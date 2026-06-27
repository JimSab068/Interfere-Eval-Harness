"""
llm.py — Thin wrapper around Gemini, supporting both:
  - Google AI Studio   (local dev)  via google-genai
  - Vertex AI          (Cloud Run)  via google-genai (vertexai=True)

Which backend is used is controlled by the USE_VERTEX env var:
  USE_VERTEX=false  →  AI Studio  (needs GEMINI_API_KEY)
  USE_VERTEX=true   →  Vertex AI  (needs GCP_PROJECT + GCP_LOCATION,
                                   auth via attached service account)

Public API (used by rag.py and attribute.py):
  generate(prompt: str, *, system: str | None) -> str
  embed(text: str) -> list[float]
  embed_batch(texts: list[str]) -> list[list[float]]

Both functions are synchronous wrappers — ChromaDB's embedding function
interface is sync, and FastAPI endpoints that call generate() do so inside
an async endpoint via asyncio.to_thread() in attribute.py.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Read config once at import time
# ---------------------------------------------------------------------------

_USE_VERTEX: bool = os.environ.get("USE_VERTEX", "false").lower() == "true"
_GCP_PROJECT: str = os.environ.get("GCP_PROJECT", "")
_GCP_LOCATION: str = os.environ.get("GCP_LOCATION", "us-central1")
_GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")

# Model names — identical string on both backends
GENERATION_MODEL = "gemini-3.1-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIM = 3072


# ---------------------------------------------------------------------------
# Backend initialisation (lazy — happens on first call, not at import)
# ---------------------------------------------------------------------------

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    """
    Lazily initializes and returns the unified GenAI Client based on backend config.
    """
    global _client
    if _client is not None:
        return _client

    if _USE_VERTEX:
        if not _GCP_PROJECT:
            raise EnvironmentError(
                "USE_VERTEX=true but GCP_PROJECT is not set.\n"
                "Add GCP_PROJECT=<your-project-id> to your .env / Cloud Run env vars."
            )
        _client = genai.Client(
            vertexai=True,
            project=_GCP_PROJECT,
            location=_GCP_LOCATION,
        )
        logger.info(
            "LLM backend: Vertex AI (project=%s location=%s model=%s)",
            _GCP_PROJECT,
            _GCP_LOCATION,
            GENERATION_MODEL,
        )
    else:
        if not _GEMINI_API_KEY:
            raise EnvironmentError(
                "USE_VERTEX=false but GEMINI_API_KEY is not set.\n"
                "Add GEMINI_API_KEY=<your key> to your .env file.\n"
                "Get a key at https://aistudio.google.com/app/apikey"
            )
        _client = genai.Client(api_key=_GEMINI_API_KEY)
        logger.info("LLM backend: Google AI Studio (model=%s)", GENERATION_MODEL)

    return _client


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate(
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1024,
) -> str:
    """
    Send a prompt to Gemini and return the text response.

    Args:
        prompt:            The user-turn content.
        system:            Optional system instruction.
        temperature:       Low (0.2) by default — we want deterministic attribution.
        max_output_tokens: Cap to keep costs predictable.

    Returns:
        The model's text response as a plain string.

    Raises:
        RuntimeError if the model returns an empty or blocked response.
    """
    client = _get_client()

    logger.debug("generate() prompt_len=%d use_vertex=%s", len(prompt), _USE_VERTEX)

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )

    # Robust text extraction supporting early token truncations
    text_content = None
    try:
        text_content = response.text
    except Exception:
        pass

    # Fallback: Extract partial text manually from parts if .text wrapper evaluates to None
    if not text_content and response.candidates:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            text_content = "".join([part.text for part in candidate.content.parts if getattr(part, "text", None)])

    if not text_content:
        finish_reason = "unknown"
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason
        raise RuntimeError(
            f"Gemini returned empty response. Finish reason: {finish_reason}"
        )
        
    return text_content.strip()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def embed(text: str) -> list[float]:
    """
    Embed a single text string.
    Returns a list[float] of length EMBEDDING_DIM.
    """
    return embed_batch([text])[0]


# def embed_batch(
#     texts: list[str],
#     *,
#     task_type: str = "RETRIEVAL_DOCUMENT",
#     retry_on_rate_limit: bool = True,
# ) -> list[list[float]]:
#     """
#     Embed a list of strings in one API call.

#     Args:
#         texts:               Strings to embed.
#         task_type:           Gemini task type hint.
#                              Use "RETRIEVAL_DOCUMENT" when indexing source chunks.
#                              Use "RETRIEVAL_QUERY"    when embedding an error context query.
#         retry_on_rate_limit: If True, sleep and retry once on 429.

#     Returns:
#         List of embedding vectors in the same order as `texts`.
#     """
#     client = _get_client()

#     if not texts:
#         return []

#     logger.debug(
#         "embed_batch() n=%d task_type=%s use_vertex=%s",
#         len(texts), task_type, _USE_VERTEX,
#     )

#     try:
#         result = client.models.embed_content(
#             model=EMBEDDING_MODEL,
#             contents=texts,
#             config=types.EmbedContentConfig(
#                 task_type=task_type,
#             ),
#         )
        
#         if not result.embeddings:
#             raise RuntimeError("Gemini embed_content returned no embeddings")
            
#         return [e.values for e in result.embeddings]

#     except Exception as exc:
#         if retry_on_rate_limit and "429" in str(exc):
#             logger.warning("Rate limit hit on embed_batch, sleeping 60s then retrying")
#             time.sleep(60)
#             return embed_batch(texts, task_type=task_type, retry_on_rate_limit=False)
#         raise


def embed_batch(
    texts: list[str],
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
    retry_on_rate_limit: bool = True,
) -> list[list[float]]:
    """
    Embed a list of strings in one API call.
    """
    client = _get_client()

    if not texts:
        return []

    logger.debug(
        "embed_batch() n=%d task_type=%s use_vertex=%s",
        len(texts), task_type, _USE_VERTEX,
    )

    # -----------------------------------------------------------------------
    # CRITICAL FIX FOR GEMINI-EMBEDDING-2:
    # Explicitly wrap strings to prevent the SDK from aggregating your list
    # into a single, multi-part document vector.
    # -----------------------------------------------------------------------
    formatted_contents = [
        types.Content(parts=[types.Part(text=t)]) for t in texts
    ]

    try:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=formatted_contents,  # Send the structured objects instead of raw strings
            config=types.EmbedContentConfig(
                task_type=task_type,
            ),
        )
        
        if not result.embeddings:
            raise RuntimeError("Gemini embed_content returned no embeddings")
            
        return [e.values for e in result.embeddings]

    except Exception as exc:
        if retry_on_rate_limit and "429" in str(exc):
            logger.warning("Rate limit hit on embed_batch, sleeping 60s then retrying")
            time.sleep(60)
            return embed_batch(texts, task_type=task_type, retry_on_rate_limit=False)
        raise

# ---------------------------------------------------------------------------
# ChromaDB-compatible embedding function
# (passed to chromadb.Collection as embedding_function=)
# ---------------------------------------------------------------------------

class GeminiEmbeddingFunction:
    """
    Implements the ChromaDB EmbeddingFunction interface.
    Used in rag.py:
        collection = chroma_client.get_or_create_collection(
            name="source_chunks",
            embedding_function=GeminiEmbeddingFunction(),
        )
    """

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return embed_batch(input, task_type="RETRIEVAL_DOCUMENT")

    def name(self) -> str:
        """Required by ChromaDB verification routines."""
        return self.__class__.__name__


class GeminiQueryEmbeddingFunction:
    """
    Same interface but uses RETRIEVAL_QUERY task type.
    Used when embedding the error context before querying ChromaDB.
    """

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return embed_batch(input, task_type="RETRIEVAL_QUERY")

    def name(self) -> str:
        """Required by ChromaDB verification routines."""
        return self.__class__.__name__