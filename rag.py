"""
rag.py — ChromaDB-backed RAG indexer for Next.js source files.

Responsibilities:
  1. Walk the stripe-clone source tree and extract function-level chunks
  2. Embed each chunk with Gemini text-embedding-004 via llm.py
  3. Store chunks in a persistent ChromaDB collection
  4. Expose retrieve() — given an error context string, return the top-k
     most semantically relevant source chunks

Called from:
  - main.py lifespan → index_source_files() on startup
  - attribute.py     → retrieve() per attribution request

Environment variables used:
  REPO_PATH   — absolute path to the stripe-clone Next.js repo on disk
                 e.g. E:/interfere/stripe-clone
  CHROMA_PATH — where to persist ChromaDB (default: ./chroma_db)

ChromaDB collection: "source_chunks"
Each document stored:
  id:        "<relative_file_path>::<function_name>::<chunk_index>"
  document:  raw source chunk text
  metadata:  { file: str, function: str, language: str, chunk_index: int }
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from llm import GeminiEmbeddingFunction, GeminiQueryEmbeddingFunction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_PATH = Path(os.environ.get("REPO_PATH", "./stripe-clone"))
CHROMA_PATH = Path(os.environ.get("CHROMA_PATH", "./chroma_db"))
COLLECTION_NAME = "source_chunks"

# Only index these extensions — skip config, lock files, etc.
INDEXABLE_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx"}

# Directories to skip entirely
SKIP_DIRS = {
    "node_modules", ".next", ".git", "dist", "build",
    "__pycache__", ".vercel", "coverage",
}

# Max characters per chunk — keeps each chunk within Gemini's token limits
# and ensures ChromaDB documents stay manageable (~300–400 tokens each)
MAX_CHUNK_CHARS = 1500

# Number of top chunks to return from retrieve()
DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SourceChunk:
    """One function-level (or fallback file-level) chunk of source code."""
    file: str           # repo-relative path, e.g. "app/api/payment/route.ts"
    function: str       # function/component name, or "__module__" for file-level
    language: str       # "typescript" or "javascript"
    text: str           # raw source text of the chunk
    chunk_index: int    # 0-based index within the same function (if split)

    @property
    def doc_id(self) -> str:
        return f"{self.file}::{self.function}::{self.chunk_index}"

    @property
    def metadata(self) -> dict:
        return {
            "file": self.file,
            "function": self.function,
            "language": self.language,
            "chunk_index": self.chunk_index,
        }


@dataclass
class RetrievedChunk:
    """A chunk returned by retrieve(), with its similarity distance."""
    file: str
    function: str
    language: str
    text: str
    chunk_index: int
    distance: float     # lower = more similar (ChromaDB L2 by default)

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "function": self.function,
            "language": self.language,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "distance": self.distance,
        }


# ---------------------------------------------------------------------------
# ChromaDB client (module-level, initialised once)
# ---------------------------------------------------------------------------

_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


def _get_collection() -> chromadb.Collection:
    if _collection is None:
        raise RuntimeError(
            "RAG not initialised. Call index_source_files() on startup."
        )
    return _collection


def init_chroma() -> chromadb.Collection:
    """
    Create (or open existing) ChromaDB persistent client and collection.
    Safe to call multiple times — returns the existing collection if already open.
    """
    global _chroma_client, _collection

    if _collection is not None:
        return _collection

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    _chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )

    _collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=GeminiEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},   # cosine similarity for code
    )

    logger.info(
        "ChromaDB collection '%s' opened at %s (existing docs: %d)",
        COLLECTION_NAME,
        CHROMA_PATH,
        _collection.count(),
    )
    return _collection


# ---------------------------------------------------------------------------
# Source file walking
# ---------------------------------------------------------------------------

def _iter_source_files(repo_path: Path):
    """Yield Path objects for every indexable source file in the repo."""
    for path in repo_path.rglob("*"):
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue
        if path.suffix not in INDEXABLE_EXTENSIONS:
            continue
        if path.is_file():
            yield path


def _relative(path: Path, repo_path: Path) -> str:
    """Return a repo-relative POSIX path string."""
    try:
        return path.relative_to(repo_path).as_posix()
    except ValueError:
        return path.as_posix()


def _detect_language(path: Path) -> str:
    return "typescript" if path.suffix in {".ts", ".tsx"} else "javascript"


# ---------------------------------------------------------------------------
# Function-level chunking
# ---------------------------------------------------------------------------

# Matches common Next.js / TypeScript function patterns:
#   export default function Foo(...)
#   export async function bar(...)
#   const baz = async (...) =>
#   export const handler = (req, res) =>
_FUNCTION_PATTERN = re.compile(
    r"""
    (?:export\s+)?                          # optional export
    (?:default\s+)?                         # optional default
    (?:async\s+)?                           # optional async
    (?:
        function\s+(\w+)                    # named function
        |
        (?:const|let|var)\s+(\w+)\s*=\s*   # arrow function assigned to var
        (?:async\s+)?
        (?:\([^)]*\)\s*=>|\([^)]*\)\s*:\s*\w+\s*=>)
    )
    """,
    re.VERBOSE,
)


def _extract_chunks(file_path: Path, repo_path: Path) -> list[SourceChunk]:
    """
    Parse a source file and split it into function-level chunks.

    Strategy:
      1. Find all function definition sites via regex
      2. Use brace counting to determine where each function body ends
      3. If no functions found, treat the whole file as one chunk
      4. Split any chunk exceeding MAX_CHUNK_CHARS into sub-chunks
    """
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read %s: %s", file_path, exc)
        return []

    rel_path = _relative(file_path, repo_path)
    language = _detect_language(file_path)
    chunks: list[SourceChunk] = []

    # Find function start positions
    matches = list(_FUNCTION_PATTERN.finditer(source))

    if not matches:
        # No recognisable functions — chunk the whole file
        return _split_text(source, rel_path, "__module__", language)

    for i, match in enumerate(matches):
        fn_name = match.group(1) or match.group(2) or f"anonymous_{i}"
        start = match.start()

        # Determine end: either the next function's start or EOF
        end = matches[i + 1].start() if i + 1 < len(matches) else len(source)

        # Refine end using brace counting to avoid grabbing too much
        end = _find_function_end(source, start, end)

        fn_text = source[start:end].strip()
        if not fn_text:
            continue

        chunks.extend(_split_text(fn_text, rel_path, fn_name, language))

    return chunks


def _find_function_end(source: str, fn_start: int, candidate_end: int) -> int:
    """
    Walk forward from fn_start tracking brace depth.
    Return the index just after the closing brace of the outermost block,
    capped at candidate_end.
    """
    depth = 0
    found_open = False

    for i in range(fn_start, candidate_end):
        ch = source[i]
        if ch == "{":
            depth += 1
            found_open = True
        elif ch == "}" and found_open:
            depth -= 1
            if depth == 0:
                return i + 1

    return candidate_end


def _split_text(
    text: str,
    file: str,
    function: str,
    language: str,
) -> list[SourceChunk]:
    """
    Split text into chunks of at most MAX_CHUNK_CHARS characters.
    Tries to split on newlines to avoid cutting mid-line.
    """
    if len(text) <= MAX_CHUNK_CHARS:
        return [SourceChunk(
            file=file,
            function=function,
            language=language,
            text=text,
            chunk_index=0,
        )]

    chunks = []
    idx = 0
    chunk_num = 0

    while idx < len(text):
        end = idx + MAX_CHUNK_CHARS
        if end < len(text):
            # Try to break on the last newline within the window
            newline = text.rfind("\n", idx, end)
            if newline > idx:
                end = newline + 1

        chunk_text = text[idx:end].strip()
        if chunk_text:
            chunks.append(SourceChunk(
                file=file,
                function=function,
                language=language,
                text=chunk_text,
                chunk_index=chunk_num,
            ))
            chunk_num += 1
        idx = end

    return chunks


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

async def index_source_files(force_reindex: bool = False) -> int:
    """
    Walk REPO_PATH, chunk all source files, embed and store in ChromaDB.

    Args:
        force_reindex: If True, delete the existing collection first and
                       rebuild from scratch. Use when source files change.

    Returns:
        Number of chunks indexed.

    Called from main.py on startup.
    """
    if not REPO_PATH.exists():
        raise FileNotFoundError(
            f"REPO_PATH does not exist: {REPO_PATH}\n"
            f"Set REPO_PATH in your .env to the absolute path of the stripe-clone repo."
        )

    collection = init_chroma()

    if force_reindex and collection.count() > 0:
        logger.info("force_reindex=True — clearing existing collection")
        _chroma_client.delete_collection(COLLECTION_NAME)
        # Re-create empty collection
        global _collection
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=GeminiEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
        collection = _collection

    if collection.count() > 0:
        logger.info(
            "Collection already has %d chunks — skipping reindex. "
            "Pass force_reindex=True to rebuild.",
            collection.count(),
        )
        return collection.count()

    # --- Walk and chunk ---
    all_chunks: list[SourceChunk] = []
    for file_path in _iter_source_files(REPO_PATH):
        file_chunks = _extract_chunks(file_path, REPO_PATH)
        logger.debug("  %s → %d chunks", file_path.name, len(file_chunks))
        all_chunks.extend(file_chunks)

    if not all_chunks:
        logger.warning(
            "No source chunks found in %s. "
            "Check REPO_PATH and INDEXABLE_EXTENSIONS.",
            REPO_PATH,
        )
        return 0

    logger.info("Indexing %d chunks from %s ...", len(all_chunks), REPO_PATH)

    # --- Batch upsert into ChromaDB ---
    # ChromaDB recommends batches of ~100 to avoid memory spikes
    BATCH_SIZE = 50
    total_indexed = 0

    for batch_start in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[batch_start : batch_start + BATCH_SIZE]

        collection.upsert(
            ids=[c.doc_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
        )
        total_indexed += len(batch)
        logger.info(
            "  Indexed batch %d–%d / %d",
            batch_start + 1,
            batch_start + len(batch),
            len(all_chunks),
        )

    logger.info("RAG index complete. Total chunks: %d", total_indexed)
    return total_indexed


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    error_context: str,
    top_k: int = DEFAULT_TOP_K,
    where: Optional[dict] = None,
) -> list[RetrievedChunk]:
    """
    Embed error_context and return the top_k most relevant source chunks.

    Args:
        error_context: A string describing the error — typically built by
                       attribute.py from the envelope's error type, message,
                       route, and top stack frame.
        top_k:         Number of chunks to return (default 5).
        where:         Optional ChromaDB metadata filter, e.g.
                       {"file": "app/api/payment/route.ts"} to restrict
                       retrieval to a specific file.

    Returns:
        List of RetrievedChunk sorted by ascending distance (most relevant first).
    """
    collection = _get_collection()

    query_kwargs: dict = {
        "query_texts": [error_context],
        "n_results": min(top_k, collection.count() or 1),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    # Use query-task embedding via a fresh function instance
    # ChromaDB uses the collection's stored embedding_function for query_texts —
    # we override by embedding manually and using query_embeddings instead.
    query_fn = GeminiQueryEmbeddingFunction()
    query_embedding = query_fn([error_context])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=query_kwargs["n_results"],
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    chunks: list[RetrievedChunk] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        chunks.append(RetrievedChunk(
            file=meta.get("file", "unknown"),
            function=meta.get("function", "unknown"),
            language=meta.get("language", "typescript"),
            text=doc,
            chunk_index=meta.get("chunk_index", 0),
            distance=dist,
        ))

    logger.debug(
        "retrieve() query_len=%d top_k=%d results=%d",
        len(error_context), top_k, len(chunks),
    )
    return chunks


def format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a readable block for inclusion in the
    Gemini attribution prompt in attribute.py.

    Example output:
        [1] app/api/payment/route.ts :: POST
        ```typescript
        export async function POST(req: Request) { ... }
        ```
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] {chunk.file} :: {chunk.function}"
        body = f"```{chunk.language}\n{chunk.text}\n```"
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)