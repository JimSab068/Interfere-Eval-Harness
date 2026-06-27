"""
attribute.py — POST /attribute endpoint.

Attribution pipeline:
  1. Extract error context from the envelope (type, message, route, frames)
  2. RAG: embed error context → ChromaDB → retrieve top-k relevant source chunks
  3. GitHub: fetch git diff between prev_tag and tag (incremental, not cumulative)
  4. LLM: send (error + RAG chunks + diff) to Gemini → structured attribution
  5. Return: culprit_file, culprit_function, confidence, explanation, reasoning_trace

Endpoint:   POST /attribute
Body:       AttributeRequest  { envelope: Envelope, tag: str }
Response:   AttributeResponse { culprit_file, culprit_function, confidence,
                                explanation, reasoning_trace }

Environment variables:
  GITHUB_TOKEN  — Personal access token with repo read scope
  GITHUB_OWNER  — Repo owner, e.g. "JimSab068"
  GITHUB_REPO   — Repo name,  e.g. "stripe-clone"
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, status

from llm import generate
from models import AttributeRequest, AttributeResponse, Envelope, Frame
from rag import RetrievedChunk, format_chunks_for_prompt, retrieve

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/attribute", tags=["attribute"])

# ---------------------------------------------------------------------------
# GitHub config
# ---------------------------------------------------------------------------

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_OWNER = os.environ.get("GITHUB_OWNER", "JimSab068")
GITHUB_REPO  = os.environ.get("GITHUB_REPO",  "stripe-clone")

# Default base tag — used when prev_tag is not supplied (e.g. direct API calls)
DEFAULT_BASE_TAG = "v1.0.0-clean"

GITHUB_COMPARE_URL = (
    "https://api.github.com/repos/{owner}/{repo}/compare/{base}...{head}"
)

# Max characters of diff to send to Gemini — keeps prompt within token budget
MAX_DIFF_CHARS = 8000

# Sentinel prefix written by _fetch_github_diff when the API returns an error
_DIFF_ERROR_PREFIX = "[GitHub diff"

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert software engineer specialising in debugging Next.js applications.
Your task is to perform root cause analysis: given a runtime error and a set of
recent code changes, identify exactly which file and function introduced the bug.

Rules:
- Base your answer ONLY on the provided error details and code diff.
- Do not guess. If you are uncertain, say so in your explanation and lower your confidence score.
- Always respond with valid JSON matching the schema below — no markdown fences, no preamble.
- Keep explanation under 100 words. Keep reasoning_trace to 5 steps maximum.


Response schema:
{
  "culprit_file": "<repo-relative file path or null>",
  "culprit_function": "<function or component name or null>",
  "confidence": <float 0.0–1.0>,
  "explanation": "<one paragraph root cause explanation>",
  "reasoning_trace": [
    "<step 1>",
    "<step 2>",
    ...
  ]
}
"""

ATTRIBUTION_PROMPT_TEMPLATE = """\
## Error Details

- **Error type:**    {error_type}
- **Error message:** {error_message}
- **Route:**         {route}
- **Request method:**  {method}
- **Runtime:**       {runtime}

### Stack frames (innermost last)
{frames_block}

---

## Relevant Source Code (retrieved via semantic search)

{rag_block}

---

## Code Changes Introduced In This Commit ({base_tag} → {tag})

The diff below shows ONLY the changes introduced by this specific commit.
The culprit is somewhere in this diff.

{diff_block}

---

Based on the error details, relevant source code, and code changes above,
identify the single most likely culprit file and function that introduced this bug.
Respond with JSON only.
"""


# ---------------------------------------------------------------------------
# Helpers — error context extraction
# ---------------------------------------------------------------------------

def _build_error_context(envelope: Envelope) -> str:
    """
    Construct a plain-text error context string for RAG embedding.
    Combines the most signal-rich fields into one query string.
    """
    parts = [
        f"Error type: {envelope.error_type}",
        f"Error message: {envelope.error_message}",
    ]
    if envelope.route:
        parts.append(f"Route: {envelope.route}")

    exc = envelope.primary_exception
    if exc.frames:
        frame_strs = []
        for f in exc.frames[-3:]:   # last 3 frames = innermost / most relevant
            bits = []
            if f.fileName:
                bits.append(f.fileName)
            if f.functionName:
                bits.append(f.functionName)
            if f.lineNumber:
                bits.append(f"line {f.lineNumber}")
            if bits:
                frame_strs.append(" > ".join(bits))
        if frame_strs:
            parts.append("Stack: " + " | ".join(frame_strs))

    return "\n".join(parts)


def _format_frames(frames: list[Frame]) -> str:
    if not frames:
        return "  (no frames available)"
    lines = []
    for f in frames:
        line = f"  at {f.functionName or '?'}"
        if f.fileName:
            line += f" ({f.fileName}"
            if f.lineNumber:
                line += f":{f.lineNumber}"
                if f.columnNumber:
                    line += f":{f.columnNumber}"
            line += ")"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GitHub diff fetching
# ---------------------------------------------------------------------------

async def _fetch_github_diff(base_tag: str, head_tag: str) -> str:
    """
    Call GitHub Compare API to get the unified diff between base_tag and head_tag.

    Using incremental diffs (prev_tag → tag) rather than cumulative
    (v1.0.0-clean → tag) means Gemini only sees the single commit that
    introduced this bug — no noise from earlier bugs.

    Returns the diff as a string (possibly truncated to MAX_DIFF_CHARS).
    On API errors returns a sentinel string starting with _DIFF_ERROR_PREFIX.
    """
    if not GITHUB_TOKEN:
        logger.warning("GITHUB_TOKEN not set — diff fetch will likely be rate-limited")

    url = GITHUB_COMPARE_URL.format(
        owner=GITHUB_OWNER,
        repo=GITHUB_REPO,
        base=base_tag,
        head=head_tag,
    )

    headers = {
        "Accept": "application/vnd.github.v3.diff",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url, headers=headers)

        if response.status_code == 404:
            return (
                f"{_DIFF_ERROR_PREFIX} not found]: "
                f"{base_tag}...{head_tag} does not exist in "
                f"{GITHUB_OWNER}/{GITHUB_REPO}. "
                f"Check that the tag is correct."
            )
        if response.status_code == 403:
            return f"{_DIFF_ERROR_PREFIX} rate-limited]: Set GITHUB_TOKEN in .env"

        response.raise_for_status()
        diff = response.text

    except httpx.TimeoutException:
        return f"{_DIFF_ERROR_PREFIX} timed out]: fetch for {base_tag}...{head_tag}"
    except Exception as exc:
        logger.warning("GitHub diff fetch failed: %s", exc)
        return f"{_DIFF_ERROR_PREFIX} error]: {exc}"

    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + f"\n\n[... diff truncated at {MAX_DIFF_CHARS} chars ...]"

    logger.debug("Fetched diff %s...%s (%d chars)", base_tag, head_tag, len(diff))
    return diff


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

def _parse_llm_response(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to extract first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # ── NEW: try to salvage truncated JSON by extracting key fields directly ──
    result = {}
    for field in ("culprit_file", "culprit_function", "explanation"):
        m = re.search(rf'"{field}"\s*:\s*"([^"]*)"', cleaned)
        if m:
            result[field] = m.group(1)
    conf = re.search(r'"confidence"\s*:\s*([0-9.]+)', cleaned)
    if conf:
        result["confidence"] = float(conf.group(1))
    if result.get("culprit_file"):
        result.setdefault("culprit_function", None)
        result.setdefault("confidence", 0.8)
        result.setdefault("explanation", "(truncated response — fields extracted)")
        result.setdefault("reasoning_trace", ["Response truncated — key fields salvaged"])
        return result

    logger.warning("Could not parse LLM response as JSON:\n%s", raw[:500])
    return {
        "culprit_file": None,
        "culprit_function": None,
        "confidence": 0.0,
        "explanation": f"LLM response could not be parsed as JSON. Raw: {raw[:300]}",
        "reasoning_trace": ["JSON parse failed"],
    }


# ---------------------------------------------------------------------------
# Core attribution logic
# ---------------------------------------------------------------------------

async def run_attribution(
    envelope: Envelope,
    tag: str,
    prev_tag: Optional[str] = None,   # ← NEW: incremental diff base
) -> AttributeResponse:
    """
    Full attribution pipeline.

    prev_tag: the tag immediately before `tag` (e.g. "v1.1.0" when tag="v1.2.0").
    If omitted, falls back to DEFAULT_BASE_TAG ("v1.0.0-clean") which gives a
    cumulative diff — fine for direct API calls but noisier for the eval harness.
    """
    reasoning_trace: list[str] = []

    # Resolve diff base
    base_tag = prev_tag if prev_tag else DEFAULT_BASE_TAG
    reasoning_trace.append(
        f"Diff range: {base_tag} → {tag} "
        f"({'incremental' if prev_tag else 'cumulative fallback'})"
    )

    # --- Step 1: Extract error context ---
    error_context = _build_error_context(envelope)
    reasoning_trace.append(
        f"Extracted error context: {envelope.error_type} on route {envelope.route}"
    )
    logger.info(
        "Attribution started: tag=%s base=%s error_type=%s route=%s",
        tag, base_tag, envelope.error_type, envelope.route,
    )

    # --- Step 2: RAG retrieval ---
    reasoning_trace.append("Querying ChromaDB for semantically relevant source chunks")
    try:
        rag_chunks: list[RetrievedChunk] = await asyncio.to_thread(
            retrieve, error_context, 5
        )
        reasoning_trace.append(
            f"Retrieved {len(rag_chunks)} chunks: "
            + ", ".join(f"{c.file}::{c.function}" for c in rag_chunks)
        )
    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)
        rag_chunks = []
        reasoning_trace.append(f"RAG retrieval failed: {exc} — proceeding without chunks")

    rag_block = format_chunks_for_prompt(rag_chunks) if rag_chunks else "(no chunks retrieved)"

    # --- Step 3: GitHub diff (incremental) ---
    reasoning_trace.append(f"Fetching incremental git diff: {base_tag} → {tag}")
    diff_block = await _fetch_github_diff(base_tag, tag)

    if diff_block.startswith(_DIFF_ERROR_PREFIX):
        detail = diff_block
        if "not found" in diff_block:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=detail,
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail,
        )

    reasoning_trace.append(f"Diff fetched ({len(diff_block)} chars)")

    # --- Step 4: Build prompt ---
    from models import NextjsContext

    nextjs_ctx: Optional[NextjsContext] = (
        envelope.context if isinstance(envelope.context, NextjsContext) else None
    )
    request_method: str = (
        nextjs_ctx.requestMethod if nextjs_ctx and nextjs_ctx.requestMethod
        else "(unknown)"
    )
    runtime_str: str = envelope.runtime.value if envelope.runtime else "node"

    exc = envelope.primary_exception
    frames_block = _format_frames(exc.frames)

    prompt = ATTRIBUTION_PROMPT_TEMPLATE.format(
        error_type=envelope.error_type,
        error_message=envelope.error_message,
        route=envelope.route or "(unknown)",
        method=request_method,
        runtime=runtime_str,
        frames_block=frames_block,
        rag_block=rag_block,
        base_tag=base_tag,
        tag=tag,
        diff_block=diff_block,
    )
    reasoning_trace.append(f"Built attribution prompt ({len(prompt)} chars)")

    # --- Step 5: Gemini generation ---
    reasoning_trace.append(f"Calling Gemini ({envelope.error_type} → attribution)")
    try:
        raw_response = await asyncio.to_thread(
            generate,
            prompt,
            system=SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=2048,
        )
    except Exception as exc_gen:
        logger.exception("Gemini generation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM generation failed: {exc_gen}",
        ) from exc_gen

    reasoning_trace.append("Received Gemini response — parsing JSON")

    # --- Step 6: Parse response ---
    parsed = _parse_llm_response(raw_response)

    confidence = float(parsed.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))

    llm_trace = parsed.get("reasoning_trace", [])
    if isinstance(llm_trace, list):
        reasoning_trace.extend(llm_trace)

    reasoning_trace.append(
        f"Attribution complete: culprit={parsed.get('culprit_file')}::"
        f"{parsed.get('culprit_function')} confidence={confidence:.2f}"
    )

    logger.info(
        "Attribution done: tag=%s culprit=%s::%s confidence=%.2f",
        tag,
        parsed.get("culprit_file"),
        parsed.get("culprit_function"),
        confidence,
    )

    return AttributeResponse(
        culprit_file=parsed.get("culprit_file"),
        culprit_function=parsed.get("culprit_function"),
        confidence=confidence,
        explanation=parsed.get("explanation", "(no explanation)"),
        reasoning_trace=reasoning_trace,
    )


# ---------------------------------------------------------------------------
# FastAPI endpoint
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=AttributeResponse,
    summary="Attribute an error envelope to a culprit commit",
)
async def attribute_envelope(body: AttributeRequest) -> AttributeResponse:
    # Direct API calls don't supply prev_tag — cumulative diff is used as fallback
    return await run_attribution(envelope=body.envelope, tag=body.tag)