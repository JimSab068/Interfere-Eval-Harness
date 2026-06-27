"""
main.py — FastAPI application entry point.

Registers all routers:
  POST /ingest          — ingest.py
  POST /ingest/raw      — ingest.py
  POST /attribute       — attribute.py
  GET  /eval            — eval.py
  GET  /eval/dashboard  — eval.py

Lifespan (startup / shutdown):
  startup:
    1. init_db()              — connect Motor client to MongoDB Atlas
    2. index_source_files()   — walk stripe-clone, chunk, embed, store in ChromaDB
                                (no-op if collection already populated)
  shutdown:
    1. close_db()             — gracefully close Motor connection

Environment variables (all read by sub-modules, main.py just checks they exist):
  MONGODB_URI      — MongoDB Atlas connection string
  GEMINI_API_KEY   — Google AI Studio key  (USE_VERTEX=false)
  USE_VERTEX       — "true" to use Vertex AI instead of AI Studio
  GCP_PROJECT      — GCP project ID        (USE_VERTEX=true)
  GCP_LOCATION     — GCP region            (default: us-central1)
  GITHUB_TOKEN     — GitHub PAT for diff fetching
  GITHUB_OWNER     — Repo owner            (default: JimSab068)
  GITHUB_REPO      — Repo name             (default: stripe-clone)
  REPO_PATH        — Absolute path to stripe-clone on disk
  CHROMA_PATH      — ChromaDB persist dir  (default: ./chroma_db)
  EVAL_CASES_PATH  — Path to eval_cases.json (default: ./eval_cases.json)
  EVAL_LLM_JUDGE   — "true" to run LLM judge in /eval (default: false)

Run locally:
  uvicorn main:app --reload --port 8000

Run in Docker:
  docker build -t interfere-eval .
  docker run -p 8000:8000 --env-file .env interfere-eval
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load .env before any sub-module reads os.environ
load_dotenv()

# Sub-module imports (after dotenv so env vars are available)
from db import close_db, init_db
from rag import index_source_files

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required env var check
# ---------------------------------------------------------------------------

_REQUIRED_VARS = ["MONGODB_URI", "REPO_PATH"]
_OPTIONAL_HINTS = {
    "GEMINI_API_KEY": "Required when USE_VERTEX=false (local dev)",
    "GITHUB_TOKEN":   "Recommended — avoids GitHub rate limits on diff fetches",
    "GCP_PROJECT":    "Required when USE_VERTEX=true (Cloud Run deployment)",
}


def _check_env() -> None:
    missing = [v for v in _REQUIRED_VARS if not os.environ.get(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Add them to your .env file and restart."
        )

    use_vertex = os.environ.get("USE_VERTEX", "false").lower() == "true"
    if use_vertex and not os.environ.get("GCP_PROJECT"):
        raise EnvironmentError(
            "USE_VERTEX=true but GCP_PROJECT is not set.\n"
            "Add GCP_PROJECT=<your-project-id> to your .env file."
        )
    if not use_vertex and not os.environ.get("GEMINI_API_KEY"):
        raise EnvironmentError(
            "USE_VERTEX=false but GEMINI_API_KEY is not set.\n"
            "Add GEMINI_API_KEY=<your-key> to your .env file.\n"
            "Get a key at https://aistudio.google.com/app/apikey"
        )

    for var, hint in _OPTIONAL_HINTS.items():
        if not os.environ.get(var):
            logger.warning("Optional env var %s not set — %s", var, hint)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic for the FastAPI app."""

    # --- Startup ---
    logger.info("=" * 60)
    logger.info("Interfere Eval Harness starting up")
    logger.info("=" * 60)

    _check_env()

    logger.info("Connecting to MongoDB...")
    await init_db()
    logger.info("MongoDB connected.")

    logger.info("Initialising RAG index (ChromaDB)...")
    force = os.environ.get("FORCE_REINDEX", "false").lower() == "true"
    n = await index_source_files(force_reindex=force)
    logger.info("RAG index ready. Chunks in collection: %d", n)

    logger.info("=" * 60)
    logger.info("All systems ready. Listening for requests.")
    logger.info("=" * 60)

    yield

    # --- Shutdown ---
    logger.info("Shutting down — closing MongoDB connection...")
    await close_db()
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Interfere Eval Harness",
    description=(
        "Codebase-aware anomaly attribution engine for Next.js apps. "
        "Receives Interfere SDK error envelopes, attributes them to the "
        "culprit commit via RAG + GitHub diff + Gemini, and scores accuracy "
        "against a ground-truth eval dataset."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins in dev; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from attribute import router as attribute_router  # noqa: E402
from eval import router as eval_router            # noqa: E402
from ingest import router as ingest_router        # noqa: E402

app.include_router(ingest_router)
app.include_router(attribute_router)
app.include_router(eval_router)


# ---------------------------------------------------------------------------
# Health / root
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "interfere-eval-harness",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "ingest":          "POST /ingest",
            "ingest_raw":      "POST /ingest/raw",
            "attribute":       "POST /attribute",
            "eval":            "GET  /eval",
            "eval_dashboard":  "GET  /eval/dashboard",
        },
    }


@app.get("/health", include_in_schema=False)
async def health():
    """
    Lightweight health check for Cloud Run / load balancer probes.
    Returns 200 as long as the process is running.
    Does not check MongoDB or ChromaDB — startup would have failed if those
    were unavailable.
    """
    return JSONResponse({"status": "ok"})