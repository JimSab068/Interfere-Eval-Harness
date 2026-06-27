# Interfere Eval Harness

A codebase-aware root cause attribution engine for Next.js applications, built as a portfolio project targeting [Interfere (YC S25)](https://interfere.com). The harness ingests runtime error envelopes from the `@interfere/next` SDK, attributes each error to its culprit commit using a RAG + GitHub diff + Gemini pipeline, and scores attribution accuracy against a ground-truth eval dataset.


## Eval Results

**7/7 file accuracy · 7/7 function accuracy** across all ground-truth cases (v1.1.0 → v1.7.0)

 [View live eval dashboard](https://htmlpreview.github.io/?https://github.com/JimSab068/eval-harness/blob/main/docs/InterfereDashboard.html)


---

## Architecture

```
stripe-clone (Next.js)          eval harness (FastAPI)
  ↓ error envelope                ↓
POST /ingest          →   MongoDB (Motor async)
                              ↓
POST /attribute       →   RAG (ChromaDB + Gemini embeddings)
                          + GitHub diff (REST API)
                          + Gemini 2.0 Flash (attribution)
                              ↓
GET /eval             →   Score vs. ground truth (eval_cases.json)
GET /eval/dashboard   →   HTML report
```

**Two repos work together:**

- [`JimSab068/stripe-clone`](https://github.com/JimSab068/stripe-clone) — A Next.js Stripe-like app with 7 intentional bugs introduced as discrete git commits (`v1.1.0`–`v1.7.0`), with `v1.0.0-clean` as the bug-free baseline.
- This repo (`eval-harness`) — The FastAPI backend that receives those errors, attributes them to the correct commit, and measures accuracy.

---

## How Attribution Works

For each error envelope received at `POST /attribute`:

1. **Error context extraction** — pulls error type, message, route, and the innermost stack frames from the envelope.
2. **RAG retrieval** — embeds the error context with `gemini-embedding-2` and queries ChromaDB for the top-5 most semantically relevant source chunks from the stripe-clone codebase.
3. **GitHub diff** — calls the GitHub Compare API to fetch the incremental diff between `prev_tag` and `tag` (e.g. `v1.1.0 → v1.2.0`), so Gemini only sees the single commit that introduced the bug.
4. **Gemini reasoning** — sends the error + RAG chunks + diff to Gemini 2.0 Flash with a structured JSON prompt. Returns `culprit_file`, `culprit_function`, `confidence`, `explanation`, and a `reasoning_trace`.
5. **Response parsing** — handles truncated responses with a regex-based fallback extractor before returning `AttributeResponse`.

---

## Eval Scoring

`GET /eval` runs attribution over every case in `eval_cases.json` and computes three metrics:

| Metric | Description |
|---|---|
| `accuracy_file` | Fraction of cases where the predicted file exactly matches ground truth (primary) |
| `accuracy_function` | Fraction of cases where the predicted function matches (secondary) |
| `accuracy_explanation` | LLM-judged correctness of the explanation; enabled via `EVAL_LLM_JUDGE=true` (tertiary) |

`GET /eval/dashboard` returns a self-contained HTML page showing the full reasoning trace, prediction, and ground truth side-by-side for each case.

---

## Project Structure

```
eval-harness/
├── main.py          # FastAPI app, lifespan (DB init + RAG indexing), routers
├── models.py        # Pydantic models mirroring the real @interfere/types schema
├── db.py            # Motor async client; insert_event, get_event_by_tag_and_route
├── ingest.py        # POST /ingest and POST /ingest/raw endpoints
├── rag.py           # ChromaDB indexer; index_source_files(), retrieve()
├── llm.py           # Gemini wrapper; generate(), embed(), GeminiEmbeddingFunction
├── attribute.py     # POST /attribute; full attribution pipeline
├── eval.py          # GET /eval and GET /eval/dashboard
├── eval_cases.json  # Ground-truth dataset (7 bugs × tag + route + culprit)
└── Dockerfile       # Cloud Run deployment
```

---

## Setup

### Prerequisites

- Python 3.12.6
- MongoDB Atlas cluster (free tier works)
- Google AI Studio API key 
- GitHub PAT with `repo:read` scope
- The `stripe-clone` repo cloned locally

### Install

```bash
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```env
# Required
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/interfere
REPO_PATH=/absolute/path/to/stripe-clone

# LLM backend — choose one
GEMINI_API_KEY=your-ai-studio-key       # local dev (USE_VERTEX=false)
USE_VERTEX=false

# USE_VERTEX=true  →  Vertex AI (Cloud Run)
# GCP_PROJECT=your-gcp-project-id
# GCP_LOCATION=us-central1

GITHUB_TOKEN=ghp_...                    # avoids GitHub rate limits
GITHUB_OWNER=JimSab068
GITHUB_REPO=stripe-clone

# Optional tuning
CHROMA_PATH=./chroma_db                 # ChromaDB persist directory
EVAL_CASES_PATH=./eval_cases.json
EVAL_LLM_JUDGE=false                    # set true to enable explanation scoring
FORCE_REINDEX=false                     # set true to rebuild ChromaDB from scratch
```

### Run locally

```bash
uvicorn main:app --reload --port 8000
```

On startup the app will:
1. Connect to MongoDB and create indexes.
2. Walk the stripe-clone source tree, chunk it by function, embed each chunk, and store it in ChromaDB (skipped if the collection is already populated).

Visit `http://localhost:8000/docs` for the interactive API docs.

---

## Usage

### 1. Ingest an error envelope

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "envelope": { ... },   # @interfere/next SDK envelope
    "tag": "v1.3.0"
  }'
```

Returns `201` with `{ received: true, uuid: "...", tag: "v1.3.0" }`. Returns `409` if the UUID was already ingested.

### 2. Attribute an error

```bash
curl -X POST http://localhost:8000/attribute \
  -H "Content-Type: application/json" \
  -d '{
    "envelope": { ... },
    "tag": "v1.3.0"
  }'
```

Returns:

```json
{
  "culprit_file": "app/api/payment/route.ts",
  "culprit_function": "POST",
  "confidence": 0.92,
  "explanation": "...",
  "reasoning_trace": ["...", "..."]
}
```

### 3. Run the eval suite

```bash
curl http://localhost:8000/eval
```

### 4. View the dashboard

Open `http://localhost:8000/eval/dashboard` in a browser.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| Schema validation | Pydantic v2 |
| Database | MongoDB Atlas (Motor async driver) |
| Vector store | ChromaDB (persistent) |
| Embeddings | Gemini `gemini-embedding-2` (3072-dim) |
| LLM | Gemini 2.0 Flash |
| GitHub integration | GitHub REST API v3 (compare endpoint) |
| HTTP client | httpx (async) |
