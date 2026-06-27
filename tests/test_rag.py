"""
test_rag.py — Integration tests for rag.py.

Tests the full ChromaDB index + retrieval pipeline against real source files.
Requires:
  - REPO_PATH pointing at the stripe-clone repo
  - GEMINI_API_KEY (or Vertex credentials)

Uses a separate ChromaDB directory (./chroma_test/) so it doesn't pollute
the production index. Cleaned up after tests run.

Run:
    pytest tests/test_rag.py -v
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import pytest_asyncio


TEST_CHROMA_PATH = Path("./chroma_test")


def _skip_if_no_prereqs():
    if not os.environ.get("REPO_PATH"):
        pytest.skip("REPO_PATH not set — skipping RAG tests")
    if not Path(os.environ["REPO_PATH"]).exists():
        pytest.skip(f"REPO_PATH {os.environ['REPO_PATH']} does not exist")
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GCP_PROJECT")):
        pytest.skip("No LLM credentials — skipping RAG tests")


@pytest.fixture(autouse=True)
def patch_chroma_path(monkeypatch):
    """Redirect ChromaDB writes to a temp test directory."""
    monkeypatch.setenv("CHROMA_PATH", str(TEST_CHROMA_PATH))

    # Reset module-level state between tests
    import rag
    rag._chroma_client = None
    rag._collection = None

    yield

    # Teardown — remove test chroma dir
    import rag
    rag._chroma_client = None
    rag._collection = None
    if TEST_CHROMA_PATH.exists():
        shutil.rmtree(TEST_CHROMA_PATH)


# ---------------------------------------------------------------------------
# Chunking unit tests (no network calls)
# ---------------------------------------------------------------------------

class TestChunking:

    def test_extract_chunks_from_typescript(self, tmp_path):
        from rag import _extract_chunks

        ts_file = tmp_path / "route.ts"
        ts_file.write_text("""
export async function POST(req: Request) {
  const body = await req.json();
  return Response.json({ ok: true });
}

export async function GET(req: Request) {
  return Response.json({ status: "healthy" });
}
""")
        chunks = _extract_chunks(ts_file, tmp_path)
        fn_names = [c.function for c in chunks]
        assert "POST" in fn_names
        assert "GET" in fn_names

    def test_module_fallback_for_json_file(self, tmp_path):
        from rag import _extract_chunks

        json_file = tmp_path / "transactions.json"
        json_file.write_text('[{"id": 1, "amount": null}]')
        chunks = _extract_chunks(json_file, tmp_path)
        assert len(chunks) == 1
        assert chunks[0].function == "__module__"

    def test_chunk_split_on_long_function(self, tmp_path):
        from rag import _extract_chunks, MAX_CHUNK_CHARS

        # Write a function body longer than MAX_CHUNK_CHARS
        long_body = "\n".join(f'  const x{i} = {i};' for i in range(500))
        ts_file = tmp_path / "big.ts"
        ts_file.write_text(f"export function bigFn() {{\n{long_body}\n}}")
        chunks = _extract_chunks(ts_file, tmp_path)
        # Should have been split into multiple chunks
        assert len(chunks) > 1
        assert all(len(c.text) <= MAX_CHUNK_CHARS + 200 for c in chunks)

    def test_relative_path_in_chunk(self, tmp_path):
        from rag import _extract_chunks

        sub = tmp_path / "app" / "api"
        sub.mkdir(parents=True)
        ts_file = sub / "route.ts"
        ts_file.write_text("export function GET() { return 'ok'; }")
        chunks = _extract_chunks(ts_file, tmp_path)
        assert all("app/api/route.ts" in c.file for c in chunks)

    def test_language_detection(self, tmp_path):
        from rag import _extract_chunks

        ts_file = tmp_path / "foo.tsx"
        ts_file.write_text("export const Comp = () => <div/>;")
        chunks = _extract_chunks(ts_file, tmp_path)
        assert all(c.language == "typescript" for c in chunks)

        js_file = tmp_path / "bar.js"
        js_file.write_text("function bar() { return 1; }")
        chunks_js = _extract_chunks(js_file, tmp_path)
        assert all(c.language == "javascript" for c in chunks_js)

    def test_doc_id_format(self, tmp_path):
        from rag import _extract_chunks

        ts_file = tmp_path / "route.ts"
        ts_file.write_text("export function handler() { return 'ok'; }")
        chunks = _extract_chunks(ts_file, tmp_path)
        for c in chunks:
            parts = c.doc_id.split("::")
            assert len(parts) == 3
            assert parts[0] == c.file
            assert parts[1] == c.function
            assert parts[2] == str(c.chunk_index)


# ---------------------------------------------------------------------------
# Indexing integration tests
# ---------------------------------------------------------------------------

class TestIndexing:

    @pytest.mark.asyncio
    async def test_index_source_files_returns_positive_count(self):
        _skip_if_no_prereqs()
        from rag import index_source_files
        n = await index_source_files()
        assert n > 0, "Expected at least one chunk to be indexed"

    @pytest.mark.asyncio
    async def test_index_is_idempotent(self):
        """Running index twice without force_reindex should not duplicate chunks."""
        _skip_if_no_prereqs()
        from rag import index_source_files
        n1 = await index_source_files()
        n2 = await index_source_files()
        assert n1 == n2

    @pytest.mark.asyncio
    async def test_force_reindex_rebuilds_collection(self):
        _skip_if_no_prereqs()
        from rag import index_source_files
        n1 = await index_source_files()
        n2 = await index_source_files(force_reindex=True)
        assert n2 > 0
        # Count should be the same after rebuild
        assert abs(n1 - n2) <= 2   # allow minor variance from file changes

    @pytest.mark.asyncio
    async def test_index_includes_api_routes(self):
        """Verify that key buggy files are indexed."""
        _skip_if_no_prereqs()
        from rag import index_source_files, _get_collection
        await index_source_files()
        col = _get_collection()

        # Check that payment route is in the index
        results = col.get(where={"file": "app/api/payment/route.ts"})
        assert len(results["ids"]) > 0, (
            "app/api/payment/route.ts not found in index — "
            "check REPO_PATH and that the file exists"
        )


# ---------------------------------------------------------------------------
# Retrieval tests
# ---------------------------------------------------------------------------

class TestRetrieval:

    @pytest_asyncio.fixture(autouse=True)
    async def indexed(self):
        """Ensure the index is populated before retrieval tests run."""
        _skip_if_no_prereqs()
        from rag import index_source_files
        await index_source_files()

    def test_retrieve_returns_chunks(self):
        from rag import retrieve
        chunks = retrieve("TypeError cannot read properties of null toFixed dashboard")
        assert len(chunks) > 0

    def test_retrieve_top_k_respected(self):
        from rag import retrieve
        chunks = retrieve("payment route error", top_k=3)
        assert len(chunks) <= 3

    def test_retrieve_chunks_have_required_fields(self):
        from rag import retrieve
        chunks = retrieve("session expired gmail auth")
        for c in chunks:
            assert c.file
            assert c.function
            assert c.text
            assert isinstance(c.distance, float)

    def test_retrieve_dashboard_bug_surfaces_dashboard_file(self):
        """
        The dashboard null-amount bug should surface app/dashboard/page.tsx
        in the top results when queried with its exact error context.
        """
        from rag import retrieve
        error_context = (
            "TypeError: Cannot read properties of null (reading 'toFixed') "
            "route: /dashboard function: Dashboard file: app/dashboard/page.tsx"
        )
        chunks = retrieve(error_context, top_k=5)
        files = [c.file for c in chunks]
        assert any("dashboard" in f for f in files), (
            f"Expected dashboard file in top results, got: {files}"
        )

    def test_retrieve_payment_bug_surfaces_payment_file(self):
        from rag import retrieve
        error_context = (
            "Error: Invalid card number length "
            "route: /api/payment method: POST file: app/api/payment/route.ts"
        )
        chunks = retrieve(error_context, top_k=5)
        files = [c.file for c in chunks]
        assert any("payment" in f for f in files), (
            f"Expected payment file in top results, got: {files}"
        )

    def test_format_chunks_for_prompt(self):
        from rag import retrieve, format_chunks_for_prompt
        chunks = retrieve("webhook malformed payload", top_k=2)
        formatted = format_chunks_for_prompt(chunks)
        assert "[1]" in formatted
        assert "```" in formatted
        assert "::" in formatted