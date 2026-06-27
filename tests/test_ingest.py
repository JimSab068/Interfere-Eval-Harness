"""
test_ingest.py — Integration tests for the POST /ingest endpoint.

Uses FastAPI's TestClient (synchronous) and httpx.AsyncClient (async) to hit
the real FastAPI app. MongoDB is patched to use 'interfere_test' database.

Run:
    pytest tests/test_ingest.py -v
"""

from __future__ import annotations

import copy
import os

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport


# ---------------------------------------------------------------------------
# App fixture — patches DB to use test database before importing app
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def app_client():
    """
    Start the FastAPI app with a test MongoDB collection injected,
    yield an async HTTP client, then tear down.
    """
    if not os.environ.get("MONGODB_URI"):
        pytest.skip("MONGODB_URI not set — skipping ingest integration tests")
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GCP_PROJECT")):
        pytest.skip("No LLM credentials set — skipping ingest integration tests")
    if not os.environ.get("REPO_PATH"):
        pytest.skip("REPO_PATH not set — skipping ingest integration tests")

    # Patch DB to use test collection before app starts
    import db as db_module
    from motor.motor_asyncio import AsyncIOMotorClient
    from pymongo import ASCENDING, IndexModel

    uri = os.environ["MONGODB_URI"]
    client = AsyncIOMotorClient(uri)
    test_col = client["interfere_test"]["ingested_events"]
    db_module._client = client
    db_module._collection = test_col

    await test_col.create_indexes([
        IndexModel([("uuid", ASCENDING)], unique=True),
        IndexModel([("tag", ASCENDING), ("route", ASCENDING)]),
    ])

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    await test_col.drop()
    client.close()
    db_module._client = None
    db_module._collection = None


# ---------------------------------------------------------------------------
# POST /ingest tests
# ---------------------------------------------------------------------------

class TestIngestEndpoint:

    @pytest.mark.asyncio
    async def test_ingest_valid_envelope(self, app_client, valid_envelope_dict):
        body = {"envelope": valid_envelope_dict, "tag": "v1.4.0"}
        resp = await app_client.post("/ingest", json=body)
        assert resp.status_code == 201
        data = resp.json()
        assert data["received"] is True
        assert data["uuid"] == valid_envelope_dict["uuid"]
        assert data["tag"] == "v1.4.0"

    @pytest.mark.asyncio
    async def test_ingest_without_tag(self, app_client, valid_envelope_dict):
        body = {"envelope": valid_envelope_dict}
        resp = await app_client.post("/ingest", json=body)
        assert resp.status_code == 201
        assert resp.json()["tag"] is None

    @pytest.mark.asyncio
    async def test_ingest_duplicate_returns_409(self, app_client, valid_envelope_dict):
        body = {"envelope": valid_envelope_dict, "tag": "v1.4.0"}
        resp1 = await app_client.post("/ingest", json=body)
        assert resp1.status_code == 201

        resp2 = await app_client.post("/ingest", json=body)
        assert resp2.status_code == 409
        assert "already ingested" in resp2.json()["detail"]

    @pytest.mark.asyncio
    async def test_ingest_malformed_envelope_returns_422(self, app_client):
        body = {"envelope": {"uuid": "missing-required-fields"}, "tag": "v1.0.0"}
        resp = await app_client.post("/ingest", json=body)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_payment_envelope(self, app_client, payment_envelope_dict):
        body = {"envelope": payment_envelope_dict, "tag": "v1.2.0"}
        resp = await app_client.post("/ingest", json=body)
        assert resp.status_code == 201
        assert resp.json()["uuid"] == payment_envelope_dict["uuid"]

    @pytest.mark.asyncio
    async def test_ingest_multiple_envelopes_different_tags(
        self, app_client, valid_envelope_dict, payment_envelope_dict
    ):
        resp1 = await app_client.post(
            "/ingest", json={"envelope": valid_envelope_dict, "tag": "v1.4.0"}
        )
        resp2 = await app_client.post(
            "/ingest", json={"envelope": payment_envelope_dict, "tag": "v1.2.0"}
        )
        assert resp1.status_code == 201
        assert resp2.status_code == 201


# ---------------------------------------------------------------------------
# POST /ingest/raw tests
# ---------------------------------------------------------------------------

class TestIngestRawEndpoint:

    @pytest.mark.asyncio
    async def test_ingest_raw_valid(self, app_client, valid_envelope_dict):
        # Use a different uuid to avoid 409 conflict with other tests
        raw = copy.deepcopy(valid_envelope_dict)
        raw["uuid"] = "raw-test-uuid-001"
        resp = await app_client.post("/ingest/raw", json=raw)
        assert resp.status_code == 201
        assert resp.json()["uuid"] == "raw-test-uuid-001"
        assert resp.json()["tag"] is None   # raw endpoint never sets a tag

    @pytest.mark.asyncio
    async def test_ingest_raw_malformed_returns_422(self, app_client):
        resp = await app_client.post("/ingest/raw", json={"not": "an envelope"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Health / root
# ---------------------------------------------------------------------------

class TestHealthEndpoints:

    @pytest.mark.asyncio
    async def test_root_returns_200(self, app_client):
        resp = await app_client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "endpoints" in data

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, app_client):
        resp = await app_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"