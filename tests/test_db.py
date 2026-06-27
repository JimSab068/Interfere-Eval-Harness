"""
test_db.py — Integration tests for db.py.

Requires a live MongoDB Atlas connection (MONGODB_URI in .env).
Uses a separate "interfere_test" database so it never touches production data.
Cleans up after itself.

Run:
    pytest tests/test_db.py -v
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from models import Envelope


# ---------------------------------------------------------------------------
# Override the DB name to use a test database
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(autouse=True)
async def use_test_db(monkeypatch):
    """
    Patch db.py to use 'interfere_test' instead of 'interfere'.
    This keeps test writes out of the production collection.

    FIX: Drop the collection BEFORE creating indexes, not just after.
    Atlas retains indexes on collections between test runs even after drop
    if a previous run crashed before teardown.  The stale 'uuid_1' index
    (created by a previous fixture run without an explicit name) conflicts
    with our 'uuid_unique' named index.  Dropping first gives us a clean slate.
    """
    import db as db_module
    import os
    from motor.motor_asyncio import AsyncIOMotorClient
    from pymongo import ASCENDING, IndexModel

    uri = os.environ.get("MONGODB_URI")
    if not uri:
        pytest.skip("MONGODB_URI not set — skipping DB integration tests")

    client = AsyncIOMotorClient(uri)
    test_db = client["interfere_test"]
    collection = test_db["ingested_events"]

    # Drop first to clear any stale indexes from previous crashed runs
    await collection.drop()

    # Inject the test collection into the module
    db_module._client = client
    db_module._collection = collection

    await collection.create_indexes([
        IndexModel([("uuid", ASCENDING)], unique=True, name="uuid_unique"),
        IndexModel([("tag", ASCENDING)], name="tag"),
        IndexModel([("tag", ASCENDING), ("route", ASCENDING)], name="tag_route"),
    ])

    yield

    # Teardown — drop the test collection
    await collection.drop()
    client.close()
    db_module._client = None
    db_module._collection = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInsertEvent:

    @pytest.mark.asyncio
    async def test_insert_returns_uuid(self, valid_envelope_dict):
        from db import insert_event
        env = Envelope(**valid_envelope_dict)
        result_uuid = await insert_event(env, tag="v1.4.0")
        assert result_uuid == env.uuid

    @pytest.mark.asyncio
    async def test_insert_stores_flattened_fields(self, valid_envelope_dict):
        from db import insert_event, get_collection
        env = Envelope(**valid_envelope_dict)
        await insert_event(env, tag="v1.4.0")

        doc = await get_collection().find_one({"uuid": env.uuid})
        assert doc is not None
        assert doc["tag"] == "v1.4.0"
        assert doc["route"] == "/dashboard"
        assert doc["error_type"] == "TypeError"
        assert "toFixed" in doc["error_message"]

    @pytest.mark.asyncio
    async def test_insert_stores_full_envelope(self, valid_envelope_dict):
        from db import insert_event, get_collection
        env = Envelope(**valid_envelope_dict)
        await insert_event(env, tag="v1.4.0")

        doc = await get_collection().find_one({"uuid": env.uuid})
        assert "envelope" in doc
        assert doc["envelope"]["uuid"] == env.uuid
        assert doc["envelope"]["payload"]["exceptions"][0]["type"] == "TypeError"

    @pytest.mark.asyncio
    async def test_insert_stores_created_at(self, valid_envelope_dict):
        from db import insert_event, get_collection
        from datetime import datetime
        env = Envelope(**valid_envelope_dict)
        await insert_event(env, tag="v1.4.0")

        doc = await get_collection().find_one({"uuid": env.uuid})
        assert isinstance(doc["created_at"], datetime)

    @pytest.mark.asyncio
    async def test_duplicate_uuid_raises(self, valid_envelope_dict):
        from db import insert_event
        from pymongo.errors import DuplicateKeyError
        env = Envelope(**valid_envelope_dict)
        await insert_event(env, tag="v1.4.0")
        with pytest.raises(DuplicateKeyError):
            await insert_event(env, tag="v1.4.0")

    @pytest.mark.asyncio
    async def test_insert_without_tag(self, valid_envelope_dict):
        from db import insert_event, get_collection
        env = Envelope(**valid_envelope_dict)
        await insert_event(env, tag=None)

        doc = await get_collection().find_one({"uuid": env.uuid})
        assert doc["tag"] is None


class TestGetEventByTagAndRoute:

    @pytest.mark.asyncio
    async def test_retrieves_correct_event(self, valid_envelope_dict):
        from db import insert_event, get_event_by_tag_and_route
        env = Envelope(**valid_envelope_dict)
        await insert_event(env, tag="v1.4.0")

        doc = await get_event_by_tag_and_route("v1.4.0", "/dashboard")
        assert doc is not None
        assert doc["uuid"] == env.uuid

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        from db import get_event_by_tag_and_route
        doc = await get_event_by_tag_and_route("v9.9.9", "/nonexistent")
        assert doc is None

    @pytest.mark.asyncio
    async def test_returns_most_recent_on_duplicate(self, valid_envelope_dict):
        """When two events share tag+route, most recent is returned."""
        import copy
        import asyncio
        from db import insert_event, get_event_by_tag_and_route

        dict1 = copy.deepcopy(valid_envelope_dict)
        dict1["uuid"] = "uuid-older"
        dict2 = copy.deepcopy(valid_envelope_dict)
        dict2["uuid"] = "uuid-newer"

        env1 = Envelope(**dict1)
        env2 = Envelope(**dict2)

        await insert_event(env1, tag="v1.4.0")
        # Small sleep to ensure created_at differs
        await asyncio.sleep(0.05)
        await insert_event(env2, tag="v1.4.0")

        doc = await get_event_by_tag_and_route("v1.4.0", "/dashboard")
        assert doc["uuid"] == "uuid-newer"


class TestGetEventByUuid:

    @pytest.mark.asyncio
    async def test_retrieves_by_uuid(self, valid_envelope_dict):
        from db import insert_event, get_event_by_uuid
        env = Envelope(**valid_envelope_dict)
        await insert_event(env, tag="v1.4.0")

        doc = await get_event_by_uuid(env.uuid)
        assert doc is not None
        assert doc["uuid"] == env.uuid

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown_uuid(self):
        from db import get_event_by_uuid
        doc = await get_event_by_uuid("nonexistent-uuid-xyz")
        assert doc is None


class TestListEventsByTag:

    @pytest.mark.asyncio
    async def test_lists_all_events_for_tag(
        self, valid_envelope_dict, payment_envelope_dict
    ):
        import copy
        from db import insert_event, list_events_by_tag

        env1 = Envelope(**valid_envelope_dict)
        env2 = Envelope(**payment_envelope_dict)

        await insert_event(env1, tag="v1.4.0")
        await insert_event(env2, tag="v1.4.0")

        docs = await list_events_by_tag("v1.4.0")
        uuids = [d["uuid"] for d in docs]
        assert env1.uuid in uuids
        assert env2.uuid in uuids

    @pytest.mark.asyncio
    async def test_empty_list_for_unknown_tag(self):
        from db import list_events_by_tag
        docs = await list_events_by_tag("v9.9.9")
        assert docs == []