"""
db.py — MongoDB async client setup using Motor.

Provides:
  - A single shared AsyncIOMotorClient (created once on startup)
  - Collection references used by ingest.py and eval.py
  - insert_event() — stores a validated envelope + tag to MongoDB
  - get_event_by_tag_and_route() — fetches one envelope for eval.py
  - Indexes created on startup to keep eval queries fast

Connection string is read from MONGODB_URI in the environment.
Database name: "interfere"
Collection: "ingested_events"

Document shape stored:
{
  "_id": ObjectId (auto),
  "uuid": str,
  "tag": str,                    # git tag passed alongside envelope
  "route": str,                  # envelope.context.routePath
  "error_type": str,             # envelope.payload.exceptions[-1].type
  "error_message": str,          # envelope.payload.exceptions[-1].value
  "created_at": datetime,
  "envelope": { ...full envelope dict... }
}
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo import ASCENDING, IndexModel

from models import Envelope

# ---------------------------------------------------------------------------
# Module-level client — initialised once by init_db() called from main.py
# ---------------------------------------------------------------------------

_client: Optional[AsyncIOMotorClient] = None
_collection: Optional[AsyncIOMotorCollection] = None


def get_collection() -> AsyncIOMotorCollection:
    if _collection is None:
        raise RuntimeError("Database not initialised. Call init_db() on startup.")
    return _collection


# ---------------------------------------------------------------------------
# Startup / teardown (called from main.py lifespan)
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """
    Create the Motor client, select the collection, and ensure indexes exist.
    Safe to call multiple times — Motor is idempotent on index creation.
    """
    global _client, _collection

    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise EnvironmentError(
            "MONGODB_URI is not set. Add it to your .env file.\n"
            "Example: MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/interfere"
        )

    _client = AsyncIOMotorClient(uri)
    db = _client["interfere"]
    _collection = db["ingested_events"]

    # Create indexes so eval queries (by tag + route) are fast.
    # create_indexes is idempotent — safe to run on every startup.
    await _collection.create_indexes([
        IndexModel([("uuid", ASCENDING)], unique=True, name="uuid_unique"),
        IndexModel([("tag", ASCENDING)], name="tag"),
        IndexModel([("tag", ASCENDING), ("route", ASCENDING)], name="tag_route"),
    ])


async def close_db() -> None:
    """Close the Motor connection. Called from main.py lifespan on shutdown."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

async def insert_event(envelope: Envelope, tag: Optional[str] = None) -> str:
    """
    Persist a validated Envelope to MongoDB.

    Stores the full envelope dict under the "envelope" key so nothing is lost,
    plus flattened top-level fields for easy querying.

    Returns the envelope UUID (not the Mongo ObjectId).

    Raises:
        DuplicateKeyError if an event with this UUID was already ingested.
    """
    collection = get_collection()

    doc = {
        "uuid": envelope.uuid,
        "tag": tag,
        "route": envelope.route,                  # context.routePath
        "error_type": envelope.error_type,        # exceptions[-1].type
        "error_message": envelope.error_message,  # exceptions[-1].value
        "created_at": datetime.now(tz=timezone.utc),
        "envelope": envelope.model_dump(),        # full envelope for later retrieval
    }

    await collection.insert_one(doc)
    return envelope.uuid


# ---------------------------------------------------------------------------
# Read (used by eval.py to retrieve envelopes for each ground-truth case)
# ---------------------------------------------------------------------------

async def get_event_by_tag_and_route(
    tag: str,
    route: str,
) -> Optional[dict]:
    """
    Fetch the most recently ingested event matching a git tag and route path.

    eval.py calls this to reconstruct the Envelope for each ground-truth case
    rather than hard-coding envelopes in eval_cases.json.

    Returns the raw MongoDB document (with "envelope" sub-dict) or None.
    """
    collection = get_collection()
    return await collection.find_one(
        {"tag": tag, "route": route},
        sort=[("created_at", -1)],   # most recent first if duplicates exist
    )


async def get_event_by_uuid(uuid: str) -> Optional[dict]:
    """Fetch a single event by its SDK-assigned UUID. Useful for debugging."""
    collection = get_collection()
    return await collection.find_one({"uuid": uuid})


async def list_events_by_tag(tag: str) -> list[dict]:
    """
    Return all ingested events for a given tag.
    Useful for inspecting what was captured before running /eval.
    """
    collection = get_collection()
    cursor = collection.find({"tag": tag}, sort=[("created_at", ASCENDING)])
    return await cursor.to_list(length=None)