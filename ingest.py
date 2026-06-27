"""
ingest.py — POST /ingest and POST /ingest/raw endpoints.

Accepts Interfere SDK error envelopes, validates them against the Pydantic
schema in models.py, and persists them to MongoDB via db.py.

Endpoints:
  POST /ingest        — accepts { envelope: Envelope, tag?: str }
                        Returns 201 IngestResponse on success.
                        Returns 409 if the UUID was already ingested.
                        Returns 422 if the envelope fails Pydantic validation.

  POST /ingest/raw    — accepts a bare Envelope dict (no wrapper object).
                        Tag is always None (no git tag context at this level).
                        Same status codes as /ingest.

Both endpoints are idempotent in the sense that re-sending the same envelope
returns 409 rather than silently creating a duplicate — the uuid unique index
in MongoDB enforces this.

Environment variables (read by db.py, not directly here):
  MONGODB_URI — MongoDB Atlas connection string
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from pymongo.errors import DuplicateKeyError

from db import insert_event
from models import Envelope, IngestRequest, IngestResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest an error envelope with an optional git tag",
    description=(
        "Validates the envelope against the @interfere/types schema and "
        "persists it to MongoDB. The optional `tag` field ties the event "
        "to a specific git tag for later attribution."
    ),
)
async def ingest_envelope(body: IngestRequest) -> IngestResponse:
    """
    Ingest a validated envelope alongside its git tag.

    The tag is stored alongside the envelope so /attribute can later fetch
    the right diff (v1.0.0-clean → tag) when running attribution.
    """
    try:
        uuid = await insert_event(body.envelope, tag=body.tag)
    except DuplicateKeyError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Event {body.envelope.uuid!r} already ingested.",
        )

    logger.info(
        "Ingested envelope uuid=%s tag=%s route=%s",
        uuid, body.tag, body.envelope.route,
    )

    return IngestResponse(
        received=True,
        uuid=uuid,
        tag=body.tag,
    )


# ---------------------------------------------------------------------------
# POST /ingest/raw
# ---------------------------------------------------------------------------

@router.post(
    "/ingest/raw",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a bare envelope dict (no tag wrapper)",
    description=(
        "Accepts the raw SDK envelope JSON directly — no IngestRequest wrapper. "
        "Used when the SDK posts directly to this endpoint without a tag. "
        "Tag is always None for events received this way."
    ),
)
async def ingest_raw(envelope: Envelope) -> IngestResponse:
    """
    Ingest a bare envelope with no git tag context.

    The @interfere/next SDK posts directly to this endpoint when it doesn't
    have a build-time tag available. Attribution can still be attempted later
    if the buildId can be correlated to a tag externally.
    """
    try:
        uuid = await insert_event(envelope, tag=None)
    except DuplicateKeyError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Event {envelope.uuid!r} already ingested.",
        )

    logger.info(
        "Ingested raw envelope uuid=%s route=%s",
        uuid, envelope.route,
    )

    return IngestResponse(
        received=True,
        uuid=uuid,
        tag=None,
    )