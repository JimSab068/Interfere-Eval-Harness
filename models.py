
"""
models.py — Pydantic models mirroring the REAL @interfere/types schema.

Verified directly against:
  node_modules/@interfere/types/dist/sdk/envelope.d.mts
  node_modules/@interfere/types/dist/sdk/plugins/payload/errors.d.mts
  node_modules/@interfere/types/dist/sdk/plugins/context/next.d.mts
  node_modules/@interfere/types/dist/data/frame.d.mts

Key facts from the real zod schemas:
  - context is OPTIONAL and a discriminated union on runtime
  - sessionId is NULLABLE (str | None)
  - runtime (top-level) is NULLABLE enum
  - environment is NULLABLE enum
  - releaseId is NULLABLE str
  - sessionSource is OPTIONAL (can be absent)
  - mechanism is OPTIONAL on ExceptionValue
  - all Frame fields are optional; real fields are fileName, functionName,
    lineNumber, columnNumber, source (no contextLine/preContext/postContext)
  - mechanism has synthetic: Optional[bool], no description/data
  - RouteType real values: render, route, action, middleware, proxy
  - SessionSource real values: client, header, async_context, fallback
  - EnvelopeType real values: error, pageview, pageleave, ui_event,
    replay_chunk, rage_click
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums — verified against real zod schemas
# ---------------------------------------------------------------------------

class SessionSource(str, Enum):
    client        = "client"
    header        = "header"
    async_context = "async_context"
    fallback      = "fallback"


class Runtime(str, Enum):
    browser = "browser"
    node    = "node"
    edge    = "edge"


class Environment(str, Enum):
    development = "development"
    preview     = "preview"
    production  = "production"


class RouteType(str, Enum):
    """Real values from nextjsContextSchema.routeType."""
    render     = "render"
    route      = "route"
    action     = "action"
    middleware = "middleware"
    proxy      = "proxy"


class RouterKind(str, Enum):
    pages_router = "Pages Router"
    app_router   = "App Router"


class RenderSource(str, Enum):
    react_server_components         = "react-server-components"
    react_server_components_payload = "react-server-components-payload"
    server_rendering                = "server-rendering"


class RenderType(str, Enum):
    dynamic        = "dynamic"
    dynamic_resume = "dynamic-resume"


class RevalidateReason(str, Enum):
    on_demand = "on-demand"
    stale     = "stale"


class EnvelopeType(str, Enum):
    """Real discriminated union types from envelopeSchema."""
    error        = "error"
    pageview     = "pageview"
    pageleave    = "pageleave"
    ui_event     = "ui_event"
    replay_chunk = "replay_chunk"
    rage_click   = "rage_click"


# ---------------------------------------------------------------------------
# Stack frame — matches ingestedFrame / exceptionValueSchema.frames exactly
# ---------------------------------------------------------------------------

class Frame(BaseModel):
    """
    Single stack frame.
    All fields are optional per the real zod schema.
    Real fields: fileName, functionName, lineNumber, columnNumber, source.
    """
    fileName:     Optional[str] = None
    functionName: Optional[str] = None
    lineNumber:   Optional[int] = None
    columnNumber: Optional[int] = None
    source:       Optional[str] = None   # raw source line from source map

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Mechanism — matches errorMechanismSchema exactly
# ---------------------------------------------------------------------------

class Mechanism(BaseModel):
    """
    Real fields: type (str), handled (bool), synthetic (Optional[bool]).
    No description, no data — those were fabricated in the old model.
    """
    type:      str            = "instrument"
    handled:   bool           = False
    synthetic: Optional[bool] = None

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Exception value — matches exceptionValueSchema exactly
# ---------------------------------------------------------------------------

class ExceptionValue(BaseModel):
    """
    Real schema: type, value required; mechanism OPTIONAL; frames required array.
    """
    type:      str                  = Field(..., description="JS error constructor, e.g. 'TypeError'")
    value:     str                  = Field(..., description="Error message string")
    mechanism: Optional[Mechanism]  = None          # OPTIONAL in real schema
    frames:    list[Frame]          = Field(default_factory=list)

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Error payload — matches errorEnvelopePayloadSchema exactly
# ---------------------------------------------------------------------------

class ErrorPayload(BaseModel):
    exceptions: list[ExceptionValue] = Field(..., min_length=1)

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Context variants — discriminated union on `runtime`
# ---------------------------------------------------------------------------

class NextjsContext(BaseModel):
    """
    Matches nextjsContextSchema exactly.
    All fields except runtime are optional.
    """
    runtime:          Literal["nextjs"]           = "nextjs"
    routePath:        Optional[str]               = None
    routeType:        Optional[RouteType]         = None
    requestMethod:    Optional[str]               = None
    requestPath:      Optional[str]               = None
    routerKind:       Optional[RouterKind]        = None
    renderSource:     Optional[RenderSource]      = None
    renderType:       Optional[RenderType]        = None
    revalidateReason: Optional[RevalidateReason]  = None
    errorDigest:      Optional[str]               = None

    model_config = {"extra": "allow"}


class BrowserContext(BaseModel):
    """Minimal browser context — we don't use this in attribution but must accept it."""
    runtime: Literal["browser"] = "browser"

    model_config = {"extra": "allow"}


class EdgeContext(BaseModel):
    """Minimal edge context."""
    runtime: Literal["edge"] = "edge"

    model_config = {"extra": "allow"}


# Discriminated union matching the real envelopeContextSchema
EnvelopeContext = Union[NextjsContext, BrowserContext, EdgeContext]


# ---------------------------------------------------------------------------
# Top-level Envelope
# ---------------------------------------------------------------------------

class Envelope(BaseModel):
    """
    Top-level event envelope matching the real @interfere/types envelopeSchema.

    Critical corrections from the real zod schema:
      - context: OPTIONAL (absent for some events)
      - sessionId: NULLABLE (str | None)
      - runtime: NULLABLE enum (Runtime | None)
      - environment: NULLABLE enum (Environment | None)
      - releaseId: NULLABLE str (str | None)
      - sessionSource: OPTIONAL (can be absent entirely)
      - type: extended enum with all 6 real event types
    """

    # Identity
    uuid: str = Field(..., description="Unique event ID (UUID format per real schema)")
    v:    int = Field(0,   description="Schema version; currently always 0")

    # Build / release
    buildId:   str           = Field(..., description="Next.js build ID")
    releaseId: Optional[str] = Field(None, description="Nullable release identifier")

    # Timing
    clientTs: int = Field(..., description="Unix timestamp ms when event occurred")

    # Session — sessionId is NULLABLE in real schema
    sessionId:     Optional[str]           = Field(None, description="UUID or 'server_<str>' or null")
    sessionSource: Optional[SessionSource] = Field(None, description="Optional — how sessionId was resolved")

    # Runtime / environment — both NULLABLE in real schema
    runtime:     Optional[Runtime]     = None
    environment: Optional[Environment] = None

    # Event type
    type: EnvelopeType = Field(EnvelopeType.error)

    # Context — OPTIONAL in real schema
    context: Optional[EnvelopeContext] = Field(
        None,
        description="Runtime context; discriminated union on runtime field",
    )

    # Payload — required for error envelopes; we focus on ErrorPayload
    payload: ErrorPayload = Field(..., description="Error payload (type='error')")

    model_config = {
        "extra": "allow",
        "populate_by_name": True,
    }

    # ------------------------------------------------------------------
    # Convenience helpers used by ingest.py / attribute.py
    # ------------------------------------------------------------------

    @property
    def primary_exception(self) -> ExceptionValue:
        return self.payload.exceptions[-1]

    @property
    def error_type(self) -> str:
        return self.primary_exception.type

    @property
    def error_message(self) -> str:
        return self.primary_exception.value

    @property
    def route(self) -> Optional[str]:
        if isinstance(self.context, NextjsContext):
            return self.context.routePath
        return None

    @property
    def top_frame(self) -> Optional[Frame]:
        frames = self.primary_exception.frames
        return frames[-1] if frames else None

    @property
    def nextjs_context(self) -> Optional[NextjsContext]:
        """Return the context only if it's a Next.js context, else None."""
        if isinstance(self.context, NextjsContext):
            return self.context
        return None


# ---------------------------------------------------------------------------
# Request / response models for FastAPI endpoints
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    envelope: Envelope
    tag: Optional[str] = Field(None, description="Git tag e.g. 'v1.4.0'")


class IngestResponse(BaseModel):
    received: bool = True
    uuid:     str
    tag:      Optional[str] = None


class AttributeRequest(BaseModel):
    envelope: Envelope
    tag:      str = Field(..., description="Git tag to diff against v1.0.0-clean")


class AttributeResponse(BaseModel):
    culprit_file:     Optional[str] = None
    culprit_function: Optional[str] = None
    confidence:       float         = Field(..., ge=0.0, le=1.0)
    explanation:      str
    reasoning_trace:  list[str]     = Field(default_factory=list)


class EvalCase(BaseModel):
    bug_id:                        str
    tag:                           str
    envelope_type:                 str           = "error"
    error_type:                    str
    route:                         str
    ground_truth_file:             str
    ground_truth_function:         str
    ground_truth_commit_introduced: str
    ground_truth_data_file:        Optional[str] = None
    envelope:                      Optional[Envelope] = None
    prev_tag: Optional[str] = None


class EvalCaseResult(BaseModel):
    bug_id:            str
    tag:               str
    predicted_file:    Optional[str]
    predicted_function: Optional[str]
    ground_truth_file: str
    ground_truth_function: str
    file_match:        bool
    function_match:    bool
    explanation_correct: Optional[bool] = None
    explanation:       str
    reasoning_trace:   list[str]


class EvalReport(BaseModel):
    total_cases:            int
    accuracy_file:          float           = Field(..., description="Fraction with correct file")
    accuracy_function:      float           = Field(..., description="Fraction with correct function")
    accuracy_explanation:   Optional[float] = None
    cases:                  list[EvalCaseResult]