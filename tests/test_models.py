"""
test_models.py — Unit tests for models.py Pydantic schemas.

Tests the Envelope and sub-model validation logic against the real
@interfere/types schema as encoded in models.py.

No external services required — pure Pydantic validation, no I/O.

Run:
    pytest tests/test_models.py -v
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from models import (
    BrowserContext,
    EdgeContext,
    Envelope,
    EnvelopeType,
    Environment,
    ErrorPayload,
    ExceptionValue,
    Frame,
    Mechanism,
    NextjsContext,
    RouteType,
    RouterKind,
    Runtime,
    SessionSource,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_envelope(**overrides) -> dict:
    """Return the smallest dict that passes Envelope validation."""
    base = {
        "uuid": "test-uuid-001",
        "buildId": "build-abc123",
        "clientTs": 1719316200000,
        "payload": {
            "exceptions": [
                {"type": "TypeError", "value": "Cannot read properties of null"}
            ]
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------

class TestFrame:

    def test_all_optional(self):
        f = Frame()
        assert f.fileName is None
        assert f.functionName is None
        assert f.lineNumber is None
        assert f.columnNumber is None
        assert f.source is None

    def test_full_frame(self):
        f = Frame(
            fileName="app/api/auth/route.ts",
            functionName="POST",
            lineNumber=34,
            columnNumber=12,
            source="  throw new Error('Session expired')",
        )
        assert f.fileName == "app/api/auth/route.ts"
        assert f.functionName == "POST"
        assert f.lineNumber == 34

    def test_extra_fields_allowed(self):
        """Frame uses extra='allow' for forward compat with new SDK versions."""
        f = Frame(fileName="app/page.tsx", unknownSdkField="future")
        assert f.fileName == "app/page.tsx"


# ---------------------------------------------------------------------------
# Mechanism
# ---------------------------------------------------------------------------

class TestMechanism:

    def test_defaults(self):
        m = Mechanism()
        assert m.type == "instrument"
        assert m.handled is False
        assert m.synthetic is None

    def test_custom_type(self):
        m = Mechanism(type="onerror", handled=True, synthetic=False)
        assert m.type == "onerror"
        assert m.handled is True
        assert m.synthetic is False

    def test_extra_fields_allowed(self):
        m = Mechanism(type="generic", futureField=42)
        assert m.type == "generic"


# ---------------------------------------------------------------------------
# ExceptionValue
# ---------------------------------------------------------------------------

class TestExceptionValue:

    def test_required_fields(self):
        exc = ExceptionValue(type="TypeError", value="null is not an object")
        assert exc.type == "TypeError"
        assert exc.value == "null is not an object"
        assert exc.mechanism is None   # optional in real schema
        assert exc.frames == []

    def test_with_frames(self):
        exc = ExceptionValue(
            type="Error",
            value="Session expired",
            frames=[
                {"fileName": "app/api/auth/route.ts", "functionName": "POST", "lineNumber": 34}
            ],
        )
        assert len(exc.frames) == 1
        assert exc.frames[0].fileName == "app/api/auth/route.ts"

    def test_missing_type_raises(self):
        with pytest.raises(ValidationError):
            ExceptionValue(value="some error")   # type required

    def test_missing_value_raises(self):
        with pytest.raises(ValidationError):
            ExceptionValue(type="Error")          # value required


# ---------------------------------------------------------------------------
# ErrorPayload
# ---------------------------------------------------------------------------

class TestErrorPayload:

    def test_valid_payload(self):
        p = ErrorPayload(
            exceptions=[{"type": "TypeError", "value": "null ref"}]
        )
        assert len(p.exceptions) == 1

    def test_empty_exceptions_raises(self):
        """min_length=1 means an empty list must fail."""
        with pytest.raises(ValidationError):
            ErrorPayload(exceptions=[])

    def test_missing_exceptions_raises(self):
        with pytest.raises(ValidationError):
            ErrorPayload()


# ---------------------------------------------------------------------------
# NextjsContext
# ---------------------------------------------------------------------------

class TestNextjsContext:

    def test_defaults(self):
        ctx = NextjsContext()
        assert ctx.runtime == "nextjs"
        assert ctx.routePath is None
        assert ctx.routeType is None
        assert ctx.requestMethod is None

    def test_api_route(self):
        ctx = NextjsContext(
            routePath="/api/payment",
            routeType="route",
            requestMethod="POST",
            routerKind="App Router",
        )
        assert ctx.routeType == RouteType.route
        assert ctx.routerKind == RouterKind.app_router

    def test_render_route_type(self):
        """'render' replaced 'page' in the real schema."""
        ctx = NextjsContext(routeType="render")
        assert ctx.routeType == RouteType.render

    def test_invalid_route_type_raises(self):
        """`page` is not a valid RouteType in the updated schema."""
        with pytest.raises(ValidationError):
            NextjsContext(routeType="page")

    def test_all_route_types_valid(self):
        for rt in ("render", "route", "action", "middleware", "proxy"):
            ctx = NextjsContext(routeType=rt)
            assert ctx.routeType.value == rt

    def test_extra_fields_allowed(self):
        ctx = NextjsContext(futureField="xyz")
        assert ctx.runtime == "nextjs"


# ---------------------------------------------------------------------------
# Context union discrimination
# ---------------------------------------------------------------------------

class TestContextUnion:

    def test_nextjs_context_discriminated(self):
        env = Envelope(**_minimal_envelope(
            context={"runtime": "nextjs", "routePath": "/api/auth"}
        ))
        assert isinstance(env.context, NextjsContext)
        assert env.context.routePath == "/api/auth"

    def test_browser_context_discriminated(self):
        env = Envelope(**_minimal_envelope(
            context={"runtime": "browser"}
        ))
        assert isinstance(env.context, BrowserContext)

    def test_edge_context_discriminated(self):
        env = Envelope(**_minimal_envelope(
            context={"runtime": "edge"}
        ))
        assert isinstance(env.context, EdgeContext)

    def test_context_absent(self):
        """context is Optional — envelope is valid without it."""
        env = Envelope(**_minimal_envelope())
        assert env.context is None


# ---------------------------------------------------------------------------
# Envelope — field validation
# ---------------------------------------------------------------------------

class TestEnvelopeRequiredFields:

    def test_minimal_valid_envelope(self):
        env = Envelope(**_minimal_envelope())
        assert env.uuid == "test-uuid-001"
        assert env.buildId == "build-abc123"
        assert env.clientTs == 1719316200000
        assert env.v == 0

    def test_missing_uuid_raises(self):
        d = _minimal_envelope()
        del d["uuid"]
        with pytest.raises(ValidationError):
            Envelope(**d)

    def test_missing_build_id_raises(self):
        d = _minimal_envelope()
        del d["buildId"]
        with pytest.raises(ValidationError):
            Envelope(**d)

    def test_missing_client_ts_raises(self):
        d = _minimal_envelope()
        del d["clientTs"]
        with pytest.raises(ValidationError):
            Envelope(**d)

    def test_missing_payload_raises(self):
        d = _minimal_envelope()
        del d["payload"]
        with pytest.raises(ValidationError):
            Envelope(**d)


class TestEnvelopeOptionalFields:

    def test_session_id_nullable(self):
        """sessionId is Optional[str] — None is valid."""
        env = Envelope(**_minimal_envelope(sessionId=None))
        assert env.sessionId is None

    def test_session_id_absent(self):
        """sessionId can be absent entirely."""
        env = Envelope(**_minimal_envelope())
        assert env.sessionId is None

    def test_runtime_nullable(self):
        env = Envelope(**_minimal_envelope(runtime=None))
        assert env.runtime is None

    def test_environment_nullable(self):
        env = Envelope(**_minimal_envelope(environment=None))
        assert env.environment is None

    def test_release_id_nullable(self):
        env = Envelope(**_minimal_envelope(releaseId=None))
        assert env.releaseId is None

    def test_session_source_optional(self):
        env = Envelope(**_minimal_envelope())
        assert env.sessionSource is None


class TestEnvelopeEnums:

    def test_valid_runtime_values(self):
        for rt in ("browser", "node", "edge"):
            env = Envelope(**_minimal_envelope(runtime=rt))
            assert env.runtime == Runtime(rt)

    def test_invalid_runtime_raises(self):
        with pytest.raises(ValidationError):
            Envelope(**_minimal_envelope(runtime="deno"))

    def test_valid_environment_values(self):
        for env_str in ("development", "preview", "production"):
            env = Envelope(**_minimal_envelope(environment=env_str))
            assert env.environment == Environment(env_str)

    def test_valid_session_source_values(self):
        for ss in ("client", "header", "async_context", "fallback"):
            env = Envelope(**_minimal_envelope(sessionSource=ss))
            assert env.sessionSource == SessionSource(ss)

    def test_old_session_source_cookie_raises(self):
        """'cookie' was in the old schema but not the real one."""
        with pytest.raises(ValidationError):
            Envelope(**_minimal_envelope(sessionSource="cookie"))

    def test_valid_envelope_types(self):
        for et in ("error", "pageview", "pageleave", "ui_event", "replay_chunk", "rage_click"):
            env = Envelope(**_minimal_envelope(type=et))
            assert env.type == EnvelopeType(et)

    def test_default_envelope_type_is_error(self):
        env = Envelope(**_minimal_envelope())
        assert env.type == EnvelopeType.error


class TestEnvelopeExtraFields:

    def test_unknown_top_level_field_allowed(self):
        """extra='allow' means forward-compat with new SDK fields."""
        env = Envelope(**_minimal_envelope(newSdkField="future_value"))
        assert env.uuid == "test-uuid-001"


# ---------------------------------------------------------------------------
# Envelope convenience properties
# ---------------------------------------------------------------------------

class TestEnvelopeProperties:

    def _make_env(self, **overrides) -> Envelope:
        d = {
            "uuid": "prop-test-001",
            "buildId": "build-props",
            "clientTs": 1000000,
            "context": {
                "runtime": "nextjs",
                "routePath": "/api/auth",
                "requestMethod": "POST",
            },
            "payload": {
                "exceptions": [
                    {
                        "type": "Error",
                        "value": "Session expired",
                        "frames": [
                            {"fileName": "app/api/auth/route.ts", "functionName": "POST", "lineNumber": 34}
                        ],
                    }
                ]
            },
        }
        d.update(overrides)
        return Envelope(**d)

    def test_primary_exception(self):
        env = self._make_env()
        assert env.primary_exception.type == "Error"
        assert env.primary_exception.value == "Session expired"

    def test_error_type(self):
        assert self._make_env().error_type == "Error"

    def test_error_message(self):
        assert self._make_env().error_message == "Session expired"

    def test_route_from_nextjs_context(self):
        assert self._make_env().route == "/api/auth"

    def test_route_none_without_nextjs_context(self):
        env = self._make_env(context={"runtime": "browser"})
        assert env.route is None

    def test_route_none_without_context(self):
        d = _minimal_envelope()
        env = Envelope(**d)
        assert env.route is None

    def test_top_frame(self):
        env = self._make_env()
        frame = env.top_frame
        assert frame is not None
        assert frame.fileName == "app/api/auth/route.ts"
        assert frame.functionName == "POST"

    def test_top_frame_none_when_no_frames(self):
        d = _minimal_envelope()   # no frames in minimal envelope
        env = Envelope(**d)
        assert env.top_frame is None

    def test_nextjs_context_property(self):
        env = self._make_env()
        ctx = env.nextjs_context
        assert ctx is not None
        assert isinstance(ctx, NextjsContext)
        assert ctx.routePath == "/api/auth"

    def test_nextjs_context_none_for_browser(self):
        env = self._make_env(context={"runtime": "browser"})
        assert env.nextjs_context is None

    def test_primary_exception_last_in_chain(self):
        """primary_exception returns the last (innermost) exception."""
        env = Envelope(**_minimal_envelope(
            payload={
                "exceptions": [
                    {"type": "Error", "value": "outer"},
                    {"type": "TypeError", "value": "inner"},
                ]
            }
        ))
        assert env.primary_exception.type == "TypeError"
        assert env.primary_exception.value == "inner"


# ---------------------------------------------------------------------------
# Real envelope shapes from the eval harness
# ---------------------------------------------------------------------------

class TestRealEnvelopeShapes:

    def test_bug1_auth_gmail_envelope(self):
        """v1.1.0 — Gmail auth bug envelope as used in e2e tests."""
        d = {
            "uuid": "e2e-bug1-auth-gmail-001",
            "v": 0,
            "buildId": "build-v1.1.0",
            "releaseId": "v1.1.0",
            "clientTs": 1719316100000,
            "sessionId": "server_e2e-session-001",
            "sessionSource": "header",
            "runtime": "node",
            "environment": "production",
            "type": "error",
            "context": {
                "runtime": "nextjs",
                "routePath": "/api/auth",
                "routeType": "route",
                "requestMethod": "POST",
                "requestPath": "/api/auth",
                "routerKind": "App Router",
            },
            "payload": {
                "exceptions": [
                    {
                        "type": "Error",
                        "value": "Session expired",
                        "mechanism": {"type": "instrument", "handled": False},
                        "frames": [
                            {
                                "fileName": "app/api/auth/route.ts",
                                "functionName": "POST",
                                "lineNumber": 34,
                                "columnNumber": 12,
                            }
                        ],
                    }
                ]
            },
        }
        env = Envelope(**d)
        assert env.uuid == "e2e-bug1-auth-gmail-001"
        assert env.route == "/api/auth"
        assert env.error_type == "Error"
        assert env.error_message == "Session expired"
        assert env.top_frame.fileName == "app/api/auth/route.ts"
        assert env.runtime == Runtime.node

    def test_bug2_payment_amex_envelope(self):
        """v1.2.0 — Amex card rejection bug envelope."""
        d = {
            "uuid": "e2e-bug2-payment-amex-001",
            "v": 0,
            "buildId": "build-v1.2.0",
            "releaseId": "v1.2.0",
            "clientTs": 1719316200000,
            "sessionId": "server_e2e-session-002",
            "sessionSource": "header",
            "runtime": "node",
            "environment": "production",
            "type": "error",
            "context": {
                "runtime": "nextjs",
                "routePath": "/api/payment",
                "routeType": "route",
                "requestMethod": "POST",
                "requestPath": "/api/payment",
                "routerKind": "App Router",
            },
            "payload": {
                "exceptions": [
                    {
                        "type": "Error",
                        "value": "Invalid card number length",
                        "mechanism": {"type": "instrument", "handled": False},
                        "frames": [
                            {
                                "fileName": "app/api/payment/route.ts",
                                "functionName": "POST",
                                "lineNumber": 28,
                                "columnNumber": 5,
                            }
                        ],
                    }
                ]
            },
        }
        env = Envelope(**d)
        assert env.route == "/api/payment"
        assert env.error_message == "Invalid card number length"
        assert env.top_frame.functionName == "POST"

    def test_dashboard_envelope_render_route_type(self):
        """Dashboard page uses routeType='render', not 'page'."""
        d = {
            "uuid": "test-uuid-dashboard-001",
            "v": 0,
            "buildId": "test-build-v1.4.0",
            "releaseId": "v1.4.0",
            "clientTs": 1719316200000,
            "sessionId": "server_test-session-001",
            "sessionSource": "header",
            "runtime": "node",
            "environment": "production",
            "type": "error",
            "context": {
                "runtime": "nextjs",
                "routePath": "/dashboard",
                "routeType": "render",
                "requestMethod": None,
                "requestPath": "/dashboard",
                "routerKind": "App Router",
            },
            "payload": {
                "exceptions": [
                    {
                        "type": "TypeError",
                        "value": "Cannot read properties of null (reading 'toFixed')",
                        "mechanism": {"type": "instrument", "handled": False},
                        "frames": [
                            {
                                "fileName": "app/dashboard/page.tsx",
                                "functionName": "Dashboard",
                                "lineNumber": 42,
                                "columnNumber": 18,
                            }
                        ],
                    }
                ]
            },
        }
        env = Envelope(**d)
        assert env.nextjs_context.routeType == RouteType.render
        assert env.error_type == "TypeError"