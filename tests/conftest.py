"""
conftest.py — shared pytest fixtures for the interfere eval harness test suite.

Place this file in the tests/ directory alongside the individual test files.
Run any single test file with:
    pytest tests/test_models.py -v
Run the full suite with:
    pytest tests/ -v

Requires a .env file in the project root with at minimum:
    MONGODB_URI, GEMINI_API_KEY (or USE_VERTEX=true + GCP_PROJECT), REPO_PATH
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Make the project root importable from tests/
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from project root before any fixture or test touches os.environ
load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Minimal valid envelope dict — reused across multiple test files
# ---------------------------------------------------------------------------

VALID_ENVELOPE_DICT = {
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
        # FIX: 'page' was removed from RouteType in models.py when the schema was
        # updated to match the real @interfere/types zod enum.
        # Real values: render | route | action | middleware | proxy
        # Dashboard page components map to "render" (RSC / server-rendering).
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


@pytest.fixture
def valid_envelope_dict() -> dict:
    """Return a copy of the canonical valid envelope dict."""
    import copy
    return copy.deepcopy(VALID_ENVELOPE_DICT)


@pytest.fixture
def payment_envelope_dict() -> dict:
    """Envelope for the v1.2.0 Amex card rejection bug."""
    return {
        "uuid": "test-uuid-payment-001",
        "v": 0,
        "buildId": "test-build-v1.2.0",
        "releaseId": "v1.2.0",
        "clientTs": 1719316300000,
        "sessionId": "server_test-session-002",
        "sessionSource": "header",
        "runtime": "node",
        "environment": "production",
        "type": "error",
        "context": {
            "runtime": "nextjs",
            "routePath": "/api/payment",
            # API route handlers correctly use "route" — unchanged.
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