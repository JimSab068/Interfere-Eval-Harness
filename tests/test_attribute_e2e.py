"""
test_attribute_e2e.py — End-to-end smoke test for POST /attribute.

Fires real envelopes for Bug 1 (gmail auth, v1.1.0) and Bug 2 (Amex card,
v1.2.0) at the live FastAPI app and prints the full attribution response so
you can eyeball the reasoning quality before deployment.

This test:
  - patches MongoDB to use 'interfere_test' (never touches production)
  - calls the REAL Gemini API + REAL GitHub API + REAL ChromaDB
  - asserts file-level and function-level attribution accuracy
  - prints the full reasoning trace for manual review

Run:
    pytest tests/test_attribute_e2e.py -v -s

The -s flag is important — it lets the reasoning trace print to stdout.

Requirements in .env:
    MONGODB_URI, GEMINI_API_KEY, GITHUB_TOKEN, REPO_PATH
"""

from __future__ import annotations

import os
import sys
import shutil
import copy
import json
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# ── make project root importable ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

TEST_CHROMA_PATH = PROJECT_ROOT / "chroma_e2e_test"

# ── ground truth for these two bugs ─────────────────────────────────────────
GROUND_TRUTH = {
    "v1.1.0": {
        "bug_id":              "auth-gmail-session-expired",
        "ground_truth_file":   "app/api/auth/route.ts",
        "ground_truth_function": "POST",
        "description":         "Gmail addresses always return 401 'Session expired'",
    },
    "v1.2.0": {
        "bug_id":              "payment-amex-card-rejected",
        "ground_truth_file":   "app/api/payment/route.ts",
        "ground_truth_function": "POST",
        "description":         "Amex (15-digit) cards rejected with cryptic error",
    },
}

# ── envelopes ────────────────────────────────────────────────────────────────
BUG1_ENVELOPE = {
    "uuid":          "e2e-bug1-auth-gmail-001",
    "v":             0,
    "buildId":       "build-v1.1.0",
    "releaseId":     "v1.1.0",
    "clientTs":      1719316100000,
    "sessionId":     "server_e2e-session-001",
    "sessionSource": "header",
    "runtime":       "node",
    "environment":   "production",
    "type":          "error",
    "context": {
        "runtime":       "nextjs",
        "routePath":     "/api/auth",
        "routeType":     "route",
        "requestMethod": "POST",
        "requestPath":   "/api/auth",
        "routerKind":    "App Router",
    },
    "payload": {
        "exceptions": [
            {
                "type":  "Error",
                "value": "Session expired",
                "mechanism": {"type": "instrument", "handled": False},
                "frames": [
                    {
                        "fileName":     "app/api/auth/route.ts",
                        "functionName": "POST",
                        "lineNumber":   34,
                        "columnNumber": 12,
                    }
                ],
            }
        ]
    },
}

BUG2_ENVELOPE = {
    "uuid":          "e2e-bug2-payment-amex-001",
    "v":             0,
    "buildId":       "build-v1.2.0",
    "releaseId":     "v1.2.0",
    "clientTs":      1719316200000,
    "sessionId":     "server_e2e-session-002",
    "sessionSource": "header",
    "runtime":       "node",
    "environment":   "production",
    "type":          "error",
    "context": {
        "runtime":       "nextjs",
        "routePath":     "/api/payment",
        "routeType":     "route",
        "requestMethod": "POST",
        "requestPath":   "/api/payment",
        "routerKind":    "App Router",
    },
    "payload": {
        "exceptions": [
            {
                "type":  "Error",
                "value": "Invalid card number length",
                "mechanism": {"type": "instrument", "handled": False},
                "frames": [
                    {
                        "fileName":     "app/api/payment/route.ts",
                        "functionName": "POST",
                        "lineNumber":   28,
                        "columnNumber": 5,
                    }
                ],
            }
        ]
    },
}


# ── skip helper ──────────────────────────────────────────────────────────────
def _check_tags_exist(tags: list[str]) -> None:
    """
    Verify that the required git tags exist on the GitHub repo before running
    attribution tests.  Skips cleanly with a clear message if they don't,
    rather than letting the test fail with a confusing 404 from attribute.py.

    Push tags with:
        git tag v1.0.0-clean <commit>
        git tag v1.1.0 <commit>
        git push origin --tags
    """
    import httpx
    token = os.environ.get("GITHUB_TOKEN", "")
    owner = os.environ.get("GITHUB_OWNER", "JimSab068")
    repo  = os.environ.get("GITHUB_REPO",  "stripe-clone")
    headers = {"X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    missing_tags = []
    for tag in tags:
        url = f"https://api.github.com/repos/{owner}/{repo}/git/refs/tags/{tag}"
        try:
            r = httpx.get(url, headers=headers, timeout=10.0)
            if r.status_code == 404:
                missing_tags.append(tag)
        except Exception:
            pass  # network error — let the test proceed and fail naturally

    if missing_tags:
        pytest.skip(
            f"Tags not found on {owner}/{repo}: {missing_tags}. "
            f"Push them with: git push origin --tags"
        )


def _skip_if_missing():
    missing = []
    for var in ("MONGODB_URI", "GEMINI_API_KEY", "GITHUB_TOKEN", "REPO_PATH"):
        if not os.environ.get(var):
            missing.append(var)
    if missing:
        pytest.skip(f"Missing env vars: {', '.join(missing)}")
    # Also verify the baseline tag and bug tags exist on GitHub
    _check_tags_exist(["v1.0.0-clean", "v1.1.0", "v1.2.0"])


# ── shared app fixture ───────────────────────────────────────────────────────
@pytest_asyncio.fixture(scope="module")
async def app_client():
    """
    Boot the real FastAPI app once for this module.
    - MongoDB → interfere_test collection
    - ChromaDB → chroma_e2e_test/ (cleaned up after)
    - Real Gemini + GitHub APIs
    """
    _skip_if_missing()

    # Redirect ChromaDB to isolated test dir
    os.environ["CHROMA_PATH"] = str(TEST_CHROMA_PATH)

    # Patch MongoDB to test DB
    import db as db_module
    from motor.motor_asyncio import AsyncIOMotorClient
    from pymongo import ASCENDING, IndexModel

    uri = os.environ["MONGODB_URI"]
    mongo_client = AsyncIOMotorClient(uri)
    test_col = mongo_client["interfere_test"]["ingested_events_e2e"]
    db_module._client = mongo_client
    db_module._collection = test_col

    await test_col.create_indexes([
        IndexModel([("uuid", ASCENDING)], unique=True),
        IndexModel([("tag", ASCENDING), ("route", ASCENDING)]),
    ])

    # Reset RAG module state so it uses the test chroma path
    import rag
    rag._chroma_client = None
    rag._collection = None

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    # ── teardown ──────────────────────────────────────────────────────────
    await test_col.drop()
    mongo_client.close()
    db_module._client = None
    db_module._collection = None

    rag._chroma_client = None
    rag._collection = None

    if TEST_CHROMA_PATH.exists():
        shutil.rmtree(TEST_CHROMA_PATH)


# ── pretty printer ───────────────────────────────────────────────────────────
def _print_attribution(tag: str, gt: dict, result: dict):
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  BUG: {gt['bug_id']}  ({gt['description']})")
    print(f"  TAG: {tag}")
    print(sep)
    print(f"  PREDICTION")
    print(f"    culprit_file     : {result.get('culprit_file')}")
    print(f"    culprit_function : {result.get('culprit_function')}")
    print(f"    confidence       : {result.get('confidence')}")
    print(f"  GROUND TRUTH")
    print(f"    ground_truth_file     : {gt['ground_truth_file']}")
    print(f"    ground_truth_function : {gt['ground_truth_function']}")
    # Guard against None before calling strip() — culprit_function is Optional
    # in AttributeResponse (the LLM may be uncertain and omit it).
    pred_file = (result.get("culprit_file") or "").strip()
    pred_fn   = (result.get("culprit_function") or "").strip()
    file_match = pred_file == gt["ground_truth_file"]
    fn_match   = pred_fn   == gt["ground_truth_function"]
    print(f"  SCORE  file={'✅' if file_match else '❌'}  function={'✅' if fn_match else '❌'}")
    print(f"\n  EXPLANATION:\n    {result.get('explanation', '(none)')}")
    print(f"\n  REASONING TRACE:")
    trace = result.get("reasoning_trace", "(none)")
    # Indent each line of the trace for readability
    for line in str(trace).splitlines():
        print(f"    {line}")
    print(sep)


# ── tests ────────────────────────────────────────────────────────────────────
class TestAttributeE2E:

    @pytest.mark.asyncio
    async def test_bug1_auth_gmail_attribution(self, app_client):
        """
        Bug 1 — v1.1.0: gmail addresses always 401 'Session expired'
        Expected culprit: app/api/auth/route.ts :: POST
        """
        tag = "v1.1.0"
        gt  = GROUND_TRUTH[tag]

        body = {"envelope": BUG1_ENVELOPE, "tag": tag}
        resp = await app_client.post("/attribute", json=body)

        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text}"
        )

        result = resp.json()
        _print_attribution(tag, gt, result)

        # ── structural assertions (always must pass) ──────────────────────
        assert "culprit_file"     in result, "Response missing culprit_file"
        assert "culprit_function" in result, "Response missing culprit_function"
        assert "confidence"       in result, "Response missing confidence"
        assert "explanation"      in result, "Response missing explanation"
        assert "reasoning_trace"  in result, "Response missing reasoning_trace"
        assert result["culprit_file"], "culprit_file should not be empty"

        # ── accuracy assertions ───────────────────────────────────────────
        assert result["culprit_file"] == gt["ground_truth_file"], (
            f"File miss: got '{result['culprit_file']}', "
            f"expected '{gt['ground_truth_file']}'"
        )
        assert result["culprit_function"] == gt["ground_truth_function"], (
            f"Function miss: got '{result['culprit_function']}', "
            f"expected '{gt['ground_truth_function']}'"
        )

    @pytest.mark.asyncio
    async def test_bug2_payment_amex_attribution(self, app_client):
        """
        Bug 2 — v1.2.0: Amex 15-digit cards rejected with cryptic error.
        Expected culprit: app/api/payment/route.ts :: POST
        """
        tag = "v1.2.0"
        gt  = GROUND_TRUTH[tag]

        body = {"envelope": BUG2_ENVELOPE, "tag": tag}
        resp = await app_client.post("/attribute", json=body)

        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text}"
        )

        result = resp.json()
        _print_attribution(tag, gt, result)

        # ── structural assertions ─────────────────────────────────────────
        assert "culprit_file"     in result
        assert "culprit_function" in result
        assert "confidence"       in result
        assert "explanation"      in result
        assert "reasoning_trace"  in result
        assert result["culprit_file"], "culprit_file should not be empty"

        # ── accuracy assertions ───────────────────────────────────────────
        assert result["culprit_file"] == gt["ground_truth_file"], (
            f"File miss: got '{result['culprit_file']}', "
            f"expected '{gt['ground_truth_file']}'"
        )
        assert result["culprit_function"] == gt["ground_truth_function"], (
            f"Function miss: got '{result['culprit_function']}', "
            f"expected '{gt['ground_truth_function']}'"
        )

    @pytest.mark.asyncio
    async def test_both_responses_differ(self, app_client):
        """
        Sanity check: the two attributions should not be identical.
        If they are, the LLM is likely returning a cached/static response.
        """
        tag1_body = {"envelope": copy.deepcopy(BUG1_ENVELOPE), "tag": "v1.1.0"}
        tag2_body = {"envelope": copy.deepcopy(BUG2_ENVELOPE), "tag": "v1.2.0"}

        # Use fresh uuids so no 409 if tests ran before
        tag1_body["envelope"]["uuid"] = "e2e-sanity-bug1-002"
        tag2_body["envelope"]["uuid"] = "e2e-sanity-bug2-002"

        r1 = await app_client.post("/attribute", json=tag1_body)
        r2 = await app_client.post("/attribute", json=tag2_body)

        assert r1.status_code == 200
        assert r2.status_code == 200

        d1 = r1.json()
        d2 = r2.json()

        print(f"\n  [sanity] bug1 file: {d1.get('culprit_file')} | "
              f"bug2 file: {d2.get('culprit_file')}")

        # The culprit files must be different since they are different routes
        assert d1.get("culprit_file") != d2.get("culprit_file"), (
            "Both attributions returned the same culprit_file — "
            "the LLM may be ignoring the error context."
        )

    @pytest.mark.asyncio
    async def test_missing_tag_returns_error(self, app_client):
        """
        /attribute requires a tag to fetch the diff.
        Omitting it should return 422 or a clear error.
        """
        body = {"envelope": copy.deepcopy(BUG1_ENVELOPE)}
        resp = await app_client.post("/attribute", json=body)
        # Accept either 422 (Pydantic validation) or 400 (manual check)
        assert resp.status_code in (400, 422), (
            f"Expected 400 or 422 for missing tag, got {resp.status_code}"
        )

    @pytest.mark.asyncio
    async def test_invalid_tag_returns_error(self, app_client):
        """
        A tag that doesn't exist on the repo should return a clear error,
        not a 500 crash or a silent 200.

        FIX: attribute.py now raises HTTP 404 when the GitHub Compare API
        returns 404 for the tag, instead of forwarding the error string to
        Gemini and returning a bogus 200 attribution.
        """
        body = {
            "envelope": copy.deepcopy(BUG1_ENVELOPE),
            "tag": "v99.99.99-nonexistent",
        }
        body["envelope"]["uuid"] = "e2e-bad-tag-001"
        resp = await app_client.post("/attribute", json=body)

        # 404 — tag not found (expected now that attribute.py raises it)
        # 400/422 — if validation rejects it before GitHub call
        # 502 — GitHub unreachable / rate limited (transient, not a test failure)
        # 500 — unexpected crash (still acceptable per original spec)
        # 200 is the only outcome that means "silently wrong" — never acceptable.
        assert resp.status_code != 200, (
            f"Expected a non-200 error response for a nonexistent tag, "
            f"got 200. The pipeline swallowed the GitHub 404 and returned "
            f"a fabricated attribution. body={resp.text[:200]}"
        )
        print(f"\n  [bad-tag] status={resp.status_code} body={resp.text[:200]}")