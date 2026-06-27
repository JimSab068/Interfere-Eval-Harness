"""
test_llm.py — Smoke tests for llm.py.

Makes real API calls to Google AI Studio (or Vertex AI).
Requires GEMINI_API_KEY (or USE_VERTEX=true + GCP_PROJECT) in .env.

These are intentionally lightweight — we're testing the integration plumbing,
not the model's reasoning quality.

Run:
    pytest tests/test_llm.py -v
"""

from __future__ import annotations

import os
import time
import pytest

# ---------------------------------------------------------------------------
# Rate limiting
#
# The Gemini free tier allows 15 RPM.  The autouse fixture sleeps 4 s before
# every test to stay within budget on average.  Tests that make MORE than one
# API call inside their body must sleep between those calls themselves — the
# fixture only fires once per test boundary.
# ---------------------------------------------------------------------------

_INTER_CALL_SLEEP = 5  # seconds — 15 RPM free tier = 1 call per 4s; 5s gives headroom


@pytest.fixture(autouse=True)
def rate_limit_sleep():
    """Sleep before every test to stay under the 15 RPM free-tier limit."""
    time.sleep(_INTER_CALL_SLEEP)


def _skip_if_no_llm():
    use_vertex = os.environ.get("USE_VERTEX", "false").lower() == "true"
    if use_vertex:
        if not os.environ.get("GCP_PROJECT"):
            pytest.skip("USE_VERTEX=true but GCP_PROJECT not set")
    else:
        if not os.environ.get("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set — skipping LLM smoke tests")


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------

class TestGenerate:

    def test_generate_returns_string(self):
        _skip_if_no_llm()
        from llm import generate
        result = generate("Say exactly: hello world")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_with_system_prompt(self):
        _skip_if_no_llm()
        from llm import generate
        result = generate(
            prompt="What colour is the sky?",
            system="Always answer in exactly one word.",
        )
        assert isinstance(result, str)
        # Should be a single word — loose check
        assert len(result.split()) <= 5

    def test_generate_low_temperature(self):
        """Same prompt twice at temperature=0 should give near-identical results.

        Uses max_output_tokens=32 — gemini-2.5-flash has a larger minimum
        output buffer than older models and silently truncates responses when
        the token cap is too tight (FinishReason.MAX_TOKENS), even for a
        single-digit answer.  32 tokens is plenty for "4" while still keeping
        the test fast.
        """
        _skip_if_no_llm()
        from llm import generate
        prompt = "What is 2 + 2? Reply with just the number."
        r1 = generate(prompt, temperature=0.0, max_output_tokens=32)
        time.sleep(_INTER_CALL_SLEEP)   # guard the second back-to-back call
        r2 = generate(prompt, temperature=0.0, max_output_tokens=32)
        assert "4" in r1
        assert "4" in r2

    def test_generate_json_response(self):
        """Verify the model can follow JSON-only instructions — critical for attribute.py.

        Sleeps an extra interval before calling the API because this test runs
        immediately after test_generate_low_temperature, which itself makes two
        API calls.  The autouse fixture only sleeps once at the test boundary,
        so without this extra sleep the quota is already exhausted.
        """
        _skip_if_no_llm()
        import json
        from llm import generate
        time.sleep(_INTER_CALL_SLEEP)   # extra guard — previous test made 2 calls
        result = generate(
            prompt='Return JSON: {"answer": 42}',
            system="Respond with valid JSON only. No markdown fences.",
            temperature=0.0,
        )
        parsed = json.loads(result)
        assert parsed.get("answer") == 42


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------

class TestEmbed:

    def test_embed_returns_vector(self):
        _skip_if_no_llm()
        from llm import EMBEDDING_DIM, embed
        vec = embed("TypeError: Cannot read properties of null")
        assert isinstance(vec, list)
        assert len(vec) == EMBEDDING_DIM
        assert all(isinstance(x, float) for x in vec)

    def test_embed_different_texts_different_vectors(self):
        _skip_if_no_llm()
        from llm import embed
        v1 = embed("payment route error")
        time.sleep(_INTER_CALL_SLEEP)
        v2 = embed("authentication session expired")
        # Vectors should not be identical
        assert v1 != v2

    def test_embed_similar_texts_closer_than_dissimilar(self):
        """Cosine similarity check — semantically similar texts should be closer."""
        _skip_if_no_llm()
        import math
        from llm import embed

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            mag_a = math.sqrt(sum(x**2 for x in a))
            mag_b = math.sqrt(sum(x**2 for x in b))
            return dot / (mag_a * mag_b + 1e-9)

        v_payment1 = embed("Amex card payment rejection error")
        time.sleep(_INTER_CALL_SLEEP)
        v_payment2 = embed("Credit card number length validation failure")
        time.sleep(_INTER_CALL_SLEEP)
        v_auth     = embed("Gmail session expired authentication failure")

        sim_same   = cosine(v_payment1, v_payment2)
        sim_diff   = cosine(v_payment1, v_auth)
        assert sim_same > sim_diff, (
            f"Expected payment vectors to be closer to each other "
            f"({sim_same:.3f}) than to auth ({sim_diff:.3f})"
        )


# ---------------------------------------------------------------------------
# embed_batch()
# ---------------------------------------------------------------------------

class TestEmbedBatch:

    def test_embed_batch_returns_correct_count(self):
        _skip_if_no_llm()
        from llm import EMBEDDING_DIM, embed_batch
        texts = [
            "TypeError in dashboard",
            "Division by zero in revenue route",
            "Unhandled webhook payload",
        ]
        results = embed_batch(texts)
        assert len(results) == 3
        for vec in results:
            assert len(vec) == EMBEDDING_DIM

    def test_embed_batch_empty_input(self):
        _skip_if_no_llm()
        from llm import embed_batch
        result = embed_batch([])
        assert result == []

    def test_embed_batch_task_type_query(self):
        _skip_if_no_llm()
        from llm import EMBEDDING_DIM, embed_batch
        result = embed_batch(
            ["error context for retrieval"],
            task_type="RETRIEVAL_QUERY",
        )
        assert len(result) == 1
        assert len(result[0]) == EMBEDDING_DIM


# ---------------------------------------------------------------------------
# ChromaDB embedding function interface
# ---------------------------------------------------------------------------

class TestEmbeddingFunctions:

    def test_gemini_embedding_function_interface(self):
        _skip_if_no_llm()
        from llm import EMBEDDING_DIM, GeminiEmbeddingFunction
        fn = GeminiEmbeddingFunction()
        result = fn(["test document chunk"])
        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]) == EMBEDDING_DIM

    def test_gemini_query_embedding_function_interface(self):
        _skip_if_no_llm()
        from llm import EMBEDDING_DIM, GeminiQueryEmbeddingFunction
        fn = GeminiQueryEmbeddingFunction()
        result = fn(["TypeError: null reference at line 42"])
        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]) == EMBEDDING_DIM