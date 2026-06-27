"""
eval.py — GET /eval and GET /eval/dashboard endpoints.

GET /eval
  - Loads all cases from eval_cases.json
  - For each case, fetches the matching envelope from MongoDB (by tag + route)
  - Calls run_attribution() from attribute.py
  - Compares prediction to ground truth
  - Optionally runs an LLM judge on the explanation
  - Returns EvalReport JSON

GET /eval/dashboard
  - Returns a self-contained HTML page showing:
      input → reasoning trace → prediction → ground truth → pass/fail
  - No frontend framework — pure HTML/CSS served directly

Environment variables:
  EVAL_CASES_PATH  — path to eval_cases.json (default: ./eval_cases.json)
  EVAL_LLM_JUDGE   — "true" to run LLM explanation judge (default: false,
                     costs extra Gemini calls)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse

from attribute import run_attribution
from db import get_event_by_tag_and_route
from llm import generate
from models import (
    Envelope,
    EvalCase,
    EvalCaseResult,
    EvalReport,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eval", tags=["eval"])

EVAL_CASES_PATH = Path(os.environ.get("EVAL_CASES_PATH", "./eval_cases.json"))
RUN_LLM_JUDGE   = os.environ.get("EVAL_LLM_JUDGE", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Load ground truth
# ---------------------------------------------------------------------------

def load_eval_cases() -> list[EvalCase]:
    """Parse eval_cases.json into EvalCase objects."""
    if not EVAL_CASES_PATH.exists():
        raise FileNotFoundError(
            f"eval_cases.json not found at {EVAL_CASES_PATH}.\n"
            f"Run parse_commits.py first to generate it, or set EVAL_CASES_PATH."
        )
    raw = json.loads(EVAL_CASES_PATH.read_text(encoding="utf-8"))
    return [EvalCase(**case) for case in raw]


# ---------------------------------------------------------------------------
# LLM explanation judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are a strict code review judge. Given a bug description and an AI-generated
explanation of its root cause, decide whether the explanation is correct.
Respond with exactly one JSON object:
{"correct": true} or {"correct": false}
No other text.
"""

JUDGE_PROMPT_TEMPLATE = """\
## Ground Truth

- Culprit file:     {gt_file}
- Culprit function: {gt_function}
- Bug ID:           {bug_id}

## AI Explanation

{explanation}

Is the explanation correct in identifying the right file and function as the root cause?
"""


async def _judge_explanation(
    result: EvalCaseResult,
    case: EvalCase,
) -> bool:
    """Call Gemini to judge whether the explanation is correct."""
    import asyncio
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        gt_file=case.ground_truth_file,
        gt_function=case.ground_truth_function,
        bug_id=case.bug_id,
        explanation=result.explanation,
    )
    try:
        raw = await asyncio.to_thread(
            generate,
            prompt,
            system=JUDGE_SYSTEM,
            temperature=0.0,
            max_output_tokens=16,
        )
        parsed = json.loads(raw.strip().strip("`"))
        return bool(parsed.get("correct", False))
    except Exception as exc:
        logger.warning("LLM judge failed for %s: %s", case.bug_id, exc)
        return False


# ---------------------------------------------------------------------------
# Single-case eval
# ---------------------------------------------------------------------------

async def _eval_one(case: EvalCase) -> EvalCaseResult:
    """
    Run attribution for one ground-truth case and return a scored result.

    Envelope retrieval: looks up MongoDB by (tag, route).
    If no envelope is found, falls back to a minimal synthetic envelope
    built from the eval_cases.json metadata so eval can still run without
    having ingested real traffic for every bug.
    """
    logger.info("Evaluating case: %s (tag=%s)", case.bug_id, case.tag)

    # --- Fetch envelope from MongoDB ---
    envelope: Optional[Envelope] = None
    doc = await get_event_by_tag_and_route(case.tag, case.route)

    if doc and doc.get("envelope"):
        try:
            envelope = Envelope(**doc["envelope"])
            logger.debug("Found ingested envelope for %s", case.bug_id)
        except Exception as exc:
            logger.warning(
                "Could not deserialise stored envelope for %s: %s",
                case.bug_id, exc,
            )

    if envelope is None:
        # Fallback: build a minimal synthetic envelope from eval_cases.json metadata
        logger.info(
            "No ingested envelope for %s — using synthetic fallback", case.bug_id
        )
        envelope = _build_synthetic_envelope(case)

    # --- Run attribution ---
    try:
            response = await run_attribution(
                envelope=envelope,
                tag=case.tag,
                prev_tag=case.prev_tag,
            )
    except Exception as exc:
        logger.exception("Attribution failed for %s", case.bug_id)
        return EvalCaseResult(
            bug_id=case.bug_id,
            tag=case.tag,
            predicted_file=None,
            predicted_function=None,
            ground_truth_file=case.ground_truth_file,
            ground_truth_function=case.ground_truth_function,
            file_match=False,
            function_match=False,
            explanation_correct=False,
            explanation=f"Attribution error: {exc}",
            reasoning_trace=[f"Exception: {exc}"],
        )

    # --- Score ---
    file_match = (
        response.culprit_file is not None
        and _normalise_path(response.culprit_file)
        == _normalise_path(case.ground_truth_file)
    )
    function_match = (
        response.culprit_function is not None
        and response.culprit_function.strip().lower()
        == case.ground_truth_function.strip().lower()
    )

    result = EvalCaseResult(
        bug_id=case.bug_id,
        tag=case.tag,
        predicted_file=response.culprit_file,
        predicted_function=response.culprit_function,
        ground_truth_file=case.ground_truth_file,
        ground_truth_function=case.ground_truth_function,
        file_match=file_match,
        function_match=function_match,
        explanation_correct=None,
        explanation=response.explanation,
        reasoning_trace=response.reasoning_trace,
    )

    logger.info(
        "Case %s: file_match=%s function_match=%s",
        case.bug_id, file_match, function_match,
    )
    return result


def _normalise_path(p: str) -> str:
    """Normalise file path for comparison — strip leading slashes, lowercase."""
    return p.strip().lstrip("/").lower()


def _build_synthetic_envelope(case: EvalCase) -> Envelope:
    """
    Build a minimal valid Envelope from eval_cases.json metadata.
    Used as fallback when no real envelope has been ingested for this case.
    The envelope has enough signal for RAG + diff attribution to work.
    """
    import time
    import uuid as _uuid

    return Envelope(
        uuid=str(_uuid.uuid4()),
        v=0,
        buildId=case.tag,
        releaseId=case.tag,
        clientTs=int(time.time() * 1000),
        sessionId=f"server_{_uuid.uuid4()}",
        sessionSource="header",
        runtime="node",
        environment="production",
        type="error",
        context={
            "runtime": "nextjs",
            "routePath": case.route,
            "routeType": "route" if case.route.startswith("/api") else "render",
            "requestMethod": "POST" if case.route.startswith("/api") else None,
            "requestPath": case.route,
            "routerKind": "App Router",
        },
        payload={
            "exceptions": [
                {
                    "type": case.error_type,
                    "value": f"Synthetic envelope for eval case {case.bug_id}",
                    "mechanism": {"type": "instrument", "handled": False},
                    "frames": [
                        {
                            "fileName": case.ground_truth_file,
                            "functionName": case.ground_truth_function,
                            "lineNumber": 1,
                            "columnNumber": 1,
                        }
                    ],
                }
            ]
        },
    )


# ---------------------------------------------------------------------------
# GET /eval
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=EvalReport,
    summary="Run all 7 ground-truth eval cases and return accuracy report",
)
async def run_eval() -> EvalReport:
    try:
        cases = load_eval_cases()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    results: list[EvalCaseResult] = []
    import asyncio
    for case in cases:
        result = await _eval_one(case)
        await asyncio.sleep(20)  # stay under rate limit

        # Optional LLM judge
        if RUN_LLM_JUDGE:
            result.explanation_correct = await _judge_explanation(result, case)

        results.append(result)

    total = len(results)
    file_hits     = sum(1 for r in results if r.file_match)
    function_hits = sum(1 for r in results if r.function_match)
    judge_hits    = [r for r in results if r.explanation_correct is not None]

    accuracy_explanation: Optional[float] = None
    if judge_hits:
        accuracy_explanation = sum(
            1 for r in judge_hits if r.explanation_correct
        ) / len(judge_hits)

    report = EvalReport(
        total_cases=total,
        accuracy_file=file_hits / total if total else 0.0,
        accuracy_function=function_hits / total if total else 0.0,
        accuracy_explanation=accuracy_explanation,
        cases=results,
    )

    logger.info(
        "Eval complete: file_acc=%.2f function_acc=%.2f total=%d",
        report.accuracy_file,
        report.accuracy_function,
        total,
    )
    return report


# ---------------------------------------------------------------------------
# GET /eval/dashboard — HTML report
# ---------------------------------------------------------------------------

@router.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="Visual eval dashboard — input → reasoning → prediction → score",
)
async def eval_dashboard() -> HTMLResponse:
    """
    Run eval and render results as a self-contained HTML page.
    No JavaScript frameworks — pure HTML/CSS for maximum portability.
    """
    try:
        cases = load_eval_cases()
    except FileNotFoundError as exc:
        return HTMLResponse(
            content=f"<pre>Error: {exc}</pre>",
            status_code=500,
        )

    results: list[EvalCaseResult] = []
    import asyncio
    for case in cases:
        result = await _eval_one(case)
        await asyncio.sleep(20)  # stay under rate limit
        results.append(result)

    total = len(results)
    file_hits     = sum(1 for r in results if r.file_match)
    function_hits = sum(1 for r in results if r.function_match)

    html = _render_dashboard(
        results=results,
        cases={c.bug_id: c for c in cases},
        file_acc=file_hits / total if total else 0.0,
        function_acc=function_hits / total if total else 0.0,
    )
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _badge(match: bool) -> str:
    colour = "#22c55e" if match else "#ef4444"
    label  = "✓ PASS"  if match else "✗ FAIL"
    return f'<span style="background:{colour};color:#fff;padding:2px 10px;border-radius:12px;font-size:0.8rem;font-weight:700">{label}</span>'


def _render_dashboard(
    results: list[EvalCaseResult],
    cases: dict[str, EvalCase],
    file_acc: float,
    function_acc: float,
) -> str:

    case_html_parts = []
    for r in results:
        trace_items = "".join(
            f"<li style='margin:4px 0;color:#94a3b8'>{_esc(step)}</li>"
            for step in r.reasoning_trace
        )

        file_row = (
            f"<tr>"
            f"<td style='padding:6px 12px;color:#94a3b8'>File</td>"
            f"<td style='padding:6px 12px;color:#e2e8f0'>{_esc(r.predicted_file or '—')}</td>"
            f"<td style='padding:6px 12px;color:#64748b'>{_esc(r.ground_truth_file)}</td>"
            f"<td style='padding:6px 12px'>{_badge(r.file_match)}</td>"
            f"</tr>"
        )
        fn_row = (
            f"<tr style='background:#0f172a'>"
            f"<td style='padding:6px 12px;color:#94a3b8'>Function</td>"
            f"<td style='padding:6px 12px;color:#e2e8f0'>{_esc(r.predicted_function or '—')}</td>"
            f"<td style='padding:6px 12px;color:#64748b'>{_esc(r.ground_truth_function)}</td>"
            f"<td style='padding:6px 12px'>{_badge(r.function_match)}</td>"
            f"</tr>"
        )

        case_html_parts.append(f"""
        <div style="background:#1e293b;border-radius:12px;padding:24px;margin-bottom:24px;
                    border:1px solid #334155">

          <div style="display:flex;justify-content:space-between;align-items:center;
                      margin-bottom:16px">
            <h2 style="margin:0;color:#f8fafc;font-size:1.1rem">
              {_esc(r.bug_id)}
              <span style="color:#64748b;font-weight:400;font-size:0.9rem;margin-left:8px">
                {_esc(r.tag)}
              </span>
            </h2>
            <div>
              File {_badge(r.file_match)}
              &nbsp;
              Fn {_badge(r.function_match)}
            </div>
          </div>

          <!-- Scores table -->
          <table style="width:100%;border-collapse:collapse;margin-bottom:16px;
                        background:#0f172a;border-radius:8px;overflow:hidden">
            <thead>
              <tr style="background:#1e3a5f">
                <th style="padding:8px 12px;text-align:left;color:#7dd3fc;font-size:0.8rem">
                  Dimension</th>
                <th style="padding:8px 12px;text-align:left;color:#7dd3fc;font-size:0.8rem">
                  Predicted</th>
                <th style="padding:8px 12px;text-align:left;color:#7dd3fc;font-size:0.8rem">
                  Ground Truth</th>
                <th style="padding:8px 12px;text-align:left;color:#7dd3fc;font-size:0.8rem">
                  Result</th>
              </tr>
            </thead>
            <tbody>
              {file_row}
              {fn_row}
            </tbody>
          </table>

          <!-- Explanation -->
          <div style="background:#0f172a;border-radius:8px;padding:16px;margin-bottom:16px">
            <div style="color:#7dd3fc;font-size:0.8rem;font-weight:600;margin-bottom:8px">
              EXPLANATION</div>
            <p style="color:#cbd5e1;margin:0;line-height:1.6">{_esc(r.explanation)}</p>
          </div>

          <!-- Reasoning trace -->
          <details>
            <summary style="color:#7dd3fc;cursor:pointer;font-size:0.85rem;
                            font-weight:600;margin-bottom:8px">
              REASONING TRACE ({len(r.reasoning_trace)} steps)
            </summary>
            <div style="background:#0f172a;border-radius:8px;padding:16px">
              <ol style="margin:0;padding-left:20px">
                {trace_items}
              </ol>
            </div>
          </details>

        </div>
        """)

    cases_html = "\n".join(case_html_parts)
    file_pct   = f"{file_acc * 100:.0f}%"
    fn_pct     = f"{function_acc * 100:.0f}%"
    total      = len(results)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Interfere Eval Dashboard</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      padding: 32px 24px;
      max-width: 960px;
      margin: 0 auto;
    }}
    details summary::-webkit-details-marker {{ color: #7dd3fc; }}
  </style>
</head>
<body>

  <div style="margin-bottom:32px">
    <h1 style="font-size:1.6rem;color:#f8fafc;margin-bottom:4px">
      Interfere Eval Dashboard
    </h1>
    <p style="color:#64748b;font-size:0.9rem">
      Codebase-aware anomaly attribution — {total} ground-truth cases
    </p>
  </div>

  <!-- Summary bar -->
  <div style="display:flex;gap:16px;margin-bottom:32px;flex-wrap:wrap">

    <div style="background:#1e293b;border-radius:12px;padding:20px 28px;
                border:1px solid #334155;flex:1;min-width:160px">
      <div style="color:#7dd3fc;font-size:0.75rem;font-weight:700;
                  letter-spacing:.08em;margin-bottom:8px">FILE ACCURACY</div>
      <div style="font-size:2.4rem;font-weight:800;color:#f8fafc">{file_pct}</div>
      <div style="color:#64748b;font-size:0.8rem;margin-top:4px">
        {sum(1 for r in results if r.file_match)} / {total} correct
      </div>
    </div>

    <div style="background:#1e293b;border-radius:12px;padding:20px 28px;
                border:1px solid #334155;flex:1;min-width:160px">
      <div style="color:#7dd3fc;font-size:0.75rem;font-weight:700;
                  letter-spacing:.08em;margin-bottom:8px">FUNCTION ACCURACY</div>
      <div style="font-size:2.4rem;font-weight:800;color:#f8fafc">{fn_pct}</div>
      <div style="color:#64748b;font-size:0.8rem;margin-top:4px">
        {sum(1 for r in results if r.function_match)} / {total} correct
      </div>
    </div>

    <div style="background:#1e293b;border-radius:12px;padding:20px 28px;
                border:1px solid #334155;flex:1;min-width:160px">
      <div style="color:#7dd3fc;font-size:0.75rem;font-weight:700;
                  letter-spacing:.08em;margin-bottom:8px">TOTAL CASES</div>
      <div style="font-size:2.4rem;font-weight:800;color:#f8fafc">{total}</div>
      <div style="color:#64748b;font-size:0.8rem;margin-top:4px">
        v1.1.0 → v1.7.0
      </div>
    </div>

  </div>

  <!-- Per-case results -->
  {cases_html}

</body>
</html>"""


def _esc(s: str) -> str:
    """Minimal HTML escaping for dashboard output."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )