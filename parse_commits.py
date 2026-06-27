"""
parse_commits.py — generates eval_cases.json from the stripe-clone git history.

Key fix: diffs are now per-tag incremental (v1.0.0 → v1.1.0, v1.1.0 → v1.2.0)
NOT cumulative (v1.0.0-clean → v1.x.0), which was causing all previous bugs'
files to appear in every case and confusing the attribution pipeline.

Run from eval-harness directory:
    python parse_commits.py
"""

import subprocess
import json

REPO_PATH = r"E:\interfere\stripe-clone"

# Ordered list of tags — order matters for incremental diff
TAGS_IN_ORDER = [
    "v1.0.0-clean",
    "v1.1.0",
    "v1.2.0",
    "v1.3.0",
    "v1.4.0",
    "v1.5.0",
    "v1.6.0",
    "v1.7.0",
]

GROUND_TRUTH = {
    "v1.1.0": {
        "bug_id": "auth-gmail-session-expiry",
        "ground_truth_file": "app/api/auth/route.ts",
        "ground_truth_function": "POST",
        "error_type": "AuthError",
        "expected_message": "Session expired",
        "route": "/api/auth",
        "route_type": "route",
        "error_source": "server",
    },
    "v1.2.0": {
        "bug_id": "payment-amex-rejection",
        "ground_truth_file": "app/api/payment/route.ts",
        "ground_truth_function": "POST",
        "error_type": "PaymentError",
        "expected_message": "Payment processor error: unsupported card type",
        "route": "/api/payment",
        "route_type": "route",
        "error_source": "server",
    },
    "v1.3.0": {
        "bug_id": "payment-random-500",
        "ground_truth_file": "app/api/payment/route.ts",
        "ground_truth_function": "POST",
        "error_type": "InternalServerError",
        "expected_message": "Internal payment error",
        "route": "/api/payment",
        "route_type": "route",
        "error_source": "server",
    },
    "v1.4.0": {
        "bug_id": "dashboard-null-amount",
        "ground_truth_file": "app/dashboard/page.tsx",
        "ground_truth_function": "Dashboard",
        "error_type": "TypeError",
        "expected_message": "Cannot read properties of null (reading 'toFixed')",
        "route": "/dashboard",
        "route_type": "render",
        "error_source": "server",
        "note": "root cause is data/transactions.json — null amount on Bob's entry",
    },
    "v1.5.0": {
        "bug_id": "analytics-division-by-zero",
        "ground_truth_file": "app/api/revenue/route.ts",
        "ground_truth_function": "GET",
        "error_type": "RangeError",
        "expected_message": "Invalid date range: no data found for this period",
        "route": "/api/revenue",
        "route_type": "route",
        "error_source": "server",
    },
    "v1.6.0": {
        "bug_id": "webhook-malformed-payload",
        "ground_truth_file": "app/api/webhooks/route.ts",
        "ground_truth_function": "POST",
        "error_type": "SyntaxError",
        "expected_message": "Malformed webhook payload",
        "route": "/api/webhooks",
        "route_type": "route",
        "error_source": "server",
    },
    "v1.7.0": {
        "bug_id": "webhook-idempotency-failure",
        "ground_truth_file": "app/api/webhooks/route.ts",
        "ground_truth_function": "POST",
        "error_type": "LogicError",
        "expected_message": "Processing duplicate event",
        "route": "/api/webhooks",
        "route_type": "route",
        "error_source": "server",
    },
}


def get_commit_sha(tag: str) -> str:
    result = subprocess.run(
        ["git", "rev-list", "-n", "1", tag],
        cwd=REPO_PATH,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def get_previous_tag(tag: str) -> str:
    """Return the tag immediately before this one in the ordered list."""
    idx = TAGS_IN_ORDER.index(tag)
    return TAGS_IN_ORDER[idx - 1]  # safe because v1.0.0-clean is never in GROUND_TRUTH


def get_diff_incremental(prev_tag: str, tag: str) -> str:
    """
    Diff only what changed between the previous tag and this one.
    This is the diff Gemini should reason over — just the bug introduced.
    """
    result = subprocess.run(
        ["git", "diff", f"{prev_tag}..{tag}", "--", "*.ts", "*.tsx", "*.json"],
        cwd=REPO_PATH,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def get_changed_files_incremental(prev_tag: str, tag: str) -> list[str]:
    """Files changed between the previous tag and this one only."""
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{prev_tag}..{tag}"],
        cwd=REPO_PATH,
        capture_output=True,
        text=True,
    )
    return [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]


def build_envelope(tag: str, truth: dict, sha: str) -> dict:
    """Build a simulated Interfere error envelope for this bug."""
    return {
        "uuid": f"eval-{truth['bug_id']}",
        "v": 0,
        "buildId": sha[:8],
        "clientTs": 1719316200000,
        "sessionId": f"server_{sha[:8]}",
        "sessionSource": "async_context",
        "runtime": "node",
        "environment": "production",
        "releaseId": tag,
        "type": "error",
        "context": {
            "runtime": "nextjs",
            "routePath": truth["route"],
            "routeType": truth["route_type"],
            "requestMethod": "POST" if truth["route_type"] == "route" else None,
            "requestPath": truth["route"],
            "routerKind": "App Router",
        },
        "payload": {
            "exceptions": [
                {
                    "type": truth["error_type"],
                    "value": truth["expected_message"],
                    "mechanism": {
                        "type": "instrument",
                        "handled": False,
                    },
                    "frames": [
                        {
                            "fileName": truth["ground_truth_file"],
                            "functionName": truth["ground_truth_function"],
                            "lineNumber": 1,
                            "columnNumber": 1,
                        }
                    ],
                }
            ]
        },
    }


def main():
    eval_cases = []

    for tag, truth in GROUND_TRUTH.items():
        prev_tag = get_previous_tag(tag)
        sha = get_commit_sha(tag)
        changed_files = get_changed_files_incremental(prev_tag, tag)
        diff = get_diff_incremental(prev_tag, tag)

        case = {
            "tag": tag,
            "commit_sha": sha,
            "bug_id": truth["bug_id"],
            "ground_truth_file": truth["ground_truth_file"],
            "ground_truth_function": truth["ground_truth_function"],
            "ground_truth_commit_introduced": tag,        # ← field models.py expects
            "error_type": truth["error_type"],
            "expected_message": truth["expected_message"],
            "route": truth["route"],
            "route_type": truth["route_type"],
            "error_source": truth["error_source"],
            "prev_tag": prev_tag,                         # ← so attribute.py knows what to diff
            "changed_files_in_tag": changed_files,
            "simulated_envelope": build_envelope(tag, truth, sha),
            "diff_preview": diff[:500] if diff else "",
        }

        if "note" in truth:
            case["note"] = truth["note"]

        eval_cases.append(case)
        print(f"✓ {tag} (vs {prev_tag}) — {truth['bug_id']} — {len(changed_files)} file(s): {changed_files}")

    output_path = "eval_cases.json"
    with open(output_path, "w") as f:
        json.dump(eval_cases, f, indent=2)

    print(f"\n✅ Wrote {len(eval_cases)} eval cases to {output_path}")
    print("\nChanged files per tag (should be 1 per bug):")
    for case in eval_cases:
        print(f"  {case['tag']}: {case['changed_files_in_tag']}")


if __name__ == "__main__":
    main()