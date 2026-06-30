"""Rate-limit handling and per-run tool call budget.

Run:
    python demos/d04_rate_limit_and_budget_guard_learner.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.budget_guard import ToolBudgetGuard
from shared.failure_simulator import SimulatedRateLimitError, fetch_product
from shared.failure_simulator import simulate_failure as trigger_failure
from shared.logging_utils import log_tool_attempt
from shared.tool_response import error_response, success_response

MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS_PER_RUN", "3"))


def get_product_details_with_budget(
    product_id: int,
    budget_guard: ToolBudgetGuard,
    simulate_rate_limit: bool = False,
) -> dict:
    """Fetch a product under a per-run call budget, with rate-limit handling."""
    input_data = {"product_id": product_id}

    if not budget_guard.can_call("get_product_details_with_budget"):
        log_tool_attempt(
            "get_product_details_with_budget", input_data, 1,
            "budget_exceeded", 0.0, error_type="budget_exceeded",
            budget_remaining=budget_guard.remaining(),
        )
        return error_response(
            "budget_exceeded",
            "Tool call budget reached for this run.",
            retryable=False,
            budget_remaining=budget_guard.remaining(),
        )
    budget_guard.record_call("get_product_details_with_budget")

    start = time.perf_counter()
    try:
        if simulate_rate_limit:
            trigger_failure("rate_limit")
        raw = fetch_product(product_id)
    except SimulatedRateLimitError:
        latency = (time.perf_counter() - start) * 1000
        log_tool_attempt(
            "get_product_details_with_budget", input_data, 1, "error",
            latency, error_type="rate_limited",
            budget_remaining=budget_guard.remaining(),
        )
        return error_response(
            "rate_limited",
            "The product service is rate limiting requests.",
            retryable=True,
            budget_remaining=budget_guard.remaining(),
        )

    normalized = {
        "product_id": raw.get("id"),
        "title": raw.get("title"),
        "category": raw.get("category"),
        "price": raw.get("price"),
    }
    latency = (time.perf_counter() - start) * 1000
    log_tool_attempt(
        "get_product_details_with_budget", input_data, 1, "success",
        latency, budget_remaining=budget_guard.remaining(),
    )
    return success_response(
        normalized, source="simulator",
        budget_remaining=budget_guard.remaining(),
    )


def main() -> None:
    print("=" * 64)
    print("DEMO 4 — Rate limit and budget guard")
    print("=" * 64)
    print(f"\n(MAX_TOOL_CALLS_PER_RUN = {MAX_TOOL_CALLS})")

    print("\n--- Rate limit (safe, retryable) ---")
    guard = ToolBudgetGuard(max_calls=MAX_TOOL_CALLS)
    print(get_product_details_with_budget(1, guard, simulate_rate_limit=True))

    print(f"\n--- Budget of {MAX_TOOL_CALLS}, attempt 5 calls ---")
    guard = ToolBudgetGuard(max_calls=MAX_TOOL_CALLS)
    for i in range(1, 6):
        result = get_product_details_with_budget(i, guard)
        if result["status"] == "success":
            print(f"Call {i} -> success     (remaining {result['budget_remaining']})")
        else:
            print(f"Call {i} -> {result}")


if __name__ == "__main__":
    main()
