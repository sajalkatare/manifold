
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.failure_simulator import SimulatedTimeoutError, fetch_product
from shared.failure_simulator import simulate_failure as trigger_failure
from shared.logging_utils import log_tool_attempt
from shared.tool_response import error_response, success_response

TIMEOUT_SECONDS = int(os.getenv("API_TIMEOUT_SECONDS", "3"))


def get_product_details_safe(product_id: int, simulate_timeout: bool = False) -> dict:
    """Fetch product details; timeouts return a safe, retryable error envelope."""
    start = time.perf_counter()
    input_data = {"product_id": product_id}

    try:
        if simulate_timeout:
            trigger_failure("timeout")
        raw = fetch_product(product_id, timeout_seconds=TIMEOUT_SECONDS)
    except (SimulatedTimeoutError, TimeoutError):
        latency = (time.perf_counter() - start) * 1000
        log_tool_attempt(
            "get_product_details_safe", input_data, 1, "error",
            latency, error_type="timeout",
        )
        return error_response(
            "timeout",
            "The product service took too long to respond.",
            retryable=True,
        )

    normalized = {
        "product_id": raw.get("id"),
        "title": raw.get("title"),
        "category": raw.get("category"),
        "price": raw.get("price"),
    }
    latency = (time.perf_counter() - start) * 1000
    log_tool_attempt("get_product_details_safe", input_data, 1, "success", latency)
    return success_response(normalized, source="simulator")


def main() -> None:
    print("=" * 64)
    print("DEMO 2 — Timeout and safe failure")
    print("=" * 64)

    print(f"\n(API_TIMEOUT_SECONDS = {TIMEOUT_SECONDS})")

    print("\n--- Healthy call ---")
    print(get_product_details_safe(1))

    print("\n--- Timeout (handled safely) ---")
    print(get_product_details_safe(1, simulate_timeout=True))


if __name__ == "__main__":
    main()
