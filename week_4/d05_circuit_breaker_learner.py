"""Circuit breaker: fail fast after repeated upstream failures.

Run:
    python demos/d05_circuit_breaker_learner.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.circuit_breaker import CircuitBreaker
from shared.failure_simulator import fetch_product
from shared.failure_simulator import simulate_failure as trigger_failure
from shared.logging_utils import log_tool_attempt
from shared.retry_utils import classify_exception, is_retryable_error
from shared.tool_response import error_response, success_response

FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "3"))


def get_product_details_with_circuit_breaker(
    product_id: int,
    circuit_breaker: CircuitBreaker,
    simulate_failure: str | None = None,
) -> dict:
    """Fetch a product under a simple circuit breaker."""
    input_data = {"product_id": product_id}
    circuit_state = circuit_breaker.state

    if not circuit_breaker.can_execute():
        log_tool_attempt(
            "get_product_details_with_circuit_breaker",
            input_data,
            1,
            "error",
            0.0,
            error_type="circuit_open",
            circuit_state=circuit_state,
        )
        return error_response(
            "circuit_open",
            "The product service circuit is open after repeated failures. "
            "Please try again later.",
            retryable=True,
            circuit_state=circuit_state,
        )

    start = time.perf_counter()
    try:
        trigger_failure(simulate_failure)
        raw = fetch_product(product_id)
        circuit_breaker.record_success()
        normalized = {
            "product_id": raw.get("id"),
            "title": raw.get("title"),
            "category": raw.get("category"),
            "price": raw.get("price"),
        }
        latency = (time.perf_counter() - start) * 1000
        log_tool_attempt(
            "get_product_details_with_circuit_breaker",
            input_data,
            1,
            "success",
            latency,
            circuit_state=circuit_breaker.state,
        )
        return success_response(
            normalized,
            source="simulator",
            circuit_state=circuit_breaker.state,
        )
    except Exception as exc:
        error_type = classify_exception(exc)
        if is_retryable_error(error_type):
            circuit_breaker.record_failure()
        latency = (time.perf_counter() - start) * 1000
        log_tool_attempt(
            "get_product_details_with_circuit_breaker",
            input_data,
            1,
            "error",
            latency,
            error_type=error_type,
            circuit_state=circuit_breaker.state,
        )
        return error_response(
            error_type,
            f"Product service failure: {error_type}.",
            retryable=is_retryable_error(error_type),
            circuit_state=circuit_breaker.state,
        )


def main() -> None:
    print("=" * 64)
    print("DEMO 5 — Circuit breaker")
    print("=" * 64)
    print(f"\n(CIRCUIT_FAILURE_THRESHOLD = {FAILURE_THRESHOLD})")

    print("\n--- Scenario 1: healthy call ---")
    breaker = CircuitBreaker(failure_threshold=FAILURE_THRESHOLD)
    print(get_product_details_with_circuit_breaker(1, breaker))
    print(f"   breaker -> {breaker.snapshot()}")

    print("\n--- Scenario 2: three server errors trip the circuit ---")
    breaker = CircuitBreaker(failure_threshold=FAILURE_THRESHOLD)
    for i in range(1, FAILURE_THRESHOLD + 1):
        print(f"\n  failure call {i}:")
        print(get_product_details_with_circuit_breaker(1, breaker, simulate_failure="server_error"))
    print(f"   breaker -> {breaker.snapshot()}")

    print("\n--- Scenario 3: call while OPEN (fast fail) ---")
    print(get_product_details_with_circuit_breaker(1, breaker))
    print(f"   breaker -> {breaker.snapshot()}")

    print("\n--- Scenario 4: non-retryable error does NOT trip the circuit ---")
    breaker = CircuitBreaker(failure_threshold=FAILURE_THRESHOLD)
    print(get_product_details_with_circuit_breaker(1, breaker, simulate_failure="invalid_input"))
    print(f"   breaker after invalid_input -> {breaker.snapshot()}")
    print(get_product_details_with_circuit_breaker(1, breaker))
    print(f"   breaker after success -> {breaker.snapshot()}")

    print("\n--- Scenario 5: reset() then success ---")
    breaker = CircuitBreaker(failure_threshold=FAILURE_THRESHOLD)
    for _ in range(FAILURE_THRESHOLD):
        get_product_details_with_circuit_breaker(1, breaker, simulate_failure="timeout")
    print(f"   breaker tripped -> {breaker.snapshot()}")
    breaker.reset()
    print(f"   breaker after reset -> {breaker.snapshot()}")
    print(get_product_details_with_circuit_breaker(1, breaker))


if __name__ == "__main__":
    main()
