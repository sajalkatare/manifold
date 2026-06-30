"""Bounded retry with classification-aware fallback.

Run:
    python demos/d03_controlled_retry_and_fallback_learner.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.failure_simulator import fetch_product
from shared.failure_simulator import simulate_failure as trigger_failure
from shared.logging_utils import log_tool_attempt
from shared.retry_utils import classify_exception, is_retryable_error
from shared.tool_response import error_response, fallback_response, success_response

DEFAULT_MAX_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "2"))


def get_product_details_with_retry(
    product_id: int,
    simulate_failure_sequence: list[str] | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> dict:
    """Fetch a product with bounded retry and safe fallback."""
    sequence = simulate_failure_sequence or []
    input_data = {"product_id": product_id}

    for attempt in range(1, max_attempts + 1):
        failure = sequence[attempt - 1] if attempt - 1 < len(sequence) else None
        start = time.perf_counter()
        try:
            trigger_failure(failure)
            raw = fetch_product(product_id)
            latency = (time.perf_counter() - start) * 1000
            log_tool_attempt(
                "get_product_details_with_retry", input_data,
                attempt, "success", latency,
            )
            normalized = {
                "product_id": raw.get("id"),
                "title": raw.get("title"),
                "category": raw.get("category"),
                "price": raw.get("price"),
            }
            return success_response(normalized, source="simulator", attempts=attempt)
        except Exception as exc:
            error_type = classify_exception(exc)
            latency = (time.perf_counter() - start) * 1000
            log_tool_attempt(
                "get_product_details_with_retry", input_data,
                attempt, "error", latency, error_type=error_type,
            )

            if not is_retryable_error(error_type):
                return error_response(
                    error_type,
                    f"Non-retryable failure: {error_type}.",
                    retryable=False,
                )

    return fallback_response(
        "max_retries_exceeded",
        "I'm unable to fetch live details right now. Please try again shortly.",
    )


def main() -> None:
    print("=" * 64)
    print("DEMO 3 — Controlled retry and fallback")
    print("=" * 64)
    print(f"\n(MAX_RETRY_ATTEMPTS default = {DEFAULT_MAX_ATTEMPTS})")

    print("\n--- ['timeout', None] -> retry then success ---")
    print(get_product_details_with_retry(1, ["timeout", None], max_attempts=2))

    print("\n--- ['timeout', 'timeout'] -> retries fail -> fallback ---")
    print(get_product_details_with_retry(1, ["timeout", "timeout"], max_attempts=2))

    print("\n--- ['invalid_input'] -> non-retryable -> immediate error ---")
    print(get_product_details_with_retry(1, ["invalid_input"], max_attempts=2))


if __name__ == "__main__":
    main()
