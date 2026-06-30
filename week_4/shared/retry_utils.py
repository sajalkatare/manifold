"""
shared/retry_utils.py

Purpose:
    The two small decisions every retry layer needs:
    - is this failure worth retrying?
    - what kind of failure is this exception?

Teaching point:
    Retry is not automatically good engineering. We retry ONLY transient,
    retryable failures — never bad input or a not-found. The actual retry loop
    lives in the demo tool (d03), built on top of these helpers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.failure_simulator import (
    SimulatedInvalidResponseError,
    SimulatedRateLimitError,
    SimulatedServerError,
    SimulatedTimeoutError,
)

# Transient — may succeed if we try again.
RETRYABLE = {"timeout", "server_error", "rate_limited"}
# Permanent — trying again will fail the same way.
NON_RETRYABLE = {"invalid_input", "unauthorized", "not_found", "invalid_response"}


def is_retryable_error(error_type: str) -> bool:
    """True if this error_type is worth retrying."""
    return error_type in RETRYABLE


def classify_exception(exc: Exception) -> str:
    """Map a raised exception to a stable error_type string."""
    if isinstance(exc, (SimulatedTimeoutError, TimeoutError)):
        return "timeout"
    if isinstance(exc, SimulatedServerError):
        return "server_error"
    if isinstance(exc, SimulatedRateLimitError):
        return "rate_limited"
    if isinstance(exc, SimulatedInvalidResponseError):
        return "invalid_response"
    if isinstance(exc, ValueError):
        return "invalid_input"
    return "server_error"  # unknown failures: treat as a transient server issue


if __name__ == "__main__":
    print("timeout retryable?", is_retryable_error("timeout"))
    print("invalid_input retryable?", is_retryable_error("invalid_input"))
    print(classify_exception(SimulatedRateLimitError()))
    print(classify_exception(ValueError()))
