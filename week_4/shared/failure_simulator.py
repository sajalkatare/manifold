"""
shared/failure_simulator.py

Purpose:
    Deterministic failure modes so live class never depends on a real API being
    flaky, plus a tiny product fetch the demos can call.

Why this exists:
    We make failure a switch WE control. The same `simulate_failure(...)` call
    drives every demo, so the lesson behaves identically every run.

Failure types -> exception raised:
    "timeout"          -> SimulatedTimeoutError
    "server_error"     -> SimulatedServerError
    "rate_limit"       -> SimulatedRateLimitError
    "invalid_response" -> SimulatedInvalidResponseError
    "invalid_input"    -> ValueError        (classified as non-retryable)
    None               -> does nothing (success path)

Real API (no key needed):
    https://dummyjson.com/products/{product_id}  — used when use_real_api=True.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import requests
except ImportError:  # only needed for the real-API path
    requests = None  # type: ignore

API_BASE = "https://dummyjson.com/products"

# A canned product so the simulated "success" path is predictable offline.
CANNED_PRODUCT = {
    "id": 1,
    "title": "Essence Mascara Lash Princess",
    "category": "beauty",
    "price": 9.99,
}


# --- Custom, named exceptions (clearer than generic ones in a live demo) ----
class SimulatedTimeoutError(Exception):
    """The external call took too long."""


class SimulatedServerError(Exception):
    """The external service returned a 5xx-style error."""


class SimulatedRateLimitError(Exception):
    """The external service is rate limiting us (429-style)."""


class SimulatedInvalidResponseError(Exception):
    """The external service returned data we cannot read."""


def simulate_failure(failure_type: str | None) -> None:
    """Raise a simulated failure, or do nothing if failure_type is None."""
    if failure_type is None:
        return
    if failure_type == "timeout":
        raise SimulatedTimeoutError("Simulated timeout")
    if failure_type == "server_error":
        raise SimulatedServerError("Simulated 500 server error")
    if failure_type == "rate_limit":
        raise SimulatedRateLimitError("Simulated 429 rate limit")
    if failure_type == "invalid_response":
        raise SimulatedInvalidResponseError("Simulated malformed response")
    if failure_type == "invalid_input":
        raise ValueError("Simulated invalid input")
    # Unknown failure types are ignored (treated as success).


def fetch_product(product_id: int, timeout_seconds: int = 3, use_real_api: bool = False) -> dict:
    """Return the RAW product dict — canned by default, or live DummyJSON.

    Raises on real-API network/HTTP problems so the caller can handle them.
    """
    if use_real_api and requests is not None:
        resp = requests.get(f"{API_BASE}/{product_id}", timeout=timeout_seconds)
        if resp.status_code == 404:
            raise SimulatedInvalidResponseError(f"No product {product_id} (404).")
        if resp.status_code != 200:
            raise SimulatedServerError(f"Service returned {resp.status_code}.")
        return resp.json()

    product = dict(CANNED_PRODUCT)
    product["id"] = product_id
    return product


if __name__ == "__main__":
    print(fetch_product(1))
    for ft in ("timeout", "server_error", "rate_limit", "invalid_response", "invalid_input", None):
        try:
            simulate_failure(ft)
            print(f"{ft!r:18} -> no exception")
        except Exception as exc:
            print(f"{ft!r:18} -> {type(exc).__name__}")
