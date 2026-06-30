"""
shared/logging_utils.py

Purpose:
    Structured logging for tool ATTEMPTS. In a retry world the trace must show
    every attempt, its status, and — where relevant — the remaining budget.

Teaching point:
    A demo gives output. A system gives a trace.
"""

import json
import time
from typing import Any


def log_tool_attempt(
    tool_name: str,
    input_data: dict,
    attempt: int,
    status: str,
    latency_ms: float,
    error_type: str | None = None,
    budget_remaining: int | None = None,
    circuit_state: str | None = None,
) -> None:
    """Print one structured log line for a single tool attempt.

    Args:
        tool_name: name of the tool.
        input_data: the arguments the tool received.
        attempt: which attempt number this was (1-based).
        status: "success" | "error" | "fallback" | "budget_exceeded".
        latency_ms: how long this attempt took, in milliseconds.
        error_type: failure category if this attempt failed.
        budget_remaining: remaining tool-call budget, if applicable.
    """
    log_line: dict[str, Any] = {
        "ts": round(time.time(), 3),
        "tool": tool_name,
        "attempt": attempt,
        "status": status,
        "latency_ms": round(latency_ms, 2),
        "input": input_data,
        "circuit_state": circuit_state,
    }
    if error_type is not None:
        log_line["error_type"] = error_type
    if budget_remaining is not None:
        log_line["budget_remaining"] = budget_remaining

    print("LOG " + json.dumps(log_line))


if __name__ == "__main__":
    log_tool_attempt("demo_tool", {"product_id": 1}, 1, "error", 12.3, error_type="timeout")
    log_tool_attempt("demo_tool", {"product_id": 1}, 2, "success", 8.1, budget_remaining=1)
