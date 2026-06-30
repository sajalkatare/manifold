"""
shared/tool_response.py

Purpose:
    One predictable response envelope for every tool — three deliberate states:
    success, error, and fallback.

Session 2 lens:
    Tool execution is a reliability boundary. The agent should always receive a
    predictable shape, and an error should say whether it is worth retrying.
"""

from typing import Any


def success_response(data: dict, source: str = "local", **extra) -> dict:
    """Wrap a successful result in the standard envelope.

    Args:
        data: The payload (e.g. {"product_id": 1, "title": "..."}).
        source: Where the data came from ("local", "dummyjson_api", ...).
        **extra: Any extra fields to attach (e.g. budget_remaining=2).
    """
    envelope: dict[str, Any] = {
        "status": "success",
        "source": source,
        "safe_for_user": True,
    }
    envelope.update(data)
    envelope.update(extra)
    return envelope


def error_response(
    error_type: str,
    message: str,
    safe_for_user: bool = True,
    retryable: bool = False,
    **extra,
) -> dict:
    """Wrap a failure in the standard envelope.

    Args:
        error_type: short machine-readable category (timeout, server_error,
            rate_limited, invalid_input, not_found, invalid_response,
            budget_exceeded, ...).
        message: human-readable, user-safe explanation (no stack traces).
        safe_for_user: whether this message can be shown to an end user.
        retryable: whether the caller may retry this failure.
        **extra: extra fields (e.g. budget_remaining=0).
    """
    envelope: dict[str, Any] = {
        "status": "error",
        "error_type": error_type,
        "message": message,
        "safe_for_user": safe_for_user,
        "retryable": retryable,
    }
    envelope.update(extra)
    return envelope


def fallback_response(
    error_type: str,
    message: str,
    safe_for_user: bool = True,
    **extra,
) -> dict:
    """Wrap a safe give-up in the standard envelope.

    A fallback is NOT an error and NOT a success. It is a deliberate, user-safe
    response returned after retries are exhausted or a limit is hit.
    """
    envelope: dict[str, Any] = {
        "status": "fallback",
        "error_type": error_type,
        "message": message,
        "safe_for_user": safe_for_user,
    }
    envelope.update(extra)
    return envelope


if __name__ == "__main__":
    print(success_response({"product_id": 1, "title": "Demo"}, source="dummyjson_api", budget_remaining=2))
    print(error_response("timeout", "The service timed out.", retryable=True))
    print(fallback_response("max_retries_exceeded", "Please try again shortly."))
