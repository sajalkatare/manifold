"""
shared/circuit_breaker.py

Purpose:
    A simple per-run circuit breaker. After enough consecutive retryable
    failures, stop calling the upstream service and fail fast with a safe
    circuit_open response.
"""


class CircuitBreaker:
    """Tracks consecutive failures and opens the circuit when a threshold is hit."""

    CLOSED = "closed"
    OPEN = "open"

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.consecutive_failures = 0
        self.state = self.CLOSED

    def can_execute(self) -> bool:
        """True when the circuit is closed and upstream calls are allowed."""
        return self.state == self.CLOSED

    def record_success(self) -> None:
        """Reset failure count and close the circuit."""
        self.consecutive_failures = 0
        self.state = self.CLOSED

    def record_failure(self) -> None:
        """Count one failure; open the circuit when the threshold is reached."""
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.failure_threshold:
            self.state = self.OPEN

    def reset(self) -> None:
        """Manually close the circuit (e.g. after a service recovery)."""
        self.consecutive_failures = 0
        self.state = self.CLOSED

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "consecutive_failures": self.consecutive_failures,
            "failure_threshold": self.failure_threshold,
        }


if __name__ == "__main__":
    breaker = CircuitBreaker(failure_threshold=3)
    for i in range(1, 6):
        if not breaker.can_execute():
            print(f"call {i}: BLOCKED (circuit open) -> {breaker.snapshot()}")
            continue
        breaker.record_failure()
        print(f"call {i}: failure recorded -> {breaker.snapshot()}")
