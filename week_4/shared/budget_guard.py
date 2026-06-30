"""
shared/budget_guard.py

Purpose:
    A simple per-run tool-call budget.

Teaching point:
    A budget protects the system from infinite loops, cost explosions, and
    rate-limit pile-ups. When the budget is hit, the block must be SAFE and
    LOGGED — never a crash, never a silent skip.
"""


class ToolBudgetGuard:
    """Tracks how many tool calls a single run is allowed to make."""

    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.used_calls = 0

    def can_call(self, tool_name: str | None = None) -> bool:
        """True if another call is permitted under the budget."""
        return self.used_calls < self.max_calls

    def record_call(self, tool_name: str | None = None) -> None:
        """Count one used call."""
        self.used_calls += 1

    def remaining(self) -> int:
        """How many calls are still allowed."""
        return max(0, self.max_calls - self.used_calls)

    def snapshot(self) -> dict:
        """A small dict for logging / debugging the budget state."""
        return {
            "max_calls": self.max_calls,
            "used_calls": self.used_calls,
            "remaining": self.remaining(),
        }


if __name__ == "__main__":
    guard = ToolBudgetGuard(max_calls=2)
    for i in range(1, 4):
        if guard.can_call():
            guard.record_call()
            print(f"call {i}: allowed -> {guard.snapshot()}")
        else:
            print(f"call {i}: BLOCKED -> {guard.snapshot()}")
