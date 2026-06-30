"""Agent with recoverable tools: retry, fallback, budget guard, and circuit breaker.

Run:
    python demos/d06_agent_with_recoverable_tools_learner.py

Requires OPENAI_API_KEY in your environment (see .env.example).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv()


from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from d03_controlled_retry_and_fallback_learner import (
    get_product_details_with_retry as _with_retry,
)
from d04_rate_limit_and_budget_guard_learner import (
    get_product_details_with_budget as _with_budget,
)
from d05_circuit_breaker_learner import (
    get_product_details_with_circuit_breaker as _with_circuit,
)
from shared.budget_guard import ToolBudgetGuard
from shared.circuit_breaker import CircuitBreaker

FAILURE_SEQUENCE: list[str] | None = None
RATE_LIMIT = False
BUDGET = ToolBudgetGuard(max_calls=int(os.getenv("MAX_TOOL_CALLS_PER_RUN", "3")))
CIRCUIT = CircuitBreaker(failure_threshold=int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "3")))
CIRCUIT_FAILURE: str | None = None


def _reset(
    failure_sequence=None,
    rate_limit=False,
    budget_max=3,
    circuit_failure=None,
    fresh_circuit=True,
):
    global FAILURE_SEQUENCE, RATE_LIMIT, BUDGET, CIRCUIT, CIRCUIT_FAILURE
    FAILURE_SEQUENCE = failure_sequence
    RATE_LIMIT = rate_limit
    BUDGET = ToolBudgetGuard(max_calls=budget_max)
    if fresh_circuit:
        CIRCUIT = CircuitBreaker(failure_threshold=int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "3")))
    CIRCUIT_FAILURE = circuit_failure


def _trip_circuit():
    """Pre-trip the circuit with consecutive retryable failures (deterministic setup)."""
    threshold = int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "3"))
    for _ in range(threshold):
        _with_circuit(1, CIRCUIT, simulate_failure="server_error")


@tool
def fetch_product_with_retry(product_id: int) -> dict:
    """Fetch normalized product details with bounded retry and safe fallback.

    Returns a dict with status success | error | fallback.
    """
    return _with_retry(product_id, simulate_failure_sequence=FAILURE_SEQUENCE, max_attempts=2)


@tool
def fetch_product_with_budget(product_id: int) -> dict:
    """Fetch product details under a per-run tool-call budget.

    Returns a dict with status success | error (incl. budget_exceeded).
    """
    return _with_budget(product_id, BUDGET, simulate_rate_limit=RATE_LIMIT)


@tool
def fetch_product_with_circuit_breaker(product_id: int) -> dict:
    """Fetch product details protected by a circuit breaker.

    After repeated upstream failures the circuit opens and calls fail fast
    with a safe circuit_open response. Returns status success | error.
    """
    return _with_circuit(product_id, CIRCUIT, simulate_failure=CIRCUIT_FAILURE)


TOOLS = [fetch_product_with_retry, fetch_product_with_budget, fetch_product_with_circuit_breaker]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def run_agent_turn(model, user_text: str) -> None:
    print("\n" + "=" * 64)
    print(f"USER: {user_text}")
    print("=" * 64)

    messages = [HumanMessage(content=user_text)]
    ai_msg = model.invoke(messages)
    messages.append(ai_msg)

    if not ai_msg.tool_calls:
        print(f"AGENT (no tool used): {ai_msg.content}")
        return

    for call in ai_msg.tool_calls:
        print(f"AGENT wants tool: {call['name']}({call['args']})")
        result = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
        print(f"TOOL result: {result}")
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    final = model.invoke(messages)
    print(f"AGENT (final): {final.content}")


def main() -> None:
    print("=" * 64)
    print("DEMO 6 — Agent with recoverable tools")
    print("=" * 64)

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "\nOPENAI_API_KEY is not set.\n"
            "1) Copy .env.example to .env\n"
            "2) Put your key in OPENAI_API_KEY\n"
            "3) Re-run:  python demos/d06_agent_with_recoverable_tools_learner.py\n"
        )
        return

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(TOOLS)

    _reset(failure_sequence=[None])
    run_agent_turn(model, "Can you fetch details for product 1?")

    _reset(failure_sequence=["timeout", None])
    run_agent_turn(model, "Fetch details for product 1. (it may be slow at first)")

    _reset(failure_sequence=["timeout", "timeout"])
    run_agent_turn(model, "Fetch details for product 1.")

    _reset(budget_max=2)
    run_agent_turn(model, "Use the budgeted tool to fetch product 1, then 2, then 3.")

    _reset(fresh_circuit=True)
    _trip_circuit()
    print(f"\n[circuit pre-tripped -> {CIRCUIT.snapshot()}]")
    run_agent_turn(
        model,
        "Use the circuit breaker tool to fetch product 1.",
    )


if __name__ == "__main__":
    main()
