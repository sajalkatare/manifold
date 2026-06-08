from typing import TypedDict
from langgraph.graph import StateGraph, END


# ===== 1. STATE — the shared clipboard every node reads/writes =====
class SupportState(TypedDict):
    user_message: str
    customer_tier: str     # 'premium' | 'standard'
    issue_category: str    # 'billing' | 'shipping' | 'technical'
    response: str


# ===== 2. NODES — read state, do work, return updates =====
def classify_issue(state: SupportState) -> dict:
    msg = state["user_message"].lower()
    if any(w in msg for w in ["refund", "charge", "billing", "payment", "invoice"]):
        category = "billing"
    elif any(w in msg for w in ["shipping", "delivery", "package", "track"]):
        category = "shipping"
    else:
        category = "technical"
    print(f"  [classify_issue]  -> {category}")
    return {"issue_category": category}


def lookup_customer(state: SupportState) -> dict:
    msg = state["user_message"].lower()
    tier = "premium" if ("premium" in msg or "vip" in msg) else "standard"
    print(f"  [lookup_customer] -> {tier}")
    return {"customer_tier": tier}


def handle_premium(state: SupportState) -> dict:
    print("  [handle_premium]  -> priority path")
    return {"response": (
        f"Dear Premium member, your {state['issue_category']} issue is prioritized. "
        "A senior agent will contact you within 1 hour (HIGH priority ticket created)."
    )}


def handle_standard(state: SupportState) -> dict:
    print("  [handle_standard] -> normal path")
    return {"response": (
        f"Thanks for contacting us about your {state['issue_category']} issue. "
        "We'll respond within 24 hours (ticket created)."
    )}


# ===== 3. CONDITIONAL EDGE — decide the next node from state =====
def route_by_tier(state: SupportState) -> str:
    return "handle_premium" if state["customer_tier"] == "premium" else "handle_standard"


# ===== 4. BUILD THE GRAPH =====
def build_graph():
    workflow = StateGraph(SupportState)
    workflow.add_node("classify_issue", classify_issue)
    workflow.add_node("lookup_customer", lookup_customer)
    workflow.add_node("handle_premium", handle_premium)
    workflow.add_node("handle_standard", handle_standard)

    workflow.set_entry_point("classify_issue")
    workflow.add_edge("classify_issue", "lookup_customer")
    workflow.add_conditional_edges(
        "lookup_customer",
        route_by_tier,
        {"handle_premium": "handle_premium", "handle_standard": "handle_standard"},
    )
    workflow.add_edge("handle_premium", END)
    workflow.add_edge("handle_standard", END)
    return workflow.compile()


# ===== 5. RUN — same graph, two inputs, two different paths =====
def run(graph, message: str, label: str) -> None:
    print(f"\n{label}")
    print("=" * 60)
    result = graph.invoke({
        "user_message": message, "customer_tier": "", "issue_category": "", "response": "",
    })
    print(f"\nFinal response: {result['response']}")


if __name__ == "__main__":
    graph = build_graph()
    run(graph, "Hi, I'm a premium member and I have a billing issue.", "TEST 1: Premium customer")
    run(graph, "I need help tracking my shipping.", "TEST 2: Standard customer")
    print("\nSame graph. Two inputs. Two visible, testable, auditable paths.")