from binascii import b2a_base64
import random
import re
from operator import add
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()
llm = ChatOpenAI(model="gpt-5.5", temperature=0)


# ===== STATE =====
class SupportState(TypedDict):
    messages: Annotated[list[BaseMessage], add]   # accumulates across nodes
    customer_id: str
    customer_tier: str       # 'premium' | 'standard'
    issue_category: str      # 'billing' | 'shipping' | 'technical' | 'general'
    ticket_id: str
    final_response: str


# ===== TOOLS =====
@tool
def get_customer_tier(customer_id: str) -> str:
    """Look up a customer's tier ('premium' or 'standard') by their customer ID."""
    premium = {"CUST-001", "CUST-003", "CUST-007"}
    return "premium" if customer_id in premium else "standard"


@tool
def create_ticket(customer_id: str, issue: str, priority: str) -> str:
    """Create a support ticket. priority is 'low', 'normal', or 'high'. Returns a ticket id."""
    ticket_id = f"TKT-{random.randint(10000, 99999)}"
    print(f"      [DB] ticket {ticket_id} | {customer_id} | {issue} | priority={priority}")
    return ticket_id


# ===== NODES =====
def extract_customer_id(state: SupportState) -> dict:
    text = state["messages"][-1].content.upper()
    match = re.search(r"CUST-\d+", text)
    customer_id = match.group(0) if match else "CUST-UNKNOWN"
    print(f"  [extract]  customer_id = {customer_id}")
    return {"customer_id": customer_id}


def lookup_tier(state: SupportState) -> dict:
    tier = get_customer_tier.invoke({"customer_id": state["customer_id"]})
    print(f"  [lookup]   tier = {tier}")
    return {"customer_tier": tier}


def classify_with_llm(state: SupportState) -> dict:
    """Classify the issue with the LLM; fall back to keywords if the call fails."""
    text = state["messages"][-1].content
    valid = {"billing", "shipping", "technical", "general"}
    try:
        resp = llm.invoke([
            SystemMessage(content=(
                "Classify this support issue into exactly one of: "
                "billing, shipping, technical, general. Return ONLY the category word."
            )),
            HumanMessage(content=f"Issue: {text}"),
        ])
        category = resp.content.strip().lower()
        if category not in valid:
            category = "general"
    except Exception as e:  # graceful degradation (Day 1 lens)
        print(f"  [classify] LLM failed ({type(e).__name__}); using keyword fallback")
        low = text.lower()
        if any(w in low for w in ["refund", "charge", "billing", "payment"]):
            category = "billing"
        elif any(w in low for w in ["shipping", "delivery", "track", "package"]):
            category = "shipping"
        elif any(w in low for w in ["error", "bug", "broken", "login", "crash"]):
            category = "technical"
        else:
            category = "general"
    print(f"  [classify] issue_category = {category}")
    return {"issue_category": category}


def _respond(state: SupportState, priority: str, system_prompt: str) -> dict:
    ticket_id = create_ticket.invoke({
        "customer_id": state["customer_id"],
        "issue": state["issue_category"],
        "priority": priority,
    })
    try:
        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Customer issue: {state['messages'][-1].content}\n"
                f"Category: {state['issue_category']}\nTicket: {ticket_id}"
            )),
        ])
        text = resp.content
    except Exception:
        text = (f"Your {state['issue_category']} issue is logged as {ticket_id}. "
                "We'll be in touch shortly.")
    return {"ticket_id": ticket_id, "final_response": text,
            "messages": [AIMessage(content=text)]}


def handle_premium_path(state: SupportState) -> dict:
    print("  [route]    -> PREMIUM (priority)")
    return _respond(state, "high",
                    "You are a senior agent for a premium customer. Warm, specific, "
                    "promise a dedicated agent within 1 hour. Max 3 sentences.")


def handle_standard_path(state: SupportState) -> dict:
    print("  [route]    -> STANDARD (normal)")
    return _respond(state, "normal",
                    "You are a support agent. Professional, give the standard 24-hour "
                    "SLA. Max 3 sentences.")


# ===== ROUTING =====
def route_by_customer_tier(state: SupportState) -> str:
    return "handle_premium_path" if state["customer_tier"] == "premium" else "handle_standard_path"


# ===== BUILD =====
def build_graph():
    b = StateGraph(SupportState)
    b.add_node("extract_customer_id", extract_customer_id)
    b.add_node("lookup_tier", lookup_tier)
    b.add_node("classify_with_llm", classify_with_llm)
    b.add_node("handle_premium_path", handle_premium_path)
    b.add_node("handle_standard_path", handle_standard_path)

    b.set_entry_point("extract_customer_id")
    b.add_edge("extract_customer_id", "lookup_tier")
    b.add_edge("lookup_tier", "classify_with_llm")
    b.add_conditional_edges(
        "classify_with_llm",
        route_by_customer_tier,
        {"handle_premium_path": "handle_premium_path",
         "handle_standard_path": "handle_standard_path"},
    )
    b.add_edge("handle_premium_path", END)
    b.add_edge("handle_standard_path", END)
    app = b.compile()
    app.get_graph().print_ascii() 
    return app


def run(graph, message: str, label: str) -> None:
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    result = graph.invoke({
        "messages": [HumanMessage(content=message)],
        "customer_id": "", "customer_tier": "", "issue_category": "",
        "ticket_id": "", "final_response": "",
    })
    print(f"\nFINAL RESPONSE: {result['final_response']}")
    print(f"tier={result['customer_tier']} | category={result['issue_category']} "
          f"| ticket={result['ticket_id']}")


if __name__ == "__main__":
    graph = build_graph()
    run(graph, "Hi, my ID is CUST-001. I was double-charged for my subscription.",
        "TEST 1: Premium + billing")
    run(graph, "Hello, ID CUST-555, my package never arrived.",
        "TEST 2: Standard + shipping")