from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# BAD: the docstring tells the model almost nothing.
# The model cannot tell when to use it, what input format to send,
# or what the output means.
# ---------------------------------------------------------------------------
@tool
def check_course_access_bad(email: str) -> str:
    """Checks access."""
    return "ok"


# ---------------------------------------------------------------------------
# GOOD: the docstring is written FOR THE MODEL. It explains:
#   - when to use the tool
#   - the exact input format expected
#   - what the output fields mean
#   - the tool's limitations
# ---------------------------------------------------------------------------
@tool
def check_course_access_good(email: str) -> dict:
    """Check whether a learner has access to the enrolled course.

    Use this tool when a user asks whether they "have access", "are enrolled",
    or "can log in" to the course. Do NOT use it to grant or change access.

    Args:
        email: The learner's full email address, e.g. "ravi@example.com".
            Must be a single, well-formed email. Do not pass names or IDs.

    Returns:
        A dict:
        {
            "email": str,          # the email that was checked
            "has_access": bool,    # True if currently enrolled and active
            "plan": str,           # "premium" | "standard" | "none"
        }

    Limitations:
        - This is a read-only check. It cannot enroll or upgrade a user.
        - If the email is unknown, has_access is False and plan is "none".
    """
    has_access = email.strip().lower().endswith("example.com")
    return {
        "email": email,
        "has_access": has_access,
        "plan": "premium" if has_access else "none",
    }


def describe(tool_obj) -> None:
    """Print the three things the model actually sees about a tool."""
    print(f"name:        {tool_obj.name}")
    print(f"description: {tool_obj.description}")
    print(f"args schema: {tool_obj.args}")


def main() -> None:
    print("=" * 60)
    print("DEMO 2 — The docstring is the agent's instruction manual")
    print("=" * 60)

    print("\n=== BAD TOOL (vague docstring) ===")
    describe(check_course_access_bad)

    print("\n=== GOOD TOOL (model-facing docstring) ===")
    describe(check_course_access_good)

    print("\n--- Actually invoking the good tool ---")
    # .invoke() is how LangChain calls a @tool with a dict of args.
    print(check_course_access_good.invoke({"email": "ravi@example.com"}))
    print(check_course_access_good.invoke({"email": "stranger@gmail.com"}))




if __name__ == "__main__":
    main()
