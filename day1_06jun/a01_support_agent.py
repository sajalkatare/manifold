########################################################
# Manifold Learning - Week 1 Assignment
# Author: Sajal Katare
# Date: 8th June 2024
# Assignment : a customer-support agent that remembers
# the conversation, survives failures, and leaves a
# trace you can read.
#
# Features:
# 1. Configurable support persona
# 2. A chat() function
# 3. A 4-turn conversation
# 4. Conversation memory using messages
# 5. Basic error handling
# 6. Structured logging
# 7. Token usage / latency logging
# 8. Clean terminal output
# 9. JSON log file output
########################################################

from dotenv import load_dotenv
import json
import logging
import time
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

from openai import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    AuthenticationError,
    BadRequestError
)

# ------------------------------------------------------
# Configuration
# ------------------------------------------------------

load_dotenv()

COMPANY = "Malabar Jewellers"
TURN_MEMORY = 4
LOG_FILE = "support_agent_logs.jsonl"

SUPPORT_PERSONA = (
    f"You are a customer support agent for {COMPANY}, "
    "an online jewelry store. "
    "You have a return policy of 15 days on all items. "
    "Be concise in responding to customers. "
    "Respond in 2-3 sentences maximum."
)

# ------------------------------------------------------
# Logging Setup
# ------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger("malabar_support_agent")

# ------------------------------------------------------
# LLM Setup
# ------------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)

conversation = [
    SystemMessage(content=SUPPORT_PERSONA)
]

# ------------------------------------------------------
# Utility Functions
# ------------------------------------------------------

def write_json_log(record: dict) -> None:
    """
    Write structured logs to a JSON Lines file.
    One JSON object per line.
    """
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def log_event(record: dict, level: str = "info") -> None:
    """
    Log to console and JSON file.
    """
    if level == "error":
        logger.error(json.dumps(record))
    else:
        logger.info(json.dumps(record))

    write_json_log(record)

# ------------------------------------------------------
# Chat Function
# ------------------------------------------------------

def chat(
    user_input: str,
    session_id: str = "sess-001"
) -> str:

    if not user_input.strip():
        return "Please enter a valid question."

    start_time = time.time()

    try:
        # Store user message
        conversation.append(
            HumanMessage(content=user_input)
        )

        # Call model
        response = llm.invoke(conversation)

        # Store AI response
        conversation.append(
            AIMessage(content=response.content)
        )

        # Keep only:
        # SystemMessage + last N turns
        conversation[:] = (
            [conversation[0]]
            + conversation[-(TURN_MEMORY * 2):]
        )

        latency_ms = round(
            (time.time() - start_time) * 1000,
            2
        )

        usage = response.usage_metadata or {}

        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "turn_success",
            "session_id": session_id,
            "user_input": user_input,
            "response": response.content,
            "latency_ms": latency_ms,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "memory_turns": TURN_MEMORY
        }

        log_event(log_record)

        return response.content

    except AuthenticationError:

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "authentication_error",
            "session_id": session_id
        }

        log_event(record, "error")

        return (
            "Authentication failed. "
            "Please contact support."
        )

    except RateLimitError:

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "rate_limit_error",
            "session_id": session_id
        }

        log_event(record, "error")

        return (
            "The service is currently busy. "
            "Please try again shortly."
        )

    except APITimeoutError:

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "timeout_error",
            "session_id": session_id
        }

        log_event(record, "error")

        return (
            "The request timed out. "
            "Please try again."
        )

    except APIConnectionError:

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "connection_error",
            "session_id": session_id
        }

        log_event(record, "error")

        return (
            "Unable to connect to the AI service."
        )

    except BadRequestError:

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "bad_request_error",
            "session_id": session_id
        }

        log_event(record, "error")

        return (
            "Invalid request sent to the model."
        )

    except Exception as e:

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "unexpected_error",
            "session_id": session_id,
            "error": str(e)
        }

        log_event(record, "error")

        return (
            "Sorry, something went wrong. "
            "Please try again later."
        )

# ------------------------------------------------------
# Demo Conversation
# ------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print(f"{COMPANY} Customer Support Agent")
    print("=" * 60)

    messages = [
        "Hi, what is your return policy?",
        "How long is the return window?",
        "I bought a ring 25 days ago. Can I still return it?",
        "Great, how do I start the return?"
    ]

    for msg in messages:

        response = chat(msg)

        print(f"\nUser : {msg}")
        print(f"Agent: {response}")

    print("\n" + "=" * 60)
    print(f"Structured logs written to: {LOG_FILE}")
    print("=" * 60)
