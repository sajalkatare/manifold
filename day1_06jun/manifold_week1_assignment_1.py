########################################################
# Manifold Learning - Week 1 Assignment
# Author: Sajal Katare
# Date: 8th June 2024
# Assignment :  a customer-support agent that remembers the conversation, survives failures, and leaves a trace you can read.   
# 1. Configurable support persona
# 2. A chat() function
# 3. A 4-turn conversation
# 4. Conversation memory using messages
# 5. Basic error handling
# 6. Structured logging
# 7. Token usage / latency logging
# 8. Clean terminal output
########################################################


from dotenv import load_dotenv
import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import time

load_dotenv()
turn = 4  # number of messages to keep in memory (excluding system prompt)

company = "Malabar Jewellers"

support_persona = f"You are a customer support agent for {company}, an online jewelry store. You have a return policy of 15 days on all items. Be concise in responding to the customer. 2-3 sentences max."

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("malabar_support_agent")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
conversation = [SystemMessage(content=support_persona)] 

def chat(user_input: str, session_id: str = "sess-001") -> str:
    conversation.append(HumanMessage(content=user_input))
    start_time = time.time()
   
    try:
        response = llm.invoke(conversation)
        conversation.append(AIMessage(content=response.content))

        conversation[:] = [SystemMessage(content=support_persona)] + conversation[-turn*2:]  # keep last 4 messages + system prompt

        latency_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(f"Session: {session_id} | User: {user_input} | Response: {response.content} | Latency: {latency_ms}ms")
        
        return response.content
        
    except Exception as e:
        logger.error(f"Session: {session_id} | Error: {str(e)}")
        return "Sorry, something went wrong. Please try again later."
    
    conversation.append(response)                  # store reply -> continuity
   
    usage = response.usage_metadata or {}
   
    logger.info(json.dumps({
        "event": "turn_success", "session_id": session_id,
        "latency_ms": round((time.time() - start_time) * 1000, 2),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }))
    return response.content


if __name__ == "__main__":
    for msg in [
        "Hi, what is your return policy?",
        "How long is the return window?",
        "I bought a laptop 25 days ago — can I still return it?",
        "Great, how do I start the return?",
    ]:
        print(f"\nUser:  {msg}\nAgent: {chat(msg)}")