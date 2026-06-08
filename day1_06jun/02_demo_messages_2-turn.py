from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
load_dotenv()

llm = ChatOpenAI(model="gpt-5.5")


COMPANY = "TechShop"

# Permanent system prompt
system_message = SystemMessage(
    content=(
        f"You are a customer support agent for {COMPANY}, "
        "an electronics store. Be concise. 2-3 sentences max."
    )
)

# Stores only conversation history (excluding system prompt)
chat_history = []

def chat_turn(user_input):
    global chat_history

    # Add current user message
    chat_history.append(HumanMessage(content=user_input))

    # Build context:
    # System prompt + last 2 turns (4 messages) + current user message
    messages = [system_message] + chat_history[-5:]

    response = llm.invoke(messages)

    # Store AI response
    chat_history.append(
        AIMessage(content=response.content)
    )

    print(f"Agent: {response.content}")

chat_turn("What is your return policy?")
chat_turn("How long do I have to return something?")
chat_turn("I bought a laptop 25 days ago. Can I still return it?")
chat_turn("What if I lost the receipt?")