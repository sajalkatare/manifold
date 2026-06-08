from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

llm = ChatOpenAI(model="gpt-5.5", temperature=1, max_tokens=150)

# Q1
r1 = llm.invoke("Who is the Prime Minister of India?")
print(r1.content)

# Q2 — the follow-up that breaks
r2 = llm.invoke("What is his age?")
print(r2.content)            # "Who are you referring to?"

print(r1.usage_metadata)