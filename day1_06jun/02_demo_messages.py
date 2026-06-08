from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
load_dotenv()

llm = ChatOpenAI(model="gpt-5.5")

conversation = [
    SystemMessage(content="You are a helpful assistant that can answer questions and respond in a funny mannerˀ."),
    HumanMessage(content="Who is the Prime Minister of India?"),
]

r1 = llm.invoke(conversation)
print(r1.content)
print("--------------------------------")
print("response object: ", r1)
conversation.append(r1)
conversation.append(HumanMessage(content="What is his age?"))
print("--------------------------------")
print("Updated conversation: ", conversation)
print("--------------------------------")
r2 = llm.invoke(conversation)
print(r2.content)
print("--------------------------------")
print("response object: ", r2)