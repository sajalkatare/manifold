from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
load_dotenv()

llm = ChatOpenAI(model="gpt-5.5")

conversation = [
    SystemMessage(content="You are a helpful assistant that can answer questions and respond in a funny mannerˀ."),
    HumanMessage(content="Can I return the mobile phone I bought from Amazon.in?"),
]

r1 = llm.invoke(conversation)
print(r1.content)
print("--------------------------------")

conversation = [
    SystemMessage(content="You are a helpful assistant that can answer questions and respond in a funny mannerˀ."),
    HumanMessage(content="""Can I return the mobile phone I bought from Amazon.in?
    
    Policy	Defective use case	Damage, Wrong Item use case
7/10 Days Service Centre Replacement

For defective items, you will have to reach out to the brand service centre for further resolutions. For few items, we may facilitate scheduling a technician visit at your doorstep for troubleshooting only. Based on the assessment/ visit, final resolution (repair, refund, replacement) will be provided by brand only.

This item is eligible for free replacement, within 7/ 10 days of delivery in an unlikely event of damaged, or different/wrong item delivered to you.

Please keep the item in its original condition, with MRP tags attached, user manual, warranty cards, and original accessories in manufacturer packaging. We may contact you and / or carry out verification checks to ascertain the damage or defect in the item prior to issuing refund/replacement.

No Returns on Electronic Items

We do not accept returns on electronic items.

"""),
]
r1 = llm.invoke(conversation)
print(r1.content)
print("--------------------------------")


conversation = [
    SystemMessage(content="You are a helpful assistant that can answer questions and respond in a funny mannerˀ."),
    HumanMessage(content=""" What is Leave Policy of manifoldailearning.com """),
]
r1 = llm.invoke(conversation)
print(r1.content)
print("--------------------------------")
