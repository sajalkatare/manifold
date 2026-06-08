from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

def get_weather(city: str) -> str:
    """
    Get the current weather for a given city
    Use this when the user asks about weather, temperature or climate.
    
    Args:
        city: the name of the city (e.g. 'Mumbai', 'Delhi', 'London')
    Returns:
        The current weather for the given city
    """
    weather_data = {
        'mumbai': 'sunny',
        'delhi': 'cloudy',
        'london': 'rainy',
    }
    weather = weather_data.get(city.lower(), 'unknown')
    return f"The weather in {city} is {weather}"

agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)
r1 = agent.invoke({"messages": [{"role": "user",
              "content": "what is the weather in mumbai"}]}) # documentation approach
print(r1)
print("--------------------------------")
r2 = agent.invoke({"messages": [HumanMessage(content="what is the weather in delhi")]}) # direct approach
print(r2)
print("--------------------------------")
r3 = agent.invoke({"messages": [HumanMessage(content="Who is the Author of the book 'The Great Gatsby'?")]}) # direct approach
print(r3)
print("--------------------------------")