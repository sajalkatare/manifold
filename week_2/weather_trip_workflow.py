"""
LangGraph workflow: ask a trip question -> extract origin/destination/dates
-> look up the weather at the destination -> answer in plain language.

Setup:
    pip install langgraph langchain-anthropic requests
    export ANTHROPIC_API_KEY=sk-ant-...

Run:
    python weather_trip_workflow.py
"""

import os
import requests
from datetime import date
from typing import Optional, TypedDict

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# 1. State — the data that flows between nodes in the graph
# ---------------------------------------------------------------------------
class TripState(TypedDict, total=False):
    user_query: str
    origin: Optional[str]
    destination: Optional[str]
    start_date: Optional[str]   # YYYY-MM-DD
    end_date: Optional[str]     # YYYY-MM-DD
    weather_summary: Optional[str]
    final_answer: str
    error: Optional[str]


# ---------------------------------------------------------------------------
# 2. Node: extract trip info from the user's free-text question with an LLM
# ---------------------------------------------------------------------------
class TripInfo(BaseModel):
    origin: Optional[str] = Field(None, description="City the user is traveling FROM, if mentioned")
    destination: str = Field(..., description="City the user is traveling TO")
    start_date: Optional[str] = Field(
        None, description="Trip start date in YYYY-MM-DD format, if mentioned or inferable. "
                           f"Today's date is {date.today().isoformat()}."
    )
    end_date: Optional[str] = Field(
        None, description="Trip end date in YYYY-MM-DD format, if mentioned or inferable."
    )


llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
structured_llm = llm.with_structured_output(TripInfo)


def extract_trip_info(state: TripState) -> TripState:
    try:
        info = structured_llm.invoke(
            "Extract the trip details from this question. If no destination is "
            "mentioned at all, set destination to an empty string.\n\n"
            f"Question: {state['user_query']}"
        )
    except Exception as e:
        return {"error": f"Could not parse the question: {e}"}

    if not info.destination:
        return {"error": "I couldn't figure out where you're traveling to — could you name a destination city?"}

    return {
        "origin": info.origin,
        "destination": info.destination,
        "start_date": info.start_date,
        "end_date": info.end_date,
    }


# ---------------------------------------------------------------------------
# 3. Node: geocode the destination and fetch weather (Open-Meteo, no API key)
# ---------------------------------------------------------------------------
def get_weather(state: TripState) -> TripState:
    if state.get("error"):
        return {}

    destination = state["destination"]

    # 3a. Geocode the city name -> lat/lon
    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": destination, "count": 1},
            timeout=10,
        )
        geo_resp.raise_for_status()
        results = geo_resp.json().get("results")
        if not results:
            return {"error": f"Couldn't find a location called '{destination}'."}
        lat, lon = results[0]["latitude"], results[0]["longitude"]
        resolved_name = f"{results[0]['name']}, {results[0].get('country', '')}".strip(", ")
    except Exception as e:
        return {"error": f"Geocoding failed: {e}"}

    # 3b. Decide which weather endpoint to use based on the dates
    today = date.today()
    start = state.get("start_date") or today.isoformat()
    end = state.get("end_date") or start

    try:
        start_d = date.fromisoformat(start)
    except ValueError:
        start_d = today

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "auto",
        "start_date": start,
        "end_date": end,
    }

    # Open-Meteo's free forecast endpoint only covers ~16 days ahead.
    # Anything further out (or in the past beyond ~90 days) falls back to the
    # historical/climate archive endpoint as a reasonable estimate.
    days_out = (start_d - today).days
    if -90 <= days_out <= 15:
        url = "https://api.open-meteo.com/v1/forecast"
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"

    try:
        weather_resp = requests.get(url, params=params, timeout=10)
        weather_resp.raise_for_status()
        daily = weather_resp.json().get("daily", {})
    except Exception as e:
        return {"error": f"Weather lookup failed: {e}"}

    if not daily.get("time"):
        return {"error": f"No weather data available for {resolved_name} on those dates."}

    # 3c. Turn the raw daily arrays into a short text summary
    code_map = {
        0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
        45: "fog", 48: "depositing rime fog",
        51: "light drizzle", 53: "drizzle", 55: "dense drizzle",
        61: "light rain", 63: "rain", 65: "heavy rain",
        71: "light snow", 73: "snow", 75: "heavy snow",
        80: "rain showers", 81: "rain showers", 82: "violent rain showers",
        95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm with hail",
    }

    lines = []
    for i, day in enumerate(daily["time"]):
        code = daily["weathercode"][i]
        tmax = daily["temperature_2m_max"][i]
        tmin = daily["temperature_2m_min"][i]
        rain_chance = daily.get("precipitation_probability_max", [None])[i]
        desc = code_map.get(code, f"weather code {code}")
        line = f"{day}: {desc}, {tmin:.0f}–{tmax:.0f}°C"
        if rain_chance is not None:
            line += f", {rain_chance:.0f}% chance of precipitation"
        lines.append(line)

    summary = f"Weather for {resolved_name}:\n" + "\n".join(lines)
    return {"weather_summary": summary, "destination": resolved_name}


# ---------------------------------------------------------------------------
# 4. Node: compose the final natural-language answer
# ---------------------------------------------------------------------------
def compose_answer(state: TripState) -> TripState:
    if state.get("error"):
        return {"final_answer": state["error"]}

    prompt = (
        "Write a short, friendly 2-4 sentence answer to the user's question using "
        "this weather data. Mention what to pack or expect if relevant.\n\n"
        f"User's question: {state['user_query']}\n\n"
        f"Weather data:\n{state['weather_summary']}"
    )
    reply = llm.invoke(prompt).content
    return {"final_answer": reply}


# ---------------------------------------------------------------------------
# 5. Build the graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(TripState)
    graph.add_node("extract_trip_info", extract_trip_info)
    graph.add_node("get_weather", get_weather)
    graph.add_node("compose_answer", compose_answer)

    graph.add_edge(START, "extract_trip_info")
    graph.add_edge("extract_trip_info", "get_weather")
    graph.add_edge("get_weather", "compose_answer")
    graph.add_edge("compose_answer", END)

    return graph.compile()


app = build_graph()


# ---------------------------------------------------------------------------
# 6. Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY before running this script.")
        raise SystemExit(1)

    while True:
        query = input("\nAsk about your trip's weather (or 'quit'): ").strip()
        if query.lower() in {"quit", "exit"}:
            break
        result = app.invoke({"user_query": query})
        print("\n" + result["final_answer"])
