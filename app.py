import os
import json
import requests
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langserve import add_routes
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent


# ============================================================
# 1. TOOLS
# ============================================================

@tool
def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""

    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali",
        "drama": "12th Fail, Dangal, Taare Zameen Par",
        "thriller": "Drishyam, Andhadhun, Ratsasan",
        "romance": "Jab We Met, Sita Ramam, 96"
    }

    return movies.get(
        genre.lower(),
        "No movies found for that genre."
    )


@tool
def change__to_f(temp_c: float) -> float:
    """Convert Celsius temperature to Fahrenheit."""

    return round((temp_c * 1.8) + 32, 2)


@tool
def get_weather(city: str) -> str:
    """Get current weather for an Indian city."""

    try:
        # ----------------------------------------------------
        # Geocoding
        # ----------------------------------------------------

        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_params = {
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json"
        }

        geo_response = requests.get(
            geo_url,
            params=geo_params,
            timeout=10
        )

        geo_response.raise_for_status()

        geo_data = geo_response.json()

        if "results" not in geo_data:
            return f"Could not find weather data for {city}."

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        # ----------------------------------------------------
        # Weather
        # ----------------------------------------------------

        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
            "temperature_unit": "celsius"
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()

        current = weather_data["current"]

        result = {
            "resolved_city": location["name"],
            "country": location.get("country", "India"),
            "temperature_celsius": current["temperature_2m"],
            "weather_code": current["weather_code"]
        }

        return json.dumps(result)

    except requests.RequestException as e:
        return f"Weather service error: {str(e)}"

    except Exception as e:
        return f"Unable to get weather information: {str(e)}"


# ============================================================
# 2. TOOLS LIST
# ============================================================

tools = [
    get_weather,
    search_movies,
    change__to_f
]


# ============================================================
# 3. GEMINI API KEY
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )


# ============================================================
# 4. GEMINI MODEL
# ============================================================

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GEMINI_API_KEY,
    temperature=0
)


# ============================================================
# 5. CREATE AGENT
# ============================================================

agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are an Indian Weather and Cinema Agent.\n\n"

        "You are ONLY authorized to answer questions related to:\n"
        "1. Weather in India\n"
        "2. Indian movies and cinema\n\n"

        "For weather questions, use the get_weather tool.\n"
        "For movie genre questions, use the search_movies tool.\n"
        "For Celsius to Fahrenheit conversion, use the "
        "change__to_f tool when relevant.\n\n"

        "If the user asks about anything outside Indian weather "
        "and Indian cinema, you must say exactly:\n"
        "'I am not authorized to answer questions outside of "
        "Indian weather and cinema.'"
    )
)


# ============================================================
# 6. INPUT MODEL
# ============================================================

class AgentInput(BaseModel):
    input: str = Field(
        description="Your message to the Indian Weather and Cinema Agent"
    )


# ============================================================
# 7. FORMAT INPUT
# ============================================================

def format_for_agent(x) -> dict:

    if isinstance(x, dict):
        user_input = x["input"]
    else:
        user_input = x.input

    return {
        "messages": [
            ("user", user_input)
        ]
    }


# ============================================================
# 8. EXTRACT AGENT RESPONSE
# ============================================================

def extract_text_response(agent_output) -> str:

    if not isinstance(agent_output, dict):
        return str(agent_output)

    # --------------------------------------------------------
    # Normal LangGraph response
    # --------------------------------------------------------

    messages = agent_output.get("messages")

    # --------------------------------------------------------
    # Search nested values if messages are not at top level
    # --------------------------------------------------------

    if messages is None:

        for value in agent_output.values():

            if (
                isinstance(value, dict)
                and "messages" in value
            ):
                messages = value["messages"]
                break

    # --------------------------------------------------------
    # Extract final message
    # --------------------------------------------------------

    if messages:

        last_message = messages[-1]

        content = getattr(
            last_message,
            "content",
            None
        )

        if content is not None:

            # Some Gemini/LangChain responses can return
            # structured content.
            if isinstance(content, str):
                return content

            return str(content)

        return str(last_message)

    return str(agent_output)


# ============================================================
# 9. CREATE LANGCHAIN CHAIN
# ============================================================

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# 10. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Indian Weather and Cinema Agent",
    description=(
        "AI agent for Indian weather and Indian cinema "
        "using LangChain, Gemini and LangServe."
    ),
    version="1.0.0"
)


# ============================================================
# 11. HOME / HEALTH ROUTE
# ============================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "message": "Indian Weather and Cinema Agent is running",
        "playground": "/agent/playground/",
        "docs": "/docs"
    }


# ============================================================
# 12. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# 13. LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default"
)


# ============================================================
# 14. RUN SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 8000)
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
