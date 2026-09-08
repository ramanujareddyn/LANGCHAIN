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
# 1. API KEY
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )


# ============================================================
# 2. TOOLS
# ============================================================

@tool
def search_movies(genre: str) -> str:
    """
    Search for Indian movies by genre.
    """

    movies = {
        "sci-fi": [
            "Cargo",
            "2.0",
            "Mr. India"
        ],

        "comedy": [
            "3 Idiots",
            "Hera Pheri",
            "Munna Bhai M.B.B.S."
        ],

        "action": [
            "RRR",
            "Vikram",
            "Baahubali"
        ],

        "drama": [
            "12th Fail",
            "Dangal",
            "Taare Zameen Par"
        ],

        "thriller": [
            "Drishyam",
            "Andhadhun",
            "Ratsasan"
        ],

        "romance": [
            "Jab We Met",
            "Sita Ramam",
            "96"
        ]
    }

    genre = genre.lower().strip()

    if genre in movies:
        return (
            f"Indian {genre} movies: "
            + ", ".join(movies[genre])
        )

    return (
        f"No movies found for genre '{genre}'. "
        f"Available genres: "
        f"{', '.join(movies.keys())}"
    )


# ============================================================
# 3. CELSIUS TO FAHRENHEIT
# ============================================================

@tool
def change__to_f(temp_c: float) -> float:
    """
    Convert Celsius temperature to Fahrenheit.
    """

    return round((temp_c * 1.8) + 32, 2)


# ============================================================
# 4. WEATHER TOOL
# ============================================================

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for an Indian city.
    """

    try:

        # ----------------------------------------------------
        # GEOCODING API
        # ----------------------------------------------------

        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search"
        )

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

        if not geo_data.get("results"):
            return (
                f"Could not find the city '{city}'."
            )

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        # ----------------------------------------------------
        # WEATHER API
        # ----------------------------------------------------

        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
        )

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "apparent_temperature,"
                "weather_code"
            ),
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
            "city": location["name"],
            "country": location.get(
                "country",
                "India"
            ),
            "temperature_celsius": current[
                "temperature_2m"
            ],
            "feels_like_celsius": current[
                "apparent_temperature"
            ],
            "humidity_percent": current[
                "relative_humidity_2m"
            ],
            "weather_code": current[
                "weather_code"
            ]
        }

        return json.dumps(result)

    except requests.RequestException as e:

        return (
            f"Weather API request failed: {str(e)}"
        )

    except Exception as e:

        return (
            f"Unable to get weather information: {str(e)}"
        )


# ============================================================
# 5. TOOL LIST
# ============================================================

tools = [
    get_weather,
    search_movies,
    change__to_f
]


# ============================================================
# 6. GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)


# ============================================================
# 7. CREATE AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools,

    system_prompt="""
You are an Indian Weather and Cinema Agent.

You are ONLY authorized to answer questions about:

1. Weather in India
2. Indian movies and Indian cinema

For weather questions:
- Use the get_weather tool.
- Give the temperature clearly.
- Mention the city.
- Give useful weather information from the tool.

For Indian movie questions:
- Use the search_movies tool when the user asks for movies by genre.
- Recommend Indian movies only.

For Celsius to Fahrenheit conversion:
- Use the change__to_f tool when appropriate.

IMPORTANT:
If the user asks about anything outside Indian weather,
Indian movies, Indian cinema, or temperature conversion,
you MUST reply exactly:

I am not authorized to answer questions outside of Indian weather and cinema.

Do not answer unrelated general knowledge questions.
"""
)


# ============================================================
# 8. INPUT MODEL
# ============================================================

class AgentInput(BaseModel):
    input: str = Field(
        description="Message for the Indian Weather and Cinema Agent"
    )


# ============================================================
# 9. FORMAT INPUT
# ============================================================

def format_for_agent(x):

    if isinstance(x, dict):

        user_input = x.get("input", "")

    else:

        user_input = x.input

    return {
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ]
    }


# ============================================================
# 10. EXTRACT RESPONSE
# ============================================================

def extract_text_response(agent_output):

    # --------------------------------------------------------
    # If already a string
    # --------------------------------------------------------

    if isinstance(agent_output, str):
        return agent_output

    # --------------------------------------------------------
    # If dictionary
    # --------------------------------------------------------

    if isinstance(agent_output, dict):

        messages = agent_output.get(
            "messages"
        )

        # ----------------------------------------------------
        # Find messages in nested objects
        # ----------------------------------------------------

        if messages is None:

            for value in agent_output.values():

                if (
                    isinstance(value, dict)
                    and "messages" in value
                ):

                    messages = value["messages"]

                    break

        # ----------------------------------------------------
        # Extract last message
        # ----------------------------------------------------

        if messages:

            last_message = messages[-1]

            content = getattr(
                last_message,
                "content",
                None
            )

            if content is None:

                if isinstance(
                    last_message,
                    dict
                ):

                    content = last_message.get(
                        "content"
                    )

            if content is not None:

                # Gemini may sometimes return
                # structured content.

                if isinstance(
                    content,
                    str
                ):

                    return content

                if isinstance(
                    content,
                    list
                ):

                    text_parts = []

                    for item in content:

                        if isinstance(
                            item,
                            dict
                        ):

                            if "text" in item:

                                text_parts.append(
                                    item["text"]
                                )

                    if text_parts:

                        return "\n".join(
                            text_parts
                        )

                return str(content)

    return str(agent_output)


# ============================================================
# 11. CREATE LANGCHAIN CHAIN
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
# 12. FASTAPI APP
# ============================================================

app = FastAPI(
    title="Indian Weather and Cinema Agent",
    description=(
        "AI Agent for Indian weather and Indian cinema "
        "using LangChain, Gemini and LangServe."
    ),
    version="1.0.0"
)


# ============================================================
# 13. HOME ROUTE
# ============================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "message": (
            "Indian Weather and Cinema Agent "
            "is running successfully."
        ),
        "playground": "/agent/playground/",
        "docs": "/docs",
        "health": "/health"
    }


# ============================================================
# 14. HEALTH ROUTE
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# 15. LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default"
)


# ============================================================
# 16. RUN SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
