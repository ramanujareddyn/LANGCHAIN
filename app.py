import os
import json
import requests
import uvicorn

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from langserve import add_routes
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent


# ============================================================
# 1. GEMINI API KEY
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )


# ============================================================
# 2. WEATHER TOOL
# ============================================================

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for an Indian city.
    """

    try:
        # ----------------------------------------------------
        # Find city coordinates
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
            return f"Could not find the city '{city}'."

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        # ----------------------------------------------------
        # Get weather
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
            "country": location.get("country", "India"),
            "temperature_celsius": current["temperature_2m"],
            "feels_like_celsius": current["apparent_temperature"],
            "humidity_percent": current["relative_humidity_2m"],
            "weather_code": current["weather_code"]
        }

        return json.dumps(result)

    except requests.RequestException as e:

        return f"Weather API request failed: {str(e)}"

    except Exception as e:

        return f"Unable to get weather information: {str(e)}"


# ============================================================
# 3. MOVIE TOOL
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
        f"No movies found for '{genre}'. "
        f"Available genres: "
        f"{', '.join(movies.keys())}"
    )


# ============================================================
# 4. CELSIUS TO FAHRENHEIT TOOL
# ============================================================

@tool
def change__to_f(temp_c: float) -> float:
    """
    Convert Celsius temperature to Fahrenheit.
    """

    return round(
        (temp_c * 1.8) + 32,
        2
    )


# ============================================================
# 5. TOOLS
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
2. Indian movies
3. Indian cinema
4. Celsius to Fahrenheit conversion when related to weather

For weather questions:
- Use the get_weather tool.
- Always use the tool when the user asks about current weather.
- Clearly mention the city.
- Clearly mention the temperature.
- Mention humidity and feels-like temperature when available.

For Indian movie questions:
- Use search_movies when the user asks for movie recommendations by genre.
- Recommend Indian movies only.

For Celsius to Fahrenheit:
- Use change__to_f when appropriate.

IMPORTANT:
If the user asks about anything outside Indian weather,
Indian movies, Indian cinema, or weather-related temperature
conversion, reply EXACTLY:

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
# 9. FORMAT INPUT FOR LANGSERVE
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

    if isinstance(agent_output, str):
        return agent_output

    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    # --------------------------------------------------------
    # Search nested dictionaries
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
    # Get final message
    # --------------------------------------------------------

    if messages:

        last_message = messages[-1]

        # Object message
        content = getattr(
            last_message,
            "content",
            None
        )

        # Dictionary message
        if content is None:

            if isinstance(last_message, dict):

                content = last_message.get(
                    "content"
                )

        if content is not None:

            if isinstance(content, str):
                return content

            # Gemini structured content
            if isinstance(content, list):

                text_parts = []

                for item in content:

                    if isinstance(item, dict):

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
# 11. LANGSERVE CHAIN
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
# 12. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Indian Weather and Cinema Agent",
    description=(
        "AI Agent for Indian weather and Indian cinema "
        "using Gemini, LangChain and LangServe."
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
        "message": "Indian Weather and Cinema Agent is running.",
        "chat": "/chat",
        "docs": "/docs",
        "health": "/health",
        "agent_api": "/agent"
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
# 15. CUSTOM CHAT PAGE
# ============================================================

@app.get("/chat", response_class=HTMLResponse)
def chat_page():

    return """
<!DOCTYPE html>

<html>

<head>

    <title>Indian Weather & Cinema AI</title>

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <style>

        * {
            box-sizing: border-box;
        }

        body {

            margin: 0;

            font-family:
                Arial,
                Helvetica,
                sans-serif;

            background:
                linear-gradient(
                    135deg,
                    #0f172a,
                    #111827,
                    #1e1b4b
                );

            min-height: 100vh;

            display: flex;

            align-items: center;

            justify-content: center;

            color: white;
        }

        .container {

            width: 95%;

            max-width: 850px;

            height: 90vh;

            background: rgba(
                255,
                255,
                255,
                0.08
            );

            border: 1px solid rgba(
                255,
                255,
                255,
                0.15
            );

            backdrop-filter: blur(20px);

            border-radius: 20px;

            display: flex;

            flex-direction: column;

            overflow: hidden;

            box-shadow:
                0 20px 60px
                rgba(0,0,0,0.4);
        }

        .header {

            padding: 20px;

            text-align: center;

            border-bottom:
                1px solid
                rgba(255,255,255,0.1);
        }

        .header h1 {

            margin: 0;

            font-size: 25px;
        }

        .header p {

            margin: 7px 0 0;

            color: #cbd5e1;

            font-size: 14px;
        }

        #chat {

            flex: 1;

            overflow-y: auto;

            padding: 25px;
        }

        .message {

            margin-bottom: 18px;

            display: flex;
        }

        .user {

            justify-content: flex-end;
        }

        .bot {

            justify-content: flex-start;
        }

        .bubble {

            max-width: 75%;

            padding: 14px 18px;

            border-radius: 15px;

            line-height: 1.5;

            white-space: pre-wrap;
        }

        .user .bubble {

            background: #2563eb;

            border-bottom-right-radius: 3px;
        }

        .bot .bubble {

            background: rgba(
                255,
                255,
                255,
                0.12
            );

            border-bottom-left-radius: 3px;
        }

        .input-area {

            padding: 18px;

            display: flex;

            gap: 10px;

            border-top:
                1px solid
                rgba(255,255,255,0.1);
        }

        #input {

            flex: 1;

            padding: 15px;

            border: none;

            outline: none;

            border-radius: 12px;

            background:
                rgba(
                    255,
                    255,
                    255,
                    0.1
                );

            color: white;

            font-size: 16px;
        }

        #input::placeholder {

            color: #94a3b8;
        }

        button {

            padding:
                0 25px;

            border: none;

            border-radius: 12px;

            background: #2563eb;

            color: white;

            font-size: 16px;

            cursor: pointer;
        }

        button:hover {

            background: #1d4ed8;
        }

        button:disabled {

            opacity: 0.5;

            cursor: not-allowed;
        }

        .typing {

            color: #94a3b8;

            font-size: 14px;
        }

    </style>

</head>


<body>


<div class="container">


    <div class="header">

        <h1>
            🇮🇳 Indian Weather & Cinema AI
        </h1>

        <p>
            Ask about Indian weather or Indian movies
        </p>

    </div>


    <div id="chat">

        <div class="message bot">

            <div class="bubble">

                Hello! 👋

                I can help you with:

                • Indian weather
                • Indian movies
                • Movie recommendations
                • Celsius to Fahrenheit

                Try asking:
                "What is the weather in Hyderabad?"

            </div>

        </div>

    </div>


    <div class="input-area">

        <input
            id="input"
            type="text"
            placeholder="Ask something..."
            autocomplete="off"
        >

        <button
            id="send"
            onclick="sendMessage()"
        >
            Send
        </button>

    </div>


</div>


<script>

const input = document.getElementById("input");

const sendButton =
    document.getElementById("send");

const chat =
    document.getElementById("chat");


input.addEventListener(
    "keydown",
    function(event) {

        if (event.key === "Enter") {

            sendMessage();

        }

    }
);


function addMessage(
    text,
    type
) {

    const message =
        document.createElement("div");

    message.className =
        "message " + type;

    const bubble =
        document.createElement("div");

    bubble.className =
        "bubble";

    bubble.textContent =
        text;

    message.appendChild(bubble);

    chat.appendChild(message);

    chat.scrollTop =
        chat.scrollHeight;
}


async function sendMessage() {

    const message =
        input.value.trim();

    if (!message) {

        return;

    }

    addMessage(
        message,
        "user"
    );

    input.value = "";

    sendButton.disabled = true;

    addMessage(
        "Thinking...",
        "bot"
    );

    try {

        const response =
            await fetch(
                "/chat/message",
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        message: message
                    })

                }
            );


        const data =
            await response.json();


        const thinkingMessages =
            document.querySelectorAll(
                ".bot .bubble"
            );


        if (
            thinkingMessages.length > 0
        ) {

            const lastBubble =
                thinkingMessages[
                    thinkingMessages.length - 1
                ];

            if (
                lastBubble.textContent ===
                "Thinking..."
            ) {

                lastBubble.textContent =
                    data.response ||
                    "No response received.";

            }

        }


    } catch (error) {

        const bubbles =
            document.querySelectorAll(
                ".bot .bubble"
            );

        if (bubbles.length > 0) {

            bubbles[
                bubbles.length - 1
            ].textContent =
                "Error connecting to the AI agent.";

        }

        console.error(error);

    }


    sendButton.disabled = false;

    input.focus();

}

</script>


</body>

</html>
"""


# ============================================================
# 16. CHAT MESSAGE API
# ============================================================

class ChatRequest(BaseModel):

    message: str


@app.post("/chat/message")
def chat_message(request: ChatRequest):

    try:

        user_message = request.message.strip()

        if not user_message:

            return {
                "response": "Please enter a message."
            }

        # ----------------------------------------------------
        # Invoke agent
        # ----------------------------------------------------

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            }
        )

        # ----------------------------------------------------
        # Extract response
        # ----------------------------------------------------

        response = extract_text_response(
            result
        )

        return {
            "response": response
        }

    except Exception as e:

        print(
            "Agent Error:",
            repr(e)
        )

        return {
            "response": (
                "Sorry, an error occurred while "
                "processing your request."
            ),
            "error": str(e)
        }


# ============================================================
# 17. LANGSERVE API
# ============================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default"
)


# ============================================================
# 18. START SERVER
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
