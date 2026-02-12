from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
import requests
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai.chat_models import ChatOpenAI

def build_model() -> ChatOpenAI:
    """Create and configure the chat model from environment settings."""
    model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
    return ChatOpenAI(temperature=temperature, model=model_name, max_retries=2)


def build_professional_chatbot(model: ChatOpenAI) -> RunnableWithMessageHistory:
    """Build a conversational chain with memory and a professional system prompt.

    This wires the prompt, model, and output parser into a Runnable that stores
    per-session chat history, enabling multi-turn conversations.
    """
    system_prompt = (
        "You are a professional, concise assistant. "
        "Ask clarifying questions when needed and be explicit about assumptions."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | model | StrOutputParser()  # Compose prompt, model, and output parsing.

    store: Dict[str, InMemoryChatMessageHistory] = {}  # In-memory per-session history store.

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        """Retrieve or create chat history for a session."""
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def chat_once(chatbot: RunnableWithMessageHistory, user_input: str, session_id: str) -> str:
    """Process a single user input and return the assistant reply."""
    if is_datetime_question(user_input):
        return get_current_datetime_response()

    if is_weather_question(user_input):
        location = extract_location(user_input)
        if not location:
            return "Please provide a location for the weather (e.g., 'weather in Toronto')."
        return fetch_open_meteo_weather(location)

    if is_news_question(user_input):
        query = extract_news_query(user_input)
        return fetch_gnews_headlines(query)

    if is_web_search_question(user_input):
        return answer_with_web_rag(chatbot, user_input, session_id)

    return chatbot.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )


def is_datetime_question(user_input: str) -> bool:
    """Detect whether the user is asking for the current date or time."""
    normalized = user_input.strip().lower()
    if not normalized:
        return False
    patterns = [
        r"\bwhat( is|'s)? the date( today)?\b",
        r"\bwhat( is|'s)? today('s)? date\b",
        r"\btoday('s)? date\b",
        r"\bwhat( is|'s)? the time( now| today)?\b",
        r"\bcurrent time\b",
        r"\btime now\b",
        r"\bcurrent date\b",
    ]
    return any(re.search(pattern, normalized) for pattern in patterns)


def is_weather_question(user_input: str) -> bool:
    """Detect whether the user is asking about weather information."""
    normalized = user_input.strip().lower()
    return bool(re.search(r"\b(weather|temperature|forecast)\b", normalized))


def is_news_question(user_input: str) -> bool:
    """Detect whether the user is asking for news or headlines."""
    normalized = user_input.strip().lower()
    return bool(re.search(r"\b(news|headline|headlines|breaking)\b", normalized))


def is_web_search_question(user_input: str) -> bool:
    """Detect whether the user is asking to search the web."""
    normalized = user_input.strip().lower()
    return bool(re.search(r"\b(search|web|internet|online|lookup|find)\b", normalized))


def get_current_datetime_response() -> str:
    """Return the current local date and time as a formatted string."""
    now = datetime.now()
    return f"Today's date is {now.strftime('%B %d, %Y')}. The current time is {now.strftime('%I:%M %p')}."


def extract_location(user_input: str) -> Optional[str]:
    """Extract a location string from the user's input if present."""
    match = re.search(r"\b(?:in|for)\s+([a-zA-Z\s,.-]+)", user_input, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_news_query(user_input: str) -> str:
    """Derive a concise news query from the user's input."""
    normalized_input = user_input.strip().lower()
    if re.search(r"\b(top|headline|headlines)\b", normalized_input, flags=re.IGNORECASE):
        remaining = re.sub(
            r"\b(what|is|the|top|headline|headlines|news|breaking|latest|today|in|of|on|for|a|an|world)\b",
            "",
            normalized_input,
            flags=re.IGNORECASE,
        )
        remaining = re.sub(r"\s+", " ", remaining).strip()
        if not remaining:
            return "top headlines"
    normalized = re.sub(r"\b(news|headline|headlines|breaking|latest|top)\b", "", user_input, flags=re.IGNORECASE)
    normalized = re.sub(r"[^a-zA-Z0-9\s]", " ", normalized)
    cleaned = re.sub(r"\s+", " ", normalized).strip()
    return cleaned or "world"


def fetch_open_meteo_weather(location: str) -> str:
    """Fetch current weather for a location using Open-Meteo (no API key)."""
    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        results = geo_data.get("results") or []
        if not results:
            return f"I couldn't find a location named '{location}'."
        place = results[0]
        latitude = place.get("latitude")
        longitude = place.get("longitude")
        place_name = place.get("name")
        country = place.get("country")

        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": latitude, "longitude": longitude, "current_weather": True},
            timeout=10,
        )
        weather_resp.raise_for_status()
        weather = weather_resp.json().get("current_weather", {})
        temperature = weather.get("temperature")
        windspeed = weather.get("windspeed")
        weathercode = weather.get("weathercode")
        description = weather_code_description(weathercode)

        return (
            f"Weather in {place_name}, {country}: {description}, "
            f"{temperature}°C with wind {windspeed} km/h."
        )
    except requests.RequestException:
        return "I couldn't reach the weather service right now. Please try again later."


def weather_code_description(code: Optional[int]) -> str:
    """Convert Open-Meteo weather codes to a short description."""
    descriptions = {
        0: "clear sky",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "fog",
        48: "depositing rime fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        56: "light freezing drizzle",
        57: "dense freezing drizzle",
        61: "slight rain",
        63: "moderate rain",
        65: "heavy rain",
        66: "light freezing rain",
        67: "heavy freezing rain",
        71: "slight snow fall",
        73: "moderate snow fall",
        75: "heavy snow fall",
        77: "snow grains",
        80: "slight rain showers",
        81: "moderate rain showers",
        82: "violent rain showers",
        85: "slight snow showers",
        86: "heavy snow showers",
        95: "thunderstorm",
        96: "thunderstorm with slight hail",
        99: "thunderstorm with heavy hail",
    }
    return descriptions.get(code, "unknown conditions")


def fetch_gnews_headlines(query: str, max_results: int = 5) -> str:
    """Fetch recent news headlines from GNews (requires API key)."""
    gnews_api_key = os.getenv("GNEWS_API_KEY")
    if not gnews_api_key:
        return "GNews API key is missing. Set GNEWS_API_KEY in your .env file."

    try:
        normalized_query = re.sub(r"\s+", " ", query.strip().lower())
        needs_freshness_note = bool(re.search(r"\b(today|latest|breaking|now|current)\b", normalized_query))
        is_top = bool(re.search(r"\b(top|headline|headlines)\b", normalized_query))
        if is_top:
            response = requests.get(
                "https://gnews.io/api/v4/top-headlines",
                params={"topic": "world", "lang": "en", "max": max_results, "token": gnews_api_key},
                timeout=10,
            )
        else:
            response = requests.get(
                "https://gnews.io/api/v4/search",
                params={"q": query, "lang": "en", "max": max_results, "token": gnews_api_key},
                timeout=10,
            )
        if not response.ok:
            try:
                error_payload = response.json()
            except ValueError:
                error_payload = {}
            error_message = error_payload.get("errors") or error_payload.get("message") or "Unknown error"
            return (
                f"GNews error (HTTP {response.status_code}): {error_message}. "
                "Please verify your API key, quota, and plan limits."
            )

        data = response.json()
        articles = data.get("articles") or []
        if not articles:
            return "I couldn't find any news for that query."
        formatted_items = []
        for idx, article in enumerate(articles, start=1):
            title = article.get("title", "Untitled")
            url = article.get("url", "")
            summary = article.get("description") or article.get("content") or "No summary available."
            formatted_items.append(
                "\n".join(
                    [
                        f"Title: {title}",
                        f"Description: {summary}",
                        f"Link: {url}",
                    ]
                )
            )
        note = "\n\n---\n\nNote: GNews free tier can be delayed by up to ~12 hours." if needs_freshness_note else ""
        return "NEWS_RESULTS\n" + "\n\n---\n\n".join(formatted_items) + note
    except requests.RequestException:
        return "I couldn't reach the news service right now. Please try again later."


def tavily_search(query: str, max_results: int = 5) -> Tuple[List[dict], Optional[str]]:
    """Search the web via Tavily and return results plus an optional error message."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return [], "Tavily API key is missing. Set TAVILY_API_KEY in your .env file."

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": False,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("results", []), None
    except requests.RequestException:
        return [], "I couldn't reach the web search service right now. Please try again later."


def answer_with_web_rag(
    chatbot: RunnableWithMessageHistory,
    query: str,
    session_id: str,
) -> str:
    """Use web search results as context to answer a query."""
    results, error = tavily_search(query)
    if error:
        return error

    if not results:
        return "I couldn't find any relevant sources on the web."

    sources = []
    for idx, item in enumerate(results, start=1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        snippet = item.get("content", "")
        sources.append(f"Source {idx}: {title}\nURL: {url}\nSnippet: {snippet}")

    rag_prompt = (
        "Use the sources below to answer the question. "
        "If the answer isn't in the sources, say you don't know.\n\n"
        f"Sources:\n{chr(10).join(sources)}\n\n"
        f"Question: {query}"
    )

    return chatbot.invoke(
        {"input": rag_prompt},
        config={"configurable": {"session_id": session_id}},
    )


if __name__ == "__main__":
    load_dotenv()

    chatbot = build_professional_chatbot(build_model())
    print(chat_once(chatbot, "Give me a short summary of LangChain.", "demo"))
