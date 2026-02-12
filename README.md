# Langchain chatbot with RAG

## Overview
A LangChain-powered chatbot with conversational memory, a FastAPI backend, and a lightweight web UI. Live data integrations include:
- Local date/time
- Weather (Open‑Meteo, no API key)
- News (GNews, API key required)
- Web search RAG (Tavily, API key required)

## Project Structure
- backend: FastAPI app and chatbot logic
- frontend: Static UI served by the backend
- Dockerfile: Container build
- requirement.txt: Python dependencies

## Prerequisites
- Docker (recommended)
- API keys for OpenAI, GNews, Tavily

## Environment Variables
Create a .env file in the project root:
- OPENAI_API_KEY=your_openai_key
- GNEWS_API_KEY=your_gnews_key
- TAVILY_API_KEY=your_tavily_key

Example:
```
OPENAI_API_KEY=sk-...
GNEWS_API_KEY=...
TAVILY_API_KEY=...
```

## Run with Docker
From the project root:
- docker build -t langchain-chatbot .
- docker run --rm -p 8000:8000 --env-file .env langchain-chatbot

Open: http://localhost:8000

## Run Locally (Python)
- Install dependencies: pip install -r requirement.txt
- Start the backend: uvicorn web_app:app --host 0.0.0.0 --port 8000

Note: Run the command from the backend folder, or adjust PYTHONPATH accordingly.

## HTTP Endpoints
- GET /: Serves the chat UI
- POST /chat: Accepts {"message": "...", "session_id": "..."} and returns {"reply": "..."}

## Usage Examples
- Date/time: "What is the date and time?"
- Weather: "Weather in Toronto"
- News: "Top news today in world"
- Web RAG: "Search the web for latest AI regulation updates"

## News Presentation Formatting
News results are formatted in the backend with labeled fields and separators so the frontend can render each article as a separate card (bold title, description, link).

Example backend response format:
- NEWS_RESULTS
- Title: ...
- Description: ...
- Link: ...
- ---

## Notes
- Weather uses Open‑Meteo and does not require an API key.
- News and web search require valid API keys and are subject to provider rate limits.
- If you see empty news results, verify your GNews key and try a simpler query.
