from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from chatbot_app import build_model, build_professional_chatbot, chat_once

app = FastAPI(title="LangChain Chatbot")
web_dir = (Path(__file__).parent / ".." / "frontend").resolve()

chatbot = build_professional_chatbot(build_model())


class ChatRequest(BaseModel):
    message: str
    session_id: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(web_dir / "index.html")


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    reply = chat_once(chatbot, request.message, request.session_id)
    return {"reply": reply}
