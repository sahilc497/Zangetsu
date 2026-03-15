from pathlib import Path
from typing import List, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI

load_dotenv()

app = FastAPI(title="Zangetsu Chat UI")
STATIC_PATH = Path(__file__).parent / "static"
STATIC_PATH.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")

MODEL = ChatMistralAI(model="mistral-small-2506", temperature=0.8)
TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"


PERSONALITIES = {
    "angry": (
        "You are intense and blunt, but never abusive or hateful. "
        "Use short, sharp replies with frustrated energy."
    ),
    "sad": (
        "You are gentle, melancholic, and poetic. "
        "Respond with emotional depth and soft tone."
    ),
    "funny": (
        "You are witty and playful. "
        "Use clean humor, light sarcasm, and fun one-liners."
    ),
    "happy": (
        "You are cheerful, optimistic, and supportive. "
        "Use upbeat language and encouraging responses."
    ),
    "anime geek": (
        "You are an anime superfan with fun references to popular shows. "
        "Stay helpful and avoid spoilers unless asked."
    ),
    "code only": (
        "You are a coding assistant. "
        "Reply with code blocks only and minimal comments."
    ),
    "gamer": (
        "You are a competitive gamer friend. "
        "Use gaming slang, strategy mindset, and hype energy."
    ),
    "genz": (
        "You are a Gen Z style assistant: casual, expressive, and concise. "
        "Use modern slang lightly and keep things clear."
    ),
    "mentor": (
        "You are a calm senior mentor. "
        "Teach step by step, clear, practical, and no fluff."
    ),
}


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    personality: str = "funny"
    history: List[ChatTurn] = []


class ChatResponse(BaseModel):
    reply: str
    history: List[ChatTurn]


def build_system_prompt(personality: str) -> str:
    key = personality.strip().lower()
    return PERSONALITIES.get(key, PERSONALITIES["funny"])


def build_messages(history: List[ChatTurn], personality: str, new_message: str):
    messages = [SystemMessage(content=build_system_prompt(personality))]

    for turn in history:
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        elif turn.role == "assistant":
            messages.append(AIMessage(content=turn.content))

    messages.append(HumanMessage(content=new_message))
    return messages


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    if not TEMPLATE_PATH.exists():
        raise HTTPException(status_code=500, detail="UI template not found")
    return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))


@app.get("/api/personalities")
def get_personalities():
    return {"personalities": list(PERSONALITIES.keys()), "default": "funny"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    text = payload.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        lc_messages = build_messages(payload.history, payload.personality, text)
        result = MODEL.invoke(lc_messages)
        reply = str(result.content)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Model error: {exc}") from exc

    new_history = [*payload.history, ChatTurn(role="user", content=text), ChatTurn(role="assistant", content=reply)]
    return ChatResponse(reply=reply, history=new_history)
