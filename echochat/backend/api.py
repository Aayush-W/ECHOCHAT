import json
from functools import lru_cache
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import (
    MEMORY_DATA_PATH,
    PERSONALITY_PROFILE_PATH,
    LLM_MODEL_NAME,
    FLASK_HOST,
    FLASK_PORT,
    BASE_DIR,
)
from .memory_store import MemoryStore
from .responder import Responder


app = FastAPI(title="EchoChat API")
UI_DIR = BASE_DIR / "ui"

if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    include_memories: bool = True


class ChatResponse(BaseModel):
    response: str
    memories_used: List[str]
    model: str
    success: bool


def _load_personality_profile() -> dict:
    if PERSONALITY_PROFILE_PATH.exists():
        try:
            with open(PERSONALITY_PROFILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


@lru_cache(maxsize=1)
def _get_responder() -> Responder:
    profile = _load_personality_profile()
    memory_store = MemoryStore(memory_data_path=str(MEMORY_DATA_PATH))
    return Responder(personality_profile=profile, memory_store=memory_store)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": LLM_MODEL_NAME}


@app.get("/")
def ui_root():
    index_path = UI_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="UI not found.")


@app.get("/ui")
def ui_page():
    return ui_root()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message must not be empty.")

    responder = _get_responder()
    result = responder.generate_response(
        message,
        include_memories=req.include_memories,
        verbose=False,
    )

    return ChatResponse(
        response=result.get("response", ""),
        memories_used=result.get("memories_used", []),
        model=result.get("model", ""),
        success=bool(result.get("success", False)),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "echochat.backend.api:app",
        host=FLASK_HOST,
        port=FLASK_PORT,
        reload=False,
    )
