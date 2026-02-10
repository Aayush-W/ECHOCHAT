import hashlib
import json
import os
import subprocess
import sys
import uuid
from collections import Counter, OrderedDict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Body, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import (
    MEMORY_DATA_PATH,
    PERSONALITY_PROFILE_PATH,
    TRAINING_DATA_PATH,
    LLM_MODEL_NAME,
    HF_LLAMA31_8B_INSTRUCT,
    OLLAMA_LLAMA31_8B,
    FLASK_HOST,
    FLASK_PORT,
    BASE_DIR,
    SESSIONS_DIR,
)
from .chat_parser import parse_whatsapp_chat
from .dataset_builder import build_datasets, save_datasets
from .personality_analyzer import PersonalityAnalyzer
from .memory_store import MemoryStore
from .example_store import ExampleStore
from .responder import Responder


app = FastAPI(title="EchoChat API")
UI_DIR = BASE_DIR / "ui"

if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

_RESPONDER_CACHE: "OrderedDict[str, Responder]" = OrderedDict()
_RESPONDER_CACHE_LIMIT = 4
_TRAINING_JOBS: Dict[str, Dict] = {}
_TRAINING_LOG_DIR = BASE_DIR / "data" / "training_logs"
_TRAINING_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _parse_ollama_size(size_str: str) -> float:
    parts = size_str.strip().split()
    if len(parts) < 2:
        return 0.0
    value_str, unit = parts[0], parts[1].lower()
    try:
        value = float(value_str)
    except ValueError:
        return 0.0
    if unit.startswith("kb"):
        return value / (1024 ** 2)
    if unit.startswith("mb"):
        return value / 1024
    if unit.startswith("gb"):
        return value
    if unit.startswith("tb"):
        return value * 1024
    return 0.0


def _pick_best_ollama_model() -> str:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return LLM_MODEL_NAME

    if result.returncode != 0 or not result.stdout:
        return LLM_MODEL_NAME

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) <= 1:
        return LLM_MODEL_NAME

    best_name = None
    best_size = 0.0
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        name = parts[0]
        size_str = parts[2] if len(parts) >= 3 else ""
        size_gb = _parse_ollama_size(size_str)
        if size_gb > best_size:
            best_size = size_gb
            best_name = name

    return best_name or LLM_MODEL_NAME


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _get_session_dir(session_id: str) -> Path:
    return SESSIONS_DIR / session_id


def _session_paths(session_dir: Path) -> Dict[str, Path]:
    return {
        "chat": session_dir / "chat.txt",
        "messages": session_dir / "messages.json",
        "meta": session_dir / "meta.json",
        "training": session_dir / "training_data.jsonl",
        "memory": session_dir / "memory_data.json",
        "profile": session_dir / "personality_profile.json",
    }


def _serialize_messages(messages: List[Dict]) -> List[Dict]:
    payload = []
    for msg in messages:
        payload.append(
            {
                **msg,
                "timestamp": msg["timestamp"].isoformat(),
            }
        )
    return payload


def _deserialize_messages(payload: List[Dict]) -> List[Dict]:
    messages = []
    for msg in payload:
        ts = msg.get("timestamp")
        if isinstance(ts, str):
            try:
                msg = {**msg, "timestamp": datetime.fromisoformat(ts)}
            except ValueError:
                continue
        messages.append(msg)
    return messages


def _save_messages(session_dir: Path, messages: List[Dict]) -> None:
    paths = _session_paths(session_dir)
    with open(paths["messages"], "w", encoding="utf-8") as f:
        json.dump(_serialize_messages(messages), f, ensure_ascii=False, indent=2)


def _load_messages(session_dir: Path) -> List[Dict]:
    paths = _session_paths(session_dir)
    if not paths["messages"].exists():
        return []
    with open(paths["messages"], "r", encoding="utf-8") as f:
        payload = json.load(f)
    return _deserialize_messages(payload)


def _load_meta(session_dir: Path) -> Dict:
    paths = _session_paths(session_dir)
    if not paths["meta"].exists():
        return {}
    try:
        with open(paths["meta"], "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def _save_meta(session_dir: Path, meta: Dict) -> None:
    paths = _session_paths(session_dir)
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _sender_stats(messages: List[Dict]) -> List[Dict]:
    counts = Counter(msg["sender"] for msg in messages)
    return [
        {"name": name, "count": count}
        for name, count in counts.most_common()
    ]


def _pick_default_echo_person(senders: List[Dict]) -> Optional[str]:
    if not senders:
        return None
    return senders[0]["name"]


def _build_session_assets(
    session_dir: Path,
    messages: List[Dict],
    echo_person: str,
) -> Dict:
    paths = _session_paths(session_dir)
    cached_meta = _load_meta(session_dir)
    if (
        cached_meta.get("echo_person") == echo_person
        and paths["memory"].exists()
        and paths["profile"].exists()
    ):
        return cached_meta

    echo_messages = [msg for msg in messages if msg["sender"] == echo_person]
    if not echo_messages:
        raise HTTPException(
            status_code=400,
            detail=f"No messages found for '{echo_person}'.",
        )

    training_data, memory_data = build_datasets(messages, echo_person=echo_person)
    if not memory_data:
        raise HTTPException(
            status_code=400,
            detail="Unable to build memory data from this chat.",
        )

    save_datasets(
        training_data,
        memory_data,
        training_path=str(paths["training"]),
        memory_path=str(paths["memory"]),
    )

    analyzer = PersonalityAnalyzer(echo_messages)
    profile = analyzer.analyze()
    analyzer.save_profile(str(paths["profile"]))

    meta = {
        "echo_person": echo_person,
        "message_count": len(messages),
        "echo_message_count": len(echo_messages),
        "training_pairs": len(training_data),
        "memory_entries": len(memory_data),
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    _save_meta(session_dir, meta)
    return meta


def _invalidate_responder(session_id: str) -> None:
    if session_id in _RESPONDER_CACHE:
        _RESPONDER_CACHE.pop(session_id, None)


def _get_responder_for_session(session_id: str) -> Responder:
    if session_id in _RESPONDER_CACHE:
        _RESPONDER_CACHE.move_to_end(session_id)
        return _RESPONDER_CACHE[session_id]

    session_dir = _get_session_dir(session_id)
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found.")

    paths = _session_paths(session_dir)
    if not paths["profile"].exists() or not paths["memory"].exists():
        raise HTTPException(
            status_code=400,
            detail="Session is not ready. Upload a chat first.",
        )

    try:
        with open(paths["profile"], "r", encoding="utf-8") as f:
            profile = json.load(f)
    except json.JSONDecodeError:
        profile = {}

    meta = _load_meta(session_dir)
    model_name = meta.get("ollama_model") or LLM_MODEL_NAME
    local_adapter_path = meta.get("local_adapter_path")
    local_base_model = meta.get("local_base_model")

    memory_store = MemoryStore(memory_data_path=str(paths["memory"]))
    example_store = None
    if paths["training"].exists():
        example_store = ExampleStore(training_data_path=str(paths["training"]))

    responder = Responder(
        personality_profile=profile,
        memory_store=memory_store,
        example_store=example_store,
        model_name=model_name,
        local_adapter_path=local_adapter_path,
        local_base_model=local_base_model,
    )

    _RESPONDER_CACHE[session_id] = responder
    if len(_RESPONDER_CACHE) > _RESPONDER_CACHE_LIMIT:
        _RESPONDER_CACHE.popitem(last=False)
    return responder


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    include_memories: bool = True
    session_id: Optional[str] = None


class SenderInfo(BaseModel):
    name: str
    count: int


class UploadResponse(BaseModel):
    session_id: str
    echo_person: str
    senders: List[SenderInfo]
    message_count: int


class SetPersonRequest(BaseModel):
    echo_person: str = Field(..., min_length=1, max_length=200)


class ChatResponse(BaseModel):
    response: str
    memories_used: List[str]
    model: str
    success: bool


class TrainRequest(BaseModel):
    session_id: Optional[str] = None
    fast: bool = True
    max_steps: Optional[int] = Field(None, ge=1)
    sample_size: Optional[int] = Field(None, ge=1)
    epochs: Optional[float] = Field(None, gt=0)
    base_model: Optional[str] = None
    export_ollama: bool = True
    ollama_base_model: Optional[str] = None
    ollama_model_name: Optional[str] = None
    no_amp: Optional[bool] = None


class TrainResponse(BaseModel):
    job_id: str
    status: str
    output_dir: str
    log_path: Optional[str]
    ollama_model_name: Optional[str] = None


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str
    output_dir: str
    log_path: Optional[str]
    started_at: Optional[str]
    finished_at: Optional[str]
    error: Optional[str]
    ollama_model_name: Optional[str] = None


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
    example_store = None
    if TRAINING_DATA_PATH.exists():
        example_store = ExampleStore(training_data_path=str(TRAINING_DATA_PATH))
    return Responder(
        personality_profile=profile,
        memory_store=memory_store,
        example_store=example_store,
    )


def _run_training_job(
    job_id: str,
    cmd: List[str],
    log_path: Path,
    session_id: Optional[str],
    adapter_dir: Path,
    local_base_model: Optional[str],
    export_ollama: bool,
    ollama_base_model: str,
    ollama_model_name: str,
) -> None:
    _TRAINING_JOBS[job_id]["status"] = "running"
    _TRAINING_JOBS[job_id]["started_at"] = datetime.utcnow().isoformat() + "Z"
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(BASE_DIR),
            )
            return_code = proc.wait()

        if return_code != 0:
            _TRAINING_JOBS[job_id]["status"] = "failed"
            _TRAINING_JOBS[job_id]["error"] = f"Training exited with code {return_code}"
            _TRAINING_JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat() + "Z"
            return

        _TRAINING_JOBS[job_id]["status"] = "completed"

        if export_ollama:
            _TRAINING_JOBS[job_id]["status"] = "exporting"
            export_cmd = [
                sys.executable,
                str(BASE_DIR / "backend" / "export_ollama.py"),
                "--adapter-dir",
                str(adapter_dir),
                "--base-model",
                ollama_base_model,
                "--model-name",
                ollama_model_name,
            ]
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write("\n\n=== Exporting adapter to Ollama ===\n")
                export_proc = subprocess.Popen(
                    export_cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=str(BASE_DIR),
                )
                export_code = export_proc.wait()

            if export_code != 0:
                _TRAINING_JOBS[job_id]["status"] = "failed"
                _TRAINING_JOBS[job_id]["error"] = f"Ollama export exited with code {export_code}"
                _TRAINING_JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat() + "Z"
                return

            _TRAINING_JOBS[job_id]["status"] = "completed"

        if session_id:
            session_dir = _get_session_dir(session_id)
            if session_dir.exists():
                meta_path = adapter_dir / "training_meta.json"
                if meta_path.exists():
                    try:
                        training_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        local_base_model = training_meta.get("base_model") or local_base_model
                    except json.JSONDecodeError:
                        pass
                meta = _load_meta(session_dir)
                meta["local_adapter_path"] = str(adapter_dir)
                if local_base_model:
                    meta["local_base_model"] = local_base_model
                if export_ollama:
                    meta["ollama_model"] = ollama_model_name
                _save_meta(session_dir, meta)
                _invalidate_responder(session_id)

        _TRAINING_JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat() + "Z"
    except Exception as exc:
        _TRAINING_JOBS[job_id]["status"] = "failed"
        _TRAINING_JOBS[job_id]["error"] = str(exc)
        _TRAINING_JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat() + "Z"


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


@app.post("/upload", response_model=UploadResponse)
async def upload_chat(file: UploadFile = File(...)) -> UploadResponse:
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    session_id = _hash_bytes(content)
    session_dir = _get_session_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    paths = _session_paths(session_dir)
    if not paths["chat"].exists():
        with open(paths["chat"], "wb") as f:
            f.write(content)

    messages = _load_messages(session_dir)
    if not messages:
        messages = parse_whatsapp_chat(str(paths["chat"]))
        if not messages:
            raise HTTPException(
                status_code=400,
                detail="No messages parsed. Check the chat export format.",
            )
        _save_messages(session_dir, messages)

    senders = _sender_stats(messages)
    echo_person = _pick_default_echo_person(senders)
    if echo_person is None:
        raise HTTPException(status_code=400, detail="No valid senders found.")

    meta = _load_meta(session_dir)
    if meta.get("echo_person") and meta.get("echo_person") in {
        sender["name"] for sender in senders
    }:
        echo_person = meta["echo_person"]

    _build_session_assets(session_dir, messages, echo_person)
    _invalidate_responder(session_id)

    return UploadResponse(
        session_id=session_id,
        echo_person=echo_person,
        senders=senders,
        message_count=len(messages),
    )


@app.post("/session/{session_id}/set_person", response_model=UploadResponse)
def set_person(session_id: str, req: SetPersonRequest = Body(...)) -> UploadResponse:
    session_dir = _get_session_dir(session_id)
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found.")

    messages = _load_messages(session_dir)
    if not messages:
        raise HTTPException(
            status_code=400,
            detail="Session has no messages. Upload a chat first.",
        )

    senders = _sender_stats(messages)
    sender_names = {sender["name"] for sender in senders}
    if req.echo_person not in sender_names:
        raise HTTPException(
            status_code=400,
            detail="Selected person not found in this chat.",
        )

    _build_session_assets(session_dir, messages, req.echo_person)
    _invalidate_responder(session_id)

    return UploadResponse(
        session_id=session_id,
        echo_person=req.echo_person,
        senders=senders,
        message_count=len(messages),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message must not be empty.")

    responder = _get_responder_for_session(req.session_id) if req.session_id else _get_responder()
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


@app.post("/train/fast", response_model=TrainResponse)
def train_fast(req: TrainRequest, background_tasks: BackgroundTasks) -> TrainResponse:
    session_id = req.session_id
    if session_id:
        session_dir = _get_session_dir(session_id)
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="Session not found.")
        data_path = _session_paths(session_dir)["training"]
        if not data_path.exists():
            raise HTTPException(
                status_code=400,
                detail="Training data not found. Upload a chat first.",
            )
        output_dir = BASE_DIR / "data" / "models" / "echobot-lora" / session_id
    else:
        data_path = TRAINING_DATA_PATH
        output_dir = BASE_DIR / "data" / "models" / "echobot-lora" / "default"
        if not data_path.exists():
            raise HTTPException(
                status_code=400,
                detail="Training data not found. Upload a chat first.",
            )

    job_id = uuid.uuid4().hex[:12]
    log_path = _TRAINING_LOG_DIR / f"train_{job_id}.log"
    ollama_base_model = req.ollama_base_model or LLM_MODEL_NAME
    if req.ollama_model_name:
        ollama_model_name = req.ollama_model_name
    else:
        suffix = session_id if session_id else "default"
        ollama_model_name = f"echochat-{suffix}"

    base_model = req.base_model or HF_LLAMA31_8B_INSTRUCT
    ollama_base_model = req.ollama_base_model or OLLAMA_LLAMA31_8B

    cmd = [
        sys.executable,
        str(BASE_DIR / "backend" / "train_qlora.py"),
        "--data-path",
        str(data_path),
        "--output-dir",
        str(output_dir),
    ]
    if req.fast:
        cmd.append("--fast")
    if req.max_steps:
        cmd += ["--max-steps", str(req.max_steps)]
    if req.sample_size:
        cmd += ["--sample-size", str(req.sample_size)]
    if req.epochs:
        cmd += ["--epochs", str(req.epochs)]
    if base_model:
        cmd += ["--base-model", base_model]
    no_amp = req.no_amp
    if no_amp is None and os.name == "nt":
        no_amp = True
    if no_amp:
        cmd.append("--no-amp")

    _TRAINING_JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "ollama_model_name": ollama_model_name if req.export_ollama else None,
        "started_at": None,
        "finished_at": None,
        "error": None,
    }

    background_tasks.add_task(
        _run_training_job,
        job_id,
        cmd,
        log_path,
        session_id,
        output_dir,
        base_model,
        req.export_ollama,
        ollama_base_model,
        ollama_model_name,
    )

    return TrainResponse(
        job_id=job_id,
        status="queued",
        output_dir=str(output_dir),
        log_path=str(log_path),
        ollama_model_name=ollama_model_name if req.export_ollama else None,
    )


@app.get("/train/status/{job_id}", response_model=TrainStatusResponse)
def train_status(job_id: str) -> TrainStatusResponse:
    if job_id not in _TRAINING_JOBS:
        raise HTTPException(status_code=404, detail="Training job not found.")
    payload = _TRAINING_JOBS[job_id]
    return TrainStatusResponse(**payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "echochat.backend.api:app",
        host=FLASK_HOST,
        port=FLASK_PORT,
        reload=False,
    )
