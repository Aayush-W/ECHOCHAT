from pathlib import Path
from typing import List, Dict, Optional
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import asyncio

DATA_DIR = Path(__file__).parent.parent / "data" / "weaviate_local"
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SimpleVectorServer")


class UpsertRequest(BaseModel):
    class_name: str
    vectors: List[List[float]]
    metadatas: List[Dict]


class QueryRequest(BaseModel):
    class_name: str
    vector: List[float]
    top_k: int = 5


class DeleteRequest(BaseModel):
    class_name: str
    ids: Optional[List[str]] = None


def _class_path(name: str) -> Path:
    return DATA_DIR / f"{name}.npz"


def _load_store(p: Path):
    if not p.exists():
        return np.empty((0, 0), dtype=float), []
    data = np.load(p, allow_pickle=True)
    vecs = data['vecs']
    metas = data['metas'].tolist()
    # normalize vecs to 2D numpy array
    try:
        arr = np.asarray(vecs)
        if arr.dtype == object:
            arr = np.vstack([np.asarray(v, dtype=float) for v in vecs]) if len(vecs) else np.empty((0, 0), dtype=float)
    except Exception:
        arr = np.empty((0, 0), dtype=float)
    return arr, metas


def _save_store(p: Path, vecs: np.ndarray, metas: List[Dict]):
    if vecs is None or vecs.size == 0:
        vecs_arr = np.empty((0, 0), dtype=float)
    else:
        vecs_arr = np.array(vecs, dtype=float)
    np.savez_compressed(p, vecs=vecs_arr, metas=np.array(metas, dtype=object))


@app.post("/upsert")
def upsert(req: UpsertRequest):
    if len(req.vectors) != len(req.metadatas):
        raise HTTPException(status_code=400, detail="vectors and metadatas length mismatch")

    p = _class_path(req.class_name)
    vecs, metas = _load_store(p)

    new_vecs = [np.array(v, dtype=float) for v in req.vectors]
    new_metas = []
    for m in req.metadatas:
        if not isinstance(m, dict):
            m = {"text": str(m)}
        if 'id' not in m:
            m['id'] = str(uuid.uuid4())
        new_metas.append(m)

    if vecs.size == 0:
        combined_vecs = np.vstack(new_vecs) if new_vecs else np.empty((0, 0), dtype=float)
    else:
        combined_vecs = np.vstack([vecs] + [np.vstack(new_vecs)]) if new_vecs else vecs

    metas = metas + new_metas
    _save_store(p, combined_vecs, metas)
    return {"success": True, "count": len(metas)}


@app.post("/query")
def query(req: QueryRequest):
    p = _class_path(req.class_name)
    if not p.exists():
        return {"results": []}

    arr, metas = _load_store(p)
    if arr.size == 0:
        return {"results": []}

    q = np.array(req.vector, dtype=float)
    norms = np.linalg.norm(arr, axis=1) * np.linalg.norm(q)
    norms = np.where(norms == 0, 1e-12, norms)
    sims = (arr @ q) / norms

    idxs = np.argsort(sims)[::-1][: req.top_k]
    out = []
    for i in idxs:
        out.append({"metadata": metas[int(i)], "score": float(sims[int(i)])})

    return {"results": out}


@app.post("/delete")
def delete(req: DeleteRequest):
    p = _class_path(req.class_name)
    if not p.exists():
        return {"deleted": 0}

    arr, metas = _load_store(p)
    if req.ids is None or len(req.ids) == 0:
        # delete all
        p.unlink(missing_ok=True)
        return {"deleted": len(metas)}

    keep_vecs = []
    keep_metas = []
    deleted = 0
    for v, m in zip(arr, metas):
        if m.get('id') in req.ids:
            deleted += 1
            continue
        keep_vecs.append(v)
        keep_metas.append(m)

    if keep_vecs:
        _save_store(p, np.vstack(keep_vecs), keep_metas)
    else:
        p.unlink(missing_ok=True)

    return {"deleted": deleted}


def _optimize_class(p: Path):
    arr, metas = _load_store(p)
    if arr.size == 0:
        return 0
    # deduplicate by id keeping last occurrence
    seen = {}
    for idx, m in enumerate(metas):
        seen[m.get('id')] = idx
    keep_idxs = sorted(seen.values())
    new_vecs = [arr[i] for i in keep_idxs]
    new_metas = [metas[i] for i in keep_idxs]
    _save_store(p, np.vstack(new_vecs) if new_vecs else np.empty((0, 0), dtype=float), new_metas)
    return len(new_metas)


@app.post("/optimize")
def optimize(class_name: Optional[str] = None):
    files = [f for f in DATA_DIR.glob("*.npz")]
    total = 0
    if class_name:
        p = _class_path(class_name)
        if p.exists():
            total = _optimize_class(p)
        return {"optimized": total}

    for p in files:
        total += _optimize_class(p)
    return {"optimized": total}


@app.get("/health")
def health():
    files = [f for f in DATA_DIR.glob("*.npz")]
    info = {}
    for p in files:
        arr, metas = _load_store(p)
        info[p.stem] = {"count": len(metas)}
    return {"ok": True, "classes": info}


async def _periodic_optimize(interval_seconds: int = 60 * 60 * 6):
    while True:
        try:
            for p in DATA_DIR.glob("*.npz"):
                _optimize_class(p)
        except Exception:
            pass
        await asyncio.sleep(interval_seconds)


@app.on_event("startup")
async def _startup_tasks():
    # start background optimization task
    asyncio.create_task(_periodic_optimize())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
