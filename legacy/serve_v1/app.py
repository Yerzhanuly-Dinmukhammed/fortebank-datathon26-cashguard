"""Cash-discipline violation service (API per the task brief, section 10, + upload page).

POST /api/v1/analyze      multipart field `file` (.mov), header X-API-Key -> {"job_id", "status"}
GET  /api/v1/jobs/{id}    header X-API-Key -> {"job_id", "status": processing|done|failed, "verdict", "violations"}
GET  /                    upload page (asks for the API key once, keeps it in the browser)

Jobs are processed one by one by a background worker (one GPU) and stored as JSON on disk,
so results survive a restart of the service.

Env: API_KEY (required), WORK (/workspace), OBJ_WEIGHTS, VLM_MODEL_ID ("" disables the VLM), DEVICE.
"""
import json
import os
import queue
import secrets
import shutil
import threading
import time
import traceback
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

WORK = Path(os.getenv("WORK", "/workspace"))
API_KEY = os.getenv("API_KEY", "")
JOBS_DIR = WORK / "service" / "jobs"
UPLOADS = WORK / "service" / "uploads"
for d in (JOBS_DIR, UPLOADS):
    d.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 300))

app = FastAPI(title="Cash discipline video check", version="1.0")
jobs_q: "queue.Queue[str]" = queue.Queue()
_analyzer = None
_state = {"ready": False, "error": None, "current": None}


# ---------------------------------------------------------------- storage
def job_path(job_id):
    return JOBS_DIR / f"{job_id}.json"


def save_job(job):
    tmp = job_path(job["job_id"]).with_suffix(".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False))
    tmp.replace(job_path(job["job_id"]))


def load_job(job_id):
    p = job_path(job_id)
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------- worker
def make_analyzer():
    from src.pipeline import Analyzer
    return Analyzer(
        person_weights=str(WORK / "yolo11m.pt"),
        pose_weights=str(WORK / "yolo11m-pose.pt"),
        object_weights=os.getenv("OBJ_WEIGHTS", str(WORK / "models" / "cash_yolo11m_auto.pt")),
        device=os.getenv("DEVICE", "0"),
        vlm_model_id=os.getenv("VLM_MODEL_ID", "Qwen/Qwen3.5-9B") or None,
    )


def worker():
    global _analyzer
    try:
        _analyzer = make_analyzer()
        _state["ready"] = True
    except Exception as e:  # keep the API alive and report the problem
        _state["error"] = f"model load failed: {e}"
        traceback.print_exc()
        return
    while True:
        job_id = jobs_q.get()
        job = load_job(job_id)
        if job is None:
            continue
        _state["current"] = job_id
        t0 = time.time()
        try:
            res = _analyzer.analyze(job["video_path"], job_id=job_id)
            job.update(status="done", verdict=res["verdict"], violations=res["violations"],
                       processing_sec=round(time.time() - t0, 1))
        except Exception as e:
            traceback.print_exc()
            job.update(status="failed", error=str(e)[:500], verdict=None, violations=[])
        job["finished_at"] = time.time()
        save_job(job)
        _state["current"] = None


@app.on_event("startup")
def startup():
    if not API_KEY:
        raise RuntimeError("set API_KEY env variable")
    # re-queue jobs that were processing when the service stopped
    for p in sorted(JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime):
        job = json.loads(p.read_text())
        if job.get("status") == "processing":
            jobs_q.put(job["job_id"])
    threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------- API
def check_key(x_api_key: str = Header(default="")):
    if not API_KEY or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.post("/api/v1/analyze", dependencies=[Depends(check_key)])
async def analyze(file: UploadFile = File(...)):
    name = (file.filename or "video.mov").lower()
    if not name.endswith((".mov", ".mp4", ".m4v", ".avi", ".mkv")):
        raise HTTPException(status_code=400, detail="expected a video file (.mov)")
    job_id = secrets.token_hex(4)
    dst = UPLOADS / f"{job_id}{Path(name).suffix}"
    size = 0
    with open(dst, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                f.close(); dst.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"file larger than {MAX_UPLOAD_MB} MB")
            f.write(chunk)
    job = {"job_id": job_id, "status": "processing", "verdict": None, "violations": [],
           "filename": file.filename, "video_path": str(dst), "created_at": time.time()}
    save_job(job)
    jobs_q.put(job_id)
    return {"job_id": job_id, "status": "processing"}


def public(job):
    out = {k: job.get(k) for k in ("job_id", "status", "verdict", "violations")}
    if job.get("status") == "failed":
        out["error"] = job.get("error")
    return out


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(check_key)])
def get_job(job_id: str):
    job = load_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse(public(job))


@app.get("/api/v1/health")
def health():
    return {"ready": _state["ready"], "error": _state["error"], "queue": jobs_q.qsize(), "current": _state["current"]}


@app.get("/", response_class=HTMLResponse)
def page():
    return (Path(__file__).with_name("index.html")).read_text(encoding="utf-8")
