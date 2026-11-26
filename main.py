import os, uuid, shutil, subprocess, json, re, threading, time
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from processor import process_video, capture_live

BASE = Path(os.getenv("LIVECLIPPER_DATA", "./data")).resolve()
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "clips"
for p in (UPLOADS, OUTPUTS): p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LiveClipper API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
jobs = {}

@app.get("/api/health")
def health():
    return {"ok": True, "service": "LiveClipper"}

@app.post("/api/process")
async def process(file: UploadFile = File(None), youtube_url: str = Form(""),
                  category: str = Form("both"), duration: int = Form(30),
                  aspect: str = Form("9:16"), subtitles: bool = Form(True)):
    if not file and not youtube_url.strip():
        raise HTTPException(400, "Choose a video file or enter a YouTube URL.")
    if duration not in [15,20,30,45,60,90,120]:
        raise HTTPException(400, "Unsupported clip duration.")
    if category not in ["funny","important","both"]:
        raise HTTPException(400, "Invalid highlight category.")
    job_id = uuid.uuid4().hex[:12]
    source = UPLOADS / f"{job_id}.mp4"
    if file:
        ext = Path(file.filename or "video.mp4").suffix or ".mp4"
        source = UPLOADS / f"{job_id}{ext}"
        with source.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    else:
        cmd = ["yt-dlp", "--no-playlist", "-f", "bv*+ba/b", "--merge-output-format", "mp4", "-o", str(source), youtube_url]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0 or not source.exists():
            raise HTTPException(400, "Could not fetch video. Check URL and permissions. " + (r.stderr[-500:] if r.stderr else ""))
    jobs[job_id] = {"status":"processing", "clips":[], "error":None}
    def worker():
        try:
            jobs[job_id]["clips"] = process_video(source, OUTPUTS, job_id, category, duration, aspect, subtitles)
            jobs[job_id]["status"] = "done"
        except Exception as e:
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["status"] = "error"
    threading.Thread(target=worker, daemon=True).start()
    return {"job_id":job_id, "status":"processing"}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs: raise HTTPException(404, "Job not found")
    return jobs[job_id]

@app.get("/clips/{filename}")
def get_clip(filename: str):
    path = (OUTPUTS / filename).resolve()
    if OUTPUTS not in path.parents or not path.exists(): raise HTTPException(404, "Clip not found")
    return FileResponse(path, media_type="video/mp4", filename=path.name)

@app.post("/api/live")
def live(url: str = Form(...), segment_minutes: int = Form(5)):
    if not url.startswith("https://"): raise HTTPException(400, "Use a valid HTTPS stream URL.")
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status":"recording", "clips":[], "error":None}
    def worker():
        try:
            capture_live(url, OUTPUTS, job_id, segment_minutes, jobs)
        except Exception as e:
            jobs[job_id]["error"] = str(e); jobs[job_id]["status"] = "error"
    threading.Thread(target=worker, daemon=True).start()
    return {"job_id":job_id, "status":"recording"}
