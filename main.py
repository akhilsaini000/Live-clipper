
import os
import uuid
import shutil
import subprocess
import threading
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from processor import process_video, capture_live


# ---------------- CONFIGURATION ----------------

BASE = Path(os.getenv("LIVECLIPPER_DATA", "./data")).resolve()
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "clips"

for folder in (UPLOADS, OUTPUTS):
    folder.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="LiveClipper API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = {}


# ---------------- HEALTH CHECK ----------------

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "LiveClipper"
    }


# ---------------- VIDEO PROCESSING ----------------

@app.post("/api/process")
async def process(
    file: UploadFile = File(None),
    youtube_url: str = Form(""),
    category: str = Form("both"),
    duration: int = Form(30),
    aspect: str = Form("9:16"),
    subtitles: bool = Form(True)
):

    if not file and not youtube_url.strip():
        raise HTTPException(
            400,
            "Choose a video file or enter a YouTube URL."
        )

    if duration not in [15, 20, 30, 45, 60, 90, 120]:
        raise HTTPException(400, "Unsupported clip duration.")

    if category not in ["funny", "important", "both"]:
        raise HTTPException(400, "Invalid highlight category.")

    job_id = uuid.uuid4().hex[:12]
    source = UPLOADS / f"{job_id}.mp4"

    # -------- LOCAL VIDEO UPLOAD --------

    if file:

        ext = Path(file.filename or "video.mp4").suffix or ".mp4"
        source = UPLOADS / f"{job_id}{ext}"

        try:
            with source.open("wb") as out:
                shutil.copyfileobj(file.file, out)
        finally:
            await file.close()

        if not source.exists() or source.stat().st_size == 0:
            raise HTTPException(400, "Uploaded video is empty.")

    # -------- YOUTUBE VIDEO DOWNLOAD --------

    else:

        output_template = UPLOADS / f"{job_id}.%(ext)s"

        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--retries", "3",
            "--fragment-retries", "3",
            "--extractor-retries", "3",
            "--retry-sleep", "2",
            "-f", "bv*+ba/b",
            "--merge-output-format", "mp4",
            "--print", "after_move:filepath",
            "-o", str(output_template),
            youtube_url
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800
            )

            downloaded_paths = [
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip()
            ]

            if result.returncode != 0 or not downloaded_paths:

                error = (
                    result.stderr or "Unknown YouTube download error"
                )[-1000:]

                raise HTTPException(
                    400,
                    "YouTube download failed: " + error
                )

            source = Path(downloaded_paths[-1])

            if not source.is_file() or source.stat().st_size == 0:
                raise HTTPException(
                    400,
                    "Downloaded video file was not found or is empty."
                )

        except subprocess.TimeoutExpired:
            raise HTTPException(
                408,
                "YouTube download timed out. Please try again."
            )

    # -------- START AI PROCESSING --------

    jobs[job_id] = {
        "status": "processing",
        "clips": [],
        "error": None
    }

    def worker():
        try:
            clips = process_video(
                source,
                OUTPUTS,
                job_id,
                category,
                duration,
                aspect,
                subtitles
            )

            jobs[job_id]["clips"] = clips
            jobs[job_id]["status"] = "done"

        except Exception as e:
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["status"] = "error"

    threading.Thread(
        target=worker,
        daemon=True
    ).start()

    return {
        "job_id": job_id,
        "status": "processing"
    }


# ---------------- JOB STATUS ----------------

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):

    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    return jobs[job_id]


# ---------------- DOWNLOAD CLIPS ----------------

@app.get("/clips/{filename}")
def get_clip(filename: str):

    path = (OUTPUTS / filename).resolve()

    if OUTPUTS not in path.parents or not path.is_file():
        raise HTTPException(404, "Clip not found")

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=path.name
    )


# ---------------- LIVE STREAM RECORDING ----------------

@app.post("/api/live")
def live(
    url: str = Form(...),
    segment_minutes: int = Form(5)
):

    if not url.startswith("https://"):
        raise HTTPException(
            400,
            "Use a valid HTTPS stream URL."
        )

    if segment_minutes < 1 or segment_minutes > 60:
        raise HTTPException(
            400,
            "Segment duration must be between 1 and 60 minutes."
        )

    job_id = uuid.uuid4().hex[:12]

    jobs[job_id] = {
        "status": "recording",
        "clips": [],
        "error": None
    }

    def worker():
        try:
            capture_live(
                url,
                OUTPUTS,
                job_id,
                segment_minutes,
                jobs
            )

        except Exception as e:
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["status"] = "error"

    threading.Thread(
        target=worker,
        daemon=True
    ).start()

    return {
        "job_id": job_id,
        "status": "recording"
    }
