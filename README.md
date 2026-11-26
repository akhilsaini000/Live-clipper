# LiveClipper — iPhone PWA + AI processing backend

This is a runnable starter implementation for an iPhone-friendly web app and a Python backend.
It supports local video upload from iPhone, clip generation, subtitles, transcript-based context scoring,
audio laughter-candidate scoring, and optional face/reaction analysis.

## Run locally (Mac/Linux)
1. Install Python 3.11+ and FFmpeg.
2. `cd backend && python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `uvicorn main:app --host 0.0.0.0 --port 8000`
5. Serve `frontend` over HTTPS (for example, deploy it to a static host) and set `API_BASE` in `frontend/app.js`
   to the public backend URL.

## iPhone
Open the deployed HTTPS frontend in Safari → Share → Add to Home Screen.
The PWA is not an App Store-native IPA. Native iOS packaging requires Xcode and Apple signing.

## AI notes
- Speech transcription uses faster-whisper (model downloads on first run).
- Laughter scoring is an audio heuristic using energy + spectral features; it flags candidates and is not a perfect laugh classifier.
- Visual reaction analysis uses OpenCV face detection and frame-change cues. It does not infer a person's identity or guarantee emotion classification.
- Context selection combines transcript cues, speech segments, laughter candidates and visual cues. It is a practical baseline, not human-level multimodal reasoning.
- YouTube URL ingestion uses yt-dlp. Only process content you own or have permission to download/edit.
- Continuous YouTube Live capture is implemented as a segmented FFmpeg capture job, but needs a reachable backend and a stable stream URL/network. Use a server with enough disk and CPU.
- This build does not include YouTube OAuth publishing or automatic uploads.

## Production deployment
Deploy the backend on a private server/VPS with persistent storage, HTTPS, authentication, upload limits and a job queue before exposing it publicly. This demo backend has no user authentication and should not be exposed publicly as-is.
