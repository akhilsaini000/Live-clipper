
import os
import subprocess
import math
import wave
import array
from pathlib import Path

_whisper = None
_face_detector = None

# Lightweight Whisper model for low-memory servers
def transcribe(path):
    global _whisper
    try:
        from faster_whisper import WhisperModel

        if _whisper is None:
            _whisper = WhisperModel(
                "tiny",
                device="cpu",
                compute_type="int8",
                cpu_threads=1,
                num_workers=1
            )

        segments, _ = _whisper.transcribe(
            str(path),
            vad_filter=True,
            beam_size=1
        )

        return [
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text.strip()
            }
            for s in segments
        ]

    except Exception as e:
        print("Whisper error:", str(e))
        return []


FUNNY = {
    "haha", "hahaha", "lol", "laugh", "funny",
    "joke", "hilarious", "मज़ाक", "हंस", "हँस",
    "मजेदार", "भाई"
}

IMPORTANT = {
    "important", "remember", "because", "therefore",
    "lesson", "key point", "in conclusion",
    "ध्यान", "जरूरी", "महत्वपूर्ण", "सीख"
}


def transcript_score(text, category):
    t = text.lower()

    f = sum(1 for w in FUNNY if w in t)
    i = sum(1 for w in IMPORTANT if w in t)

    if category == "funny":
        return f * 2 + (1 if "?" in t else 0)

    if category == "important":
        return i * 2 + min(len(t) / 180, 2)

    return f * 2 + i * 2 + min(len(t) / 220, 1)


def extract_audio(path, wav):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-threads", "1",
            str(wav)
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )


# Read audio in small chunks instead of loading the entire audio into RAM
def laughter_candidates(path):
    wav = path.with_suffix(".lc.wav")

    try:
        extract_audio(path, wav)

        candidates = []

        with wave.open(str(wav), "rb") as w:
            rate = w.getframerate()
            channels = w.getnchannels()
            sample_width = w.getsampwidth()

            if sample_width != 2 or channels != 1:
                return []

            chunk_size = int(rate * 0.5)
            idx = 0

            while True:
                raw = w.readframes(chunk_size)

                if not raw:
                    break

                samples = array.array("h")
                samples.frombytes(raw)

                if not samples:
                    break

                count = len(samples)

                energy = sum(
                    float(v) * float(v)
                    for v in samples
                ) / count

                rms = math.sqrt(energy) / 32768.0

                crossings = sum(
                    1 for a, b in zip(samples, samples[1:])
                    if (a < 0 <= b) or (a >= 0 > b)
                )

                zcr = crossings / max(1, count - 1)

                if rms > 0.045 and zcr > 0.035:
                    candidates.append(idx / rate)

                idx += count

        groups = []

        for t in candidates:
            if not groups or t - groups[-1][-1] > 0.8:
                groups.append([t])
            else:
                groups[-1].append(t)

        return [
            (g[0], g[-1] + 0.5)
            for g in groups if len(g) >= 2
        ]

    except Exception as e:
        print("Laughter analysis error:", str(e))
        return []

    finally:
        try:
            wav.unlink()
        except Exception:
            pass


# Analyse small frames to reduce OpenCV memory usage
def visual_candidates(path):
    global _face_detector

    try:
        import cv2

        if _face_detector is None:
            cascade_path = (
                cv2.data.haarcascades
                + "haarcascade_frontalface_default.xml"
            )
            _face_detector = cv2.CascadeClassifier(cascade_path)

        if _face_detector.empty():
            print("Face detector could not be loaded")
            return []

        cap = cv2.VideoCapture(str(path))

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if total_frames <= 0:
            cap.release()
            return []

        total_seconds = total_frames / fps
        out = []
        prev = None

        for second in range(int(total_seconds) + 1):
            cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)

            ok, frame = cap.read()

            if not ok:
                continue

            height, width = frame.shape[:2]

            new_width = 320
            new_height = max(1, int(height * new_width / width))

            small = cv2.resize(
                frame,
                (new_width, new_height)
            )

            gray = cv2.cvtColor(
                small,
                cv2.COLOR_BGR2GRAY
            )

            faces = _face_detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(25, 25)
            )

            change = (
                0 if prev is None
                else float(cv2.absdiff(gray, prev).mean())
            )

            if len(faces) > 0 or change > 18:
                out.append(float(second))

            prev = gray

            del frame, small, gray

        cap.release()
        return out

    except Exception as e:
        print("Visual analysis error:", str(e))
        return []


def choose_windows(path, category, duration):
    transcript = transcribe(path)

    laughs = laughter_candidates(path)
    visuals = visual_candidates(path)

    duration_s = float(
        subprocess.check_output([
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ]).decode().strip()
    )

    windows = []

    for s in transcript:
        center = (s["start"] + s["end"]) / 2
        score = transcript_score(s["text"], category)

        if score > 0:
            windows.append((
                max(0, center - duration / 2),
                min(duration_s, center + duration / 2),
                score,
                "speech"
            ))

    for a, b in laughs:
        center = (a + b) / 2

        windows.append((
            max(0, center - duration / 2),
            min(duration_s, center + duration / 2),
            5,
            "laughter-candidate"
        ))

    for t in visuals:
        windows.append((
            max(0, t - duration / 2),
            min(duration_s, t + duration / 2),
            1.5,
            "visual-reaction-candidate"
        ))

    windows.sort(key=lambda x: x[2], reverse=True)

    selected = []

    for w in windows:
        if w[1] - w[0] < 5:
            continue

        if all(
            abs(
                (w[0] + w[1]) / 2 -
                (q[0] + q[1]) / 2
            ) > duration * 0.65
            for q in selected
        ):
            selected.append(w)

        if len(selected) >= 12:
            break

    selected.sort(key=lambda x: x[0])

    return selected, transcript


def make_srt(transcript, start, end, path):
    lines = []
    n = 1

    for s in transcript:
        if s["end"] <= start or s["start"] >= end:
            continue

        a = max(0, s["start"] - start)
        b = max(a + 0.2, s["end"] - start)

        def ts(v):
            ms = int(v * 1000)
            h = ms // 3600000
            m = (ms % 3600000) // 60000
            sec = (ms % 60000) // 1000
            milli = ms % 1000

            return f"{h:02}:{m:02}:{sec:02},{milli:03}"

        lines += [
            str(n),
            f"{ts(a)} --> {ts(b)}",
            s["text"],
            ""
        ]

        n += 1

    path.write_text("\n".join(lines), encoding="utf-8")


def process_video(source, outdir, job, category, duration, aspect, subtitles):
    windows, transcript = choose_windows(
        source, category, duration
    )

    results = []

    # Lower output resolution for the Render Free memory limit
    vf = {
        "9:16": "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280",
        "16:9": "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720",
        "1:1": "scale=720:720:force_original_aspect_ratio=increase,crop=720:720"
    }.get(aspect, "scale=720:-2")

    for idx, (start, end, score, reason) in enumerate(windows, 1):
        out = outdir / f"{job}_{idx:02}.mp4"
        length = max(5, end - start)

        final_vf = vf

        if subtitles and transcript:
            srt = out.with_suffix(".srt")
            make_srt(transcript, start, end, srt)

            escaped = (
                str(srt.resolve())
                .replace("\\", "/")
                .replace(":", "\\:")
            )

            final_vf += f",subtitles='{escaped}'"

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(source),
            "-t", str(length),
            "-vf", final_vf,
            "-threads", "1",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "26",
            "-c:a", "aac",
            "-threads", "1",
            str(out)
        ]

        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        results.append({
            "filename": out.name,
            "url": f"/clips/{out.name}",
            "score": round(score, 2),
            "reason": reason,
            "start": round(start, 1),
            "end": round(end, 1)
        })

    return results


def capture_live(url, outdir, job, segment_minutes, jobs):
    segment_minutes = max(1, min(15, int(segment_minutes)))

    pattern = str(outdir / f"{job}_live_%03d.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-i", url,
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(segment_minutes * 60),
        "-reset_timestamps", "1",
        pattern
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    jobs[job]["status"] = "recording"

    import time

    while proc.poll() is None:
        time.sleep(5)

        clips = []

        for p in sorted(outdir.glob(f"{job}_live_*.mp4")):
            clips.append({
                "filename": p.name,
                "url": f"/clips/{p.name}",
                "reason": "live segment captured"
            })

        jobs[job]["clips"] = clips

    if proc.returncode not in (0, None):
        raise RuntimeError(
            "Live capture stopped; check stream URL/network."
        )

    jobs[job]["status"] = "done"
