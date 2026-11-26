import os, subprocess, json, re, math, uuid
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
import cv2

_whisper = None
def transcribe(path):
    global _whisper
    try:
        from faster_whisper import WhisperModel
        if _whisper is None: _whisper = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = _whisper.transcribe(str(path), vad_filter=True)
        return [{"start":float(s.start),"end":float(s.end),"text":s.text.strip()} for s in segments]
    except Exception as e:
        return []

FUNNY = {"haha","hahaha","lol","laugh","funny","joke"," hilarious"," मज़ाक","हंस","हँस","मजेदार","भाई"}
IMPORTANT = {"important","remember","because","therefore","lesson","key point","in conclusion","ध्यान","जरूरी","महत्वपूर्ण","सीख"}
def transcript_score(text, category):
    t=text.lower()
    f=sum(1 for w in FUNNY if w.strip() and w.strip() in t)
    i=sum(1 for w in IMPORTANT if w in t)
    if category=="funny": return f*2 + (1 if "?" in t else 0)
    if category=="important": return i*2 + min(len(t)/180,2)
    return f*2+i*2+min(len(t)/220,1)

def extract_audio(path, wav):
    subprocess.run(["ffmpeg","-y","-i",str(path),"-vn","-ac","1","-ar","16000",str(wav)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
def laughter_candidates(path):
    wav=path.with_suffix(".lc.wav")
    try:
        extract_audio(path,wav)
        import wave
        with wave.open(str(wav),"rb") as w:
            rate=w.getframerate(); raw=w.readframes(w.getnframes())
        x=np.frombuffer(raw,dtype=np.int16).astype(np.float32)/32768
        win=int(rate*0.5); hop=int(rate*0.25); candidates=[]
        for idx in range(0,max(0,len(x)-win),hop):
            frame=x[idx:idx+win]
            rms=float(np.sqrt(np.mean(frame**2)+1e-9))
            zcr=float(np.mean(np.abs(np.diff(np.sign(frame))))/2)
            # laughter often appears as repeated energetic bursts; candidate heuristic only
            if rms>0.045 and zcr>0.035:
                candidates.append(idx/rate)
        groups=[]
        for t in candidates:
            if not groups or t-groups[-1][-1]>.8: groups.append([t])
            else: groups[-1].append(t)
        return [(g[0],g[-1]+.5) for g in groups if len(g)>=2]
    except Exception: return []
    finally:
        try: wav.unlink()
        except: pass

def visual_candidates(path):
    cap=cv2.VideoCapture(str(path)); fps=cap.get(cv2.CAP_PROP_FPS) or 25
    face=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    out=[]; prev=None; frame_no=0
    while cap.isOpened():
        ok, frame=cap.read()
        if not ok: break
        if frame_no % max(1,int(fps*1.0))==0:
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=face.detectMultiScale(gray,1.2,5,minSize=(40,40))
            change=0 if prev is None else float(cv2.absdiff(gray,prev).mean())
            if len(faces)>0 or change>18: out.append(frame_no/fps)
            prev=gray
        frame_no+=1
    cap.release()
    return out

def choose_windows(path, category, duration):
    transcript=transcribe(path)
    laughs=laughter_candidates(path)
    visuals=visual_candidates(path)
    duration_s=float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(path)]).decode().strip())
    windows=[]
    for s in transcript:
        center=(s["start"]+s["end"])/2
        score=transcript_score(s["text"],category)
        if score>0: windows.append((max(0,center-duration/2), min(duration_s,center+duration/2), score, "speech"))
    for a,b in laughs:
        c=(a+b)/2; windows.append((max(0,c-duration/2),min(duration_s,c+duration/2),5,"laughter-candidate"))
    for t in visuals:
        windows.append((max(0,t-duration/2),min(duration_s,t+duration/2),1.5,"visual-reaction-candidate"))
    # Merge near-duplicate windows and keep strongest.
    windows.sort(key=lambda x:x[2],reverse=True); selected=[]
    for w in windows:
        if w[1]-w[0]<5: continue
        if all(abs((w[0]+w[1])/2-(q[0]+q[1])/2)>duration*.65 for q in selected):
            selected.append(w)
        if len(selected)>=12: break
    selected.sort(key=lambda x:x[0])
    return selected, transcript

def make_srt(transcript, start, end, path):
    lines=[]; n=1
    for s in transcript:
        if s["end"]<=start or s["start"]>=end: continue
        a=max(0,s["start"]-start); b=max(a+.2,s["end"]-start)
        def ts(v):
            ms=int(v*1000); h=ms//3600000; m=(ms%3600000)//60000; sec=(ms%60000)//1000; milli=ms%1000
            return f"{h:02}:{m:02}:{sec:02},{milli:03}"
        lines += [str(n),f"{ts(a)} --> {ts(b)}",s["text"],""]; n+=1
    path.write_text("\n".join(lines),encoding="utf-8")

def process_video(source, outdir, job, category, duration, aspect, subtitles):
    windows, transcript=choose_windows(source,category,duration)
    results=[]
    for idx,(start,end,score,reason) in enumerate(windows,1):
        out=outdir/f"{job}_{idx:02}.mp4"; length=max(5,end-start)
        vf={"9:16":"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
            "16:9":"scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080",
            "1:1":"scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080"}.get(aspect,"scale=1280:-2")
        cmd=["ffmpeg","-y","-ss",str(start),"-i",str(source),"-t",str(length),"-vf",vf]
        if subtitles and transcript:
            srt=out.with_suffix(".srt"); make_srt(transcript,start,end,srt)
            # use absolute path escaped for ffmpeg subtitles filter
            escaped=str(srt.resolve()).replace("\\","/").replace(":","\\:")
            cmd += ["-vf",vf+f",subtitles='{escaped}'"]
        cmd += ["-c:v","libx264","-preset","veryfast","-crf","23","-c:a","aac",str(out)]
        subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
        results.append({"filename":out.name,"url":f"/clips/{out.name}","score":round(score,2),"reason":reason,"start":round(start,1),"end":round(end,1)})
    return results

def capture_live(url, outdir, job, segment_minutes, jobs):
    segment_minutes=max(1,min(15,int(segment_minutes)))
    pattern=str(outdir/f"{job}_live_%03d.mp4")
    cmd=["ffmpeg","-y","-i",url,"-c","copy","-f","segment","-segment_time",str(segment_minutes*60),"-reset_timestamps","1",pattern]
    proc=subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    jobs[job]["status"]="recording"
    while proc.poll() is None:
        import time; time.sleep(5)
        clips=[]
        for p in sorted(outdir.glob(f"{job}_live_*.mp4")):
            clips.append({"filename":p.name,"url":f"/clips/{p.name}","reason":"live segment captured"})
        jobs[job]["clips"]=clips
    if proc.returncode not in (0,None): raise RuntimeError("Live capture stopped; check stream URL/network.")
    jobs[job]["status"]="done"
