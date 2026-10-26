"""Image-to-video candidates on fal (queue REST API) and their screening.

Layout per call:  CASE/generation/<round>/
    first_frame.png  first_frame.json (sha256)  upload.json  plan.json
    cand<k>/ input.json  submission_started.json  queue.json  status.json  result.json  video.mp4  download.json
             error.json (on failure)
Safety rules (paid API):
  * input.json + submission_started.json are written BEFORE the POST, queue.json right after it, all before
    any polling. A candidate with submission_started.json but no queue.json is UNCERTAIN (the POST may have
    gone through) and is never resubmitted automatically; delete the cand dir by hand once you have checked.
  * polling is bounded by a deadline; a timed-out job keeps queue.json so a rerun resumes it.
  * API calls go only to queue.fal.run; downloads only from fal media hosts; every download is sha256'd.
  * FAL_KEY is read from the environment and never printed or written.
`--dry-run` writes first_frame.png, plan.json and every cand*/input.json without network or key.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import mimetypes
import os
import random
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

QUEUE_HOST = "queue.fal.run"
REST_STORAGE = "https://rest.alpha.fal.ai/storage/upload/initiate?storage_type=fal-cdn-v3"
MEDIA_HOSTS = ("fal.media", "fal.ai", "fal.run")


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def save_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def _host_ok(url, hosts) -> bool:
    h = (urllib.parse.urlparse(url).hostname or "").lower()
    return any(h == s or h.endswith("." + s) for s in hosts)


def _key() -> str:
    k = os.environ.get("FAL_KEY")
    if not k:
        raise RuntimeError("FAL_KEY is not set in the environment")
    return k


def api(url, payload=None, timeout=90):
    """GET (payload None) or POST JSON to the fal queue API."""
    if urllib.parse.urlparse(url).hostname != QUEUE_HOST or not url.startswith("https://"):
        raise ValueError(f"refusing API call to unexpected host: {urllib.parse.urlparse(url).hostname}")
    headers = {"Authorization": "Key " + _key()}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=timeout) as r:
        return json.load(r)


def upload(path) -> str:
    """Upload a local file to fal storage; returns its URL. Uses fal_client if installed, else REST."""
    try:
        import fal_client  # type: ignore
        return fal_client.upload_file(str(path))
    except ImportError:
        pass
    path = Path(path)
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    req = urllib.request.Request(REST_STORAGE, data=json.dumps({"content_type": ctype, "file_name": path.name}).encode(),
                                 headers={"Authorization": "Key " + _key(), "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        info = json.load(r)
    put = urllib.request.Request(info["upload_url"], data=path.read_bytes(), method="PUT",
                                 headers={"Content-Type": ctype})
    with urllib.request.urlopen(put, timeout=300):
        pass
    return info["file_url"]


def _find_video_url(res):
    if isinstance(res, dict):
        v = res.get("video")
        if isinstance(v, dict) and "url" in v:
            return v["url"]
        for val in res.values():
            u = _find_video_url(val)
            if u:
                return u
    elif isinstance(res, list):
        for val in res:
            u = _find_video_url(val)
            if u:
                return u
    elif isinstance(res, str) and res.startswith("https://") and res.split("?")[0].endswith((".mp4", ".webm", ".mov")):
        return res
    return None


def run_job(d: Path, endpoint: str, inp: dict, timeout_s=1800, poll_s=8) -> str:
    """Submit (once), poll (bounded) and download one candidate in directory d. Returns a status line."""
    d.mkdir(parents=True, exist_ok=True)
    video, qf, rf = d / "video.mp4", d / "queue.json", d / "result.json"
    if video.exists():
        return f"{d.name}: already downloaded"
    try:
        if not qf.exists():
            if (d / "submission_started.json").exists():
                return f"{d.name}: UNCERTAIN earlier submission (no queue.json); not resubmitting"
            save_json(d / "input.json", inp)
            save_json(d / "submission_started.json", {"utc": _utc(), "endpoint": endpoint})
            q = api(f"https://{QUEUE_HOST}/{endpoint}", inp)
            save_json(qf, q)
        else:
            q = json.loads(qf.read_text())
        deadline = time.time() + timeout_s
        while not rf.exists():
            s = api(q["status_url"])
            save_json(d / "status.json", s)
            if s.get("status") == "COMPLETED":
                if s.get("error"):
                    raise RuntimeError(str(s["error"]))
                save_json(rf, api(q["response_url"]))
                break
            if time.time() > deadline:
                return f"{d.name}: TIMEOUT (request {q.get('request_id')} kept in queue.json; rerun to resume)"
            time.sleep(poll_s)
        url = _find_video_url(json.loads(rf.read_text()))
        if not url or not _host_ok(url, MEDIA_HOSTS):
            raise ValueError(f"no video URL on an expected media host in result ({url and urllib.parse.urlparse(url).hostname})")
        h, part = hashlib.sha256(), d / "video.part"
        with urllib.request.urlopen(url, timeout=300) as src, open(part, "wb") as dst:
            while chunk := src.read(1 << 20):
                h.update(chunk)
                dst.write(chunk)
        part.replace(video)
        save_json(d / "download.json", {"source_url": url, "sha256": h.hexdigest(), "bytes": video.stat().st_size,
                                        "utc": _utc()})
        return f"{d.name}: downloaded"
    except Exception as e:  # recorded, never retried blindly
        save_json(d / "error.json", {"type": type(e).__name__, "detail": str(e)[:1500], "utc": _utc()})
        return f"{d.name}: ERROR {type(e).__name__}: {e}"


def case_generation_config(case) -> dict:
    p = Path(case) / "config" / "case.json"
    return json.loads(p.read_text()).get("generation", {}) if p.exists() else {}


def generate(case, first_frame, prompt, n=3, seed=None, endpoint=None, duration=10, round_name="",
             settings=None, dry_run=False, workers=4, timeout_s=1800, log=print) -> dict:
    """Plan (and unless dry_run, run) n image-to-video candidates for CASE. Returns the plan dict."""
    case = Path(case)
    cfg = case_generation_config(case)
    endpoint = endpoint or cfg.get("endpoint")
    if not endpoint and not dry_run:
        raise ValueError("no endpoint: pass --endpoint or set generation.endpoint in config/case.json")
    gdir = case / "generation" / round_name if round_name else case / "generation"
    gdir.mkdir(parents=True, exist_ok=True)
    frame = gdir / ("first_frame" + Path(first_frame).suffix.lower())
    if Path(first_frame).resolve() != frame.resolve():
        shutil.copy2(first_frame, frame)
    sha = hashlib.sha256(frame.read_bytes()).hexdigest()
    save_json(gdir / "first_frame.json", {"source": str(first_frame), "sha256": sha})

    up = gdir / "upload.json"
    if up.exists() and json.loads(up.read_text()).get("frame_sha256") == sha:
        image_url = json.loads(up.read_text())["url"]
    elif dry_run:
        image_url = "DRY-RUN"
    else:
        image_url = upload(frame)
        save_json(up, {"url": image_url, "frame_sha256": sha, "utc": _utc()})

    base = {**cfg.get("settings", {}), **(settings or {})}
    if duration is not None:
        base["duration"] = duration
    seeds = [seed + k for k in range(n)] if seed is not None else [random.randrange(1, 2 ** 31) for _ in range(n)]
    jobs = []
    for k in range(n):
        inp = {"prompt": prompt, "image_url": image_url, **base, "seed": seeds[k]}
        jobs.append((gdir / f"cand{k}", inp))
    plan = {"endpoint": endpoint, "n": n, "seeds": seeds, "settings": base, "prompt": prompt,
            "first_frame_sha256": sha, "dry_run": dry_run, "utc": _utc()}
    save_json(gdir / "plan.json", plan)
    if dry_run:
        for d, inp in jobs:
            d.mkdir(parents=True, exist_ok=True)
            if not (d / "submission_started.json").exists():
                save_json(d / "input.json", inp)
            log(f"{d.name}: dry-run seed {inp['seed']} -> {d / 'input.json'}")
        return plan
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(lambda j: run_job(j[0], endpoint, j[1], timeout_s), jobs):
            log(line)
    return plan


# ------------------------------------------------------------------ screening
def screen(case, round_name="", first_frame=None, sheet_frames=8, log=print) -> dict:
    """Score every candidate: frame-0 diff vs the first frame, camera drift (affine ECC on a static mask),
    motion timeline. Writes screen.json and screen_sheet.jpg in the generation round dir."""
    import cv2
    import numpy as np
    from .stabilize import affine_drift
    from .video import read_frames, source_fps
    from .compose import label

    gdir = Path(case) / "generation" / round_name if round_name else Path(case) / "generation"
    ff = Path(first_frame) if first_frame else next(
        (p for p in sorted(gdir.glob("first_frame.*")) if p.suffix.lower() != ".json"), None)
    ref = cv2.imread(str(ff)) if ff else None
    report, rows = {}, []
    for cand in sorted(p for p in gdir.glob("cand*") if p.is_dir()):
        v = cand / "video.mp4"
        if not v.exists():
            report[cand.name] = {"status": "missing"}
            continue
        F = read_frames(v)
        if not F:
            report[cand.name] = {"status": "undecodable"}
            continue
        fps = source_fps(v)
        entry = {"status": "ok", "frames": len(F), "fps": fps, "size": [F[0].shape[1], F[0].shape[0]]}
        if ref is not None:
            f0 = cv2.resize(F[0], (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
            entry["frame0_mean_abs_diff_8bit"] = round(float(np.abs(f0.astype(np.float32) - ref.astype(np.float32)).mean()), 2)
        motion = np.array([float(np.abs(F[i].astype(np.float32) - F[i - 1].astype(np.float32)).mean())
                           for i in range(1, len(F))] or [0.0])
        stack = np.stack([cv2.cvtColor(F[i], cv2.COLOR_BGR2GRAY) for i in range(0, len(F), max(1, len(F) // 24))])
        mask = stack.astype(np.float32).std(0) < 6
        entry["static_fraction"] = round(float(mask.mean()), 3)
        drift = [affine_drift(F[0], F[i], mask if mask.mean() > 0.05 else None) for i in (len(F) // 2, len(F) - 1)]
        entry.update({"camera_drift_px_mid_end": [round(x, 2) for x in drift],
                      "motion_mean": round(float(motion.mean()), 3),
                      "motion_timeline": [round(float(x), 3) for x in motion],
                      "peak_motion_t_s": round(float(np.argmax(motion) + 1) / fps, 3),
                      "high_motion_frames": int((motion > np.median(motion) * 3 + 1).sum())})
        report[cand.name] = entry
        picks = np.linspace(0, len(F) - 1, sheet_frames).round().astype(int)
        tw = 320
        th = max(2, round(tw * F[0].shape[0] / F[0].shape[1]))
        row = np.hstack([cv2.resize(F[i], (tw, th), interpolation=cv2.INTER_AREA) for i in picks])
        txt = f"{cand.name}  f0diff {entry.get('frame0_mean_abs_diff_8bit', 'n/a')}  drift {drift[1]:.1f}px"
        rows.append(label(row, txt, (6, 6), 14))
    save_json(gdir / "screen.json", report)
    if rows:
        cv2.imwrite(str(gdir / "screen_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    log(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "motion_timeline"} for k, v in report.items()}))
    return report
