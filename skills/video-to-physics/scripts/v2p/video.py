"""Video I/O: probe, frame reading (video file or image directory), ffmpeg-backed writing, fonts.

Conventions: frame n of a clip is at t = n / fps. Frames are numpy uint8 arrays in BGR order (OpenCV),
unless a function says otherwise.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi")


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def require_ffmpeg() -> str:
    """Return the ffmpeg executable path or raise a clear error."""
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH; install it (macOS: brew install ffmpeg, "
                           "Debian/Ubuntu: apt-get install ffmpeg) to encode videos")
    return exe


def image_files(directory) -> list[Path]:
    return sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def probe(path, exact=True) -> dict:
    """fps, frame count, size, duration and sha256 of a video. `exact` counts frames by decoding."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if exact:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        n = 0
        while cap.grab():
            n += 1
    cap.release()
    return {"path": str(path), "fps": round(fps, 6), "frames": n, "width": w, "height": h,
            "duration_s": round(n / fps, 6) if fps else None, "sha256": sha256_file(path)}


def iter_frames(source, start=0, count=None):
    """Yield BGR frames from a video file or a directory of images (sorted by name)."""
    source = Path(source)
    if source.is_dir():
        files = image_files(source)[start:]
        for k, f in enumerate(files):
            if count is not None and k >= count:
                return
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"cannot read image {f}")
            yield img
        return
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {source}")
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    k = 0
    while count is None or k < count:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
        k += 1
    cap.release()


def read_frames(source, start=0, count=None) -> list[np.ndarray]:
    return list(iter_frames(source, start, count))


def source_fps(source, default=24.0) -> float:
    """fps of a video; image directories use `default` (or fps from a sibling meta.json if present)."""
    source = Path(source)
    if source.is_dir():
        meta = source / "meta.json"
        if meta.exists():
            return float(json.loads(meta.read_text()).get("fps", default))
        return float(default)
    cap = cv2.VideoCapture(str(source))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps or float(default)


def source_len(source) -> int:
    source = Path(source)
    if source.is_dir():
        return len(image_files(source))
    return probe(source)["frames"]


class FrameSource:
    """Random access to frames of a video or image dir: frame(i) clamps i to [0, n-1].

    Sequential forward access is cheap; seeking backwards reopens. Returns BGR (or RGB if rgb=True).
    """

    def __init__(self, source, rgb=False):
        self.source, self.rgb = Path(source), rgb
        self.files = image_files(self.source) if self.source.is_dir() else None
        if self.files is not None:
            self.n = len(self.files)
        else:
            self.cap = cv2.VideoCapture(str(self.source))
            self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.i, self.last = -1, None
        if self.n <= 0:
            raise ValueError(f"no frames in {self.source}")

    def frame(self, i) -> np.ndarray:
        i = min(max(int(i), 0), self.n - 1)
        if self.files is not None:
            img = cv2.imread(str(self.files[i]), cv2.IMREAD_COLOR)
        else:
            if i < self.i:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                self.i = i - 1
            while self.i < i:
                ok, f = self.cap.read()
                if not ok:      # container frame count can overstate; hold the last decoded frame
                    self.n = max(self.i + 1, 1)
                    break
                self.last, self.i = f, self.i + 1
            img = self.last
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if self.rgb else img


class VideoWriter:
    """Pipe raw frames into ffmpeg (H.264, yuv420p). Use as a context manager.

    Odd sizes are rejected (yuv420p needs even width/height). `pix_fmt_in` is 'bgr24' or 'rgb24'.
    """

    def __init__(self, path, width, height, fps, crf=18, pix_fmt_in="bgr24", preset="medium"):
        if width % 2 or height % 2:
            raise ValueError(f"frame size must be even for yuv420p, got {width}x{height}")
        self.path, self.size = Path(path), (int(width), int(height))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [require_ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix_fmt_in,
               "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", preset,
               "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.count = 0

    def write(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        if (w, h) != self.size or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame {w}x{h} does not match writer {self.size[0]}x{self.size[1]}x3")
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.count += 1

    def close(self):
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace") if self.proc.stderr else ""
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}: {err.strip()[-800:]}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_video(frames, path, fps, crf=18, rgb=False) -> Path:
    """Encode an iterable of equally sized frames to an MP4; returns the path."""
    it = iter(frames)
    first = next(it)
    h, w = first.shape[:2]
    with VideoWriter(path, w, h, fps, crf=crf, pix_fmt_in="rgb24" if rgb else "bgr24") as vw:
        vw.write(first)
        for f in it:
            vw.write(f)
    return Path(path)


def even(x) -> int:
    return int(x) - int(x) % 2


# ------------------------------------------------------------------ fonts (macOS, Linux CI, fallback)
_FONT_CANDIDATES = [
    # (regular, bold, variable-font?)
    ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/SFNS.ttf", True),
    ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Helvetica.ttc", False),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", False),
    ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", False),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf", False),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf", False),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf", False),
    ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", False),   # resolved by Pillow's search path if installed
]
_font_cache: dict = {}


def font(px: int, weight: str = "Regular"):
    """A PIL font at `px` pixels. weight: Regular | Medium | Semibold | Bold (best effort per font)."""
    from PIL import ImageFont
    key = (max(int(px), 1), weight)
    if key in _font_cache:
        return _font_cache[key]
    bold = weight in ("Semibold", "Bold")
    f = None
    for regular, boldpath, variable in _FONT_CANDIDATES:
        path = boldpath if bold else regular
        if not (Path(path).exists() or "/" not in path):
            continue
        try:
            index = 1 if (bold and path.endswith("Helvetica.ttc")) else 0
            f = ImageFont.truetype(path, key[0], index=index)
            if variable:
                try:
                    f.set_variation_by_name(weight)
                except Exception:
                    pass
            break
        except Exception:
            f = None
    if f is None:
        try:
            f = ImageFont.load_default(size=key[0])
        except TypeError:       # Pillow < 10.1
            f = ImageFont.load_default()
    _font_cache[key] = f
    return f
