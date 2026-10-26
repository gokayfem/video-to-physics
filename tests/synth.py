"""Synthetic test data shared by the tool tests (no external files)."""
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "video-to-physics" / "scripts"


def tmpdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="v2p_test_"))


def need_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise unittest.SkipTest("ffmpeg not installed")


def texture(w, h, seed=0) -> np.ndarray:
    """Smooth random texture (BGR uint8), good for ECC alignment."""
    rng = np.random.default_rng(seed)
    base = rng.random((h // 8 + 2, w // 8 + 2)).astype(np.float32)
    img = cv2.resize(base, (w + 16, h + 16), interpolation=cv2.INTER_CUBIC)
    img += 0.5 * cv2.resize(rng.random((h // 3 + 2, w // 3 + 2)).astype(np.float32), (w + 16, h + 16))
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    img = (img - img.min()) / (img.max() - img.min())
    g = (img * 220 + 20).astype(np.uint8)
    return cv2.merge([g, np.roll(g, 7, 1), np.roll(g, 5, 0)])


def shifted(img, dx, dy, w, h) -> np.ndarray:
    """Content moved by (+dx, +dy) px, cropped to w x h (sub-pixel, bicubic)."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT)[:h, :w]


def moving_video(path, n=12, w=320, h=240, fps=24, step=(0.6, -0.35), crf=8):
    from v2p.video import write_video
    tex = texture(w + 64, h + 64)
    frames = [shifted(tex, 20 + step[0] * k, 20 + step[1] * k, w, h) for k in range(n)]
    write_video(frames, path, fps, crf=crf)
    return frames


def gradient_video(path, n=10, w=160, h=96, fps=24, colour=(40, 160, 220)):
    from v2p.video import write_video
    frames = []
    for k in range(n):
        f = np.zeros((h, w, 3), np.uint8)
        f[:] = colour
        cv2.circle(f, (10 + 12 * k, h // 2), 12, (255, 255, 255), -1)
        frames.append(f)
    write_video(frames, path, fps)
    return frames
