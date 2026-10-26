"""Review compositions: reference | ours | 50/50 blend side-by-side videos, contact sheets, burned labels.

No retiming anywhere: output frame n shows input frame n of both sources, at the reference's true fps.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .video import FrameSource, VideoWriter, even, font, source_fps, source_len


def burn_label(img, text, xy, px=18, fg=(255, 255, 255), bg=(25, 25, 25), weight="Semibold") -> np.ndarray:
    """Burn `text` with a dark box at xy (top-left) into a BGR image; returns a new array."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    f = font(px, weight)
    x0, y0, x1, y1 = d.textbbox(xy, text, font=f)
    pad = max(3, px // 4)
    d.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], fill=bg)
    d.text(xy, text, font=f, fill=fg)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


label = burn_label


def fit_size(w, h, max_w) -> tuple[int, int]:
    s = min(1.0, max_w / w)
    return max(2, even(round(w * s))), max(2, even(round(h * s)))


def sbs(ref, ours, out, label_ours="ours (physics)", label_ref="reference", frames=None, panel_w=960,
        fps=None, footer=None) -> dict:
    """Write a reference | ours | 50/50 blend MP4. `ours` may be a video or a directory of PNG frames.

    Panels are scaled to the reference aspect (max `panel_w` wide). fps defaults to the reference's.
    """
    R, O = FrameSource(ref), FrameSource(ours)
    n = min(source_len(ref), source_len(ours)) if frames is None else int(frames)
    fps = fps or source_fps(ref)
    f0 = R.frame(0)
    W, H = fit_size(f0.shape[1], f0.shape[0], panel_w)
    bar_h = even(max(40, H // 9))
    px = max(12, H // 30)
    footer = footer or "same physical time in every panel, no retiming · visual similarity is not proof of physics"
    with VideoWriter(out, 3 * W, H + bar_h, fps) as vw:
        for i in range(n):
            a = cv2.resize(R.frame(i), (W, H), interpolation=cv2.INTER_AREA)
            b = cv2.resize(O.frame(i), (W, H), interpolation=cv2.INTER_AREA)
            row = np.hstack([a, b, cv2.addWeighted(a, 0.5, b, 0.5, 0)])
            row = burn_label(row, label_ref, (10, 8), px)
            row = burn_label(row, label_ours, (W + 10, 8), px)
            row = burn_label(row, "50/50 blend", (2 * W + 10, 8), px)
            bar = np.full((bar_h, 3 * W, 3), 30, np.uint8)
            bar = burn_label(bar, f"frame {i:04d}   t = {i / fps:7.3f} s   ·   {footer}", (10, (bar_h - px) // 2), px,
                             bg=(30, 30, 30), weight="Regular")
            vw.write(np.vstack([row, bar]))
    return {"out": str(out), "frames": n, "fps": fps, "size": [3 * W, H + bar_h]}


def sheet(video, out, times=None, label=None, cols=4, thumb_w=480, fps=None) -> dict:
    """Contact sheet of `video` at the given times (s); default 8 evenly spaced frames. Each tile is
    labelled with frame number and time."""
    S = FrameSource(video)
    fps = fps or source_fps(video)
    if times is None:
        idx = np.linspace(0, S.n - 1, 8).round().astype(int).tolist()
    else:
        idx = [min(int(round(t * fps)), S.n - 1) for t in times]
    f0 = S.frame(0)
    tw, th = fit_size(f0.shape[1], f0.shape[0], thumb_w)
    tiles = []
    for i in idx:
        t = cv2.resize(S.frame(i), (tw, th), interpolation=cv2.INTER_AREA)
        tiles.append(burn_label(t, f"f{i:04d}  t={i / fps:.2f}s", (6, 6), max(11, th // 18)))
    cols = max(1, min(cols, len(tiles)))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(len(tiles) // cols)])
    if label:
        head = np.full((max(32, th // 7), grid.shape[1], 3), 30, np.uint8)
        grid = np.vstack([burn_label(head, label, (10, 6), max(14, th // 14), bg=(30, 30, 30)), grid])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"out": str(out), "frames": idx, "size": [grid.shape[1], grid.shape[0]]}
