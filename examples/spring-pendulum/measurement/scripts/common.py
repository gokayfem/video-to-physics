"""Shared helpers for clip-02 reference measurement (adapted from clip-01 measurement/scripts/common.py)."""
import cv2, numpy as np, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MEAS = os.path.join(ROOT, "measurement")
VIDEO = os.path.abspath(os.path.join(ROOT, "..", "..", "reference-batch", "clips", "02-vertical-spring.mp4"))
VIDEO_SHA = "fee726c2c30d36e5cf403a7152ff8a02d100682ee4333e1a495aa4746f6865cd"
CACHE = os.path.join(MEAS, "cache")
FPS = 24.0
N = 241
VERSION = "v001"
STATUS = "reference measurement - draft"

def load_frames():
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, "frames_bgr.npy")
    if os.path.exists(p):
        return np.load(p, mmap_mode="r")
    cap = cv2.VideoCapture(VIDEO); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    fr = np.array(fr); assert fr.shape == (241, 1088, 1920, 3), fr.shape
    np.save(p, fr); return np.load(p, mmap_mode="r")

def luma(f):
    return cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_BGR2GRAY).astype(np.float32)

MOVING_POLY = np.array([[915, 10], [1000, 10], [1060, 170], [1175, 740], [1175, 1020], [745, 1020], [745, 740], [870, 170]], np.int32)

def static_mask(shape=(1088, 1920)):
    """255 where the scene is static (frame-0 geometry): excludes spring + hook + weight sweep + weight shadow
    region with generous margins, and an 8 px border."""
    m = np.full(shape, 255, np.uint8)
    cv2.fillPoly(m, [MOVING_POLY], 0)
    m[:8] = 0; m[-8:] = 0; m[:, :8] = 0; m[:, -8:] = 0
    return m

def bilinear(img, u, v):
    """Sample img at contract pixel coords (u,v) (pixel centres at +0.5), bilinear, replicate border."""
    sh = np.shape(u); x = np.asarray(u, np.float32).ravel() - 0.5; y = np.asarray(v, np.float32).ravel() - 0.5
    K = 1024; m = int(np.ceil(len(x) / K)) * K
    xp = np.zeros(m, np.float32); yp = np.zeros(m, np.float32); xp[:len(x)] = x; yp[:len(y)] = y
    out = cv2.remap(np.asarray(img, np.float32), xp.reshape(-1, K), yp.reshape(-1, K), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.ravel()[:len(x)].reshape(sh)

def Happly(H, P):
    P = np.atleast_2d(np.asarray(P, float)); q = np.c_[P, np.ones(len(P))] @ H.T
    return q[:, :2] / q[:, 2:3]

def label(img, lines, org=(12, 30), scale=0.8):
    """Burn in labels (version / frame-time / status) with a dark outline."""
    y = org[1]
    for s in lines:
        cv2.putText(img, s, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, s, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += int(34 * scale)
    return img
