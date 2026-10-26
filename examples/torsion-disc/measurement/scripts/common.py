"""Shared helpers for clip-03 reference measurement (adapted from clip-01 measurement/scripts/common.py)."""
import cv2, numpy as np, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MEAS = os.path.join(ROOT, "measurement")
VIDEO = os.path.abspath(os.path.join(ROOT, "..", "..", "reference-batch", "clips", "03-torsion-disc.mp4"))
VIDEO_SHA = "c640e5ce848f7a86c0c6d6a2156aa4d89c022c51ba164b5b119bef16b0a2045c"
CACHE = os.path.join(MEAS, "cache")
FPS = 24.0
N = 241
VERSION = "v002"
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

# clip 03: wire (below the clamp) + disc sweep + disc shadow on the table, generous margins (frame-0 px)
MOVING_POLY = np.array([[925, 185], [1005, 185], [1015, 540], [1300, 540], [1320, 1030], [630, 1030], [640, 540], [915, 540]], np.int32)

def static_mask(shape=(1088, 1920)):
    """255 where the scene is static (frame-0 geometry): excludes wire + disc sweep + disc shadow region with generous margins, and an 8 px border."""
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
