"""Shared helpers for clip-01 reference measurement (measurement/ role)."""
import cv2, numpy as np, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MEAS = os.path.join(ROOT, "measurement")
VIDEO = os.path.abspath(os.path.join(ROOT, "..", "..", "reference-batch", "clips", "01-brass-pendulum.mp4"))
VIDEO_SHA = "ced22d105d7bf9e93d7c152fab02b62e077cbe54162056114201c50b61f56bae"
CACHE = os.path.join(MEAS, "cache")
FPS = 24.0

def load_frames():
    """Decode all frames with OpenCV (BGR uint8). Cached as .npy in measurement/cache (not a deliverable)."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, "frames_bgr.npy")
    if os.path.exists(p):
        return np.load(p, mmap_mode="r")
    cap = cv2.VideoCapture(VIDEO)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    fr = np.array(fr)
    assert fr.shape == (241, 1088, 1920, 3), fr.shape
    np.save(p, fr)
    return np.load(p, mmap_mode="r")

def luma(f):
    return cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_BGR2GRAY).astype(np.float32)

def static_mask(shape=(1088, 1920)):
    """255 where the scene is expected static: excludes the swing region (wire sweep, bob, bob shadow)
    and the wire loop on the pin. Pixel coords, frame-0 geometry (generous margins)."""
    m = np.full(shape, 255, np.uint8)
    cv2.fillPoly(m, [np.array([[1062, 112], [1112, 112], [1260, 1000], [890, 1000]], np.int32)], 0)
    cv2.rectangle(m, (1068, 74), (1098, 132), 0, -1)
    return m

def bilinear(img, u, v):
    """Sample img at pixel coords (u,v) with the contract convention (pixel centres at +0.5)."""
    x = np.asarray(u, np.float32) - 0.5
    y = np.asarray(v, np.float32) - 0.5
    return cv2.remap(img, x.reshape(-1, 1), y.reshape(-1, 1), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE).reshape(np.shape(u))
