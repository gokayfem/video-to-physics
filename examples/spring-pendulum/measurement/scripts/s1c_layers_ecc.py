"""Stage 1c: per-layer ECC affine (frame 0 -> n) on backdrop-only, frame-only and table-only masks; reports the
isotropic scale (sqrt det) per layer. Backdrop has little texture (gradient + faint vignette), so its scale is the
weakest; used only to bound a dolly."""
import cv2, numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *
F = load_frames(); H = np.load(os.path.join(CACHE, "H_ecc.npy"))
def mask(k):
    m = np.zeros((1088, 1920), np.uint8)
    if k == "backdrop": m[30:800, 20:460] = 255; m[30:800, 1460:1900] = 255
    if k == "frame": m[40:1000, 490:625] = 255; m[40:1000, 1295:1425] = 255; m[60:120, 620:1300] = 255
    if k == "table": m[870:1075, 20:460] = 255; m[870:1075, 1460:1900] = 255
    return m
T = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1.0]]); Ti = np.linalg.inv(T)
g0 = cv2.GaussianBlur(luma(F[0]) / 255, (0, 0), 1.5).astype(np.float32); res = {}
for n in [60, 120, 180, 240]:
    gn = cv2.GaussianBlur(luma(F[n]) / 255, (0, 0), 1.5).astype(np.float32); res[n] = {}
    for k in ["frame", "table", "backdrop"]:
        Hc = Ti @ H[n] @ T; W = (Hc / Hc[2, 2])[:2].astype(np.float32)
        try:
            cc, W = cv2.findTransformECC(g0, gn, W, cv2.MOTION_AFFINE, (3, 300, 1e-8), mask(k), 5)
        except cv2.error: cc = np.nan
        s = float(np.sqrt(abs(np.linalg.det(W[:, :2])))); res[n][k] = round(s, 5)
    print(n, res[n])
json.dump(res, open(os.path.join(MEAS, "drift_layers_ecc.json"), "w"), indent=1)
