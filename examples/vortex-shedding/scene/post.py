"""Camera-like finishing: radial vignette (cos^4-like) + fine luminance grain. Deterministic (fixed seed).
usage: python scene/post.py raw.png out.png [frame]"""
import json, os, sys, cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pp = json.load(open(os.path.join(ROOT, "config", "scene_params.json")))["look"]["post"]
im = cv2.imread(sys.argv[1]).astype(np.float32); H, W = im.shape[:2]
y, x = np.mgrid[0:H, 0:W]; r2 = ((x + 0.5 - W / 2) ** 2 + (y + 0.5 - H / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)
im *= (1 - pp["vignette_strength"] * r2 ** 1.2)[..., None]
frame = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # per-frame grain (frame 0 identical to the first frame)
g = np.random.default_rng(pp["seed"] + frame).normal(0, pp["grain_sigma_8bit"], (H, W)).astype(np.float32)
g = cv2.GaussianBlur(g, (0, 0), 0.6) * 1.6
im += g[..., None]
cv2.imwrite(sys.argv[2], np.clip(im + 0.5, 0, 255).astype(np.uint8))
