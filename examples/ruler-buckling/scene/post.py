"""Camera-like finishing (deterministic): mild radial vignette + fine luminance grain. Geometry untouched.
python scene/post.py scene/out/frame0_raw.png firstframe/frame0.png"""
import json, pathlib, sys
import cv2, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
Q = json.loads((ROOT / "config/scene_params.json").read_text())["post"]
img = cv2.imread(sys.argv[1]).astype(np.float32)
H, W = img.shape[:2]
y, x = np.mgrid[0:H, 0:W].astype(np.float32)
r2 = ((x + 0.5 - W / 2) ** 2 + (y + 0.5 - H / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)
img *= (1.0 - (1.0 - Q["vignette_corner_gain"]) * r2 ** 1.5)[..., None]
g = np.random.default_rng(Q["seed"]).normal(0, Q["grain_sigma_8bit"], (H, W)).astype(np.float32)
img += cv2.GaussianBlur(g, (0, 0), 0.6)[..., None] * 1.3
cv2.imwrite(sys.argv[2], np.clip(np.rint(img), 0, 255).astype(np.uint8))
print("post ok", W, H)
