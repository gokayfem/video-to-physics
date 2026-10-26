"""Zoom-grid helper: crop.py <frame.png> <u0> <v0> <w> <h> <zoom> <out>; grid every 10 px, labels every 50 px."""
import cv2, sys
im = cv2.imread(sys.argv[1]); u0, v0, w, h, z = map(int, sys.argv[2:7])
c = cv2.resize(im[v0:v0 + h, u0:u0 + w], None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
for k in range(0, w, 10): cv2.line(c, (k * z, 0), (k * z, h * z), (0, 255, 0) if (u0 + k) % 50 == 0 else (0, 110, 0), 1)
for k in range(0, h, 10): cv2.line(c, (0, k * z), (w * z, k * z), (0, 255, 0) if (v0 + k) % 50 == 0 else (0, 110, 0), 1)
for k in range(0, w, 10):
    if (u0 + k) % 50 == 0: cv2.putText(c, str(u0 + k), (k * z + 2, 14), 0, 0.45, (0, 0, 255), 1)
for k in range(0, h, 10):
    if (v0 + k) % 50 == 0: cv2.putText(c, str(v0 + k), (2, k * z + 14), 0, 0.45, (0, 0, 255), 1)
cv2.imwrite(sys.argv[7], c)
