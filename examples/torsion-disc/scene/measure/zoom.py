"""Zoomed reference crop with a pixel grid for reading landmarks (scene role C).
.venv/bin/python scene/measure/zoom.py <frame> u0 v0 u1 v1 <scale> <out.png> [grid_step]"""
import sys, pathlib, cv2, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
REF = ROOT.parents[1] / "reference-batch/clips/03-torsion-disc.mp4"
n, u0, v0, u1, v1 = map(int, sys.argv[1:6]); s = float(sys.argv[6]); out = sys.argv[7]
step = int(sys.argv[8]) if len(sys.argv) > 8 else 10
cap = cv2.VideoCapture(str(REF)); cap.set(cv2.CAP_PROP_POS_FRAMES, n); ok, f = cap.read(); assert ok
c = cv2.resize(f[v0:v1, u0:u1], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST if s >= 2 else cv2.INTER_AREA)
for u in range((u0 // step + 1) * step, u1, step):
    x = int((u - u0) * s); big = u % (step * 5) == 0
    cv2.line(c, (x, 0), (x, 8 if not big else 16), (0, 255, 255), 1)
    if big: cv2.putText(c, str(u), (x + 2, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
for v in range((v0 // step + 1) * step, v1, step):
    y = int((v - v0) * s); big = v % (step * 5) == 0
    cv2.line(c, (0, y), (8 if not big else 16, y), (0, 255, 255), 1)
    if big: cv2.putText(c, str(v), (18, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
cv2.imwrite(out, c)
