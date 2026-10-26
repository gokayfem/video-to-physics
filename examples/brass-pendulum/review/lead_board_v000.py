"""Lead visual v000: current observed state of clip 01 (no model fit yet)."""
import csv, cv2, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
REF = ROOT.parents[1] / 'reference-batch/clips/01-brass-pendulum.mp4'
cap = cv2.VideoCapture(str(REF)); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
assert len(frames) == 241
F = np.stack(frames).astype(np.float32)

def label(img, lines, y0=34):
    cv2.rectangle(img, (0, 0), (img.shape[1], y0 + 34 * (len(lines) - 1) + 14), (30, 30, 30), -1)
    for i, s in enumerate(lines):
        y = y0 + i * 34
        cv2.putText(img, s, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, s, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return img

# A: frame 0 with static landmarks tracked by the measurement specialist (frame 0 positions)
a = frames[0].copy()
rows = [r for r in csv.DictReader(open(ROOT / 'measurement/static_landmarks.csv')) if r['frame'] == '0']
for r in rows:
    u, v = float(r['u_px']), float(r['v_px'])
    cv2.circle(a, (round(u), round(v)), 9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(a, r['landmark'], (round(u) + 12, round(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
label(a, ['A  REFERENCE frame 0 (t=0.000 s) - observed', 'yellow: static landmarks tracked by measurement (draft)'])

# B: temporal composite over all 241 frames: per-pixel darkest value shows the swept bob/wire range
b = F.min(axis=0).astype(np.uint8)
b = cv2.addWeighted(b, 0.65, frames[0], 0.35, 0)
label(b, ['B  REFERENCE all 241 frames, darkest-pixel composite - observed', 'shows the full swept range of wire and bob (no model)'])

def fit(img, w=1920, h=1088):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
c = fit(cv2.imread(str(ROOT / 'camera/out/drift_lines.png')))
c = cv2.copyMakeBorder(c, 90, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255)); c = fit(c)
label(c, ['C  CAMERA drift check (camera specialist, draft)', 'image shift/scale vs mid-clip: several px + ~1% scale: under review'])
d = fit(cv2.imread(str(ROOT / 'solver/out/selfcheck.png')))
d = cv2.copyMakeBorder(d, 90, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255)); d = fit(d)
label(d, ['D  SOLVER self-check on PLACEHOLDER params (not fitted)', 'isolated component - draft; independent verification pending'])
board = np.vstack([np.hstack([a, b]), np.hstack([c, d])])
board = cv2.resize(board, (1920, 1088), interpolation=cv2.INTER_AREA)
cv2.rectangle(board, (0, 1048), (1920, 1088), (30, 30, 30), -1)
cv2.putText(board, 'Sim 01 brass pendulum | lead board v000 | 2026-10-26 | status: building - no fitted model, no Blender render yet',
            (14, 1076), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
out = ROOT / 'review/visuals/v000_lead_board.jpg'
cv2.imwrite(str(out), board, [cv2.IMWRITE_JPEG_QUALITY, 90]); print(out)
