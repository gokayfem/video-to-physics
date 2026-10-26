"""Compose reference | render | 50/50 blend side-by-side MP4 at true 24 fps with burned-in labels.
usage: compose_sbs.py RENDER_DIR OUT.mp4 "label for render" """
import subprocess, sys, cv2, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
REF = ROOT.parents[1] / 'reference-batch/clips/01-brass-pendulum.mp4'
rdir, out, lab = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
W, H = 960, 544
cap = cv2.VideoCapture(str(REF))
ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W*3}x{H+60}',
                       '-r', '24', '-i', '-', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
def txt(img, s, x, y, sc=0.55):
    (w, h), b = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)
    cv2.rectangle(img, (x - 4, y - h - 5), (x + w + 4, y + b + 3), (25, 25, 25), -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), 1, cv2.LINE_AA)
for n in range(241):
    ok, ref = cap.read(); assert ok
    ref = cv2.resize(ref, (W, H), interpolation=cv2.INTER_AREA)
    ren = cv2.resize(cv2.imread(str(rdir / f'f{n:04d}.png')), (W, H), interpolation=cv2.INTER_AREA)
    mix = cv2.addWeighted(ref, 0.5, ren, 0.5, 0)
    row = np.hstack([ref, ren, mix])
    bar = np.full((60, W * 3, 3), 30, np.uint8)
    txt(row, 'REFERENCE (AI clip, observed)', 10, 24); txt(row, lab, W + 10, 24); txt(row, '50/50 blend', 2 * W + 10, 24)
    txt(bar, f'frame {n:03d}  t = {n/24:6.3f} s  (same physical time in all panels, no retiming)', 10, 24, 0.6)
    txt(bar, 'Simulation = passive physical pendulum (fitted, Q assumed). Visual similarity is not proof of physics.', 10, 50, 0.5)
    ff.stdin.write(np.vstack([row, bar]).tobytes())
ff.stdin.close(); ff.wait(); print(out)
