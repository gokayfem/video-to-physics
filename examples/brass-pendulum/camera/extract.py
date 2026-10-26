"""Read all 241 reference frames (read-only), save the temporal median (static plate) and a
downsampled grey stack used for drift analysis."""
import cv2, numpy as np, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = ROOT.parents[1] / "reference-batch/clips/01-brass-pendulum.mp4"
cap = cv2.VideoCapture(str(REF)); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
A = np.stack(frames)
print("frames", A.shape)
med = np.median(A, axis=0).astype(np.uint8)
out = ROOT / "camera/frames"; out.mkdir(exist_ok=True)
cv2.imwrite(str(out / "median.png"), med)
np.save(out / "grey_stack.npy", np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]))
