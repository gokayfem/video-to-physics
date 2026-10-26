"""Track static stand features from reference frame 120 into every frame (pyramidal LK directly
120 -> n, forward-backward check < 0.3 px).  Features: Shi-Tomasi corners inside the stand
(post, arm, base) excluding the swing region (wire, bob, bob shadow) and the pin/ring.
Output camera/out/static_tracks.npz: pts (N_frames, N_feat, 2) contract px (nan = lost), ref_frame."""
import cv2, numpy as np, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
S = np.load(ROOT / "camera/frames/grey_stack.npy", mmap_mode="r")
REF = 120
g0 = np.ascontiguousarray(S[REF])
rb = cv2.imread(str(ROOT / "camera/frames/f_120.png")).astype(np.float32)
wood = cv2.GaussianBlur(rb[..., 2] - rb[..., 0], (0, 0), 2) > 30
mask = (cv2.dilate(wood.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0).astype(np.uint8) * 255
cv2.fillPoly(mask, [np.array([[1040, 70], [1140, 70], [1290, 1088], [860, 1088]], np.int32)], 0)  # swing/shadow region
mask[:, 1480:] = 0
pts = cv2.goodFeaturesToTrack(g0, 600, 0.01, 12, mask=mask, blockSize=9)
p0 = pts.reshape(-1, 2).astype(np.float32)
lk = dict(winSize=(31, 31), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01))
N = S.shape[0]; T = np.full((N, len(p0), 2), np.nan, np.float32)
for n in range(N):
    g = np.ascontiguousarray(S[n])
    if n == REF: T[n] = p0; continue
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g, p0, None, **lk)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(g, g0, p1, None, **lk)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(pb - p0, axis=1) < 0.3)
    T[n, ok] = p1[ok]
T += 0.5   # array index -> contract pixel-centre convention
np.savez(ROOT / "camera/out/static_tracks.npz", pts=T, ref_frame=REF)
good = np.isfinite(T[..., 0]).mean(0)
print("features", len(p0), "tracked in >90% frames:", (good > 0.9).sum())
d = T - T[REF]
print("median |disp| frame0 %.2f frame240 %.2f" % (np.nanmedian(np.linalg.norm(d[0], axis=1)), np.nanmedian(np.linalg.norm(d[240], axis=1))))
