"""Static reprojection check (R4 <= 2 px) of the camera of record (clip 03, orbiting camera).
Model: the 7 identity-stable frame edges triangulated by camera/pose_bundle.py (camera/out/bundle_lines.json) + origin A.
Measured: sub-pixel edge points tracked in every frame (measurement/cache/edges_track.json, every 3rd point) and the wire-top pixel.
Residual = point distance to the projected model line. Reported for frames 0/60/120/180/240 (and all frames), camera of record vs the
locked frame-0 camera. Points within 8 px of the border excluded. Visual: measurement/visuals/v002_reprojection_overlay.jpg (frame 240).
Writes camera/out/reprojection_by_frame.json."""
import numpy as np, json, pathlib, sys, cv2, csv
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "measurement/scripts"))
import project as PJ
from common import load_frames, label
F = load_frames(); T = json.load(open(ROOT / "measurement/cache/edges_track.json")); BL = json.load(open(ROOT / "camera/out/bundle_lines.json"))
WT = list(csv.DictReader(open(ROOT / "measurement/wire_track.csv")))
def model_line(cam, k):
    X0 = np.array(BL[k]["point"]); d = np.array(BL[k]["direction"]); q = np.array(PJ.project(np.array([X0 - 0.8 * d, X0 + 0.8 * d]), cam)).T
    e = q[1] - q[0]; nrm = np.array([-e[1], e[0]]) / np.linalg.norm(e); return nrm, -nrm @ q[0], q
def frame_res(n, kind):
    cam = PJ.load_camera(frame=n, kind=kind); allr = []; per = {}
    for k in BL:
        v = T[n][k]
        if v["n_pts"] < 10 or v["rms"] > 1.5: continue
        P = np.array(v["pts"]); ok = (P[:, 0] > 8) & (P[:, 0] < 1912) & (P[:, 1] > 8) & (P[:, 1] < 1080); P = P[ok]
        nr, c, _ = model_line(cam, k); d = P @ nr + c; allr.append(d); per[k] = float(np.sqrt(np.mean(d ** 2)))
    a = np.array(PJ.project(np.zeros(3), cam)) - [float(WT[n]["u_top_px"]), float(WT[n]["v_top_px"])]
    allr = np.concatenate(allr); return dict(rms_px=float(np.sqrt(np.mean(allr ** 2))), max_px=float(np.abs(allr).max()), n=int(len(allr)), per_edge=per, A_resid_px=a.tolist())
res = {}
for n in (0, 60, 120, 180, 240):
    res[n] = {kind: frame_res(n, kind) for kind in ("reference", "locked")}
    print(n, "reference %.3f px (max %.2f, n=%d, A %s)  locked %.2f px" % (res[n]["reference"]["rms_px"], res[n]["reference"]["max_px"], res[n]["reference"]["n"], np.round(res[n]["reference"]["A_resid_px"], 2), res[n]["locked"]["rms_px"]))
allf = [frame_res(n, "reference")["rms_px"] for n in range(len(T))]
json.dump({"criterion": "R4: static edge reprojection RMS <= 2 px", "model": "bundle-triangulated identity-stable frame edges + A", "frames": res,
           "all_frames_rms_px": {"median": float(np.median(allf)), "max": float(np.max(allf)), "frame_of_max": int(np.argmax(allf))},
           "all_pass_reference": bool(max(allf) <= 2)}, open(ROOT / "camera/out/reprojection_by_frame.json", "w"), indent=1)
print("all frames: median %.3f max %.3f (frame %d)" % (np.median(allf), max(allf), int(np.argmax(allf))))
n = 240; cam = PJ.load_camera(frame=n); im = np.ascontiguousarray(F[n]).copy(); camL = PJ.load_camera(frame=n, kind="locked")
for k in BL:
    nr, c, q = model_line(cam, k); _, _, qL = model_line(camL, k); v = T[n][k]; P = np.array(v["pts"])
    if len(P) == 0: continue
    e = (q[1] - q[0]) / np.linalg.norm(q[1] - q[0]); s_ = (P - q[0]) @ e; a = q[0] + (s_.min() - 25) * e; b = q[0] + (s_.max() + 25) * e
    eL = (qL[1] - qL[0]) / np.linalg.norm(qL[1] - qL[0]); sL = (P - qL[0]) @ eL; aL = qL[0] + (sL.min() - 25) * eL; bL = qL[0] + (sL.max() + 25) * eL
    cv2.line(im, tuple(np.round((aL - 0.5) * 16).astype(int)), tuple(np.round((bL - 0.5) * 16).astype(int)), (255, 0, 255), 2, cv2.LINE_AA, 4)
    cv2.line(im, tuple(np.round((a - 0.5) * 16).astype(int)), tuple(np.round((b - 0.5) * 16).astype(int)), (255, 255, 0), 1, cv2.LINE_AA, 4)
    d = P @ nr + c
    for p, dd in zip(P, d):
        cv2.circle(im, tuple(np.round((p - 0.5) * 16).astype(int)), 24, (0, 0, 255), -1, cv2.LINE_AA, 4)
        q2 = p - 20 * dd * nr; cv2.line(im, tuple(np.round((p - 0.5) * 16).astype(int)), tuple(np.round((q2 - 0.5) * 16).astype(int)), (0, 255, 255), 1, cv2.LINE_AA, 4)
A_uv = PJ.project(np.zeros(3), cam); cv2.drawMarker(im, (int(A_uv[0]), int(A_uv[1])), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
r = res[n]["reference"]; im_nl = im.copy()
label(im, ["v002  frame 240  t=10.000 s   reference measurement - draft", "camera of record (per-frame orbit pose): static edge RMS %.2f px (n=%d), R4 <= 2 px %s; all frames median %.2f / max %.2f px" % (r["rms_px"], r["n"], "PASS" if max(allf) <= 2 else "FAIL", np.median(allf), max(allf)),
           "cyan = projected bundle edges, red = measured sub-pixel edge points, yellow = residual x20, green + = origin A", "magenta = same edges through the LOCKED frame-0 camera (%.1f px rms): the camera orbits" % res[n]["locked"]["rms_px"]], scale=0.75)
Z = []
for (u0, v0) in ((380, 20), (780, 120), (540, 300), (900, 60)):
    c = cv2.resize(im_nl[v0:v0 + 160, u0:u0 + 260], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST); cv2.putText(c, "(%d,%d) x3" % (u0, v0), (6, 22), 0, 0.6, (255, 255, 255), 2); Z.append(c)
zim = np.vstack([np.hstack(Z[:2]), np.hstack(Z[2:4])])
cv2.imwrite(str(ROOT / "measurement/visuals/v002_reprojection_overlay.jpg"), np.vstack([cv2.resize(im, None, fx=0.75, fy=0.75), cv2.resize(zim, (1440, int(zim.shape[0] * 1440 / zim.shape[1])))]), [cv2.IMWRITE_JPEG_QUALITY, 90])
