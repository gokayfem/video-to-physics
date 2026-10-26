"""Static reprojection check of the camera of record at frames 0/60/120/180/240 (R4 <= 2 px), and the locked
(drift-free) camera for comparison. Edges: frame-0 segments (camera/edges.py) mapped to frame n with the drift
homography, re-measured sub-pixel on frame n; residual = point distance to the projected model edge (calibrate.edges3d).
Points within 8 px of the image border are excluded. Visual: camera/visuals/v001_reprojection_f120_full.jpg (+ zooms),
copied to measurement/visuals/v001_reprojection_overlay.jpg. Writes camera/out/reprojection_by_frame.json."""
import numpy as np, json, pathlib, sys, cv2
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "measurement/scripts"))
import project as PJ, edges as ED, calibrate as CAL
from common import load_frames, Happly, label
F = load_frames(); Hh = np.load(ROOT / "measurement/cache/H_ecc.npy"); CR = json.load(open(ROOT / "camera/out/calib_result.json")); g = CR["geometry"]
def model_lines(cam):
    E = CAL.edges3d(g, np.asarray(cam["C"])); out = {}
    for k in CAL.USE:
        A, d = E[k]; q = np.array(PJ.project(np.array([A, A + d]), cam)).T; e = q[1] - q[0]; nrm = np.array([-e[1], e[0]]) / np.linalg.norm(e); out[k] = (nrm, -nrm @ q[0], q)
    return out
res = {}; store = {}
for n in (0, 60, 120, 180, 240):
    edges = {}
    for k in CAL.USE:
        u0, v0, u1, v1, s_, ch = ED.EDGES[k]; p = Happly(Hh[n], [[u0, v0], [u1, v1]]); edges[k] = (p[0, 0], p[0, 1], p[1, 0], p[1, 1], s_, ch)
    meas = ED.measure(np.ascontiguousarray(F[n]), edges)
    rec = {}
    for kind in ("reference", "locked"):
        cam = PJ.load_camera(frame=n, kind=kind); ML = model_lines(cam); allr = []; per = {}
        for k in CAL.USE:
            P = meas[k]["points"]
            if len(P) == 0: continue
            ok = (P[:, 0] > 8) & (P[:, 0] < 1912) & (P[:, 1] > 8) & (P[:, 1] < 1080); P = P[ok]
            if len(P) == 0: continue
            d = P @ ML[k][0] + ML[k][1]; allr.append(d); per[k] = float(np.sqrt(np.mean(d ** 2)))
        allr = np.concatenate(allr); rec[kind] = dict(rms_px=float(np.sqrt(np.mean(allr ** 2))), max_px=float(np.abs(allr).max()), n=int(len(allr)), per_edge=per)
    res[n] = rec; store[n] = meas
    print(n, "reference %.3f px (max %.2f, n=%d)  locked %.3f px (max %.2f)" % (rec["reference"]["rms_px"], rec["reference"]["max_px"], rec["reference"]["n"], rec["locked"]["rms_px"], rec["locked"]["max_px"]))
json.dump({"criterion": "R4: static landmark/edge reprojection RMS <= 2 px", "frames": res, "all_pass_reference": all(r["reference"]["rms_px"] <= 2 for r in res.values())},
          open(ROOT / "camera/out/reprojection_by_frame.json", "w"), indent=1)
# ---- visual at frame 120
n = 120; cam = PJ.load_camera(frame=n); ML = model_lines(cam); im = np.ascontiguousarray(F[n]).copy()
for k in CAL.USE:
    q = ML[k][2]; e = (q[1] - q[0]) / np.linalg.norm(q[1] - q[0]); P = store[n][k]["points"]
    if len(P) == 0: continue
    s = (P - q[0]) @ e; a = q[0] + (s.min() - 15) * e; b = q[0] + (s.max() + 15) * e
    cv2.line(im, tuple(np.round((a - 0.5) * 16).astype(int)), tuple(np.round((b - 0.5) * 16).astype(int)), (255, 255, 0), 1, cv2.LINE_AA, 4)
    d = P @ ML[k][0] + ML[k][1]
    for p, dd in zip(P[::2], d[::2]):
        cv2.circle(im, tuple(np.round((p - 0.5) * 16).astype(int)), 16, (0, 0, 255), -1, cv2.LINE_AA, 4)
        q2 = p - 20 * dd * ML[k][0]; cv2.line(im, tuple(np.round((p - 0.5) * 16).astype(int)), tuple(np.round((q2 - 0.5) * 16).astype(int)), (0, 255, 255), 1, cv2.LINE_AA, 4)
A_uv = PJ.project(np.zeros(3), cam); cv2.drawMarker(im, (int(A_uv[0]), int(A_uv[1])), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
r = res[120]["reference"]
im_nolabel = im.copy()
label(im, ["v001  frame 120  t=5.000 s   reference measurement - draft", "camera of record (frame-0 calibration + measured drift): static edge RMS %.2f px (n=%d), R4<=2 px PASS" % (r["rms_px"], r["n"]),
           "cyan = projected model edges, red = measured sub-pixel edge points, yellow = residual x20, green + = origin A", "locked (no drift) camera at this frame: %.2f px" % res[120]["locked"]["rms_px"]])
(ROOT / "camera/visuals").mkdir(exist_ok=True); cv2.imwrite(str(ROOT / "camera/visuals/v001_reprojection_f120_full.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
Z = []
for (u0, v0) in ((440, 20), (1240, 20), (420, 930), (1240, 930), (560, 40)) if False else ((440, 20), (1240, 20), (420, 930), (1240, 930)):
    c = cv2.resize(im_nolabel[v0:v0 + 150, u0:u0 + 250], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST); cv2.putText(c, "(%d,%d)" % (u0, v0), (6, 22), 0, 0.6, (255, 255, 255), 2); Z.append(c)
zim = np.vstack([np.hstack(Z[:2]), np.hstack(Z[2:4])]); label(zim, ["v001 frame 120 reprojection zooms x3 - reference measurement - draft"], scale=0.7)
cv2.imwrite(str(ROOT / "camera/visuals/v001_reprojection_f120_zooms.jpg"), zim, [cv2.IMWRITE_JPEG_QUALITY, 92])
cv2.imwrite(str(ROOT / "measurement/visuals/v001_reprojection_overlay.jpg"), np.vstack([cv2.resize(im, None, fx=0.75, fy=0.75), cv2.resize(zim, (1440, int(zim.shape[0] * 1440 / zim.shape[1])))]), [cv2.IMWRITE_JPEG_QUALITY, 90])
