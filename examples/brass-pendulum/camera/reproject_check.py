"""Reprojection evidence (camera role).
For frames 0, 60, 120, 180, 240: re-measure the static silhouette edges (camera/edges.py, segment
endpoints mapped from frame 120 with the drift similarity), and compute the distance of every edge
point to the model edge (calibrated stand model, params/camera.json camera of that frame) for
  (a) the reference-matched camera (with drift) and (b) the locked camera (no drift).
Also overlays the bob-centre track predicted by the calibration fit vs measured.
Writes camera/out/reprojection_by_frame.json and visuals in camera/visuals/ (versioned)."""
import numpy as np, cv2, json, pathlib, sys, datetime
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0], "--phi0"]
import calibrate as CAL, edges as E, project as PJ
ROOT = CAL.ROOT
VER = "v003"
REF = ROOT.parents[1] / "reference-batch/clips/01-brass-pendulum.mp4"
CJ = json.load(open(ROOT / "params/camera.json"))
Lc = CJ["scale"]["L_eff_m"]
A = json.load(open(ROOT / "camera/out/calib_adopted.json"))
x = np.array(A["x"]); f0, cx0, cy0, rv, C_L, g, phi, e, Rb, th = CAL.unpack(x)
D = np.loadtxt(ROOT / "camera/out/drift_similarity_raw.csv", delimiter=",", skiprows=1)
FR = [0, 60, 120, 180, 240]
cap = cv2.VideoCapture(str(REF)); imgs = {}; n = 0
while True:
    ok, fimg = cap.read()
    if not ok: break
    if n in FR: imgs[n] = fimg
    n += 1

def model_line(name, cam):
    A_, d, n1, n2, rk = CAL.edge_defs(g)[name]
    r = abs(g[rk]) if rk else 0.0
    C = np.asarray(cam["C"]) / Lc
    m = CAL.tangent_plane_normal(A_, d, n1, n2, r, C)
    K = np.array([[cam["f_px"], 0, cam["cx"]], [0, cam["f_px"], cam["cy"]], [0, 0, 1.0]])
    l = np.linalg.inv(K).T @ (np.asarray(cam["R"]) @ m)
    return l / np.hypot(l[0], l[1])

def map_pt(n, u, v):
    s, rho, tx, ty = D[n, 2:6]; c, sn = np.cos(rho), np.sin(rho)
    return s*(c*u - sn*v) + tx, s*(sn*u + c*v) + ty

res = {}; meas = {}
for n in FR:
    Fi = E.feature_images(imgs[n]); res[n] = {}; meas[n] = {}
    for kind in ("reference", "locked"):
        cam = PJ.load_camera(frame=n, kind=kind); allr = []
        for name in CAL.USE_EDGES:
            if name not in meas[n]:
                u0, v0, u1, v1, s_, ch = E.EDGES[name]
                a = map_pt(n, u0, v0); b = map_pt(n, u1, v1)
                pts = E.sample_edge(Fi, a[0], a[1], b[0], b[1], s_ + 2, ch)
                pts = pts[(pts[:, 1] < 1088 - 8) & (pts[:, 1] > 8)]      # drop points at the frame border (edge leaves frame during the push-in)
                _, _, keep, _ = E.fit_line(pts); meas[n][name] = pts[keep]
            l = model_line(name, cam); P = meas[n][name]
            allr.append(P @ l[:2] + l[2])
        allr = np.concatenate(allr)
        res[n][kind] = {"rms_px": float(np.sqrt(np.mean(allr**2))), "max_px": float(np.abs(allr).max()), "n": int(len(allr))}
    print(n, res[n])
json.dump({"frames": FR, "edges": CAL.USE_EDGES, "result": {str(k): v for k, v in res.items()},
           "note": "distance of re-measured silhouette edge points to the calibrated model edges; frame 120 is the calibration frame"},
          open(ROOT / "camera/out/reprojection_by_frame.json", "w"), indent=1)

# ------------------------------------------------------------------ visuals
VIS = ROOT / "camera/visuals"; VIS.mkdir(exist_ok=True)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%MZ")
def label(img, lines, y0=30, scale=0.8):
    for i, t in enumerate(lines):
        y = y0 + int(i*34*scale)
        cv2.putText(img, t, (20, y), 0, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, t, (20, y), 0, scale, (255, 255, 255), 2, cv2.LINE_AA)

def draw_frame(n, kind, mag=20):
    img = imgs[n].copy(); cam = PJ.load_camera(frame=n, kind=kind)
    for name in CAL.USE_EDGES:
        l = model_line(name, cam); P = meas[n][name]
        # draw model line across the extent of the measured points (+-40 px)
        d = np.array([-l[1], l[0]]); t = P @ d; p0 = P[np.argmin(t)] - (P[np.argmin(t)] @ l[:2] + l[2])*l[:2]
        p1 = P[np.argmax(t)] - (P[np.argmax(t)] @ l[:2] + l[2])*l[:2]
        p0 -= 40*d; p1 += 40*d
        cv2.line(img, tuple(np.round((p0 - .5)*16).astype(int)), tuple(np.round((p1 - .5)*16).astype(int)), (255, 255, 0), 1, cv2.LINE_AA, 4)
        for p in P[::2]:
            r = p @ l[:2] + l[2]; q = p - r*l[:2]
            cv2.circle(img, tuple(np.round((p - .5)*16).astype(int)), 2*16, (0, 0, 255), -1, cv2.LINE_AA, 4)
            tip = q + mag*(p - q)
            cv2.line(img, tuple(np.round((q - .5)*16).astype(int)), tuple(np.round((tip - .5)*16).astype(int)), (0, 255, 255), 1, cv2.LINE_AA, 4)
    # pivot
    u, v = PJ.project(np.zeros((1, 3)), cam)
    cv2.drawMarker(img, (int(u[0]), int(v[0])), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
    ue, ve = PJ.project(np.array([[0, 0, -e*Lc]]), cam)
    cv2.drawMarker(img, (int(ue[0]), int(ve[0])), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
    return img

# V1: frame 120 calibration overlay + zooms
img = draw_frame(120, "reference")
# bob: measured centre track (f120 coords, all frames) vs model-predicted
Bm = np.loadtxt(ROOT / "camera/out/bob_meas_f120.csv", delimiter=",", skiprows=1)
cam120 = PJ.load_camera(frame=120, kind="locked")
th_ad = np.radians(A["theta_deg_frame"])
Bw = np.c_[np.sin(th_ad), np.zeros_like(th_ad), -np.cos(th_ad)]*Lc + [0, 0, -e*Lc]
bu, bv = PJ.project(Bw, cam120)
for k in range(len(Bm)):
    cv2.circle(img, (int(round((Bm[k, 2]-.5)*4)), int(round((Bm[k, 3]-.5)*4))), 4, (0, 0, 255), -1, cv2.LINE_AA, 2)
    cv2.circle(img, (int(round((bu[k]-.5)*4)), int(round((bv[k]-.5)*4))), 4, (255, 255, 0), -1, cv2.LINE_AA, 2)
fq = CJ["fit_quality"]
full = img.copy()
label(full, [f"{VER} camera calibration - reference frame 120 (t=5.000 s) - status: isolated component, draft",
             f"cyan = calibrated model silhouette edges, red = measured sub-pixel edge points, yellow = residual x20",
             f"static edge RMS {fq['static_edges_points_rms_px'] if 'static_edges_points_rms_px' in fq else fq['static_edge_points_rms_px']:.2f} px over {fq['n_static_edge_points']} points (R4 target <= 2 px)",
             f"green + = pivot P_ring (wire leaves loop), magenta x = P_eff (wire-line/arc rotation centre, {e:.3f} L below)",
             f"bob centres all 241 frames mapped to frame 120: red = measured (measurement/bob_track.csv), cyan = fitted circle model, rms {fq['bob_centre_rms_px']:.2f} px",
             f"f = {CJ['reference_camera']['intrinsics']['f_px']:.0f} px (45.6 mm @36 mm), swing plane {CJ['swing_plane']['angle_between_swing_plane_and_image_plane_deg']:.1f} deg to image plane"])
cv2.imwrite(str(VIS / f"{VER}_reprojection_f120_full.jpg"), cv2.resize(full, (1440, 816), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
cv2.imwrite(str(ROOT / "camera/out/landmarks.jpg"), cv2.resize(full, (1440, 816), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
# zoom panels
crops = [("post top + arm", (660, 30, 1140, 210)), ("base left", (460, 860, 760, 1088)), ("base right", (1180, 860, 1520, 1088)), ("bob arc", (940, 760, 1260, 900))]
panels = []
for name, (a, b, c, d) in crops:
    z = img[b:d, a:c]; s = 600/(c - a); z = cv2.resize(z, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    z = cv2.copyMakeBorder(z, 0, max(0, 380 - z.shape[0]), 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))[:380]
    label(z, [f"{VER} f120 zoom: {name}"], 24, 0.6); panels.append(z)
grid = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
cv2.imwrite(str(VIS / f"{VER}_reprojection_f120_zooms.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
# V2: drift - frames 0 and 240, reference-matched vs locked camera, zooms at base corners
rows = []
for n in (0, 240):
    row = []
    for kind in ("reference", "locked"):
        im = draw_frame(n, kind, mag=5)
        for (a, b, c, d) in [(440, 860, 760, 1088), (1200, 860, 1520, 1088), (640, 20, 1140, 210)]:
            z = cv2.resize(im[b:d, a:c], None, fx=420/(c - a), fy=420/(c - a), interpolation=cv2.INTER_CUBIC)
            z = cv2.copyMakeBorder(z, 0, max(0, 300 - z.shape[0]), 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))[:300]
            row.append(z)
        r_ = res[n][kind]
        label(row[-3], [f"{VER} frame {n} (t={n/24:.3f} s) {kind} camera", f"static RMS {r_['rms_px']:.2f} px, max {r_['max_px']:.1f} px"], 24, 0.55)
    rows.append(np.hstack(row))
g2 = np.vstack(rows)
hdr = np.full((60, g2.shape[1], 3), 30, np.uint8)
label(hdr, [f"{VER} camera drift check - status: isolated component, draft. Cyan model edges, red measured, yellow residual x5. Left 3: reference-matched (drift) camera; right 3: locked camera"], 36, 0.6)
cv2.imwrite(str(VIS / f"{VER}_drift_ref_vs_locked_f000_f240.jpg"), np.vstack([hdr, g2]), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("visuals written")
