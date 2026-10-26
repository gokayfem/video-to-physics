"""Stage 10: visuals (measurement/visuals/v002_*), labelled version / frame-time / 'reference measurement - draft'."""
import numpy as np, cv2, json, os, sys, csv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "camera"))
from common import *
import camfr, project as PJ
from scipy.ndimage import median_filter
VIS = os.path.join(MEAS, "visuals"); os.makedirs(VIS, exist_ok=True)
F = load_frames(); A = np.load(os.path.join(CACHE, "analysis.npz")); AN = json.load(open(os.path.join(MEAS, "analysis.json"))); AW = json.load(open(os.path.join(MEAS, "anchor_window.json")))
POSE = np.load(os.path.join(CACHE, "disc_pose.npy")); E = np.load(os.path.join(CACHE, "disc_edges.npy")); apex = np.load(os.path.join(CACHE, "apex_offset.npy"))
WT = list(csv.DictReader(open(os.path.join(MEAS, "wire_track.csv")))); t = A["t"]; psi = A["psi"]; N = len(t); Ta = AW["T_a_s"]
import s3_disc_pose as S3
def P16(p): return tuple(np.round((np.asarray(p) - 0.5) * 16).astype(int))
def poly(im, pts, col, th=1, closed=True): cv2.polylines(im, [np.round((np.asarray(pts) - 0.5) * 16).astype(np.int32)], closed, col, th, cv2.LINE_AA, 4)
# ---------- 1. frame 0 annotated
n = 0; im = np.ascontiguousarray(F[n]).copy(); k = camfr.cam(n); x = POSE[n, 2:7]; h = POSE[n, 17]
a, b = S3.circles(x, h, k); poly(im, a, (0, 255, 0), 2); poly(im, b, (255, 128, 0), 1)
for p in E[n][::6]:
    if np.isfinite(p[0]): cv2.circle(im, P16(p), 24, (0, 0, 255), -1, cv2.LINE_AA, 4)
nn = S3.normal(*x[3:5]); ex, ey = S3.basis(nn); Q = A["Q"]
d = np.cos(np.radians(psi[n])) * ex + np.sin(np.radians(psi[n])) * ey; L = camfr.proj(Q[n] + np.outer(np.linspace(-0.5, 0.5, 50), d), k); poly(im, L, (255, 0, 255), 2, False)
W0 = WT[n]; cv2.line(im, P16([float(W0["u_top_px"]), float(W0["v_top_px"])]), P16([float(W0["u_bot_px"]), float(W0["v_bot_px"])]), (0, 255, 255), 1, cv2.LINE_AA, 4)
for key, col in (("top", (0, 255, 255)), ("bot", (0, 200, 255))): cv2.drawMarker(im, tuple(int(v) for v in (float(W0["u_%s_px" % key]), float(W0["v_%s_px" % key]))), col, cv2.MARKER_TILTED_CROSS, 16, 2)
trk = np.array([camfr.proj(Q[i], k) for i in range(N)]); poly(im, trk, (255, 255, 255), 1, False)
for i in range(0, N, 24): cv2.circle(im, P16(trk[i]), 40, (255, 255, 255), -1, cv2.LINE_AA, 4)
A_uv = camfr.proj(np.zeros(3), k); cv2.drawMarker(im, tuple(int(v) for v in A_uv), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
# inset: 4x zoom of the centre track
c = trk.mean(0); u0, v0 = int(c[0] - 60), int(c[1] - 40); z = cv2.resize(im[v0:v0 + 80, u0:u0 + 120], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
im[1088 - 330:1088 - 10, 1920 - 490:1920 - 10] = z[:320, :480]; cv2.rectangle(im, (1920 - 490, 1088 - 330), (1920 - 10, 1088 - 10), (255, 255, 255), 1)
cv2.putText(im, "x4: disc rotation-centre track 0-10 s (frame-0 view)", (1920 - 485, 1088 - 336), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
label(im, ["v002  frame 0  t=0.000 s   reference measurement - draft",
           "green = fitted top-face circle (radius 0.5 D), orange = bottom face (thickness %.3f D), red = silhouette points; silhouette rms %.2f px" % (h, POSE[n, 12]),
           "magenta = engraved line, psi(0) = %.1f deg (world azimuth); yellow = wire, x = wire top (A) / bottom; green + = A" % psi[0],
           "white = disc rotation-centre track over all 241 frames projected into frame 0 (dots every 1 s); sway amp x %.3f D, y %.3f D, period %.3f s" % (AN["sway"]["Qx_D"]["amp"], AN["sway"]["Qy_D"]["amp"], AN["sway"]["Qx_D"]["period_s"])], scale=0.7)
cv2.imwrite(os.path.join(VIS, "v002_annotated_f000.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
# ---------- 2. rectified strip
C = np.load(os.path.join(CACHE, "rect_cart.npy"), mmap_mode="r"); tiles = []
for i in range(0, N, 10):
    r = cv2.cvtColor(cv2.resize(np.ascontiguousarray(C[i]), (300, 300)), cv2.COLOR_GRAY2BGR); cc = np.array([150, 150]) + np.array([apex[0], -apex[1]]) * 300
    dd = np.array([np.cos(np.radians(psi[i])), -np.sin(np.radians(psi[i]))])
    cv2.line(r, tuple((cc - 150 * dd).astype(int)), tuple((cc + 150 * dd).astype(int)), (255, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(r, "f%d t=%.2fs" % (i, t[i]), (4, 16), 0, 0.45, (0, 0, 255), 1, cv2.LINE_AA); cv2.putText(r, "psi %.1f" % psi[i], (4, 292), 0, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    tiles.append(r)
while len(tiles) % 5: tiles.append(np.zeros_like(tiles[0]))
grid = np.vstack([np.hstack(tiles[j:j + 5]) for j in range(0, len(tiles), 5)])
pol = np.load(os.path.join(CACHE, "rect_polar.npy"), mmap_mode="r"); RR = np.linspace(0.06, 0.48, 85); band = (RR > 0.15) & (RR < 0.45)
kym = np.array([(lambda p: p - median_filter(p, 31, mode="wrap"))(np.asarray(pol[i])[band].mean(0)) for i in range(N)])
kym = np.clip(128 + 6 * kym, 0, 255).astype(np.uint8); kym = cv2.cvtColor(cv2.resize(kym, (1500, 482), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
for i in range(N):
    for half in (0, 180):
        th = (psi[i] + half) % 360; cv2.circle(kym, (int(th / 360 * 1500), int(i * 2)), 1, (255, 0, 255), -1)
for s_ in range(0, 361, 45): cv2.putText(kym, "%d" % s_, (int(s_ / 360 * 1500) + 2, 476), 0, 0.4, (0, 255, 0), 1)
for s_ in range(0, 11, 2): cv2.putText(kym, "%ds" % s_, (2, int(s_ * 24 * 2) + 12), 0, 0.4, (0, 255, 0), 1)
hdr = np.zeros((70, 1500, 3), np.uint8)
label(hdr, ["v002  frames 0-240 (t=0-10 s)   reference measurement - draft", "top: disc rectified top-down (D=600 px, world-azimuth frame, +X right, +Y up), every 10 frames, magenta = measured psi line; bottom: angular profile (theta 0-360, line darkness) vs time, magenta = psi, psi+180"], org=(8, 24), scale=0.55)
cv2.imwrite(os.path.join(VIS, "v002_rectified_disc_strip.jpg"), np.vstack([hdr, grid, kym]), [cv2.IMWRITE_JPEG_QUALITY, 90])
# ---------- 3. psi(t), sway, T_a
fig, ax = plt.subplots(5, 1, figsize=(13, 17), sharex=True)
m = AN["psi"]["models"]; so = m["slow_oscillation_c+A_sin(2pi t/T+phi)"]["p"]; cu = m["cubic"]["p"]
ax[0].plot(t, psi, "k.", ms=3, label="psi (mean of A,B)"); ax[0].plot(t, so[0] + so[1] * np.sin(2 * np.pi * t / so[2] + so[3]), "r-", lw=1, label="slow-oscillation fit T=%.1f s, A=%.0f deg (rms %.2f)" % (so[2], abs(so[1]), m["slow_oscillation_c+A_sin(2pi t/T+phi)"]["rms_deg"]))
ax[0].plot(t, cu[0] + cu[1] * t + cu[2] * t ** 2 + cu[3] * t ** 3, "b--", lw=1, label="cubic (rms %.2f)" % m["cubic"]["rms_deg"])
ax[0].set_ylabel("twist psi [deg]"); ax[0].legend(fontsize=8); ax[0].set_title("v002 reference measurement - draft | clip 03 torsion disc | twist psi(t): NO turning point in 0-10 s (monotonic clockwise spin, %.0f deg = %.2f turns)" % (AN["psi"]["total_rotation_deg"], AN["psi"]["turns"]), fontsize=10)
ax[1].plot(t, A["vs"], "k-", label="rate (8-frame mean)"); rb = AN["psi"]["rate_by_second_deg_s"]; ax[1].step(np.arange(11), rb + [rb[-1]], where="post", color="r", label="mean rate per second")
ax[1].set_ylabel("dpsi/dt [deg/s]"); ax[1].legend(fontsize=8)
ax[2].plot(t, A["pA"] - A["pB"], "g.", ms=3, label="A - B (rms %.2f deg)" % AN["psi_methods"]["A_minus_B_rms_deg"]); ax[2].plot(t, A["sig"], "k-", lw=1, label="sigma_psi per frame"); ax[2].plot(t, -A["sig"], "k-", lw=1)
nm = np.arange(N) % 4; ax[2].plot(t[nm == 0], A["res7"][nm == 0], "rx", ms=3, label="psi - smooth, n%4==0 (cadence)"); ax[2].plot(t, A["kinkA"] / 5, "c-", lw=0.7, label="line kink A / 5")
ax[2].set_ylabel("deg"); ax[2].legend(fontsize=7, ncol=2); ax[2].set_ylim(-3, 3)
Qx, Qy = A["Q"][:, 0], A["Q"][:, 1]; ax[3].plot(t, Qx, "b-", lw=1, label="disc centre X [D]"); ax[3].plot(t, Qy, "m-", lw=1, label="disc centre Y (depth) [D]")
for mode, col in (("sway_Qx_D", "b"), ("sway_Qy_D", "m")):
    tp = AW["modes"][mode]["turning_points"]; ax[3].plot([q["t"] for q in tp], [q["value"] for q in tp], col + "o", ms=3)
    for q0, q1, am in zip(tp[:-1], tp[1:], AW["modes"][mode]["halfcycle_amplitude"]): ax[3].text(0.5 * (q0["t"] + q1["t"]), (0.07 if col == "b" else -0.1), "%.3f" % am, fontsize=5, color=col, ha="center")
ax[3].set_ylabel("sway [D]"); ax[3].legend(fontsize=8, loc="upper right"); ax[3].set_title("sway: period %.3f s (x), %.3f s (y); half-cycle amplitudes printed; disc tilt follows the wire (r = %.2f)" % (AN["sway"]["Qx_D"]["period_s"], AN["sway"]["Qy_D"]["period_s"], AN["tilt"]["correlations"]["tilt_Y_vs_swing_x"]), fontsize=9)
EN = AW["sway_2D_amplitude_supplementary"]; ax[4].errorbar(EN["window_centres_s"], EN["amp_D"], EN["sd_D"], fmt="ko-", ms=3, label="2D sway amplitude (isotropic invariant), 11-frame windows")
hx = AW["modes"]["sway_Qx_D"]; ax[4].plot(hx["halfcycle_end_t"], hx["halfcycle_amplitude"], "b.-", lw=0.7, label="x half-cycle amplitude"); hy = AW["modes"]["sway_Qy_D"]; ax[4].plot(hy["halfcycle_end_t"], hy["halfcycle_amplitude"], "m.-", lw=0.7, label="y half-cycle amplitude")
for a_ in ax: a_.axvline(Ta, color="r", lw=1.5); a_.axvspan(Ta, 10, color="r", alpha=0.05)
for tr in AW["triggers_sorted"][:4]: ax[4].axvline(tr["t"], color="orange", ls="--", lw=0.8); ax[4].text(tr["t"], 0.07, "%s %s" % (tr["mode"].replace("sway_", ""), tr["test"][0]), fontsize=6, rotation=90, color="orange")
ax[4].text(Ta + 0.05, 0.075, "T_a = %.2f s (%s)" % (Ta, AW["T_a_reason"]), fontsize=7, color="r"); ax[4].set_ylim(0, 0.085)
ax[4].set_ylabel("amplitude [D]"); ax[4].set_xlabel("t [s] (frame n at t = n/24)"); ax[4].legend(fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(VIS, "v002_psi_sway_Ta.png"), dpi=110); plt.close(fig)
# ---------- 4. camera drift
D = list(csv.DictReader(open(os.path.join(ROOT, "camera/drift.csv")))); g = lambda kk: np.array([float(r[kk]) for r in D])
fig, ax = plt.subplots(2, 2, figsize=(13, 7))
ax[0, 0].plot(t, g("azimuth_deg"), "k-"); ax[0, 0].set_ylabel("camera azimuth about A [deg]"); ax[0, 0].set_title("v002 reference measurement - draft | camera ORBIT (per-frame pose)", fontsize=10)
ax[0, 1].plot(t, g("dist_to_A"), "k-"); ax[0, 1].set_ylabel("distance to A [D]"); ax2 = ax[0, 1].twinx(); ax2.plot(t, g("elevation_deg"), "b-"); ax2.set_ylabel("elevation [deg]", color="b")
ax[1, 0].plot(t, g("f_free_vp_px"), "c-", label="free per-frame VP focal (ill-conditioned late)"); ax[1, 0].axhline(float(D[0]["f_px"]), color="k", label="adopted constant f"); ax[1, 0].set_ylim(2300, 3600); ax[1, 0].legend(fontsize=8); ax[1, 0].set_ylabel("f [px]"); ax[1, 0].set_xlabel("t [s]")
ax[1, 1].plot(t, g("vp_edge_rms_px"), "k-", label="VP edge rms"); ax[1, 1].plot(t, g("bundle_edge_rms_px"), "r-", label="bundle edge-endpoint rms"); ax[1, 1].plot(t, np.hypot(g("A_resid_u_px"), g("A_resid_v_px")), "g-", label="|A residual|")
ax[1, 1].legend(fontsize=8); ax[1, 1].set_ylabel("px"); ax[1, 1].set_xlabel("t [s]")
fig.tight_layout(); fig.savefig(os.path.join(VIS, "v002_camera_drift_plot.png"), dpi=100); plt.close(fig)
f0 = np.ascontiguousarray(F[0]); f240 = np.ascontiguousarray(F[240]); an = np.dstack([luma(f240), luma(f240), luma(f0)]).astype(np.uint8)  # red = f0, cyan = f240
# lines of the static model through frame-0 and frame-240 cameras on frame 240
BL = json.load(open(os.path.join(ROOT, "camera/out/bundle_lines.json"))); ov = f240.copy()
for kk, v in BL.items():
    X0 = np.array(v["point"]); dd = np.array(v["direction"])
    for cam, col in ((PJ.load_camera(240), (255, 255, 0)), (PJ.load_camera(240, kind="locked"), (255, 0, 255))):
        q = np.array(PJ.project(np.array([X0 - 0.9 * dd, X0 + 0.9 * dd]), cam)).T; cv2.line(ov, P16(q[0]), P16(q[1]), col, 2, cv2.LINE_AA, 4)
out = np.hstack([cv2.resize(an, (960, 544)), cv2.resize(ov, (960, 544))]); label(out, ["v002  frames 0 vs 240 (t=0 / 10 s)   reference measurement - draft", "left: anaglyph red = frame 0, cyan = frame 240 (raw): parallax between crossbar, posts and feet = orbit, not zoom",
                                                                                         "right: frame 240 with static edges via camera of record (cyan) vs locked frame-0 camera (magenta)"], scale=0.6)
cv2.imwrite(os.path.join(VIS, "v002_camera_drift_raw_vs_corrected.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("visuals written")
