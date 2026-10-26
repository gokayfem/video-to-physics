"""Stage 10: required visuals (measurement/visuals/v001_*), burned-in labels (version / frame-time / status)."""
import numpy as np, os, sys, json, csv, pickle
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
VIS = os.path.join(MEAS, "visuals"); os.makedirs(VIS, exist_ok=True)
F = load_frames(); H = np.load(os.path.join(CACHE, "H_ecc.npy")); Hi = np.array([np.linalg.inv(h) for h in H])
A = np.load(os.path.join(CACHE, "analysis.npz")); AN = json.load(open(os.path.join(MEAS, "analysis.json"))); AW = json.load(open(os.path.join(MEAS, "anchor_window.json")))
W = np.load(os.path.join(CACHE, "weight_raw.npz")); SP = list(csv.DictReader(open(os.path.join(MEAS, "spring_track.csv")))); spf = lambda k: np.array([float(r[k]) for r in SP])
t = np.arange(N) / FPS
HEX = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4", green="#008300", violet="#6250d6")
def bgr(h): h = h.lstrip("#"); return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))
STATUS = "reference measurement - draft"
def P16(p): return tuple(np.round((np.asarray(p) - 0.5) * 16).astype(int))
def wrect(c, th, hl, hw):
    a = np.array([np.sin(th), np.cos(th)]); p = np.array([np.cos(th), -np.sin(th)])
    return np.array([c + sx * hl * a + sy * hw * p for sx, sy in ((-1, -1), (-1, 1), (1, 1), (1, -1))])
Peff = A["Peff"]; cW0 = A["cW0"]; cR0 = A["cR0"]; thW0 = A["thW0"]
# ---------- V1 annotated frame 0 with all tracks (frame-0 geometry)
im = np.ascontiguousarray(F[0]).copy(); ov = im.copy()
for n in range(N - 1):
    cv2.line(ov, P16(cW0[n]), P16(cW0[n + 1]), bgr(HEX["blue"]), 2, cv2.LINE_AA, 4)
    cv2.line(ov, P16(cR0[n]), P16(cR0[n + 1]), bgr(HEX["orange"]), 2, cv2.LINE_AA, 4)
    cv2.line(ov, P16(A["Pbot0"][n]), P16(A["Pbot0"][n + 1]), bgr(HEX["aqua"]), 1, cv2.LINE_AA, 4)
for n in range(0, N, 6): cv2.circle(ov, P16(A["Ptop0"][n]), 24, bgr(HEX["violet"]), -1, cv2.LINE_AA, 4)
ext = [int(np.argmin(A["phid"])), int(np.argmax(A["phid"]))]
for n in ext:
    for nn, col in ((n, HEX["yellow"]),):
        d = np.array([np.sin(A["thS0"][nn]), np.cos(A["thS0"][nn])]); cv2.line(ov, P16(Peff - 20 * d), P16(A["Pbot0"][nn] + 30 * d), bgr(col), 2, cv2.LINE_AA, 4)
        Q = wrect(cW0[nn], thW0[nn], W["lengthB"][nn] / 2 / A["sdrift"][nn], W["widthB"][nn] / 2 / A["sdrift"][nn]); cv2.polylines(ov, [np.round((Q - 0.5) * 16).astype(np.int32)], True, bgr(HEX["magenta"]), 2, cv2.LINE_AA, 4)
im = cv2.addWeighted(ov, 0.9, im, 0.1, 0)
cv2.drawMarker(im, tuple(np.round(Peff).astype(int)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2); cv2.putText(im, "P_eff (spring-axis pivot)", (int(Peff[0]) + 14, int(Peff[1]) + 6), 0, 0.55, (0, 0, 255), 2)
CP = json.load(open(os.path.join(ROOT, "params/camera.json"))); Ap = CP["key_points"]["A_image_frame0_px"]
cv2.drawMarker(im, (int(Ap[0]), int(Ap[1])), (0, 255, 0), cv2.MARKER_CROSS, 22, 2); cv2.putText(im, "A (origin)", (int(Ap[0]) + 14, int(Ap[1]) - 6), 0, 0.55, (0, 255, 0), 2)
for r in csv.DictReader(open(os.path.join(MEAS, "static_landmarks.csv"))):
    if r["frame"] == "0" and "wood" not in r["landmark"] and "table" not in r["landmark"]: cv2.circle(im, (int(float(r["u_px"])), int(float(r["v_px"]))), 6, (255, 255, 255), 1, cv2.LINE_AA)
label(im, ["v001  frame 0 (t = 0.000 s)  all 241 frames of tracks in frame-0 geometry (drift removed)   " + STATUS,
           "blue = weight centre, orange = eye-ring centre (spring lower attachment), aqua = spring coil-body bottom, violet = coil-body top (every 6th frame)",
           "yellow = spring axis at left/right extremes (frames %d/%d), magenta = weight silhouette at those frames, white o = static landmarks" % tuple(ext),
           "swing %.1f..%.1f deg about P_eff; period %.4f s; weight tilt follows swing x1.3; no independent vertical bob" % (A["phid"].min(), A["phid"].max(), AN["swing_damped_sine"]["period_s"])], scale=0.62)
cv2.imwrite(os.path.join(VIS, "v001_annotated_f000_all_tracks.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
# ---------- V2 turning points / zero crossings strip (raw frames, own geometry)
tps = AN["turning_points_swing"]; zcs = AN["zero_crossings_swing"]
picks = [(0, "t=0 start")] + [(int(round(tp["t"] * FPS)), "TP %s" % ("R" if tp["kind"] == "max" else "L")) for tp in tps[:2]] + [(int(round(z["t"] * FPS)), "ZC") for z in zcs[:2]]
big = max(tps, key=lambda tp: abs(tp["y"] - AN["swing_center_deg"])); picks += [(int(round(big["t"] * FPS)), "largest TP"), (int(round(tps[-1]["t"] * FPS)), "last TP"), (240, "end")]
cells = []
for n, lab in picks:
    fr = np.ascontiguousarray(F[n]).copy(); cB = A["cB"][n]; cR = A["cRA"][n]
    Q = wrect(cB, W["thB"][n], W["lengthB"][n] / 2, W["widthB"][n] / 2); cv2.polylines(fr, [np.round((Q - 0.5) * 16).astype(np.int32)], True, bgr(HEX["magenta"]), 2, cv2.LINE_AA, 4)
    cv2.circle(fr, P16(cR), 32 * 16, bgr(HEX["orange"]), 2, cv2.LINE_AA, 4)
    pt, pb = A["Ptop"][n], A["Pbot"][n]; cv2.line(fr, P16(pt), P16(pb), bgr(HEX["yellow"]), 2, cv2.LINE_AA, 4)
    Pe = Happly(H[n], Peff)[0]; cv2.line(fr, P16(Pe), P16(cR), bgr(HEX["aqua"]), 1, cv2.LINE_AA, 4)
    c = fr[0:1010, int(cB[0]) - 230:int(cB[0]) + 230]; c = cv2.resize(c, None, fx=0.5, fy=0.5)
    label(c, ["%s" % lab, "f%d t=%.3f" % (n, n / FPS), "swing %.2f deg" % A["phid"][n]], org=(6, 20), scale=0.5); cells.append(c)
strip = np.hstack(cells); label(strip, ["v001  swing turning points (TP) and zero crossings (ZC); magenta weight silhouette, orange ring, yellow spring axis, aqua P_eff->ring   " + STATUS], org=(6, strip.shape[0] - 10), scale=0.5)
cv2.imwrite(os.path.join(VIS, "v001_turning_points_zero_crossings.jpg"), strip, [cv2.IMWRITE_JPEG_QUALITY, 92])
# ---------- V3 camera drift raw vs corrected
g0 = luma(F[0]).astype(np.uint8); g240 = luma(F[240]).astype(np.uint8)
T = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1.0]]); Hcv = np.linalg.inv(T) @ H[240] @ T
g240c = cv2.warpPerspective(g240, Hcv, (1920, 1088), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
def ana(a, b): return cv2.merge([b, b, a])
rawA = ana(g0, g240); corA = ana(g0, g240c)
dr = json.load(open(os.path.join(ROOT, "camera/out/drift_fit.json")))
z1 = cv2.resize(rawA[20:180, 440:700], (960, 590), interpolation=cv2.INTER_NEAREST); z2 = cv2.resize(corA[20:180, 440:700], (960, 590), interpolation=cv2.INTER_NEAREST)
top = np.hstack([cv2.resize(rawA, (960, 544)), cv2.resize(corA, (960, 544))]); bot = np.hstack([z1, z2])
label(top, ["v001  frames 0 vs 240 (t = 10 s)  red = frame 0, cyan = frame 240   " + STATUS, "LEFT raw: static frame misregistered up to ~25 px (push-in)",
            "RIGHT corrected: frame 240 mapped to frame-0 geometry with measured H_240^-1 (zoom +%.2f %%; sim. vs H rms %.2f px)" % (100 * (dr["scale_frame240"] - 1), dr["H_vs_sim_rms_median_px"])], scale=0.55)
label(bot, ["left post top x3.7: raw (left) / corrected (right) - only the moving spring and weight remain doubled"], scale=0.6)
cv2.imwrite(os.path.join(VIS, "v001_camera_drift_raw_vs_corrected.jpg"), np.vstack([top, bot]), [cv2.IMWRITE_JPEG_QUALITY, 90])
D = np.genfromtxt(os.path.join(ROOT, "camera/out/drift_similarity_raw.csv"), delimiter=",", names=True)
fig, ax = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
ax[0].plot(D["t_s"], (D["scale"] - 1) * 100, color=HEX["blue"], lw=2); ax[0].set_ylabel("image scale vs frame 0 [%]")
ax[1].plot(D["t_s"], D["tx_px"], color=HEX["blue"], lw=2, label="tx"); ax[1].plot(D["t_s"], D["ty_px"], color=HEX["orange"], lw=2, label="ty"); ax[1].legend(frameon=False); ax[1].set_ylabel("similarity translation [px]")
ax[2].plot(D["t_s"], D["H_vs_sim_rms_px"], color=HEX["blue"], lw=2, label="similarity vs homography, static region rms"); ax[2].plot(D["t_s"], D["landmarks_vs_sim_rms_px"], color=HEX["orange"], lw=2, label="tracked landmarks vs similarity rms"); ax[2].legend(frameon=False); ax[2].set_ylabel("residual [px]"); ax[2].set_xlabel("t [s]")
for a in ax: a.grid(alpha=0.25)
fig.suptitle("v001  camera drift per frame (vs frame 0) - " + STATUS); fig.tight_layout(); fig.savefig(os.path.join(VIS, "v001_camera_drift_plot.png"), dpi=110); plt.close(fig)
# ---------- V5 amplitude / period per half-cycle with T_a decision
Ta = AW["T_a_s"]
def hcplot(a, hc, key, lab, col, s_key=None):
    tm = [h["t_start"] * 0.5 + h["t_end"] * 0.5 for h in hc]; y = [h[key] for h in hc]; e = [h[s_key] for h in hc] if s_key else None
    a.errorbar(tm, y, yerr=e, fmt="o-", color=col, ms=6, lw=2, capsize=3, label=lab)
fig, ax = plt.subplots(5, 1, figsize=(11, 14), sharex=True)
ax[0].plot(t, A["phid"], color=HEX["blue"], lw=1.5, label="swing angle (ring about P_eff, image) [deg]")
ax[0].plot([tp["t"] for tp in tps], [tp["y"] for tp in tps], "o", color=HEX["orange"], ms=6, label="turning points")
ax[0].plot([z["t"] for z in zcs], [AN["swing_center_deg"]] * len(zcs), "s", color=HEX["aqua"], ms=6, label="zero crossings"); ax[0].set_ylabel("deg"); ax[0].legend(frameon=False, ncol=3, fontsize=8)
hcplot(ax[1], AN["half_cycles_swing"], "amp", "swing half-cycle amplitude [deg]", HEX["blue"], "s_amp"); ax[1].set_ylabel("deg")
nm = max(AW["modes"][0]["steps"], key=lambda e: e["z"]); ax[1].annotate("largest step +%.2f sigma (rule needs > 3)" % nm["z"], (nm["turning_point_t"], nm["amp"]), xytext=(nm["turning_point_t"] + 0.6, nm["amp"] + 0.12), arrowprops=dict(arrowstyle="->"), fontsize=9)
hcplot(ax[2], AN["half_cycles_swing"], "half_period", "swing half-period [s]", HEX["violet"], "s_half_period"); ax[2].set_ylabel("s")
hcplot(ax[3], AN["half_cycles_weight_tilt_at_swing_tps"], "amp", "weight tilt (rocking) amplitude at swing TPs [deg]", HEX["magenta"], "s_amp"); hcplot(ax[3], AN["half_cycles_weight_rel_tilt_at_swing_tps"], "amp", "weight tilt relative to pivot-ring line [deg]", HEX["green"], "s_amp"); ax[3].set_ylabel("deg"); ax[3].legend(frameon=False, fontsize=8)
ax[4].plot(t, A["rr"] - A["rr"].mean(), color=HEX["orange"], lw=1.2, label="pivot-ring distance r - mean [px] (swing-locked w + 2w; no independent bob)")
ax[4].plot(t, A["Lbody0"] - A["Lbody0"].mean(), color=HEX["aqua"], lw=1.2, label="spring coil-body length - mean [px]"); ax[4].set_ylabel("px"); ax[4].legend(frameon=False, fontsize=8); ax[4].set_xlabel("t [s]")
for a in ax:
    a.grid(alpha=0.25); a.axvspan(7.5, 10.0, color="0.85", zorder=0); a.axvline(Ta, color="k", lw=2, ls="--")
ax[1].legend(frameon=False, fontsize=8); ax[2].legend(frameon=False, fontsize=8)
ax[0].text(7.55, ax[0].get_ylim()[1] * 0.8, "held-out 7.5-10 s", fontsize=9); ax[1].text(9.0, ax[1].get_ylim()[0] + 0.05, "T_a = %.1f s" % Ta, fontsize=10, weight="bold")
fig.suptitle("v001  modes per half-cycle and anchor-window decision (rule CHANGELOG 2026-10-25T23:41:09Z): T_a = %.1f s, no trigger\n%s" % (Ta, STATUS)); fig.tight_layout(); fig.savefig(os.path.join(VIS, "v001_amplitude_period_per_halfcycle_Ta.png"), dpi=110); plt.close(fig)
# ---------- V6 spring coil crops over time (rectified along the axis)
cells = []
for n in range(0, N, 30):
    r = SP[n]; Pt = np.array([float(r["axis_u_at_top"]), float(r["v_top"])]); Pb = np.array([float(r["axis_u_at_bot"]), float(r["v_bot"])])
    d = Pb - Pt; Lx = np.linalg.norm(d); ax_ = d / Lx; pp = np.array([ax_[1], -ax_[0]])
    s = np.arange(-30, 520, 1.0); X = np.arange(-60, 60, 1.0)
    U = Pt[0] + s[:, None] * ax_[0] + X[None] * pp[0]; V = Pt[1] + s[:, None] * ax_[1] + X[None] * pp[1]
    c = cv2.cvtColor(np.clip(bilinear(luma(F[n]), U, V), 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    pr = pickle.load(open(os.path.join(CACHE, "spring_profiles.pkl"), "rb"))[n] if n == 0 or True else None
    for g0_, g1_ in pr["gaps"]: cv2.line(c, (0, int(0.5 * (g0_ + g1_)) + 30), (8, int(0.5 * (g0_ + g1_)) + 30), bgr(HEX["orange"]), 2)
    if np.isfinite(pr["s_dense0"]): cv2.line(c, (0, int(pr["s_dense0"]) + 30), (119, int(pr["s_dense0"]) + 30), bgr(HEX["aqua"]), 1)
    cv2.line(c, (0, int(Lx) + 30), (119, int(Lx) + 30), bgr(HEX["magenta"]), 1)
    c = cv2.resize(c, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    label(c, ["f%d t=%.2fs" % (n, n / FPS), "gaps %s dense~%s" % (r["n_open_gaps"], r["n_dense_scallops_R"].split(".")[0]), "L=%.0fpx" % (float(r["body_length_px"]) / A["sdrift"][n])], org=(4, 16), scale=0.42)
    cells.append(np.pad(c, ((0, 0), (0, 3), (0, 0))))
sp = np.hstack(cells); sp = np.vstack([np.zeros((60, sp.shape[1], 3), np.uint8), sp])
label(sp, ["v001  spring coil body rectified along its axis, every 30 frames (orange ticks = open-coil gaps, aqua = start of close-wound section, magenta = body end)",
           "open gaps 11 (12 only when swung right: boundary turn opens), close-wound ~10 turns; no coil-count change; L = body length in frame-0 px   " + STATUS], org=(6, 20), scale=0.5)
cv2.imwrite(os.path.join(VIS, "v001_spring_coil_crops.jpg"), sp, [cv2.IMWRITE_JPEG_QUALITY, 92])
# ---------- V7 two-method agreement + cadence
CA = json.load(open(os.path.join(MEAS, "cadence_analysis.json")))
fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
ax[0].plot(t, A["dW"][:, 0], color=HEX["blue"], lw=1.2, label="weight u: C - B"); ax[0].plot(t, A["dW"][:, 1], color=HEX["orange"], lw=1.2, label="weight v: C - B"); ax[0].fill_between(t, -A["sigW"][:, 0], A["sigW"][:, 0], color="0.85", label="+-sigma u")
ax[1].plot(t, np.degrees(A["dTh"]), color=HEX["magenta"], lw=1.2, label="weight tilt: C - B [deg]"); ax[1].plot(t, np.degrees(A["dThS"]), color=HEX["green"], lw=1.2, label="spring tilt: extent - centroid [deg]")
ax[2].plot(t, A["dR"][:, 0], color=HEX["blue"], lw=1.2, label="ring u: B - A"); ax[2].plot(t, A["dR"][:, 1], color=HEX["orange"], lw=1.2, label="ring v: B - A")
v = np.diff(np.genfromtxt(os.path.join(MEAS, "ring_track.csv"), delimiter=",", names=True)["u_stab_frame0_px"]); nn = np.arange(1, N)
for j, col in zip(range(4), [HEX["blue"], HEX["orange"], HEX["aqua"], HEX["violet"]]): ax[3].plot(t[1:][nn % 4 == j], v[nn % 4 == j], "o", ms=4, color=col, label="step into n = %d mod 4" % j)
ax[3].set_ylabel("ring u step [px/frame]"); ax[3].set_xlabel("t [s]")
for a in ax: a.grid(alpha=0.25); a.legend(frameon=False, fontsize=8, ncol=2)
ax[0].set_ylabel("px"); ax[1].set_ylabel("deg"); ax[2].set_ylabel("px")
fig.suptitle("v001  two-method tracking differences and generator cadence (offsets %s frames by n mod 4)\n%s" % (CA["ring_u_stab_px"]["delta_frames_by_n_mod_4"], STATUS)); fig.tight_layout(); fig.savefig(os.path.join(VIS, "v001_tracking_methods_and_cadence.png"), dpi=110); plt.close(fig)
print("visuals written:", sorted(os.listdir(VIS)))
