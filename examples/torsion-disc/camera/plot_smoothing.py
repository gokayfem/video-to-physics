"""Visual for VR03-04: camera path raw vs smoothed, static-scene window motion (reference vs camera), jerk, R4 per frame.
Writes measurement/visuals/v002_camera_jitter_smoothing.png."""
import numpy as np, json, pathlib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = json.load(open(ROOT / "camera/out/camera_poses_raw.json")); SM = json.load(open(ROOT / "camera/out/camera_poses.json")); CS = json.load(open(ROOT / "camera/out/camera_smoothing.json"))
REF = np.load(ROOT / "measurement/cache/ref_window_motion.npy", allow_pickle=True).item()
f, cx, cy = RAW["f_px"], RAW["cx"], RAW["cy"]; N = len(RAW["frames"]); t = np.arange(N) / 24
def proj(X, fr): R = np.array(fr["R"]); C = np.array(fr["C"]); Xc = (X - C) @ R.T; return np.array([cx + f * Xc[0] / Xc[2], cy + f * Xc[1] / Xc[2]])
X = np.array([0.926, -0.109, -0.6])
vr = np.diff(np.array([proj(X, fr) for fr in RAW["frames"]]), axis=0); vs = np.diff(np.array([proj(X, fr) for fr in SM["frames"]]), axis=0); vref = REF["right_post"]
jr, js, jref = (np.hypot(*np.diff(v, axis=0).T) for v in (vr, vs, vref))
Craw = np.array([fr["C"] for fr in RAW["frames"]]); Csm = np.array([fr["C"] for fr in SM["frames"]])
az = lambda C: np.degrees(np.arctan2(C[:, 0], -C[:, 1]))
fig, ax = plt.subplots(5, 1, figsize=(13, 16), sharex=True)
ax[0].plot(t, az(Craw), "c-", lw=1, label="raw per-frame azimuth"); ax[0].plot(t, az(Csm), "k-", lw=1.5, label="smoothed (tau = %.1f s)" % CS["chosen_tau_s"])
ax[0].set_ylabel("camera azimuth [deg]"); ax[0].legend(fontsize=8)
ax[0].set_title("v002 reference measurement - draft | clip 03 camera path smoothing (VR03-04): max deviation from raw azimuth %.2f deg, elevation %.2f deg, distance %.4f D, rotation %.2f deg"
                % tuple(CS["max_deviation_from_raw"][k] for k in ("azimuth_deg", "elevation_deg", "distance_D", "rotation_deg")), fontsize=9)
tm = t[:-1] + 1 / 48
for i, c in enumerate("uv"):
    a = ax[1 + i]; a.plot(tm, vref[:, i], color="#2a78d6", lw=1, label="reference image motion (phase corr.)"); a.plot(tm, vr[:, i], "c-", lw=0.8, label="raw camera (projected)"); a.plot(tm, vs[:, i], "k-", lw=1.4, label="smoothed camera (projected)")
    a.set_ylabel("right-post d%s [px/frame]" % c); a.legend(fontsize=8)
m = CS["chosen"]["metrics"]["right_post"]; m0 = CS["raw"]["metrics"]["right_post"]
ax[3].plot(t[1:-1], jref, color="#2a78d6", lw=1, label="reference jerk: median %.2f / max %.2f px" % (np.median(jref), jref.max()))
ax[3].plot(t[1:-1], jr, "c-", lw=0.8, label="raw camera jerk: median %.2f / max %.2f px" % (np.median(jr), jr.max())); ax[3].plot(t[1:-1], js, "k-", lw=1.4, label="smoothed camera jerk: median %.3f / max %.3f px" % (np.median(js), js.max()))
ax[3].set_ylabel("right-post jerk [px]"); ax[3].legend(fontsize=8)
r4r = np.array(CS["r4_per_frame_raw"]); r4s = np.array(CS["r4_per_frame_smoothed"])
ax[4].plot(t, r4r, "c-", lw=0.8, label="raw camera: median %.2f / max %.2f px" % (np.median(r4r), r4r.max())); ax[4].plot(t, r4s, "k-", lw=1.4, label="smoothed camera: median %.2f / max %.2f px" % (np.median(r4s), r4s.max()))
ax[4].axhline(2.0, color="r", ls="--", label="R4 limit 2 px"); ax[4].set_ylim(0, 2.2); ax[4].set_ylabel("R4 static edge rms [px]"); ax[4].set_xlabel("t [s] (frame n at t = n/24)"); ax[4].legend(fontsize=8)
fig.tight_layout(); fig.savefig(ROOT / "measurement/visuals/v002_camera_jitter_smoothing.png", dpi=100)
print("ref vel vs smoothed rms %.3f, raw %.3f" % (np.sqrt(np.mean(np.sum((vs - vref) ** 2, 1))), np.sqrt(np.mean(np.sum((vr - vref) ** 2, 1)))))
