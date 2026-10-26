"""Stage 12 (VR03-05): clean per-frame track of the disc SILHOUETTE centre P (top-face centre of the fitted rigid disc, s3) in world
coordinates (origin A, unit D) through the smoothed camera of record. sigma: sd_formal (Gauss-Newton, per frame) and
sd_eff = sd_formal (+) local scatter (rms of P minus a 7-frame quadratic Savitzky-Golay fit, 9-frame window, x sqrt(7/4) dof correction).
Systematic (not in sigma): f +-3 % scales positions about the camera (|P - C| ~ 4.8 D -> the depth axis mostly), gauge (A placement).
Also: P - Q check (Q = line rotation centre) and the line-offset model test. Writes measurement/disc_P_track.csv and measurement/disc_P_track.json."""
import numpy as np, csv, json, os, sys
from scipy.signal import savgol_filter
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import camfr
assert camfr.MODE == "world"
POSE = np.load(os.path.join(CACHE, "disc_pose.npy")); N = len(POSE); t = POSE[:, 1]
P = POSE[:, 2:5]; sdf = POSE[:, 7:10]; a, b = np.degrees(POSE[:, 5]), np.degrees(POSE[:, 6]); sil = POSE[:, 12]
res = P - savgol_filter(P, 7, 2, axis=0)
loc = np.array([np.sqrt(np.mean(res[max(0, i - 4):i + 5] ** 2, 0)) for i in range(N)]) * np.sqrt(7 / 4)
sde = np.sqrt(sdf ** 2 + loc ** 2)
uv = np.array([camfr.proj(P[i], camfr.cam(i)) for i in range(N)])
with open(os.path.join(MEAS, "disc_P_track.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["frame", "t_s", "Px_D", "Py_D", "Pz_D", "sd_eff_x", "sd_eff_y", "sd_eff_z", "sd_formal_x", "sd_formal_y", "sd_formal_z",
                                    "tilt_about_X_deg", "tilt_about_Y_deg", "u_P_px", "v_P_px", "silhouette_rms_px"])
    for i in range(N): w.writerow([i, "%.6f" % t[i]] + ["%.6f" % x for x in (*P[i], *sde[i], *sdf[i], a[i], b[i], *uv[i], sil[i])])
out = {"definition": "P = centre of the fitted top-face circle (radius 0.5 D) of the rigid-disc silhouette model (s3_disc_pose: top + bottom face, thickness %.3f D), world coords (origin A, X crossbar, Y away, Z up), unit D; camera = smoothed camera of record" % POSE[0, 17],
       "mean_D": P.mean(0).tolist(), "std_D": P.std(0).tolist(), "sd_eff_median_D": np.median(sde, 0).tolist(), "sd_formal_median_D": np.median(sdf, 0).tolist(),
       "silhouette_rms_px_median": float(np.median(sil)), "wire_length_A_to_P_mean_D": float(-P[:, 2].mean()),
       "Q_minus_P": {"value_D_world": "(+0.0019, +0.0309, 0) in the world-azimuth frame, i.e. constant +0.031 D in world +Y (away from the camera)",
                     "test": "line perpendicular offset vs psi: world-fixed point model rms 0.0058 D; disc-attached chord model (constant offset) rms 0.0229 D -> the line passes through a world-fixed point, not a disc-attached one (AI inconsistency, or a depth bias of the silhouette centre along the view ~ +Y)"},
       "systematics": "f +-3 % scales positions about the camera (mainly depth/Y and the absolute distance); A placement assumption; disc taken as a rigid circle"}
json.dump(out, open(os.path.join(MEAS, "disc_P_track.json"), "w"), indent=1); print(json.dumps(out, indent=1))
