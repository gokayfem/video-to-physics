"""Map measurement/bob_track.csv (raw per-frame image coords, measurement role) into reference
frame-120 image coordinates with the camera-role drift similarity (camera/out/drift_similarity_raw.csv).
Radius is divided by the drift scale.  Output camera/out/bob_meas_f120.csv (same columns as
bob_quick_f120.csv) used by camera/calibrate.py when present."""
import numpy as np, pathlib, csv
ROOT = pathlib.Path(__file__).resolve().parents[1]
class _T: pass
B = _T()
_r = list(csv.DictReader(open(ROOT / "measurement/bob_track.csv")))
for col in ("u_px", "v_px", "radius_px", "sigma_u_px", "sigma_v_px"):
    setattr(B, col, np.array([float(r[col]) for r in _r]))
Dr = np.loadtxt(ROOT / "camera/out/drift_similarity_raw.csv", delimiter=",", skiprows=1)
s, rho, tx, ty = Dr[:, 2], Dr[:, 3], Dr[:, 4], Dr[:, 5]
x = (B.u_px - tx)/s; y = (B.v_px - ty)/s
c, sn = np.cos(rho), np.sin(rho)
u = c*x + sn*y; v = -sn*x + c*y
with open(ROOT / "camera/out/bob_meas_f120.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame", "t_s", "u_f120", "v_f120", "r_f120", "sigma_px"])
    for k in range(len(B.u_px)):
        w.writerow([k, f"{k/24:.6f}", f"{u[k]:.4f}", f"{v[k]:.4f}", f"{B.radius_px[k]/s[k]:.4f}", f"{np.hypot(B.sigma_u_px[k], B.sigma_v_px[k]):.4f}"])
q = np.loadtxt(ROOT / "camera/out/bob_quick_f120.csv", delimiter=",", skiprows=1)
print("measurement vs camera-quick track (f120 coords): du rms %.3f dv rms %.3f dr rms %.3f" % (
    np.sqrt(np.mean((u-q[:, 2])**2)), np.sqrt(np.mean((v-q[:, 3])**2)), np.sqrt(np.mean((B.radius_px/s-q[:, 4])**2))))
