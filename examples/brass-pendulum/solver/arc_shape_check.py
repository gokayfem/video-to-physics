"""Arc-shape residual (reference-defect effect of the AI's apparent rotation point).

Dynamics-free: every frame gets its own free angle theta_n; only the hinge geometry is shared.
  H1 (production geometry): hinge at P_ring, radius k (L_eff units) free.
  H2 (AI apparent hinge, diagnostic): hinge at (0,0,dz) with dz and k free (camera role: dz ~ -0.081).
The difference between H1 and H2 residuals is the part of the pixel misfit caused purely by the arc shape
(rotation point), independent of period/amplitude defects.

Usage: .venv/bin/python solver/arc_shape_check.py   -> solver/out/fit_projected/arc_shape.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from fit import load_track  # noqa: E402
from fit_projected import Cams, bob_world  # noqa: E402


def main():
    d = load_track(ROOT / "measurement" / "bob_track.csv")
    fr = np.round(d["t"] * 24).astype(int)
    cams = Cams(fr)
    X = cams.ray_plane_y0(d["y"])
    th0 = np.arctan2(X[:, 0], -X[:, 2])
    n = len(fr)

    def run(free_dz):
        def res(x):
            k = x[0]; dz = x[1] if free_dz else 0.0; th = x[2 if free_dz else 1:]
            uv = cams.project(bob_world(th, k, dz))
            return ((uv - d["y"]) / d["s"]).ravel()
        x0 = np.concatenate([[1.07] + ([-0.05] if free_dz else []), th0])
        r = least_squares(res, x0, method="trf", x_scale="jac", xtol=1e-12, ftol=1e-12)
        k = r.x[0]; dz = r.x[1] if free_dz else 0.0; th = r.x[2 if free_dz else 1:]
        uv = cams.project(bob_world(th, k, dz)); e = uv - d["y"]
        # radial (along pivot->bob image direction) vs tangential residual
        up = cams.project(np.tile([0.0, 0.0, dz], (n, 1)))
        rad = (uv - up); rad /= np.linalg.norm(rad, axis=1, keepdims=True)
        er = np.sum(e * rad, axis=1)
        J = r.jac[:, :2 if free_dz else 1]
        s2 = 2 * r.cost / (2 * n - len(r.x))
        cov = np.linalg.pinv(r.jac.T @ r.jac)[:J.shape[1], :J.shape[1]] * max(s2, 1)
        return {"k_Leff": float(k), "k_sd": float(math.sqrt(cov[0, 0])), "dz_Leff": float(dz),
                "dz_sd": float(math.sqrt(cov[1, 1])) if free_dz else None,
                "rms_px": float(np.sqrt(np.mean(np.sum(e ** 2, axis=1)))),
                "rms_radial_px": float(np.sqrt(np.mean(er ** 2))),
                "max_abs_radial_px": float(np.max(np.abs(er))),
                "radial_resid_vs_theta_corr": float(np.corrcoef(np.abs(th), er)[0, 1]),
                "theta_range_deg": [float(np.degrees(th.min())), float(np.degrees(th.max()))]}
    out = {"H1_hinge_P_ring (production)": run(False), "H2_free_hinge_dz (AI apparent, diagnostic)": run(True),
           "note": "per-frame free theta (no dynamics); difference H1-H2 = arc-shape (rotation point) effect, a reference defect"}
    p = HERE / "out" / "fit_projected" / "arc_shape.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
