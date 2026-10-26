"""Verify the level-tilt sign convention and that the tilted projection reproduces the left/right asymmetry.

Convention (params/physics.json level_tilt_rad = beta): the static scene (stand, pivot hardware, table, cameras;
everything described in camera.json / geometry.json stand frame) is rotated by +beta about world +Y through P
(right-hand rule; Blender: parent empty at P with rotation_euler = (0, beta, 0)). Gravity is exactly -Z_world;
the pendulum angle theta is relative to gravity. Then a bob at world angle theta appears at stand-frame angle
theta + beta, i.e. X_stand = R_y(beta)^T X_world.

Checks: (1) R_y(beta)^T bob_world(theta) == bob_stand(theta + beta) (numpy); (2) the rotation matrix of a
Blender Euler (0, beta, 0) equals R_y(beta) (run with --blender-matrix JSON produced by Blender, optional);
(3) anchor-window turning points: right/left stand-frame amplitudes of the data vs the projected model with
the fitted tilt vs the same model without tilt; mean u residual with/without tilt.
Usage: .venv/bin/python solver/tilt_check.py [--fit-dir solver/out/fit_tilt/Ta_4.0]
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))


def Ry(b):
    c, s = math.cos(b), math.sin(b)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fit-dir", default=str(HERE / "out/fit_tilt/Ta_4.0")); a = ap.parse_args()
    import fit_projected as FP
    from fit import load_track, theta_at
    r = json.loads((Path(a.fit_dir) / "fit_projected.json").read_text()); P = r["params"]; beta = P["th_off"]
    out = {"beta_rad": beta, "beta_deg": math.degrees(beta)}
    th = np.linspace(-0.3, 0.3, 61)
    Xw = np.column_stack([np.sin(th), np.zeros_like(th), -np.cos(th)])
    Xs = Xw @ Ry(beta)                      # rows: (R^T x)^T = x^T R
    Xs_expected = FP.bob_world(th + beta, 1.0)
    out["check1_max_abs_diff"] = float(np.max(np.abs(Xs - Xs_expected)))
    # (2) Blender Euler convention
    code = ("import mathutils,json,sys;b=float(sys.argv[-1]);"
            "print('MAT'+json.dumps([list(r) for r in mathutils.Euler((0.0,b,0.0),'XYZ').to_matrix()]))")
    try:
        res = subprocess.run(["blender", "-b", "--factory-startup", "--python-expr", code, "--", str(beta)],
                             capture_output=True, text=True, timeout=120)
        line = [l for l in res.stdout.splitlines() if l.startswith("MAT")][0]
        Mb = np.array(json.loads(line[3:]))
        out["check2_blender_euler_vs_Ry_max_abs_diff"] = float(np.max(np.abs(Mb - Ry(beta))))
        # stand 'up' axis in world after rotation: tilts toward +X for beta > 0
        out["stand_up_axis_in_world"] = (Mb @ np.array([0, 0, 1.0])).tolist()
    except Exception as e:  # noqa: BLE001
        out["check2_error"] = str(e)
    # (3) asymmetry
    d = load_track(ROOT / "measurement" / "bob_track.csv"); fr = np.round(d["t"] * 24).astype(int)
    F = FP.Fitter(d, FP.Cams(fr))
    X = F.cams.ray_plane_y0(d["y"]); alpha = np.arctan2(X[:, 0], -X[:, 2])      # stand-frame measured angle
    s = json.loads((ROOT / "measurement/summary.json").read_text())
    T_a = r["t_anchor_s"]
    tpf = [tp["frame"] for tp in s["amplitude"]["turning_points"] if tp["t_s"] <= T_a]

    def tp_amp(series):
        vals = []
        for n0 in tpf:
            js = np.arange(n0 - 3, n0 + 4)
            c2, c1, c0 = np.polyfit(js / 24 - n0 / 24, series[js], 2)
            vals.append(c0 - c1 ** 2 / (4 * c2))
        return np.degrees(np.array(vals))
    th_m, _ = theta_at(d["t"], P["L_eq"], P["theta0"], P["omega0"], P["gamma"], 0.0)
    rows = {"data_stand_frame": tp_amp(alpha), "model_with_tilt_stand_frame": tp_amp(th_m + beta),
            "model_without_tilt_stand_frame": tp_amp(th_m)}
    asym = {}
    for k, v in rows.items():
        R_ = v[v > 0]; L_ = v[v < 0]
        asym[k] = {"turning_points_deg": v.tolist(), "right_mean_deg": float(R_.mean()), "left_mean_deg": float(L_.mean()),
                   "asymmetry_right_minus_absleft_deg": float(R_.mean() + L_.mean())}
    out["check3_anchor_turning_point_asymmetry"] = asym
    an = d["t"] <= T_a + 1e-9
    for lab, PP in (("with_tilt", P), ("without_tilt_same_params", dict(P, th_off=0.0))):
        uv, _, _ = F.model(PP); e = uv - d["y"]
        out[f"anchor_{lab}"] = {"mean_u_resid_px": float(e[an, 0].mean()), "rms_px": float(np.sqrt(np.mean(np.sum(e[an] ** 2, 1))))}
    out["verdict"] = {"sign_ok": out["check1_max_abs_diff"] < 1e-12 and out.get("check2_blender_euler_vs_Ry_max_abs_diff", 1) < 1e-6,
                      "asymmetry_reproduced": abs(asym["model_with_tilt_stand_frame"]["asymmetry_right_minus_absleft_deg"]
                                                  - asym["data_stand_frame"]["asymmetry_right_minus_absleft_deg"]) < 0.3}
    p = Path(a.fit_dir) / "tilt_check.json"; p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
