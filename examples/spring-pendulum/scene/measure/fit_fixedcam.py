"""Refit of the static frame proportions with the camera FIXED to the camera of record (params/camera.json via
camera/project.py load_camera(frame=0)); scene role. Same box model, edge landmarks and residuals as fit_frame0.py,
geometry parameters only (11). Unit W_in (= S, clear span between the posts' inner faces; identical definitions).
.venv/bin/python scene/measure/fit_fixedcam.py -> scene/measure/fit_fixedcam.json
Spring/weight points are measured per frame through load_camera(frame=n) on the plane y = 0 (no homography needed:
the camera of record carries the measured drift)."""
import json, pathlib, sys
import numpy as np
from scipy.optimize import least_squares
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "camera"))
import fit_frame0 as F0
import project as PJ
LM = F0.LM
CAM0 = PJ.load_camera(frame=0)
cam0 = {"R": np.asarray(CAM0["R"]), "C": np.asarray(CAM0["C"]), "f_px": CAM0["f_px"], "cx": CAM0["cx"], "cy": CAM0["cy"]}


def resid(g, named=False):
    out, nm = [], {}
    for k, (P0, d) in F0.lines(g).items():
        P0, d = np.array(P0, float), np.array(d, float)
        a, b = F0.proj(P0 - 0.3 * d, cam0), F0.proj(P0 + 0.3 * d, cam0)
        n = np.array([-(b - a)[1], (b - a)[0]]); n /= np.linalg.norm(n)
        e = LM["edges"][k]
        r = [(np.array(q) - a) @ n / e["sigma"] for q in e["pts"]]
        nm[k] = [round(float(x * e["sigma"]), 2) for x in r]
        out += r
    out += [(g[2] - g[1]) / 0.1, g[8] / 0.1]          # weak priors (d_p ~ w_p, plate centred), sigma 0.1 W_in
    return (np.array(out), nm) if named else np.array(out)


def plane_y0(uv, cam):
    o, w = PJ.ray(uv[0], uv[1], cam)
    s = -o[1] / w[1]
    return o + s * w


def main():
    g0 = np.array([0.0, 0.18, 0.18, 0.012, 0.082, -1.42, 1.39, 0.42, -0.05, 0.025, 1.5])
    sol = least_squares(resid, g0, x_scale="jac", method="lm")
    r, nm = resid(sol.x, True)
    dof = len(r) - len(g0); s2 = float(r @ r / dof)
    cov = np.linalg.pinv(sol.jac.T @ sol.jac) * max(1.0, s2)
    sd = np.sqrt(np.diag(cov))
    g = dict(zip(F0.GN, sol.x)); gs = dict(zip(F0.GN, sd))
    pts = {}
    for k, v in LM["points_plane_y0"].items():
        if not isinstance(v, dict):
            continue
        for fk, uv in v.items():
            if fk.startswith("f"):
                X = plane_y0(uv, PJ.load_camera(frame=int(fk[1:])))
                pts.setdefault(k, {})[fk] = {"uv": uv, "xz_Win": [round(float(X[0]), 5), round(float(X[2]), 5)]}
    # local image scale (px per W_in) at the spring plane, frame 0 and frame 120
    scale = {}
    for f in (0, 120):
        cam = PJ.load_camera(frame=f)
        for v in (450, 860):
            X = plane_y0((960, v), cam)
            u1, v1 = PJ.project(X + [0.01, 0, 0], cam); u0, v0 = PJ.project(X, cam)
            scale[f"f{f:03d}_v{v}"] = float(np.hypot(u1 - u0, v1 - v0) / 0.01)
    res = {"unit": "W_in = clear span between the posts' inner faces (camera.json scale.unit)", "camera": "params/camera.json load_camera(frame=0) FIXED",
           "rms_sigma": float(np.sqrt(s2)), "n_obs": len(r), "n_par": len(g0), "geometry_S": g, "geometry_sd_S": gs,
           "edge_residuals_px": nm, "spring_plane_points": pts, "px_per_Win": scale,
           "camera_json_implied_frame": json.load(open(ROOT / "params/camera.json"))["frame_geometry_implied_by_calibration"]["values"]}
    (HERE / "fit_fixedcam.json").write_text(json.dumps(res, indent=1))
    print("rms(sigma)", round(np.sqrt(s2), 3))
    for k in F0.GN:
        print(f"  {k:7s} {g[k]:+.4f} +- {gs[k]:.4f}")
    for k, v in nm.items():
        print("  res", k, v)
    print(scale)
    for k, v in pts.items():
        print("  pt", k, v)


if __name__ == "__main__":
    main()
