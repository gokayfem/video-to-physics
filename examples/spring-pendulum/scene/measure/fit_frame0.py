"""Joint fit of the static frame (box model) and an approximate pinhole camera at reference frame 0 (scene role).

.venv/bin/python scene/measure/fit_frame0.py  ->  scene/measure/fit_frame0.json

Unit S = clear span between the two posts' inner faces (= 1). World = config/coordinates.md: origin A = top
bearing point of the spring's top loop on the crossbar (bar top, y = 0 plane through the bar centre), X along the
bar to image right, Y away from the camera, Z up. Box model:
  posts: x in [x_c-0.5-w_p, x_c-0.5] and [x_c+0.5, x_c+0.5+w_p], y in [-d_p/2, d_p/2], z in [z_bt, z_pt]
  bar  : z in [-h_b, 0], y in [-d_b/2, d_b/2] with d_b = 0.5 h_b (ASSUMED, not visible from the front)
  base : plate x in [x_c-W/2, x_c+W/2], y in [y_pl-D/2, y_pl+D/2], z in [z_bt-t, z_bt]
  table: top z = z_bt - t, back edge (meets the backdrop) at y = y_wall
Camera: pinhole, square pixels, principal point at the image centre (960, 544), no distortion. f, C, yaw/pitch/roll.
Edge observations are point-to-projected-line distances (px). Points on the spring plane y = 0 are then measured by
ray/plane intersection; frames 120/240 are first mapped to frame-0 pixels with the measurement role's ECC homography.
"""
import json, pathlib
import numpy as np
from scipy.optimize import least_squares

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LM = json.loads((HERE / "landmarks_f000.json").read_text())
CX, CY = 960.0, 544.0

GN = ["x_c", "w_p", "d_p", "z_pt", "h_b", "z_bt", "W_pl", "D_pl", "y_pl", "t_pl", "y_wall"]
CN = ["f", "Cx", "Cy", "Cz", "yaw", "pitch", "roll"]


def rot(yaw, pitch, roll):
    """world->camera (OpenCV axes). Camera looks along +Y at zero angles; yaw about world Z, pitch about camera x
    (positive = looking down), roll about the optical axis."""
    R0 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
    cp, sp = np.cos(pitch), np.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    cr, sr = np.cos(roll), np.sin(roll)
    Rr = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rr @ Rx @ R0 @ Rz


def cam_of(c):
    f, Cx, Cy, Cz, yaw, pitch, roll = c
    return {"R": rot(yaw, pitch, roll), "C": np.array([Cx, Cy, Cz]), "f_px": f, "cx": CX, "cy": CY}


def proj(X, cam):
    Xc = (np.asarray(X, float) - cam["C"]) @ cam["R"].T
    return np.stack([cam["cx"] + cam["f_px"] * Xc[..., 0] / Xc[..., 2], cam["cy"] + cam["f_px"] * Xc[..., 1] / Xc[..., 2]], -1)


def lines(g):
    x_c, w_p, d_p, z_pt, h_b, z_bt, W, D, y_pl, t, y_wall = g
    d_b = 0.5 * h_b
    yf = -d_p / 2
    L = {
        "postL_outer_front": ((x_c - 0.5 - w_p, yf, 0), (0, 0, 1)),
        "postL_inner_front": ((x_c - 0.5, yf, 0), (0, 0, 1)),
        "postL_inner_back": ((x_c - 0.5, -yf, 0), (0, 0, 1)),
        "postR_inner_front": ((x_c + 0.5, yf, 0), (0, 0, 1)),
        "postR_inner_back": ((x_c + 0.5, -yf, 0), (0, 0, 1)),
        "postR_outer_front": ((x_c + 0.5 + w_p, yf, 0), (0, 0, 1)),
        "postL_top_front": ((0, yf, z_pt), (1, 0, 0)),
        "postR_top_front": ((0, yf, z_pt), (1, 0, 0)),
        "postL_bottom_front": ((0, yf, z_bt), (1, 0, 0)),
        "postR_bottom_front": ((0, yf, z_bt), (1, 0, 0)),
        "bar_top_front": ((0, -d_b / 2, 0), (1, 0, 0)),
        "bar_bottom_front": ((0, -d_b / 2, -h_b), (1, 0, 0)),
        "base_top_back": ((0, y_pl + D / 2, z_bt), (1, 0, 0)),
        "base_top_front": ((0, y_pl - D / 2, z_bt), (1, 0, 0)),
        "base_bottom_front": ((0, y_pl - D / 2, z_bt - t), (1, 0, 0)),
        "base_left_front": ((x_c - W / 2, y_pl - D / 2, 0), (0, 0, 1)),
        "base_right_front": ((x_c + W / 2, y_pl - D / 2, 0), (0, 0, 1)),
        "table_back_edge": ((0, y_wall, z_bt - t), (1, 0, 0)),
    }
    return L


def resid(p, return_named=False):
    g, c = p[:len(GN)], p[len(GN):]
    cam = cam_of(c)
    out, named = [], {}
    for k, (P0, d) in lines(g).items():
        P0, d = np.array(P0, float), np.array(d, float)
        a, b = proj(P0 - 0.3 * d, cam), proj(P0 + 0.3 * d, cam)
        n = np.array([-(b - a)[1], (b - a)[0]]); n /= np.linalg.norm(n)
        e = LM["edges"][k]
        r = [(np.array(q) - a) @ n / e["sigma"] for q in e["pts"]]
        named[k] = [round(float(x * e["sigma"]), 2) for x in r]
        out += r
    # X gauge: A (origin) is under the top loop, observed at u = 955.5 (sigma 1.5 px)
    uA = proj(np.zeros(3), cam)[0]
    out.append((uA - LM["points_plane_y0"]["A_top_loop_inner_top"]["f000"][0]) / 1.5)
    named["A_u_gauge"] = [round(float(uA - LM["points_plane_y0"]["A_top_loop_inner_top"]["f000"][0]), 2)]
    # weak priors (sigma in S): posts square-ish, plate centred under the posts in depth
    out += [(g[2] - g[1]) / 0.05, (g[8] - 0.0) / 0.05]
    return (np.array(out), named) if return_named else np.array(out)


def ray_plane_y0(uv, cam):
    d = np.array([(uv[0] - cam["cx"]) / cam["f_px"], (uv[1] - cam["cy"]) / cam["f_px"], 1.0]) @ cam["R"]
    s = -cam["C"][1] / d[1]
    return cam["C"] + s * d


def main():
    g0 = [0.05, 0.18, 0.18, 0.015, 0.085, -1.36, 1.42, 0.40, 0.0, 0.022, 0.8]
    c0 = [3000.0, 0.05, -4.4, -0.55, 0.0, 0.07, 0.0]
    p0 = np.array(g0 + c0)
    sol = least_squares(resid, p0, x_scale="jac", method="lm", max_nfev=20000)
    r, named = resid(sol.x, True)
    J = sol.jac
    dof = max(1, len(r) - len(p0))
    s2 = float(r @ r / dof)
    cov = np.linalg.pinv(J.T @ J) * max(1.0, s2)
    sd = np.sqrt(np.diag(cov))
    g = dict(zip(GN, sol.x[:len(GN)])); c = dict(zip(CN, sol.x[len(GN):]))
    gs = dict(zip(GN, sd[:len(GN)])); cs = dict(zip(CN, sd[len(GN):]))
    cam = cam_of(sol.x[len(GN):])
    # spring-plane points (y = 0), frames 120/240 mapped to frame-0 pixels by the measurement ECC homography
    H = np.load(ROOT / "measurement" / "cache" / "H_ecc.npy")
    pts = {}
    for k, v in LM["points_plane_y0"].items():
        if not isinstance(v, dict):
            continue
        for fk, uv in v.items():
            if not fk.startswith("f"):
                continue
            n = int(fk[1:])
            q = np.linalg.inv(H[n]) @ np.array([uv[0], uv[1], 1.0]); q = q[:2] / q[2]
            X = ray_plane_y0(q, cam)
            pts.setdefault(k, {})[fk] = {"uv_frame0": q.round(2).tolist(), "xz_S": [round(float(X[0]), 5), round(float(X[2]), 5)]}
    # local image scale at the spring plane near the spring (px per S) at frame 0
    Xm = ray_plane_y0((960, 450), cam)
    s_px = float(np.linalg.norm(proj(Xm + [0.01, 0, 0], cam) - proj(Xm, cam)) / 0.01)
    Xw = ray_plane_y0((900, 860), cam)
    s_px_w = float(np.linalg.norm(proj(Xw + [0.01, 0, 0], cam) - proj(Xw, cam)) / 0.01)
    res = {"unit": "S = clear span between the posts' inner faces", "rms_sigma": float(np.sqrt(s2)), "n_obs": len(r),
           "n_par": len(p0), "geometry_S": g, "geometry_sd_S": gs, "camera": c, "camera_sd": cs,
           "camera_R": cam["R"].tolist(), "edge_residuals_px": named, "spring_plane_points": pts,
           "px_per_S_spring_plane": {"v450": s_px, "v860": s_px_w},
           "assumed": {"bar_depth_over_height": 0.5, "principal_point": [CX, CY]},
           "priors": "d_p = w_p +- 0.05 S; y_pl = 0 +- 0.05 S"}
    (HERE / "fit_frame0.json").write_text(json.dumps(res, indent=1))
    print("rms(sigma)", round(np.sqrt(s2), 3), "n", len(r))
    for k in GN:
        print(f"  {k:7s} {g[k]:+.4f} +- {gs[k]:.4f}")
    for k in CN:
        v = np.degrees(c[k]) if k in ("yaw", "pitch", "roll") else c[k]
        print(f"  {k:7s} {v:+.4f}")
    for k, v in named.items():
        print("  res", k, v)
    print("px/S", s_px, s_px_w)
    for k, v in pts.items():
        print("  pt", k, v)


if __name__ == "__main__":
    main()
