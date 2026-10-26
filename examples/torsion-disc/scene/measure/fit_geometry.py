"""Joint fit of the frame/clamp/disc proportions (unit D = disc diameter) and APPROXIMATE per-frame pinhole cameras
at reference frames 0/120/240 (scene role C). The AI camera orbits (right-front at frame 0 -> nearly frontal at 240),
so each frame has its own camera (R, C, f; principal point fixed at the image centre); the geometry is shared.

Observations: hand-read landmarks (scene/measure/landmarks.json) + disc silhouette points (disc_sil_fNNN.json, distance
to the convex hull of the projected top and bottom rim circles). The disc centre may sway per frame (dx, dy free; the
disc is level). World: origin A (wire exit at the clamp bottom), X along the crossbar (image right), Y away from the
camera, Z up. Writes scene/measure/fit_geometry.json.
.venv/bin/python scene/measure/fit_geometry.py [--camera-of-record]   (-> fit_geometry.json | fit_geometry_camref.json)
"""
import json, pathlib, sys
import numpy as np, cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

HERE = pathlib.Path(__file__).resolve().parent
W, H = 1920, 1088
FR = [0, 120, 240]
G0 = {  # unit D
    "S": 1.75, "xc": 0.0, "pw": 0.2, "pd": 0.2, "zt": 0.28, "bh": 0.22, "z_ft": -1.45, "fw": 0.2, "y_ff": -0.9,
    "y_fb": 0.45, "cw": 0.23, "cy": -0.12, "cz_top": 0.30, "cz_bot": 0.03, "L": 0.85, "t": 0.03}
GK = list(G0)


def model_points(g):
    S, xc, pw, pd, zt, bh = g["S"], g["xc"], g["pw"], g["pd"], g["zt"], g["bh"]
    xl_i, xl_o = xc - S / 2, xc - S / 2 - pw
    xr_i, xr_o = xc + S / 2, xc + S / 2 + pw
    yf, yb = -pd / 2, pd / 2
    xlc, xrc = xc - S / 2 - pw / 2, xc + S / 2 + pw / 2
    fw, zf = g["fw"], g["z_ft"]
    return {
        "LP_top_fo": (xl_o, yf, zt), "LP_top_fi": (xl_i, yf, zt), "LP_top_bo": (xl_o, yb, zt), "LP_top_bi": (xl_i, yb, zt),
        "LB_bot_fi": (xl_i, yf, zt - bh), "RB_bot_fi": (xr_i, yf, zt - bh),
        "RP_top_fi": (xr_i, yf, zt), "RP_top_fo": (xr_o, yf, zt), "RP_top_bi": (xr_i, yb, zt), "RP_top_bo": (xr_o, yb, zt),
        "CL_front_tl": (-g["cw"] / 2, g["cy"], g["cz_top"]), "CL_front_tr": (g["cw"] / 2, g["cy"], g["cz_top"]),
        "CL_front_bl": (-g["cw"] / 2, g["cy"], g["cz_bot"]), "CL_front_br": (g["cw"] / 2, g["cy"], g["cz_bot"]),
        "A": (0.0, 0.0, 0.0),
        "LP_bot_fo": (xl_o, yf, zf), "LP_bot_fi": (xl_i, yf, zf), "RP_bot_fi": (xr_i, yf, zf), "RP_bot_fo": (xr_o, yf, zf),
        "LF_top_front_o": (xlc - fw / 2, g["y_ff"], zf), "LF_top_back_i": (xlc + fw / 2, g["y_fb"], zf),
        "LF_top_back_o": (xlc - fw / 2, g["y_fb"], zf),
        "RF_top_back_i": (xrc - fw / 2, g["y_fb"], zf), "RF_top_back_o": (xrc + fw / 2, g["y_fb"], zf),
    }


def project(P, rv, C, f):
    R = Rot.from_rotvec(rv).as_matrix()
    X = (np.atleast_2d(P) - C) @ R.T
    return np.c_[f * X[:, 0] / X[:, 2] + W / 2, f * X[:, 1] / X[:, 2] + H / 2]


def lookat(C, T):
    z = np.asarray(T, float) - C; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return Rot.from_matrix(np.vstack([x, y, z])).as_rotvec()


def unpack(p):
    g = dict(zip(GK, p[:len(GK)]))
    cams, k = {}, len(GK)
    for n in FR:
        cams[n] = (p[k:k + 3], p[k + 3:k + 6], p[k + 6], p[k + 7:k + 9])   # rotvec, C, f, disc (dx, dy)
        k += 9
    return g, cams


def disc_hull(g, cam, d):
    a = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    r = 0.5
    top = np.c_[d[0] + r * np.cos(a), d[1] + r * np.sin(a), np.full_like(a, -g["L"])]
    bot = top - [0, 0, g["t"]]
    uv = project(np.vstack([top, bot]), cam[0], cam[1], cam[2])
    return cv2.convexHull(uv.astype(np.float32))


def residuals(p, LM, SIL, w_sil=0.3, sil_sigma=1.5):
    g, cams = unpack(p)
    MP = model_points(g)
    res = []
    for n in FR:
        rv, C, f, d = cams[n]
        for k, (u, v, s) in LM[str(n)].items():
            X = (d[0], d[1], -g["L"]) if k == "DISC_centre" else MP[k]
            uv = project(np.array(X), rv, C, f)[0]
            res += [(uv[0] - u) / s, (uv[1] - v) / s]
        hull = disc_hull(g, cams[n], d)
        for q in SIL[n]:
            res.append(w_sil * cv2.pointPolygonTest(hull, (float(q[0]), float(q[1])), True) / sil_sigma)
    # weak priors (sigma in D) keep unobserved/assumed dims sane
    res += [(g["t"] - 0.03) / 0.02, (g["fw"] - g["pw"]) / 0.05]
    return np.array(res)


def main():
    LM = json.loads((HERE / "landmarks.json").read_text())["frames"]
    SIL = {n: np.array(json.loads((HERE / f"disc_sil_f{n:03d}.json").read_text())["points_px"])[::2] for n in FR}
    # start: f ~ 2700 px from the vertical-edge convergence (posts lean in ~45-60 px over 925 px) and the disc ellipse
    # aspect (~20 deg elevation at the disc vs ~6 deg at the post tops => camera ~5 D away)
    p = [G0[k] for k in GK]
    for n, cx in zip(FR, (1.5, 0.7, 0.2)):
        C = np.array([cx, -5.0, 1.0])
        p += list(lookat(C, (0, 0, -0.6))) + list(C) + [2700.0, 0.0, 0.0]
    # robust loss: the AI frame is not perfectly rigid across the orbit (feet/back corners disagree by up to ~30 px at 240)
    r = least_squares(residuals, np.array(p), args=(LM, SIL), kwargs=dict(w_sil=0.3), loss="soft_l1", f_scale=3.0,
                      x_scale="jac", max_nfev=8000)
    g, cams = unpack(r.x)
    J = r.jac
    dof = max(1, len(r.fun) - len(r.x))
    s2 = 2 * r.cost / dof
    try:
        cov = np.linalg.pinv(J.T @ J) * s2
        sd = np.sqrt(np.diag(cov))
    except Exception:
        sd = np.full(len(r.x), np.nan)
    MP = model_points(g)
    out = {"unit": "D = disc diameter", "method": __doc__.strip().splitlines()[0], "cost": r.cost, "dof": dof,
           "rms_normalised": float(np.sqrt(s2)), "geometry_D": {k: round(float(g[k]), 5) for k in GK},
           "geometry_sd_D": {k: round(float(sd[i]), 5) for i, k in enumerate(GK)}, "cameras": {}, "residuals_px": {}}
    for n in FR:
        rv, C, f, d = cams[n]
        R = Rot.from_rotvec(rv).as_matrix()
        out["cameras"][str(n)] = {"R_world_to_cv": R.round(9).tolist(), "C_D": np.round(C, 6).tolist(), "f_px": float(f),
                                  "principal_point_px": [W / 2, H / 2], "disc_offset_D": np.round(d, 5).tolist(),
                                  "lens_mm_36": float(f * 36 / W)}
        rr = {}
        for k, (u, v, s) in LM[str(n)].items():
            X = (d[0], d[1], -g["L"]) if k == "DISC_centre" else MP[k]
            uv = project(np.array(X), rv, C, f)[0]
            rr[k] = [round(float(uv[0] - u), 2), round(float(uv[1] - v), 2)]
        hull = disc_hull(g, cams[n], d)
        sil = [cv2.pointPolygonTest(hull, (float(q[0]), float(q[1])), True) for q in SIL[n]]
        rr["disc_silhouette_rms"] = round(float(np.sqrt(np.mean(np.square(sil)))), 3)
        out["residuals_px"][str(n)] = rr
        e = np.degrees(Rot.from_matrix(R).as_euler("xyz"))
        print(n, "f", round(f), "C", np.round(C, 3), "disc", np.round(d, 3), "sil rms", rr["disc_silhouette_rms"])
    print(json.dumps(out["geometry_D"]))
    print(json.dumps(out["residuals_px"]))
    (HERE / "fit_geometry.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__" and "--camera-of-record" not in sys.argv:
    main()


# ---------------------------------------------------------------- refit through the camera of record
def main_camref(out_name="fit_geometry_camref.json"):
    """Cameras FIXED to params/camera.json via camera/project.py load_camera(frame=n) (unit D); geometry (16) + per-frame
    disc offsets (2 x 3) fitted to the same landmarks and silhouettes, robust loss."""
    ROOT = HERE.parents[1]
    sys.path.insert(0, str(ROOT / "camera"))
    import project as PJ
    LM = json.loads((HERE / "landmarks.json").read_text())["frames"]
    SIL = {n: np.array(json.loads((HERE / f"disc_sil_f{n:03d}.json").read_text())["points_px"])[::2] for n in FR}
    cams = {}
    for n in FR:
        c = PJ.load_camera(frame=n)
        assert abs(c["cx"] - W / 2) < 1e-6 and abs(c["cy"] - H / 2) < 1e-6, "principal point off-centre: extend project()"
        cams[n] = (Rot.from_matrix(np.asarray(c["R"])).as_rotvec(), np.asarray(c["C"], float), float(c["f_px"]))

    def unpack_c(p):
        g = dict(zip(GK, p[:len(GK)]))
        return g, {n: (cams[n][0], cams[n][1], cams[n][2], p[len(GK) + 2 * i:len(GK) + 2 * i + 2]) for i, n in enumerate(FR)}

    def res_c(p):
        g, cc = unpack_c(p)
        full = list(p[:len(GK)])
        for n in FR:
            full += list(cc[n][0]) + list(cc[n][1]) + [cc[n][2]] + list(cc[n][3])
        return residuals(np.array(full), LM, SIL)
    p0 = np.array([G0[k] for k in GK] + [0.0] * (2 * len(FR)))
    r = least_squares(res_c, p0, loss="soft_l1", f_scale=3.0, x_scale="jac", max_nfev=8000)
    g, cc = unpack_c(r.x)
    MP = model_points(g)
    out = {"unit": "D = disc diameter", "method": "camera of record FIXED (params/camera.json sha256 below); geometry + disc offsets fitted",
           "camera_json_sha256": __import__("hashlib").sha256((ROOT / "params/camera.json").read_bytes()).hexdigest(),
           "cost": r.cost, "geometry_D": {k: round(float(g[k]), 5) for k in GK}, "disc_offset_D": {}, "residuals_px": {}}
    for n in FR:
        rv, C, f, d = cc[n]
        out["disc_offset_D"][str(n)] = np.round(d, 5).tolist()
        rr = {}
        for k, (u, v, s) in LM[str(n)].items():
            X = (d[0], d[1], -g["L"]) if k == "DISC_centre" else MP[k]
            uv = project(np.array(X), rv, C, f)[0]
            rr[k] = [round(float(uv[0] - u), 2), round(float(uv[1] - v), 2)]
        hull = disc_hull(g, cc[n], d)
        sil = [cv2.pointPolygonTest(hull, (float(q[0]), float(q[1])), True) for q in SIL[n]]
        rr["disc_silhouette_rms"] = round(float(np.sqrt(np.mean(np.square(sil)))), 3)
        out["residuals_px"][str(n)] = rr
    (HERE / out_name).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["geometry_D"]))
    for n in FR:
        rr = out["residuals_px"][str(n)]
        print(n, "sil rms", rr["disc_silhouette_rms"], {k: v for k, v in rr.items() if k != "disc_silhouette_rms"})


if __name__ == "__main__" and "--camera-of-record" in sys.argv:
    main_camref()
