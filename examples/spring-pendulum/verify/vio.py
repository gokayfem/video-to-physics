"""Verifier I/O: canonical trajectory CSV, production column aliasing, Hermite helpers, small utilities."""
import csv, json, math, os
import numpy as np

CANON = ["t_s", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s", "xc_m", "yc_m", "zc_m", "qw", "qx", "qy", "qz",
         "wx", "wy", "wz", "ell_m", "elldot_m_s", "tension_N", "KE_J", "PE_J", "E_J", "P_diss_W", "W_diss_J", "Lx", "Ly", "Lz"]

# production column aliases -> canonical (first match wins). Positions: the EYE / spring attachment point.
COL_ALIASES = {
    "t_s": ["t_s", "t", "time_s", "time"],
    "x_m": ["x_m", "H_x_m", "eye_x_m", "xE_m", "E_x_m", "hook_x_m", "x_eye_m", "px_m"],
    "y_m": ["y_m", "H_y_m", "eye_y_m", "yE_m", "E_y_m", "hook_y_m", "y_eye_m", "py_m"],
    "z_m": ["z_m", "H_z_m", "eye_z_m", "zE_m", "E_z_m", "hook_z_m", "z_eye_m", "pz_m"],
    "vx_m_s": ["vx_m_s", "Hdot_x_m_s", "vx", "eye_vx_m_s", "vE_x_m_s", "v_x_m_s"],
    "vy_m_s": ["vy_m_s", "Hdot_y_m_s", "vy", "eye_vy_m_s", "vE_y_m_s", "v_y_m_s"],
    "vz_m_s": ["vz_m_s", "Hdot_z_m_s", "vz", "eye_vz_m_s", "vE_z_m_s", "v_z_m_s"],
    "xc_m": ["xc_m", "X_x_m", "com_x_m", "xC_m", "C_x_m"], "yc_m": ["yc_m", "X_y_m", "com_y_m", "yC_m", "C_y_m"], "zc_m": ["zc_m", "X_z_m", "com_z_m", "zC_m", "C_z_m"],
    "qw": ["qw", "q_w", "quat_w", "q_w"], "qx": ["qx", "q_x", "quat_x", "q_x"], "qy": ["qy", "q_y", "quat_y", "q_y"], "qz": ["qz", "q_z", "quat_z", "q_z"],
    "wx": ["wx", "wx_rad_s", "omega_x_rad_s"], "wy": ["wy", "wy_rad_s", "omega_y_rad_s"], "wz": ["wz", "wz_rad_s", "omega_z_rad_s"],
    "ell_m": ["ell_m", "spring_len_m", "spring_length_m", "length_m", "L_m", "l_m"],
    "elldot_m_s": ["elldot_m_s", "spring_rate_m_s", "spring_length_rate_m_s", "ldot_m_s"],
    "tension_N": ["tension_N", "spring_tension_N", "T_N", "spring_force_N"],
    "KE_J": ["KE_J", "KE", "kinetic_J"], "PE_J": ["PE_J", "PE", "potential_J"], "E_J": ["E_J", "E", "energy_J", "E_total_J"],
    "P_diss_W": ["P_diss_W", "P_diss", "dissipation_W"], "W_diss_J": ["W_diss_J", "W_diss", "dissipated_J", "E_diss_J"],
    "Lx": ["Lx", "Lx_kg_m2_s"], "Ly": ["Ly", "Ly_kg_m2_s"], "Lz": ["Lz", "Lz_kg_m2_s", "L_z"],
}


def read_csv(path):
    with open(path) as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [[float(x) for x in row] for row in r if row and not row[0].startswith("#")]
    A = np.array(rows, dtype=float)
    return {h: A[:, i] for i, h in enumerate(header)}, header


def read_traj(path, colmap=None):
    """-> (canonical dict, mapping used). colmap: optional JSON path or dict canonical->production name."""
    raw, header = read_csv(path)
    cm = {}
    if colmap:
        cm = json.load(open(colmap)) if isinstance(colmap, str) else dict(colmap)
    out, used = {}, {}
    for c, al in COL_ALIASES.items():
        names = [cm[c]] if c in cm else al
        for n in names:
            if n in raw:
                out[c] = raw[n]; used[c] = n; break
    if "wx" not in out and all(f"{a}_{x}" in raw for a in ("n", "ndot") for x in "xyz"):
        # zero-spin body (clip-02 production): world angular velocity = n x ndot (documented in solver/NOTES.md)
        n = np.stack([raw["n_x"], raw["n_y"], raw["n_z"]], -1); nd = np.stack([raw["ndot_x"], raw["ndot_y"], raw["ndot_z"]], -1)
        w = np.cross(n, nd)
        out["wx"], out["wy"], out["wz"] = w[:, 0], w[:, 1], w[:, 2]
        used["wx,wy,wz"] = "derived: n x ndot"
    return out, used


def write_traj(path, cols, header=None):
    header = header or [c for c in CANON if c in cols]
    n = len(cols[header[0]])
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for i in range(n):
            f.write(",".join(repr(float(cols[h][i])) for h in header) + "\n")


def pos(d):
    return np.stack([d["x_m"], d["y_m"], d["z_m"]], -1)


def vel(d):
    return np.stack([d["vx_m_s"], d["vy_m_s"], d["vz_m_s"]], -1)


def quat(d):
    if all(k in d for k in ("qw", "qx", "qy", "qz")):
        return np.stack([d["qw"], d["qx"], d["qy"], d["qz"]], -1)
    return None


def status(ok):
    return "PASS" if ok else "FAIL"


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.floating):
        o = float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    return o


def dump_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(jsonable(obj), f, indent=2)


# ---------------------------------------------------------------- integrals / fits
_GLX, _GLW = np.polynomial.legendre.leggauss(8)


def herm5_coeffs(y0, d0, dd0, y1, d1, dd1, h):
    """quintic Hermite in s in [0,1]; arrays broadcast."""
    a0, a1, a2 = y0, d0 * h, 0.5 * dd0 * h * h
    b0, b1, b2 = y1, d1 * h, 0.5 * dd1 * h * h
    # solve for c3,c4,c5 from end conditions
    r0 = b0 - (a0 + a1 + a2)
    r1 = b1 - (a1 + 2 * a2)
    r2 = b2 - a2
    c3 = 10 * r0 - 4 * r1 + r2
    c4 = -15 * r0 + 7 * r1 - 2 * r2
    c5 = 6 * r0 - 3 * r1 + r2
    return [a0, a1, a2, c3, c4, c5]


def cumulative_integral_quintic(t, f, fd, fdd):
    """int_0^t f dt with f represented by quintic Hermite (f, f', f'') per interval, Gauss-8 per interval."""
    h = np.diff(t)
    c = herm5_coeffs(f[:-1], fd[:-1], fdd[:-1], f[1:], fd[1:], fdd[1:], h)
    acc = np.zeros(len(h))
    for xg, wg in zip(_GLX, _GLW):
        s = 0.5 * (xg + 1)
        v = c[5]
        for k in (4, 3, 2, 1, 0):
            v = v * s + c[k]
        acc += 0.5 * wg * v
    return np.r_[0.0, np.cumsum(acc * h)]


def simpson_cumulative(t, y):
    h = t[1] - t[0]
    n = (len(t) - 1) // 2 * 2
    idx = np.arange(0, n + 1, 2)
    inc = h / 3 * (y[idx[:-1]] + 4 * y[idx[:-1] + 1] + y[idx[:-1] + 2])
    return idx, np.r_[0.0, np.cumsum(inc)]


def fit_sinusoid(t, x, w0, decay=False, w_rel_window=0.2):
    """Least-squares fit x ~ c + e^{-gt}(a cos wt + b sin wt). Returns (w, g, params, rms). Uses scipy."""
    from scipy.optimize import least_squares
    x0, sc = float(np.mean(x)), float(np.std(x)) or 1.0
    x = (x - x0) / sc          # normalise: termination tolerances are then meaningful for tiny amplitudes
    t = np.asarray(t, float)

    def model(p):
        w, g = p[0], (p[1] if decay else 0.0)
        C, S = np.cos(w * t), np.sin(w * t)
        env = np.exp(-g * t)
        M = np.stack([np.ones_like(t), env * C, env * S], 1)
        coef, *_ = np.linalg.lstsq(M, x, rcond=None)
        return M @ coef - x, coef
    # coarse scan around w0 to avoid aliasing to a wrong local minimum
    ws = np.linspace(w0 * (1 - w_rel_window), w0 * (1 + w_rel_window), 400)   # even count: w0 itself is not a start point
    best = min(ws, key=lambda w: np.sum(model([w, 0.0])[0] ** 2))
    p0 = [best, 0.0] if decay else [best]
    for _ in range(2):   # restart once from the optimum to make termination independent of the start
        r = least_squares(lambda p: model(p)[0], p0, xtol=1e-15, ftol=1e-15, gtol=1e-15, x_scale="jac", method="lm")
        p0 = r.x
    res, coef = model(r.x)
    coef = coef * sc; coef[0] += x0
    return float(r.x[0]), (float(r.x[1]) if decay else 0.0), coef, float(np.sqrt(np.mean(res ** 2)) * sc)
