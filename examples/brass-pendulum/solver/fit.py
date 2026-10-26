"""Fit the simplest physically consistent pendulum model to the reference measurements.

Modes
  angle      : image-space angle series (measurement/angle_series.csv).  Projection-free: gives period,
               phase and *image-space* amplitude.  y = theta(t) + delta  (delta = image roll of vertical).
  track      : bob centre pixels (measurement/bob_track.csv) through a 2-D similarity map (no camera):
               u = c_u + R_px sin(theta + psi),  v = c_v + R_px cos(theta + psi)   (v down).
               Amplitude is identified through the arc curvature / pivot pixel; valid if the swing plane
               is close to fronto-parallel (foreshortening would bias amplitude, not period/phase).
  projected  : bob centre pixels through the camera model in params/camera.json (world -> pixel).
               Bob = P + L_c (sin th, 0, -cos th); L_c follows from the fitted L_eq under the declared
               inertia model (model-conditioned scale).

Dynamics parameters (all fits):  L_eq = I_P/(m d) [m] (from the period, g fixed 9.80665),
  theta0 [rad], omega0 [rad/s], gamma = c / (2 I_P) [1/s] >= 0 (FORMULATION convention; Q = omega0/(2 gamma)),
  beta = k_q / I_P [1/rad] >= 0.
  Coulomb friction is not fitted (not identifiable from ~8 periods of a 6 deg swing; documented).
  Physical coefficients are c = 2 gamma I_P, k_q = beta I_P with I_P from the ASSUMED bob mass.

Nested models M0 (no damping), M1 (gamma), M2 (gamma, beta) are fitted on t < 7.5 s and compared
by BIC; the simplest within 2 BIC of the best is selected.  Hold-out t >= 7.5 s is predicted without
refit.  Uncertainties: Jacobian covariance (formal and residual-scaled) and moving-block bootstrap.
Parameters at a bound, with relative sd > 0.5 or |corr| > 0.95 are flagged, not reported as values.

CLI:  ROOT/.venv/bin/python solver/fit.py --mode track [--track measurement/bob_track.csv]
          [--angle measurement/angle_series.csv] [--camera params/camera.json] [--nboot 40]
          [--pivot-px U V [--fix-pivot]] [--out solver/out] [--tag NAME]
Never writes params/physics.json (the lead decides; see NOTES.md).
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, curve_fit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import pendulum as pd  # noqa: E402

FPS = 24.0
T_SPLIT = 7.5
H_FIT = 1.0 / 768.0     # RK4 step used inside the fit (global error < 1e-9 rad, see selfcheck ladder)
G = pd.G_STD


# ------------------------------------------------------------------ data loading
def _read_csv(path):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    head = [h.strip() for h in rows[0]]
    cols = {h: [] for h in head}
    for r in rows[1:]:
        for h, x in zip(head, r):
            try:
                cols[h].append(float(x))
            except ValueError:
                cols[h].append(float("nan") if x.strip().lower() in ("", "nan", "none", "null") else x)
    return {h: np.array(v, dtype=object if any(isinstance(z, str) for z in v) else float) for h, v in cols.items()}


def _pick(cols, *cands, required=True):
    for c in cands:
        if c in cols:
            return c
    low = {k.lower(): k for k in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    if required:
        raise KeyError(f"none of {cands} in columns {list(cols)}")
    return None


def _times(cols):
    fc = _pick(cols, "frame", "frame_idx", "n", required=False)
    if fc is not None:
        return cols[fc].astype(float) / FPS
    return cols[_pick(cols, "t_s", "t", "time_s")].astype(float)


def _valid(cols, n):
    vc = _pick(cols, "valid", "ok", "good", required=False)
    if vc is None:
        return np.ones(n, bool)
    v = cols[vc]
    return np.array([str(x).strip().lower() in ("1", "1.0", "true", "yes") for x in v])


def load_track(path):
    c = _read_csv(path)
    t = _times(c)
    u = c[_pick(c, "u_px", "u", "x_px", "bob_u_px")].astype(float)
    v = c[_pick(c, "v_px", "v", "y_px", "bob_v_px")].astype(float)
    su = _pick(c, "sigma_u_px", "sigma_px", "sigma", "unc_px", required=False)
    sv = _pick(c, "sigma_v_px", "sigma_px", "sigma", "unc_px", required=False)
    su = c[su].astype(float) if su else np.full_like(u, 0.5)
    sv = c[sv].astype(float) if sv else np.full_like(v, 0.5)
    ok = _valid(c, len(t)) & np.isfinite(u) & np.isfinite(v) & np.isfinite(su) & (su > 0) & (sv > 0)
    return dict(kind="track", t=t[ok], y=np.column_stack([u, v])[ok], s=np.column_stack([su, sv])[ok],
                source=str(path))


def load_angle(path):
    c = _read_csv(path)
    t = _times(c)
    ac = _pick(c, "theta_img_rad", "angle_rad", "theta_rad", "angle_img_rad", "theta_img_deg", "angle_deg")
    a = c[ac].astype(float)
    sc = _pick(c, "sigma_rad", "sigma_theta_rad", "sigma_deg", "sigma", required=False)
    s = c[sc].astype(float) if sc else np.full_like(a, 1e-3 if "deg" not in ac else 0.05)
    if "deg" in ac:
        a = np.radians(a)
    if sc and "deg" in sc:
        s = np.radians(s)
    ok = _valid(c, len(t)) & np.isfinite(a) & np.isfinite(s) & (s > 0)
    return dict(kind="angle", t=t[ok], y=a[ok, None], s=s[ok, None], source=str(path))


# ------------------------------------------------------------------ camera model
class Camera:
    """World -> pixel (CONTRACT pixel convention: origin at top-left corner, u right, v down).

    Accepted params/camera.json layouts (first match wins):
      intrinsics: "K" 3x3 | {"fx_px","fy_px","cx_px","cy_px"} (top level or under "intrinsics")
                  | {"focal_mm","sensor_width_mm","resolution":[W,H]} (+ optional principal point cx_px, cy_px)
      extrinsics: "world_to_camera" 4x4 | {"R_world_to_cam" 3x3, "t_world_to_cam" 3}
                  (top level or under "extrinsics"); "camera_convention": "opencv" (x right, y down,
                  z forward; default) or "blender" (x right, y up, looking along -z).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        I = cfg.get("intrinsics", cfg)
        E = cfg.get("extrinsics", cfg)
        if "K" in I:
            K = np.array(I["K"], float)
        elif "fx_px" in I:
            K = np.array([[I["fx_px"], 0, I["cx_px"]], [0, I["fy_px"], I["cy_px"]], [0, 0, 1]], float)
        elif "focal_mm" in I:
            W, H = I.get("resolution", [1920, 1088])
            f = I["focal_mm"] / I["sensor_width_mm"] * W
            K = np.array([[f, 0, I.get("cx_px", W / 2)], [0, f, I.get("cy_px", H / 2)], [0, 0, 1]], float)
        else:
            raise KeyError("camera.json: no intrinsics (K | fx_px.. | focal_mm..)")
        if "world_to_camera" in E:
            M = np.array(E["world_to_camera"], float); R, t = M[:3, :3], M[:3, 3]
        elif "R_world_to_cam" in E:
            R = np.array(E["R_world_to_cam"], float); t = np.array(E["t_world_to_cam"], float)
        else:
            raise KeyError("camera.json: no extrinsics (world_to_camera | R_world_to_cam,t_world_to_cam)")
        conv = cfg.get("camera_convention", E.get("camera_convention", "opencv"))
        if conv == "blender":
            F = np.diag([1.0, -1.0, -1.0]); R, t = F @ R, F @ t
        elif conv != "opencv":
            raise ValueError(f"unknown camera_convention {conv}")
        self.K, self.R, self.t = K, R, t

    def project(self, X):
        Xc = X @ self.R.T + self.t
        z = Xc[:, 2]
        if np.any(z <= 0):
            raise ValueError("point behind camera")
        u = self.K[0, 0] * Xc[:, 0] / z + self.K[0, 1] * Xc[:, 1] / z + self.K[0, 2]
        v = self.K[1, 1] * Xc[:, 1] / z + self.K[1, 2]
        return np.column_stack([u, v])


# ------------------------------------------------------------------ mass model (L_eq <-> L_c)
def load_mass_model(params_path, ratio=None):
    """Mass model for the L_eq -> L_c mapping.  Production schema (FORMULATION s7): solid sphere on a
    massless rod, L_c = L_eq / (1 + 0.4/ratio^2) with ratio = L_c/r (image-measured; --ratio overrides)."""
    p = json.loads(Path(params_path).read_text()) if Path(params_path).exists() else {}
    if "model_id" in p:
        rr = ratio if ratio is not None else pd._val(p.get("ratio_Lc_over_r"))
        return {"inertia_model": "solid_sphere", "ratio": rr, "r": float(pd._val(p.get("bob_radius"), 0.0)),
                "rho": float(pd._val(p.get("rho_bob"), 8500.0)), "m_bob": None, "mu": 0.0}
    return {"inertia_model": pd._v(p, "inertia_model", "point_mass"), "ratio": None,
            "r": float(pd._v(p, "bob_radius_m", 0.0) or 0.0),
            "rho": float(pd._v(p, "bob_density_kg_m3", 8500.0) or 8500.0),
            "m_bob": pd._v(p, "bob_mass_kg"),
            "mu": float(pd._v(p, "wire_linear_density_kg_m", 0.0) or 0.0)}


def model_for_Lc(mm, L_c, **kw):
    if mm.get("ratio"):
        mm = dict(mm, r=L_c / mm["ratio"], m_bob=None)
    m_bob = mm["m_bob"] if mm["m_bob"] is not None else mm["rho"] * 4 / 3 * math.pi * mm["r"] ** 3
    if not m_bob:
        m_bob = 1.0
    return pd.model_from_values(L_c=L_c, r=mm["r"], m_bob=m_bob, inertia_model=mm["inertia_model"],
                                mu=mm["mu"], **kw)


def Lc_from_Leq(mm, L_eq):
    """Solve L_eq(L_c) = L_eq (monotone for realistic geometry) by Newton iteration."""
    if mm.get("ratio") and mm["inertia_model"] == "solid_sphere" and not mm.get("mu"):
        return L_eq / (1.0 + 0.4 / mm["ratio"] ** 2)
    L = L_eq
    for _ in range(50):
        f = model_for_Lc(mm, L, theta0=0.0).L_eq - L_eq
        df = (model_for_Lc(mm, L * (1 + 1e-7), theta0=0.0).L_eq - model_for_Lc(mm, L, theta0=0.0).L_eq) / (L * 1e-7)
        dL = f / df; L -= dL
        if abs(dL) < 1e-15:
            break
    return L


# ------------------------------------------------------------------ dynamics at data times
def theta_at(t_data, L_eq, th0, om0, gam, beta, h=H_FIT):
    """Point-mass equivalent (I = L_eq^2, m g d = g L_eq): identical dynamics for any mass model
    with the same L_eq and the same gamma = c/(2 I_P), beta = k_q/I_P."""
    n = np.round(t_data * FPS).astype(int)
    if np.all(np.abs(t_data * FPS - n) < 1e-6):
        grid = np.arange(n.max() + 1) / FPS
        idx = n
    else:
        grid = np.concatenate([[0.0], np.unique(t_data[t_data > 0])])
        idx = np.searchsorted(grid, t_data)
    M = pd.Model(m=1.0, d=L_eq, I_P=L_eq * L_eq, L_c=L_eq, m_bob=1.0, c_v=2.0 * gam * L_eq * L_eq,
                 k_q=beta * L_eq * L_eq, tau_c=0.0, omega_s=1e-3, g=G, theta0=th0, omega0=om0)
    th, om, _ = pd.integrate(M, grid, h)
    return th[idx], om[idx]


# ------------------------------------------------------------------ problem definition
class Problem:
    def __init__(self, data, mode, camera=None, mass_model=None, pivot_px=None, fix_pivot=False):
        self.data, self.mode, self.cam, self.mm = data, mode, camera, mass_model or {
            "inertia_model": "point_mass", "r": 0.0, "rho": 8500.0, "m_bob": None, "mu": 0.0}
        self.pivot_px, self.fix_pivot = pivot_px, fix_pivot
        self.names = ["L_eq", "theta0", "omega0", "gamma", "beta"]
        if mode == "angle":
            self.names += ["delta"]
        elif mode == "track":
            self.names += ["c_u", "c_v", "R_px", "psi"]
        self.lo = {"L_eq": 0.01, "theta0": -1.0, "omega0": -10.0, "gamma": 0.0, "beta": 0.0,
                   "delta": -0.5, "c_u": -1e5, "c_v": -1e5, "R_px": 1.0, "psi": -0.5}
        self.hi = {"L_eq": 50.0, "theta0": 1.0, "omega0": 10.0, "gamma": 5.0, "beta": 50.0,
                   "delta": 0.5, "c_u": 1e5, "c_v": 1e5, "R_px": 1e5, "psi": 0.5}

    def predict(self, P, t):
        th, om = theta_at(t, P["L_eq"], P["theta0"], P["omega0"], P["gamma"], P["beta"])
        return self.observe(P, th), th

    def observe(self, P, th):
        if self.mode == "angle":
            return (th + P["delta"])[:, None]
        if self.mode == "track":
            a = th + P["psi"]
            return np.column_stack([P["c_u"] + P["R_px"] * np.sin(a), P["c_v"] + P["R_px"] * np.cos(a)])
        L_c = Lc_from_Leq(self.mm, P["L_eq"])
        X = np.column_stack([L_c * np.sin(th), np.zeros_like(th), -L_c * np.cos(th)])
        return self.cam.project(X)

    def swing_signal(self, P, y=None, th=None):
        """Scalar signal that crosses zero at the equilibrium, used for phase comparison."""
        if th is not None and y is None:
            return th
        if self.mode == "angle":
            return y[:, 0] - P["delta"]
        if self.mode == "track":
            return np.arctan2(y[:, 0] - P["c_u"], y[:, 1] - P["c_v"]) - P["psi"]
        u_eq = self.observe(P, np.zeros(1))[0, 0]
        return y[:, 0] - u_eq


def _initial_guess(prob: Problem, fit_mask):
    d = prob.data
    t, y = d["t"][fit_mask], d["y"][fit_mask]
    out = {"gamma": 0.0, "beta": 0.0}
    if prob.mode == "track":
        if prob.pivot_px is not None:
            cu, cv = prob.pivot_px
            R = float(np.median(np.hypot(y[:, 0] - cu, y[:, 1] - cv)))
        else:  # algebraic (Kasa) circle fit to the arc
            A = np.column_stack([y[:, 0], y[:, 1], np.ones(len(y))])
            b = -(y[:, 0] ** 2 + y[:, 1] ** 2)
            D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
            cu, cv = -D / 2, -E / 2
            R = math.sqrt(max(cu ** 2 + cv ** 2 - F, 1.0))
            if cv > np.median(y[:, 1]):  # pivot must be above the bob (v down)
                cv = np.median(y[:, 1]) - R
        sig = np.arctan2(y[:, 0] - cu, y[:, 1] - cv)
        out.update(c_u=cu, c_v=cv, R_px=R, psi=0.0)
        scale = 1.0
    elif prob.mode == "angle":
        sig = y[:, 0]; scale = 1.0
    else:
        L0 = 0.388
        mm = prob.mm
        u0 = prob.cam.project(np.array([[0.0, 0.0, -L0]]))[0, 0]
        u1 = prob.cam.project(np.array([[L0 * math.sin(1e-3), 0.0, -L0 * math.cos(1e-3)]]))[0, 0]
        scale = (u1 - u0) / 1e-3      # px per rad (approx; L refined by the fit)
        sig = y[:, 0] - u0
    # damped sinusoid on the signal: A e^{-lam t} cos(w t + phi) + c
    tt = t - t[0]
    f = np.fft.rfftfreq(4096, d=1 / FPS)
    sp = np.abs(np.fft.rfft((sig - sig.mean()) * np.hanning(len(sig)), 4096))
    w0 = 2 * math.pi * f[np.argmax(sp[1:]) + 1]
    A0 = (sig.max() - sig.min()) / 2

    def fs(tq, A, lam, w, phi, c):
        return A * np.exp(-lam * tq) * np.cos(w * tq + phi) + c
    best = None
    for phi0 in np.linspace(-math.pi, math.pi, 8, endpoint=False):
        try:
            p, cov = curve_fit(fs, tt, sig, p0=[A0, 0.0, w0, phi0, float(np.mean(sig))], maxfev=20000)
            r = np.sum((fs(tt, *p) - sig) ** 2)
            if best is None or r < best[0]:
                best = (r, p, cov)
        except RuntimeError:
            pass
    _, p, cov = best
    A, lam, w, phi, c = p
    if A < 0:
        A, phi = -A, phi + math.pi
    T = 2 * math.pi / w
    amp = A / scale
    # elliptic correction of the period -> small-angle-equivalent L_eq
    L_eq = G * (T / (2 * math.pi)) ** 2 / (1 + amp ** 2 / 16) ** 2
    ph0 = phi - w * t[0]  # phase at t = 0
    out.update(L_eq=L_eq, theta0=amp * math.cos(ph0), omega0=-amp * w * math.sin(ph0))
    if prob.mode == "angle":
        out["delta"] = c
    if prob.mode == "track":
        out["theta0"] += 0.0
        out["psi"] = c   # mean angle offset in the image
        out["theta0"] = amp * math.cos(ph0)
    sinus = {"A_signal": float(A), "A_rad_est": float(amp), "lambda_1_s": float(lam),
             "lambda_sd": float(math.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) else None,
             "omega_rad_s": float(w), "period_s": float(T),
             "period_sd_s": float(T * math.sqrt(cov[2, 2]) / w) if np.isfinite(cov[2, 2]) else None,
             "note": "unconstrained damped-sinusoid fit to the swing signal on t < 7.5 s (projection-free); "
                     "lambda < 0 means the amplitude GROWS"}
    return out, sinus


# ------------------------------------------------------------------ fitting
def _fit_one(prob, init, free, mask, y=None):
    d = prob.data
    t = d["t"][mask]; ys = d["y"][mask] if y is None else y; s = d["s"][mask]
    names = [n for n in prob.names if n in free]
    fixed = {n: init[n] for n in prob.names if n not in free}
    x0 = np.array([init[n] for n in names])
    lo = np.array([prob.lo[n] for n in names]); hi = np.array([prob.hi[n] for n in names])
    x0 = np.clip(x0, lo + 1e-12 * (hi - lo) * (lo != 0), hi)

    def P_of(x):
        P = dict(fixed); P.update(zip(names, x)); return P

    def res(x):
        m, _ = prob.predict(P_of(x), t)
        return ((m - ys) / s).ravel()
    r = least_squares(res, x0, bounds=(lo, hi), x_scale="jac", method="trf",
                      xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400, diff_step=1e-7)
    return r, names, P_of(r.x)


def _cov_report(r, names, P, n_obs, prob):
    J = r.jac
    chi2 = float(2 * r.cost)
    dof = max(n_obs - len(names), 1)
    s2 = chi2 / dof
    at_bound = {n: bool(abs(P[n] - prob.lo[n]) < 1e-10 or abs(P[n] - prob.hi[n]) < 1e-10) for n in names}
    try:
        C = np.linalg.pinv(J.T @ J, rcond=1e-14)
    except np.linalg.LinAlgError:
        C = np.full((len(names), len(names)), np.nan)
    sd_formal = np.sqrt(np.abs(np.diag(C)))
    sd_scaled = sd_formal * math.sqrt(max(s2, 1.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = C / np.outer(sd_formal, sd_formal)
    cond = float(np.linalg.cond(J)) if J.size else float("nan")
    return dict(chi2=chi2, dof=dof, reduced_chi2=s2, cond_J=cond, at_bound=at_bound,
                sd_formal=dict(zip(names, sd_formal.tolist())), sd_scaled=dict(zip(names, sd_scaled.tolist())),
                corr={a: {b: float(corr[i, j]) for j, b in enumerate(names)} for i, a in enumerate(names)})


def crossings(t, s):
    i = np.where(np.sign(s[:-1]) * np.sign(s[1:]) < 0)[0]
    return t[i] - s[i] * (t[i + 1] - t[i]) / (s[i + 1] - s[i]), np.sign(s[i + 1] - s[i])


def run_fit(data, mode, camera=None, mass_model=None, nboot=40, seed=1, pivot_px=None, fix_pivot=False,
            models=("M0", "M1", "M2")):
    prob = Problem(data, mode, camera, mass_model, pivot_px, fix_pivot)
    t = data["t"]
    fit = t < T_SPLIT
    hold = ~fit
    init, sinus = _initial_guess(prob, fit)
    base_free = [n for n in prob.names if n not in ("gamma", "beta")]
    if mode == "track" and fix_pivot and pivot_px is not None:
        base_free = [n for n in base_free if n not in ("c_u", "c_v")]
    free_sets = {"M0": base_free, "M1": base_free + ["gamma"], "M2": base_free + ["gamma", "beta"]}
    n_obs = int(fit.sum() * data["y"].shape[1])
    results = {}
    start = dict(init)
    for name in models:
        st = dict(start)
        if name == "M1" and "M0" in results:
            st = dict(results["M0"]["P"]); st["gamma"] = 0.01   # start inside the feasible set
        if name == "M2" and "M1" in results:
            st = dict(results["M1"]["P"]); st["gamma"] = max(st["gamma"], 0.005); st["beta"] = 0.01
        r, names, P = _fit_one(prob, st, free_sets[name], fit)
        if name == "M0":
            start = dict(P)
        cr = _cov_report(r, names, P, n_obs, prob)
        bic = cr["chi2"] + len(names) * math.log(n_obs)
        results[name] = dict(P=P, names=names, cov=cr, bic=bic, success=bool(r.success), nfev=int(r.nfev),
                             _r=r)
    best = min(results, key=lambda k: results[k]["bic"])
    order = [m for m in ("M0", "M1", "M2") if m in results]
    chosen = next(m for m in order if results[m]["bic"] <= results[best]["bic"] + 2.0)
    R = results[chosen]
    P = R["P"]

    # residuals, hold-out, phase
    m_all, th_all = prob.predict(P, t)
    resid = m_all - data["y"]
    unit = "rad" if mode == "angle" else "px"
    rms = lambda a: float(np.sqrt(np.mean(np.sum(a ** 2, axis=1)))) if len(a) else None   # Euclidean per sample
    sig_med = float(np.median(np.sqrt(np.mean(data["s"] ** 2, axis=1))))
    s_data = prob.swing_signal(P, y=data["y"])
    tc_d, dir_d = crossings(t, s_data)
    tg = np.arange(0, int(round(t.max() * 96)) + 1) / 96.0
    thg, _ = theta_at(tg, P["L_eq"], P["theta0"], P["omega0"], P["gamma"], P["beta"])
    s_model = prob.swing_signal(P, y=prob.observe(P, thg)) if mode == "projected" else thg
    tc_m, dir_m = crossings(tg, s_model)
    T_model = float(np.mean(np.diff(tc_m[dir_m > 0]))) if np.sum(dir_m > 0) > 1 else float("nan")
    ph = []
    for tc, dd in zip(tc_d, dir_d):
        same = tc_m[dir_m == dd]
        if len(same):
            j = np.argmin(np.abs(same - tc)); ph.append((tc, (same[j] - tc) / T_model))
    ph = np.array(ph) if ph else np.zeros((0, 2))
    last = ph[ph[:, 0] >= t.max() - T_model] if len(ph) else ph
    phase_end = float(np.mean(last[:, 1])) if len(last) else None
    # growth diagnostic from peaks of the data swing signal (per half period)
    env = _envelope(t, s_data, T_model)

    # bootstrap (moving blocks of ~1 period, both channels together) on the fit segment
    boot = _bootstrap(prob, R, fit, nboot, seed, block=max(2, int(round(T_model * FPS))))

    flags = _flags(R, boot, sinus)
    mm = prob.mm
    L_c = Lc_from_Leq(mm, P["L_eq"])
    Mphys = model_for_Lc(mm, L_c, theta0=P["theta0"], omega0=P["omega0"])
    out = {
        "mode": mode, "source": data["source"], "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fit_window": f"t < {T_SPLIT} s", "holdout_window": f"t >= {T_SPLIT} s", "n_fit": int(fit.sum()),
        "n_holdout": int(hold.sum()), "g_m_s2": G,
        "projection_free_sinusoid": sinus,
        "model_selection": {m: {"bic": results[m]["bic"], "chi2": results[m]["cov"]["chi2"],
                                "reduced_chi2": results[m]["cov"]["reduced_chi2"],
                                "n_params": len(results[m]["names"]), "success": results[m]["success"],
                                "gamma": results[m]["P"]["gamma"], "beta": results[m]["P"]["beta"]}
                            for m in results},
        "chosen_model": chosen,
        "params": {k: float(v) for k, v in P.items()},
        "uncertainty": {k: R["cov"][k] for k in ("sd_formal", "sd_scaled", "reduced_chi2", "cond_J", "at_bound")},
        "correlation": R["cov"]["corr"],
        "bootstrap": boot,
        "flags": flags,
        "derived": {"L_eq_m": P["L_eq"], "L_c_m": L_c, "inertia_model": mm["inertia_model"],
                    "L_c_provenance": "model-conditioned (period + assumed g + declared inertia model)",
                    "period_model_s": T_model, "small_angle_period_s": 2 * math.pi * math.sqrt(P["L_eq"] / G),
                    "theta0_deg": math.degrees(P["theta0"]), "omega0_rad_s": P["omega0"],
                    "amplitude_note": {"angle": "IMAGE-SPACE amplitude (projection not removed)",
                                       "track": "similarity-map amplitude (assumes fronto-parallel swing plane)",
                                       "projected": "world amplitude through params/camera.json"}[mode],
                    "I_P_kg_m2_assumed_mass": Mphys.I_P,
                    "c_N_m_s": 2 * P["gamma"] * Mphys.I_P, "k_q_N_m_s2": P["beta"] * Mphys.I_P,
                    "Q": (math.sqrt(G / P["L_eq"]) / (2 * P["gamma"])) if P["gamma"] > 0 else None,
                    "damping_ratio_zeta": P["gamma"] / math.sqrt(G / P["L_eq"]),
                    "c_provenance": "fitted gamma = c/(2 I_P) x assumed I_P (assumed brass density; r from ratio)"},
        "residuals": {"unit": unit, "rms_fit": rms(resid[fit]), "rms_holdout": rms(resid[hold]),
                      "max_abs_fit": float(np.max(np.abs(resid[fit]))),
                      "max_abs_holdout": float(np.max(np.abs(resid[hold]))) if hold.any() else None,
                      "median_sigma": sig_med,
                      "R2_target": None if mode == "angle" else max(2.0, 3 * sig_med),
                      "R3_target": None if mode == "angle" else max(3.0, 3 * sig_med),
                      "per_channel_rms_fit": np.sqrt(np.mean(resid[fit] ** 2, axis=0)).tolist()},
        "phase": {"phase_error_end_cycles": phase_end,
                  "R3_phase_target_cycles": 0.05,
                  "per_crossing": [[float(a), float(b)] for a, b in ph],
                  "note": "(model crossing - data crossing)/T for crossings of the swing signal; end = mean over the last period"},
        "envelope": env,
    }
    if mode != "angle":
        out["residuals"]["R2_pass"] = out["residuals"]["rms_fit"] <= out["residuals"]["R2_target"]
        out["residuals"]["R3_pass"] = (out["residuals"]["rms_holdout"] is not None and
                                       out["residuals"]["rms_holdout"] <= out["residuals"]["R3_target"])
    out["phase"]["R3_phase_pass"] = phase_end is not None and abs(phase_end) < 0.05
    out["_series"] = dict(t=t, y=data["y"], m=m_all, fit=fit)
    return out


def _envelope(t, s, T):
    """Peak |swing| per half period and an exponential fit -> growth/decay rate (projection-free)."""
    if not np.isfinite(T):
        return None
    half = T / 2
    k = np.floor(t / half).astype(int)
    tp, ap = [], []
    for kk in np.unique(k):
        m = k == kk
        if m.sum() >= 3:
            i = np.argmax(np.abs(s[m])); tp.append(t[m][i]); ap.append(abs(s[m][i]))
    tp, ap = np.array(tp), np.array(ap)
    if len(tp) < 4:
        return None
    A = np.column_stack([np.ones_like(tp), tp])
    coef, *_ = np.linalg.lstsq(A, np.log(ap), rcond=None)
    r = np.log(ap) - A @ coef
    s2 = r @ r / max(len(tp) - 2, 1)
    sd = math.sqrt(s2 * np.linalg.inv(A.T @ A)[1, 1])
    rate = float(coef[1])
    return {"log_amplitude_slope_1_s": rate, "sd": sd, "n_peaks": len(tp),
            "amplitude_first_last": [float(ap[0]), float(ap[-1])],
            "growth_detected": bool(rate > 3 * sd and rate > 0),
            "note": "positive slope = amplitude GROWTH (non-physical for a free pendulum: reference defect)"}


def _bootstrap(prob, R, fit, nboot, seed, block):
    if nboot <= 0:
        return None
    d = prob.data
    t = d["t"][fit]
    P = R["P"]
    m, _ = prob.predict(P, t)
    res = d["y"][fit] - m
    n = len(t)
    rng = np.random.default_rng(seed)
    draws = []
    for b in range(nboot):
        idx = []
        while len(idx) < n:
            s0 = rng.integers(0, n - block + 1)
            idx.extend(range(s0, s0 + block))
        idx = np.array(idx[:n])
        yb = m + res[idx]
        r, names, Pb = _fit_one(prob, P, R["names"], fit, y=yb)
        draws.append([Pb[k] for k in R["names"]])
    D = np.array(draws)
    out = {"n": nboot, "block_len_samples": block, "method": "moving-block residual bootstrap, refit from best"}
    for i, k in enumerate(R["names"]):
        col = D[:, i]
        out[k] = {"sd": float(col.std(ddof=1)), "p2.5": float(np.percentile(col, 2.5)),
                  "p97.5": float(np.percentile(col, 97.5)),
                  "frac_at_lower_bound": float(np.mean(np.abs(col - prob.lo[k]) < 1e-10))}
    if len(R["names"]) > 1:
        C = np.corrcoef(D.T)
        out["corr"] = {a: {b: float(C[i, j]) for j, b in enumerate(R["names"])} for i, a in enumerate(R["names"])}
    return out


def _flags(R, boot, sinus):
    P, cov = R["P"], R["cov"]
    flags = {}
    for n in R["names"]:
        f = []
        sd = max(cov["sd_scaled"][n], (boot or {}).get(n, {}).get("sd", 0.0) or 0.0)
        if cov["at_bound"][n]:
            f.append(f"AT_BOUND (constraint active; approx one-sided 95% upper limit {P[n] + 1.645 * sd:.3g})")
        elif abs(P[n]) > 0 and sd / abs(P[n]) > 0.5:
            f.append(f"UNIDENTIFIED (relative sd {sd / abs(P[n]):.2g}); do not use value")
        for m, c in cov["corr"][n].items():
            if m != n and np.isfinite(c) and abs(c) > 0.95:
                f.append(f"STRONGLY_CORRELATED with {m} ({c:+.3f})")
        flags[n] = f
    if sinus.get("lambda_sd") and sinus["lambda_1_s"] < -3 * sinus["lambda_sd"]:
        flags["_reference"] = ["AMPLITUDE_GROWTH: unconstrained sinusoid gives lambda = "
                               f"{sinus['lambda_1_s']:.4g} +/- {sinus['lambda_sd']:.2g} 1/s (<0 = growth). "
                               "Non-physical for a free pendulum: reference defect, not modelled; damping held >= 0."]
    return flags


def well_conditioned(out):
    core = ["L_eq", "theta0", "omega0"]
    return all(not out["flags"].get(k) for k in core) and "_reference" not in out["flags"]


def save(out, outdir, tag):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    ser = out.pop("_series")
    (outdir / f"fit_{tag}.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, y, m, fit = ser["t"], ser["y"], ser["m"], ser["fit"]
    k = y.shape[1]
    fig, ax = plt.subplots(k + 1, 1, figsize=(11, 3 * (k + 1)), sharex=True)
    lab = ["angle [rad]"] if k == 1 else ["u [px]", "v [px]"]
    for i in range(k):
        ax[i].plot(t, y[:, i], ".", ms=3, label="data"); ax[i].plot(t, m[:, i], "-", lw=1, label="model")
        ax[i].axvline(T_SPLIT, color="k", ls=":"); ax[i].set_ylabel(lab[i]); ax[i].legend(fontsize=7)
    for i in range(k):
        ax[k].plot(t, m[:, i] - y[:, i], ".", ms=3, label=f"resid {lab[i]}")
    ax[k].axvline(T_SPLIT, color="k", ls=":"); ax[k].legend(fontsize=7); ax[k].set_xlabel("t [s]")
    ax[0].set_title(f"fit {tag}: model {out['chosen_model']}, holdout right of dotted line")
    fig.tight_layout(); fig.savefig(outdir / f"fit_{tag}.png", dpi=100); plt.close(fig)
    out["_series"] = ser
    return outdir / f"fit_{tag}.json"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["angle", "track", "projected"], required=True)
    ap.add_argument("--track", default=str(ROOT / "measurement" / "bob_track.csv"))
    ap.add_argument("--angle", default=str(ROOT / "measurement" / "angle_series.csv"))
    ap.add_argument("--camera", default=str(ROOT / "params" / "camera.json"))
    ap.add_argument("--params", default=str(ROOT / "params" / "physics.json"))
    ap.add_argument("--ratio", type=float, default=None, help="L_c/r override for the L_eq -> L_c mapping")
    ap.add_argument("--pivot-px", nargs=2, type=float, default=None)
    ap.add_argument("--fix-pivot", action="store_true")
    ap.add_argument("--nboot", type=int, default=40)
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    if a.mode == "angle":
        data = load_angle(a.angle); cam = None
    else:
        data = load_track(a.track)
        cam = Camera(json.loads(Path(a.camera).read_text())) if a.mode == "projected" else None
    out = run_fit(data, a.mode, cam, load_mass_model(a.params, a.ratio), a.nboot, pivot_px=a.pivot_px, fix_pivot=a.fix_pivot)
    out["well_conditioned_core"] = well_conditioned(out)
    p = save(out, a.out, a.tag or a.mode)
    s = {k: out[k] for k in ("chosen_model", "params", "flags", "residuals", "phase")}
    s["phase"] = {k: v for k, v in s["phase"].items() if k != "per_crossing"}
    print(json.dumps(s, indent=1, default=float))
    print("wrote", p)


if __name__ == "__main__":
    main()
