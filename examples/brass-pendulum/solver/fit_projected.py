"""Full projected fit of the passive pendulum model to measurement/bob_track.csv through the camera of
record (params/camera.json + camera/project.py, per-frame reference-matched camera incl. measured zoom).

Model (production, pivot fixed at P_ring = world origin):
    bob centre (calibration units L_eff) = k * (sin th, 0, -cos th)
    th(t) from the production solver (pendulum.py, RK4) with L_eq (period, g fixed), th0, om0,
    gamma = c/(2 I_P) >= 0.
    k = |bob - P_ring| / L_eff: the pendulum length in the camera's calibration gauge (an image-geometry
    parameter, like a pixel radius). Metric L_c comes ONLY from the period: L_c = L_eq / (1 + 0.4/ratio^2)
    (ratio = L_c/r = 12.30 +/- 0.45 measured, params/geometry.json). Camera metres: C_m = C_Leff * L_c / k.
Fit on t < 7.5 s; hold-out t >= 7.5 s predicted without refit.
Damping: M0 (gamma = 0) vs M1 (gamma >= 0) by BIC.  If gamma is not identified (bound / not preferred),
the production fit re-estimates th0, om0, L_eq, k with Q FIXED at the declared assumption 1000
(sensitivity at Q = 300 and 6000 reported).
Diagnostics (labelled, never production): D1 free pivot offset dz (along world Z, L_eff units);
D2 free dz + swing-plane azimuth phi.

Usage: .venv/bin/python solver/fit_projected.py [--nboot 40] [--out solver/out/fit_projected]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
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
from fit import load_track, theta_at, crossings, _envelope  # noqa: E402

G = pd.G_STD
T_SPLIT = 7.5
Q_ASSUMED, Q_RANGE = 1000.0, (300.0, 6000.0)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_project():
    spec = importlib.util.spec_from_file_location("cam_project", ROOT / "camera" / "project.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class Cams:
    """Per-frame cameras in L_eff units (positions divided by camera.json scale.L_eff_m)."""

    def __init__(self, frames, kind="reference"):
        prj = load_project()
        cj = json.loads((ROOT / "params" / "camera.json").read_text())
        self.Leff_m_json = float(cj["scale"]["L_eff_m"])
        cams = [prj.load_camera(frame=int(n), kind=kind) for n in frames]
        self.R = np.array([c["R"] for c in cams]); self.C = np.array([c["C"] for c in cams]) / self.Leff_m_json
        self.f = np.array([c["f_px"] for c in cams]); self.cx = np.array([c["cx"] for c in cams])
        self.cy = np.array([c["cy"] for c in cams])
        self.C_Leff_json = np.array(cj["reference_camera"]["extrinsics"]["C_world_Leff"], float)

    def project(self, X):            # X (N,3) in L_eff units, one point per frame
        Xc = np.einsum("nij,nj->ni", self.R, X - self.C)
        return np.column_stack([self.cx + self.f * Xc[:, 0] / Xc[:, 2], self.cy + self.f * Xc[:, 1] / Xc[:, 2]])

    def depth(self, X):
        return np.einsum("nij,nj->ni", self.R, X - self.C)[:, 2]

    def ray_plane_y0(self, uv):
        """Triangulate pixel rays with the swing plane Y = 0 (diagnostic 'measured theta')."""
        d = np.column_stack([(uv[:, 0] - self.cx) / self.f, (uv[:, 1] - self.cy) / self.f, np.ones(len(uv))])
        w = np.einsum("nji,nj->ni", self.R, d)
        s = -self.C[:, 1] / w[:, 1]
        return self.C + s[:, None] * w


def bob_world(th, k, dz=0.0, phi=0.0):
    s = np.sin(th)
    return np.column_stack([k * s * math.cos(phi), k * s * math.sin(phi), dz - k * np.cos(th)])


NAMES_ALL = ["L_eq", "theta0", "omega0", "gamma", "k", "dz", "phi", "th_off"]
LO = dict(L_eq=0.05, theta0=-1.0, omega0=-10.0, gamma=0.0, k=0.5, dz=-0.5, phi=-0.5, th_off=-0.2)
HI = dict(L_eq=5.0, theta0=1.0, omega0=10.0, gamma=2.0, k=2.0, dz=0.5, phi=0.5, th_off=0.2)


class Fitter:
    def __init__(self, data, cams):
        self.d, self.cams = data, cams
        self.t = data["t"]; self.frames = np.round(self.t * 24).astype(int)

    def model(self, P, idx=None):
        th, om = theta_at(self.t, P["L_eq"], P["theta0"], P["omega0"], P["gamma"], 0.0)
        X = bob_world(th + P.get("th_off", 0.0), P["k"], P.get("dz", 0.0), P.get("phi", 0.0))
        uv = self.cams.project(X)
        return uv, th, om

    def fit(self, init, free, mask, fixed_gamma_Q=None, y=None):
        names = [n for n in NAMES_ALL if n in free]
        base = dict(dz=0.0, phi=0.0, th_off=0.0); base.update(init)
        x0 = np.clip(np.array([base[n] for n in names]), [LO[n] for n in names], [HI[n] for n in names])
        Y = self.d["y"] if y is None else y
        S = self.d["s"]

        def P_of(x):
            P = dict(base); P.update(zip(names, x))
            if fixed_gamma_Q is not None:
                P["gamma"] = math.sqrt(G / P["L_eq"]) / (2 * fixed_gamma_Q)
            return P

        def res(x):
            uv, _, _ = self.model(P_of(x))
            return ((uv - Y) / S)[mask].ravel()
        r = least_squares(res, x0, bounds=([LO[n] for n in names], [HI[n] for n in names]), x_scale="jac",
                          xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=500, diff_step=1e-7)
        P = P_of(r.x)
        n_obs = int(mask.sum() * 2)
        chi2 = float(2 * r.cost); dof = n_obs - len(names); s2 = chi2 / dof
        C = np.linalg.pinv(r.jac.T @ r.jac, rcond=1e-14)
        sdf = np.sqrt(np.abs(np.diag(C)))
        corr = C / np.outer(sdf, sdf)
        at_b = {n: bool(abs(P[n] - LO[n]) < 1e-10 or abs(P[n] - HI[n]) < 1e-10) for n in names}
        return dict(P=P, names=names, chi2=chi2, dof=dof, reduced_chi2=s2, bic=chi2 + len(names) * math.log(n_obs),
                    sd_formal=dict(zip(names, sdf.tolist())), sd_scaled=dict(zip(names, (sdf * math.sqrt(max(s2, 1))).tolist())),
                    corr={a: {b: float(corr[i, j]) for j, b in enumerate(names)} for i, a in enumerate(names)},
                    at_bound=at_b, success=bool(r.success), nfev=int(r.nfev), cond_J=float(np.linalg.cond(r.jac)))


def rms(a):
    return float(np.sqrt(np.mean(np.sum(a ** 2, axis=1)))) if len(a) else None


def metrics(F, P, fitm, hold, anchor=None):
    """fitm/hold = the DECLARED R2/R3 segments (t < 7.5 s / t >= 7.5 s); anchor = the window actually fitted."""
    uv, th, _ = F.model(P)
    r = uv - F.d["y"]
    n4 = (F.frames % 4) == 0
    sig = float(np.median(np.sqrt(np.mean(F.d["s"] ** 2, axis=1))))
    out = {"rms_fit_px": rms(r[fitm]), "rms_holdout_px": rms(r[hold]),
           "rms_fit_u_v_px": np.sqrt(np.mean(r[fitm] ** 2, axis=0)).tolist(),
           "rms_holdout_u_v_px": np.sqrt(np.mean(r[hold] ** 2, axis=0)).tolist(),
           "max_abs_px": float(np.max(np.abs(r))), "median_sigma_px": sig,
           "R2_target_px": max(2.0, 3 * sig), "R3_target_px": max(3.0, 3 * sig),
           "supplementary_n_mod4_eq0": {"rms_fit_px": rms(r[fitm & n4]), "rms_holdout_px": rms(r[hold & n4]),
                                        "n_fit": int((fitm & n4).sum()), "n_holdout": int((hold & n4).sum())},
           "mean_residual_by_n_mod4_px": {str(q): np.mean(r[(F.frames % 4) == q], axis=0).tolist() for q in range(4)},
           "measured_cadence_offsets_frames_disclosed": {"0": 0.277, "1": 0.113, "2": -0.094, "3": -0.296,
                                                         "source": "measurement/NOTES.md (diagnostic; simulation NOT retimed)"}}
    if anchor is not None:
        out["rms_anchor_px"] = rms(r[anchor]); out["rms_anchor_u_v_px"] = np.sqrt(np.mean(r[anchor] ** 2, axis=0)).tolist()
        out["rms_after_anchor_px"] = rms(r[~anchor])
        out["supplementary_n_mod4_eq0"]["rms_anchor_px"] = rms(r[anchor & n4])
    out["R2_pass"] = out["rms_fit_px"] <= out["R2_target_px"]
    out["R3_rms_pass"] = out["rms_holdout_px"] <= out["R3_target_px"]
    # phase from triangulated measured theta vs model theta (both directions of zero crossing)
    X = F.cams.ray_plane_y0(F.d["y"])
    th_meas = np.arctan2(X[:, 0], -(X[:, 2] - P.get("dz", 0.0))) - P.get("th_off", 0.0)
    tg = np.arange(0, 961) / 96.0
    thg, _ = theta_at(tg, P["L_eq"], P["theta0"], P["omega0"], P["gamma"], 0.0)
    tm, dm = crossings(tg, thg); td, dd = crossings(F.t, th_meas)
    T_model = float(np.mean(np.diff(tm[dm > 0])))
    ph = []
    for tc, s_ in zip(td, dd):
        same = tm[dm == s_]
        j = np.argmin(np.abs(same - tc)); ph.append((float(tc), float((same[j] - tc) / T_model)))
    ph = np.array(ph)
    last = ph[ph[:, 0] >= 10.0 - T_model]
    out["phase"] = {"T_model_s": T_model, "per_crossing_t_s_and_cycles": ph.tolist(),
                    "phase_error_end_cycles": float(np.mean(last[:, 1])), "R3_phase_target_cycles": 0.05,
                    "note": "(model - data) zero-crossing time / T_model; data theta = pixel ray ∩ plane Y=0 (per-frame camera)"}
    out["R3_phase_pass"] = abs(out["phase"]["phase_error_end_cycles"]) < 0.05
    out["R3_pass"] = out["R3_rms_pass"] and out["R3_phase_pass"]
    return out, th_meas, uv, r


def bootstrap(F, R, mask, nboot, seed, block, fixed_Q):
    if nboot <= 0:
        return None
    uv, _, _ = F.model(R["P"])
    res = F.d["y"] - uv
    idx_fit = np.where(mask)[0]; n = len(idx_fit)
    rng = np.random.default_rng(seed); D = []
    for b in range(nboot):
        ii = []
        while len(ii) < n:
            s0 = rng.integers(0, n - block + 1); ii.extend(range(s0, s0 + block))
        yb = F.d["y"].copy()
        yb[idx_fit] = uv[idx_fit] + res[idx_fit][np.array(ii[:n])]
        Rb = F.fit(R["P"], R["names"], mask, fixed_gamma_Q=fixed_Q, y=yb)
        D.append([Rb["P"][k] for k in R["names"]] + [Rb["P"]["gamma"]])
    D = np.array(D); names = R["names"] + ["gamma_derived"]
    out = {"n": nboot, "block_len_frames": block, "method": "moving-block residual bootstrap on t<7.5 s, refit from best"}
    for i, k in enumerate(names):
        out[k] = {"sd": float(D[:, i].std(ddof=1)), "p2.5": float(np.percentile(D[:, i], 2.5)),
                  "p97.5": float(np.percentile(D[:, i], 97.5))}
    C = np.corrcoef(D[:, :len(R["names"])].T)
    out["corr"] = {a: {b: float(C[i, j]) for j, b in enumerate(R["names"])} for i, a in enumerate(R["names"])}
    return out


def flags_for(R, boot):
    fl = {}
    for n in R["names"]:
        f = []
        sd = max(R["sd_scaled"][n], ((boot or {}).get(n) or {}).get("sd", 0.0))
        if R["at_bound"][n]:
            f.append(f"AT_BOUND (constraint active; approx one-sided 95% upper limit {R['P'][n] + 1.645 * sd:.3g})")
        elif abs(R["P"][n]) > 0 and sd / abs(R["P"][n]) > 0.5:
            f.append(f"UNIDENTIFIED (relative sd {sd / abs(R['P'][n]):.2g})")
        for m, c in R["corr"][n].items():
            if m != n and abs(c) > 0.95:
                f.append(f"STRONGLY_CORRELATED with {m} ({c:+.3f})")
        fl[n] = f
    return fl


def strip(R):
    return {k: v for k, v in R.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=40)
    ap.add_argument("--ratio", type=float, default=12.30); ap.add_argument("--sigma-ratio", type=float, default=0.45)
    ap.add_argument("--out", default=str(HERE / "out" / "fit_projected"))
    ap.add_argument("--t-anchor", type=float, default=None,
                    help="fit window t <= T_a (anchored mode, CHANGELOG 2026-10-25T23:06:55Z); default: t < 7.5 s")
    ap.add_argument("--q-plausible", type=float, nargs=2, default=[100.0, 6000.0])
    ap.add_argument("--fit-tilt", action="store_true",
                    help="fit the static-scene level tilt th_off (CHANGELOG 2026-10-25T23:14:43Z): stand+camera rotated by "
                         "+th_off about world +Y through P; theta stays relative to true gravity")
    ap.add_argument("--damping", choices=["auto", "fitted", "assumed"], default="auto")
    a = ap.parse_args(argv)
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    data = load_track(ROOT / "measurement" / "bob_track.csv")
    frames = np.round(data["t"] * 24).astype(int)
    cams = Cams(frames)
    F = Fitter(data, cams)
    decl_fit = data["t"] < T_SPLIT; decl_hold = ~decl_fit          # R2/R3 segments exactly as declared
    if a.t_anchor is not None:
        fitm = data["t"] <= a.t_anchor + 1e-9
    else:
        fitm = decl_fit.copy()
    hold = decl_hold
    anchor = fitm

    # ---- initial guess from triangulated theta (plane Y = 0)
    X = cams.ray_plane_y0(data["y"])
    th_tri = np.arctan2(X[:, 0], -X[:, 2]); k0 = float(np.median(np.hypot(X[:, 0], X[:, 2])))
    tt = data["t"][fitm]; yy = th_tri[fitm]
    fs = lambda t, A, w, ph, c: A * np.cos(w * t + ph) + c
    best = None
    for ph0 in np.linspace(-math.pi, math.pi, 8, endpoint=False):
        try:
            p, _ = curve_fit(fs, tt, yy, p0=[0.18, 2 * math.pi / 1.40, ph0, 0.0], maxfev=20000)
            rr = np.sum((fs(tt, *p) - yy) ** 2)
            if best is None or rr < best[0]:
                best = (rr, p)
        except RuntimeError:
            pass
    A, w, ph, c0 = best[1]
    A = abs(A) if A > 0 else -A
    if best[1][0] < 0:
        ph += math.pi
    init = dict(L_eq=G / w ** 2 / (1 + A ** 2 / 16) ** 2, theta0=A * math.cos(ph), omega0=-A * w * math.sin(ph),
                gamma=0.0, k=k0)
    base = ["L_eq", "theta0", "omega0", "k"] + (["th_off"] if a.fit_tilt else [])
    if a.fit_tilt:
        init["th_off"] = 0.0126
    R0 = F.fit(init, base, fitm)
    st = dict(R0["P"]); st["gamma"] = 0.002
    R1 = F.fit(st, base + ["gamma"], fitm)
    sel = {"M0": {"bic": R0["bic"], "chi2": R0["chi2"], "reduced_chi2": R0["reduced_chi2"]},
           "M1": {"bic": R1["bic"], "chi2": R1["chi2"], "reduced_chi2": R1["reduced_chi2"], "gamma": R1["P"]["gamma"],
                  "gamma_sd_scaled": R1["sd_scaled"]["gamma"], "gamma_at_bound": R1["at_bound"]["gamma"]}}
    stat_ident = (not R1["at_bound"]["gamma"]) and R1["bic"] < R0["bic"] - 2.0 and \
        R1["P"]["gamma"] > 3 * R1["sd_scaled"]["gamma"]
    Q1 = math.sqrt(G / R1["P"]["L_eq"]) / (2 * R1["P"]["gamma"]) if R1["P"]["gamma"] > 0 else math.inf
    # damping evidence: bootstrap of M1, and M1 on the n = 0 mod 4 frames only (cadence check)
    n4 = (frames % 4) == 0
    R1n4 = F.fit(R1["P"], base + ["gamma"], fitm & n4)
    boot1 = bootstrap(F, R1, fitm, a.nboot, 11, block=int(round(1.4 * 24)), fixed_Q=None)
    g_lo = boot1["gamma_derived"]["p2.5"] if boot1 else None
    boot_ident = bool(boot1 and g_lo is not None and g_lo > 0 and boot1["gamma_derived"]["sd"] < 0.5 * max(R1["P"]["gamma"], 1e-30))
    plausible = a.q_plausible[0] <= Q1 <= a.q_plausible[1]
    gamma_identified = bool(stat_ident and boot_ident and plausible)
    sel["M1"]["Q_implied"] = Q1
    sel["damping_evidence"] = {
        "statistically_identified_BIC_3sigma": bool(stat_ident), "bootstrap_identified": boot_ident,
        "bootstrap_M1_gamma": (boot1 or {}).get("gamma_derived"),
        "bootstrap_M1_Q_95": ([math.sqrt(G / R1["P"]["L_eq"]) / (2 * boot1["gamma_derived"]["p97.5"]),
                               (math.sqrt(G / R1["P"]["L_eq"]) / (2 * g_lo) if g_lo and g_lo > 0 else math.inf)] if boot1 else None),
        "Q_plausible_range": a.q_plausible, "Q_plausible": bool(plausible),
        "M1_on_n_mod4_eq0_frames_only": {"gamma": R1n4["P"]["gamma"], "gamma_sd_scaled": R1n4["sd_scaled"]["gamma"],
                                         "Q": (math.sqrt(G / R1n4["P"]["L_eq"]) / (2 * R1n4["P"]["gamma"]) if R1n4["P"]["gamma"] > 0 else math.inf),
                                         "L_eq": R1n4["P"]["L_eq"], "n_frames": int((fitm & n4).sum())},
        "adopted": gamma_identified}
    if a.damping == "fitted":
        gamma_identified = True
    elif a.damping == "assumed":
        gamma_identified = False
    sel["damping_evidence"]["mode"] = a.damping
    sel["damping_evidence"]["adopted"] = bool(gamma_identified)
    if gamma_identified:
        prod_label, R = "M1 (gamma fitted)", R1; fixed_Q = None
    else:
        prod_label = f"MQ (Q fixed at declared assumption {Q_ASSUMED:g}; gamma not identified)"
        fixed_Q = Q_ASSUMED
        R = F.fit(R0["P"], base, fitm, fixed_gamma_Q=Q_ASSUMED)
    sens = {}
    for Qs in Q_RANGE:
        Rs = F.fit(R0["P"], base, fitm, fixed_gamma_Q=Qs)
        ms, *_ = metrics(F, Rs["P"], decl_fit, decl_hold, anchor)
        sens[f"Q={Qs:g}"] = {"params": Rs["P"], "rms_fit_px": ms["rms_fit_px"], "rms_holdout_px": ms["rms_holdout_px"],
                            "chi2": Rs["chi2"], "phase_end": ms["phase"]["phase_error_end_cycles"]}
    T_est = 2 * math.pi / math.sqrt(G / R["P"]["L_eq"])
    boot = bootstrap(F, R, fitm, a.nboot, 7, block=int(round(1.4 * 24)), fixed_Q=fixed_Q)
    M, th_meas, uv, resid = metrics(F, R["P"], decl_fit, decl_hold, anchor)
    # without the drift model (locked camera), same params, for contrast
    camsL = Cams(frames, kind="locked"); FL = Fitter(data, camsL)
    uvL, _, _ = FL.model(R["P"]); rL = uvL - data["y"]
    M["contrast_locked_camera_no_drift_model"] = {"rms_fit_px": rms(rL[decl_fit]), "rms_holdout_px": rms(rL[decl_hold]), "rms_anchor_px": rms(rL[anchor]),
                                                  "note": "same params projected with the drift-free camera (raw track still contains the zoom)"}
    M["note_drift"] = "primary residuals use the per-frame reference-matched camera, i.e. the measured camera drift IS removed"

    # diagnostics
    D1 = F.fit(dict(R["P"], dz=0.0), base + ["dz"], fitm, fixed_gamma_Q=fixed_Q)
    m1, *_ = metrics(F, D1["P"], decl_fit, decl_hold, anchor)
    D2 = F.fit(dict(D1["P"], phi=0.0), base + ["dz", "phi"], fitm, fixed_gamma_Q=fixed_Q)
    m2, *_ = metrics(F, D2["P"], decl_fit, decl_hold, anchor)
    D3 = F.fit(dict(R["P"], th_off=R["P"].get("th_off", 0.0)), sorted(set(base + ["th_off"]), key=NAMES_ALL.index), fitm, fixed_gamma_Q=fixed_Q)
    m3, *_ = metrics(F, D3["P"], decl_fit, decl_hold, anchor)
    diag = {"D3_free_equilibrium_offset": {"label": "DIAGNOSTIC ONLY: equilibrium direction offset th_off free (non-physical for a vertical-gravity pendulum; quantifies image-asymmetry of the reference swing)",
                                 "params": D3["P"], "sd_scaled": D3["sd_scaled"], "rms_fit_px": m3["rms_fit_px"],
                                 "rms_holdout_px": m3["rms_holdout_px"], "phase_end": m3["phase"]["phase_error_end_cycles"],
                                 "chi2": D3["chi2"]},
            "D1_free_pivot_dz": {"label": "DIAGNOSTIC ONLY (not production): pivot offset along world Z free",
                                 "params": D1["P"], "sd_scaled": D1["sd_scaled"], "rms_fit_px": m1["rms_fit_px"],
                                 "rms_holdout_px": m1["rms_holdout_px"], "phase_end": m1["phase"]["phase_error_end_cycles"],
                                 "chi2": D1["chi2"]},
            "D2_free_dz_phi": {"label": "DIAGNOSTIC ONLY: pivot dz + swing-plane azimuth phi free",
                               "params": D2["P"], "sd_scaled": D2["sd_scaled"], "rms_fit_px": m2["rms_fit_px"],
                               "rms_holdout_px": m2["rms_holdout_px"], "phase_end": m2["phase"]["phase_error_end_cycles"],
                               "chi2": D2["chi2"]}}

    # divergence map: measured turning points (frames from measurement/summary.json, refined by a parabola on the
    # triangulated angle) vs the corrected-physics model; and per-frame pixel distance
    summ = json.loads((ROOT / "measurement" / "summary.json").read_text())
    tgd = np.arange(0, 961) / 96.0
    thd, omd = theta_at(tgd, R["P"]["L_eq"], R["P"]["theta0"], R["P"]["omega0"], R["P"]["gamma"], 0.0)
    itp = np.where(np.sign(omd[:-1]) != np.sign(omd[1:]))[0]
    tp_model = [(tgd[i], thd[i] if abs(thd[i]) > abs(thd[i + 1]) else thd[i + 1]) for i in itp]
    div = []
    for tpd in summ["amplitude"]["turning_points"]:
        n0 = int(tpd["frame"]); js = np.arange(max(n0 - 3, 0), min(n0 + 4, len(th_meas)))
        c2, c1, c0 = np.polyfit(data["t"][js] - n0 / 24, th_meas[js], 2)
        tt0 = n0 / 24 - c1 / (2 * c2); A = c0 - c1 ** 2 / (4 * c2)
        same = [(tm_, am_) for tm_, am_ in tp_model if np.sign(am_) == np.sign(A)]
        tm_, am_ = min(same, key=lambda z: abs(z[0] - tt0))
        div.append({"t_data_s": float(tt0), "A_data_deg": float(np.degrees(A)), "t_model_s": float(tm_),
                    "A_model_deg": float(np.degrees(am_)), "dt_model_minus_data_s": float(tm_ - tt0),
                    "dA_data_minus_model_deg": float(np.degrees(abs(A) - abs(am_))),
                    "px_dist_at_frame": float(np.hypot(*resid[n0])), "in_anchor": bool(tt0 <= (a.t_anchor or 7.5))})
    divergence = {"turning_points": div, "per_frame_px_dist": np.hypot(resid[:, 0], resid[:, 1]).tolist(),
                  "note": "after the anchor window this is a DIVERGENCE MAP (AI motion vs corrected physics), not a fit target"}

    # growth / envelope of the measured (triangulated) angle
    env = _envelope(data["t"], th_meas, T_est)
    P = R["P"]
    ratio = a.ratio
    L_c = P["L_eq"] / (1 + 0.4 / ratio ** 2)
    sdL = max(R["sd_scaled"]["L_eq"], (boot or {}).get("L_eq", {}).get("sd", 0))
    dLc_ratio = L_c * (0.8 / ratio ** 3) * a.sigma_ratio / (1 + 0.4 / ratio ** 2)
    tg = np.arange(0, 961) / 96.0
    thg, _ = theta_at(tg, P["L_eq"], P["theta0"], P["omega0"], P["gamma"], 0.0)
    result = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {"bob_track": "measurement/bob_track.csv", "bob_track_sha256": sha(ROOT / "measurement/bob_track.csv"),
                   "camera_json_sha256": sha(ROOT / "params/camera.json"),
                   "camera_project_py_sha256": sha(ROOT / "camera/project.py"),
                   "ratio_Lc_over_r": [ratio, a.sigma_ratio], "ratio_source": "params/geometry.json (measured), per lead",
                   "track_method": "method A (u_px, v_px) with sigma_u_px/sigma_v_px"},
        "model": "pivot fixed at P_ring; bob = k (sin th, 0, -cos th) in L_eff units; passive rigid compound pendulum, damping >= 0",
        "fit_window": (f"ANCHOR t <= {a.t_anchor} s (frames 0..{int(round(a.t_anchor * 24))})" if a.t_anchor is not None else "t < 7.5 s (frames 0..179)"),
        "t_anchor_s": a.t_anchor,
        "declared_R2_segment": "t < 7.5 s", "declared_R3_segment": "t >= 7.5 s (predicted, no refit)",
        "holdout": "t >= 7.5 s (frames 180..240)",
        "initial_guess": init, "model_selection": sel, "gamma_identified": bool(gamma_identified),
        "level_tilt": ({"th_off_rad": R["P"]["th_off"], "sd_scaled": R["sd_scaled"].get("th_off"),
                        "convention": "static scene (stand, pivot hardware, table, cameras) rotated by +th_off about world +Y through P "
                                      "(right-hand rule, Blender rotation_euler[1] = +th_off); gravity exactly -Z_world; theta relative to gravity"}
                       if a.fit_tilt else None),
        "production_fit": prod_label, "params": P, "sd_scaled": R["sd_scaled"], "sd_formal": R["sd_formal"],
        "reduced_chi2": R["reduced_chi2"], "cond_J": R["cond_J"], "correlation": R["corr"],
        "bootstrap": boot, "flags": flags_for(R, boot),
        "Q_sensitivity": sens,
        "derived": {"L_eq_m": P["L_eq"], "L_eq_sd_m": sdL, "T0_small_angle_s": T_est,
                    "T_model_s": M["phase"]["T_model_s"], "L_c_m": L_c,
                    "L_c_sd_m_from_L_eq": sdL / (1 + 0.4 / ratio ** 2), "L_c_sd_m_from_ratio": dLc_ratio,
                    "r_m": L_c / ratio, "theta0_deg": math.degrees(P["theta0"]), "omega0_rad_s": P["omega0"],
                    "theta_max_abs_deg_0_10s": float(np.degrees(np.max(np.abs(thg)))),
                    "k_Lc_over_Leff": P["k"],
                    "camera_metre_scale_L_eff_m": L_c / P["k"],
                    "camera_rescale_note": "camera C_m = C_world_Leff * (L_c / k); camera.json currently uses L_eff_m = "
                                           f"{cams.Leff_m_json:.6f} -> camera role must re-run write_params with L_eff_m = L_c/k",
                    "Q_used": fixed_Q if fixed_Q else math.sqrt(G / P["L_eq"]) / (2 * P["gamma"]),
                    "provenance": "L_eq, L_c: model-conditioned (period + assumed g + rigid compound model + measured ratio)"},
        "metrics": M, "envelope_measured_theta": env, "diagnostics": diag, "divergence_map": divergence,
    }
    (out_dir / "fit_projected.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    np.savez(out_dir / "fit_projected_series.npz", t=data["t"], frames=frames, y=data["y"], s=data["s"], uv=uv,
             th_meas=th_meas, tg=tg, thg=thg, fitm=fitm, anchor=anchor,
             uv_D1=F.model(D1["P"])[0])
    print(json.dumps({"production": prod_label, "params": P, "sd_scaled": R["sd_scaled"],
                      "boot_sd": {k: v["sd"] for k, v in (boot or {}).items() if isinstance(v, dict) and "sd" in v},
                      "flags": result["flags"], "selection": sel, "derived": result["derived"],
                      "metrics": {k: M[k] for k in ("rms_fit_px", "rms_holdout_px", "R2_pass", "R3_pass", "R3_phase_pass",
                                                    "supplementary_n_mod4_eq0", "mean_residual_by_n_mod4_px",
                                                    "contrast_locked_camera_no_drift_model")},
                      "phase_end": M["phase"]["phase_error_end_cycles"], "envelope": env,
                      "diag": {k: {kk: v[kk] for kk in ("params", "rms_fit_px", "rms_holdout_px", "phase_end")} for k, v in diag.items()},
                      "Q_sens": {k: {kk: v[kk] for kk in ("rms_fit_px", "rms_holdout_px", "phase_end")} for k, v in sens.items()}},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
