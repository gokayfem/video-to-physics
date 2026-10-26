#!/usr/bin/env python
"""Verifier reference solver for the clip-02 elastic pendulum (independent of solver/).

Methods:  GL4  = 4-stage Gauss-Legendre collocation (order 8, implicit, symplectic), fixed-point iteration
          GBS  = Gragg-Bulirsch-Stoer extrapolation (modified midpoint, Aitken-Neville in h^2, 8 columns)
Output grid: every 1/96 s from 0 to t_end (default 240.75/24 s, so frame n subframe s -> row 4(n+s)).

  .venv/bin/python verify/ref_solver.py --params params/physics.json [--model-json M.json] --out verify/out/ref.csv
  .venv/bin/python verify/ref_solver.py --selftest --report verify/out/ref_solver_selftest.json
"""
import argparse, datetime, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vmodel import Model, axial_exact, conical, rigid_planar_modes, autoparametric_rate, quat_to_R, rot_angle, axis_angle_quat  # noqa: E402
from vio import write_traj, dump_json, fit_sinusoid, cumulative_integral_quintic, CANON  # noqa: E402
from vparams import load_model, sha256_file  # noqa: E402

OUT_DT = 1.0 / 96.0
T_END_DEFAULT = 240.75 / 24.0


# ------------------------------------------------------------------ Gauss-Legendre tableau
def gauss_tableau(s):
    x, w = np.polynomial.legendre.leggauss(s)
    c = 0.5 * (x + 1); b = 0.5 * w
    A = np.zeros((s, s))
    for j in range(s):
        others = [c[m] for m in range(s) if m != j]
        poly = np.poly1d([1.0])
        for cm in others:
            poly = poly * np.poly1d([1.0, -cm]) / (c[j] - cm)
        P = poly.integ()
        for i in range(s):
            A[i, j] = P(c[i]) - P(0.0)
    return A, b, c


def integrate_gl(model, t_end, sub=8, s=4, tol=1e-15, maxit=60):
    A, b, c = gauss_tableau(s)
    H = OUT_DT; h = H / sub
    nout = int(round(t_end / H)) + 1
    y = model.initial_state().astype(float)
    Y = np.empty((nout, len(y))); Y[0] = y
    K = np.tile(model.rhs(y), (s, 1))
    maxits = 0
    for i in range(1, nout):
        for _ in range(sub):
            Kn = K
            for it in range(maxit):
                Z = y + h * (A @ K)
                Kn = model.rhs(Z)
                dlt = np.max(np.abs(Kn - K)) * h
                K = Kn
                if dlt <= tol * (1.0 + np.max(np.abs(y))):
                    break
            maxits = max(maxits, it + 1)
            y = y + h * (b @ K)
            # predictor for next step: extrapolate stage derivatives (evaluate collocation poly) -> simple reuse
        Y[i] = y
    t = np.arange(nout) * H
    return t, Y, dict(method=f"Gauss-Legendre s={s} (order {2*s})", h=h, sub=sub, max_fixed_point_iters=maxits)


def integrate_gbs(model, t_end, kmax=8, macro_sub=1):
    H = OUT_DT / macro_sub
    nout = int(round(t_end / OUT_DT)) + 1
    y = model.initial_state().astype(float)
    Y = np.empty((nout, len(y))); Y[0] = y
    seq = [2 * (j + 1) for j in range(kmax)]
    for i in range(1, nout):
        for _ in range(macro_sub):
            T = []
            for j, n in enumerate(seq):
                hh = H / n
                z0 = y.copy(); z1 = y + hh * model.rhs(y)
                for _m in range(1, n):
                    z0, z1 = z1, z0 + 2 * hh * model.rhs(z1)
                row = [0.5 * (z1 + z0 + hh * model.rhs(z1))]
                for k in range(1, j + 1):
                    r = (seq[j] / seq[j - k]) ** 2
                    row.append(row[k - 1] + (row[k - 1] - T[j - 1][k - 1]) / (r - 1))
                T.append(row)
            y = T[-1][-1]
        Y[i] = y
    t = np.arange(nout) * OUT_DT
    return t, Y, dict(method=f"Gragg-Bulirsch-Stoer, {kmax} columns (order {2*kmax})", H=H)


def integrate(model, t_end=T_END_DEFAULT, method="gl", **kw):
    return integrate_gl(model, t_end, **kw) if method == "gl" else integrate_gbs(model, t_end, **kw)


def to_columns(model, t, Y):
    cols = dict(t_s=t, **model.observables(Y))
    if model.damped:
        cols["W_diss_J"] = dissipated_work(model, Y)
    else:
        cols["W_diss_J"] = np.zeros_like(t)
    return cols


class _WithWork:
    """augmented system y' = f(y), W' = P_diss(y): W integrated by the same integrator (no quadrature of samples)."""
    def __init__(self, model):
        self.m = model

    def initial_state(self):
        return np.r_[self.m.initial_state(), 0.0]

    def rhs(self, y):
        y = np.asarray(y, float)
        out = np.empty_like(y)
        out[..., :-1] = self.m.rhs(y[..., :-1])
        out[..., -1] = self.m.observables(y[..., :-1])["P_diss_W"]
        return out


def dissipated_work(model, Y=None, t_end=None, sub=8):
    """W(t_n) on the output grid, from the augmented GL4 integration (Y argument kept for API symmetry)."""
    n = len(Y) if Y is not None else int(round(t_end / OUT_DT)) + 1
    _, Ya, _ = integrate_gl(_WithWork(model), (n - 1) * OUT_DT, sub=sub)
    return Ya[:, -1]


def run_for_params(model, out, t_end=T_END_DEFAULT, method="gl", sub=8, meta_extra=None):
    t, Y, info = integrate(model, t_end, method, **({"sub": sub} if method == "gl" else {}))
    cols = to_columns(model, t, Y)
    write_traj(out, cols, [c for c in CANON if c in cols])
    meta = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                integrator=info, model=model.as_dict(), provenance=model.provenance,
                code_sha256={f: sha256_file(os.path.join(HERE, f)) for f in ("ref_solver.py", "vmodel.py", "vparams.py")},
                derived=dict(ell_eq=model.ell_eq, omega_z=model.omega_z, omega_s=model.omega_s, E_static=model.E_static()))
    if meta_extra:
        meta.update(meta_extra)
    dump_json(meta, out + ".meta.json")
    return t, Y, cols


def lyapunov_estimate(model, t_end=10.0, d0=1e-10):
    """finite-time separation growth for a 1e-10 m perturbation of the eye's x position (GL4)."""
    m2 = model.replace(E0=(model.E0 + np.array([d0, 0, 0])).tolist())
    t, Y1, _ = integrate_gl(model, t_end, sub=4)
    _, Y2, _ = integrate_gl(m2, t_end, sub=4)
    O1, O2 = model.observables(Y1), m2.observables(Y2)
    sep = np.linalg.norm(np.stack([O1[k] - O2[k] for k in ("x_m", "y_m", "z_m")], -1), axis=1)
    return dict(t=t.tolist()[::24], sep=sep.tolist()[::24], growth_factor=float(sep.max() / d0),
                ftle_per_s=float(math.log(max(sep[-1], 1e-300) / d0) / t[-1]))


# ------------------------------------------------------------------ self-test (validation vs analytic limits)
def base_point(**kw):
    d = dict(k=40.0, L0=0.20, M=0.50, m_s=0.0, F_i=0.0)
    d.update(kw)
    return Model(**d)


def selftest(report_path, quick=False):
    R = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
             code_sha256={f: sha256_file(os.path.join(HERE, f)) for f in ("ref_solver.py", "vmodel.py")}, tests={})
    T = R["tests"]
    TOL = 1e-7

    # B1 static equilibrium (point and rigid, with m_s and preload)
    for name, m in (("point", base_point(m_s=0.03, F_i=0.5)),
                    ("rigid", Model(k=40, L0=0.2, M=0.5, m_s=0.03, F_i=0.5, body="rigid", e_b=[0, 0, 0.04], I_c=[4e-4, 5e-4, 2e-4]))):
        t, Y, _ = integrate_gl(m, 2.0, sub=4)
        E = m.observables(Y)
        dev = np.max(np.abs(np.stack([E["x_m"], E["y_m"], E["z_m"] + m.ell_eq], -1)))
        T[f"B1_static_{name}"] = dict(ell_eq=m.ell_eq, max_dev_m=float(dev), rel=float(dev / m.ell_eq), status="PASS" if dev / m.ell_eq < TOL else "FAIL")

    # B2 axial exact: point undamped, point damped (c_a + c_v), rigid undamped, with spring mass
    cases = [("point_undamped", base_point(m_s=0.03)),
             ("point_damped", base_point(m_s=0.03, c_a=0.15, c_v=0.05)),
             ("rigid_undamped", Model(k=40, L0=0.2, M=0.5, m_s=0.03, body="rigid", e_b=[0, 0, 0.04], I_c=[4e-4, 5e-4, 2e-4]))]
    for name, m in cases:
        a = 0.03
        m = m.replace(E0=[0, 0, -(m.ell_eq + a)])
        for meth in (("gl", "gbs") if name == "point_undamped" else ("gl",)):
            t, Y, _ = integrate(m, 10.0, meth)
            z = m.observables(Y)["z_m"]
            ze, wd, gam = axial_exact(m, t, a)
            err = np.max(np.abs(z - ze)) / a
            wfit, gfit, _, _ = fit_sinusoid(t, z, wd, decay=m.damped)
            T[f"B2_axial_{name}_{meth}"] = dict(omega_z=m.omega_z, omega_d=wd, gamma=gam, max_err_over_amp=float(err),
                                                omega_fit=wfit, rel_omega_err=abs(wfit / wd - 1), gamma_fit=gfit,
                                                rel_gamma_err=(abs(gfit / gam - 1) if gam else None),
                                                status="PASS" if (err < TOL and abs(wfit / wd - 1) < TOL) else "FAIL")
            if name == "point_undamped" and meth == "gl":
                R["_plot_axial"] = dict(t=t[::2].tolist(), z=z[::2].tolist(), ze=ze[::2].tolist(), err=(np.abs(z - ze))[::2].tolist(), a=a)

    # B3 conical orbit exact
    for name, m0 in (("massless", base_point()), ("ms_preload", base_point(m_s=0.03, F_i=0.5))):
        for alpha_deg in (5.0, 30.0, 60.0):
            al = math.radians(alpha_deg)
            cc = conical(m0, al)
            m = m0.replace(E0=[cc["rho"], 0, cc["z"]], vE0=[0, cc["rho"] * cc["Omega"], 0])
            t, Y, _ = integrate_gl(m, 10.0, sub=8)
            O = m.observables(Y)
            ex = np.stack([cc["rho"] * np.cos(cc["Omega"] * t), cc["rho"] * np.sin(cc["Omega"] * t), np.full_like(t, cc["z"])], -1)
            err = np.max(np.linalg.norm(np.stack([O["x_m"], O["y_m"], O["z_m"]], -1) - ex, axis=1)) / cc["rho"]
            lz = np.max(np.abs(O["Lz"] - cc["Lz"])) / cc["Lz"]
            T[f"B3_conical_{name}_{int(alpha_deg)}deg"] = dict(Omega=cc["Omega"], rho=cc["rho"], max_err_over_rho=float(err), Lz_rel_err=float(lz),
                                                               status="PASS" if err < TOL and lz < TOL else "FAIL")

    # B4 small-angle swing frequency -> omega_s, error ~ theta^2
    m0 = base_point(m_s=0.03)
    amps = [1e-1, 3e-2, 1e-2, 1e-3, 1e-4]
    errs = []
    for th in amps:
        l = m0.ell_eq
        m = m0.replace(E0=[l * math.sin(th), 0, -l * math.cos(th)])
        t, Y, _ = integrate_gl(m, 10.0, sub=8)
        x = m.observables(Y)["x_m"]
        wfit, _, _, _ = fit_sinusoid(t, x, m.omega_s)
        errs.append(abs(wfit / m.omega_s - 1))
    T["B4_small_angle_swing"] = dict(omega_s=m0.omega_s, amplitudes_rad=amps, rel_err=errs,
                                     slope_loglog_last=float(math.log(errs[-3] / errs[-2]) / math.log(amps[-3] / amps[-2])) if errs[-2] > 0 else None,
                                     status="PASS" if errs[-1] < TOL else "FAIL")
    R["_plot_B4"] = dict(amps=amps, errs=list(errs))

    # B5 rigid linear modes (planar xz -> rotation about y) at 1e-5 rad
    mr = Model(k=40, L0=0.2, M=0.5, m_s=0.03, body="rigid", e_b=[0, 0, 0.04], I_c=[4e-4, 5e-4, 2e-4])
    wm, V = rigid_planar_modes(mr, "y")
    res5 = []
    for j in range(2):
        amp = 1e-5
        ph1, ph2 = V[:, j] / np.max(np.abs(V[:, j])) * amp
        l = mr.ell_eq
        E0 = [l * math.sin(ph1), 0, -l * math.cos(ph1)]
        q0 = axis_angle_quat([0, 1, 0], -ph2)  # rotation about +y by -phi: body 'down' vector (0,0,-1) -> (sin phi, 0, -cos phi)
        m = mr.replace(E0=E0, q0=q0.tolist())
        t, Y, _ = integrate_gl(m, 10.0, sub=8)
        O = m.observables(Y)
        wfit, _, _, _ = fit_sinusoid(t, O["x_m"], wm[j], w_rel_window=0.05)
        res5.append(dict(mode=j, omega_pred=float(wm[j]), omega_fit=wfit, rel_err=abs(wfit / wm[j] - 1)))
    T["B5_rigid_modes"] = dict(modes=res5, status="PASS" if all(r["rel_err"] < TOL for r in res5) else "FAIL")

    # B6/B7 energy & momentum: generic 3-D motion (point + rigid), undamped and damped
    gen_pt = base_point(m_s=0.03, F_i=0.2).replace(E0=[0.05, -0.03, -0.27], vE0=[0.1, 0.25, -0.05])
    gen_rg = Model(k=40, L0=0.2, M=0.5, m_s=0.03, F_i=0.2, body="rigid", e_b=[0.003, -0.002, 0.04], I_c=[4e-4, 5e-4, 2e-4],
                   E0=[0.05, -0.03, -0.27], vE0=[0.1, 0.25, -0.05], q0=axis_angle_quat([1, 2, 0.3], 0.2).tolist(), w0=[0.5, -0.3, 1.0])
    R["_plot_energy"] = {}
    for name, m in (("point", gen_pt), ("rigid", gen_rg)):
        t, Y, _ = integrate_gl(m, 10.0, sub=8)
        O = m.observables(Y)
        E0s = O["E_J"][0] - m.E_static()
        dE = np.max(np.abs(O["E_J"] - O["E_J"][0])) / E0s
        Lref = m.Mt * m.ell_eq * np.max(np.linalg.norm(np.stack([O["vx_m_s"], O["vy_m_s"], O["vz_m_s"]], -1), axis=1))
        dLz = np.max(np.abs(O["Lz"] - O["Lz"][0])) / Lref
        T[f"B6_energy_undamped_{name}"] = dict(E_osc_J=float(E0s), max_rel_drift=float(dE), status="PASS" if dE < 1e-10 else "FAIL")
        T[f"B7_Lz_undamped_{name}"] = dict(L_ref=float(Lref), max_rel_drift=float(dLz), status="PASS" if dLz < 1e-10 else "FAIL")
        R["_plot_energy"][name] = dict(t=t[::4].tolist(), dE=((O["E_J"] - O["E_J"][0]) / E0s)[::4].tolist(), dLz=((O["Lz"] - O["Lz"][0]) / Lref)[::4].tolist())
        # GL vs GBS agreement on generic motion
        _, Y2, _ = integrate_gbs(m, 10.0)
        O2 = m.observables(Y2)
        dpos = np.max(np.linalg.norm(np.stack([O[k] - O2[k] for k in ("x_m", "y_m", "z_m")], -1), axis=1))
        T[f"GL_vs_GBS_{name}"] = dict(max_pos_diff_m=float(dpos), status="PASS" if dpos < 1e-11 else "FAIL")
        if name == "rigid":
            qa = np.stack([O["qw"], O["qx"], O["qy"], O["qz"]], -1); qb = np.stack([O2["qw"], O2["qx"], O2["qy"], O2["qz"]], -1)
            T[f"GL_vs_GBS_{name}"]["max_att_diff_rad"] = float(np.max(rot_angle(quat_to_R(qa), quat_to_R(qb))))
            T[f"GL_quaternion_norm_{name}"] = dict(max_abs_norm_minus_1=float(np.max(np.abs(np.linalg.norm(Y[:, 6:10], axis=1) - 1))))
        # damped energy balance
        md = m.replace(c_v=0.04, c_a=0.12, c_r=(2e-4 if m.body == "rigid" else 0.0))
        t, Y, _ = integrate_gl(md, 10.0, sub=8)
        O = md.observables(Y)
        W = dissipated_work(md, Y)
        E0s = O["E_J"][0] - md.E_static()
        bal = np.max(np.abs(O["E_J"] + W - O["E_J"][0])) / E0s
        growth = np.max(np.diff(O["E_J"])) / E0s
        T[f"B6_energy_balance_damped_{name}"] = dict(max_rel_residual=float(bal), max_forward_dE_over_E0=float(growth),
                                                     E_end_over_E0=float((O["E_J"][-1] - md.E_static()) / E0s),
                                                     status="PASS" if bal < 1e-9 and growth <= 0 else "FAIL")
        R["_plot_energy"][name + "_damped"] = dict(t=t[::4].tolist(), E=((O["E_J"] - md.E_static()) / E0s)[::4].tolist(),
                                                   balance=((O["E_J"] + W - O["E_J"][0]) / E0s)[::4].tolist())
        # axial damping only keeps Lz
        ma = m.replace(c_a=0.12)
        t, Y, _ = integrate_gl(ma, 10.0, sub=8)
        O = ma.observables(Y)
        T[f"B7_Lz_axial_damping_only_{name}"] = dict(max_rel_drift=float(np.max(np.abs(O["Lz"] - O["Lz"][0])) / Lref),
                                                     status="PASS" if np.max(np.abs(O["Lz"] - O["Lz"][0])) / Lref < 1e-10 else "FAIL")
    # full L conservation with g = 0 (rigid, tests torque/inertia coupling in all axes)
    m0g = gen_rg.replace(g=0.0)
    t, Y, _ = integrate_gl(m0g, 10.0, sub=8)
    O = m0g.observables(Y)
    L = np.stack([O["Lx"], O["Ly"], O["Lz"]], -1)
    dL = np.max(np.linalg.norm(L - L[0], axis=1)) / np.linalg.norm(L[0])
    T["B7_full_L_g0_rigid"] = dict(max_rel_drift=float(dL), status="PASS" if dL < 1e-10 else "FAIL")

    # GL4 convergence order: stiff variant (k = 4000 N/m, omega_z ~ 88 rad/s) so that h*omega ~ 0.9 at h = 1/96
    md = gen_pt.replace(k=4000.0, c_v=0.04, c_a=0.12)
    md = md.replace(E0=[0.05, -0.03, -(md.ell_eq + 0.004)])
    subs = [1, 2, 4]
    sols = []
    for sb in subs:
        t, Y, _ = integrate_gl(md, 2.0, sub=sb)
        sols.append(Y)
    _, Yref, _ = integrate_gbs(md, 2.0, macro_sub=16)
    errs = [float(np.max(np.abs(S[:, :3] - Yref[:, :3]))) for S in sols]
    po = [math.log(errs[i] / errs[i + 1]) / math.log(2) for i in range(len(errs) - 1)]
    T["GL4_order"] = dict(case="k=4000 N/m, damped, 3-D, 2 s", h=[OUT_DT / s for s in subs], err_vs_GBS_m=errs, observed_order=po,
                          status="PASS" if abs(po[0] - 8) <= 0.5 else "FAIL")
    R["_plot_order"] = dict(h=[OUT_DT / s for s in subs], err=errs)

    # B8 autoparametric 2:1 resonance (point, m_s = 0): k = 3 M g / L0.
    # Averaged linear theory for release theta(0)=th0, theta'(0)=0, axial at max stretch:
    #   swing amplitude A(t) = th0 * sqrt(cosh(2 sigma t)),  sigma = (3/4) eps omega_s.
    M, L0 = 0.5, 0.2
    kres = 3 * M * 9.80665 / L0
    mres = Model(k=kres, L0=L0, M=M)
    rows = []
    from scipy.optimize import curve_fit
    for eps in ((0.04, 0.02) if quick else (0.04, 0.02, 0.01)):
        a = eps * mres.ell_eq
        seed = 1e-7
        l = mres.ell_eq + a
        m = mres.replace(E0=[l * math.sin(seed), 0, -l * math.cos(seed)])
        sig = autoparametric_rate(m, eps)
        tend = 6.0 / sig
        t, Y, _ = integrate_gl(m, tend, sub=2)
        O = m.observables(Y)
        th = np.arctan2(O["x_m"], -O["z_m"])
        Ts = 2 * math.pi / m.omega_s
        nper = int(round(Ts / OUT_DT))
        env_t, env = [], []
        for s0 in range(0, len(t) - nper, nper):
            seg = np.abs(th[s0:s0 + nper]); j = int(np.argmax(seg))
            env_t.append(t[s0 + j]); env.append(seg[j])
        env_t, env = np.array(env_t), np.array(env)
        f = lambda tt, la0, sg: la0 + 0.5 * np.log(np.cosh(2 * sg * tt))
        (la0, sfit), _ = curve_fit(f, env_t, np.log(env), p0=[math.log(seed), sig])
        rows.append(dict(eps=eps, sigma_pred=sig, sigma_meas=float(sfit), rel_err=abs(sfit / sig - 1), t_end=tend,
                         growth=float(env[-1] / env[0])))
        if eps == 0.02:
            R["_plot_autoparam"] = dict(env_t=env_t.tolist(), env=env.tolist(), fit=np.exp(f(env_t, la0, sfit)).tolist(),
                                        pred=(seed * np.sqrt(np.cosh(2 * sig * env_t))).tolist(), sigma=sig, eps=eps, seed=seed)
    # energy exchange (finite seed): fraction of oscillation energy in the swing
    eps, seed = 0.05, 1e-3
    l = mres.ell_eq + eps * mres.ell_eq
    m = mres.replace(E0=[l * math.sin(seed), 0, -l * math.cos(seed)])
    t, Y, _ = integrate_gl(m, 60.0, sub=2)
    O = m.observables(Y)
    th = np.arctan2(O["x_m"], -O["z_m"])
    Eosc = O["E_J"][0] - m.E_static()
    Eswing = 0.5 * m.Mt * (O["vx_m_s"] ** 2 + O["vy_m_s"] ** 2) + m.W * m.ell_eq * (1 - np.cos(th))  # lowest-order swing energy
    R["_plot_exchange"] = dict(t=t[::4].tolist(), swing_frac=(Eswing / Eosc)[::4].tolist(), z=(O["z_m"] + m.ell_eq)[::4].tolist(), theta=th[::4].tolist(), eps=eps, seed=seed)
    errs8 = [r["rel_err"] for r in rows]
    exch = float(np.max(Eswing / Eosc))
    T["B8_autoparametric"] = dict(k_resonant=kres, omega_z=mres.omega_z, two_omega_s=2 * mres.omega_s, rows=rows,
                                  exchange_eps=eps, exchange_seed=seed, max_swing_energy_fraction=exch,
                                  status="PASS" if (all(errs8[i + 1] < errs8[i] for i in range(len(errs8) - 1)) and errs8[-1] < 0.05 and exch > 0.9) else "FAIL",
                                  criterion="growth-rate rel error decreases monotonically with eps and < 5% at the smallest eps; swing energy fraction > 0.9")
    R["status"] = "PASS" if all(v.get("status", "PASS") == "PASS" for v in T.values()) else "FAIL"
    dump_json(R, report_path)
    for k, v in T.items():
        print(f"{k:42s} {v.get('status','-')}  " + ", ".join(f"{kk}={vv:.3e}" for kk, vv in v.items() if isinstance(vv, float))[:160])
    print("SELFTEST", R["status"])
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params"); ap.add_argument("--model-json"); ap.add_argument("--out")
    ap.add_argument("--method", choices=["gl", "gbs"], default="gl"); ap.add_argument("--sub", type=int, default=8)
    ap.add_argument("--t-end", type=float, default=T_END_DEFAULT)
    ap.add_argument("--lyapunov", action="store_true")
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--report")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.report or os.path.join(HERE, "out", "ref_solver_selftest.json")); return
    if a.params:
        model, _ = load_model(a.params, a.model_json)
    elif a.model_json:
        d = json.load(open(a.model_json)); model = Model(provenance={"model_json": a.model_json, "sha256": sha256_file(a.model_json)}, **d)
    else:
        raise SystemExit("need --params or --model-json")
    extra = {}
    if a.lyapunov:
        extra["lyapunov"] = lyapunov_estimate(model)
    run_for_params(model, a.out, a.t_end, a.method, a.sub, extra)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
