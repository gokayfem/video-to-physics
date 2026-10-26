#!/usr/bin/env python
"""Independent reference solutions for the planar rigid pendulum (verifier-owned, no solver/ imports).

    I_P th'' = -mgd sin th - c_v th' - k_q |th'| th' - tau_c sign(th')

Methods (none of them Runge-Kutta/DOP853 explicit schemes):
  * taylor_solve : recursive Taylor-series integrator (automatic differentiation of sin/cos), order N
  * gl_solve     : implicit Gauss-Legendre collocation, s stages (order 2s), Newton-solved
  * closed_form  : exact Jacobi-elliptic solution of the conservative pendulum,
                   sin(th/2) = k sn(wn t + u0 | m=k^2),  k^2 = sin^2(th0/2) + om0^2/(4 wn^2)  (k<1)
                   release from rest: th = 2 asin(k sn(K - wn t)), K = K(m=k^2)
Both integrators split the integration at every omega = 0 turning point, so the |omega| and
sign(omega) terms are smooth inside every segment (direction s is fixed per segment; analytic
continuation past the root is used only to locate it). Coulomb stick: if tau_c >= mgd|sin th*| at a
turning point the pendulum stays at rest.

CLI:
  python verify/ref_solver.py --params params/physics.json --out verify/out/ref_trajectory.csv
  python verify/ref_solver.py --selftest     # validation vs closed form, convergence, cross-method
"""
import argparse, json, math, sys, os, time
import numpy as np
from scipy.special import ellipk, ellipe, ellipj, ellipkinc
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vparams import Model, load_model, sha256_file  # noqa: E402


# ----------------------------------------------------------------------------------------------
# closed form and period formulas
def agm(a, b):
    for _ in range(60):
        a, b = 0.5 * (a + b), math.sqrt(a * b)
        if abs(a - b) <= 1e-17 * a:
            break
    return 0.5 * (a + b)


def period_exact(model, amp):
    """Exact conservative period at amplitude amp (rad): 4 sqrt(I/mgd) K(m=sin^2(amp/2))."""
    m = math.sin(amp / 2) ** 2
    return 4.0 * float(ellipk(m)) / model.wn


def period_agm(model, amp):
    """Same period by an independent route: T = 2 pi / (wn * AGM(1, cos(amp/2)))."""
    return 2 * math.pi / (model.wn * agm(1.0, math.cos(amp / 2)))


def period_small(model):
    return 2 * math.pi / model.wn


def conservative_amplitude(model, th0=None, om0=None):
    th0 = model.theta0 if th0 is None else th0
    om0 = model.omega0 if om0 is None else om0
    k2 = math.sin(th0 / 2) ** 2 + om0 ** 2 / (4 * model.wn ** 2)
    if k2 >= 1:
        raise ValueError("rotating (k>=1) regime: closed form for libration not valid")
    return 2 * math.asin(math.sqrt(k2))


def closed_form(model, t):
    """Exact conservative solution theta(t), omega(t) for any (th0, om0) with k<1."""
    t = np.asarray(t, float)
    wn = model.wn
    k2 = math.sin(model.theta0 / 2) ** 2 + model.omega0 ** 2 / (4 * wn ** 2)
    k = math.sqrt(k2)
    if k == 0:
        return np.zeros_like(t), np.zeros_like(t)
    # sn(u0) = sin(th0/2)/k, cn(u0) = om0/(2 k wn); amplitude phi0 = am(u0)
    phi0 = math.atan2(math.sin(model.theta0 / 2) / k, model.omega0 / (2 * k * wn))
    K = float(ellipk(k2))
    u0 = float(ellipkinc(phi0, k2))  # F(phi0|m); valid for phi0 in (-pi, pi]
    u = np.mod(wn * t + u0, 4 * K)  # reduce to one period for accuracy
    sn, cn, dn, ph = ellipj(u, k2)
    theta = 2 * np.arcsin(k * sn)
    omega = 2 * k * wn * cn
    return theta, omega


# ----------------------------------------------------------------------------------------------
# common segment logic
class _Rhs:
    def __init__(self, model):
        I = model.I_P
        self.a, self.b, self.q, self.tau = model.mgd / I, model.c_v / I, model.k_q / I, model.tau_c / I

    def f(self, th, om, s):
        return -self.a * math.sin(th) - self.b * om - self.q * s * om * om - self.tau * s

    def new_direction(self, th):
        """Direction after a turning point at angle th (omega=0). 0 means at rest / stuck."""
        acc_grav = -self.a * math.sin(th)
        if self.tau > 0 and abs(acc_grav) <= self.tau:
            return 0.0
        if acc_grav == 0.0:
            return 0.0
        return math.copysign(1.0, acc_grav)


def _initial_direction(rhs, th, om):
    if om != 0.0:
        return math.copysign(1.0, om)
    return rhs.new_direction(th)


# ----------------------------------------------------------------------------------------------
# Taylor series integrator
def _taylor_coeffs(rhs, th0, om0, s, N):
    th = np.zeros(N + 1); om = np.zeros(N + 1); S = np.zeros(N + 1); C = np.zeros(N + 1)
    th[0], om[0] = th0, om0
    S[0], C[0] = math.sin(th0), math.cos(th0)
    a, b, q, tau = rhs.a, rhs.b, rhs.q, rhs.tau
    j = np.arange(N + 1, dtype=float)
    for k in range(N):
        w2 = float(np.dot(om[:k + 1], om[k::-1]))
        F = -a * S[k] - b * om[k] - q * s * w2 - (tau * s if k == 0 else 0.0)
        th[k + 1] = om[k] / (k + 1)
        om[k + 1] = F / (k + 1)
        kk = k + 1
        jt = j[1:kk + 1] * th[1:kk + 1]
        S[kk] = float(np.dot(jt, C[kk - 1::-1][:kk])) / kk
        C[kk] = -float(np.dot(jt, S[kk - 1::-1][:kk])) / kk
    return th, om


def _horner(c, h):
    r = 0.0
    for x in c[::-1]:
        r = r * h + x
    return r


def _dhorner(c, h):
    r = 0.0
    n = len(c)
    for k in range(n - 1, 0, -1):
        r = r * h + k * c[k]
    return r


def _taylor_step(rhs, th, om, s, h, N):
    """Advance by h unless a turning point occurs; returns (h_used, th, om, event)."""
    cth, com = _taylor_coeffs(rhs, th, om, s, N)
    om_end = _horner(com, h)
    if s != 0 and (om_end * s <= 0.0):
        # locate root of omega polynomial in (0, h]
        lo, hi = 0.0, h
        x = h * abs(om) / (abs(om) + abs(om_end)) if (abs(om) + abs(om_end)) > 0 else h
        for _ in range(100):
            fx = _horner(com, x)
            if fx * s > 0:
                lo = x
            else:
                hi = x
            d = _dhorner(com, x)
            xn = x - fx / d if d != 0 else 0.5 * (lo + hi)
            if not (lo < xn < hi):
                xn = 0.5 * (lo + hi)
            if abs(xn - x) <= 1e-18 + 1e-16 * abs(x):
                x = xn
                break
            x = xn
        return x, _horner(cth, x), 0.0, True
    return h, _horner(cth, h), om_end, False


def taylor_solve(model, t_out, order=24, substeps=2):
    return _integrate(model, t_out, substeps, lambda rhs, th, om, s, h: _taylor_step(rhs, th, om, s, h, order))


# ----------------------------------------------------------------------------------------------
# Gauss-Legendre collocation
def gl_tableau(s):
    x, w = np.polynomial.legendre.leggauss(s)
    c = (x + 1) / 2
    A = np.zeros((s, s)); b = np.zeros(s)
    for jj in range(s):
        others = [c[m] for m in range(s) if m != jj]
        p = np.poly1d([1.0])
        for cm in others:
            p = p * np.poly1d([1.0, -cm]) / (c[jj] - cm)
        P = p.integ()
        b[jj] = P(1.0) - P(0.0)
        for i in range(s):
            A[i, jj] = P(c[i]) - P(0.0)
    return A, b, c


_GL_CACHE = {}


def _gl_step_raw(rhs, th, om, s, h, stages):
    if stages not in _GL_CACHE:
        _GL_CACHE[stages] = gl_tableau(stages)
    A, b, c = _GL_CACHE[stages]
    n = stages
    y0 = np.array([th, om])
    # stage increments Z (n x 2): Y_i = y0 + Z_i ; Z = h A F(Y)
    Z = np.zeros((n, 2))
    # initial guess: explicit linear extrapolation
    f0 = np.array([om, rhs.f(th, om, s)])
    Z = h * np.outer(c, f0)
    for it in range(50):
        Y = y0 + Z
        F = np.array([[Y[i, 1], rhs.f(Y[i, 0], Y[i, 1], s)] for i in range(n)])
        G = (Z - h * A @ F).ravel()
        # Jacobian of G wrt Z: I - h (A kron J_i) with J at each stage
        Jm = np.eye(2 * n)
        for i in range(n):
            for jj in range(n):
                Jj = np.array([[0.0, 1.0], [-rhs.a * math.cos(Y[jj, 0]), -rhs.b - 2 * rhs.q * s * Y[jj, 1]]])
                Jm[2 * i:2 * i + 2, 2 * jj:2 * jj + 2] -= h * A[i, jj] * Jj
        dZ = np.linalg.solve(Jm, -G).reshape(n, 2)
        Z = Z + dZ
        if np.max(np.abs(dZ)) <= 1e-17 + 2e-16 * np.max(np.abs(Z)):
            break
    Y = y0 + Z
    F = np.array([[Y[i, 1], rhs.f(Y[i, 0], Y[i, 1], s)] for i in range(n)])
    y1 = y0 + h * (b @ F)
    return y1[0], y1[1]


def _gl_step(rhs, th, om, s, h, stages):
    th1, om1 = _gl_step_raw(rhs, th, om, s, h, stages)
    if s != 0 and om1 * s <= 0.0:
        g = lambda x: _gl_step_raw(rhs, th, om, s, x, stages)[1] if x > 0 else om
        if om == 0.0:
            # started at a turning point: omega moves away with sign s immediately; no event
            return h, th1, om1, False
        x = brentq(g, 0.0, h, xtol=1e-18, rtol=1e-15, maxiter=200)
        thx, _ = _gl_step_raw(rhs, th, om, s, x, stages)
        return x, thx, 0.0, True
    return h, th1, om1, False


def gl_solve(model, t_out, stages=4, substeps=2):
    return _integrate(model, t_out, substeps, lambda rhs, th, om, s, h: _gl_step(rhs, th, om, s, h, stages))


# ----------------------------------------------------------------------------------------------
def _integrate(model, t_out, substeps, step):
    """Integrate exactly onto t_out (must start at 0) with `substeps` equal steps per interval,
    splitting at turning points."""
    t_out = np.asarray(t_out, float)
    assert t_out[0] == 0.0
    rhs = _Rhs(model)
    th, om = model.theta0, model.omega0
    s = _initial_direction(rhs, th, om)
    TH = np.empty_like(t_out); OM = np.empty_like(t_out); AL = np.empty_like(t_out)
    events = []
    def store(i):
        TH[i], OM[i] = th, om
        AL[i] = rhs.f(th, om, s) if s != 0 else 0.0
    store(0)
    for i in range(1, len(t_out)):
        t0, t1 = t_out[i - 1], t_out[i]
        t = t0
        n_left = substeps
        while n_left > 0:
            if s == 0:  # at rest (stuck or equilibrium)
                break
            # remaining time split evenly over remaining substeps (keeps grid exact)
            h = (t1 - t) / n_left
            hu, th, om_new, ev = step(rhs, th, om, s, h)
            om = om_new
            if ev:
                t = t + hu
                events.append(t)
                s = rhs.new_direction(th)
                om = 0.0
                if abs(t1 - t) < 1e-15:
                    n_left = 0
            else:
                t = t + h
                n_left -= 1
        store(i)
    return TH, OM, AL, np.array(events)


# ----------------------------------------------------------------------------------------------
def selftest(report_path=None):
    rep = {}
    t = np.arange(0, 961) / 96.0
    # representative model: brass bob, L ~ 0.39 m (T ~ 1.25 s)
    L, m, r = 0.39, 0.10, 0.012
    base = Model(I_P=m * (L * L + 0.4 * r * r), mgd=m * 9.80665 * L, theta0=math.radians(6))
    # (1) closed form vs both integrators, several amplitudes and an om0 != 0 case
    cf = {}
    for deg, om0 in [(0.5, 0), (2, 0), (6, 0), (15, 0), (60, 0), (170, 0), (6, 0.3), (-4, -0.2)]:
        mdl = base.replace(theta0=math.radians(deg), omega0=om0)
        thc, omc = closed_form(mdl, t)
        tt0 = time.time(); tht, omt, _, _ = taylor_solve(mdl, t); tT = time.time() - tt0
        tt0 = time.time(); thg, omg, _, _ = gl_solve(mdl, t); tG = time.time() - tt0
        cf[f"theta0={deg}deg,omega0={om0}"] = dict(
            taylor_vs_closed_theta=float(np.max(np.abs(tht - thc))),
            gl4_vs_closed_theta=float(np.max(np.abs(thg - thc))),
            taylor_vs_gl4_theta=float(np.max(np.abs(tht - thg))),
            taylor_vs_closed_omega=float(np.max(np.abs(omt - omc))),
            closed_form_t0_theta_err=float(abs(thc[0] - mdl.theta0)),
            closed_form_t0_omega_err=float(abs(omc[0] - mdl.omega0)),
            runtime_s=[round(tT, 2), round(tG, 2)])
    rep["closed_form_vs_integrators_0_10s"] = cf
    # closed form satisfies the ODE: residual of th'' + wn^2 sin th using omega' by finite diff of exact omega
    mdl = base
    tt = np.linspace(0, 10, 20001)
    th, om = closed_form(mdl, tt)
    e = 0.5 * mdl.I_P * om ** 2 + mdl.mgd * (1 - np.cos(th))
    rep["closed_form_energy_rel_spread"] = float((e.max() - e.min()) / e[0])
    # (2) convergence orders: coarse output grid (0.5 s) so truncation error dominates round-off
    conv = {}
    tc = np.arange(0, 21) * 0.5
    mm = base.replace(theta0=math.radians(30), c_v=2 * base.I_P * 0.02, k_q=base.I_P * 0.05)  # damped, with turning-point events
    ref_c = taylor_solve(mm, tc, order=30, substeps=16)[0]
    conv["reference"] = "Taylor order 30, h=1/32 s (damped 30 deg case with c_v and k_q)"
    for name, fn, subs, p in [("gl4", lambda n: gl_solve(mm, tc, 4, n), [4, 8, 16], 8),
                              ("gl2", lambda n: gl_solve(mm, tc, 2, n), [8, 16, 32], 4),
                              ("taylor_N6", lambda n: taylor_solve(mm, tc, order=6, substeps=n), [4, 8, 16], 6)]:
        errs = [float(np.max(np.abs(fn(n)[0] - ref_c))) for n in subs]
        conv[name] = dict(h_s=[0.5 / n for n in subs], err=errs,
                          observed_order=[math.log2(errs[i] / errs[i + 1]) for i in range(2)], theoretical=p)
    mm0 = base.replace(theta0=math.radians(30))
    thc, _ = closed_form(mm0, tc)
    conv["taylor_order_sweep_h0.125_vs_closed"] = {N: float(np.max(np.abs(taylor_solve(mm0, tc, order=N, substeps=4)[0] - thc))) for N in (6, 10, 14, 18, 24)}
    rep["convergence"] = conv
    # (3) damped cases: Taylor vs GL4 (no closed form); includes non-smooth k_q and Coulomb
    damp = {}
    I = base.I_P
    for label, kw in [("viscous", dict(c_v=2 * I * 0.005)), ("viscous_x10", dict(c_v=2 * I * 0.05)),
                      ("quadratic", dict(k_q=I * 0.05)), ("coulomb", dict(tau_c=base.mgd * 2e-4)),
                      ("coulomb_stick", dict(tau_c=base.mgd * 0.02)),
                      ("all", dict(c_v=2 * I * 0.005, k_q=I * 0.05, tau_c=base.mgd * 1e-4))]:
        mm = base.replace(**kw)
        a = taylor_solve(mm, t); b = gl_solve(mm, t)
        bb = gl_solve(mm, t, 4, 4)
        damp[label] = dict(taylor_vs_gl4=float(np.max(np.abs(a[0] - b[0]))),
                           gl4_n2_vs_n4=float(np.max(np.abs(b[0] - bb[0]))),
                           n_turning_points=int(len(a[3])), final_theta=float(a[0][-1]))
    rep["damped_cross_method"] = damp
    # (4) period formulas: ellipk vs AGM, small-angle limit
    per = {}
    for deg in (1e-3, 0.01, 0.1, 1, 2, 6, 15, 60, 170):
        A = math.radians(deg)
        T1, T2, T0 = period_exact(base, A), period_agm(base, A), period_small(base)
        per[f"{deg}deg"] = dict(T_ellipk=T1, rel_ellipk_vs_agm=abs(T1 - T2) / T2,
                                T_over_T0_minus_1=T1 / T0 - 1, series_A2_16=A * A / 16 + 11 * A ** 4 / 3072,
                                A6_coeff_observed=(T1 / T0 - 1 - (A * A / 16 + 11 * A ** 4 / 3072)) / max(A ** 6, 1e-300),
                                A6_coeff_theory=173 / 737280)
    rep["period_formulas"] = per
    if report_path:
        with open(report_path, "w") as f:
            json.dump(rep, f, indent=2)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params")
    ap.add_argument("--model-json", help="explicit override of model constants (JSON)")
    ap.add_argument("--out")
    ap.add_argument("--rate", type=float, default=96.0)
    ap.add_argument("--n", type=int, default=964, help="number of samples (default: 0..963 -> covers frame 240 subframe .75)")
    ap.add_argument("--method", choices=["taylor", "gl4"], default="taylor")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--report")
    a = ap.parse_args()
    if a.selftest:
        rep = selftest(a.report)
        print(json.dumps(rep, indent=1))
        return
    model, _ = load_model(a.params, a.model_json)
    t = np.arange(a.n) / a.rate
    th, om, al, ev = (taylor_solve if a.method == "taylor" else gl_solve)(model, t)
    th2, om2, _, _ = (gl_solve if a.method == "taylor" else taylor_solve)(model, t)
    cross = float(np.max(np.abs(th - th2)))
    with open(a.out, "w") as f:
        f.write("t_s,theta_rad,omega_rad_s,alpha_rad_s2\n")
        for i in range(len(t)):
            f.write(",".join(repr(float(v)) for v in (t[i], th[i], om[i], al[i])) + "\n")
    meta = dict(model=model.as_dict(), provenance=model.provenance, method=a.method,
                cross_method_max_dtheta=cross, n_turning_points=int(len(ev)), rate=a.rate, n=a.n,
                ref_solver_sha256=sha256_file(os.path.abspath(__file__)),
                vparams_sha256=sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vparams.py")))
    with open(a.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
