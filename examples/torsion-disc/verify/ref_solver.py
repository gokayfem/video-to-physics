#!/usr/bin/env python
"""Verifier reference solver for the clip-03 torsion disc (independent of solver/).

Methods:  GL4 = 4-stage Gauss-Legendre collocation (order 8, implicit, symplectic; |q|^2 of each quaternion is an exact
                quadratic invariant -> attitude normalisation exact without projection), fixed-point iteration
          GBS = Gragg-Bulirsch-Stoer extrapolation (modified midpoint, Aitken-Neville in h^2, 8 columns)
Output grid: every 1/96 s from 0 to t_end (default 240.75/24 s, so frame n subframe s -> row 4(n+s)).
W_diss (dissipated work) and Tz_int (time integral of the torque about the vertical) are integrated as augmented states.

  .venv/bin/python verify/ref_solver.py --params params/physics.json [--model-json M.json] --out verify/out/ref.csv
  .venv/bin/python verify/ref_solver.py --model-json M.json --out X.csv [--method gbs]
  .venv/bin/python verify/ref_solver.py --selftest [--report verify/out/ref_solver_selftest.json]
"""
import argparse, datetime, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tmodel import Model, conical_rigid_pivot, conical_string, quat_to_R, rot_angle, axis_angle_quat, R_to_quat, EZ  # noqa: E402
from tio import write_traj, dump_json, fit_sinusoid, CANON  # noqa: E402
from tparams import load_model, sha256_file  # noqa: E402

OUT_DT = 1.0 / 96.0
T_END_DEFAULT = 240.75 / 24.0


# ------------------------------------------------------------------ integrators
def gauss_tableau(s):
    x, w = np.polynomial.legendre.leggauss(s)
    c = 0.5 * (x + 1); b = 0.5 * w
    A = np.zeros((s, s))
    for j in range(s):
        poly = np.poly1d([1.0])
        for m in range(s):
            if m != j:
                poly = poly * np.poly1d([1.0, -c[m]]) / (c[j] - c[m])
        P = poly.integ()
        for i in range(s):
            A[i, j] = P(c[i]) - P(0.0)
    return A, b, c


class Aug:
    """y' = f(y), W' = P_diss(y), Z' = tau_z(y): integrated by the same integrator (no sample quadrature)."""
    def __init__(self, model):
        self.m = model

    def initial_state(self):
        return np.r_[self.m.initial_state(), 0.0, 0.0]

    def rhs(self, y):
        y = np.asarray(y, float)
        out = np.empty_like(y)
        out[..., :-2], out[..., -2], out[..., -1] = self.m.rhs(y[..., :-2], aug=True)
        return out


def integrate_gl(model, t_end, sub=8, s=4, tol=1e-15, maxit=80, dt_out=OUT_DT):
    A, b, c = gauss_tableau(s)
    h = dt_out / sub
    nout = int(round(t_end / dt_out)) + 1
    y = model.initial_state().astype(float)
    Y = np.empty((nout, len(y))); Y[0] = y
    K = np.tile(model.rhs(y), (s, 1))
    maxits = 0
    for i in range(1, nout):
        for _ in range(sub):
            for it in range(maxit):
                Kn = model.rhs(y + h * (A @ K))
                dlt = np.max(np.abs(Kn - K)) * h
                K = Kn
                if dlt <= tol * (1.0 + np.max(np.abs(y))):
                    break
            maxits = max(maxits, it + 1)
            y = y + h * (b @ K)
        Y[i] = y
    return np.arange(nout) * dt_out, Y, dict(method=f"Gauss-Legendre s={s} (order {2*s})", h=h, sub=sub, max_fixed_point_iters=maxits)


def integrate_gbs(model, t_end, kmax=8, macro_sub=1, dt_out=OUT_DT):
    H = dt_out / macro_sub
    nout = int(round(t_end / dt_out)) + 1
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
    return np.arange(nout) * dt_out, Y, dict(method=f"Gragg-Bulirsch-Stoer, {kmax} columns (order {2*kmax})", H=H)


def integrate(model, t_end=T_END_DEFAULT, method="gl", **kw):
    return integrate_gl(model, t_end, **kw) if method == "gl" else integrate_gbs(model, t_end, **kw)


def solve(model, t_end=T_END_DEFAULT, method="gl", sub=4):
    """-> t, Y (model state), cols (canonical columns incl. W_diss_J and Tz_int_N_m_s)."""
    t, Ya, info = integrate(Aug(model), t_end, method, **({"sub": sub} if method == "gl" else {}))
    Y = Ya[:, :-2]
    cols = dict(t_s=t, **model.observables(Y))
    cols["W_diss_J"] = Ya[:, -2]; cols["Tz_int_N_m_s"] = Ya[:, -1]
    return t, Y, cols, info


def run_for_params(model, out, t_end=T_END_DEFAULT, method="gl", sub=4, meta_extra=None):
    t, Y, cols, info = solve(model, t_end, method, sub)
    write_traj(out, cols, [c for c in CANON if c in cols])
    meta = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                integrator=info, model=model.as_dict(), provenance=model.provenance,
                code_sha256={f: sha256_file(os.path.join(HERE, f)) for f in ("ref_solver.py", "tmodel.py", "tparams.py")},
                derived=dict(omega_t=model.omega_t, gamma_t=model.gamma_t, swing_modes_y=model.swing_modes("y")[0].tolist(),
                             E_static=model.E_static()))
    if meta_extra:
        meta.update(meta_extra)
    dump_json(meta, out + ".meta.json")
    return t, Y, cols


def lyapunov_estimate(model, t_end=10.0, d0=1e-10):
    """finite-time separation growth for a 1e-10 rad perturbation of the initial twist."""
    from tmodel import quat_mul
    m2 = model.replace(q0=quat_mul(axis_angle_quat(EZ, d0), model.q0).tolist())   # keep the quaternion hemisphere (lifted twist)
    _, _, O1, _ = solve(model, t_end, sub=4)
    _, _, O2, _ = solve(m2, t_end, sub=4)
    sep = np.abs(O1["psi_rad"] - O2["psi_rad"])
    return dict(growth_factor=float(sep.max() / d0), ftle_per_s=float(math.log(max(sep[-1], 1e-300) / d0) / O1["t_s"][-1]))


# ------------------------------------------------------------------ self-test (validation vs analytic limits)
RHO, RAD, TH = 8500.0, 0.05, 0.003       # brass disc 10 cm across, 3 mm thick (test values only)


def base(mode="rigid_pivot", **kw):
    M = RHO * math.pi * RAD ** 2 * TH
    Izz = 0.5 * M * RAD ** 2
    Ixx = 0.25 * M * RAD ** 2 + M * TH ** 2 / 12
    d = dict(mode=mode, L=0.25, M=M, I_c=[Ixx, Ixx, Izz], p_b=[0, 0, TH / 2], kappa=Izz * (2 * math.pi / 3.0) ** 2, twist="transport")
    if mode == "string":
        d["k_b"] = 0.0
    d.update(kw)
    return Model(**d)


def twist_q(psi):
    return axis_angle_quat(EZ, psi).tolist()


def selftest(report_path, quick=False):
    R = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
             code_sha256={f: sha256_file(os.path.join(HERE, f)) for f in ("ref_solver.py", "tmodel.py")}, tests={}, plots={})
    T, PL = R["tests"], R["plots"]
    TOL = 1e-7
    TE = 10.0

    def rec(name, status, **kw):
        T[name] = dict(status="PASS" if status else "FAIL", **kw)
        print(f"{name:46s} {T[name]['status']}  " + ", ".join(f"{k}={v:.3e}" for k, v in kw.items() if isinstance(v, float))[:170], flush=True)

    # B1 static
    for mode in ("rigid_pivot", "string"):
        m = base(mode, k_b=(2e-3 if mode == "string" else 0.0))
        _, _, O, _ = solve(m, 2.0, sub=2)
        l = m.L + m.p_b[2]
        dev = max(np.max(np.abs(O["x_m"])), np.max(np.abs(O["y_m"])), np.max(np.abs(O["z_m"] + l)), np.max(np.abs(O["psi_rad"])))
        rec(f"B1_static_{mode}", dev < 1e-12, max_dev=float(dev))

    # B2/B3 torsion exact (pure twist), small and large amplitude, damped
    for mode in ("rigid_pivot", "string"):
        for tw in (("transport", "vertical", "euler_zyx") if mode == "string" else ("transport",)):
            for amp in (0.05, 1.0):
                for damp in (False, True):
                    if (tw != "transport") and (amp != 1.0 or damp):
                        continue
                    kw = dict(c_t=2e-5, c_r=5e-6, c_rp=3e-5, c_v=0.004) if damp else {}
                    m = base(mode, twist=tw, q0=twist_q(amp), w0=[0, 0, 0.3], **kw)
                    t, _, O, _ = solve(m, TE, sub=2)
                    pe, wd, gam = m.torsion_exact(t, amp, 0.3)
                    err = np.max(np.abs(O["psi_rad"] - pe)) / amp
                    wfit, gfit, _, _ = fit_sinusoid(t, O["psi_rad"], wd, decay=damp)
                    sw = max(np.max(np.abs(O["ux"])), np.max(np.abs(O["uy"])), np.max(O["tilt_rad"]))
                    name = f"B{'3' if damp else '2'}_torsion_{mode}_{tw}_amp{amp:g}"
                    rec(name, err < TOL and abs(wfit / wd - 1) < TOL and sw < 1e-12, omega_t=m.omega_t, omega_d=wd, max_err_over_amp=float(err),
                        rel_omega_err=float(abs(wfit / wd - 1)), rel_gamma_err=float(abs(gfit / gam - 1)) if damp else 0.0, swing_tilt_leak=float(sw))
                    if mode == "rigid_pivot" and amp == 1.0:
                        PL[f"torsion_{'damped' if damp else 'undamped'}"] = dict(t=t[::2].tolist(), psi=O["psi_rad"][::2].tolist(), exact=pe[::2].tolist(),
                                                                                err=np.abs(O["psi_rad"] - pe)[::2].tolist(), amp=amp)

    # B4 small-angle swing (planar, rotation about y; and out-of-plane about x)
    for mode in ("rigid_pivot", "string", "string_prodlike"):
        m0 = base(mode, k_b=(2e-3 if mode == "string" else 0.0)) if mode != "string_prodlike" else \
            base("string", L=0.0909, k_a=0.0157, k_b=0.0157, bend_law="cos", twist="vertical")
        for ax in ("y", "x"):
            ws, V = m0.swing_modes(ax)
            amps = [3e-2, 1e-2, 3e-3, 1e-3, 1e-5] if (mode == "rigid_pivot" and ax == "y") else [1e-5]
            nm = len(ws)
            for j in range(nm):
                errs = []
                for th in amps:
                    axis = [0, 1, 0] if ax == "y" else [1, 0, 0]
                    if m0.mode == "rigid_pivot":
                        q0 = axis_angle_quat(axis, th); u0 = [0, 0, 1]
                    else:
                        ph1, ph2 = V[:, j] / np.max(np.abs(V[:, j])) * th
                        u0 = quat_to_R(axis_angle_quat(axis, ph1)) @ EZ
                        q0 = axis_angle_quat(axis, ph2)
                    m = m0.replace(q0=q0.tolist(), u0=list(u0))
                    t, _, O, _ = solve(m, TE, sub=2)
                    sig = O["x_m"] if ax == "y" else O["y_m"]
                    wfit, _, _, _ = fit_sinusoid(t, sig, ws[j], w_rel_window=0.05)
                    errs.append(abs(wfit / ws[j] - 1))
                    if th == amps[-1]:
                        tw_leak = float(np.max(np.abs(O["psi_rad"])))
                rec(f"B4_swing_{mode}_axis{ax}_mode{j}", errs[-1] < TOL and tw_leak < 1e-12, omega_pred=float(ws[j]), rel_err=float(errs[-1]),
                    psi_leak=tw_leak)
                if len(amps) > 1:
                    PL["B4"] = dict(amps=amps, errs=errs, omega=float(ws[j]))

    # B5 conical orbits (kappa = 0)
    for alpha_deg in (5.0, 30.0):
        m0 = base("rigid_pivot", kappa=0.0)
        cc = conical_rigid_pivot(m0, math.radians(alpha_deg))
        m = m0.replace(q0=cc["q0"], w0=cc["w0"])
        t, Y, O, _ = solve(m, TE, sub=2)
        rho = np.linalg.norm(cc["rho"][:2])
        ex = np.stack([cc["rho"][0] * np.cos(cc["Omega"] * t), cc["rho"][0] * np.sin(cc["Omega"] * t)], -1)
        err = np.max(np.linalg.norm(np.stack([O["x_m"], O["y_m"]], -1) - ex, axis=1)) / rho
        dz = np.max(np.abs(O["z_m"] - cc["rho"][2])) / rho
        lz = np.max(np.abs(O["Lz"] - cc["Lz"])) / abs(cc["Lz"])
        rec(f"B5_conical_rigid_pivot_{int(alpha_deg)}deg", err < TOL and dz < TOL and lz < TOL, Omega=cc["Omega"], max_err_over_rho=float(err), z_err=float(dz), Lz_rel_err=float(lz))
        if alpha_deg == 30.0:
            PL["conical"] = dict(x=O["x_m"][::4].tolist(), y=O["y_m"][::4].tolist(), err=(np.linalg.norm(np.stack([O["x_m"], O["y_m"]], -1) - ex, axis=1) / rho)[::4].tolist(), t=t[::4].tolist())
    for alpha_deg in (5.0, 20.0):
        m0 = base("string", kappa=0.0)
        cc = conical_string(m0, math.radians(alpha_deg))
        m = m0.replace(q0=cc["q0"], w0=cc["w0"], u0=cc["u0"], udot0=cc["udot0"])
        tend = TE
        t, Y, O, _ = solve(m, tend, sub=2)
        rho = np.hypot(cc["C"][0], cc["C"][1])
        ex = np.stack([cc["C"][0] * np.cos(cc["Omega"] * t), cc["C"][0] * np.sin(cc["Omega"] * t)], -1)
        e = np.linalg.norm(np.stack([O["x_m"], O["y_m"]], -1) - ex, axis=1) / rho
        Tn = np.max(np.abs(O["tension_N"] - cc["T"])) / cc["T"]
        rec(f"B5_conical_string_{int(alpha_deg)}deg", e.max() < TOL and Tn < TOL, Omega=cc["Omega"], tilt_rad=float(cc["theta"]), t_end=tend,
            max_err_over_rho=float(e.max()), tension_rel_err=float(Tn))

    # B6/B7 energy & momentum: generic 3-D motion, every mode/convention, undamped + damped
    def generic(mode, tw, **kw):
        q0 = quat_to_R(axis_angle_quat([0.3, -0.5, 1.0], 0.0)) @ np.eye(3)
        qd = axis_angle_quat(EZ, 0.5)
        qd = R_to_quat(quat_to_R(axis_angle_quat([1, 0.4, 0], 0.03)) @ quat_to_R(qd))
        extra = dict(u0=(quat_to_R(axis_angle_quat([0.2, 1, 0], 0.04)) @ EZ).tolist(), udot0=[0.05, -0.08, 0.0]) if mode == "string" else {}
        if mode == "string":
            u0 = np.array(extra["u0"]); ud = np.array(extra["udot0"]); extra["udot0"] = (ud - (ud @ u0) * u0).tolist()
        return base(mode, twist=tw, q0=qd.tolist(), w0=[0.1, -0.2, 0.4], **extra, **kw)
    gens = [("rigid_pivot", "transport", {}), ("rigid_pivot", "euler_zyx", {}), ("string", "transport", dict(k_b=2e-3)),
            ("string", "vertical", dict(k_b=2e-3, bend_law="cos")), ("string", "transport", {}),
            ("string", "vertical", dict(L=0.0909, k_a=0.0157, k_b=0.0157, bend_law="cos"))]
    PL["energy"] = {}
    for mode, tw, kw in gens:
        nm = f"{mode}_{tw}" + ("_ka" if kw.get("k_a") else "") + ("_kb" if kw.get("k_b") else "") + ("_cos" if kw.get("bend_law") == "cos" else "")
        m = generic(mode, tw, **kw)
        t, Y, O, _ = solve(m, TE, sub=2)
        Eosc = O["E_J"][0] - m.E_static()
        dE = np.max(np.abs(O["E_J"] - O["E_J"][0])) / Eosc
        Lref = math.sqrt(2 * Eosc * m.I_c[2, 2])
        bal = np.max(np.abs(O["Lz"] - O["Lz"][0] - O["Tz_int_N_m_s"])) / Lref
        qn = np.max(np.abs(np.linalg.norm(Y[:, (0 if mode == "rigid_pivot" else 6):(4 if mode == "rigid_pivot" else 10)], axis=1) - 1))
        rec(f"B6_energy_undamped_{nm}", dE < 1e-10, E_osc_J=float(Eosc), max_rel_drift=float(dE))
        rec(f"B7_Lz_balance_undamped_{nm}", bal < 1e-10, L_ref=float(Lref), max_rel_residual=float(bal),
            Lz_change_over_Lref=float(np.max(np.abs(O["Lz"] - O["Lz"][0])) / Lref))
        rec(f"GL4_quaternion_norm_{nm}", qn < 1e-13, max_abs_norm_minus_1=float(qn))
        PL["energy"][nm] = dict(t=t[::4].tolist(), dE=((O["E_J"] - O["E_J"][0]) / Eosc)[::4].tolist())
        if nm == "string_transport_kb":
            PL["conv_diff"] = dict(t=t[::4].tolist(), dV=(O["psi_vertical_rad"] - O["psi_transport_rad"])[::4].tolist(),
                                   dE=(O["psi_euler_rad"] - O["psi_transport_rad"])[::4].tolist())
            T["info_twist_convention_spread"] = dict(max_abs_vertical_minus_transport_rad=float(np.max(np.abs(O["psi_vertical_rad"] - O["psi_transport_rad"]))),
                                                     max_abs_euler_minus_transport_rad=float(np.max(np.abs(O["psi_euler_rad"] - O["psi_transport_rad"]))),
                                                     max_swing_rad=float(np.max(np.arccos(np.clip(O["uz"], -1, 1)))), max_tilt_rad=float(np.max(O["tilt_rad"])))
            # GL vs GBS on generic motion
            t2, Y2, O2, _ = solve(m, TE, method="gbs")
            dpos = np.max(np.linalg.norm(np.stack([O[k] - O2[k] for k in ("x_m", "y_m", "z_m")], -1), axis=1))
            qa = np.stack([O[k] for k in ("qw", "qx", "qy", "qz")], -1); qb = np.stack([O2[k] for k in ("qw", "qx", "qy", "qz")], -1)
            datt = float(np.max(rot_angle(quat_to_R(qa), quat_to_R(qb))))
            rec("GL_vs_GBS_string_generic", dpos < 1e-11 and datt < 1e-10, max_pos_diff_m=float(dpos), max_att_diff_rad=datt)
        # damped balance
        md = m.replace(c_t=2e-5, c_s=3e-4, c_v=0.01, c_r=5e-6, c_rp=3e-5)
        t, Y, O, _ = solve(md, TE, sub=2)
        Eosc = O["E_J"][0] - md.E_static()
        bal = np.max(np.abs(O["E_J"] + O["W_diss_J"] - O["E_J"][0])) / Eosc
        growth = np.max(np.diff(O["E_J"])) / Eosc
        lb = np.max(np.abs(O["Lz"] - O["Lz"][0] - O["Tz_int_N_m_s"])) / Lref
        rec(f"B6_energy_balance_damped_{nm}", bal < 1e-9 and growth <= 0, max_rel_residual=float(bal), max_forward_dE=float(growth),
            E_end_over_E0=float((O["E_J"][-1] - md.E_static()) / Eosc))
        rec(f"B7_Lz_balance_damped_{nm}", lb < 1e-10, max_rel_residual=float(lb))
        if nm == "string_transport_kb":
            PL["energy"]["damped"] = dict(t=t[::4].tolist(), E=((O["E_J"] - md.E_static()) / Eosc)[::4].tolist(),
                                         balance=((O["E_J"] + O["W_diss_J"] - O["E_J"][0]) / Eosc)[::4].tolist())
    # L_z strictly conserved with kappa = 0 (undamped)
    for mode in ("rigid_pivot", "string"):
        m = generic(mode, "transport", kappa=0.0)
        t, Y, O, _ = solve(m, TE, sub=2)
        Lref = abs(O["Lz"][0]) + 1e-30
        dl = np.max(np.abs(O["Lz"] - O["Lz"][0])) / Lref
        rec(f"B7_Lz_conserved_kappa0_{mode}", dl < 1e-10, Lz0=float(O["Lz"][0]), max_rel_drift=float(dl))

    # B8 decoupling: planar swing keeps psi = 0 and y = 0; twist + tiny swing frequencies
    for mode in ("rigid_pivot", "string"):
        m0 = base(mode)
        if mode == "rigid_pivot":
            m = m0.replace(q0=axis_angle_quat([0, 1, 0], 0.2).tolist())
        else:
            m = m0.replace(u0=(quat_to_R(axis_angle_quat([0, 1, 0], 0.2)) @ EZ).tolist(), q0=axis_angle_quat([0, 1, 0], 0.1).tolist())
        _, _, O, _ = solve(m, TE, sub=2)
        leak = max(np.max(np.abs(O["psi_rad"])), np.max(np.abs(O["y_m"])))
        rec(f"B8_planar_swing_no_twist_{mode}", leak < 1e-13, max_psi_or_y=float(leak))
        ws, V = m0.swing_modes("y")
        if mode == "rigid_pivot":
            m = m0.replace(q0=R_to_quat(quat_to_R(axis_angle_quat([0, 1, 0], 1e-5)) @ quat_to_R(axis_angle_quat(EZ, 0.05))).tolist())
        else:
            ph1, ph2 = V[:, 0] / np.max(np.abs(V[:, 0])) * 1e-5
            m = m0.replace(u0=(quat_to_R(axis_angle_quat([0, 1, 0], ph1)) @ EZ).tolist(),
                           q0=R_to_quat(quat_to_R(axis_angle_quat([0, 1, 0], ph2)) @ quat_to_R(axis_angle_quat(EZ, 0.05))).tolist())
        t, _, O, _ = solve(m, TE, sub=2)
        wt, _, _, _ = fit_sinusoid(t, O["psi_rad"], m0.omega_t, w_rel_window=0.05)
        rec(f"B8_twist_plus_tiny_swing_{mode}", abs(wt / m0.omega_t - 1) < TOL, omega_t=m0.omega_t, twist_rel_err=float(abs(wt / m0.omega_t - 1)))

    # GL4 convergence order (stiff variant so that h*omega ~ 1 at h = 1/96)
    md = generic("string", "transport", k_b=2e-3).replace(kappa=base().I_c[2][2] * 70.0 ** 2, c_t=2e-5, c_v=0.01)
    subs = [1, 2, 4]
    sols = [integrate_gl(md, 2.0, sub=sb)[1] for sb in subs]
    _, Yref, _ = integrate_gbs(md, 2.0, macro_sub=16)
    errs = [float(np.max(np.abs(S - Yref))) for S in sols]
    po = [math.log(errs[i] / errs[i + 1]) / math.log(2) for i in range(len(errs) - 1)]
    rec("GL4_order", abs(po[-1] - 8) <= 0.5 or abs(po[0] - 8) <= 0.5, err_h1=errs[0], err_h2=errs[1], err_h4=errs[2], p01=po[0], p12=po[1])
    T["GL4_order"].update(h=[OUT_DT / s for s in subs], case="string, kappa -> omega_t = 70 rad/s, k_b, damped, 3-D, 2 s; reference GBS H=1/1536")
    PL["order"] = dict(h=[OUT_DT / s for s in subs], err=errs)

    R["status"] = "PASS" if all(v.get("status", "PASS") == "PASS" for v in T.values()) else "FAIL"
    dump_json(R, report_path)
    print("SELFTEST", R["status"])
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params"); ap.add_argument("--model-json"); ap.add_argument("--out")
    ap.add_argument("--method", choices=["gl", "gbs"], default="gl"); ap.add_argument("--sub", type=int, default=4)
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
    extra = {"lyapunov": lyapunov_estimate(model)} if a.lyapunov else {}
    run_for_params(model, a.out, a.t_end, a.method, a.sub, extra)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
