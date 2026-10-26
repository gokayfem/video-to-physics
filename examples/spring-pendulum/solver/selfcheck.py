"""Author-side developer checks of solver/spring_pendulum.py (P1, P3, P4, P7, P8; the verifier does its own).

Writes solver/out/selfcheck.json and solver/out/selfcheck.png.
Usage: ROOT/.venv/bin/python solver/selfcheck.py [--quick]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import spring_pendulum as sp  # noqa: E402
from run import time_grid, solver_source_sha256, sha256_file  # noqa: E402


def maxdiff(Ya, Yb):
    """max over samples of position-like differences: X, H (m) and n (rad)."""
    dX = np.max(np.abs(Ya[:, 0:3] - Yb[:, 0:3]))
    dn = np.max(np.abs(Ya[:, 6:9] - Yb[:, 6:9]))
    return float(dX), float(dn)


def convergence(Ms, names, t, hs):
    runs = [sp.integrate(Ms, t, h) for h in hs]           # (N,17,B)
    out = {}
    for j, nm in enumerate(names):
        rows = []
        for i in range(len(hs) - 1):
            dX, dn = maxdiff(runs[i][:, :, j], runs[i + 1][:, :, j])
            rows.append({"h_coarse_s": hs[i], "h_fine_s": hs[i + 1], "max_dX_m": dX, "max_dn": dn})
        for i in range(1, len(rows)):
            a, b = rows[i - 1]["max_dX_m"], rows[i]["max_dX_m"]
            rows[i]["observed_order_X"] = math.log2(a / b) if b > 0 else None
        good = [r["observed_order_X"] for r in rows if r.get("observed_order_X") and r["max_dX_m"] > 1e-11]
        fin = rows[-1]
        out[nm] = {"ladder": rows, "finest_pair_max_dX_m": fin["max_dX_m"], "finest_pair_max_dn": fin["max_dn"],
                   "observed_orders_above_roundoff": good,
                   "pass_tol_1e-7": bool(fin["max_dX_m"] < 1e-7 and fin["max_dn"] < 1e-7),
                   "pass_order_within_0.5": bool(good and all(abs(o - 4) <= 0.5 for o in good))}
    return out


def crossing_period(t, y, yd, direction=+1):
    """mean period from zero crossings, located by cubic Hermite interpolation on (y, y')."""
    tc = []
    for i in range(len(t) - 1):
        if y[i] * y[i + 1] < 0 and np.sign(yd[i]) == direction:
            h = t[i + 1] - t[i]
            p0, p1, m0, m1 = y[i], y[i + 1], yd[i] * h, yd[i + 1] * h
            s = -p0 / (p1 - p0)
            for _ in range(50):
                h00 = 2 * s ** 3 - 3 * s ** 2 + 1; h10 = s ** 3 - 2 * s ** 2 + s; h01 = -2 * s ** 3 + 3 * s ** 2; h11 = s ** 3 - s ** 2
                f = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
                df = (6 * s ** 2 - 6 * s) * p0 + (3 * s ** 2 - 4 * s + 1) * m0 + (-6 * s ** 2 + 6 * s) * p1 + (3 * s ** 2 - 2 * s) * m1
                ds = f / df; s -= ds
                if abs(ds) < 1e-15:
                    break
            tc.append(t[i] + s * h)
    tc = np.array(tc)
    return float((tc[-1] - tc[0]) / (len(tc) - 1)), len(tc) - 1


def mode_state(M, ell, V, j, amp):
    phi, psi = amp * V[0, j] / np.max(np.abs(V[:, j])), amp * V[1, j] / np.max(np.abs(V[:, j]))
    H = ell * np.array([math.sin(phi), 0, -math.cos(phi)]); n = np.array([math.sin(psi), 0, -math.cos(psi)])
    return replace(M, X0=tuple(H + M.b * n), V0=(0.0, 0.0, 0.0), n0=tuple(n), u0=(0.0, 0.0, 0.0))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--params", default=str(ROOT / "params" / "physics.json"))
    a = ap.parse_args(argv)
    params = json.loads(Path(a.params).read_text())
    Mp = sp.model_from_physics_json(params)
    h_prod = float(params.get("integrator", {}).get("h_s", sp.DEFAULT_H))
    t = time_grid(10.0, 96)
    res = {"created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "solver_id": sp.SOLVER_ID, "solver_source_sha256": solver_source_sha256(),
           "params_file": str(Path(a.params).resolve().relative_to(ROOT)), "params_sha256": sha256_file(a.params),
           "params_status": params.get("status"), "h_production_s": h_prod, "theoretical_order": 4, "checks": {}}
    C = res["checks"]

    # test variants (same batch: rigid bodies)
    strong = replace(Mp, c_s=Mp.c_s + 0.5, c_x=Mp.c_x + 0.05, c_r=Mp.c_r + 2e-4)
    big3d = replace(Mp, V0=(Mp.V0[0], Mp.V0[1] + 0.15, Mp.V0[2]))          # out-of-plane, 3D motion
    cons = replace(Mp, c_s=0.0, c_x=0.0, c_r=0.0)
    names = ["production_params", "strong_damping", "out_of_plane_3d"]
    hs = [h_prod * 16, h_prod * 8, h_prod * 4, h_prod * 2, h_prod] + ([] if a.quick else [h_prod / 2])
    conv = convergence([Mp, strong, big3d], names, t, hs)
    H0 = np.array(Mp.X0) - Mp.b * np.array(Mp.n0); Hd0 = np.array(Mp.V0) - Mp.b * np.array(Mp.u0)
    pt = replace(Mp, body="point", b=0.0, I_perp=0.0, X0=tuple(H0), V0=tuple(Hd0))   # point at the hinge (spring stays in tension)
    conv.update(convergence([pt], ["point_mass_3d"], t, hs))
    C["P1_convergence"] = conv
    C["P1_note"] = ("ladder h_prod*16 ... finest; finest pair = (h_prod, h_prod/2) unless --quick. "
                    "Production h = %.6g s; the production-vs-finer difference IS the finest pair." % h_prod)

    # ---- P3 analytic limits
    p3 = {}
    # (a) static equilibrium: start at equilibrium, must stay
    d0 = Mp.derived(); ell = d0["ell_eq_m"]
    eq = replace(Mp, X0=(0.0, 0.0, -ell - Mp.b), V0=(0.0, 0.0, 0.0), n0=(0.0, 0.0, -1.0), u0=(0.0, 0.0, 0.0))
    Ye = sp.integrate(eq, t, h_prod)
    p3["static_equilibrium"] = {"delta_formula": "(m + m_s/2) g / k", "static_stretch_m": d0["static_stretch_m"],
                                "max_drift_m": float(np.max(np.abs(Ye[:, 0:3] - np.array(eq.X0)))),
                                "pass": bool(np.max(np.abs(Ye[:, 0:3] - np.array(eq.X0))) < 1e-12)}
    # (b) axial mode: exact damped linear oscillator (body hanging straight; H moves on the vertical)
    z0 = 0.01
    ax = replace(Mp, X0=(0.0, 0.0, -ell - Mp.b - z0), V0=(0.0, 0.0, 0.0), n0=(0.0, 0.0, -1.0), u0=(0.0, 0.0, 0.0),
                 c_s=0.08, c_x=0.0, c_r=0.0)
    Ya = sp.integrate(ax, t, h_prod)
    w0 = math.sqrt(ax.k / (ax.m + ax.mu)); gam = (ax.c_s + ax.c_x) / (2 * (ax.m + ax.mu)); wd = math.sqrt(w0 ** 2 - gam ** 2)
    zex = -ell - ax.b - z0 * np.exp(-gam * t) * (np.cos(wd * t) + gam / wd * np.sin(wd * t))
    err = float(np.max(np.abs(Ya[:, 2] - zex)))
    p3["axial_damped_linear"] = {"omega_axial_formula": "sqrt(k/(m + m_s/3))", "omega0": w0, "gamma": gam, "amp_m": z0,
                                 "max_abs_err_m": err, "rel_err": err / z0, "lateral_leak_m": float(np.max(np.abs(Ya[:, 0:2]))),
                                 "pass": bool(err / z0 < 1e-7)}
    # (c) lateral linear modes (rigid hinged body), tiny amplitude in each mode shape
    wl, Vm, ell = sp.lateral_mode_shapes(cons)
    tf = np.arange(0, 961 * 4) / 384.0
    rows = []
    for j in range(2):
        Mj = mode_state(cons, ell, Vm, j, 1e-6)
        Yj = sp.integrate(Mj, tf, h_prod)
        H = Yj[:, 0:3] - Mj.b * Yj[:, 6:9]
        Hd = Yj[:, 3:6] - Mj.b * Yj[:, 9:12]
        yv, ydv = (H[:, 0], Hd[:, 0]) if j == 0 else (Yj[:, 6], Yj[:, 9])
        T_num, ncyc = crossing_period(tf, yv, ydv)
        T_ex = 2 * math.pi / wl[j]
        rows.append({"mode": j + 1, "omega_exact": float(wl[j]), "T_exact_s": T_ex, "T_numeric_s": T_num, "n_periods": ncyc,
                     "rel_err": abs(T_num - T_ex) / T_ex})
    p3["lateral_modes_rigid"] = {"rows": rows, "pass": bool(max(r["rel_err"] for r in rows) < 1e-7),
                                 "formula": "det(K - w^2 M) = 0, M = [[(m+mu) l^2, m l b],[m l b, m b^2 + I_perp]], K = diag((m+m_s/2) g l, m g b)"}
    # (d) point-mass elastic pendulum, massless spring: omega_swing = sqrt(g / (L0 + delta))
    pm = replace(cons, body="point", b=0.0, I_perp=0.0, m_s=0.0)
    ellp = pm.L0 + pm.m * pm.g / pm.k
    amp = 1e-6
    pm = replace(pm, X0=(ellp * math.sin(amp), 0.0, -ellp * math.cos(amp)), V0=(0.0, 0.0, 0.0))
    Yp = sp.integrate(pm, tf, h_prod)
    T_num, ncyc = crossing_period(tf, Yp[:, 0], Yp[:, 3])
    T_ex = 2 * math.pi * math.sqrt(ellp / pm.g)
    p3["point_swing_massless"] = {"formula": "omega = sqrt(g/(L0 + m g/k))", "T_exact_s": T_ex, "T_numeric_s": T_num,
                                  "rel_err": abs(T_num - T_ex) / T_ex, "pass": bool(abs(T_num - T_ex) / T_ex < 1e-7)}
    p3["pass"] = all(v["pass"] for v in p3.values() if isinstance(v, dict) and "pass" in v)
    C["P3_analytic"] = p3

    # ---- P4 energy
    p4 = {}
    for nm, M in (("conservative", cons), ("production_params", Mp), ("strong_damping", strong), ("out_of_plane_3d", big3d)):
        d = sp.simulate(M, t, h_prod)
        E0 = d["E"][0]
        bal = d["E"] - E0 + d["W"]
        r = {"E0_J": float(E0), "E_end_over_E0": float(d["E"][-1] / E0),
             "max_drift_rel": float(np.max(np.abs(d["E"] - E0)) / E0),
             "balance_max_rel": float(np.max(np.abs(bal)) / E0),
             "max_sample_increment_over_E0": float(np.max(np.diff(d["E"])) / E0),
             "min_P_diss_W": float(d["P_diss"].min()), "norm_n_err": float(d["norm_n_err"]),
             "n_vs_quaternion_err": float(d["n_q_consistency"]),
             "tension_min_N": float(d["tension"].min()), "tension_max_N": float(d["tension"].max())}
        if nm == "conservative":
            r["pass"] = r["max_drift_rel"] < 1e-8
        else:
            r["pass"] = bool(r["balance_max_rel"] < 1e-7 and r["max_sample_increment_over_E0"] <= 1e-12 and r["min_P_diss_W"] >= 0)
        p4[nm] = r
        if nm == "production_params":
            dprod = d
    C["P4_energy"] = p4
    C["P7_forces"] = {"tension_min_N": float(dprod["tension"].min()), "tension_max_N": float(dprod["tension"].max()),
                      "spring_len_min_m": float(dprod["L"].min()), "pass_tension_positive": bool(dprod["tension"].min() > 0),
                      "note": "a close-wound extension spring also needs T > preload F_i to keep coils apart; F_i is absorbed in L0 (FORMULATION s2)"}
    Y = dprod["Y"]
    C["P8_initial_conditions"] = {"X0_err_m": float(np.max(np.abs(Y[0, 0:3] - np.array(Mp.X0)))),
                                  "V0_err": float(np.max(np.abs(Y[0, 3:6] - np.array(Mp.V0)))),
                                  "time_grid_max_err_s": float(np.max(np.abs(t - np.arange(len(t)) / 96.0)))}
    C["P8_initial_conditions"]["pass"] = C["P8_initial_conditions"]["X0_err_m"] == 0.0
    Yb = sp.integrate(Mp, t, h_prod)
    C["repeatable_bitwise"] = bool(np.array_equal(Y, Yb))
    # ---- v2 guards and P9-style variants (CHANGELOG 2026-10-26T01:53:54Z)
    import model_build as mb
    g = {"constraint_after_projection": {"max_abs_norm_n_minus_1": float(np.max(np.abs(np.linalg.norm(Y[:, 6:9], axis=1) - 1))),
                                         "max_abs_n_dot_ndot": float(np.max(np.abs(np.sum(Y[:, 6:9] * Y[:, 9:12], 1)))),
                                         "max_abs_norm_q_minus_1": float(np.max(np.abs(np.linalg.norm(Y[:, 12:16], axis=1) - 1)))}}
    g["constraint_after_projection"]["pass_1e-12"] = bool(g["constraint_after_projection"]["max_abs_norm_n_minus_1"] <= 1e-12)
    raised = {}
    for nm, kw in (("k_x4_nominal_IC", {"k": 4 * Mp.k}), ("L0_x1.5_nominal_IC", {"L0": 1.5 * Mp.L0})):
        try:
            sp.integrate(replace(Mp, **kw), t, h_prod); raised[nm] = "NOT RAISED"
        except sp.SpringCompressionError as e:
            raised[nm] = "SpringCompressionError: " + str(e)[:80]
    g["compression_guard"] = raised
    g["compression_guard_pass"] = all(v.startswith("SpringCompressionError") for v in raised.values())
    var = {}
    phi0 = params.get("primary", {}).get("phi0", {}).get("value", 0.1); phid0 = params.get("primary", {}).get("phid0", {}).get("value", 0.0)
    for nm, kw in (("k_x4", {"k": 4 * Mp.k}), ("L0_x1.5", {"L0": 1.5 * Mp.L0}), ("k_x0.25", {"k": 0.25 * Mp.k})):
        Mv, info = mb.reequilibrate(replace(Mp, **kw), phi0, phid0)
        st = {}
        try:
            Ya = sp.integrate(Mv, t, h_prod, stats=st); Yb2 = sp.integrate(Mv, t, h_prod / 2)
            var[nm] = {"h_vs_h2_max_dX_m": float(np.max(np.abs(Ya[:, 0:3] - Yb2[:, 0:3]))), "min_tension_N": st["min_tension_N"][0],
                       "max_step_drift": st["max_step_drift"], "ell_eq_m": info["ell_eq"]}
            var[nm]["pass_1e-7"] = var[nm]["h_vs_h2_max_dX_m"] < 1e-7
        except (sp.SpringCompressionError, sp.SolverGuardError) as e:
            var[nm] = {"raised": str(e)[:120], "pass_1e-7": False, "note": "physically slack spring for this variant (expected only for soft k)"}
    g["reequilibrated_variants"] = var
    C["v2_guards"] = g
    summ = {"P1": all(v["pass_tol_1e-7"] and v["pass_order_within_0.5"] for v in conv.values()),
            "P3": p3["pass"], "P4": all(v["pass"] for v in p4.values()), "P7": C["P7_forces"]["pass_tension_positive"],
            "P8": C["P8_initial_conditions"]["pass"], "repeatable": C["repeatable_bitwise"],
            "constraint_1e-12": g["constraint_after_projection"]["pass_1e-12"], "compression_guard": g["compression_guard_pass"],
            "variants_k_x4_L0_x1.5_converge": bool(var["k_x4"]["pass_1e-7"] and var["L0_x1.5"]["pass_1e-7"])}
    res["summary"] = summ
    out = HERE / "out"; out.mkdir(exist_ok=True)
    (out / "selfcheck.json").write_text(json.dumps(res, indent=2) + "\n")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for nm, v in conv.items():
        axs[0, 0].loglog([r["h_fine_s"] for r in v["ladder"]], [max(r["max_dX_m"], 1e-17) for r in v["ladder"]], "o-", label=nm)
    hh = np.array(hs[1:]); r0 = conv["production_params"]["ladder"][0]["max_dX_m"]
    axs[0, 0].loglog(hh, r0 * (hh / hh[0]) ** 4, "k--", label="slope 4")
    axs[0, 0].axhline(1e-7, color="r", lw=0.8); axs[0, 0].set_xlabel("h fine [s]"); axs[0, 0].set_ylabel("max |X_h - X_2h| [m]")
    axs[0, 0].legend(fontsize=7); axs[0, 0].set_title("P1 step halving")
    dc = sp.simulate(cons, t, h_prod)
    axs[0, 1].plot(t, (dc["E"] - dc["E"][0]) / dc["E"][0]); axs[0, 1].set_title("P4 conservative (E-E0)/E0")
    ds_ = sp.simulate(strong, t, h_prod)
    axs[1, 0].plot(t, ds_["E"] / ds_["E"][0], label="E/E0 strong damping"); axs[1, 0].plot(t, ds_["W"] / ds_["E"][0], label="W/E0")
    axs[1, 0].plot(t, (ds_["E"] - ds_["E"][0] + ds_["W"]) / ds_["E"][0] * 1e8, label="balance x1e8"); axs[1, 0].legend(fontsize=7)
    axs[1, 1].plot(t, np.degrees(np.arctan2(dprod["H"][:, 0], -dprod["H"][:, 2])), label="spring angle phi [deg]")
    axs[1, 1].plot(t, np.degrees(np.arctan2(dprod["n"][:, 0], -dprod["n"][:, 2])), label="body angle psi [deg]")
    axs[1, 1].plot(t, (dprod["L"] - ell) * 1000, label="|AH| - ell [mm]"); axs[1, 1].legend(fontsize=7); axs[1, 1].set_xlabel("t [s]")
    fig.suptitle(f"clip02 solver selfcheck | params status {params.get('status')} | {json.dumps(summ)}", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "selfcheck.png", dpi=100)
    print(json.dumps(summ, indent=1))
    print("P1 finest:", {k: (v["finest_pair_max_dX_m"], v["observed_orders_above_roundoff"]) for k, v in conv.items()})
    print("P3:", json.dumps({k: (v.get("rel_err") or v.get("max_drift_m") or [r["rel_err"] for r in v.get("rows", [])]) for k, v in p3.items() if isinstance(v, dict)}))
    print("P4:", json.dumps({k: (v["max_drift_rel"], v["balance_max_rel"], v["max_sample_increment_over_E0"]) for k, v in p4.items()}))
    return 0 if all(summ.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
