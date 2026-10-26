"""Author-side developer checks of solver/torsion_disc.py (P1, P3, P4, P7, P8, guards, P9-style variants; the verifier
does its own).  Writes solver/out/selfcheck.json and solver/out/selfcheck.png.

Usage: ROOT/.venv/bin/python solver/selfcheck.py [--quick] [--params params/physics.json]
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
from scipy.special import ellipk

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import torsion_disc as td  # noqa: E402
import model_build as mb  # noqa: E402
from run import time_grid, solver_source_sha256, sha256_file  # noqa: E402


def rot_angle_diff(qa, qb):
    """max rotation angle [rad] between two quaternion series (N,4)."""
    d = np.abs(np.sum(qa * qb, 1)).clip(0, 1)
    return float(np.max(2 * np.arccos(d))) if np.any(d < 1) else 0.0


def maxdiff(Ya, Yb):
    dX = float(np.max(np.abs(Ya[:, 0:3] - Yb[:, 0:3])))
    qa, qb = Ya[:, 6:10], Yb[:, 6:10]
    dq = float(np.max(np.abs(qa - qb))) * 2.0          # ~ rotation angle for small differences (robust, no arccos roundoff)
    return dX, dq


def convergence(Ms, names, t, hs):
    runs = [td.integrate(Ms, t, h) for h in hs]
    out = {}
    for j, nm in enumerate(names):
        rows = []
        for i in range(len(hs) - 1):
            dX, dq = maxdiff(runs[i][:, :, j], runs[i + 1][:, :, j])
            rows.append({"h_coarse_s": hs[i], "h_fine_s": hs[i + 1], "max_dX_m": dX, "max_drot_rad": dq})
        for i in range(1, len(rows)):
            for key, o in (("max_dX_m", "observed_order_X"), ("max_drot_rad", "observed_order_rot")):
                a, b = rows[i - 1][key], rows[i][key]
                rows[i][o] = math.log2(a / b) if b > 0 else None
        good = [r["observed_order_rot"] for r in rows if r.get("observed_order_rot") and r["max_drot_rad"] > 1e-10]
        fin = rows[-1]
        out[nm] = {"ladder": rows, "finest_pair_max_dX_m": fin["max_dX_m"], "finest_pair_max_drot_rad": fin["max_drot_rad"],
                   "observed_orders_rot_above_roundoff": good,
                   "pass_tol_1e-7": bool(fin["max_dX_m"] < 1e-7 and fin["max_drot_rad"] < 1e-7),
                   "pass_order_within_0.5": bool(good and all(abs(o - 4) <= 0.5 for o in good))}
    return out


def crossing_period(t, y, yd, direction=+1):
    """mean period from same-direction zero crossings, located by cubic Hermite interpolation on (y, y')."""
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


def at_rest(M):
    return replace(M, X0=(0.0, 0.0, -M.ell - M.h_c), V0=(0.0, 0.0, 0.0), q0=(1.0, 0.0, 0.0, 0.0), w0=(0.0, 0.0, 0.0))


def planar_state(M, phi, theta):
    X0, V0, q0, w0 = mb.state_from_angles(M.ell, M.h_c, 0.0, 0.0, phi, 0.0, 0.0, 0.0, theta, 0.0, 0.0, 0.0)
    return replace(M, X0=tuple(X0), V0=tuple(V0), q0=tuple(q0), w0=tuple(w0))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--params", default=str(ROOT / "params" / "physics.json"))
    a = ap.parse_args(argv)
    params = json.loads(Path(a.params).read_text())
    Mp = td.model_from_physics_json(params)
    h_prod = float(params.get("integrator", {}).get("h_s", td.DEFAULT_H))
    t = time_grid(10.0, 96)
    res = {"created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "solver_id": td.SOLVER_ID, "solver_source_sha256": solver_source_sha256(),
           "params_file": str(Path(a.params).resolve().relative_to(ROOT)), "params_sha256": sha256_file(a.params),
           "params_status": params.get("status"), "h_production_s": h_prod, "theoretical_order": 4, "checks": {}}
    C = res["checks"]
    cons = replace(Mp, c_t=0.0, c_x=0.0, c_r=0.0)
    strong = replace(Mp, c_t=Mp.c_t + 20 * Mp.I_z, c_x=Mp.c_x + 0.5 * Mp.m, c_r=Mp.c_r + 2 * Mp.I_perp)
    k1 = float(td.lateral_modes(replace(Mp, c_t=0.0, c_x=0.0, c_r=0.0))[1][1, 0])
    X0, V0, q0, w0 = mb.state_from_angles(Mp.ell, Mp.h_c, 0.9, 0.5, 0.12, -0.08, 0.3, 0.4,
                                          k1 * 0.12 + 0.0005, -k1 * 0.08 - 0.0003, k1 * 0.3 + 0.005, k1 * 0.4)
    big3d = replace(Mp, X0=tuple(X0), V0=tuple(V0), q0=tuple(q0), w0=tuple(w0))   # 52 deg twist, 7 deg 3D sway, tilt mode excited (free offset)

    # ---- P1 step-halving ladder
    # ladder starts at 4 h: the rigid wire-disc joint mode (~300 rad/s) trips the per-step drift guard at 8-16 h by design
    hs = [h_prod * 4, h_prod * 2, h_prod, h_prod / 2]
    conv = convergence([Mp, strong, big3d], ["production_params", "strong_damping", "large_3d_free_tilt"], t, hs)
    C["P1_convergence"] = conv
    C["P1_note"] = ("ladder 4 h_prod, 2 h_prod, h_prod, h_prod/2; finest pair = (h_prod, h_prod/2); rotation difference = "
                    "2 max|dq| (~ angle). Production h = %.6g s" % h_prod)

    # ---- P3 analytic limits (one batch, fine output grid for period measurement)
    tf = np.arange(0, 7681) / 768.0
    gam = 0.05                                    # 1/s, exact damped linear oscillator check
    tor30 = replace(at_rest(Mp), q0=tuple(td.quat_from_tilt_twist([0, 0, -1.0], math.radians(30))), c_t=2 * gam * Mp.I_z, c_x=0.0, c_r=0.0)
    tor60 = replace(tor30, q0=tuple(td.quat_from_tilt_twist([0, 0, -1.0], math.radians(60))), w0=(0.0, 0.0, 0.3))
    wl, Vm = td.lateral_modes(cons)
    md = [planar_state(cons, 1e-6, 1e-6 * Vm[1, j]) for j in range(2)]
    pend0 = replace(cons, t=0.0, k_A=0.0, k_B=0.0)
    phiA = 0.3
    pend = planar_state(pend0, phiA, phiA)        # ideal string, point-like attachment at the CoM (t = 0): planar pendulum
    eq = at_rest(Mp)
    batch = [eq, tor30, tor60, md[0], md[1], pend]
    Yb = td.integrate(batch, tf, h_prod)
    p3 = {}
    p3["static_equilibrium"] = {"max_drift_m": float(np.max(np.abs(Yb[:, 0:3, 0] - np.array(eq.X0)))),
                                "max_rot": float(np.max(np.abs(Yb[:, 7:10, 0])))}
    p3["static_equilibrium"]["pass"] = p3["static_equilibrium"]["max_drift_m"] < 1e-12 and p3["static_equilibrium"]["max_rot"] < 1e-12
    for j, (nm, psi0, wz0) in enumerate((("torsion_damped_30deg", math.radians(30), 0.0), ("torsion_damped_60deg_with_rate", math.radians(60), 0.3)), start=1):
        Mj = batch[j]
        psi = td.twist_angle(Yb[:, 6:10, j].T)
        w0_ = math.sqrt(Mj.kappa / Mj.I_z); wd = math.sqrt(w0_ ** 2 - gam ** 2)
        ex = np.exp(-gam * tf) * (psi0 * np.cos(wd * tf) + (wz0 + gam * psi0) / wd * np.sin(wd * tf))
        err = float(np.max(np.abs(psi - ex)))
        p3[nm] = {"formula": "I_z psi'' + c_t psi' + kappa psi = 0 (exact for pure twist at ANY amplitude: linear wire)",
                  "omega0": w0_, "gamma": gam, "T_s": 2 * math.pi / w0_, "max_abs_err_rad": err, "rel_err": err / psi0,
                  "lateral_leak_m": float(np.max(np.abs(Yb[:, 0:2, j]))), "pass": bool(err / psi0 < 1e-7)}
    rows = []
    for j in range(2):
        Y = Yb[:, :, 3 + j]
        if j == 0:
            y, yd = Y[:, 0], Y[:, 3]                                  # X_x and its exact rate
        else:
            n = -td.rot_batch(Y[:, 6:10].T)[:, :, 2]                   # disc down-normal
            w = Y[:, 10:13]
            y, yd = n[:, 0], w[:, 1] * n[:, 2] - w[:, 2] * n[:, 1]      # n_x and (w x n)_x
        T_num, ncyc = crossing_period(tf, y, yd)
        T_ex = 2 * math.pi / wl[j]
        rows.append({"mode": j + 1, "omega_exact": wl[j], "theta_over_phi": float(Vm[1, j]), "T_exact_s": T_ex, "T_numeric_s": T_num,
                     "n_periods": ncyc, "rel_err": abs(T_num - T_ex) / T_ex})
    p3["lateral_modes_linear"] = {"rows": rows, "pass": bool(max(r["rel_err"] for r in rows) < 1e-7),
                                  "formula": "det(K - w^2 M) = 0, M = [[m l^2, m l h],[m l h, m h^2 + I_perp]], K = [[m g l + k_A + k_B, -k_B],[-k_B, m g h + k_B]]"}
    Lh = cons.ell + cons.h_c
    w_rig = math.sqrt((cons.m * cons.g * Lh + cons.k_A) / (cons.m * Lh ** 2 + cons.I_perp))
    p3["lateral_modes_linear"]["rigid_joint_compound_pendulum_info"] = {
        "formula": "omega^2 = (m g (l+h) + k_A) / (m (l+h)^2 + I_perp)  (k_B -> infinity; informational, finite k_B differs by O((omega/omega_B)^2))",
        "omega_rigid": w_rig, "omega_mode1": wl[0], "rel_diff": abs(wl[0] - w_rig) / w_rig}
    Yp = Yb[:, :, 5]
    T_num, ncyc = crossing_period(tf, Yp[:, 0], Yp[:, 3])
    T_ex = 4 * math.sqrt(pend.ell / pend.g) * ellipk(math.sin(phiA / 2) ** 2)
    p3["planar_pendulum_elliptic"] = {"formula": "T = 4 sqrt(l/g) K(sin^2(phi0/2)) (ideal string, t = 0, k = 0, phi0 = 0.3 rad)",
                                      "T_exact_s": T_ex, "T_numeric_s": T_num, "n_periods": ncyc, "rel_err": abs(T_num - T_ex) / T_ex,
                                      "pass": bool(abs(T_num - T_ex) / T_ex < 1e-7)}
    p3["pass"] = all(v["pass"] for v in p3.values() if isinstance(v, dict) and "pass" in v)
    C["P3_analytic"] = p3

    # ---- P4 energy
    p4 = {}
    dprod = None
    for nm, M in (("conservative", cons), ("production_params", Mp), ("strong_damping", strong), ("large_3d_free_tilt", big3d),
                  ("large_3d_conservative", replace(big3d, c_t=0.0, c_x=0.0, c_r=0.0))):
        d = td.simulate(M, t, h_prod)
        E0 = d["E"][0]; bal = d["E"] - E0 + d["W"]
        r = {"E0_J": float(E0), "E_end_over_E0": float(d["E"][-1] / E0), "max_drift_rel": float(np.max(np.abs(d["E"] - E0)) / E0),
             "balance_max_rel": float(np.max(np.abs(bal)) / E0), "max_sample_increment_over_E0": float(np.max(np.diff(d["E"])) / E0),
             "min_P_diss_W": float(d["P_diss"].min()), "norm_q_err": d["norm_q_err"], "wire_length_err_m": d["len_err"],
             "tension_min_N": float(d["tension"].min()), "tension_max_N": float(d["tension"].max())}
        if nm.endswith("conservative"):
            r["pass"] = r["max_drift_rel"] < 1e-8
        else:
            r["pass"] = bool(r["balance_max_rel"] < 1e-7 and r["max_sample_increment_over_E0"] <= 1e-12 and r["min_P_diss_W"] >= 0)
        p4[nm] = r
        if nm == "production_params":
            dprod = d
    C["P4_energy"] = p4
    C["P7_forces"] = {"tension_min_N": float(dprod["tension"].min()), "tension_max_N": float(dprod["tension"].max()),
                      "static_tension_N": Mp.m * Mp.g, "pass_tension_positive": bool(dprod["tension"].min() > 0)}
    Y = dprod["Y"]
    C["P8_initial_conditions"] = {"X0_err_m": float(np.max(np.abs(Y[0, 0:3] - np.array(Mp.X0)))),
                                  "q0_err": float(np.max(np.abs(Y[0, 6:10] - np.array(Mp.q0)))),
                                  "time_grid_max_err_s": float(np.max(np.abs(t - np.arange(len(t)) / 96.0)))}
    C["P8_initial_conditions"]["pass"] = C["P8_initial_conditions"]["X0_err_m"] == 0.0 and C["P8_initial_conditions"]["q0_err"] == 0.0
    C["repeatable_bitwise"] = bool(np.array_equal(Y, td.integrate(Mp, t, h_prod)))

    # ---- guards + P9-style variants with re-equilibrated ICs (clip-02 lessons)
    g = {"constraint_after_projection": {"max_abs_norm_q_minus_1": dprod["norm_q_err"], "max_abs_wire_length_err_m": dprod["len_err"]}}
    g["constraint_after_projection"]["pass_1e-12"] = bool(dprod["norm_q_err"] <= 1e-12 and dprod["len_err"] <= 1e-12 * max(1, Mp.ell))
    X0s, V0s, q0s, w0s = mb.state_from_angles(Mp.ell, Mp.h_c, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    upw = np.array([0.0, 0.0, 1.0]) * 2.0 * math.sqrt(Mp.g * Mp.ell)
    try:
        slack2 = replace(cons, V0=tuple(upw))     # violates B.dB = 0 -> initial_state raises ValueError (guard on ICs)
        td.integrate(slack2, t[:20], h_prod); g["ic_guard"] = "NOT RAISED"
    except ValueError as e:
        g["ic_guard"] = "ValueError: " + str(e)[:80]
    s_ = np.array([math.sin(2.6), 0.0, -math.cos(2.6)]); B_ = Mp.ell * s_
    slack3 = replace(cons, k_A=0.0, k_B=0.0, t=0.0, X0=tuple(B_), V0=(0.0, 0.0, 0.0))
    try:
        td.integrate(slack3, t[:60], h_prod); g["slack_guard"] = "NOT RAISED"
    except td.SlackWireError as e:
        g["slack_guard"] = "SlackWireError: " + str(e)[:80]
    except td.SolverGuardError as e:
        g["slack_guard"] = "SolverGuardError: " + str(e)[:80]
    g["slack_guard_pass"] = g["slack_guard"].startswith("SlackWireError") and g["ic_guard"].startswith("ValueError")
    pr = params.get("primary", {})
    gv = lambda k: float((pr.get(k) or {}).get("value") or 0.0)
    var = {}
    for nm, kw in (("kappa_x4", {"kappa": 4 * Mp.kappa}), ("ell_x1.5", {"ell": 1.5 * Mp.ell}), ("m_x2", {"m": 2 * Mp.m}),
                   ("R_x1.3", {"R": 1.3 * Mp.R}), ("kAB_x4", {"k_A": 4 * Mp.k_A, "k_B": 4 * Mp.k_B})):
        Mv, info = mb.reequilibrate(replace(Mp, **kw), gv("psi0"), gv("psid0"), gv("phx0"), gv("phy0"), gv("phxd0"), gv("phyd0"))
        st = {}
        try:
            Ya = td.integrate(Mv, t, h_prod, stats=st); Yh = td.integrate(Mv, t, h_prod / 2)
            dX, dq = maxdiff(Ya, Yh)
            # control response: pure-twist period of the variant vs 2 pi sqrt(I_z/kappa) (the equation's prediction)
            var[nm] = {"h_vs_h2_max_dX_m": dX, "h_vs_h2_max_drot": dq, "min_tension_N": st["min_tension_N"][0],
                       "max_step_drift": st["max_step_drift"], "omega_torsion_pred": info["omega_torsion"],
                       "omega_lateral_pred": info["omega_lateral"], "kappa1": info["kappa1"],
                       "pass_1e-7": bool(dX < 1e-7 and dq < 1e-7)}
        except (td.SlackWireError, td.SolverGuardError) as e:
            var[nm] = {"raised": str(e)[:120], "pass_1e-7": False}
    g["reequilibrated_variants"] = var
    C["guards_and_variants"] = g
    summ = {"P1": all(v["pass_tol_1e-7"] and v["pass_order_within_0.5"] for v in conv.values()),
            "P3": p3["pass"], "P4": all(v["pass"] for v in p4.values()), "P7": C["P7_forces"]["pass_tension_positive"],
            "P8": C["P8_initial_conditions"]["pass"], "repeatable": C["repeatable_bitwise"],
            "constraint_1e-12": g["constraint_after_projection"]["pass_1e-12"], "slack_and_ic_guards": g["slack_guard_pass"],
            "variants_converge": all(v["pass_1e-7"] for v in var.values())}
    res["summary"] = summ
    out = HERE / "out"; out.mkdir(exist_ok=True)
    (out / "selfcheck.json").write_text(json.dumps(res, indent=2, default=float) + "\n")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for nm, v in conv.items():
        axs[0, 0].loglog([r["h_fine_s"] for r in v["ladder"]], [max(r["max_drot_rad"], 1e-17) for r in v["ladder"]], "o-", label=nm)
    hh = np.array(hs[1:]); r0 = conv["production_params"]["ladder"][0]["max_drot_rad"]
    axs[0, 0].loglog(hh, r0 * (hh / hh[0]) ** 4, "k--", label="slope 4")
    axs[0, 0].axhline(1e-7, color="r", lw=0.8); axs[0, 0].set_xlabel("h fine [s]"); axs[0, 0].set_ylabel("max rotation diff [rad]")
    axs[0, 0].legend(fontsize=7); axs[0, 0].set_title("P1 step halving")
    dc = td.simulate(cons, t, h_prod)
    axs[0, 1].plot(t, (dc["E"] - dc["E"][0]) / dc["E"][0]); axs[0, 1].set_title("P4 conservative (E-E0)/E0")
    ds_ = td.simulate(strong, t, h_prod)
    axs[1, 0].plot(t, ds_["E"] / ds_["E"][0], label="E/E0 strong damping"); axs[1, 0].plot(t, ds_["W"] / ds_["E"][0], label="W/E0")
    axs[1, 0].plot(t, (ds_["E"] - ds_["E"][0] + ds_["W"]) / ds_["E"][0] * 1e8, label="balance x1e8"); axs[1, 0].legend(fontsize=7)
    axs[1, 1].plot(t, np.degrees(dprod["psi"]), label="twist psi [deg]")
    axs[1, 1].plot(t, 10 * np.degrees(dprod["phi_x"]), label="10 x wire angle phi_x [deg]")
    axs[1, 1].plot(t, 10 * np.degrees(dprod["theta_x"]), label="10 x disc tilt theta_x [deg]"); axs[1, 1].legend(fontsize=7); axs[1, 1].set_xlabel("t [s]")
    fig.suptitle(f"clip03 solver selfcheck | params status {params.get('status')} | {json.dumps(summ)}", fontsize=7)
    fig.tight_layout(); fig.savefig(out / "selfcheck.png", dpi=100)
    print(json.dumps(summ, indent=1))
    print("P1 finest:", {k: (v["finest_pair_max_dX_m"], v["finest_pair_max_drot_rad"], v["observed_orders_rot_above_roundoff"]) for k, v in conv.items()})
    print("P3:", json.dumps({k: (v.get("rel_err") or v.get("max_drift_m") or [r["rel_err"] for r in v.get("rows", [])]) for k, v in p3.items() if isinstance(v, dict)}))
    print("P4:", json.dumps({k: (v["max_drift_rel"], v["balance_max_rel"], v["max_sample_increment_over_E0"]) for k, v in p4.items()}))
    print("guards:", g["slack_guard"], "|", g["ic_guard"])
    print("variants:", json.dumps({k: (v.get("h_vs_h2_max_dX_m"), v.get("h_vs_h2_max_drot"), v.get("raised")) for k, v in var.items()}))
    return 0 if all(summ.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
