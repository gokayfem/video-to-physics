"""Author-side developer checks of solver/pendulum.py (the independent verifier does its own).

Writes solver/out/selfcheck.json and solver/out/selfcheck.png.
Usage: ROOT/.venv/bin/python solver/selfcheck.py
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import pendulum as pd  # noqa: E402
from run import time_grid, solver_source_sha256, sha256_file  # noqa: E402

ACC = json.loads((ROOT / "config" / "acceptance.json").read_text())["physics_validation"]


def convergence(M, t, levels):
    """levels: list of h. Returns per-level diffs between successive levels and observed orders."""
    runs = [pd.integrate(M, t, h) for h in levels]
    rows = []
    for k in range(len(levels) - 1):
        (a1, b1, w1), (a2, b2, w2) = runs[k], runs[k + 1]
        E1 = pd.derived(M, a1, b1)["E"]; E2 = pd.derived(M, a2, b2)["E"]
        rows.append({"h_coarse_s": levels[k], "h_fine_s": levels[k + 1],
                     "max_dtheta_rad": float(np.max(np.abs(a1 - a2))),
                     "max_domega_rad_s": float(np.max(np.abs(b1 - b2))),
                     "max_dE_rel": float(np.max(np.abs(E1 - E2)) / E1[0])})
    for k in range(len(rows) - 1):
        d1, d2 = rows[k]["max_dtheta_rad"], rows[k + 1]["max_dtheta_rad"]
        rows[k + 1]["observed_order_theta"] = float(math.log2(d1 / d2)) if d2 > 0 else None
    return rows, runs


def main():
    params = json.loads((ROOT / "params" / "physics.json").read_text())
    Mp = pd.model_from_params(params)
    t = time_grid(10.0, 96)
    h_prod = float(params.get("integrator", {}).get("h_s", pd.DEFAULT_H))
    res = {"created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "solver_id": pd.SOLVER_ID, "solver_source_sha256": solver_source_sha256(),
           "params_sha256": sha256_file(ROOT / "params" / "physics.json"),
           "h_production_s": h_prod, "theoretical_order": 4, "checks": {}}
    C = res["checks"]

    # Test models: production params, a damped smooth case, a compound sphere+wire case, Coulomb case
    damped = pd.Model(**{**{k: getattr(Mp, k) for k in Mp.__dataclass_fields__},
                         "c_v": 2 * Mp.I_P * 0.05, "k_q": Mp.I_P * 0.3})
    compound = pd.model_from_values(L_c=Mp.L_c, theta0=Mp.theta0, r=0.02, m_bob=Mp.m_bob,
                                    inertia_model="solid_sphere", mu=0.002)
    coulomb = pd.Model(**{**{k: getattr(Mp, k) for k in Mp.__dataclass_fields__},
                          "tau_c": 2e-5, "omega_s": 0.05})

    # ---- P1 convergence: coarse ladder (order) and production-level pair
    ladder = [1 / 96 / 2 ** k for k in range(0, 6)]       # 1/96 ... 1/3072
    conv = {}
    for name, M in (("production_params", Mp), ("damped_cv_kq", damped), ("compound_sphere_wire", compound),
                    ("coulomb_smoothed", coulomb)):
        rows, _ = convergence(M, t, ladder)
        a2, _, _ = pd.integrate(M, t, h_prod / 2)
        a4, _, _ = pd.integrate(M, t, h_prod / 4)
        orders = [r["observed_order_theta"] for r in rows if r.get("observed_order_theta")]
        # use orders from pairs whose differences are well above round-off (>1e-12 rad)
        good = [r["observed_order_theta"] for r in rows
                if r.get("observed_order_theta") and r["max_dtheta_rad"] > 1e-12]
        conv[name] = {"ladder": rows,
                      "prod_half_vs_quarter_max_dtheta_rad": float(np.max(np.abs(a2 - a4))),
                      "observed_orders_above_roundoff": good,
                      "pass_tol_1e-7": bool(np.max(np.abs(a2 - a4)) < 1e-7),
                      "pass_order_within_0.5": bool(good and all(abs(o - 4) <= 0.5 for o in good))}
    C["P1_convergence"] = conv

    # ---- P3 period vs exact elliptic integral (undamped), several amplitudes and models
    per = []
    for name, base in (("point_mass", Mp), ("compound_sphere_wire", compound)):
        for deg in (0.5, 2.0, 6.0, 15.0, 60.0):
            M = pd.Model(**{**{k: getattr(base, k) for k in base.__dataclass_fields__},
                            "theta0": math.radians(deg), "omega0": 0.0, "c_v": 0.0, "k_q": 0.0, "tau_c": 0.0})
            th, om, _ = pd.integrate(M, t, h_prod)
            tc = pd.crossing_times(M, t, th, om, h_prod, direction=-1)
            T_num = float(np.mean(np.diff(tc)))
            T_ex = pd.exact_period(M, M.theta0)
            T_small = 2 * math.pi * math.sqrt(M.I_P / (M.m * M.g * M.d))
            per.append({"model": name, "theta0_deg": deg, "T_numeric_s": T_num, "T_exact_s": T_ex,
                        "rel_err": abs(T_num - T_ex) / T_ex, "n_periods": len(tc) - 1,
                        "T_small_angle_s": T_small, "exact_over_small": T_ex / T_small})
    C["P3_period"] = {"rows": per, "max_rel_err": max(r["rel_err"] for r in per),
                      "pass_tol_1e-7": bool(max(r["rel_err"] for r in per) < 1e-7)}

    # ---- P4 energy
    th, om, W = pd.integrate(Mp, t, h_prod)
    d = pd.derived(Mp, th, om)
    E0 = d["E"][0]
    cons = pd.Model(**{**{k: getattr(Mp, k) for k in Mp.__dataclass_fields__}, "c_v": 0.0, "k_q": 0.0, "tau_c": 0.0})
    thc, omc, _ = pd.integrate(cons, t, h_prod)
    Ec = pd.derived(cons, thc, omc)["E"]
    energy = {"conservative_max_drift_rel": float(np.max(np.abs(Ec - Ec[0])) / Ec[0]),
              "conservative_max_sample_increment_over_E0": float(np.max(np.diff(Ec)) / Ec[0])}
    energy["pass_conservative_increment_1e-12"] = energy["conservative_max_sample_increment_over_E0"] <= 1e-12
    energy["pass_conservative_1e-8"] = energy["conservative_max_drift_rel"] < 1e-8
    for name, M in (("damped_cv_kq", damped), ("coulomb_smoothed", coulomb), ("production_params", Mp)):
        th2, om2, W2 = pd.integrate(M, t, h_prod)
        d2 = pd.derived(M, th2, om2)
        bal = d2["E"] - d2["E"][0] + W2
        # analytic dE/dt at samples = I om alpha + m g d sin(th) om  (should equal -P_diss <= 0)
        dEdt = M.I_P * om2 * d2["alpha"] + M.m * M.g * M.d * np.sin(th2) * om2
        energy[name] = {"E0_J": float(d2["E"][0]), "E_end_over_E0": float(d2["E"][-1] / d2["E"][0]),
                        "balance_max_rel": float(np.max(np.abs(bal)) / d2["E"][0]),
                        "max_dEdt_over_E0_per_s": float(np.max(dEdt) / d2["E"][0]),
                        "max_sample_increment_E_over_E0": float(np.max(np.diff(d2["E"])) / d2["E"][0]),
                        "min_P_diss_W": float(d2["P_diss"].min())}
        energy[name]["pass_balance_1e-7"] = energy[name]["balance_max_rel"] < 1e-7
        energy[name]["pass_no_growth_1e-12"] = (energy[name]["max_dEdt_over_E0_per_s"] <= 1e-12 and
                                                energy[name]["max_sample_increment_E_over_E0"] <= 1e-12)  # CHANGELOG reading
    C["P4_energy"] = energy
    pc = {"production_params": (th, om, d)}

    # ---- linear damping theory check: amplitude envelope for pure c_v, small amplitude
    gam = 0.05
    lin = pd.Model(**{**{k: getattr(Mp, k) for k in Mp.__dataclass_fields__},
                      "theta0": math.radians(0.5), "c_v": 2 * Mp.I_P * gam, "k_q": 0.0, "tau_c": 0.0})
    thl, oml, _ = pd.integrate(lin, t, h_prod)
    # successive maxima: E ~ exp(-2 gam t) on average; compare ratio of energies over 8 s
    dl = pd.derived(lin, thl, oml)
    i8 = int(8 * 96)
    # energy averaged over integer periods is the robust envelope; use ratio of E at t=0 and t=8s corrected:
    w0 = math.sqrt(lin.m * lin.g * lin.d / lin.I_P)
    wd = math.sqrt(w0 ** 2 - gam ** 2)
    # exact linear solution for comparison (small amplitude 0.5 deg -> nonlinearity ~ 1e-5 rel)
    A = lin.theta0; tt = t
    th_lin = A * np.exp(-gam * tt) * (np.cos(wd * tt) + gam / wd * np.sin(wd * tt))
    C["linear_damping_theory"] = {"gamma_1_s": gam, "theta0_deg": 0.5,
                                  "max_abs_diff_vs_linear_solution_rad": float(np.max(np.abs(thl - th_lin))),
                                  "relative_to_amplitude": float(np.max(np.abs(thl - th_lin)) / A),
                                  "note": "difference is the sin(th) nonlinearity at 0.5 deg (~th^2/16 period shift), not solver error"}

    # ---- P7 tension, P8 IC
    C["P7_tension"] = {"min_N": float(d["tension"].min()), "max_N": float(d["tension"].max()),
                       "pass_positive": bool(d["tension"].min() > 0)}
    C["P8_initial_conditions"] = {"theta0_err_rad": float(abs(th[0] - Mp.theta0)),
                                  "omega0_err_rad_s": float(abs(om[0] - Mp.omega0)),
                                  "time_grid_max_err_s": float(np.max(np.abs(t - np.arange(len(t)) / 96.0))),
                                  "pass": bool(abs(th[0] - Mp.theta0) < 1e-9 and abs(om[0] - Mp.omega0) < 1e-9)}
    # repeatability
    th_b, om_b, _ = pd.integrate(Mp, t, h_prod)
    C["repeatability_bitwise"] = bool(np.array_equal(th, th_b) and np.array_equal(om, om_b))

    summary = {
        "P1": all(v["pass_tol_1e-7"] and v["pass_order_within_0.5"] for k, v in conv.items()),
        "P1_note": "non-smooth damping terms are handled by splitting steps at located turning points (omega=0); without that the k_q case showed order ~3.4",
        "P3": C["P3_period"]["pass_tol_1e-7"],
        "P4_conservative": energy["pass_conservative_1e-8"] and energy["pass_conservative_increment_1e-12"],
        "P4_damped": all(energy[n]["pass_balance_1e-7"] and energy[n]["pass_no_growth_1e-12"]
                         for n in ("damped_cv_kq", "coulomb_smoothed", "production_params")),
        "P7": C["P7_tension"]["pass_positive"], "P8": C["P8_initial_conditions"]["pass"],
        "repeatable": C["repeatability_bitwise"]}
    res["summary"] = summary
    out = HERE / "out"; out.mkdir(exist_ok=True)
    (out / "selfcheck.json").write_text(json.dumps(res, indent=2) + "\n")

    # ---- plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for name, v in conv.items():
        hs = [r["h_fine_s"] for r in v["ladder"]]; ds = [r["max_dtheta_rad"] for r in v["ladder"]]
        ax[0, 0].loglog(hs, ds, "o-", label=name)
    hh = np.array([ladder[1], ladder[-1]])
    ax[0, 0].loglog(hh, conv["production_params"]["ladder"][0]["max_dtheta_rad"] * (hh / hh[0]) ** 4, "k--", label="slope 4")
    ax[0, 0].set_xlabel("h (fine) [s]"); ax[0, 0].set_ylabel("max |theta_h - theta_2h| [rad]")
    ax[0, 0].set_title("P1 step-halving convergence"); ax[0, 0].legend(fontsize=7)
    ax[0, 1].plot(t, (Ec - Ec[0]) / Ec[0]); ax[0, 1].set_title("conservative energy drift (E-E0)/E0")
    ax[0, 1].set_xlabel("t [s]")
    th2, om2, W2 = pd.integrate(damped, t, h_prod); d2 = pd.derived(damped, th2, om2)
    ax[1, 0].plot(t, d2["E"] / d2["E"][0], label="E/E0"); ax[1, 0].plot(t, W2 / d2["E"][0], label="W_diss/E0")
    ax[1, 0].plot(t, (d2["E"] - d2["E"][0] + W2) / d2["E"][0] * 1e7, label="balance residual x1e7")
    ax[1, 0].set_title("damped test case energy balance"); ax[1, 0].legend(fontsize=7); ax[1, 0].set_xlabel("t [s]")
    ax[1, 1].plot(t, np.degrees(th), label="theta (production params)")
    ax[1, 1].plot(t, np.degrees(th2), label="theta (damped test)")
    ax[1, 1].set_xlabel("t [s]"); ax[1, 1].set_ylabel("deg"); ax[1, 1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "selfcheck.png", dpi=110)
    print(json.dumps(summary, indent=1))
    print("P1 prod half-vs-quarter:", {k: v["prod_half_vs_quarter_max_dtheta_rad"] for k, v in conv.items()})
    print("orders:", {k: [round(o, 3) for o in v["observed_orders_above_roundoff"]] for k, v in conv.items()})
    print("P3 max rel err:", C["P3_period"]["max_rel_err"])
    print("energy:", json.dumps(energy, indent=1))
    return 0 if all(v for k, v in summary.items() if isinstance(v, bool)) else 1


if __name__ == "__main__":
    sys.exit(main())
