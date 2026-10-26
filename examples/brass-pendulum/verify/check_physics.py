#!/usr/bin/env python
"""P2, P3, P4, P7, P8 checks on the production trajectory (verifier-owned; no solver/ imports).

  ROOT/.venv/bin/python verify/check_physics.py \
      --params params/physics.json --trajectory solver/out/trajectory.csv \
      [--conservative-trajectory solver/out/trajectory_conservative.csv] \
      [--meta solver/out/trajectory_meta.json] [--model-json override.json] \
      --out verify/out/check_physics.json

Tolerances are those of config/acceptance.json v1 (copied below, not relaxed). Verifier-internal
consistency tolerances (column vs recomputation) are declared in PLAN.md.
"""
import argparse, json, math, os, sys, datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vparams import load_model, sha256_file  # noqa: E402
from ref_solver import taylor_solve, gl_solve, closed_form, period_exact, period_agm, period_small, conservative_amplitude  # noqa: E402
from vlib import read_traj, up_crossings, extrema, cumulative_dissipation, simpson_cumulative, boole_cumulative, status, jsonable  # noqa: E402

TOL = dict(P2=1e-6, P3=1e-7, P4_cons=1e-8, P4_damped=1e-7, P4_growth=1e-12, P8_solver=1e-9, P8_t=1e-12)
CONSIST = 1e-9          # verifier: column vs recomputation, relative to E0 (energy) / E0 per s (power)
XZ_TOL = 1e-9           # verifier: x_m,z_m vs L_c (sin, -cos), metres
TENSION_REL = 1e-6      # verifier: tension column vs recomputation (informational unless model differs)


def jerk_of(model, th, om, al):
    I = model.I_P
    return -(model.mgd / I) * np.cos(th) * om - (model.c_v / I) * al - 2 * (model.k_q / I) * np.abs(om) * al


def P_of(model):
    return lambda w: model.c_v * w * w + model.k_q * abs(w) ** 3 + model.tau_c * abs(w)


def energy_checks(model, d, conservative):
    t, th, om, al = d["t_s"], d["theta_rad"], d["omega_rad_s"], d["alpha_rad_s2"]
    E0 = 0.5 * model.I_P * model.omega0 ** 2 + model.mgd * (1 - math.cos(model.theta0))
    KE = 0.5 * model.I_P * om ** 2
    PE = model.mgd * (1 - np.cos(th))
    E = KE + PE
    Pd = model.c_v * om ** 2 + model.k_q * np.abs(om) ** 3 + model.tau_c * np.abs(om)
    r = dict(E0_recomputed_J=E0, E0_column_J=float(d["E_J"][0]))
    r["col_KE_vs_recomputed_rel"] = float(np.max(np.abs(d["KE_J"] - KE)) / E0)
    r["col_dPE_vs_recomputed_rel"] = float(np.max(np.abs((d["PE_J"] - d["PE_J"][0]) - (PE - PE[0]))) / E0)
    r["col_PE_offset_J"] = float(d["PE_J"][0] - PE[0])
    r["col_E_vs_KE_plus_PE_rel"] = float(np.max(np.abs(d["E_J"] - d["KE_J"] - d["PE_J"])) / E0)
    r["col_Pdiss_vs_recomputed_rel_per_s"] = float(np.max(np.abs(d["P_diss_W"] - Pd)) / E0)
    consistent = max(r["col_KE_vs_recomputed_rel"], r["col_dPE_vs_recomputed_rel"], r["col_E_vs_KE_plus_PE_rel"],
                     r["col_Pdiss_vs_recomputed_rel_per_s"]) <= CONSIST
    r["columns_consistent_with_recomputation"] = bool(consistent)
    Ec = d["E_J"]
    dt = np.diff(t)
    r["max_forward_dEdt_column_over_E0"] = float(np.max(np.diff(Ec) / dt) / E0)
    r["max_per_sample_increment_over_E0"] = float(np.max(np.diff(Ec)) / E0)
    r["no_growth_reading"] = "CHANGELOG 2026-10-25T22:31:55Z: E(t_k+1)-E(t_k) <= 1e-12*E0 per 1/96 s sample, damped AND conservative"
    r["no_growth_ok"] = bool(r["max_per_sample_increment_over_E0"] <= TOL["P4_growth"])
    r["max_analytic_dEdt_over_E0"] = float(np.max(om * (model.I_P * al + model.mgd * np.sin(th))) / E0)
    if conservative:
        r["drift_column_rel"] = float(np.max(np.abs(Ec - Ec[0])) / E0)
        r["drift_recomputed_rel"] = float(np.max(np.abs(E - E[0])) / E0)
        ok = consistent and max(r["drift_column_rel"], r["drift_recomputed_rel"]) < TOL["P4_cons"] and r["no_growth_ok"]
    else:
        W = cumulative_dissipation(t, om, al, jerk_of(model, th, om, al), P_of(model))
        bal_col = (Ec - Ec[0]) + W
        bal_rec = (E - E[0]) + W
        r["balance_residual_column_E_eventaware_int_rel"] = float(np.max(np.abs(bal_col)) / E0)
        r["balance_residual_recomputed_E_rel"] = float(np.max(np.abs(bal_rec)) / E0)
        r["dissipated_fraction_at_end"] = float(W[-1] / E0)
        # pure-column route (Simpson / Boole on P_diss_W) with quadrature-error estimate
        uniform = np.max(np.abs(np.diff(t) - (t[1] - t[0]))) < 1e-12
        if uniform:
            iS, WS = simpson_cumulative(t, d["P_diss_W"]); iB, WB = boole_cumulative(t, d["P_diss_W"])
            r["balance_residual_pure_columns_simpson_rel"] = float(np.max(np.abs(Ec[iS] - Ec[0] + WS)) / E0)
            common = np.intersect1d(iS, iB)
            r["simpson_vs_boole_quadrature_rel"] = float(np.max(np.abs(WS[np.searchsorted(iS, common)] - WB[np.searchsorted(iB, common)])) / E0)
        ok = consistent and r["balance_residual_column_E_eventaware_int_rel"] < TOL["P4_damped"] and \
            r["balance_residual_recomputed_E_rel"] < TOL["P4_damped"] and r["no_growth_ok"]
    r["status"] = status(ok)
    return r


def periods(t, th, om, al):
    tc = up_crossings(t, th, om, al)
    return tc, np.diff(tc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--model-json")
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--conservative-trajectory")
    ap.add_argument("--meta")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model, params = load_model(a.params, a.model_json)
    d, header, missing = read_traj(a.trajectory)
    rep = dict(label=a.label, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               inputs={p: sha256_file(p) for p in [a.params, a.trajectory, a.conservative_trajectory, a.meta, a.model_json] if p},
               verifier_code={f: sha256_file(os.path.join(HERE, f)) for f in ["check_physics.py", "ref_solver.py", "vparams.py", "vlib.py"]},
               model=model.as_dict(), model_provenance=model.provenance, tolerances=TOL,
               header=header, missing_columns=missing)
    if a.meta:
        with open(a.meta) as f:
            meta = json.load(f)
        rep["trajectory_meta"] = meta
        ph = meta.get("params_sha256") or meta.get("params_hash")
        rep["meta_params_hash_matches"] = (ph == sha256_file(a.params)) if ph else "no params hash in meta"
    if missing:
        rep["fatal"] = f"missing columns {missing}"
        json.dump(jsonable(rep), open(a.out, "w"), indent=2); print(json.dumps(jsonable(rep), indent=1)); sys.exit(2)
    t, th, om, al = d["t_s"], d["theta_rad"], d["omega_rad_s"], d["alpha_rad_s2"]
    res = {}

    # ---------------- P8 initial conditions & time grid (+ x,z consistency)
    n = np.arange(len(t))
    p8 = dict(theta0_param=model.theta0, theta0_traj=float(th[0]), d_theta0=float(abs(th[0] - model.theta0)),
              omega0_param=model.omega0, omega0_traj=float(om[0]), d_omega0=float(abs(om[0] - model.omega0)),
              n_rows=int(len(t)), t_first=float(t[0]), t_last=float(t[-1]),
              max_abs_t_minus_n_over_96=float(np.max(np.abs(t - n / 96.0))))
    if model.L_c is not None:
        p8["max_xz_vs_Lc_sin_cos_m"] = float(max(np.max(np.abs(d["x_m"] - model.L_c * np.sin(th))),
                                                 np.max(np.abs(d["z_m"] + model.L_c * np.cos(th)))))
    # period in data vs my reference at the params' L, g (no time scaling)
    ref_th, ref_om, ref_al, ref_ev = taylor_solve(model, t)
    tc_d, T_d = periods(t, th, om, al)
    tc_r, T_r = periods(t, ref_th, ref_om, ref_al)
    if len(T_d) and len(T_d) == len(T_r):
        p8["periods_data_s"] = T_d.tolist()
        p8["max_rel_period_data_vs_ref"] = float(np.max(np.abs(T_d / T_r - 1)))
        p8["mean_period_rel_data_vs_ref"] = float((tc_d[-1] - tc_d[0]) / (tc_r[-1] - tc_r[0]) - 1)
    else:
        p8["period_count_mismatch"] = [len(T_d), len(T_r)]
    ok8 = (p8["d_theta0"] < TOL["P8_solver"] and p8["d_omega0"] < TOL["P8_solver"] and
           p8["max_abs_t_minus_n_over_96"] <= TOL["P8_t"] and abs(t[0]) == 0.0 and t[-1] >= 10.0 - 1e-12 and
           p8.get("max_xz_vs_Lc_sin_cos_m", 0.0) <= XZ_TOL and
           p8.get("max_rel_period_data_vs_ref", 1.0) < TOL["P3"])
    p8["criteria"] = "dtheta0,domega0<1e-9; |t-n/96|<=1e-12; t from 0 to >=10 s; x,z = L_c(sin,-cos) within 1e-9 m; period vs independent ref < 1e-7 rel"
    p8["status"] = status(ok8)
    res["P8_initial_conditions_time"] = p8

    # ---------------- P2 independent benchmark
    th_gl, _, _, _ = gl_solve(model, t)
    p2 = dict(max_abs_dtheta_rad=float(np.max(np.abs(th - ref_th))),
              t_at_max=float(t[int(np.argmax(np.abs(th - ref_th)))]),
              max_abs_domega=float(np.max(np.abs(om - ref_om))),
              ref_taylor_vs_gl4_max_dtheta=float(np.max(np.abs(ref_th - th_gl))),
              ref_turning_points=int(len(ref_ev)), method="Taylor-series order 24 (event split at omega=0), cross-checked by Gauss-Legendre s=4")
    p2["status"] = status(p2["max_abs_dtheta_rad"] < TOL["P2"] and p2["ref_taylor_vs_gl4_max_dtheta"] < 1e-9)
    res["P2_independent_benchmark"] = p2

    # ---------------- P4 energy (damped/production trajectory)
    res["P4_energy_damped" if not model.conservative else "P4_energy_conservative"] = energy_checks(model, d, model.conservative)

    # ---------------- P3 conservative period (needs conservative production run)
    if a.conservative_trajectory or model.conservative:
        if a.conservative_trajectory:
            dc, _, missc = read_traj(a.conservative_trajectory)
        else:
            dc, missc = d, []
        mc = model.replace(c_v=0.0, k_q=0.0, tau_c=0.0)
        tcc, Tc = periods(dc["t_s"], dc["theta_rad"], dc["omega_rad_s"], dc["alpha_rad_s2"])
        amp = conservative_amplitude(mc)
        T_ell, T_agm = period_exact(mc, amp), period_agm(mc, amp)
        p3 = dict(amplitude_rad=amp, T_exact_ellipk_s=T_ell, T_exact_agm_s=T_agm, T_small_angle_s=period_small(mc),
                  n_periods=int(len(Tc)), periods_s=Tc.tolist(),
                  max_rel_err=float(np.max(np.abs(Tc / T_ell - 1))) if len(Tc) else None,
                  mean_rel_err=float(((tcc[-1] - tcc[0]) / len(Tc)) / T_ell - 1) if len(Tc) else None,
                  conservative_traj_theta0_err=float(abs(dc["theta_rad"][0] - mc.theta0)))
        thc, _ = closed_form(mc, dc["t_s"])
        p3["conservative_traj_vs_closed_form_max_dtheta"] = float(np.max(np.abs(dc["theta_rad"] - thc)))
        p3["status"] = status(len(Tc) > 0 and p3["max_rel_err"] < TOL["P3"])
        res["P4_energy_conservative"] = energy_checks(mc, dc, True)
    else:
        p3 = dict(status="NOT_RUN", reason="no conservative production trajectory supplied (--conservative-trajectory)")
        res["P4_energy_conservative"] = dict(status="NOT_RUN", reason="no conservative production trajectory supplied")
    # informational: damped data period vs elliptic at measured cycle amplitude
    ex = extrema(t, th, om, al)
    if len(T_d) and ex:
        amp_first = max(abs(v) for _, v in ex[:3])
        p3["info_damped_first_period_vs_elliptic_at_first_amplitude_rel"] = float(T_d[0] / period_exact(model, amp_first) - 1)
    res["P3_finite_amplitude_period"] = p3

    # ---------------- P7 tension
    p7 = {}
    if model.m_bob is not None and model.L_c is not None:
        Tb = model.m_bob * (model.g * np.cos(th) + model.L_c * om ** 2)
        Tb_ref = model.m_bob * (model.g * np.cos(ref_th) + model.L_c * ref_om ** 2)
        p7.update(recomputed_min_N=float(Tb.min()), recomputed_max_N=float(Tb.max()),
                  ref_min_N=float(Tb_ref.min()), column_min_N=float(d["tension_N"].min()),
                  column_max_N=float(d["tension_N"].max()),
                  column_vs_recomputed_max_rel=float(np.max(np.abs(d["tension_N"] - Tb) / np.abs(Tb))),
                  formula="T_bob = m_bob (g cos th + L_c om^2)")
        p7["column_matches_recomputation"] = bool(p7["column_vs_recomputed_max_rel"] < TENSION_REL)
        ok7 = Tb.min() > 0 and Tb_ref.min() > 0 and d["tension_N"].min() > 0
    else:
        p7["note"] = "m_bob or L_c missing: only column sign checked"
        p7.update(column_min_N=float(d["tension_N"].min()))
        ok7 = d["tension_N"].min() > 0
    p7["status"] = status(ok7)
    res["P7_tension"] = p7

    rep["results"] = res
    rep["summary"] = {k: v.get("status") for k, v in res.items()}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(jsonable(rep), f, indent=2)
    print(json.dumps(jsonable(rep["summary"]), indent=1))


if __name__ == "__main__":
    main()
