#!/usr/bin/env python
"""P2, P3, P4, P7, P8 for clip 02 (verifier-owned; never imports solver/).

  .venv/bin/python verify/check_physics.py --params params/physics.json --trajectory solver/out/trajectory.csv \
      [--meta solver/out/trajectory_meta.json] [--colmap colmap.json] [--model-json override.json] \
      [--ref verify/out/ref_trajectory.csv]  (computed if absent) \
      [--conservative-trajectory RUN_WITH_DAMPING_ZEROED.csv --conservative-params P0.json] \
      [--bench-manifest verify/out/bench/bench_manifest.json] \
      --out verify/out/check_physics.json

Positions are of the EYE E (spring attachment on the weight). Columns are mapped by alias (vio.COL_ALIASES)
or an explicit --colmap {canonical: production_name}.
"""
import argparse, datetime, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vparams import load_model, sha256_file  # noqa: E402
from vmodel import quat_to_R, rot_angle, axial_exact, conical, rigid_planar_modes  # noqa: E402
from vio import read_traj, read_csv, pos, vel, quat, jsonable, dump_json, fit_sinusoid  # noqa: E402
from vcore import state_from_cols, local_propagate, grid_index, time_scale_estimate, OUT_DT  # noqa: E402
import ref_solver  # noqa: E402

TOL = dict(P2_pos=1e-6, P2_rad=1e-6, P3_rel=1e-7, P4_cons=1e-8, P4_bal=1e-7, P4_growth=1e-12, P4_cols=1e-9, P4_Lz=1e-8,
           P8_ic=1e-9, P8_t=1e-12, P8_geom=1e-9, P8_scale=1e-6, P7_rel=1e-9)


def ref_for(model, ref_path=None, t_end=None):
    if ref_path and os.path.exists(ref_path):
        d, _ = read_csv(ref_path)
        return d
    t_end = t_end or ref_solver.T_END_DEFAULT
    t, Y, info = ref_solver.integrate_gl(model, t_end, sub=8)
    d = dict(t_s=t, **model.observables(Y))
    if ref_path:
        ref_solver.run_for_params(model, ref_path, t_end)
    return d


# ------------------------------------------------------------------ P2
def p2(model, d, ref):
    m, k = grid_index(d["t_s"])
    m &= k < len(ref["t_s"])
    Xp, Xr = pos(d)[m], pos(ref)[k[m]]
    dp = np.linalg.norm(Xp - Xr, axis=1)
    r = dict(n_compared=int(m.sum()), n_rows=len(d["t_s"]), t_max_compared=float(d["t_s"][m].max()),
             max_pos_diff_m=float(dp.max()), t_at_max=float(d["t_s"][m][np.argmax(dp)]), rms_pos_diff_m=float(np.sqrt(np.mean(dp ** 2))))
    if all(c in d for c in ("vx_m_s", "vy_m_s", "vz_m_s")):
        r["max_vel_diff_m_s"] = float(np.max(np.linalg.norm(vel(d)[m] - vel(ref)[k[m]], axis=1)))
    ok = r["max_pos_diff_m"] < TOL["P2_pos"]
    if model.body == "rigid":
        qp = quat(d)
        if qp is None:
            r["attitude"] = "MISSING quaternion columns"; ok = False
        else:
            ang = rot_angle(quat_to_R(qp[m]), quat_to_R(quat(ref)[k[m]]))
            r["max_attitude_diff_rad"] = float(ang.max())
            ok = ok and r["max_attitude_diff_rad"] < TOL["P2_rad"]
    r["series"] = dict(t=d["t_s"][m][::4], dpos=dp[::4])
    r["status"] = "PASS" if ok else "FAIL"
    return r


# ------------------------------------------------------------------ P4
def p4_columns(model, d):
    """recompute KE/PE/E/tension/ell from state columns; compare with columns."""
    Y = state_from_cols(model, d)
    O = model.observables(Y)
    return Y, O


def p4(model, d, conservative_d=None, conservative_model=None):
    r = {}
    Y, O = p4_columns(model, d)
    Es = model.E_static()
    E0 = O["E_J"][0] - Es
    r["E_osc0_J"] = float(E0)
    r["recomputed_vs_columns"] = {}
    ok_cols = True
    for c in ("KE_J", "E_J", "PE_J"):
        if c in d:
            diff = d[c] - O[c]
            off = float(np.median(diff)) if c != "KE_J" else 0.0
            dev = float(np.max(np.abs(diff - off))) / E0
            r["recomputed_vs_columns"][c] = dict(constant_offset_J=off, max_dev_over_E0=dev)
            ok_cols &= dev < TOL["P4_cols"]
        else:
            r["recomputed_vs_columns"][c] = "column missing"
            if c in ("E_J",):
                ok_cols = False
    r["columns_ok"] = ok_cols
    # balance: local GL propagation from each production state integrates P_diss over each interval
    m, k = grid_index(d["t_s"])
    idx = np.where(m)[0]
    if len(idx) > 1 and np.all(np.diff(k[idx]) == 1):
        Yn, dW = local_propagate(model, Y[idx[:-1]])
        W = np.r_[0.0, np.cumsum(dW)]
        Ecol = d["E_J"][idx] if "E_J" in d else O["E_J"][idx]
        Erec = O["E_J"][idx]
        bal_rec = np.abs(Erec + W - Erec[0]) / E0
        r["balance_recomputed_E_max_over_E0"] = float(bal_rec.max())
        r["balance_column_E_max_over_E0"] = float(np.max(np.abs(Ecol + W - Ecol[0]) / E0))
        if "W_diss_J" in d:
            r["column_W_vs_verifier_W_max_over_E0"] = float(np.max(np.abs(d["W_diss_J"][idx] - W)) / E0)
            r["balance_all_columns_max_over_E0"] = float(np.max(np.abs(Ecol + d["W_diss_J"][idx] - Ecol[0])) / E0)
        # local one-step consistency (info): verifier propagation of production state n vs production state n+1
        On = model.observables(Yn)
        r["info_local_step_pos_diff_max_m"] = float(np.max(np.linalg.norm(np.stack([On[c] - O[c][idx[1:]] for c in ("x_m", "y_m", "z_m")], -1), axis=1)))
        r["series"] = dict(t=d["t_s"][idx][::4], balance=((Erec + W - Erec[0]) / E0)[::4], E=((Ecol - Es) / E0)[::4])
        bal = r["balance_recomputed_E_max_over_E0"]
        bal_ok = bal < TOL["P4_bal"] and r["balance_column_E_max_over_E0"] < TOL["P4_bal"]
        if "W_diss_J" in d:
            bal_ok = bal_ok and r["balance_all_columns_max_over_E0"] < TOL["P4_bal"]
    else:
        r["balance"] = "production rows are not a contiguous 1/96 s grid"; bal_ok = False
    dEf = np.diff(d["E_J"] if "E_J" in d else O["E_J"]) / E0
    r["max_forward_dE_over_E0"] = float(dEf.max())
    growth_ok = r["max_forward_dE_over_E0"] <= TOL["P4_growth"]
    if not model.damped:
        r["note"] = "main run is undamped: balance == drift; forward-difference growth reported but judged on drift only"
        drift = float(np.max(np.abs(O["E_J"] - O["E_J"][0])) / E0)
        r["undamped_drift_main_over_E0"] = drift
        growth_ok = True
        bal_ok = drift < TOL["P4_cons"] and bal_ok
    ok = ok_cols and bal_ok and growth_ok
    # conservative production run (damping zeroed) -> drift + Lz
    if conservative_d is not None:
        cm = conservative_model
        Yc = state_from_cols(cm, conservative_d)
        Oc = cm.observables(Yc)
        E0c = Oc["E_J"][0] - cm.E_static()
        drift = np.abs(Oc["E_J"] - Oc["E_J"][0]) / E0c
        Lref = cm.Mt * cm.ell_eq * float(np.max(np.linalg.norm(vel(conservative_d), axis=1)))
        dLz = np.abs(Oc["Lz"] - Oc["Lz"][0]) / Lref
        cons = dict(E_osc0_J=float(E0c), max_drift_over_E0=float(drift.max()), max_Lz_drift_over_Lref=float(dLz.max()), L_ref=Lref,
                    series=dict(t=conservative_d["t_s"][::4], drift=((Oc["E_J"] - Oc["E_J"][0]) / E0c)[::4], dLz=((Oc["Lz"] - Oc["Lz"][0]) / Lref)[::4]))
        if "E_J" in conservative_d:
            cons["column_drift_over_E0"] = float(np.max(np.abs(conservative_d["E_J"] - conservative_d["E_J"][0])) / E0c)
        cons["status"] = "PASS" if (cons["max_drift_over_E0"] < TOL["P4_cons"] and cons.get("column_drift_over_E0", 0) < TOL["P4_cons"]
                                    and cons["max_Lz_drift_over_Lref"] < TOL["P4_Lz"]) else "FAIL"
        r["conservative_run"] = cons
        ok = ok and cons["status"] == "PASS"
    else:
        r["conservative_run"] = "NOT RUN (no --conservative-trajectory)"
    r["status"] = "PASS" if ok else "FAIL"
    return r


# ------------------------------------------------------------------ P7
def p7(model, d):
    X = pos(d)
    ell = np.linalg.norm(X - model.A, axis=1)
    T = model.F_i + model.k * (ell - model.L0)
    r = dict(min_tension_recomputed_N=float(T.min()), max_tension_recomputed_N=float(T.max()), min_ell_m=float(ell.min()), L0=model.L0,
             t_min_tension=float(d["t_s"][np.argmin(T)]))
    ok = T.min() >= 0
    if model.F_i > 0:
        r["min_ell_minus_L0_m"] = float(ell.min() - model.L0)
        ok = ok and ell.min() >= model.L0
    if "tension_N" in d:
        r["min_tension_column_N"] = float(d["tension_N"].min())
        r["column_vs_recomputed_max_rel"] = float(np.max(np.abs(d["tension_N"] - T)) / max(np.max(np.abs(T)), 1e-30))
        ok = ok and d["tension_N"].min() >= 0 and r["column_vs_recomputed_max_rel"] < TOL["P7_rel"]
    else:
        r["tension_column"] = "missing (only recomputation judged)"
    r["series"] = dict(t=d["t_s"][::4], T=T[::4])
    r["status"] = "PASS" if ok else "FAIL"
    return r


# ------------------------------------------------------------------ P8
def p8(model, d, ref):
    r = {}
    t = d["t_s"]
    dt = t[1] - t[0]
    sub = int(round(OUT_DT / dt))
    r["dt_s"] = float(dt); r["samples_per_1_96"] = sub
    nexp = int(round(10.0 / dt)) + 1
    ii = np.arange(len(t))
    r["max_abs_t_minus_i_dt"] = float(np.max(np.abs(t - ii * (OUT_DT / max(sub, 1)))))
    frames = np.arange(241) / 24.0
    has = [bool(np.any(np.abs(t - f) <= TOL["P8_t"])) for f in frames]
    r["all_frame_times_present_exact"] = bool(all(has))
    r["n_rows"] = len(t); r["covers_0_10s"] = bool(t[0] == 0.0 and t[-1] >= 10.0 - 1e-12)
    t_ok = sub >= 1 and r["max_abs_t_minus_i_dt"] <= TOL["P8_t"] and r["all_frame_times_present_exact"] and r["covers_0_10s"] and len(t) >= nexp
    # initial conditions
    y0 = model.initial_state()
    O0 = model.observables(y0[None])
    ic = dict(E0_param=model.E0, E0_traj=pos(d)[0], dE0=float(np.max(np.abs(pos(d)[0] - model.E0))))
    if all(c in d for c in ("vx_m_s", "vy_m_s", "vz_m_s")):
        ic["dvE0"] = float(np.max(np.abs(vel(d)[0] - model.vE0)))
    else:
        ic["dvE0"] = "velocity columns missing"
    ic_ok = ic["dE0"] < TOL["P8_ic"] and isinstance(ic["dvE0"], float) and ic["dvE0"] < TOL["P8_ic"]
    if model.body == "rigid":
        q = quat(d)
        ic["dq0_rad"] = float(rot_angle(quat_to_R(q[0]), quat_to_R(model.q0))) if q is not None else "missing"
        ic["dw0"] = float(np.max(np.abs(np.array([d["wx"][0], d["wy"][0], d["wz"][0]]) - model.w0))) if "wx" in d else "missing"
        ic_ok = ic_ok and isinstance(ic["dq0_rad"], float) and ic["dq0_rad"] < TOL["P8_ic"] and isinstance(ic["dw0"], float) and ic["dw0"] < TOL["P8_ic"]
    r["initial_conditions"] = ic
    # geometric consistency of reported columns
    geo = {}
    geo_ok = True
    if "ell_m" in d:
        geo["ell_col_vs_norm_max_m"] = float(np.max(np.abs(d["ell_m"] - np.linalg.norm(pos(d) - model.A, axis=1))))
        geo_ok &= geo["ell_col_vs_norm_max_m"] < TOL["P8_geom"]
    if all(c in d for c in ("xc_m", "yc_m", "zc_m")):
        Y = state_from_cols(model, d)
        O = model.observables(Y)
        geo["com_col_vs_E_minus_R_eb_max_m"] = float(np.max(np.abs(np.stack([d["xc_m"] - O["xc_m"], d["yc_m"] - O["yc_m"], d["zc_m"] - O["zc_m"]], -1))))
        geo_ok &= geo["com_col_vs_E_minus_R_eb_max_m"] < TOL["P8_geom"]
    r["geometry_consistency"] = geo
    # time scaling vs verifier reference
    m, k = grid_index(t)
    m &= (k < len(ref["t_s"])) & (t <= 10.0 + 1e-12)
    s, c_s, c_1 = time_scale_estimate(t[m], pos(d)[m], ref["t_s"], pos(ref), vel(ref))
    r["time_scale"] = dict(best_s=s, rel=abs(s - 1), rms_at_best=c_s, rms_at_1=c_1,
                           method="s minimising RMS|E_prod(t) - E_ref(s t)| (cubic Hermite on ref)")
    scale_ok = abs(s - 1) < TOL["P8_scale"]
    r["checks"] = dict(time_grid=t_ok, initial_conditions=ic_ok, geometry=geo_ok, no_time_scaling=scale_ok)
    r["status"] = "PASS" if all(r["checks"].values()) else "FAIL"
    return r


# ------------------------------------------------------------------ P3 benchmarks
def p3(manifest):
    out = {}
    allok = True
    for ent in manifest["runs"]:
        kind, pp, tp = ent["kind"], ent["params"], ent["trajectory"]
        model, _ = load_model(pp, ent.get("model_json"))
        d, used = read_traj(tp, ent.get("colmap"))
        t = d["t_s"]; X = pos(d)
        res = dict(kind=kind, params=pp, trajectory=tp, sha256=sha256_file(tp))
        if kind == "rest":
            dev = np.max(np.linalg.norm(X - (model.A - np.array([0, 0, model.ell_eq])), axis=1)) / model.ell_eq
            res.update(ell_eq=model.ell_eq, max_dev_over_ell_eq=float(dev), status="PASS" if dev < TOL["P3_rel"] else "FAIL")
        elif kind == "axial":
            a = -(model.E0[2] - (model.A[2] - model.ell_eq))
            ze, wd, gam = axial_exact(model, t, a)
            err = float(np.max(np.abs(X[:, 2] - ze)) / abs(a))
            lat = float(np.max(np.abs(X[:, :2] - model.A[:2])) / abs(a))
            wfit, gfit, _, _ = fit_sinusoid(t, X[:, 2], wd, decay=model.damped)
            res.update(a=a, omega_pred=wd, gamma_pred=gam, omega_fit=wfit, gamma_fit=gfit, max_err_over_amp=err, lateral_over_amp=lat,
                       rel_omega_err=abs(wfit / wd - 1),
                       series=dict(t=t[::4], err=(X[:, 2] - ze)[::4] / abs(a)))
            res["status"] = "PASS" if err < TOL["P3_rel"] and lat < TOL["P3_rel"] and res["rel_omega_err"] < TOL["P3_rel"] else "FAIL"
        elif kind == "conical":
            rho0 = float(np.hypot(*(model.E0[:2] - model.A[:2])))
            alpha = math.atan2(rho0, model.A[2] - model.E0[2])
            cc = conical(model, alpha)
            ph0 = math.atan2(model.E0[1] - model.A[1], model.E0[0] - model.A[0])
            sense = np.sign(np.cross(np.r_[model.E0[:2] - model.A[:2], 0], np.r_[model.vE0[:2], 0])[2]) or 1.0
            ex = np.stack([model.A[0] + cc["rho"] * np.cos(ph0 + sense * cc["Omega"] * t), model.A[1] + cc["rho"] * np.sin(ph0 + sense * cc["Omega"] * t),
                           np.full_like(t, cc["z"])], -1)
            ic_rel = float(max(abs(np.linalg.norm(model.E0 - model.A) - cc["ell"]), abs(np.linalg.norm(model.vE0) - cc["rho"] * cc["Omega"])) / cc["rho"])
            err = float(np.max(np.linalg.norm(X - ex, axis=1)) / cc["rho"])
            res.update(alpha_rad=alpha, Omega=cc["Omega"], rho=cc["rho"], ic_consistency_rel=ic_rel, max_err_over_rho=err,
                       series=dict(t=t[::4], err=np.linalg.norm(X - ex, axis=1)[::4] / cc["rho"]))
            res["status"] = "PASS" if err < TOL["P3_rel"] and ic_rel < 1e-12 else "FAIL"
        elif kind in ("swing", "rigid_mode"):
            th_amp = float(math.atan2(np.hypot(*(model.E0[:2] - model.A[:2])), model.A[2] - model.E0[2]))
            if kind == "swing":
                wp = model.omega_s
                gam = model.c_v / (2 * model.Mt) if model.body != "rigid" else 0.0
                wp = math.sqrt(wp * wp - gam * gam)
            else:
                ws, _ = rigid_planar_modes(model, ent.get("axis", "y"))
                wp = float(ws[int(ent.get("mode", 0))])
            ax = 0 if ent.get("axis", "y") == "y" else 1
            wfit, gfit, _, _ = fit_sinusoid(t, X[:, ax], wp, decay=model.damped, w_rel_window=0.05)
            res.update(amplitude_rad=th_amp, omega_pred=wp, omega_fit=wfit, rel_omega_err=abs(wfit / wp - 1),
                       regime_ok=th_amp <= 1e-4 + 1e-12)
            res["status"] = "PASS" if res["rel_omega_err"] < TOL["P3_rel"] and res["regime_ok"] else "FAIL"
        else:
            res["status"] = f"UNKNOWN kind {kind}"
        allok &= res["status"] == "PASS"
        out[ent.get("name", kind)] = res
    out["status"] = "PASS" if allok and manifest["runs"] else "FAIL"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True); ap.add_argument("--model-json")
    ap.add_argument("--trajectory", required=True); ap.add_argument("--colmap"); ap.add_argument("--meta")
    ap.add_argument("--ref")
    ap.add_argument("--conservative-trajectory"); ap.add_argument("--conservative-params")
    ap.add_argument("--bench-manifest")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, P = load_model(a.params, a.model_json)
    d, used = read_traj(a.trajectory, a.colmap)
    rep = dict(label=a.label, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               inputs={a.params: sha256_file(a.params), a.trajectory: sha256_file(a.trajectory)}, column_mapping=used,
               model=model.as_dict(), provenance=model.provenance, tolerances=TOL,
               derived=dict(ell_eq=model.ell_eq, omega_z=model.omega_z, omega_s=model.omega_s, ratio_wz_over_ws=model.omega_z / model.omega_s),
               verifier_code_sha256={f: sha256_file(os.path.join(HERE, f)) for f in ("check_physics.py", "vmodel.py", "vparams.py", "vcore.py", "ref_solver.py", "vio.py")})
    if a.meta and os.path.exists(a.meta):
        meta = json.load(open(a.meta)); rep["inputs"][a.meta] = sha256_file(a.meta)
        rep["meta_params_sha256_matches"] = any(v == sha256_file(a.params) for v in _walk_strings(meta))
    ref = ref_for(model, a.ref)
    if a.ref:
        rep["inputs"][a.ref] = sha256_file(a.ref)
    rep["P2_independent_benchmark"] = p2(model, d, ref)
    cd = cm = None
    if a.conservative_trajectory:
        cm, _ = load_model(a.conservative_params or a.params, a.model_json if not a.conservative_params else None)
        if not a.conservative_params:
            cm = cm.replace(c_v=0.0, c_a=0.0, c_r=0.0)
        if cm.damped:
            raise SystemExit("conservative params still contain damping")
        cd, _ = read_traj(a.conservative_trajectory, a.colmap)
        rep["inputs"][a.conservative_trajectory] = sha256_file(a.conservative_trajectory)
    rep["P4_energy"] = p4(model, d, cd, cm)
    rep["P7_forces"] = p7(model, d)
    rep["P8_initial_conditions_time"] = p8(model, d, ref)
    if a.bench_manifest:
        rep["P3_analytic_limit"] = p3(json.load(open(a.bench_manifest)))
    else:
        rep["P3_analytic_limit"] = dict(status="NOT RUN", reason="no --bench-manifest (production runs in benchmark configurations)")
    rep["summary"] = {k: rep[v]["status"] for k, v in (("P2", "P2_independent_benchmark"), ("P3", "P3_analytic_limit"), ("P4", "P4_energy"),
                                                       ("P7", "P7_forces"), ("P8", "P8_initial_conditions_time"))}
    dump_json(rep, a.out)
    print("CHECK_PHYSICS_SUMMARY", json.dumps(rep["summary"]))


def _walk_strings(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _walk_strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk_strings(v)
    elif isinstance(o, str):
        yield o


if __name__ == "__main__":
    main()
