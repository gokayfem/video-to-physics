#!/usr/bin/env python
"""P2, P4, P7, P8 on a production trajectory (+ P3 via --bench). Verifier-owned; never imports solver/.

  .venv/bin/python verify/check_physics.py --params params/physics.json --trajectory solver/out/trajectory.csv \
      [--ref verify/out/ref_ck001.csv] [--model-json M.json] [--colmap C.json] \
      [--undamped-trajectory U.csv] [--solver-cmd "{py} solver/run.py --params {params} --out {outdir}"] [--bench] \
      [--lyapunov] --out verify/out/check_physics_ck001.json

--solver-cmd lets the checker produce the undamped run (P4 conservative) and the P3 benchmark runs itself on edited params
COPIES (verify/out/runs/...). Without it, P3 is NOT RUN (never PASS) and P4-conservative uses --undamped-trajectory if given.
"""
import argparse, copy, datetime, json, math, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from tmodel import Model, quat_to_R, rot_angle, twist_of, axis_angle_quat, EZ  # noqa: E402
from tio import read_traj, read_csv, pos, vel, dump_json, fit_sinusoid, jsonable  # noqa: E402
from tparams import load_model, write_variant, sha256_file  # noqa: E402
from tcore import state_from_cols, local_propagate, grid_index, time_scale_estimate, unit_quats, attach_pos, OUT_DT  # noqa: E402
import ref_solver  # noqa: E402

TOL = dict(P2_pos=1e-6, P2_rad=1e-6, P3_rel=1e-7, P4_cons=1e-8, P4_bal=1e-7, P4_step=1e-12, P8_ic=1e-9, P8_t=1e-12,
           P8_scale=1e-6, P8_cons=1e-9, P8_qnorm=1e-12)


def ref_for(model, t_end, path=None):
    if path and os.path.exists(path):
        return read_csv(path)[0]
    _, _, cols, _ = ref_solver.solve(model, t_end)
    if path:
        from tio import write_traj, CANON
        write_traj(path, cols, [c for c in CANON if c in cols])
    return cols


def common_rows(dp, dr):
    ok, k = grid_index(dp["t_s"])
    sel = ok & (k < len(dr["t_s"]))
    return np.where(sel)[0], k[sel]


def p2(model, dp, dr):
    ip, ir = common_rows(dp, dr)
    out = dict(n_common=int(len(ip)), t_max=float(dp["t_s"][ip[-1]]))
    dC = np.linalg.norm(pos(dp)[ip] - pos(dr)[ir], axis=1)
    out["C_max_m"] = float(dC.max()); out["C_t_at_max"] = float(dp["t_s"][ip][np.argmax(dC)])
    Pp, Pr = attach_pos(model, dp)[ip], attach_pos(model, dr)[ir]
    out["P_max_m"] = float(np.linalg.norm(Pp - Pr, axis=1).max())
    ang = rot_angle(quat_to_R(unit_quats(dr)[ir]), quat_to_R(unit_quats(dp)[ip]))
    out["quaternion_hemisphere_mismatches"] = int(np.sum(np.sum(unit_quats(dr)[ir] * unit_quats(dp)[ip], 1) < 0))
    out["attitude_max_rad"] = float(ang.max()); out["attitude_t_at_max"] = float(dp["t_s"][ip][np.argmax(ang)])
    if "psi_rad" in dp:
        dpsi = np.abs(dp["psi_rad"][ip] - dr["psi_rad"][ir])          # lifted twist: no 2 pi wrapping (a 2 pi slip must FAIL)
        out["psi_max_rad"] = float(dpsi.max())
    out["info_velocity_max_m_s"] = float(np.linalg.norm(vel(dp)[ip] - vel(dr)[ir], axis=1).max()) if "vx_m_s" in dp else None
    ok = out["quaternion_hemisphere_mismatches"] == 0 and out["C_max_m"] < TOL["P2_pos"] and out["P_max_m"] < TOL["P2_pos"] and out["attitude_max_rad"] < TOL["P2_rad"] and out.get("psi_max_rad", 0) < TOL["P2_rad"]
    out["status"] = "PASS" if ok else "FAIL"
    out["criterion"] = "max |dC|,|dP| < 1e-6 m; attitude angle and |dpsi| < 1e-6 rad vs verifier GL4 at identical params"
    return out, (ip, ir)


def p4(model, d, damped_expected, label):
    """energy from MY recomputation of the production states; balance via per-interval GL4 propagation of each state."""
    Y = state_from_cols(model, d)
    O = model.observables(Y)
    E = O["E_J"]
    Eosc = E[0] - model.E_static()
    out = dict(label=label, E_osc_J=float(Eosc))
    if "E_J" in d:
        off = d["E_J"] - E
        out["column_E_minus_recomputed_offset_J"] = float(np.median(off))
        out["column_E_vs_recomputed_max_rel"] = float(np.max(np.abs(off - np.median(off))) / Eosc)
        out["column_E_consistent"] = bool(out["column_E_vs_recomputed_max_rel"] < 1e-9)
    if "KE_J" in d:
        out["column_KE_vs_recomputed_max_rel"] = float(np.max(np.abs(d["KE_J"] - O["KE_J"])) / Eosc)
    ok_t, _ = grid_index(d["t_s"])
    Yn, dW, dZ = local_propagate(model, Y[:-1])
    W = np.r_[0.0, np.cumsum(dW)]
    bal = (E + W - E[0]) / Eosc
    out["max_rel_drift_E"] = float(np.max(np.abs(E - E[0])) / Eosc)
    out["max_rel_balance_residual"] = float(np.max(np.abs(bal)))
    dE = np.diff(E) / Eosc
    out["max_forward_dE_over_Eosc"] = float(dE.max())
    out["max_step_balance_residual_over_Eosc"] = float(np.max(np.abs(np.diff(E) + dW)) / Eosc)
    if "W_diss_J" in d:
        out["column_W_diss_vs_recomputed_max_rel"] = float(np.max(np.abs(d["W_diss_J"] - W)) / Eosc)
    # L_z balance (generic tau_z) -- added check, not a relaxation
    Lref = math.sqrt(2 * abs(Eosc) * model.I_c[2, 2])
    Z = np.r_[0.0, np.cumsum(dZ)]
    out["Lz_balance_max_rel"] = float(np.max(np.abs(O["Lz"] - O["Lz"][0] - Z)) / Lref)
    # local one-interval defect: production state(t+dt) vs my propagation of production state(t)
    ang = rot_angle(quat_to_R(Yn[:, (0 if model.mode == "rigid_pivot" else 6):(4 if model.mode == "rigid_pivot" else 10)]),
                    quat_to_R(Y[1:, (0 if model.mode == "rigid_pivot" else 6):(4 if model.mode == "rigid_pivot" else 10)]))
    On = model.observables(Yn)
    out["info_local_defect_C_max_m"] = float(np.max(np.linalg.norm(np.stack([On["x_m"] - O["x_m"][1:], On["y_m"] - O["y_m"][1:], On["z_m"] - O["z_m"][1:]], -1), axis=1)))
    out["info_local_defect_att_max_rad"] = float(ang.max())
    if damped_expected:
        ok = out["max_rel_balance_residual"] < TOL["P4_bal"] and out["max_forward_dE_over_Eosc"] <= TOL["P4_step"] and out["Lz_balance_max_rel"] < 1e-8
        out["criterion"] = "|E + W_diss - E0| < 1e-7 E_osc; forward dE <= 1e-12 E_osc; L_z balance < 1e-8 L_ref"
    else:
        ok = out["max_rel_drift_E"] < TOL["P4_cons"] and out["Lz_balance_max_rel"] < 1e-8
        out["criterion"] = "|E - E0| < 1e-8 E_osc (conservative run); L_z balance < 1e-8 L_ref"
    if "column_E_consistent" in out:
        ok = ok and out["column_E_consistent"]
    out["status"] = "PASS" if ok else "FAIL"
    out["_series"] = dict(t=d["t_s"][::4].tolist(), dE=((E - E[0]) / Eosc)[::4].tolist(), bal=bal[::4].tolist())
    return out


def p7(model, d):
    Y = state_from_cols(model, d)
    O = model.observables(Y)
    out = dict(recomputed_tension_min_N=float(O["tension_N"].min()), recomputed_tension_max_N=float(O["tension_N"].max()))
    ok = out["recomputed_tension_min_N"] >= 0
    if "tension_N" in d:
        out["column_tension_min_N"] = float(d["tension_N"].min())
        out["column_vs_recomputed_max_rel"] = float(np.max(np.abs(d["tension_N"] - O["tension_N"])) / (model.M * model.g))
        ok = ok and out["column_tension_min_N"] >= 0 and out["column_vs_recomputed_max_rel"] < 1e-6
    out["criterion"] = "tension >= 0 (column and recomputation M a_C + M g z + c_v v_C along u); column vs recomputed < 1e-6 Mg"
    out["status"] = "PASS" if ok else "FAIL"
    return out


def p8(model, dp, dr):
    out = {}
    t = dp["t_s"]
    n = np.arange(len(t))
    out["t_grid_max_dev_s"] = float(np.max(np.abs(t - n * OUT_DT)))
    out["n_rows"] = int(len(t)); out["t_last_s"] = float(t[-1])
    q = unit_quats(dp)
    out["quat_norm_max_dev"] = float(np.max(np.abs(np.linalg.norm(np.stack([dp[k] for k in ("qw", "qx", "qy", "qz")], -1), axis=1) - 1)))
    y0 = model.initial_state()
    O0 = model.observables(y0[None])
    R0 = quat_to_R(model.q0)
    ic = dict(C=float(np.linalg.norm(pos(dp)[0] - np.array([O0["x_m"][0], O0["y_m"][0], O0["z_m"][0]]))),
              att_rad=float(rot_angle(R0, quat_to_R(q[0]))),
              w=float(np.linalg.norm(np.array([dp["wx"][0], dp["wy"][0], dp["wz"][0]]) - model.w0)))
    if "vx_m_s" in dp:
        ic["vC"] = float(np.linalg.norm(vel(dp)[0] - np.array([O0["vx_m_s"][0], O0["vy_m_s"][0], O0["vz_m_s"][0]])))
    out["ic_deviation"] = ic
    # column consistency
    R = quat_to_R(q)
    P = attach_pos(model, dp)
    out["wire_length_max_dev_m"] = float(np.max(np.abs(np.linalg.norm(P - model.A, axis=1) - model.L)))
    if all(k in dp for k in ("px_m", "py_m", "pz_m")):
        out["C_vs_P_minus_Rpb_max_m"] = float(np.max(np.linalg.norm(pos(dp) - (P - np.einsum("nij,j->ni", R, model.p_b)), axis=1)))
    u = (model.A - P) / np.linalg.norm(model.A - P, axis=1, keepdims=True)
    if model.mode == "rigid_pivot":
        out["rigid_pivot_disc_normal_vs_wire_max_rad"] = float(np.max(np.arccos(np.clip(np.sum(R[:, :, 2] * u, 1), -1, 1))))
    if "psi_rad" in dp:
        conv = {c: float(np.max(np.abs(np.angle(np.exp(1j * (dp["psi_rad"] - twist_of(c, u, R[:, :, 2], R[:, :, 0]))))))) for c in ("transport", "vertical", "euler_zyx")}
        conv["vertical_lifted_from_quaternion (no wrap)"] = float(np.max(np.abs(dp["psi_rad"] - twist_of("vertical", u, R[:, :, 2], R[:, :, 0], q))))
        if model.twist == "vertical":
            conv["vertical"] = conv["vertical_lifted_from_quaternion (no wrap)"]
        out["psi_column_vs_conventions_max_rad"] = conv
        out["psi_column_vs_model_convention_max_rad"] = conv[model.twist]
    # time scale estimate vs ref (twist and CoM)
    ip, ir = common_rows(dp, dr)
    Rd = 0.05
    if "psidot_rad_s" in dp or True:
        Xp = np.c_[pos(dp)[ip], Rd * dp["psi_rad"][ip]] if "psi_rad" in dp else pos(dp)[ip]
        Xr = np.c_[pos(dr), Rd * dr["psi_rad"]] if "psi_rad" in dp else pos(dr)
        Vr = np.c_[vel(dr), Rd * dr["psidot_rad_s"]] if "psi_rad" in dp else vel(dr)
        s, c_s, c_1 = time_scale_estimate(dp["t_s"][ip], Xp, dr["t_s"], Xr, Vr)
        out["time_scale_estimate"] = dict(s=float(s), rms_at_s=c_s, rms_at_1=c_1, rel=float(abs(s - 1)))
    ok = (out["t_grid_max_dev_s"] <= TOL["P8_t"] and out["quat_norm_max_dev"] < TOL["P8_qnorm"] and max(ic.values()) < TOL["P8_ic"]
          and out["wire_length_max_dev_m"] < TOL["P8_cons"] and out.get("C_vs_P_minus_Rpb_max_m", 0) < TOL["P8_cons"]
          and out.get("rigid_pivot_disc_normal_vs_wire_max_rad", 0) < TOL["P8_cons"]
          and out.get("psi_column_vs_model_convention_max_rad", 0) < TOL["P8_cons"]
          and (out["time_scale_estimate"]["rel"] < TOL["P8_scale"] or out["time_scale_estimate"]["rms_at_1"] < 1e-9))
    out["criterion"] = ("t[n] = n/96 (1e-12 s); |q| = 1 (1e-12); IC vs params < 1e-9; |P-A| = L, C = P - R p_b, psi column = model "
                        "convention (1e-9); best-fit time scale |s-1| < 1e-6")
    out["status"] = "PASS" if ok else "FAIL"
    return out


# ------------------------------------------------------------------ production runs (subprocess only)
def run_production(solver_cmd, params_copy, outdir, traj_name="trajectory.csv"):
    cmd = solver_cmd.format(py=os.path.join(ROOT, ".venv", "bin", "python"), params=params_copy, outdir=outdir,
                            traj=os.path.join(outdir, traj_name))
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    if r.returncode:
        raise RuntimeError(f"production command failed ({r.returncode}): {cmd}\n{r.stderr[-1500:]}")
    return os.path.join(outdir, traj_name), cmd


def variant(P, model, edits, tag, solver_cmd, workdir, colmap=None, editor=None):
    d = os.path.join(workdir, tag); os.makedirs(d, exist_ok=True)
    pp = os.path.join(d, "physics.json")
    done = (editor or write_variant)(P, model, edits, pp, note=f"verifier variant {tag}")
    tr, cmd = run_production(solver_cmd, pp, d)
    return read_traj(tr, colmap)[0], dict(tag=tag, edits=done, cmd=cmd, params_copy=pp, trajectory=tr, sha256=sha256_file(tr))


DAMP = ("c_t", "c_s", "c_v", "c_r", "c_rp")


def zero_damping(model=None):
    return {k: 0.0 for k in DAMP if model is None or getattr(model, k) != 0 or k in model.provenance}


def bench(P, model, solver_cmd, workdir, colmap, editor=None):
    out = {}
    base_ic = dict(u0=[0.0, 0.0, 1.0], udot0=[0.0, 0.0, 0.0], w0=[0.0, 0.0, 0.0])
    present = [k for k in DAMP if k in model.provenance or getattr(model, k) != 0]
    ws, V = model.swing_modes("y")
    if model.mode == "string":
        ph1, ph2 = V[:, 0] / np.max(np.abs(V[:, 0])) * 1e-5
        sw_ic = dict(u0=(quat_to_R(axis_angle_quat([0, 1, 0], ph1)) @ EZ).tolist(), q0=axis_angle_quat([0, 1, 0], ph2).tolist())
    else:
        sw_ic = dict(q0=axis_angle_quat([0, 1, 0], 1e-5).tolist())
    cases = [("B2_torsion_undamped_0.5rad", dict(base_ic, q0=axis_angle_quat(EZ, 0.5).tolist(), **{k: 0.0 for k in present}), "twist"),
             ("B3_torsion_damped_0.5rad", dict(base_ic, q0=axis_angle_quat(EZ, 0.5).tolist()), "twist"),
             ("B4_swing_1e-5rad_lowest_mode", dict(base_ic, **sw_ic, **{k: 0.0 for k in present}), "swing")]
    for tag, edits, kind in cases:
        if model.mode == "rigid_pivot":
            edits = {k: v for k, v in edits.items() if k not in ("u0", "udot0")}
        try:
            d, info = variant(P, model, edits, tag, solver_cmd, workdir, colmap, editor)
        except (SystemExit, RuntimeError, KeyError, ValueError) as e:
            out[tag] = dict(status="NOT RUN", reason=str(e)[:600]); continue
        m = model.replace(**edits)
        t = d["t_s"]
        if kind == "twist":
            pe, wd, gam = m.torsion_exact(t, 0.5, 0.0)
            psi = d["psi_rad"] if "psi_rad" in d else np.array([twist_of(m.twist, EZ, quat_to_R(q)[:, 2], quat_to_R(q)[:, 0]) for q in unit_quats(d)])
            err = float(np.max(np.abs(psi - pe)) / 0.5)
            wfit, gfit, _, _ = fit_sinusoid(t, psi, wd, decay=gam > 0)
            res = dict(omega_pred=wd, gamma_pred=gam, max_err_over_amp=err, omega_fit=wfit, rel_omega_err=abs(wfit / wd - 1),
                       gamma_fit=gfit, rel_gamma_err=(abs(gfit / gam - 1) if gam else None),
                       swing_leak_m=float(np.max(np.hypot(d["x_m"] - d["x_m"][0], d["y_m"] - d["y_m"][0]))))
            res["status"] = "PASS" if (err < TOL["P3_rel"] and res["rel_omega_err"] < TOL["P3_rel"]) else "FAIL"
        else:
            ws, _ = m.swing_modes("y")
            wfit, _, _, _ = fit_sinusoid(t, d["x_m"], ws[0], w_rel_window=0.05)
            res = dict(omega_pred=float(ws[0]), omega_fit=wfit, rel_omega_err=abs(wfit / ws[0] - 1))
            res["status"] = "PASS" if res["rel_omega_err"] < TOL["P3_rel"] else "FAIL"
            if model.mode == "string":
                res["note"] = "string mode: released in the lowest linear mode shape (wire angle, disc tilt) from the B4 eigenproblem"
        res["run"] = info
        out[tag] = res
    sts = [v["status"] for v in out.values()]
    out["status"] = "NOT RUN" if all(s == "NOT RUN" for s in sts) else ("PASS" if all(s == "PASS" for s in sts) else ("FAIL" if "FAIL" in sts else "PARTIAL"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True); ap.add_argument("--model-json"); ap.add_argument("--trajectory", required=True)
    ap.add_argument("--ref"); ap.add_argument("--colmap"); ap.add_argument("--undamped-trajectory")
    ap.add_argument("--solver-cmd"); ap.add_argument("--bench", action="store_true"); ap.add_argument("--lyapunov", action="store_true")
    ap.add_argument("--workdir", default=os.path.join(HERE, "out", "runs")); ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, P = load_model(a.params, a.model_json)
    dp, used = read_traj(a.trajectory, a.colmap)
    t_end = float(dp["t_s"][-1])
    dr = ref_for(model, max(t_end, 1e-9), a.ref)
    rep = dict(label=a.label, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               inputs={a.params: sha256_file(a.params), a.trajectory: sha256_file(a.trajectory)}, columns_used=used,
               model=model.as_dict(), provenance=model.provenance, tolerances=TOL,
               verifier_code_sha256={f: sha256_file(os.path.join(HERE, f)) for f in ("check_physics.py", "tmodel.py", "ref_solver.py", "tcore.py", "tparams.py")})
    if a.ref:
        rep["inputs"][a.ref] = sha256_file(a.ref)
    rep["P2_independent_benchmark"], _ = p2(model, dp, dr)
    rep["P4_energy_production_run"] = p4(model, dp, model.damped, "production run (as given)")
    und = None
    if a.undamped_trajectory:
        und = read_traj(a.undamped_trajectory, a.colmap)[0]; src = a.undamped_trajectory
    elif a.solver_cmd and model.damped:
        try:
            und, info = variant(P, model, zero_damping(model), "P4_undamped", a.solver_cmd, a.workdir, a.colmap)
            src = info
        except (SystemExit, RuntimeError, KeyError, ValueError) as e:
            rep["P4_energy_conservative"] = dict(status="NOT RUN", reason=str(e)[:600])
    if und is not None:
        mu = model.replace(**zero_damping())
        rep["P4_energy_conservative"] = p4(mu, und, False, "conservative variant")
        rep["P4_energy_conservative"]["source"] = src
        rep["P2_conservative_variant"], _ = p2(mu, und, ref_for(mu, float(und["t_s"][-1])))
    elif not model.damped:
        rep["P4_energy_conservative"] = dict(status=rep["P4_energy_production_run"]["status"], note="production run is itself undamped")
    rep["P7_forces"] = p7(model, dp)
    rep["P8_initial_conditions_time"] = p8(model, dp, dr)
    if a.bench:
        rep["P3_analytic_limit"] = bench(P, model, a.solver_cmd, a.workdir, a.colmap) if a.solver_cmd else dict(status="NOT RUN", reason="no --solver-cmd")
    else:
        rep["P3_analytic_limit"] = dict(status="NOT RUN", reason="--bench not requested")
    if a.lyapunov:
        rep["info_lyapunov"] = ref_solver.lyapunov_estimate(model)
    p4s = [rep["P4_energy_production_run"]["status"], rep.get("P4_energy_conservative", {}).get("status", "NOT RUN")]
    rep["summary"] = dict(P2=rep["P2_independent_benchmark"]["status"], P3=rep["P3_analytic_limit"]["status"],
                          P4="FAIL" if "FAIL" in p4s else ("PASS" if all(s == "PASS" for s in p4s) else "PARTIAL"),
                          P7=rep["P7_forces"]["status"], P8=rep["P8_initial_conditions_time"]["status"])
    dump_json(rep, a.out)
    print("CHECK_PHYSICS_SUMMARY", json.dumps(rep["summary"]))


if __name__ == "__main__":
    main()
