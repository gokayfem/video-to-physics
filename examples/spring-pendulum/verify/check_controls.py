#!/usr/bin/env python
"""P9 control response through the real solver (+ optional build/bake + check_blend). Verifier-owned.

Variants (task brief + PLAN.md), each in a clean analytic regime, expected values from formulas only:
  k_x4        axial release (a = 2% ell_eq): omega_d from B2  -> axial period / 2 (exactly / sqrt-law with damping)
  M_x4        axial release: omega_z^2 = k/(4M + m_s/3)     -> period x2 for m_s = 0
  L0_plus50   1e-4 rad swing (point; rigid: lowest rigid mode): omega from B4/B5 at the new ell_eq
  damping_x10 axial release: gamma = 10 (c_a + c_v) / (2 M_t), omega_d from B2
Each variant is also run at the PRODUCTION initial conditions and compared with the verifier GL4 solution at the
variant params (max |dE| < 1e-6 m), i.e. 'the equations predict' in the full nonlinear regime.
Tolerance 1e-4 relative (acceptance P9) on frequencies/decay rates; P2-type 1e-6 m on full runs.

  .venv/bin/python verify/check_controls.py --params params/physics.json --tag ck001 \
      --solver-cmd "{py} solver/run.py --params {params} --out {outdir}" \
      [--bake-cmd "tools/with_render_lock.sh blender -b --factory-startup --python scene/build.py -- --out {blend} --bake {traj} ..."] \
      [--blend-check-args "--eye-name VSP_Weight_Eye ..."] --out verify/out/check_controls_ck001.json
"""
import argparse, datetime, json, math, os, shlex, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from vparams import load_model, write_variant, sha256_file  # noqa: E402
from vmodel import rigid_planar_modes, axis_angle_quat  # noqa: E402
from vio import read_traj, pos, dump_json, fit_sinusoid, jsonable  # noqa: E402
from vcore import grid_index  # noqa: E402
import ref_solver  # noqa: E402

TOL, TOL_POS = 1e-4, 1e-6
B = "blender"


def regime_edits(m, kind):
    A, l = m.A, m.ell_eq
    e = dict(vE0=[0.0, 0.0, 0.0])
    if m.body == "rigid":
        e.update(q0=[1.0, 0.0, 0.0, 0.0], w0=[0.0, 0.0, 0.0])
    if kind == "axial":
        e["E0"] = [A[0], A[1], A[2] - l * 1.02]
    else:
        if m.body == "rigid":
            ws, V = rigid_planar_modes(m, "y")
            ph1, ph2 = V[:, 0] / np.max(np.abs(V[:, 0])) * 1e-4
            e["E0"] = [A[0] + l * math.sin(ph1), A[1], A[2] - l * math.cos(ph1)]
            e["q0"] = axis_angle_quat([0, 1, 0], -ph2).tolist()
        else:
            e["E0"] = [A[0] + l * math.sin(1e-4), A[1], A[2] - l * math.cos(1e-4)]
    return e


def predicted(m, kind):
    if kind == "axial":
        gam = (m.c_a + m.c_v) / (2 * m.Mt)
        return math.sqrt(m.omega_z ** 2 - gam * gam), gam
    if m.body == "rigid":
        return float(rigid_planar_modes(m, "y")[0][0]), None   # damping-free lowest mode (c_r, c_v small: reported)
    gam = m.c_v / (2 * m.Mt)
    return math.sqrt(m.omega_s ** 2 - gam * gam), gam


VAR_SETS = {"base": [], "k_x4": ["--set k=x4"], "M_x4": ["--set m=x4", "--set I_perp=x4"], "L0_plus50": ["--set L0=x1.5"],
            "damping_x10": ["--set c_s=x10", "--set c_x=x10", "--set c_r=x10"]}


def ic_rule_check(m, P, pp, edits):
    """Independent check of a production re-equilibrated variant (CHANGELOG 2026-10-26T01:53:54Z rule):
    edited parameters as requested, others unchanged; same swing angle/rate of the spring about P as the base state;
    rocking on the variant's linear swing-mode shape (psi = kappa phi, verifier eigenvector); axial at rest; tension > 0."""
    mv, Q = load_model(pp)
    out = {}
    for k_, val in edits.items():
        if k_ == "I_c":
            out["I_perp"] = [float(mv.I_c[0, 0]), float(np.array(val)[0][0])]
        else:
            out[k_] = [getattr(mv, k_), val]
    out["edits_applied_max_rel"] = float(max(abs(a_ / b_ - 1) for a_, b_ in out.values())) if out else 0.0
    same = {k_: (getattr(mv, k_), getattr(m, k_)) for k_ in ("k", "L0", "M", "m_s", "c_v", "c_a", "c_r", "g") if k_ not in edits}
    out["unedited_params_max_rel_change"] = float(max((abs(x - y) / max(abs(y), 1e-300) for x, y in same.values()), default=0.0))
    def swing(mm):
        d = mm.E0 - mm.A; phi = math.atan2(d[0], -d[2])
        phid = (d[2] * mm.vE0[0] - d[0] * mm.vE0[2]) / (d[0] ** 2 + d[2] ** 2) * -1
        nb = quat_to_R_(mm.q0) @ np.array([0, 0, -1.0]); psi = math.atan2(nb[0], -nb[2])
        u = d / np.linalg.norm(d)
        return phi, phid, psi, float(u @ mm.vE0), float(np.linalg.norm(d))
    pb, pdb, _, _, _ = swing(m); pv, pdv, psv, ldot, ell0 = swing(mv)
    ws, V = rigid_planar_modes(mv, "y"); kappa = V[1, 0] / V[0, 0]
    T0 = mv.F_i + mv.k * (ell0 - mv.L0)
    out.update(phi0_base=pb, phi0_variant=pv, phid0_base=pdb, phid0_variant=pdv, psi_over_phi=psv / pv, kappa_verifier=kappa,
               rel_kappa=abs(psv / pv / kappa - 1), spring_rate0_m_s=ldot, ell0_m=ell0, ell_eq_variant_m=mv.ell_eq, tension0_N=T0)
    out["consistent"] = bool(out["edits_applied_max_rel"] < 1e-12 and out["unedited_params_max_rel_change"] < 1e-12 and abs(pv - pb) < 1e-9
                             and abs(pdv - pdb) < 1e-9 and out["rel_kappa"] < 1e-6 and T0 > 0)
    return out


def quat_to_R_(q):
    from vmodel import quat_to_R
    return quat_to_R(np.array(q))


def run_solver(cmd_t, pp, od):
    cmd = cmd_t.format(py=sys.executable, params=pp, outdir=od, traj=os.path.join(od, "trajectory.csv"))
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)
    return cmd, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True); ap.add_argument("--tag", default="run")
    ap.add_argument("--solver-cmd", default="{py} solver/run.py --params {params} --out {outdir}")
    ap.add_argument("--colmap"); ap.add_argument("--bake-cmd"); ap.add_argument("--blend-check-args", default="")
    ap.add_argument("--variants", nargs="*", default=["base", "k_x4", "M_x4", "L0_plus50", "damping_x10"])
    ap.add_argument("--skip-production-ic", action="store_true")
    ap.add_argument("--variant-cmd", help="production re-equilibrated variant generator, e.g. 'cd CK && {py} solver/make_variant.py --params params/physics.json {sets} --out {out}'")
    ap.add_argument("--nominal-ic-too", action="store_true", help="also run each variant from the nominal initial state and record the solver outcome")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m, P = load_model(a.params)
    base_dir = os.path.join(HERE, "out", "controls", a.tag)
    rep = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), tag=a.tag,
               base_params={a.params: sha256_file(a.params)}, solver_cmd_template=a.solver_cmd, bake_cmd_template=a.bake_cmd,
               tolerance=dict(rel=TOL, pos_m=TOL_POS), verifier_code_sha256=sha256_file(os.path.abspath(__file__)), variants={})
    var_edit = {"base": {}, "k_x4": {"k": m.k * 4}, "M_x4": {"M": m.M * 4}, "L0_plus50": {"L0": m.L0 * 1.5},
                "damping_x10": {k: getattr(m, k) * 10 for k in ("c_v", "c_a", "c_r") if getattr(m, k) != 0}}
    if m.body == "rigid":
        var_edit["M_x4"]["I_c"] = (m.I_c * 4).tolist()   # same body shape, 4x density
    base_meas = {}
    for v in a.variants:
        ent = dict(edits=var_edit[v], runs={})
        if v == "damping_x10" and not var_edit[v]:
            ent["status"] = "NOT APPLICABLE"; ent["reason"] = "params have no damping terms"; rep["variants"][v] = ent; continue
        mv = m.replace(**{k: (val if not isinstance(val, np.ndarray) else val.tolist()) for k, val in var_edit[v].items()})
        kinds = ["axial"] if v in ("base", "k_x4", "M_x4", "damping_x10") else []
        if v in ("base", "L0_plus50"):
            kinds.append("swing")
        ok = True
        for kind in kinds:
            vd = os.path.join(base_dir, v, kind); os.makedirs(vd, exist_ok=True)
            pp = os.path.join(vd, "physics.json")
            edits = dict(var_edit[v]); edits.update(regime_edits(mv, kind))
            write_variant(P, m, edits, pp, note=f"P9 {v} {kind}")
            cmd, r = run_solver(a.solver_cmd, pp, vd)
            run = dict(params=pp, cmd=cmd, rc=r.returncode, stderr_tail=r.stderr[-800:])
            tp = os.path.join(vd, "trajectory.csv")
            if r.returncode or not os.path.exists(tp):
                run["status"] = "FAIL"; ok = False; ent["runs"][kind] = run; continue
            mm, _ = load_model(pp)
            d, _ = read_traj(tp, a.colmap)
            X = pos(d); t = d["t_s"]
            wp, gp = predicted(mm, kind)
            sig = X[:, 2] if kind == "axial" else X[:, 0]
            decay = (gp is not None and gp > 0) or (kind == "swing" and mm.body == "rigid" and mm.damped)  # rigid: fit the envelope too
            wf, gf, _, rms = fit_sinusoid(t, sig, wp, decay=decay, w_rel_window=0.1)
            run.update(omega_pred=wp, omega_fit=wf, rel_omega=abs(wf / wp - 1), gamma_pred=gp, gamma_fit=gf,
                       rel_gamma=(abs(gf / gp - 1) if (decay and gp) else None), period_fit_s=2 * math.pi / wf, fit_rms=rms)
            run["status"] = "PASS" if run["rel_omega"] < TOL and (run["rel_gamma"] is None or run["rel_gamma"] < TOL) else "FAIL"
            if v == "base":
                base_meas[kind] = run
            elif kind in base_meas:
                run["period_ratio_meas"] = base_meas[kind]["omega_fit"] / wf
                run["period_ratio_pred"] = base_meas[kind]["omega_pred"] / wp
            ok &= run["status"] == "PASS"
            ent["runs"][kind] = run
        # full-regime: variant at production ICs vs verifier GL4 at the same variant params
        if not a.skip_production_ic:
            if a.nominal_ic_too and v != "base":
                nd = os.path.join(base_dir, v, "nominal_ic"); os.makedirs(nd, exist_ok=True)
                npp = os.path.join(nd, "physics.json")
                write_variant(P, m, var_edit[v], npp, note=f"P9 {v} nominal ICs (diagnostic)")
                ncmd, nr = run_solver(a.solver_cmd, npp, nd)
                ntp = os.path.join(nd, "trajectory.csv")
                nent = dict(cmd=ncmd, rc=nr.returncode, raised=[e for e in ("SpringCompressionError", "SolverGuardError") if e in nr.stderr],
                            stderr_tail=nr.stderr[-400:])
                if os.path.exists(ntp):
                    dn, _ = read_traj(ntp, a.colmap); nent["nonfinite_rows"] = int(np.sum(~np.isfinite(pos(dn)).all(1)))
                mvn, _ = load_model(npp)
                tq = ref_solver.integrate_gl(mvn, 0.5, sub=8)
                nent["verifier_min_tension_first_0p5s_N"] = float(mvn.observables(tq[1])["tension_N"].min())
                ent["runs"]["nominal_ic_diagnostic"] = nent
            vd = os.path.join(base_dir, v, "production_ic"); os.makedirs(vd, exist_ok=True)
            pp = os.path.join(vd, "physics.json")
            if a.variant_cmd:
                sets = " ".join(VAR_SETS[v])
                vc = a.variant_cmd.format(py=sys.executable, sets=sets, out=pp)
                vr = subprocess.run(vc, shell=True, cwd=ROOT, capture_output=True, text=True)
                if vr.returncode or not os.path.exists(pp):
                    raise SystemExit(f"variant command failed: {vc}\n{vr.stderr[-1500:]}")
            else:
                vc = None
                write_variant(P, m, var_edit[v], pp, note=f"P9 {v} production ICs")
            cmd, r = run_solver(a.solver_cmd, pp, vd)
            tp = os.path.join(vd, "trajectory.csv")
            run = dict(params=pp, cmd=cmd, rc=r.returncode, variant_cmd=vc, stderr_tail=r.stderr[-400:])
            if vc:
                run["ic_rule_check"] = ic_rule_check(m, P, pp, var_edit[v])
            if r.returncode or not os.path.exists(tp):
                run["status"] = "FAIL"; ok = False
            else:
                mm, _ = load_model(pp)
                d, _ = read_traj(tp, a.colmap)
                refp = os.path.join(vd, "ref.csv")
                ref_solver.run_for_params(mm, refp)
                from vio import read_csv
                ref, _ = read_csv(refp)
                msk, k = grid_index(d["t_s"]); msk &= k < len(ref["t_s"])
                dp = float(np.max(np.linalg.norm(pos(d)[msk] - pos(ref)[k[msk]], axis=1)))
                run.update(max_pos_diff_vs_verifier_m=dp, status="PASS" if dp < TOL_POS else "FAIL")
                ok &= run["status"] == "PASS"
                if a.bake_cmd:
                    bp = os.path.join(vd, "variant.blend")
                    bc = a.bake_cmd.format(params=pp, traj=tp, blend=bp)
                    br = subprocess.run(bc, shell=True, cwd=ROOT, capture_output=True, text=True)
                    cb = os.path.join(vd, "check_blend.json")
                    cr = subprocess.run([B, "-b", bp, "--python", os.path.join(HERE, "check_blend.py"), "--", "--params", pp, "--ref", refp,
                                         "--trajectory", tp, "--out", cb, "--label", f"P9:{v}"] + shlex.split(a.blend_check_args),
                                        cwd=ROOT, capture_output=True, text=True)
                    run["bake_cmd"] = bc; run["bake_rc"] = br.returncode
                    if os.path.exists(cb):
                        s = json.load(open(cb))["summary"]; run["blend_summary"] = s
                        ok &= s.get("P6") == "PASS"
                    else:
                        run["blend_summary"] = "check_blend failed: " + cr.stderr[-600:]; ok = False
            ent["runs"]["production_ic"] = run
        ent["status"] = "PASS" if ok else "FAIL"
        rep["variants"][v] = ent
        print(f"{v:12s} {ent['status']}  " + "  ".join(f"{k}:{r.get('rel_omega', r.get('max_pos_diff_vs_verifier_m'))}" for k, r in ent["runs"].items()))
    judged = [e["status"] for k, e in rep["variants"].items() if e["status"] != "NOT APPLICABLE"]
    rep["status"] = "PASS" if judged and all(s == "PASS" for s in judged) else "FAIL"
    rep["through_bake"] = bool(a.bake_cmd)
    dump_json(rep, a.out)
    print("P9", rep["status"], "(solver + bake + check_blend)" if a.bake_cmd else "(solver only)")


if __name__ == "__main__":
    main()
