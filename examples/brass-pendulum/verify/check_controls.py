#!/usr/bin/env python
"""P9 control response through the real build path (verifier-owned; never imports solver/).

Variants (acceptance P9): L_c x4, theta0 = 2 deg, theta0 = 15 deg, damping x10.
For each variant this script
  1. writes an edited COPY of the params to verify/out/controls/<tag>/<variant>/physics.json
     (derived fields that the params' own model_equation defines from the edited value, i.e. I_P,
     are recomputed and the edit list is recorded),
  2. runs the production solver via --solver-cmd   (default: "{py} solver/run.py --params {params} --out {outdir}"; run.py treats --out as a directory),
  3. optionally builds+bakes a .blend via --bake-cmd (placeholders {params} {traj} {blend}) and runs
     verify/check_blend.py on it against MY reference solution at the variant params,
  4. compares the production trajectory with formulas only:
       * every period (between successive upward zero crossings) vs
         T = 4 sqrt(I_P/mgd) K(m=sin^2(A/2)) / sqrt(1 - zeta^2),  A = cycle amplitude from the peaks in
         that cycle (mean of sin^2(|th_peak|/2)), zeta = c_v/(2 I_P wn)
       * every turning-point amplitude vs the first-order averaged nonlinear decay law
         dE/dt = -c_v<w^2> - k_q<|w|^3> - tau_c<|w|>, E = 2 mgd k^2 (exact elliptic averages)
     Tolerance 1e-4 relative (acceptance P9). Regime: light damping (zeta << 1), libration.

  .venv/bin/python verify/check_controls.py --params params/physics.json --tag prod \
      [--bake-cmd "blender -b --factory-startup --python scene/build.py -- --out {blend} --physics {params} --geometry params/geometry.json --bake {traj}"] \
      --out verify/out/check_controls.json
Self-test (stand-in solver, NOT a P9 result):
  .venv/bin/python verify/check_controls.py --params verify/test/synthetic/good/physics.json --tag selftest \
      --solver-cmd "{py} verify/test/fake_production_solver.py --params {params} --step 0.0026041666666666665 --out {traj}" \
      --out verify/out/negctl/controls_selftest.json
"""
import argparse, copy, datetime, json, math, os, subprocess, sys
import numpy as np
from scipy.special import ellipk, ellipe
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from vparams import load_model, sha256_file, find  # noqa: E402
from ref_solver import period_exact  # noqa: E402
from vlib import read_traj, up_crossings, extrema, jsonable  # noqa: E402
from check_physics import jerk_of  # noqa: E402

TOL = 1e-4
VARIANTS = ["L_x4", "amp_2deg", "amp_15deg", "damping_x10"]


def set_path(P, path, val):
    keys = [k for k in path.split("/") if k]
    o = P
    for k in keys[:-1]:
        o = o[k]
    if isinstance(o[keys[-1]], dict):
        o[keys[-1]]["value"] = val
    else:
        o[keys[-1]] = val


def make_variant(P, model, variant):
    Q = copy.deepcopy(P)
    prov = model.provenance
    edits = []
    def put(name, val):
        path = prov.get(name)
        if not path or not path.startswith("/"):
            raise SystemExit(f"cannot edit {name}: no params path (provenance {path!r})")
        set_path(Q, path, val); edits.append((path, val))
    if variant == "L_x4":
        L = model.L_c * 4
        put("L_c", L)
        if isinstance(prov.get("I_P"), str) and prov["I_P"].startswith("/"):
            r, _ = find(P, "r_bob")
            put("I_P", model.m_bob * (L * L + 0.4 * (r or 0.0) ** 2))
        if isinstance(prov.get("d"), str) and prov["d"].startswith("/"):
            put("d", L)
        for extra in ("L_eq", "wire_length"):
            if extra in Q and isinstance(Q[extra], dict):
                r, _ = find(P, "r_bob")
                Q[extra]["value"] = (model.m_bob * (L * L + 0.4 * r * r)) / (model.m_bob * L) if extra == "L_eq" else L - r
                edits.append((f"/{extra}", Q[extra]["value"]))
    elif variant == "amp_2deg":
        put("theta0", math.radians(2)); put("omega0", 0.0) if prov.get("omega0") else None
    elif variant == "amp_15deg":
        put("theta0", math.radians(15)); put("omega0", 0.0) if prov.get("omega0") else None
    elif variant == "damping_x10":
        for nm in ("c_v", "k_q", "tau_c"):
            if prov.get(nm):
                put(nm, getattr(model, nm) * 10)
    Q["_verifier_variant"] = dict(variant=variant, edits=edits, note="edited COPY made by verify/check_controls.py")
    return Q, edits


def averaged_rate(model, A):
    """d(k^2)/dt from first-order averaging at amplitude A."""
    k2 = math.sin(A / 2) ** 2
    wn2 = model.wn ** 2
    K, E = float(ellipk(k2)), float(ellipe(k2))
    T = 4 * K / model.wn
    dE = 0.0
    if model.c_v:
        dE -= model.c_v * 4 * wn2 * (E - (1 - k2) * K) / K  # <w^2> = 4 wn^2 [E(m)-(1-m)K(m)]/K(m)
    if model.k_q:
        dE -= model.k_q * 8 * wn2 * (math.sin(A) - A * math.cos(A)) / T
    if model.tau_c:
        dE -= model.tau_c * 4 * A / T
    return dE / (2 * model.mgd)


def analyse(model, d):
    t, th, om, al = d["t_s"], d["theta_rad"], d["omega_rad_s"], d["alpha_rad_s2"]
    ex = extrema(t, th, om, al, jerk_of(model, th, om, al))
    tc = up_crossings(t, th, om, al)
    zeta = model.c_v / (2 * model.I_P * model.wn)
    per = []
    for i in range(len(tc) - 1):
        pk = [v for (tt, v) in ex if tc[i] < tt < tc[i + 1]]
        if not pk:
            continue
        k2m = np.mean([math.sin(abs(v) / 2) ** 2 for v in pk])
        A = 2 * math.asin(math.sqrt(k2m))
        Tp = period_exact(model, A) / math.sqrt(1 - zeta ** 2)
        Tm = tc[i + 1] - tc[i]
        per.append(dict(t_start=float(tc[i]), T_meas=float(Tm), T_pred=Tp, amp=A, rel=float(Tm / Tp - 1)))
    # decay envelope
    t0, A0 = ex[0][0], abs(ex[0][1])
    tk = np.array([e[0] for e in ex]); Ak = np.array([abs(e[1]) for e in ex])
    if model.conservative:
        Apred = np.full_like(Ak, A0)
    else:
        sol = solve_ivp(lambda tt, y: [averaged_rate(model, 2 * math.asin(math.sqrt(max(y[0], 0))))],
                        (t0, tk[-1]), [math.sin(A0 / 2) ** 2], t_eval=tk, rtol=1e-12, atol=1e-16, method="DOP853")
        Apred = 2 * np.arcsin(np.sqrt(sol.y[0]))
    rel_env = Ak / Apred - 1
    r = dict(zeta=zeta, n_periods=len(per), periods=per,
             max_rel_period_err=float(max(abs(p["rel"]) for p in per)) if per else None,
             n_turning_points=len(ex), turning_times=tk.tolist(), amp_meas=Ak.tolist(), amp_pred=Apred.tolist(),
             max_rel_envelope_err=float(np.max(np.abs(rel_env))),
             amp_ratio_end_over_start=float(Ak[-1] / Ak[0]))
    r["status"] = "PASS" if (per and r["max_rel_period_err"] < TOL and r["max_rel_envelope_err"] < TOL) else "FAIL"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--model-json")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--solver-cmd", default="{py} solver/run.py --params {params} --out {outdir}")
    ap.add_argument("--bake-cmd")
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--include-base", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, P = load_model(a.params, a.model_json)
    base_dir = os.path.join(HERE, "out", "controls", a.tag)
    rep = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), tag=a.tag,
               base_params={a.params: sha256_file(a.params)}, solver_cmd_template=a.solver_cmd,
               bake_cmd_template=a.bake_cmd, tolerance=TOL, verifier_code_sha256=sha256_file(os.path.abspath(__file__)),
               variants={})
    vlist = (["base"] if a.include_base else []) + a.variants
    for v in vlist:
        vd = os.path.join(base_dir, v); os.makedirs(vd, exist_ok=True)
        pp = os.path.join(vd, "physics.json"); od = os.path.join(vd, "solver_out"); os.makedirs(od, exist_ok=True); tp = os.path.join(od, "trajectory.csv"); bp = os.path.join(vd, "pendulum.blend")
        if v == "base":
            Q, edits = copy.deepcopy(P), []
        else:
            Q, edits = make_variant(P, model, v)
        json.dump(Q, open(pp, "w"), indent=2)
        ent = dict(params=pp, params_sha256=sha256_file(pp), edits=edits, commands=[])
        cmd = a.solver_cmd.format(py=sys.executable, params=pp, traj=tp, outdir=od)
        pr = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)
        ent["commands"].append(dict(cmd=cmd, rc=pr.returncode, stderr_tail=pr.stderr[-1500:]))
        if pr.returncode != 0 or not os.path.exists(tp):
            ent["status"] = "FAIL"; ent["reason"] = "solver command failed"; rep["variants"][v] = ent; continue
        mv, _ = load_model(pp)
        ent["model"] = mv.as_dict()
        d, _, miss = read_traj(tp)
        ent["trajectory_sha256"] = sha256_file(tp)
        ent["analysis"] = analyse(mv, d)
        ent["predicted_small_angle_period_s"] = 2 * math.pi / mv.wn
        st = ent["analysis"]["status"]
        if a.bake_cmd:
            bc = a.bake_cmd.format(params=pp, traj=tp, blend=bp)
            pr = subprocess.run(bc, shell=True, cwd=ROOT, capture_output=True, text=True)
            ent["commands"].append(dict(cmd=bc, rc=pr.returncode, stderr_tail=pr.stderr[-1500:]))
            refp = os.path.join(vd, "ref.csv")
            subprocess.run([sys.executable, os.path.join(HERE, "ref_solver.py"), "--params", pp, "--out", refp], check=True, capture_output=True)
            cb = os.path.join(vd, "check_blend.json")
            pr2 = subprocess.run(["blender", "-b", bp, "--python", os.path.join(HERE, "check_blend.py"), "--",
                                  "--params", pp, "--ref", refp, "--trajectory", tp, "--out", cb, "--label", f"P9:{v}"],
                                 cwd=ROOT, capture_output=True, text=True)
            if os.path.exists(cb):
                r = json.load(open(cb)); ent["blend"] = r.get("summary"); ent["blend_P6_max_dtheta"] = r.get("P6_blend_vs_solver", {}).get("max_abs_dtheta_vs_verifier_ref_rad")
                st = "PASS" if (st == "PASS" and r["summary"]["P6"] == "PASS") else "FAIL"
            else:
                st = "FAIL"; ent["blend"] = "check_blend failed: " + pr2.stderr[-800:]
        ent["status"] = st
        rep["variants"][v] = ent
        an = ent["analysis"]
        print(f"{v:12s} {st}  max_rel_period={an['max_rel_period_err']:.3e}  max_rel_env={an['max_rel_envelope_err']:.3e}  T0={ent['predicted_small_angle_period_s']:.6f}s")
    judged = [rep["variants"][v]["status"] for v in a.variants if v in rep["variants"]]
    rep["status"] = "PASS" if judged and all(s == "PASS" for s in judged) else "FAIL"
    rep["through_bake"] = bool(a.bake_cmd)
    json.dump(jsonable(rep), open(a.out, "w"), indent=2)
    print("P9", rep["status"], "(solver only)" if not a.bake_cmd else "(solver + bake + check_blend)")


if __name__ == "__main__":
    main()
