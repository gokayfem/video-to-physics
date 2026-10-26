#!/usr/bin/env python
"""P9 control response through the REAL production solver (subprocess only). Verifier-owned; never imports solver/.

Variants (PLAN.md), each in a clean analytic regime; expected values from closed forms only:
  base         pure twist, psi0 = 0.1 rad on the vertical          -> omega_d, gamma from B3
  kappa_x4     kappa x4                                              -> twist period / 2 (exactly, undamped; B3 law with damping)
  I_x4         all inertia components x4                             -> twist period x 2
  L_plus50     wire length x1.5, 1e-5 rad planar swing, undamped     -> B4 swing frequency at the new L
  damping_x10  every damping coefficient x10                         -> envelope e^{-10 gamma t}, omega_d from B3
Tolerance 1e-4 relative (acceptance P9). Additionally each variant is run at the PRODUCTION initial conditions and compared
with my GL4 at the variant params (P2-type, 1e-6 m / rad).
Variant params are produced either by my own edits of a params COPY (tparams.write_variant at the provenance paths) or, if
given, by the dynamics make_variant interface:
  --variant-cmd "{py} solver/make_variant.py --params {params} --out {out} {set}"  with --variant-sets V.json
  (V.json maps variant name -> list of argument strings substituted for {set}; the regime ICs are then applied by my edits).

  .venv/bin/python verify/check_controls.py --params params/physics.json --solver-cmd "{py} solver/run.py --params {params} --out {outdir}" \
      [--model-json M.json] [--colmap C.json] [--skip-full] --out verify/out/check_controls_ck001.json
"""
import argparse, datetime, json, math, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from tmodel import axis_angle_quat, quat_to_R, EZ  # noqa: E402
from tio import read_traj, dump_json, fit_sinusoid  # noqa: E402
from tparams import load_model, write_variant, sha256_file  # noqa: E402
import check_physics as cp  # noqa: E402

TOL, TOL_POS = 1e-4, 1e-6
DAMP = ("c_t", "c_s", "c_v", "c_r", "c_rp")


def variant_edits(m, name):
    if name == "base":
        return {}
    if name == "kappa_x4":
        return dict(kappa=4 * m.kappa)
    if name == "I_x4":
        return dict(I_c=(4 * m.I_c).tolist())
    if name == "L_plus50":
        return dict(L=1.5 * m.L)
    if name == "damping_x10":
        return {k: 10 * getattr(m, k) for k in DAMP if k in m.provenance or getattr(m, k) != 0}
    raise ValueError(name)


def regime_edits(m, kind):
    e = dict(w0=[0.0, 0.0, 0.0])
    if m.mode == "string":
        e.update(u0=[0.0, 0.0, 1.0], udot0=[0.0, 0.0, 0.0])
    if kind == "twist":
        e["q0"] = axis_angle_quat(EZ, 0.1).tolist()
    else:
        ws, V = m.swing_modes("y")
        if m.mode == "string":
            ph1, ph2 = V[:, 0] / np.max(np.abs(V[:, 0])) * 1e-5
            e["u0"] = (quat_to_R(axis_angle_quat([0, 1, 0], ph1)) @ EZ).tolist(); e["q0"] = axis_angle_quat([0, 1, 0], ph2).tolist()
        else:
            e["q0"] = axis_angle_quat([0, 1, 0], 1e-5).tolist()
        e.update({k: 0.0 for k in DAMP if k in m.provenance})
    return e


def expected(m, kind):
    if kind == "twist":
        g = m.gamma_t
        return math.sqrt(m.omega_t ** 2 - g * g), g
    return float(m.swing_modes("y")[0][0]), 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True); ap.add_argument("--model-json"); ap.add_argument("--colmap")
    ap.add_argument("--solver-cmd", required=True); ap.add_argument("--variant-cmd"); ap.add_argument("--variant-sets")
    ap.add_argument("--variants", default="base,kappa_x4,I_x4,L_plus50,damping_x10"); ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--workdir", default=os.path.join(HERE, "out", "controls")); ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m0, P0 = load_model(a.params, a.model_json)
    sets = json.load(open(a.variant_sets)) if a.variant_sets else {}
    rep = dict(label=a.label, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               inputs={a.params: sha256_file(a.params)}, model=m0.as_dict(), tolerance_rel=TOL, tolerance_pos=TOL_POS,
               verifier_code_sha256=sha256_file(os.path.abspath(__file__)), variants={})
    os.makedirs(a.workdir, exist_ok=True)
    fits = {}
    for name in a.variants.split(","):
        ve = variant_edits(m0, name)
        P, base_params = P0, a.params
        if a.variant_cmd and name in sets:
            vp = os.path.join(a.workdir, f"{name}_made_variant.json")
            cmd = a.variant_cmd.format(py=os.path.join(ROOT, ".venv", "bin", "python"), params=a.params, out=vp, set=" ".join(sets[name]))
            r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)
            if r.returncode:
                rep["variants"][name] = dict(status="FAIL", reason=f"variant command failed: {cmd}\n{r.stderr[-800:]}"); continue
            mv, P = load_model(vp, a.model_json)
            base_params = vp
            ve_made = {k: getattr(mv, k) for k in ("kappa", "L", "M") if getattr(mv, k) != getattr(m0, k)}
            mvar = mv
        else:
            mvar = m0.replace(**ve)
        kind = "swing" if name == "L_plus50" else "twist"
        ent = dict(edits={k: (v if not hasattr(v, "tolist") else v.tolist()) for k, v in ve.items()}, regime=kind)
        try:
            re_ = regime_edits(mvar, kind)
            mreg = mvar.replace(**re_)
            d, info = cp.variant(P, m0 if P is P0 else load_model(base_params, a.model_json)[0], {**(ve if P is P0 else {}), **re_},
                                 f"{name}_regime", a.solver_cmd, a.workdir, a.colmap)
            w_exp, g_exp = expected(mreg, kind)
            sig = d["psi_rad"] if kind == "twist" else d["x_m"]
            wfit, gfit, _, _ = fit_sinusoid(d["t_s"], sig, w_exp, decay=(kind == "twist" and g_exp > 0), w_rel_window=0.05)
            ent.update(omega_expected=w_exp, omega_fit=wfit, rel_omega_err=abs(wfit / w_exp - 1), gamma_expected=g_exp, gamma_fit=gfit,
                       rel_gamma_err=(abs(gfit / g_exp - 1) if g_exp else None), run=info)
            ok = ent["rel_omega_err"] < TOL and (ent["rel_gamma_err"] is None or ent["rel_gamma_err"] < TOL)
            fits[name] = (wfit, gfit)
            if not a.skip_full:
                df, finfo = cp.variant(P, m0 if P is P0 else load_model(base_params, a.model_json)[0], ve if P is P0 else {},
                                       f"{name}_full", a.solver_cmd, a.workdir, a.colmap)
                p2, _ = cp.p2(mvar, df, cp.ref_for(mvar, float(df["t_s"][-1])))
                ent["full_ic_vs_verifier"] = {k: v for k, v in p2.items() if k != "criterion"}
                ent["full_run"] = finfo
                ok = ok and p2["status"] == "PASS"
            ent["status"] = "PASS" if ok else "FAIL"
        except (SystemExit, RuntimeError, KeyError, ValueError) as e:
            ent.update(status="NOT RUN", reason=str(e)[:800])
        rep["variants"][name] = ent
        print(f"{name:12s} {ent['status']}  " + ", ".join(f"{k}={ent[k]:.3e}" for k in ("omega_expected", "omega_fit", "rel_omega_err") if k in ent), flush=True)
    # ratios (informational, the judged numbers are the per-variant formula checks)
    if "base" in fits:
        wb = fits["base"][0]
        rep["ratios"] = {k: dict(period_ratio_var_over_base=wb / v[0]) for k, v in fits.items() if k != "base"}
    sts = [v["status"] for v in rep["variants"].values()]
    rep["status"] = "PASS" if sts and all(s == "PASS" for s in sts) else ("NOT RUN" if all(s == "NOT RUN" for s in sts) else "FAIL")
    rep["summary"] = dict(P9=rep["status"], variants={k: v["status"] for k, v in rep["variants"].items()})
    dump_json(rep, a.out)
    print("CHECK_CONTROLS_SUMMARY", json.dumps(rep["summary"]))


if __name__ == "__main__":
    main()
