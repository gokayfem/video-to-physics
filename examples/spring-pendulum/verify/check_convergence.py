#!/usr/bin/env python
"""P1 — convergence of the PRODUCTION integrator (verifier-owned; never imports solver/).

(a) files already produced by the production command interface (>= 3, any order; VALUE = step or tol):
    .venv/bin/python verify/check_convergence.py --order 4 --run 0.0026:a.csv --run 0.0013:b.csv --run 0.00065:c.csv --out X.json
(b) execute production through a template ({value}, {out} substituted), e.g. with verify/run_production.py:
    .venv/bin/python verify/check_convergence.py --order 4 --values 0.0052083 0.0026042 0.0013021 0.00065104 \
       --cmd-template ".venv/bin/python verify/run_production.py --params params/physics.json --set /integrator/h_s={value} \
                      --solver-cmd '{{py}} solver/run.py --params {{params}} --out {{outdir}}' --out {out}" --out X.json
Criterion (acceptance P1): max over samples/xyz of |E_{h/2} - E_{h/4}| < 1e-7 m (and attitude < 1e-7 rad if quaternions)
AND |p_obs - p_theory| <= 0.5 on the finest triple.  Finest diff < 1e-12 -> INCONCLUSIVE (order not observable).
--kind tol: p is the error-vs-tol exponent (declared by dynamics), disclosed as such.
"""
import argparse, datetime, json, math, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from vparams import sha256_file  # noqa: E402
from vio import read_traj, pos, quat, jsonable, dump_json  # noqa: E402
from vmodel import quat_to_R, rot_angle  # noqa: E402
ROUNDOFF = 1e-12


def analyse(runs, order):
    t0 = runs[0][1]["t_s"]
    for v, d in runs:
        if len(d["t_s"]) != len(t0) or np.max(np.abs(d["t_s"] - t0)) > 1e-12:
            raise ValueError("runs are not on a common output grid")
    res = {"values": [v for v, _ in runs]}
    errs = [float(np.max(np.abs(pos(runs[i][1]) - pos(runs[i + 1][1])))) for i in range(len(runs) - 1)]
    res["pos_successive_max_diffs_m"] = errs
    qs = [quat(d) for _, d in runs]
    if all(q is not None for q in qs):
        res["att_successive_max_diffs_rad"] = [float(np.max(rot_angle(quat_to_R(qs[i]), quat_to_R(qs[i + 1])))) for i in range(len(runs) - 1)]
    po = []
    for i in range(len(errs) - 1):
        r = runs[i][0] / runs[i + 1][0]
        po.append(math.log(errs[i] / errs[i + 1]) / math.log(r) if errs[i] > 0 and errs[i + 1] > 0 else None)
    res["observed_order"] = po
    e_last, p_last = errs[-1], po[-1]
    res.update(e_last_m=e_last, p_obs_last=p_last, p_theory=order,
               criterion="max|E_{h/2}-E_{h/4}| < 1e-7 m (att < 1e-7 rad) and |p_obs - p_theory| <= 0.5 (finest triple)")
    att_ok = res.get("att_successive_max_diffs_rad", [0])[-1] < 1e-7
    if e_last < ROUNDOFF:
        res["status"] = "INCONCLUSIVE"; res["reason"] = "finest difference at round-off; supply coarser resolutions"
    else:
        res["status"] = "PASS" if (e_last < 1e-7 and att_ok and p_last is not None and abs(p_last - order) <= 0.5) else "FAIL"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["step", "tol"], default="step"); ap.add_argument("--order", type=float, required=True)
    ap.add_argument("--run", action="append", default=[]); ap.add_argument("--values", type=float, nargs="*")
    ap.add_argument("--cmd-template"); ap.add_argument("--colmap"); ap.add_argument("--workdir", default=os.path.join(HERE, "out", "conv_runs"))
    ap.add_argument("--label", default=""); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files, cmds = [], []
    if a.cmd_template:
        os.makedirs(a.workdir, exist_ok=True)
        for v in a.values:
            out = os.path.join(os.path.abspath(a.workdir), f"run_{a.kind}_{v:.6e}.csv")
            cmd = a.cmd_template.format(value=repr(v), out=out)
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            cmds.append(dict(cmd=cmd, returncode=p.returncode, stderr_tail=p.stderr[-1500:]))
            if p.returncode:
                raise SystemExit(f"production command failed: {cmd}\n{p.stderr[-2000:]}")
            files.append((v, out))
    for r in a.run:
        v, f = r.split(":", 1); files.append((float(v), f))
    if len(files) < 3:
        raise SystemExit("need >= 3 resolutions")
    files.sort(key=lambda x: -x[0])
    runs = [(v, read_traj(f, a.colmap)[0]) for v, f in files]
    rep = dict(label=a.label, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), kind=a.kind,
               inputs=[dict(value=v, file=f, sha256=sha256_file(f)) for v, f in files], commands=cmds,
               verifier_code_sha256=sha256_file(os.path.abspath(__file__)))
    rep["P1"] = analyse(runs, a.order)
    dump_json(rep, a.out)
    print(json.dumps(jsonable({k: v for k, v in rep["P1"].items()}), indent=0))


if __name__ == "__main__":
    main()
