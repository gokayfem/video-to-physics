#!/usr/bin/env python
"""P1 — convergence of the PRODUCTION integrator (verifier-owned; never imports solver/).

Two ways to feed it (>= 3 resolutions, coarse -> fine, all written on the same 1/96 s output grid):

 (a) files already produced by the production command interface:
     .venv/bin/python verify/check_convergence.py --kind step --order 8 \
        --run 0.0104166667:solver/out/conv_h1.csv --run 0.0052083333:solver/out/conv_h2.csv \
        --run 0.0026041667:solver/out/conv_h3.csv \
        --cmd "<exact command that produced each file>" ... --out verify/out/check_convergence.json

 (b) let this script execute the production solver through a command template
     (the template is the dynamics author's documented interface; {value} and {out} are substituted):
     .venv/bin/python verify/check_convergence.py --kind step --order 8 --values 0.01 0.005 0.0025 \
        --cmd-template ".venv/bin/python solver/run.py --step {value} --out {out}" --out verify/out/check_convergence.json

--kind tol : adaptive runs at tolerance levels; the observed exponent log(e1/e2)/log(tol1/tol2) is
compared with --order (the exponent the dynamics author declares for error vs tol), disclosed as such.
Tolerance (acceptance P1): max|theta_{h/2} - theta_{h/4}| < 1e-7 rad AND |p_obs - p_theory| <= 0.5.
If e2 is at round-off level (< 1e-12 rad) the order is not observable -> INCONCLUSIVE (never PASS).
"""
import argparse, datetime, json, math, os, shlex, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vparams import sha256_file  # noqa: E402
from vlib import read_traj, jsonable  # noqa: E402

ROUNDOFF = 1e-12


def analyse(runs, kind, order):
    """runs: list of (value, dict-of-columns), coarse -> fine."""
    t0 = runs[0][1]["t_s"]
    for v, d in runs:
        if len(d["t_s"]) != len(t0) or np.max(np.abs(d["t_s"] - t0)) > 1e-12:
            raise ValueError("runs are not on a common output grid")
    res = {"values": [v for v, _ in runs]}
    for col in ("theta_rad", "omega_rad_s", "E_J"):
        if col not in runs[0][1]:
            continue
        errs = [float(np.max(np.abs(runs[i][1][col] - runs[i + 1][1][col]))) for i in range(len(runs) - 1)]
        pobs = []
        for i in range(len(errs) - 1):
            r = runs[i][0] / runs[i + 1][0]  # refinement ratio between the pair giving errs[i] and errs[i+1]
            pobs.append(math.log(errs[i] / errs[i + 1]) / math.log(r) if errs[i + 1] > 0 and errs[i] > 0 else None)
        res[col] = dict(successive_max_diffs=errs, observed_order=[p for p in pobs if p is not None])
    th = res["theta_rad"]
    e_last = th["successive_max_diffs"][-1]
    p_last = th["observed_order"][-1] if th["observed_order"] else None
    res["criterion"] = "max|theta_{h/2}-theta_{h/4}| < 1e-7 rad and |p_obs - p_theory| <= 0.5 (finest triple)"
    res["e_last_rad"] = e_last
    res["p_obs_last"] = p_last
    res["p_theory"] = order
    res["kind"] = kind
    if e_last < ROUNDOFF:
        res["status"] = "INCONCLUSIVE"
        res["reason"] = "finest difference at round-off; order not observable -> supply coarser resolutions"
    else:
        res["status"] = "PASS" if (e_last < 1e-7 and p_last is not None and abs(p_last - order) <= 0.5) else "FAIL"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["step", "tol"], default="step")
    ap.add_argument("--order", type=float, required=True, help="theoretical order declared by dynamics")
    ap.add_argument("--run", action="append", default=[], help="VALUE:FILE (coarse->fine)")
    ap.add_argument("--cmd", action="append", default=[], help="command that produced each --run (recorded)")
    ap.add_argument("--values", type=float, nargs="*")
    ap.add_argument("--cmd-template")
    ap.add_argument("--workdir", default=os.path.join(HERE, "out", "conv_runs"))
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs, files, cmds = [], [], list(a.cmd)
    if a.cmd_template:
        os.makedirs(a.workdir, exist_ok=True)
        for v in a.values:
            out = os.path.join(a.workdir, f"run_{a.kind}_{v:.6e}.csv")
            cmd = a.cmd_template.format(value=repr(v), out=out)
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            cmds.append(dict(cmd=cmd, returncode=p.returncode, stderr_tail=p.stderr[-2000:]))
            if p.returncode != 0:
                raise SystemExit(f"production command failed: {cmd}\n{p.stderr[-2000:]}")
            files.append((v, out))
    for r in a.run:
        v, f = r.split(":", 1)
        files.append((float(v), f))
    if len(files) < 3:
        raise SystemExit("need >= 3 resolutions")
    files.sort(key=lambda x: -x[0])  # coarse -> fine
    for v, f in files:
        d, _, missing = read_traj(f, required=["t_s", "theta_rad"])
        runs.append((v, d))
    rep = dict(label=a.label, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               inputs=[dict(value=v, file=f, sha256=sha256_file(f)) for v, f in files], commands=cmds,
               verifier_code_sha256=sha256_file(os.path.abspath(__file__)))
    rep["P1"] = analyse(runs, a.kind, a.order)
    with open(a.out, "w") as fo:
        json.dump(jsonable(rep), fo, indent=2)
    print(json.dumps(jsonable(rep["P1"]), indent=1))


if __name__ == "__main__":
    main()
