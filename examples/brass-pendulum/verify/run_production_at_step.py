#!/usr/bin/env python
"""Run the PRODUCTION solver (subprocess, no import) on a params COPY whose integrator.h_s is replaced.
  .venv/bin/python verify/run_production_at_step.py --params params/physics.json --h VALUE --out FILE.csv
Used by check_convergence.py --cmd-template. The production CLI (solver/run.py --params P --out DIR)
takes the step from params integrator.h_s, so the step is varied through a params copy."""
import argparse, json, os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser(); ap.add_argument("--params", required=True); ap.add_argument("--h", type=float, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--solver-root", default=ROOT, help="tree whose solver/run.py is executed (e.g. a COPY of a checkpoint); cwd for the run")
a = ap.parse_args()
P = json.load(open(a.params))
P["integrator"]["h_s"] = a.h
P["integrator"]["h_note"] = f"verifier P1 convergence copy, h={a.h!r}"
d = os.path.splitext(a.out)[0] + "_dir"; os.makedirs(d, exist_ok=True)
pp = os.path.join(d, "physics.json"); json.dump(P, open(pp, "w"), indent=2)
sr = os.path.abspath(a.solver_root)
r = subprocess.run([sys.executable, os.path.join(sr, "solver", "run.py"), "--params", os.path.abspath(pp), "--out", os.path.abspath(d)], cwd=sr, capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
sys.stderr.write(r.stderr[-2000:])
if r.returncode: sys.exit(r.returncode)
shutil.copyfile(os.path.join(d, "trajectory.csv"), a.out)
meta = json.load(open(os.path.join(d, "trajectory_meta.json")))
print(json.dumps(dict(h_requested=a.h, meta_h=meta.get("tolerances", {}).get("h_s"), substeps=meta.get("tolerances", {}).get("substeps_per_output"))))
