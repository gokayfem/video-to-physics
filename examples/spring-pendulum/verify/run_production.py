#!/usr/bin/env python
"""Run the PRODUCTION solver (subprocess only, never imported) on a params COPY with JSON-path edits.
  .venv/bin/python verify/run_production.py --params params/physics.json --set /integrator/h_s=0.001 \
      --solver-cmd "{py} solver/run.py --params {params} --out {outdir}" [--traj-name trajectory.csv] [--cwd DIR] --out FILE.csv
Values in --set are parsed as JSON (numbers, lists). The edited copy and the solver output dir sit next to --out."""
import argparse, json, os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from vparams import set_path  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--params", required=True); ap.add_argument("--set", action="append", default=[])
ap.add_argument("--solver-cmd", required=True); ap.add_argument("--traj-name", default="trajectory.csv"); ap.add_argument("--cwd", default=ROOT)
ap.add_argument("--out", required=True)
a = ap.parse_args()
P = json.load(open(a.params))
edits = []
for s in a.set:
    path, val = s.split("=", 1); v = json.loads(val); set_path(P, path, v); edits.append((path, v))
P["_verifier_variant"] = dict(edits=edits, note="verify/run_production.py copy")
d = os.path.splitext(os.path.abspath(a.out))[0] + "_dir"; os.makedirs(d, exist_ok=True)
pp = os.path.join(d, "physics.json"); json.dump(P, open(pp, "w"), indent=2)
cmd = a.solver_cmd.format(py=sys.executable, params=pp, outdir=d, traj=os.path.join(d, a.traj_name))
r = subprocess.run(cmd, shell=True, cwd=a.cwd, capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
sys.stderr.write(r.stderr[-2000:])
if r.returncode:
    sys.exit(r.returncode)
shutil.copyfile(os.path.join(d, a.traj_name), a.out)
print(json.dumps(dict(cmd=cmd, edits=edits)))
