#!/usr/bin/env python
"""Make P3 benchmark params COPIES (edited at the params' own paths) and run the PRODUCTION solver on each
via a command template (subprocess only; never imports solver/). Writes <outdir>/bench_manifest.json for
check_physics.py --bench-manifest.

  .venv/bin/python verify/run_benchmarks.py --params params/physics.json --outdir verify/out/bench \
      --solver-cmd "{py} solver/run.py --params {params} --out {outdir}" [--traj-name trajectory.csv] [--colmap X.json]
Configurations (PLAN.md):  rest (B1), axial a=0.02*ell_eq (B2, params damping kept), conical 20 deg (B3, undamped,
point only), swing 1e-4 rad (B4, point), rigid_mode 0/1 at 1e-5 rad (B5, rigid). Edits needed: E0, vE0 (and q0,w0 for rigid),
damping terms (zeroed where the benchmark is undamped).  If the params schema cannot be edited at these paths
the run stops and asks for --model-json style variant params from dynamics.
"""
import argparse, json, math, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from vparams import load_model, write_variant, sha256_file  # noqa: E402
from vmodel import conical, rigid_planar_modes, axis_angle_quat  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True); ap.add_argument("--outdir", required=True)
ap.add_argument("--solver-cmd", required=True); ap.add_argument("--traj-name", default="trajectory.csv"); ap.add_argument("--colmap")
ap.add_argument("--cwd", default=ROOT)
a = ap.parse_args()
m, P = load_model(a.params)
os.makedirs(a.outdir, exist_ok=True)
A = m.A; l = m.ell_eq
zero_damp = {k: 0.0 for k in ("c_v", "c_a", "c_r") if m.provenance.get(k, "").startswith("/")}
runs = []
def add(name, kind, edits, **extra):
    d = os.path.join(os.path.abspath(a.outdir), name); os.makedirs(d, exist_ok=True)
    pp = os.path.join(d, "physics.json")
    write_variant(P, m, edits, pp, note=f"P3 benchmark {kind}")
    cmd = a.solver_cmd.format(py=sys.executable, params=pp, outdir=d, traj=os.path.join(d, a.traj_name))
    r = subprocess.run(cmd, shell=True, cwd=a.cwd, capture_output=True, text=True)
    ent = dict(name=name, kind=kind, params=pp, trajectory=os.path.join(d, a.traj_name), cmd=cmd, rc=r.returncode, stderr_tail=r.stderr[-800:], **extra)
    if a.colmap:
        ent["colmap"] = a.colmap
    runs.append(ent)
    print(name, "rc", r.returncode)
rigid = m.body == "rigid"
rest_ic = dict(E0=[A[0], A[1], A[2] - l], vE0=[0.0, 0.0, 0.0])
if rigid:
    rest_ic.update(q0=[1.0, 0.0, 0.0, 0.0], w0=[0.0, 0.0, 0.0])
add("B1_rest", "rest", dict(rest_ic))
ax = dict(rest_ic); ax["E0"] = [A[0], A[1], A[2] - l - 0.02 * l]
add("B2_axial", "axial", ax)
if not rigid:
    al = math.radians(20); cc = conical(m.replace(c_v=0, c_a=0, c_r=0), al)
    add("B3_conical", "conical", dict(E0=[A[0] + cc["rho"], A[1], cc["z"]], vE0=[0.0, cc["rho"] * cc["Omega"], 0.0], **zero_damp))
    th = 1e-4
    add("B4_swing", "swing", dict(E0=[A[0] + l * math.sin(th), A[1], A[2] - l * math.cos(th)], vE0=[0.0, 0.0, 0.0], **zero_damp))
else:
    ws, V = rigid_planar_modes(m, "y")
    for j in range(2):
        ph1, ph2 = V[:, j] / np.max(np.abs(V[:, j])) * 1e-5
        e = dict(E0=[A[0] + l * math.sin(ph1), A[1], A[2] - l * math.cos(ph1)], vE0=[0.0, 0.0, 0.0],
                 q0=axis_angle_quat([0, 1, 0], -ph2).tolist(), w0=[0.0, 0.0, 0.0], **zero_damp)
        add(f"B5_rigid_mode{j}", "rigid_mode", e, mode=j, axis="y")
json.dump(dict(base_params=a.params, base_sha256=sha256_file(a.params), solver_cmd=a.solver_cmd, runs=runs), open(os.path.join(a.outdir, "bench_manifest.json"), "w"), indent=2)
print("wrote", os.path.join(a.outdir, "bench_manifest.json"))
