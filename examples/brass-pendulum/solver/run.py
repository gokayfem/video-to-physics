"""Write solver/out/trajectory.csv and trajectory_meta.json from params/physics.json.

Usage:  ROOT/.venv/bin/python solver/run.py [--params params/physics.json] [--out solver/out]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import pendulum as pd  # noqa: E402

COLUMNS = "t_s,theta_rad,omega_rad_s,alpha_rad_s2,x_m,z_m,KE_J,PE_J,E_J,tension_N,P_diss_W"
SOURCE_FILES = ["pendulum.py", "run.py"]


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def solver_source_sha256() -> str:
    h = hashlib.sha256()
    for f in SOURCE_FILES:
        h.update(f.encode()); h.update(b"\0"); h.update((HERE / f).read_bytes()); h.update(b"\0")
    return h.hexdigest()


def time_grid(t_end=10.0, per_second=96):
    n = int(round(t_end * per_second))
    return np.arange(n + 1, dtype=float) / per_second   # exact k/96


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=str(ROOT / "params" / "physics.json"))
    ap.add_argument("--out", default=str(HERE / "out"))
    a = ap.parse_args(argv)
    params = json.loads(Path(a.params).read_text())
    integ = params.get("integrator", {})
    h = float(integ.get("h_s", pd.DEFAULT_H))
    t_end = float(integ.get("t_end_s", 10.0))
    t = time_grid(t_end, 96)
    r = pd.simulate(params, t, h=h)
    M = r["model"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([t, r["theta"], r["omega"], r["alpha"], r["x"], r["z"], r["KE"], r["PE"],
                            r["E"], r["tension"], r["P_diss"]])
    csv = out / "trajectory.csv"
    with open(csv, "w", newline="\n") as f:
        f.write(COLUMNS + "\n")
        for row in data:
            f.write(",".join(repr(float(v)) for v in row) + "\n")   # repr = shortest round-trip
    E0 = r["E"][0]
    meta = {
        "solver_id": pd.SOLVER_ID,
        "params_file": os.path.relpath(a.params, ROOT),
        "params_sha256": sha256_file(a.params),
        "params_status": params.get("status"),
        "solver_source_files": [f"solver/{f}" for f in SOURCE_FILES],
        "solver_source_sha256": solver_source_sha256(),
        "trajectory_sha256": sha256_file(csv),
        "method": "classical RK4, fixed step; output samples are integration nodes",
        "tolerances": {"h_s": h, "substeps_per_output": int(round((1 / 96) / h)),
                       "expected_global_error_rad": "< 1e-10 (see solver/out/selfcheck.json)"},
        "sampling": {"dt_s": 1 / 96, "t_start_s": 0.0, "t_end_s": t_end, "n_samples": len(t),
                     "note": "t = k/96 exactly; frame n of the reference at row 4n"},
        "derived_model": M.as_dict(),
        "summary": {"theta_max_abs_rad": float(np.max(np.abs(r["theta"]))),
                    "tension_min_N": float(r["tension"].min()), "tension_max_N": float(r["tension"].max()),
                    "E0_J": float(E0), "E_end_J": float(r["E"][-1]),
                    "W_diss_end_J": float(r["W_diss"][-1]),
                    "energy_balance_max_rel": float(np.max(np.abs(r["E"] - E0 + r["W_diss"])) / E0)},
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out / "trajectory_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("params_sha256", "solver_source_sha256", "trajectory_sha256")}, indent=1))
    print(json.dumps(meta["summary"], indent=1))


if __name__ == "__main__":
    main()
