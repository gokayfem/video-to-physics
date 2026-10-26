"""Write solver/out/trajectory.csv and trajectory_meta.json from params/physics.json (clip 02).

Usage:  ROOT/.venv/bin/python solver/run.py [--params params/physics.json] [--out solver/out]

Columns (SI, world frame = gravity frame, A at the origin; t = k/96 s exactly, frame n at row 4n):
  t_s
  X_x_m X_y_m X_z_m            weight centre of mass (= cylinder centre)
  V_x_m_s V_y_m_s V_z_m_s      its velocity
  n_x n_y n_z                  body axis (unit, hinge -> CoM)
  ndot_x ndot_y ndot_z         dn/dt [1/s]
  H_x_m H_y_m H_z_m            hinge = eye inner top (spring lower end; spring upper end = pivot P, fixed)
  Hdot_x_m_s Hdot_y_m_s Hdot_z_m_s
  q_w q_x q_y q_z              weight orientation (local -Z -> n, zero spin, geometric twist included)
  eul_x_rad eul_y_rad eul_z_rad   Blender XYZ Euler of q (unwrapped)
  euld_x_rad_s euld_y_rad_s euld_z_rad_s   their time derivatives
  spring_len_m spring_rate_m_s  |P H| and d|P H|/dt  (P = swing pivot, params pivot_world_m)
  spring_tension_N             k (|P_eff H| - L0), force the spring exerts along its axis (>0 = tension; never < 0, solver raises)
  phi_x_rad phi_y_rad          spring angle from vertical in XZ / YZ (atan2(H_x,-H_z), atan2(H_y,-H_z))
  psi_x_rad psi_y_rad          body-axis angle from vertical, same convention
  KE_J PE_J E_J                PE zero at static equilibrium
  P_diss_W W_diss_J            dissipated power and cumulative dissipated work
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
import spring_pendulum as sp  # noqa: E402

COLUMNS = ["t_s", "X_x_m", "X_y_m", "X_z_m", "V_x_m_s", "V_y_m_s", "V_z_m_s", "n_x", "n_y", "n_z",
           "ndot_x", "ndot_y", "ndot_z", "H_x_m", "H_y_m", "H_z_m", "Hdot_x_m_s", "Hdot_y_m_s", "Hdot_z_m_s",
           "q_w", "q_x", "q_y", "q_z", "eul_x_rad", "eul_y_rad", "eul_z_rad", "euld_x_rad_s", "euld_y_rad_s", "euld_z_rad_s",
           "spring_len_m", "spring_rate_m_s", "spring_tension_N", "phi_x_rad", "phi_y_rad", "psi_x_rad", "psi_y_rad",
           "KE_J", "PE_J", "E_J", "P_diss_W", "W_diss_J"]
SOURCE_FILES = ["spring_pendulum.py", "run.py"]


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def solver_source_sha256() -> str:
    h = hashlib.sha256()
    for f in SOURCE_FILES:
        h.update(f.encode()); h.update(b"\0"); h.update((HERE / f).read_bytes()); h.update(b"\0")
    return h.hexdigest()


def time_grid(t_end=10.0, per_second=96):
    n = int(round(t_end * per_second))
    return np.arange(n + 1, dtype=float) / per_second


def table(d):
    n = d["n"]; H = d["H"] - d.get("pivot", 0.0)   # relative to the pivot: used ONLY for the angles; H columns are world
    cols = [d["t"], *d["X"].T, *d["V"].T, *n.T, *d["u"].T, *d["H"].T, *d["Hdot"].T, *d["q"].T, *d["euler"].T, *d["euler_rate"].T,
            d["L"], d["sr"], d["tension"], np.arctan2(H[:, 0], -H[:, 2]), np.arctan2(H[:, 1], -H[:, 2]),
            np.arctan2(n[:, 0], -n[:, 2]), np.arctan2(n[:, 1], -n[:, 2]), d["KE"], d["PE"], d["E"], d["P_diss"], d["W"]]
    return np.column_stack(cols)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=str(ROOT / "params" / "physics.json"))
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--h", type=float, default=None, help="override the fixed RK4 step [s] (convergence runs); must divide 1/96 s")
    a = ap.parse_args(argv)
    params = json.loads(Path(a.params).read_text())
    M = sp.model_from_physics_json(params)
    integ = params.get("integrator", {})
    h = float(a.h) if a.h else float(integ.get("h_s", sp.DEFAULT_H)); t_end = float(integ.get("t_end_s", 10.0))
    t = time_grid(t_end, 96)
    d = sp.simulate(M, t, h)
    piv = np.array((params.get("pivot_world_m") or {}).get("value", [0.0, 0.0, 0.0]), float)
    d["X"] = d["X"] + piv; d["H"] = d["H"] + piv; d["pivot"] = piv          # world frame (A = origin); solver state is relative to the pivot P
    data = table(d)
    assert data.shape[1] == len(COLUMNS)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    csv = out / "trajectory.csv"
    with open(csv, "w", newline="\n") as f:
        f.write(",".join(COLUMNS) + "\n")
        for row in data:
            f.write(",".join(repr(float(v)) for v in row) + "\n")
    E0 = float(d["E"][0])
    meta = {
        "solver_id": sp.SOLVER_ID, "model_id": sp.MODEL_ID, "body": M.body,
        "params_file": os.path.relpath(a.params, ROOT), "params_sha256": sha256_file(a.params),
        "params_status": params.get("status"),
        "solver_source_files": [f"solver/{f}" for f in SOURCE_FILES], "solver_source_sha256": solver_source_sha256(),
        "trajectory_sha256": sha256_file(csv), "columns": COLUMNS,
        "method": "classical RK4, fixed step; output samples are integration nodes; orientation quaternion integrated as an auxiliary state",
        "tolerances": {"h_s": h, "substeps_per_output": int(round((1 / 96) / h)),
                       "expected_global_error_m": "< 1e-9 (see solver/out/selfcheck.json)"},
        "sampling": {"dt_s": 1 / 96, "t_start_s": 0.0, "t_end_s": t_end, "n_samples": len(t),
                     "note": "t = k/96 exactly; reference frame n at row 4n; no retiming"},
        "rig": {"A_world_m": [0.0, 0.0, 0.0], "pivot_world_m": piv.tolist(),
                "pivot_note": "swing pivot P = spring top coil on the fixed S-hook (lead decision, CHANGELOG); spring axis = P -> H", "hinge": "H = lower hook (VSP_HookLower location, VSP_Weight origin)",
                "weight_axis": "local -Z of VSP_Weight = n (hinge -> CoM)", "b_m": M.b,
                "geometry_m": params.get("geometry_m"), "level_tilt": params.get("level_tilt_rad"),
                "camera_scale": params.get("camera_scale")},
        "derived_model": M.as_dict(),
        "initial_state_hinge": {"H0_m": (np.array(M.X0) - M.b * np.array(M.n0) + piv).tolist(),
                                "Hdot0_m_s": (np.array(M.V0) - M.b * np.array(M.u0)).tolist(),
                                "n0": list(M.n0), "ndot0_1_s": list(M.u0), "X0_com_m": (np.array(M.X0) + piv).tolist(), "V0_com_m_s": list(M.V0),
                                "omega0_world_rad_s": np.cross(M.n0, M.u0).tolist(), "spin_omega3": 0.0,
                                "note": "H = X - b n (eye/hinge); world = gravity frame, A at the origin; solver/physics.json initial_state is RELATIVE to the pivot P = pivot_world_m"},
        "summary": {"spring_len_min_m": float(d["L"].min()), "spring_len_max_m": float(d["L"].max()),
                    "tension_min_N": float(d["tension"].min()), "tension_max_N": float(d["tension"].max()),
                    "phi_x_max_abs_deg": float(np.degrees(np.max(np.abs(data[:, COLUMNS.index("phi_x_rad")])))),
                    "psi_minus_phi_max_abs_deg": float(np.degrees(np.max(np.abs(data[:, COLUMNS.index("psi_x_rad")] - data[:, COLUMNS.index("phi_x_rad")])))),
                    "E0_J": E0, "E_end_J": float(d["E"][-1]), "W_diss_end_J": float(d["W"][-1]),
                    "energy_balance_max_abs_J": float(np.max(np.abs(d["E"] - E0 + d["W"]))),
                    "norm_n_err": float(d["norm_n_err"]), "n_vs_quaternion_err": float(d["n_q_consistency"]),
                    "integrator_stats": {k: (v if not isinstance(v, list) else [float(x) for x in v]) for k, v in d["integrator_stats"].items()},
                    "guards": "v2: projection of (n, n', q) every step; raise on non-finite, | |n|-1 | > 1e-12, step drift > 1e-8, tension < 0"},
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out / "trajectory_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("params_sha256", "solver_source_sha256", "trajectory_sha256")}, indent=1))
    print(json.dumps(meta["summary"], indent=1))


if __name__ == "__main__":
    main()
