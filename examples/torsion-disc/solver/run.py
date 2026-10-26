"""Write solver/out/trajectory.csv and trajectory_meta.json from params/physics.json (clip 03, torsion disc).

Usage:  ROOT/.venv/bin/python solver/run.py [--params params/physics.json] [--out solver/out] [--h 3.2552083e-4]

Columns (SI, world frame = gravity frame, clamp A at the origin; t = k/96 s exactly, reference frame n at row 4n):
  t_s
  X_x_m X_y_m X_z_m              disc centre of mass (= disc centre, half a thickness below the top face)
  V_x_m_s V_y_m_s V_z_m_s        its velocity
  B_x_m B_y_m B_z_m              wire attachment = centre of the disc TOP face (TOR_Disc origin; wire lower end)
  Bdot_x_m_s Bdot_y_m_s Bdot_z_m_s
  q_w q_x q_y q_z                disc orientation, body -> world (body z = disc up-normal, body x = zero-twist reference)
  w_x_rad_s w_y_rad_s w_z_rad_s  WORLD angular velocity
  eul_x_rad eul_y_rad eul_z_rad  Blender XYZ Euler of q (unwrapped)
  euld_x_rad_s euld_y_rad_s euld_z_rad_s   their time derivatives (analytic from w)
  psi_rad psidot_rad_s           wire twist (swing-twist decomposition about the disc normal) and its rate
  mark_azimuth_rad               world azimuth (atan2(y, x) of the horizontal projection) of the engraved-mark direction
                                 (body azimuth mu from params mark.mu_body_rad)
  s_x s_y s_z                    wire direction A -> B (unit)
  phi_x_rad phi_y_rad            wire angle from vertical in XZ / YZ (atan2(s_x, -s_z), atan2(s_y, -s_z))
  theta_x_rad theta_y_rad        disc down-normal angle from vertical, same convention
  tilt_rad bend_B_rad            disc tilt from level; angle between wire and disc normal at B (bending boundary layer)
  wire_tension_N                 tension (>= 0; solver raises if slack), acting on the disc at B along -s
  wire_torque_z_N_m              torsional restoring torque -kappa psi (about the disc normal)
  KE_J PE_J E_J                  PE zero at static equilibrium (gravity + twist + bending)
  PE_twist_J PE_bend_J
  P_diss_W W_diss_J              dissipated power and cumulative dissipated work (integrated as a state)
  qdot_w_1_s .. qdot_z_1_s       dq/dt (for quaternion Hermite keys)
Positions (X, B) are WORLD = pivot_world_m (params; the suspension point, = A unless a pivot offset was fitted) + solver state.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import torsion_disc as td  # noqa: E402

COLUMNS = ["t_s", "X_x_m", "X_y_m", "X_z_m", "V_x_m_s", "V_y_m_s", "V_z_m_s", "B_x_m", "B_y_m", "B_z_m",
           "Bdot_x_m_s", "Bdot_y_m_s", "Bdot_z_m_s", "q_w", "q_x", "q_y", "q_z", "w_x_rad_s", "w_y_rad_s", "w_z_rad_s",
           "eul_x_rad", "eul_y_rad", "eul_z_rad", "euld_x_rad_s", "euld_y_rad_s", "euld_z_rad_s",
           "psi_rad", "psidot_rad_s", "mark_azimuth_rad", "s_x", "s_y", "s_z", "phi_x_rad", "phi_y_rad",
           "theta_x_rad", "theta_y_rad", "tilt_rad", "bend_B_rad", "wire_tension_N", "wire_torque_z_N_m",
           "KE_J", "PE_J", "E_J", "PE_twist_J", "PE_bend_J", "P_diss_W", "W_diss_J",
           "qdot_w_1_s", "qdot_x_1_s", "qdot_y_1_s", "qdot_z_1_s"]
SOURCE_FILES = ["torsion_disc.py", "run.py"]


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


def mark_azimuth(d, mu):
    """world azimuth of the mark direction cos(mu) e_x + sin(mu) e_y (body), horizontal projection, unwrapped."""
    Rm = td.rot_batch(d["q"].T)
    v = np.cos(mu) * Rm[:, :, 0] + np.sin(mu) * Rm[:, :, 1]
    return np.unwrap(np.arctan2(v[:, 1], v[:, 0]))


def table(d, mu):
    cols = [d["t"], *d["X"].T, *d["V"].T, *d["B"].T, *d["Bdot"].T, *d["q"].T, *d["w"].T, *d["euler"].T, *d["euler_rate"].T,
            d["psi"], d["psidot"], mark_azimuth(d, mu), *d["s"].T, d["phi_x"], d["phi_y"], d["theta_x"], d["theta_y"],
            d["tilt"], d["bend_B"], d["tension"], d["torque_wire_z"], d["KE"], d["PE"], d["E"], d["PE_twist"], d["PE_bend"],
            d["P_diss"], d["W"], *qdot(d["q"], d["w"]).T]
    return np.column_stack(cols)


def qdot(q, w):
    """dq/dt = 1/2 (0, w_world) (x) q   (N,4)."""
    qw, qx, qy, qz = q.T; wx, wy, wz = w.T
    return 0.5 * np.column_stack([-wx * qx - wy * qy - wz * qz, wx * qw + wy * qz - wz * qy,
                                  -wx * qz + wy * qw + wz * qx, wx * qy - wy * qx + wz * qw])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=str(ROOT / "params" / "physics.json"))
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--h", type=float, default=None, help="override the fixed RK4 step [s] (convergence runs); 1/96 s / h integer")
    a = ap.parse_args(argv)
    params = json.loads(Path(a.params).read_text())
    M = td.model_from_physics_json(params)
    integ = params.get("integrator", {})
    h = float(a.h) if a.h else float(integ.get("h_s", td.DEFAULT_H)); t_end = float(integ.get("t_end_s", 10.0))
    t = time_grid(t_end, 96)
    d = td.simulate(M, t, h)
    mu = float(((params.get("mark") or {}).get("mu_body_rad") or {}).get("value", 0.0))
    piv = np.array((params.get("pivot_world_m") or {}).get("value", [0.0, 0.0, 0.0]), float)
    d["X"] = d["X"] + piv; d["B"] = d["B"] + piv            # world frame (A = origin); solver state is relative to the pivot
    data = table(d, mu)
    assert data.shape[1] == len(COLUMNS)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    csv = out / "trajectory.csv"
    with open(csv, "w", newline="\n") as f:
        f.write(",".join(COLUMNS) + "\n")
        for row in data:
            f.write(",".join(repr(float(v)) for v in row) + "\n")
    E0 = float(d["E"][0])
    st = d["integrator_stats"]
    meta = {
        "solver_id": td.SOLVER_ID, "model_id": td.MODEL_ID,
        "params_file": os.path.relpath(a.params, ROOT), "params_sha256": sha256_file(a.params),
        "params_status": params.get("status"),
        "solver_source_files": [f"solver/{f}" for f in SOURCE_FILES], "solver_source_sha256": solver_source_sha256(),
        "trajectory_sha256": sha256_file(csv), "columns": COLUMNS,
        "method": ("classical RK4, fixed step; output samples are integration nodes; index-1 constrained Newton-Euler "
                   "(wire tension multiplier) with exact projection of |q| = 1, |B| = ell, B.dB/dt = 0 after every step"),
        "tolerances": {"h_s": h, "substeps_per_output": int(round((1 / 96) / h)),
                       "expected_global_error": "< 1e-9 m / rad (see solver/out/selfcheck.json P1)",
                       "guards": "raise on non-finite state, post-projection constraint error > 1e-12, per-step drift > 1e-8, tension < 0, |psi| > 1.9 pi"},
        "sampling": {"dt_s": 1 / 96, "t_start_s": 0.0, "t_end_s": t_end, "n_samples": len(t),
                     "note": "t = k/96 exactly; reference frame n at row 4n; no retiming"},
        "rig": {"A_world_m": [0.0, 0.0, 0.0], "pivot_world_m": piv.tolist(),
                "pivot_note": "suspension point = wire top end (TOR_Anchor should sit here); = A + fitted horizontal offset (params pivot_world_m)",
                "TOR_Disc": "origin = B (top-face centre, wire attachment); local +Z = disc up-normal; local +X = zero-twist reference; "
                            "location keyed = B(t), rotation_quaternion keyed from q (columns q_*, qdot_*) 4 keys/frame; mark drawn at local azimuth mu_body_rad",
                "TOR_Wire": "straight segment pivot -> B (scene GN reads TOR_Anchor and TOR_Disc origins)",
                "mark_mu_body_rad": mu, "R_m": M.R, "t_m": M.t, "ell_m": M.ell, "h_c_m": M.h_c,
                "geometry_m": params.get("geometry_m"), "camera_scale": params.get("camera_scale")},
        "derived_model": M.as_dict(),
        "initial_state_world": {"X0_com_m": list(M.X0), "V0_com_m_s": list(M.V0), "q0_wxyz": list(M.q0),
                                "w0_world_rad_s": list(M.w0),
                                "B0_m": (np.array(M.X0) + td.quat_to_matrix(M.q0) @ np.array([0, 0, M.h_c]) + piv).tolist(),
                                "note": "X0/V0/q0/w0 are the SOLVER state relative to the pivot; world = pivot_world_m + X",
                                "psi0_rad": float(d["psi"][0]), "psidot0_rad_s": float(d["psidot"][0])},
        "summary": {"psi_min_deg": float(np.degrees(d["psi"].min())), "psi_max_deg": float(np.degrees(d["psi"].max())),
                    "swing_max_deg": float(np.degrees(d["swing"].max())), "tilt_max_deg": float(np.degrees(d["tilt"].max())),
                    "B_horizontal_max_mm": float(1e3 * np.max(np.hypot(d["B"][:, 0], d["B"][:, 1]))),
                    "tension_min_N": float(d["tension"].min()), "tension_max_N": float(d["tension"].max()),
                    "E0_J": E0, "E_end_J": float(d["E"][-1]), "W_diss_end_J": float(d["W"][-1]),
                    "energy_balance_max_abs_J": float(np.max(np.abs(d["E"] - E0 + d["W"]))),
                    "norm_q_err": d["norm_q_err"], "wire_length_err_m": d["len_err"],
                    "integrator_stats": {k: (v if not isinstance(v, list) else [float(x) for x in v]) for k, v in st.items()}},
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out / "trajectory_meta.json").write_text(json.dumps(meta, indent=2, default=float) + "\n")
    print(json.dumps({k: meta[k] for k in ("params_sha256", "solver_source_sha256", "trajectory_sha256")}, indent=1))
    print(json.dumps(meta["summary"], indent=1, default=float))


if __name__ == "__main__":
    main()
