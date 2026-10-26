"""Generate params/physics.json (schema: physics/FORMULATION.md section 7) from primary inputs, applying
the model-conditioned scale chain so every derived number is reproducible:

  T0   = T_meas / (2 K(sin(theta_amp/2)) / pi)         (finite-amplitude correction)
  L_eq = g (T0 / 2 pi)^2
  L_c  = L_eq / (1 + 0.4 / ratio^2)    (ratio = L_c / r from the image)   or
  L_c  = (L_eq + sqrt(L_eq^2 - 1.6 r^2)) / 2   (if --r given instead)
  r = L_c / ratio,  l = L_c - r,  m = 4/3 pi r^3 rho_bob,  I_P = m L_c^2 + 0.4 m r^2
  c = I_P omega0 / Q,  omega0 = sqrt(g / L_eq)          (Q assumed unless decay is fitted)

Usage (placeholder):
  .venv/bin/python solver/make_params.py --status placeholder --T-meas 1.25 --theta-amp-deg 6 \
      --ratio 13 --Q 1000 --theta-init-deg 6 --omega-init 0
Each primary input takes a provenance tag (--prov-T, --prov-amp, --prov-ratio, --prov-Q, --prov-init)
and an optional sigma/range; --source-* strings document where the number came from.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import sys
import json
import math
from pathlib import Path

from scipy.special import ellipk

ROOT = Path(__file__).resolve().parent.parent
G = 9.80665


def build(a) -> dict:
    amp = math.radians(a.theta_amp_deg) if a.theta_amp_deg is not None else a.theta_amp
    ratioK = 2.0 * float(ellipk(math.sin(amp / 2) ** 2)) / math.pi
    if a.L_eq is not None:       # fitted L_eq given directly (projected fit); T_meas becomes the model period
        L_eq = a.L_eq
        T0 = 2 * math.pi * math.sqrt(L_eq / G)
        a.T_meas = T0 * ratioK
    else:
        T0 = a.T_meas / ratioK
        L_eq = G * (T0 / (2 * math.pi)) ** 2
    if a.r is not None:
        r = a.r
        L_c = (L_eq + math.sqrt(L_eq ** 2 - 1.6 * r * r)) / 2
        ratio = L_c / r
        prov_r, prov_ratio = a.prov_r, "derived"
    else:
        ratio = a.ratio
        L_c = L_eq / (1 + 0.4 / ratio ** 2)
        r = L_c / ratio
        prov_r, prov_ratio = "model-conditioned", a.prov_ratio
    m = 4 / 3 * math.pi * r ** 3 * a.rho_bob
    I_P = m * L_c ** 2 + 0.4 * m * r * r
    L_eq_chk = I_P / (m * L_c)
    om0 = math.sqrt(G / L_eq_chk)
    c = I_P * om0 / a.Q if a.Q not in (None, math.inf) else 0.0
    th_init = math.radians(a.theta_init_deg) if a.theta_init_deg is not None else a.theta_init
    sig = lambda x: x if x is not None else None
    p = {
        "schema": "physics/FORMULATION.md s7 (physics_json_schema_proposal v1)",
        "owner": "dynamics",
        "status": a.status,
        "status_note": a.status_note,
        "generated_by": "solver/make_params.py",
        "model_id": "planar_compound_sphere_rigid_massless_rod_v1",
        "model_equation": "I_P th'' = -m g L_c sin th - c th';  I_P = m L_c^2 + 0.4 m r^2;  bob centre = P + L_c (sin th, 0, -cos th)",
        "g": {"value": G, "unit": "m/s^2", "provenance": "assumed", "source": "CONTRACT v1 (standard gravity)",
              "range": [9.78, 9.832]},
        "T_meas": {"value": a.T_meas, "unit": "s", "provenance": a.prov_T, "sigma": sig(a.sigma_T),
                   "source": a.source_T},
        "theta_amp": {"value": amp, "unit": "rad", "provenance": a.prov_amp, "sigma": sig(a.sigma_amp),
                      "source": a.source_amp, "value_deg": math.degrees(amp)},
        "T0": {"value": T0, "unit": "s", "provenance": "model-conditioned",
               "derivation": "T_meas / (2 K(sin(theta_amp/2)) / pi)", "T_over_T0": ratioK},
        "L_eq": {"value": L_eq_chk, "unit": "m", "provenance": "model-conditioned",
                 "sigma": a.sigma_L_eq if a.sigma_L_eq is not None else ((2 * a.sigma_T / a.T_meas * L_eq) if a.sigma_T else None),
                 "derivation": (a.L_eq_source or "fitted directly") if a.L_eq is not None else
                               "g (T0 / 2 pi)^2; stored value = I_P/(m L_c) recomputed (identical to round-off)"},
        "ratio_Lc_over_r": {"value": ratio, "unit": "-", "provenance": prov_ratio, "sigma": sig(a.sigma_ratio),
                            "source": a.source_ratio},
        "L_c": {"value": L_c, "unit": "m", "provenance": "model-conditioned",
                "derivation": "L_eq / (1 + 0.4 / ratio^2)" if a.r is None else "(L_eq + sqrt(L_eq^2 - 1.6 r^2)) / 2"},
        "L_c_in_L_eff_units": ({"value": a.k_Leff, "unit": "camera.json length unit (L_eff)", "provenance": "fitted",
                                "sigma": a.sigma_k_Leff, "source": a.source_k_Leff,
                                "definition": "|P_ring - bob centre| measured through the camera of record (CHANGELOG 2026-10-25T22:47:40Z)"}
                               if a.k_Leff is not None else None),
        "L_eff_m": ({"value": L_c / a.k_Leff, "unit": "m per camera.json length unit", "provenance": "model-conditioned",
                     "sigma": (L_c / a.k_Leff) * math.sqrt(((a.sigma_L_eq or 0) / L_eq) ** 2 + ((a.sigma_k_Leff or 0) / a.k_Leff) ** 2),
                     "derivation": "L_c_m / L_c_in_L_eff_units; scene/build.py scales camera translations by THIS, never by L_c"}
                    if a.k_Leff is not None else None),
        "level_tilt_rad": ({"value": a.level_tilt, "unit": "rad", "provenance": "fitted", "sigma": a.sigma_level_tilt,
                            "value_deg": math.degrees(a.level_tilt), "source": a.source_level_tilt,
                            "convention": ("The STATIC scene (table, stand, pivot hardware, backdrop, ALL cameras; i.e. everything in the "
                                           "camera.json/geometry.json stand frame) is rotated by +level_tilt_rad about the world +Y axis "
                                           "through P (right-hand rule; Blender: parent empty at P, rotation_euler = (0, +level_tilt_rad, 0)). "
                                           "Gravity is exactly g along -Z_world. PND_Swing is NOT tilted: theta (theta_init, trajectory) is "
                                           "relative to TRUE vertical. A positive value tilts the stand's up-axis toward +X_world, so the bob's "
                                           "rest direction appears at +level_tilt_rad (image right) in the stand/camera frame."),
                            "decision": "CHANGELOG 2026-10-25T23:14:43Z (not-quite-level table)"}
                           if a.level_tilt is not None else None),
        "bob_radius": {"value": r, "unit": "m", "provenance": prov_r,
                       "derivation": "L_c / ratio" if a.r is None else a.source_r},
        "wire_length": {"value": L_c - r, "unit": "m", "provenance": "model-conditioned", "note": "L_c - r"},
        "rho_bob": {"value": a.rho_bob, "unit": "kg/m^3", "provenance": "assumed", "range": [8400, 8730],
                    "source": "brass; mass is unidentifiable from kinematics (FORMULATION s6)"},
        "bob_mass": {"value": m, "unit": "kg", "provenance": "assumed", "derivation": "4/3 pi r^3 rho_bob"},
        "I_P": {"value": I_P, "unit": "kg m^2", "provenance": "model-conditioned", "derivation": "m L_c^2 + 0.4 m r^2"},
        "wire": {"diameter": {"value": 3e-4, "unit": "m", "provenance": "assumed", "range": [2e-4, 5e-4]},
                 "E": {"value": 2.0e11, "unit": "Pa", "provenance": "assumed", "range": [1.9e11, 2.1e11]},
                 "rho": {"value": 7850, "unit": "kg/m^3", "provenance": "assumed"},
                 "mass_included": False, "inextensible": True},
        "damping": {"law": "linear_viscous_pivot",
                    "Q": {"value": a.Q, "unit": "-", "provenance": a.prov_Q, "range": a.Q_range,
                          "source": a.source_Q},
                    "c": {"value": c, "unit": "N m s/rad", "provenance": "derived", "derivation": "I_P omega0 / Q"},
                    "gamma": {"value": c / (2 * I_P), "unit": "1/s", "derivation": "c / (2 I_P)"},
                    "k_q": {"value": 0.0, "unit": "N m s^2/rad^2", "provenance": "assumed",
                            "note": "quadratic air drag available in solver, OFF (sub-pixel per FORMULATION s5)"},
                    "tau_c": {"value": 0.0, "unit": "N m", "provenance": "assumed",
                              "note": "smoothed Coulomb available in solver, OFF (rejected in FORMULATION s5)"},
                    "coulomb_omega_s": {"value": 1e-3, "unit": "rad/s", "provenance": "assumed"},
                    "note": "never negative; replace Q provenance by 'fitted' only if decay is significant"},
        "theta_init": {"value": th_init, "unit": "rad", "provenance": a.prov_init, "sigma": sig(a.sigma_init),
                       "value_deg": math.degrees(th_init), "source": a.source_init},
        "omega_init": {"value": a.omega_init, "unit": "rad/s", "provenance": a.prov_init, "source": a.source_init},
        "air": {"rho": 1.204, "mu": 1.813e-5, "T_K": 293.15, "p_Pa": 101325, "provenance": "assumed",
                "used_for": "plausibility only (Q bound); not in production EOM"},
        "systematics_on_L_mapping": {"buoyancy_added_mass_rel_T": 1.1e-4, "wire_mass_rel_Leq": 3e-4,
                                     "flexible_wire_rel_T": 5e-6, "g_local_rel": 3e-3},
        "integrator": {"solver_id": "rk4-fixed-step-v1",
                       "method": "classical RK4, fixed step dividing each output interval exactly; steps split at turning points when non-smooth terms are on",
                       "h_s": 1.0 / 3072.0, "h_note": "1/3072 s = 1/32 of the 1/96 s output spacing",
                       "t_end_s": 10.0, "output_dt_s": 1.0 / 96.0},
    }
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "params" / "physics.json"))
    ap.add_argument("--status", required=True, choices=["placeholder", "fitted", "test"])
    ap.add_argument("--status-note", default="")
    ap.add_argument("--T-meas", type=float, default=None)
    ap.add_argument("--L-eq", type=float, default=None, help="fitted L_eq [m]; overrides the T_meas chain")
    ap.add_argument("--sigma-L-eq", type=float, default=None)
    ap.add_argument("--L-eq-source", default="")
    ap.add_argument("--k-Leff", type=float, default=None, help="L_c in camera.json length units (fitted)")
    ap.add_argument("--sigma-k-Leff", type=float, default=None)
    ap.add_argument("--source-k-Leff", default="")
    ap.add_argument("--level-tilt", type=float, default=None); ap.add_argument("--sigma-level-tilt", type=float, default=None)
    ap.add_argument("--source-level-tilt", default="")
    ap.add_argument("--extra-json", default=None, help="JSON file merged in as 'fit_provenance'"); ap.add_argument("--sigma-T", type=float)
    ap.add_argument("--prov-T", default="measured"); ap.add_argument("--source-T", default="")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--theta-amp-deg", type=float); g.add_argument("--theta-amp", type=float)
    ap.add_argument("--sigma-amp", type=float); ap.add_argument("--prov-amp", default="fitted")
    ap.add_argument("--source-amp", default="")
    g2 = ap.add_mutually_exclusive_group(required=True)
    g2.add_argument("--ratio", type=float); g2.add_argument("--r", type=float)
    ap.add_argument("--sigma-ratio", type=float); ap.add_argument("--prov-ratio", default="measured")
    ap.add_argument("--source-ratio", default=""); ap.add_argument("--prov-r", default="assumed")
    ap.add_argument("--source-r", default="")
    ap.add_argument("--rho-bob", type=float, default=8500.0)
    ap.add_argument("--Q", type=float, default=1000.0); ap.add_argument("--prov-Q", default="assumed")
    ap.add_argument("--Q-range", type=float, nargs=2, default=[300, 6000])
    ap.add_argument("--source-Q", default="FORMULATION s5: nominal assumption, no significant decay measured")
    g3 = ap.add_mutually_exclusive_group(required=True)
    g3.add_argument("--theta-init-deg", type=float); g3.add_argument("--theta-init", type=float)
    ap.add_argument("--omega-init", type=float, default=0.0)
    ap.add_argument("--sigma-init", type=float); ap.add_argument("--prov-init", default="fitted")
    ap.add_argument("--source-init", default="")
    a = ap.parse_args(argv)
    if a.T_meas is None and a.L_eq is None:
        ap.error("need --T-meas or --L-eq")
    p = build(a)
    if a.extra_json:
        p["fit_provenance"] = json.loads(Path(a.extra_json).read_text())
    # delivery F1: exact argv + input hashes inside the file; timestamp only in a sidecar (not byte-reproducible)
    ins = {"solver/make_params.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if a.extra_json:
        ins[str(Path(a.extra_json).resolve().relative_to(ROOT))] = hashlib.sha256(Path(a.extra_json).read_bytes()).hexdigest()
    p["generator"] = {"script": "solver/make_params.py", "argv": list(argv if argv is not None else sys.argv[1:]),
                      "inputs_sha256": ins,
                      "note": "byte-reproducible: rerun with this argv from ROOT; generation time is in solver/out/make_params_run.json"}
    Path(a.out).write_text(json.dumps(p, indent=2) + "\n")
    side = ROOT / "solver" / "out" / "make_params_run.json"
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps({"generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "out": str(a.out), "out_sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest(),
                                "argv": p["generator"]["argv"]}, indent=2) + "\n")
    print(json.dumps({k: p[k]["value"] for k in ("T0", "L_eq", "L_c", "bob_radius", "bob_mass", "I_P")}, indent=1),
          "c =", p["damping"]["c"]["value"])


if __name__ == "__main__":
    main()
