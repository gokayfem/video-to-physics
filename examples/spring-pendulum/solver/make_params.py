"""Generate params/physics.json (model elastic_pendulum_hinged_body_v1, physics/FORMULATION.md s7).

The parameter -> model mapping is solver/model_build.py (the SAME code the fit uses), so the production model is identical
to the fitted one.  Scale chain (model-conditioned):
  ell (equilibrium |AH|) from the swing frequency with g assumed; r, h, b, spring d/D from measured image ratios x ell;
  m = rho pi r^2 h (assumed density -> mass unidentifiable); I_perp = m (3r^2+h^2)/12; m_s = rho_s pi d^2/4 pi D N;
  k = (m + m_s/3) omega_axial^2 with omega_axial = rax sqrt(g/ell); L0 = ell - (m + m_s/2) g / k;
  c_x = 2 m gamma_sw, c_s = max(0, 2(m + m_s/3) gamma_ax - c_x), c_r = 2 I_perp gamma_rock.

Fitted:      make_params.py --status fitted --fit-json solver/out/fit/Ta_X/fit_projected.json [provenance args]
Placeholder: make_params.py --status placeholder --P '{"ell":0.3,"Lcam":1,"bcam":0.2268,"rax":2.0,...}'
Exact argv + input hashes are stored in the file (generator); the timestamp goes to solver/out/make_params_run.json.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import spring_pendulum as sp  # noqa: E402
import model_build as mb  # noqa: E402


def build(a):
    fit = None
    if a.fit_json:
        fit = json.loads(Path(a.fit_json).read_text())
        P = {k: v for k, v in fit["params"].items() if k != "Q_fixed"}
        geom = fit["inputs"]["geometry"]
        sdb = {k: v["sd"] for k, v in (fit.get("bootstrap") or {}).items() if isinstance(v, dict) and "sd" in v}
        sds = fit["sd_scaled"]
        sd = lambda k: max(sdb.get(k, 0.0), sds.get(k, 0.0)) if (k in sdb or k in sds) else None
    else:
        P = json.loads(a.P); geom = json.loads(a.geom) if a.geom else {}
        sd = lambda k: None
    for k in ("beta", "phiy0", "psiy0", "phidy0", "psidy0", "pz", "px"):
        P.setdefault(k, 0.0)
    M, info = mb.build_model(P, geom)
    der = M.derived()
    g = info["geom"]
    V = lambda v, unit, prov, **kw: dict({"value": v, "unit": unit, "provenance": prov}, **kw)
    fitted = a.status == "fitted"
    prov_fit = "fitted" if fitted else "assumed"
    p = {
        "schema": "physics/FORMULATION.md s7 (clip 02 v1)",
        "owner": "physics+dynamics (role B)",
        "status": a.status, "status_note": a.status_note,
        "generated_by": "solver/make_params.py (mapping: solver/model_build.py)",
        "model_id": sp.MODEL_ID, "body": "rigid",
        "model_equation": ("T = 1/2 m|X'|^2 + 1/2 I_perp|n'|^2 + 1/2 (m_s/3)|H'|^2;  V = 1/2 k(|H|-L0)^2 + m g X_z + m_s g H_z/2;  "
                           "H = X - b n (lower hook = hinge), A = origin; Rayleigh D = 1/2 c_s (s_hat.H')^2 + 1/2 c_x|X'|^2 + 1/2 c_r|n'|^2"),
        "primary": {k: V(P.get(k), u, prov_fit if k not in ("gamma_sw", "gamma_ax") else a.prov_damping, sigma=sd(k))
                    for k, u in (("ell", "m"), ("Lcam", "camera units"), ("bcam", "camera units"), ("rax", "-"), ("phi0", "rad"),
                                 ("psi0", "rad"), ("de0", "m"), ("phid0", "rad/s"), ("psid0", "rad/s"), ("ded0", "m/s"),
                                 ("gamma_sw", "1/s"), ("gamma_ax", "1/s"), ("beta", "rad"), ("phiy0", "rad"), ("psiy0", "rad"),
                                 ("phidy0", "rad/s"), ("psidy0", "rad/s"), ("pz", "camera units"), ("px", "camera units"))},
        "parameters": {
            "g": V(sp.G_STD, "m/s^2", "assumed", source="CONTRACT v1 standard gravity", range=[9.78, 9.832]),
            "ell_eq": V(info["ell"], "m", "model-conditioned", sigma=sd("ell"),
                        definition="equilibrium |A H| = L0 + (m + m_s/2) g / k; set by the swing frequency with g assumed"),
            "omega_axial": V(info["omega_axial"], "rad/s", "model-conditioned" if fitted else "assumed",
                             definition="sqrt(k/(m + m_s/3)) = rax sqrt(g/ell)"),
            "k": V(info["k"], "N/m", "model-conditioned", derivation="(m + m_s/3) omega_axial^2 (scales with the assumed mass)"),
            "L0": V(info["L0"], "m", "model-conditioned", derivation="ell - (m + m_s/2) g / k (effective free length incl. hooks and coil preload)"),
            "m": V(info["m"], "kg", "assumed", derivation=f"rho pi r^2 h, rho = {g['rho']} (steel assumed); unidentifiable from kinematics"),
            "m_s": V(info["m_s"], "kg", "assumed", derivation=f"rho_s pi d^2/4 * pi D N, d = {info['spring_d']:.5g} m, D = {info['spring_D']:.5g} m, N = {g['spring_turns']}"),
            "b": V(info["b"], "m", "measured (camera units) / model-conditioned scale" if g["b_mode"] == "measured" else "fitted (diagnostic)",
                   derivation=("hinge H -> CoM = b_cam (measured, role C) / camera_units_per_m" if g["b_mode"] == "measured"
                               else "bcam / (Lcam/ell)") + "; CoM = cylinder centre (eye-ring mass neglected)"),
            "I_perp": V(info["I_perp"], "kg m^2", "model-conditioned", derivation="m (3 r^2 + h^2) / 12"),
            "damping": {
                "law": "Rayleigh: axial dashpot c_s on the spring rate, linear drag c_x on the CoM, rotational drag c_r on n'",
                "gamma_swing": V(P["gamma_sw"], "1/s", a.prov_damping, source=a.source_damping,
                                 Q_swing=(der["omega_lateral_rad_s"][0] / (2 * P["gamma_sw"]) if P["gamma_sw"] > 0 else None)),
                "gamma_axial": V(P["gamma_ax"], "1/s", a.prov_damping, source=a.source_damping),
                "gamma_rock": V(g["gamma_rock"], "1/s", "assumed", source="ring-on-hook rubbing; nominal, not observable"),
                "c_x": V(info["c_x"], "N s/m", "derived", derivation="2 m gamma_swing"),
                "c_s": V(info["c_s"], "N s/m", "derived", derivation="max(0, 2 (m + m_s/3) gamma_axial - c_x)", clipped=info["c_s_clipped"]),
                "c_r": V(info["c_r"], "N m s", "derived", derivation="2 I_perp gamma_rock"),
                "note": "never negative; no drives"},
        },
        "geometry_m": {"provenance": g.get("provenance"), "b_mode": g["b_mode"], "ell_eq": info["ell"],
                       "ell_eq_definition": "equilibrium |P - H|, P = swing pivot (spring top coil P_eff), H = hinge (eye inner top)", "cylinder_radius": info["r"], "cylinder_height": info["h"], "H_to_cylinder_centre": info["b"],
                       "spring_wire_d": info["spring_d"], "spring_coil_D": info["spring_D"], "spring_turns": g["spring_turns"],
                       "camera_units": {k: g[k] for k in ("unit", "r_cam", "h_cam", "b_cam", "e_cam", "spring_d_cam", "spring_D_cam")},
                       "H_to_eye_centre": info["e"], "pivot_world_m": info["pivot_world"], "camera_units_per_m": info["lam"],
                       "S_m": 1.0 / info["lam"], "k_mode": g["k_mode"], "ratios_source": g.get("source"),
                       "note": "metric lengths are model-conditioned (scale from the swing frequency with assumed g)"},
        "S_m": {"value": 1.0 / info["lam"], "unit": "m per W_in (= S, clear span between the posts)", "provenance": "model-conditioned",
                "derivation": "Lcam / ell inverse: swing frequency + assumed g; scene/build.py gauge"},
        "pivot_world_m": {"value": info["pivot_world"], "unit": "m", "provenance": "measured (camera units, P_eff) x model-conditioned scale",
                          "definition": "swing pivot P (spring top coil on the fixed S-hook), world frame with A at the origin; "
                                        "trajectory positions (X, H) are world = P + solver state"},
        "camera_scale": {"camera_units_per_m": info["lam"], "provenance": prov_fit,
                         "definition": "camera.json length units per metre = Lcam / ell; scene metres = camera units / this"},
        "level_tilt_rad": V(P["beta"], "rad", "fitted" if (fitted and fit and fit.get("fit_tilt")) else "assumed (0)", sigma=sd("beta"),
                            convention=("static scene (frame, crossbar, S-hook, table, backdrop, cameras) rotated by +beta about world +Y through the pivot P "
                                        "(Blender: parent empty at A, rotation_euler = (0, +beta, 0)); gravity exactly -Z_world; the solver "
                                        "state and the baked VSP_Weight/VSP_HookLower transforms are in the gravity frame")),
        "initial_state": {"X0": V(list(M.X0), "m", prov_fit), "V0": V(list(M.V0), "m/s", prov_fit),
                          "n0": V(list(M.n0), "-", prov_fit), "u0": V(list(M.u0), "1/s", prov_fit),
                          "spin_omega3": V(0.0, "rad/s", "assumed", note="released without twist; conserved for the axisymmetric hinged body")},
        "derived": der,
        "integrator": {"solver_id": sp.SOLVER_ID, "method": "classical RK4, fixed step dividing each output interval exactly",
                       "h_s": 1.0 / 3072.0, "t_end_s": 10.0, "output_dt_s": 1.0 / 96.0},
    }
    if fit:
        p["fit_provenance"] = {"fit_json": str(Path(a.fit_json).resolve().relative_to(ROOT)), "production_fit": fit["production_fit"],
                               "t_anchor_s": fit["t_anchor_s"], "metrics": fit["metrics"], "flags": fit["flags"],
                               "model_selection": {k: fit["model_selection"][k] for k in ("MQ", "M1", "damping_evidence")},
                               "inputs": fit["inputs"], "reduced_chi2": fit["reduced_chi2"]}
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "params" / "physics.json"))
    ap.add_argument("--status", required=True, choices=["placeholder", "fitted", "test"])
    ap.add_argument("--status-note", default="")
    ap.add_argument("--fit-json", default=None)
    ap.add_argument("--P", default=None, help="placeholder primary parameters (JSON)")
    ap.add_argument("--geom", default=None, help="placeholder geometry override (JSON)")
    ap.add_argument("--prov-damping", default="assumed"); ap.add_argument("--source-damping", default="")
    a = ap.parse_args(argv)
    if not a.fit_json and not a.P:
        ap.error("need --fit-json or --P")
    p = build(a)
    ins = {f"solver/{f}": hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in ("make_params.py", "model_build.py", "spring_pendulum.py")}
    if a.fit_json:
        ins[str(Path(a.fit_json).resolve().relative_to(ROOT))] = hashlib.sha256(Path(a.fit_json).read_bytes()).hexdigest()
    p["generator"] = {"script": "solver/make_params.py", "argv": list(argv if argv is not None else sys.argv[1:]),
                      "inputs_sha256": ins, "note": "byte-reproducible: rerun with this argv from ROOT"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(p, indent=2) + "\n")
    side = HERE / "out" / "make_params_run.json"; side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps({"generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "out": str(a.out), "out_sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest(),
                                "argv": p["generator"]["argv"]}, indent=2) + "\n")
    print(json.dumps({k: p["parameters"][k]["value"] for k in ("ell_eq", "k", "L0", "m", "m_s", "b", "I_perp")}, indent=1))
    print(json.dumps({k: p["derived"][k] for k in ("omega_axial_rad_s", "omega_lateral_rad_s", "ratio_axial_over_swing", "static_stretch_m")}))


if __name__ == "__main__":
    main()
