"""Generate params/physics.json (model disc_on_wire_rigid6_v1, physics/FORMULATION.md s7).

The parameter -> model mapping is solver/model_build.py (the SAME code the fit uses), so the production model is identical
to the fitted one.  Scale/material chain (model-conditioned, FORMULATION s6):
  lam (camera units per metre) from the swing frequency (scale_mode swing) or from the declared disc radius (prompt);
  R, t = measured camera-unit ratios / lam;  m = rho pi R^2 t (brass assumed);  I_z = m R^2/2;  kappa = I_z om_t^2;
  wire d = (32 ell kappa / (pi G))^(1/4);  k_A = k_B = sqrt(E (pi d^4/64) m g);  c_* from declared/fitted Q.

Fitted:      make_params.py --status fitted --fit-json solver/out/fit/<run>/fit_projected.json [--status-note ...]
Placeholder: make_params.py --status placeholder --P '{"Lcam":1,"om_t":2.1,"psi0":0.5,...}' [--geom '{...}']
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
import torsion_disc as td  # noqa: E402
import model_build as mb  # noqa: E402

PRIMARY_UNITS = (("Ax", "camera units"), ("Ay", "camera units"), ("ell", "m"), ("Lcam", "camera units"), ("om_t", "rad/s"), ("psi0", "rad"), ("psid0", "rad/s"), ("mu", "rad"),
                 ("phx0", "rad"), ("phy0", "rad"), ("phxd0", "rad/s"), ("phyd0", "rad/s"),
                 ("thx0", "rad"), ("thy0", "rad"), ("thxd0", "rad/s"), ("thyd0", "rad/s"), ("Q_t", "-"), ("Q_sw", "-"), ("Q_tilt", "-"), ("gamma_t", "1/s"))


def build(a):
    fit = None
    if a.fit_json:
        fit = json.loads(Path(a.fit_json).read_text())
        P = dict(fit["params"]); geom = fit["inputs"]["geometry"]
        sdb = {k: v["sd"] for k, v in (fit.get("bootstrap") or {}).items() if isinstance(v, dict) and "sd" in v}
        sds = fit["sd_scaled"]
        sd = lambda k: max(sdb.get(k, 0.0), sds.get(k, 0.0)) if (k in sdb or k in sds) else None
        free = set(fit["free"])
    else:
        P = json.loads(a.P); geom = json.loads(a.geom) if a.geom else {}
        sd = lambda k: None; free = set()
    M, info = mb.build_model(P, geom)
    g = info["g"]
    der = M.derived()
    V = lambda v, unit, prov, **kw: dict({"value": v, "unit": unit, "provenance": prov}, **kw)
    fitted = a.status == "fitted"

    def prov(k):
        if k in free:
            return "fitted"
        if k in ("Q_t", "Q_sw", "Q_tilt"):
            return a.prov_damping if k == "Q_t" else "assumed"
        if k == "ell" and g["scale_mode"] == "prompt":
            return "model-conditioned (declared disc radius R_prompt)"
        return "assumed" if not fitted else "fixed (not fitted; see fit flags)"
    psi_max = max(abs(P.get("psi0", 0.0)), 1e-9)
    p = {
        "schema": "physics/FORMULATION.md s7 (clip 03 v1)",
        "owner": "physics+dynamics (role B)",
        "status": a.status, "status_note": a.status_note,
        "generated_by": "solver/make_params.py (mapping: solver/model_build.py)",
        "model_id": td.MODEL_ID,
        "model_equation": ("rigid disc (m, R, t) hung at its top-face centre B from clamp A by an inextensible wire |AB| = ell; "
                           "V = m g X_z + 1/2 kappa psi^2 + k_A (1 + s_z) + k_B (1 - s.n); "
                           "Rayleigh D = 1/2 c_t psidot^2 + 1/2 c_x |V|^2 + 1/2 c_r |w_perp|^2 (FORMULATION s3)"),
        "primary": {k: V(P.get(k), u, prov(k), sigma=sd(k)) for k, u in PRIMARY_UNITS if k in P or k in ("psid0",)},
        "parameters": {
            "g": V(td.G_STD, "m/s^2", "assumed", source="CONTRACT v1 standard gravity", range=[9.78, 9.832]),
            "ell": V(info["ell"], "m", "model-conditioned", sigma=sd("ell"),
                     definition="wire length |A B|, clamp exit -> disc top-face centre"),
            "R": V(info["R"], "m", "measured ratio (camera units) / model-conditioned scale", derivation="R_cam / lam"),
            "t": V(info["t"], "m", "measured ratio (camera units) / model-conditioned scale", derivation="t_cam / lam"),
            "m": V(info["m"], "kg", "assumed", derivation=f"rho pi R^2 t, rho = {g['rho']} kg/m^3 (brass assumed); unidentifiable from kinematics"),
            "I_z": V(info["I_z"], "kg m^2", "model-conditioned", derivation="m R^2 / 2"),
            "I_perp": V(info["I_perp"], "kg m^2", "model-conditioned", derivation="m (3 R^2 + t^2) / 12"),
            "omega_torsion": V(P["om_t"], "rad/s", "fitted" if "om_t" in free else "assumed", sigma=sd("om_t"),
                               definition="sqrt(kappa / I_z); identified by the twist period (scale- and mass-free)"),
            "kappa": V(info["kappa"], "N m/rad", "model-conditioned", derivation="I_z omega_torsion^2 (scales with the assumed mass)"),
            "k_A": V(info["k_A"], "N m", "model-conditioned", derivation=f"sqrt(E I_w m g), bend_ends = {g['bend_ends']}"),
            "k_B": V(info["k_B"], "N m", "model-conditioned", derivation=f"sqrt(E I_w m g), bend_ends = {g['bend_ends']}"),
            "damping": {
                "law": "Rayleigh: torsional dashpot c_t on psidot, linear drag c_x on the CoM velocity, tilt drag c_r on w_perp",
                "Q_torsion": V(P.get("Q_t"), "-", prov("Q_t"), source=a.source_damping, sigma=sd("Q_t")),
                "Q_swing": V(P.get("Q_sw"), "-", "assumed", source="air drag; nominal (FORMULATION s4)"),
                "Q_tilt": V(P.get("Q_tilt"), "-", "assumed", source="face-normal air drag on the disc; nominal (FORMULATION s5); rigid joint: referenced to the swing mode"),
                "c_t": V(info["c_t"], "N m s", "derived", derivation="I_z omega_torsion / Q_torsion"),
                "c_x": V(info["c_x"], "N s/m", "derived", derivation="m omega_lateral[0] / Q_swing"),
                "c_r": V(info["c_r"], "N m s", "derived", derivation="I_perp omega_ref / Q_tilt, omega_ref = omega_lateral[0] for the rigid joint (else [1])"),
                "note": "never negative; no drives"},
        },
        "wire": {"d_m": V(info["d_wire"], "m", "model-conditioned", derivation=f"(32 ell kappa / (pi G))^(1/4), G = {g['G_shear']:.4g} Pa (steel assumed)"),
                 "EI_N_m2": V(info["EI"], "N m^2", "model-conditioned", derivation=f"E pi d^4 / 64, E = {g['E_young']:.4g} Pa"),
                 "boundary_layer_m": V(info["lambda_bl"], "m", "model-conditioned", derivation="sqrt(EI / (m g))"),
                 "max_shear_strain": V(info["shear_strain_per_rad"] * psi_max, "-", "derived",
                                       derivation="(d/2) |psi0| / ell (linear-elastic torsion valid: << 2e-3)")},
        "geometry_m": {"provenance": g.get("provenance"), "scale_mode": g["scale_mode"], "ell": info["ell"], "disc_radius": info["R"],
                       "disc_thickness": info["t"], "wire_diameter": info["d_wire"], "camera_units_per_m": info["lam"],
                       "camera_units": {k: g[k] for k in ("unit", "R_cam", "t_cam")} | {"Lcam": P.get("Lcam")},
                       "ratios_source": g.get("source"),
                       "note": "metric lengths are model-conditioned (scale from the swing frequency with assumed g, or the declared disc radius)"},
        "camera_scale": {"camera_units_per_m": info["lam"], "provenance": ("fitted" if "ell" in free else "model-conditioned"),
                         "definition": "camera.json length units per metre = Lcam / ell; scene metres = camera units / this"},
        "pivot_world_m": V([P.get("Ax", 0.0) / info["lam"], P.get("Ay", 0.0) / info["lam"], 0.0], "m",
                           "fitted" if "Ax" in free else "assumed (= A)", sigma=[sd("Ax"), sd("Ay")],
                           camera_units=[P.get("Ax", 0.0), P.get("Ay", 0.0), 0.0],
                           definition="suspension point (wire top end) in world, A = origin; solver state is relative to it; "
                                      "fitted horizontal offset = correction of A's assumed placement (camera role: A placed by assumption "
                                      "vertically above the frame-0 disc centre; reported A miss 0.037/0.010 D)"),
        "mark": {"mu_body_rad": V(P.get("mu", 0.0), "rad", "fitted" if "mu" in free else "assumed", sigma=sd("mu"),
                                  definition="azimuth of the engraved mark (one end of the diameter line) in TOR_Disc local XY, from local +X toward +Y")},
        "initial_state": {"X0": V(list(M.X0), "m", "fitted" if fitted else "assumed"), "V0": V(list(M.V0), "m/s", "fitted" if fitted else "assumed"),
                          "q0": V(list(M.q0), "-", "fitted" if fitted else "assumed", convention="w,x,y,z body -> world"),
                          "w0": V(list(M.w0), "rad/s", "fitted" if fitted else "assumed", convention="WORLD angular velocity"),
                          "tilt_mode": g["tilt_mode"], "theta0_used": list(info["theta0_used"]), "kappa1_tilt_over_swing": info["kappa1_tilt_over_swing"]},
        "derived": der,
        "integrator": {"solver_id": td.SOLVER_ID, "method": "classical RK4, fixed step dividing each output interval exactly",
                       "h_s": 1.0 / 3072.0, "t_end_s": 10.0, "output_dt_s": 1.0 / 96.0},
    }
    if fit:
        p["fit_provenance"] = {"fit_json": str(Path(a.fit_json).resolve().relative_to(ROOT)), "production_fit": fit["production_fit"],
                               "t_anchor_s": fit["t_anchor_s"], "metrics": fit["metrics"], "flags": fit["flags"],
                               "model_selection": fit["model_selection"], "inputs": fit["inputs"], "reduced_chi2": fit["reduced_chi2"]}
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
    ins = {f"solver/{f}": hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in ("make_params.py", "model_build.py", "torsion_disc.py")}
    if a.fit_json:
        ins[str(Path(a.fit_json).resolve().relative_to(ROOT))] = hashlib.sha256(Path(a.fit_json).read_bytes()).hexdigest()
    p["generator"] = {"script": "solver/make_params.py", "argv": list(argv if argv is not None else sys.argv[1:]),
                      "inputs_sha256": ins, "note": "byte-reproducible: rerun with this argv from ROOT"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(p, indent=2, default=float) + "\n")
    side = HERE / "out" / "make_params_run.json"; side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps({"generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "out": str(a.out), "out_sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest(),
                                "argv": p["generator"]["argv"]}, indent=2) + "\n")
    print(json.dumps({k: p["parameters"][k]["value"] for k in ("ell", "R", "t", "m", "kappa", "k_A")}, indent=1))
    print(json.dumps({k: p["wire"][k]["value"] for k in p["wire"]}))
    print(json.dumps({k: p["derived"][k] for k in ("T_torsion_s", "omega_lateral_rad_s", "T_lateral_s", "Q_torsion")}))


if __name__ == "__main__":
    main()
