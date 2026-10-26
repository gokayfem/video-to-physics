"""Write a P9 parameter-VARIANT params file with a RE-EQUILIBRATED initial state (clip-02 lesson): the variant's own hanging
equilibrium (|B| = its ell, level disc), the SAME twist amplitude/phase (psi0, psid0) and wire swing angles as production,
tilt mode at rest (disc normal on the variant's own swing-mode shape).

  .venv/bin/python solver/make_variant.py --set kappa=x4 [--set ell=x1.5 --set m=x2 ...] --out solver/out/variants/kappa_x4.json
  .venv/bin/python solver/run.py --params solver/out/variants/kappa_x4.json --out solver/out/variants/kappa_x4
--set NAME=xF multiplies, NAME=V sets (NAME in m, R, t, ell, kappa, k_A, k_B, c_t, c_x, c_r, g).  Predicted control response
(equations): omega_torsion = sqrt(kappa / (m R^2 / 2)); lateral modes from det(K - w^2 M) = 0 (FORMULATION s4); both are
written to the variant's "derived" block.
"""
import argparse, copy, hashlib, json, sys
from dataclasses import replace
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import torsion_disc as td, model_build as mb

ap = argparse.ArgumentParser()
ap.add_argument("--params", default=str(ROOT / "params/physics.json")); ap.add_argument("--set", action="append", default=[])
ap.add_argument("--out", required=True); a = ap.parse_args()
p = json.loads(Path(a.params).read_text()); M = td.model_from_physics_json(p)
kw = {}
for item in a.set:
    k, v = item.split("=")
    kw[k] = getattr(M, k) * float(v[1:]) if v.startswith("x") else float(v)
Mv = replace(M, **kw)
pr = p["primary"]; gv = lambda k: float((pr.get(k) or {}).get("value") or 0.0)
ic = {k: gv(k) for k in ("psi0", "psid0", "phx0", "phy0", "phxd0", "phyd0")}
Mv, info = mb.reequilibrate(Mv, ic["psi0"], ic["psid0"], ic["phx0"], ic["phy0"], ic["phxd0"], ic["phyd0"])
q = copy.deepcopy(p)
q["status"] = "test"
q["status_note"] = (f"P9 VARIANT of {Path(a.params).name} (sha256 {hashlib.sha256(Path(a.params).read_bytes()).hexdigest()[:12]}): {a.set}; "
                    "re-equilibrated IC (own equilibrium, same twist psi0/psid0 and swing angles, tilt mode at rest)")
P = q["parameters"]
for k in ("m", "R", "t", "ell", "kappa", "k_A", "k_B", "g"):
    P[k]["value"] = getattr(Mv, k)
P["I_z"]["value"] = Mv.I_z; P["I_perp"]["value"] = Mv.I_perp
for k in ("c_t", "c_x", "c_r"):
    P["damping"][k]["value"] = getattr(Mv, k)
for k, v in (("X0", Mv.X0), ("V0", Mv.V0), ("q0", Mv.q0), ("w0", Mv.w0)):
    q["initial_state"][k]["value"] = list(v); q["initial_state"][k]["provenance"] = "re-equilibrated variant (solver/make_variant.py)"
q["derived"] = Mv.derived()
q["variant"] = {"set": a.set, "reequilibrate": info, "ic_angles": ic, "generator": "solver/make_variant.py", "argv": sys.argv[1:]}
Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(q, indent=2, default=float) + "\n")
print(a.out, json.dumps(info, default=float))
