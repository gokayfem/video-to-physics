"""Write a P9 parameter-VARIANT params file with a RE-EQUILIBRATED initial state (verifier / lead, CHANGELOG
2026-10-26T01:53:54Z): the variant's own static stretch, the SAME swing amplitude and phase as production, axial and
rocking modes at rest.  Without re-equilibration a stiffer/longer spring started from the nominal state is compressed
(tension < 0), which a hooked spring cannot sustain (the solver raises SpringCompressionError).

  .venv/bin/python solver/make_variant.py --set k=x4 [--set L0=x1.5 --set m=x2 ...] --out solver/out/variants/k_x4.json
  .venv/bin/python solver/run.py --params solver/out/variants/k_x4.json --out solver/out/variants/k_x4
--set NAME=xF multiplies, NAME=V sets (NAME in k, L0, m, m_s, b, I_perp, c_s, c_x, c_r, g).  The swing state (phi0, phid0)
is read from params/physics.json primary.phi0/phid0 (production values; out-of-plane 0).
"""
import argparse, copy, hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import spring_pendulum as sp, model_build as mb

ap = argparse.ArgumentParser()
ap.add_argument("--params", default=str(ROOT / "params/physics.json")); ap.add_argument("--set", action="append", default=[])
ap.add_argument("--out", required=True); a = ap.parse_args()
p = json.loads(Path(a.params).read_text()); M = sp.model_from_physics_json(p)
kw = {}
for item in a.set:
    k, v = item.split("=")
    kw[k] = getattr(M, k) * float(v[1:]) if v.startswith("x") else float(v)
from dataclasses import replace
Mv = replace(M, **kw)
phi0, phid0 = p["primary"]["phi0"]["value"], p["primary"]["phid0"]["value"]
Mv, info = mb.reequilibrate(Mv, phi0, phid0)
q = copy.deepcopy(p)
q["status"] = "test"; q["status_note"] = f"P9 VARIANT of {Path(a.params).name} (sha256 {hashlib.sha256(Path(a.params).read_bytes()).hexdigest()[:12]}): {a.set}; re-equilibrated IC (own static stretch, same swing phi0/phid0, axial+rocking at rest)"
P = q["parameters"]
for k in ("k", "L0", "m", "m_s", "b", "I_perp", "g"):
    P[k]["value"] = getattr(Mv, k)
for k in ("c_s", "c_x", "c_r"):
    P["damping"][k]["value"] = getattr(Mv, k)
P["ell_eq"]["value"] = info["ell_eq"]
for k, v in (("X0", Mv.X0), ("V0", Mv.V0), ("n0", Mv.n0), ("u0", Mv.u0)):
    q["initial_state"][k]["value"] = list(v); q["initial_state"][k]["provenance"] = "re-equilibrated variant (solver/make_variant.py)"
q["derived"] = Mv.derived(); q["variant"] = {"set": a.set, "reequilibrate": info, "phi0": phi0, "phid0": phid0,
                                            "generator": "solver/make_variant.py", "argv": sys.argv[1:]}
Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(q, indent=2) + "\n")
print(a.out, json.dumps(info))
