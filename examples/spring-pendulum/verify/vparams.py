"""Verifier-owned parameter loader (stdlib + numpy; also used inside Blender). Never imports solver/.

params/physics.json schema is owned by dynamics and unknown when this was written, so scalars/vectors are
found by alias search (recursive; a value may be a number, a list, or a dict with "value").  Every
mapping is recorded in Model.provenance.  A canonical override JSON (keys = vmodel.Model.FIELDS) wins.
Subtrees named like uncertainty/sigma/provenance/bounds/prior/range are ignored.
"""
import json, hashlib, copy
from vmodel import Model, G_STD

ALIASES = {
    "g": ["g", "g_m_s2", "gravity", "gravity_m_s2"],
    "A": ["A", "anchor", "anchor_m", "anchor_world_m", "A_m", "anchor_position_m"],
    "k": ["k", "k_N_m", "k_N_per_m", "spring_k", "stiffness", "stiffness_N_m", "spring_constant_N_m"],
    "L0": ["L0", "L0_m", "rest_length", "rest_length_m", "spring_rest_length_m", "L_rest_m", "free_length_m"],
    "F_i": ["F_i", "F_i_N", "preload", "preload_N", "initial_tension_N", "F0_N"],
    "M": ["M", "m", "m_kg", "mass", "mass_kg", "m_weight", "m_weight_kg", "weight_mass_kg", "M_kg"],
    "m_s": ["m_s", "m_s_kg", "m_spring", "m_spring_kg", "spring_mass", "spring_mass_kg"],
    "e_b": ["e_b", "eye_offset_body_m", "eye_body_m", "hook_offset_body_m", "r_eye_body_m", "eye_from_com_m"],
    "I_c": ["I_c", "I_com", "I_com_kg_m2", "inertia_com_kg_m2", "I_body_kg_m2", "inertia_kg_m2"],
    "c_v": ["c_v", "c_v_N_s_m", "c_drag", "c_air", "drag_linear_N_s_m", "c_lin_N_s_m"],
    "c_a": ["c_a", "c_a_N_s_m", "c_axial", "c_spring", "spring_damping_N_s_m", "c_axial_N_s_m"],
    "c_r": ["c_r", "c_r_N_m_s", "c_rot", "rocking_damping_N_m_s", "c_rock"],
    "E0": ["E0", "eye0", "eye0_m", "r0", "r0_m", "x0_m", "pos0", "pos0_m", "position0_m", "p0_m", "r_init_m"],
    "vE0": ["vE0", "v0", "v0_m_s", "vel0", "vel0_m_s", "velocity0_m_s", "v_init_m_s"],
    "q0": ["q0", "q0_wxyz", "quat0", "quat0_wxyz", "q_init_wxyz"],
    "w0": ["w0", "omega0", "omega0_rad_s", "w0_rad_s", "omega_world0_rad_s", "omega_init_rad_s"],
    "body": ["body", "body_model", "weight_model", "model_kind"],
}
SKIP = ("uncert", "sigma", "provenance", "/range", "bounds", "prior", "bootstrap", "history", "_verifier")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _val(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float, str)):
        return v
    if isinstance(v, list) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
        return v
    if isinstance(v, list) and v and all(isinstance(x, list) for x in v):
        return v
    if isinstance(v, dict) and "value" in v:
        return _val(v["value"])
    return None


def _find_all(obj, key, path=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}/{k}"
            if k == key and _val(v) is not None:
                out.append((p, _val(v)))
            out.extend(_find_all(v, key, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                out.extend(_find_all(v, key, f"{path}[{i}]"))
    return out


def find(params, name):
    for alias in ALIASES[name]:
        hits = [h for h in _find_all(params, alias) if not any(s in h[0].lower() for s in SKIP)]
        if hits:
            vals = {json.dumps(h[1]) for h in hits}
            if len(vals) > 1:
                raise ValueError(f"ambiguous key {alias!r} for {name}: {hits}")
            return hits[0][1], hits[0][0]
    return None, None


def get_path(P, path):
    o = P
    for k in [k for k in path.split("/") if k]:
        o = o[k]
    return o["value"] if isinstance(o, dict) and "value" in o else o


def set_path(P, path, val):
    keys = [k for k in path.split("/") if k]
    o = P
    for k in keys[:-1]:
        o = o[k]
    if isinstance(o[keys[-1]], dict) and "value" in o[keys[-1]]:
        o[keys[-1]]["value"] = val
    else:
        o[keys[-1]] = val


def load_model(params_path, override_path=None):
    with open(params_path) as f:
        P = json.load(f)
    prov = {"params_path": params_path, "params_sha256": sha256_file(params_path)}
    vals = {}
    for name in ALIASES:
        try:
            v, p = find(P, name)
        except ValueError as e:
            v, p = None, f"AMBIGUOUS: {e}"
        vals[name] = v
        if p is not None:
            prov[name] = p
    ov = {}
    if override_path:
        with open(override_path) as f:
            ov = json.load(f)
        prov["override"] = {"path": override_path, "sha256": sha256_file(override_path), "keys": sorted(ov)}
    kw = {}
    for name in Model.FIELDS:
        v = ov[name] if name in ov else vals.get(name)
        if v is not None:
            kw[name] = v
    if "g" not in kw:
        kw["g"] = G_STD; prov["g"] = "verifier default 9.80665 (contract)"
    body = str(kw.get("body", "point")).lower()
    kw["body"] = "rigid" if "rigid" in body else "point"
    missing = [n for n in ("k", "L0", "M") if n not in kw]
    if missing:
        raise ValueError(f"cannot build verifier model: missing {missing}; supply --model-json")
    return Model(provenance=prov, **kw), P


def write_variant(P, model, edits, out_path, note=""):
    """edits: {canonical_name: value}. Writes an edited COPY of params at the provenance paths."""
    Q = copy.deepcopy(P)
    done = []
    for name, val in edits.items():
        path = model.provenance.get(name)
        if not path or not path.startswith("/"):
            raise SystemExit(f"cannot edit {name}: no params path (provenance {path!r}); use --model-json params")
        set_path(Q, path, val); done.append((path, val))
    Q["_verifier_variant"] = dict(edits=done, note=note or "edited COPY made by verify/")
    with open(out_path, "w") as f:
        json.dump(Q, f, indent=2)
    return done


# ---------------------------------------------------------------- clip-02 production schema adapter
# Documented in solver/NOTES.md "For the independent verifier" (model_id elastic_pendulum_hinged_body_v1). Mapping:
#   A (verifier spring anchor) = pivot_world_m (P_eff); e_b = (0,0,b) (body local -Z points hinge->CoM, so CoM->eye = +b z);
#   I_c = diag(I_perp, I_perp, I_perp) (spin is zero and stays zero for an axisymmetric body: axial value irrelevant);
#   c_v = damping.c_x (CoM drag), c_a = damping.c_s (axial dashpot), c_r = damping.c_r (on n' == omega for zero spin);
#   initial state: X0,V0 (CoM rel. P), n0, u0 = n0' -> E0 = P + X0 - b n0, vE0 = V0 - b u0, q0 = minimal rotation (-z -> n0),
#   w0 = n0 x u0.
import math as _m
import numpy as _np
CK_MODEL_ID = "elastic_pendulum_hinged_body_v1"
CK_PATHS = {"g": "/parameters/g", "k": "/parameters/k", "L0": "/parameters/L0", "M": "/parameters/m", "m_s": "/parameters/m_s",
            "b": "/parameters/b", "I_perp": "/parameters/I_perp", "c_v": "/parameters/damping/c_x", "c_a": "/parameters/damping/c_s",
            "c_r": "/parameters/damping/c_r", "A": "/pivot_world_m"}


def _minrot_quat(n):
    a = _np.array([0.0, 0.0, -1.0]); n = _np.asarray(n, float) / _np.linalg.norm(n)
    ax = _np.cross(a, n); s = _np.linalg.norm(ax); c = float(a @ n)
    if s < 1e-300:
        return [1.0, 0.0, 0.0, 0.0] if c > 0 else [0.0, 1.0, 0.0, 0.0]
    ang = _m.atan2(s, c); ax = ax / s
    return [_m.cos(ang / 2), *(_m.sin(ang / 2) * ax)]


def _ck_model(P, path, override):
    g = lambda p: get_path(P, p)
    prov = {"params_path": path, "params_sha256": sha256_file(path), "adapter": f"vparams._ck_model ({CK_MODEL_ID})"}
    prov.update({k: v for k, v in CK_PATHS.items()})
    b = float(g(CK_PATHS["b"])); Ip = float(g(CK_PATHS["I_perp"])); A = _np.array(g(CK_PATHS["A"]), float)
    st = P["initial_state"]
    X0, V0, n0, u0 = (_np.array(_val(st[k]), float) for k in ("X0", "V0", "n0", "u0"))
    if abs(float(_val(st.get("spin_omega3", 0.0)) or 0.0)) > 0:
        raise ValueError("nonzero spin_omega3 not supported by the adapter")
    kw = dict(g=g(CK_PATHS["g"]), k=g(CK_PATHS["k"]), L0=g(CK_PATHS["L0"]), M=g(CK_PATHS["M"]), m_s=g(CK_PATHS["m_s"]), F_i=0.0,
              body="rigid", A=A.tolist(), e_b=[0.0, 0.0, b], I_c=[Ip, Ip, Ip], c_v=g(CK_PATHS["c_v"]), c_a=g(CK_PATHS["c_a"]),
              c_r=g(CK_PATHS["c_r"]), E0=(A + X0 - b * n0).tolist(), vE0=(V0 - b * u0).tolist(), q0=_minrot_quat(n0),
              w0=_np.cross(n0, u0).tolist())
    prov.update(E0="/initial_state (P + X0 - b n0)", vE0="/initial_state (V0 - b u0)", q0="/initial_state (minrot -z->n0)",
                w0="/initial_state (n0 x u0)", I_c="/parameters/I_perp (isotropic; spin 0)", e_b="/parameters/b (0,0,b)")
    kw.update({k: v for k, v in override.items() if k in Model.FIELDS})
    return Model(provenance=prov, **kw), P


def _ck_edits(P, model, edits):
    """canonical edits -> production schema edits (list of (path, value))."""
    out = []
    st_keys = {"E0", "vE0", "q0", "w0"}
    for name, val in edits.items():
        if name in st_keys:
            continue
        if name == "I_c":
            val = float(_np.array(val, float).reshape(-1)[0] if _np.array(val).size == 3 else _np.array(val)[0][0]); name = "I_perp"
        if name == "A":
            raise SystemExit("editing the pivot is not supported")
        out.append((CK_PATHS[name], val))
    if st_keys & set(edits):
        m2 = model.replace(**{k: v for k, v in edits.items() if k in Model.FIELDS})
        from vmodel import quat_to_R
        b = float(_np.linalg.norm(m2.e_b))
        if "b" in edits:
            b = float(edits["b"])
        R = quat_to_R(_np.array(m2.q0)); n0 = R @ _np.array([0.0, 0.0, -1.0])
        w0 = _np.array(m2.w0); u0 = _np.cross(w0, n0)
        X0 = _np.array(m2.E0) + b * n0 - m2.A; V0 = _np.array(m2.vE0) + b * u0
        if abs(w0 @ n0) > 1e-12:
            raise SystemExit("edit implies spin about the body axis; not representable")
        out += [("/initial_state/X0", X0.tolist()), ("/initial_state/V0", V0.tolist()), ("/initial_state/n0", n0.tolist()), ("/initial_state/u0", u0.tolist())]
    return out


_load_model_generic = load_model
_write_variant_generic = write_variant


def load_model(params_path, override_path=None):  # noqa: F811
    with open(params_path) as f:
        P = json.load(f)
    if P.get("model_id") == CK_MODEL_ID:
        ov = json.load(open(override_path)) if override_path else {}
        return _ck_model(P, params_path, ov)
    return _load_model_generic(params_path, override_path)


def write_variant(P, model, edits, out_path, note=""):  # noqa: F811
    if P.get("model_id") != CK_MODEL_ID:
        return _write_variant_generic(P, model, edits, out_path, note)
    Q = copy.deepcopy(P)
    done = _ck_edits(P, model, edits)
    for path, val in done:
        set_path(Q, path, val)
    Q["_verifier_variant"] = dict(edits=done, canonical_edits={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in edits.items()},
                                  note=note or "edited COPY made by verify/")
    with open(out_path, "w") as f:
        json.dump(Q, f, indent=2)
    return done
