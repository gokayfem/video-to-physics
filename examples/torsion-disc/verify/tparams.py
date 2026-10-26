"""Verifier-owned parameter loader (stdlib + numpy; also used inside Blender). Never imports solver/.

params/physics.json schema is owned by dynamics and was unknown when this was written: scalars/vectors are found by
alias search (recursive; value may be a number, list, or {"value": ...}).  Every mapping is recorded in
Model.provenance.  A canonical override JSON (keys = tmodel.Model.FIELDS) wins over everything.
Subtrees named like uncertainty/sigma/provenance/bounds/prior/range are ignored.
A schema adapter for the real production file is added here once solver/NOTES.md documents it.
"""
import json, hashlib, copy
from tmodel import Model, G_STD

ALIASES = {
    "mode": ["mode", "wire_model", "suspension_model", "model_kind"],
    "g": ["g", "g_m_s2", "gravity_m_s2"],
    "A": ["A", "anchor", "anchor_m", "anchor_world_m", "clamp_world_m", "A_m"],
    "L": ["L", "L_m", "wire_length_m", "L_wire_m", "wire_L_m"],
    "M": ["M", "m", "m_kg", "mass_kg", "disc_mass_kg", "M_kg"],
    "I_c": ["I_c", "I_com_kg_m2", "inertia_com_kg_m2", "I_body_kg_m2", "I_disc_kg_m2", "inertia_kg_m2"],
    "p_b": ["p_b", "attach_body_m", "attachment_body_m", "p_attach_body_m", "r_attach_body_m"],
    "kappa": ["kappa", "kappa_N_m_rad", "kappa_Nm_per_rad", "torsion_constant_N_m_rad", "k_t", "k_torsion"],
    "k_b": ["k_b", "k_b_N_m_rad", "bending_stiffness_N_m_rad"],
    "bend_law": ["bend_law"],
    "twist": ["twist", "twist_convention", "psi_convention"],
    "c_t": ["c_t", "c_t_N_m_s", "c_torsion", "torsion_damping_N_m_s"],
    "c_s": ["c_s", "c_s_N_m_s", "c_swing", "swing_damping_N_m_s"],
    "c_v": ["c_v", "c_v_N_s_m", "c_air", "drag_linear_N_s_m"],
    "c_r": ["c_r", "c_r_N_m_s", "c_rot", "rotational_damping_N_m_s"],
    "q0": ["q0", "q0_wxyz", "quat0_wxyz", "disc_q0_wxyz"],
    "w0": ["w0", "omega0", "omega0_rad_s", "w0_rad_s", "omega_world0_rad_s"],
    "u0": ["u0", "wire_u0", "wire_dir0"],
    "udot0": ["udot0", "wire_udot0"],
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
    missing = [n for n in ("L", "M", "I_c", "kappa") if n not in kw]
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


# ---------------------------------------------------------------- clip-03 production schema adapter
# physics/FORMULATION.md (model_id disc_on_wire_rigid6_v1) and params/physics.json s7 schema. Mapping to tmodel.Model:
#   mode string; twist 'vertical' (psi = 2 atan2(q_z, q_w) = swing-twist about the disc normal); bend_law 'cos';
#   A = origin; L = ell; M = m; I_c = diag(I_perp, I_perp, I_z); p_b = (0, 0, t/2) (CoM -> top-face centre B);
#   kappa; k_a = k_A (clamp, k_A (1 + s_z) = k_A (1 - u_z)); k_b = k_B (k_B (1 - s.n) = k_B (1 - u.a));
#   damping: c_t (psidot), c_v = c_x (CoM velocity), c_rp = c_r (disc omega perpendicular to the normal), c_s = c_r(iso) = 0.
#   initial_state X0, V0, q0 (w,x,y,z body->world), w0 (world)  ->  u0 = (A - B0)/ell, udot0 = -Bdot0/ell.
import numpy as _np
CK_MODEL_ID = "disc_on_wire_rigid6_v1"
CK_PATHS = {"g": "/parameters/g", "L": "/parameters/ell", "M": "/parameters/m", "I_z": "/parameters/I_z", "I_perp": "/parameters/I_perp",
            "t": "/parameters/t", "kappa": "/parameters/kappa", "k_a": "/parameters/k_A", "k_b": "/parameters/k_B",
            "c_t": "/parameters/damping/c_t", "c_v": "/parameters/damping/c_x", "c_rp": "/parameters/damping/c_r",
            "X0": "/initial_state/X0", "V0": "/initial_state/V0", "q0": "/initial_state/q0", "w0": "/initial_state/w0",
            "A": "/pivot_world_m"}
# ck001 (solver/NOTES.md): suspension point P = pivot_world_m (A + fitted offset); X0 is RELATIVE to the pivot; trajectory is WORLD.


def _ck_model(P, path, override):
    from tmodel import quat_to_R
    g = lambda k: get_path(P, CK_PATHS[k])
    prov = {"params_path": path, "params_sha256": sha256_file(path), "adapter": f"tparams._ck_model ({CK_MODEL_ID})"}
    prov.update({k: v for k, v in CK_PATHS.items()})
    t = float(g("t")); Ip, Iz = float(g("I_perp")), float(g("I_z")); ell = float(g("L"))
    X0, V0, q0, w0 = (_np.array(g(k), float) for k in ("X0", "V0", "q0", "w0"))
    try:
        A = _np.array(g("A"), float).reshape(3)
    except (KeyError, TypeError):
        A = _np.zeros(3); prov["A"] = "no pivot_world_m: origin"
    q0 = q0 / _np.linalg.norm(q0)
    r = quat_to_R(q0) @ _np.array([0.0, 0.0, t / 2])
    B0 = A + X0 + r; Bd0 = V0 + _np.cross(w0, r)
    u0 = (A - B0) / _np.linalg.norm(A - B0)
    kw = dict(mode="string", twist="vertical", bend_law="cos", g=float(g("g")), A=A.tolist(), L=ell, M=float(g("M")),
              I_c=[Ip, Ip, Iz], p_b=[0.0, 0.0, t / 2], kappa=float(g("kappa")), k_a=float(g("k_a")), k_b=float(g("k_b")),
              c_t=float(g("c_t")), c_v=float(g("c_v")), c_rp=float(g("c_rp")), c_s=0.0, c_r=0.0,
              q0=q0.tolist(), w0=w0.tolist(), u0=u0.tolist(), udot0=(-Bd0 / ell).tolist())
    prov.update(u0="/initial_state (A - B0)/|.|", udot0="/initial_state -(V0 + w0 x r)/ell", I_c="diag(I_perp, I_perp, I_z)",
                p_b="(0,0,t/2)", B0_minus_A_len_minus_ell=float(_np.linalg.norm(B0 - A) - ell))
    kw.update({k: v for k, v in override.items() if k in Model.FIELDS})
    return Model(provenance=prov, **kw), P


def _ck_edits(P, model, edits):
    """canonical edits -> production schema edits [(path, value)]. IC edits are converted to X0, V0, q0, w0."""
    from tmodel import quat_to_R
    out = []
    ic = {"q0", "w0", "u0", "udot0"}
    for name, val in edits.items():
        if name in ic:
            continue
        if name == "I_c":
            I = _np.array(val, float); I = _np.diag(I) if I.shape == (3, 3) else I
            out += [(CK_PATHS["I_perp"], float(I[0])), (CK_PATHS["I_z"], float(I[2]))]; continue
        if name in ("c_s", "c_r") and float(val) == 0.0:
            continue
        if name not in CK_PATHS or name == "A":
            raise SystemExit(f"cannot edit {name} in the production schema")
        out.append((CK_PATHS[name], float(val)))
    if ic & set(edits):
        m2 = model.replace(**{k: v for k, v in edits.items() if k in Model.FIELDS})
        R = quat_to_R(_np.array(m2.q0)); r = R @ m2.p_b
        B = m2.A - m2.L * m2.u0; Bd = -m2.L * (m2.udot0 - (m2.udot0 @ m2.u0) * m2.u0)
        w = _np.array(m2.w0)
        out += [(CK_PATHS["X0"], (B - r - m2.A).tolist()), (CK_PATHS["V0"], (Bd - _np.cross(w, r)).tolist()),
                (CK_PATHS["q0"], _np.array(m2.q0).tolist()), (CK_PATHS["w0"], w.tolist())]
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
