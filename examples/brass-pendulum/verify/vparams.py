"""Verifier-owned parameter loader (independent of solver/).

Maps params/physics.json onto the verifier's model
    I_P th'' = -mgd sin th - c_v th' - k_q |th'| th' - tau_c sign(th')
The schema of physics.json is owned by dynamics and was unknown when this was written, so keys are
found by alias search (recursive; a value may be a number or a dict with "value").  Every mapping
is recorded in Model.provenance so reports show exactly what was used.  An explicit JSON override
({"I_P":..,"mgd":..,...}) wins over everything.
"""
import json, hashlib, math

G_STD = 9.80665

ALIASES = {
    "g": ["g", "g_m_s2", "g_mps2", "gravity", "gravity_m_s2"],
    "L_c": ["L_c", "L_c_m", "Lc", "L_c_model", "pivot_to_bob_centre_m", "pivot_to_bob_center_m"],
    "m_total": ["m_total", "m_total_kg", "mass_total", "mass_total_kg", "total_mass_kg", "m"],
    "m_bob": ["m_bob", "m_bob_kg", "bob_mass", "bob_mass_kg", "mass_bob_kg", "bob_m_kg"],
    "r_bob": ["r_bob", "r_bob_m", "bob_radius", "bob_radius_m", "R_bob", "R_bob_m"],
    "I_P": ["I_P", "I_P_kg_m2", "I_pivot", "I_pivot_kg_m2", "I_eff", "I_eff_kg_m2"],
    "d": ["d", "d_m", "d_com", "d_com_m", "com_distance_m", "d_cm_m"],
    "mgd": ["mgd", "m_g_d", "mgd_N_m", "gravity_torque_coeff_N_m"],
    "c_v": ["c_v", "c", "c_v_N_m_s", "c_v_Nms", "c_visc", "viscous", "c_viscous", "c_v_N_m_s_per_rad"],
    "k_q": ["k_q", "k_q_N_m_s2", "k_quad", "quadratic", "c_q", "k_q_Nms2"],
    "tau_c": ["tau_c", "tau_c_N_m", "coulomb", "tau_coulomb", "tau_coulomb_N_m"],
    "theta0": ["theta_init", "theta0", "theta0_rad", "theta_0", "theta_0_rad", "theta_init_rad"],
    "omega0": ["omega_init", "omega0", "omega0_rad_s", "omega_0", "omega_0_rad_s", "omega_init_rad_s"],
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict) and "value" in v and isinstance(v["value"], (int, float)) and not isinstance(v["value"], bool):
        return float(v["value"])
    return None


def _find_all(obj, key, path=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}/{k}"
            if k == key and _num(v) is not None:
                out.append((p, _num(v)))
            out.extend(_find_all(v, key, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_find_all(v, key, f"{path}[{i}]"))
    return out


def find(params, name):
    """Return (value, jsonpath) for the first alias found, or (None, None). Ambiguity -> error."""
    for alias in ALIASES[name]:
        hits = _find_all(params, alias)
        # ignore hits under provenance/uncertainty sub-trees
        hits = [h for h in hits if not any(s in h[0].lower() for s in ("uncert", "sigma", "provenance", "/range", "bounds", "prior"))]
        if hits:
            vals = {round(v, 15) for _, v in hits}
            if len(vals) > 1:
                raise ValueError(f"ambiguous key {alias!r} for {name}: {hits}")
            return hits[0][1], hits[0][0]
    return None, None


class Model:
    def __init__(self, I_P, mgd, c_v=0.0, k_q=0.0, tau_c=0.0, theta0=0.0, omega0=0.0,
                 g=G_STD, L_c=None, m_bob=None, provenance=None):
        self.I_P, self.mgd = float(I_P), float(mgd)
        self.c_v, self.k_q, self.tau_c = float(c_v), float(k_q), float(tau_c)
        self.theta0, self.omega0 = float(theta0), float(omega0)
        self.g, self.L_c, self.m_bob = g, L_c, m_bob
        self.provenance = provenance or {}

    @property
    def wn(self):
        return math.sqrt(self.mgd / self.I_P)

    @property
    def conservative(self):
        return self.c_v == 0 and self.k_q == 0 and self.tau_c == 0

    def replace(self, **kw):
        d = self.as_dict()
        d.update(kw)
        prov = dict(self.provenance); prov["replaced"] = sorted(kw)
        d.pop("wn", None)
        return Model(provenance=prov, **d)

    def as_dict(self):
        return dict(I_P=self.I_P, mgd=self.mgd, c_v=self.c_v, k_q=self.k_q, tau_c=self.tau_c,
                    theta0=self.theta0, omega0=self.omega0, g=self.g, L_c=self.L_c, m_bob=self.m_bob)


def load_model(params_path, override_path=None):
    with open(params_path) as f:
        P = json.load(f)
    prov = {"params_path": params_path, "params_sha256": sha256_file(params_path)}
    vals = {}
    for name in ALIASES:
        v, p = find(P, name)
        vals[name] = v
        if v is not None:
            prov[name] = p
    ov = {}
    if override_path:
        with open(override_path) as f:
            ov = json.load(f)
        prov["override"] = {"path": override_path, "sha256": sha256_file(override_path), "keys": sorted(ov)}
    def get(n):
        if n in ov:
            return float(ov[n])
        return vals.get(n)
    g = get("g") if get("g") is not None else G_STD
    if get("g") is None:
        prov["g"] = "verifier default 9.80665 (contract)"
    L_c = get("L_c")
    m_bob = get("m_bob")
    m_tot = get("m_total")
    r = get("r_bob")
    I_P, d, mgd = get("I_P"), get("d"), get("mgd")
    if mgd is None:
        m_for = m_tot if m_tot is not None else m_bob
        dd = d if d is not None else L_c
        if m_for is None or dd is None:
            raise ValueError("cannot form m g d: need (m_total|m_bob) and (d|L_c), or mgd/override")
        mgd = m_for * g * dd
        prov["mgd"] = f"verifier: m({'m_total' if m_tot is not None else 'm_bob'})*g*({'d' if d is not None else 'L_c'})"
    if I_P is None:
        if m_bob is None and m_tot is None or L_c is None:
            raise ValueError("cannot form I_P: need I_P or (mass, L_c[, r_bob])")
        mm = m_bob if m_bob is not None else m_tot
        rr = r if r is not None else 0.0
        I_P = mm * (L_c ** 2 + 0.4 * rr ** 2)
        prov["I_P"] = "verifier-derived: solid sphere on massless wire m(L_c^2 + 2/5 r^2)" + ("" if r is not None else " with r=0 (point mass)")
    if m_bob is None:
        m_bob = m_tot
    return Model(I_P=I_P, mgd=mgd,
                 c_v=get("c_v") or 0.0, k_q=get("k_q") or 0.0, tau_c=get("tau_c") or 0.0,
                 theta0=get("theta0") if get("theta0") is not None else float("nan"),
                 omega0=get("omega0") or 0.0, g=g, L_c=L_c, m_bob=m_bob, provenance=prov), P
