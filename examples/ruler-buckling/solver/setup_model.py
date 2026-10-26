"""Builds the rod model from params/physics.json (+ optional measurement/summary.json schedule)."""
import json, copy, hashlib
from pathlib import Path
import numpy as np
from rod import Rod

ROOT = Path(__file__).resolve().parents[1]
PJ = ROOT / "params" / "physics.json"


def v(x):
    return x["value"] if isinstance(x, dict) and "value" in x else x


def load_params():
    raw = json.loads(PJ.read_text())
    P = {
        "strip": {k: v(raw["strip"][k]) for k in raw["strip"]},
        "tip_body": {k: (v(x) if k != "pieces" else x) for k, x in raw["tip_body"].items() if not k.startswith("_")},
        "imperfection": {"tip_offset_m": v(raw["imperfection"]["tip_offset_m"])},
        "damping": {k: v(raw["damping"][k]) for k in raw["damping"]},
        "loading": {k: v(x) for k, x in raw["loading"].items()},
        "g": v(raw["g"]),
    }
    return P, raw


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schedule(P):
    """weight arrival times. measurement/summary.json (key 'weight_arrival_times_s') overrides the default."""
    ms = ROOT / "measurement" / "summary.json"
    L = P["loading"]
    src = "default t_j = %.2f + %.2f j" % (L["t_first_s"], L["spacing_s"])
    times = L["t_first_s"] + L["spacing_s"] * np.arange(L["n_weights"])
    if ms.exists():
        try:
            d = json.loads(ms.read_text())
            for key in ("weight_arrival_times_s", "weight_contact_times_s", "weight_times_s"):
                if key in d:
                    tt = [float(x) for x in d[key]][: L["n_weights"]]
                    if len(tt) >= 1:
                        times = np.array(tt); src = "measurement/summary.json:" + key
                    break
        except Exception as ex:  # keep default, record why
            src += " (summary.json unreadable: %s)" % ex
    return times, src


def body_props(P, fracs, fracs_inertial=None):
    """rigid tip body -> (m_inertial, m_grav, e_grav, J_about_inertial_CoM, e_inertial).
    fracs[j] = contact-force fraction of weight j (slot j of the stack); fracs_inertial[j] = 1 once released."""
    TB = P["tip_body"]
    base = []
    for p in TB["pieces"]:
        m = p["mass_kg"]; L = p.get("len_m", 0.0); d = p.get("lateral_offset_m", 0.0)
        thick = 0.002 if p["name"].startswith("tabs") else (0.025 if p["name"] == "plate_Al" else 0.0)
        Jown = m * (L ** 2 + thick ** 2) / 12 + m * d ** 2
        if p["name"] == "peg_steel":
            Jown = m * (3 * 0.002 ** 2 + L ** 2) / 12
        base.append((m, p["offset_m"], Jown))
    mw, hw = TB["weight_mass_kg"], TB["weight_thickness_m"]
    ro, ri = TB["weight_outer_d_m"] / 2, TB["weight_hole_d_m"] / 2
    z0 = TB["first_weight_base_offset_m"]
    def collect(fr):
        parts = list(base)
        for j, f in enumerate(fr):
            if f > 0:
                m = f * mw
                parts.append((m, z0 + (j + 0.5) * hw, m * (3 * (ro ** 2 + ri ** 2) + hw ** 2) / 12))
        mt = sum(p[0] for p in parts)
        e = sum(p[0] * p[1] for p in parts) / mt
        J = sum(p[2] + p[0] * (p[1] - e) ** 2 for p in parts)
        return mt, e, J
    mg, eg, Jg = collect(fracs)
    if fracs_inertial is None:
        return mg, mg, eg, Jg, eg
    mi, ei, Ji = collect(fracs_inertial)
    return mi, mg, eg, Ji, ei


def fracs_static(lam, n=14):
    return np.clip(lam - np.arange(n), 0.0, 1.0)


def ramp(x):  # smootherstep (FORMULATION.md)
    x = np.clip(x, 0.0, 1.0)
    return x ** 3 * (x * (6 * x - 15) + 10)


def fracs_time(t, times, tau):
    return ramp((t - np.asarray(times)) / tau)


def make_rod(P, N, mode="actual", imperfection=None, **kw):
    """mode 'actual': static lam = continuous weight count; 'dyn': lam = time."""
    if mode == "actual":
        tp = lambda lam: body_props(P, fracs_static(lam))
    elif mode == "dyn":
        times, _ = schedule(P)
        tau = P["loading"]["ramp_s"]
        tp = lambda t: body_props(P, fracs_time(t, times, tau), (np.asarray(t) >= np.asarray(times) + tau).astype(float))
    else:
        raise ValueError(mode)
    return Rod(P, N, tip_pieces=tp, imperfection_m=imperfection, **kw)


def make_point_load_rod(P, N, e=0.0, Lfree=None, imperfection=0.0):
    """verification: point dead load lam [N] at rigid offset e on top of a flexible length Lfree, no self-weight."""
    Q = copy.deepcopy(P)
    Q["tip_body"]["rigid_top_strip_m"] = 0.0
    if Lfree is not None:
        Q["strip"]["free_length_m"] = Lfree
    g = P["g"]
    tp = lambda lam: (0.0, lam / g, e, 0.0, e)
    return Rod(Q, N, tip_pieces=tp, self_weight=False, imperfection_m=imperfection, drag=False, eta=0.0)
