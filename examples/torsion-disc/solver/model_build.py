"""Single source of the parameter -> Model mapping (used by fit_projected.py AND make_params.py AND make_variant.py, so the
production model is bit-identical to the fitted one).  physics/FORMULATION.md s6-s7.

Primary parameters P (dict):
  ell      wire length |A B| [m] (clamp exit -> top-face centre); model-conditioned scale.  In scale_mode "prompt" it is
           NOT free: ell = Lcam * R_prompt / R_cam (disc radius declared from the prompt's ~10 cm diameter).
  Lcam     the same length in camera units  ->  lam = Lcam / ell camera units per metre
  om_t     torsion angular frequency sqrt(kappa / I_z) [rad/s]  (identified by the twist period)
  psi0, psid0          initial twist [rad] and twist rate [rad/s] (swing-twist decomposition about the disc normal)
  mu                   mark azimuth at zero twist [rad] in the body frame (body x = world X at psi = 0, no tilt); the
                       observable mark azimuth is mu + psi (seen from above, counter-clockwise X -> Y)
  phx0, phy0, phxd0, phyd0   initial wire angle (direction ~ (tan phx, tan phy, -1)) and rates
  thx0, thy0, thxd0, thyd0   initial disc-normal angles (same convention), ONLY used if geom tilt_mode == "free";
                       default tilt_mode "mode1": tilt on the linear swing-mode shape (theta = kappa1 phi per axis),
                       i.e. the tilt (wobble) mode starts at rest
  gamma_t              (optional, fit) torsion decay rate c_t/(2 I_z) [1/s]; overrides Q_t when present
  Q_t, Q_sw, Q_tilt    quality factors (damping) of torsion / swing / tilt; c_t = I_z om_t / Q_t,
                       c_x = m om_1 / Q_sw, c_r = I_perp om_2 / Q_tilt (om_1, om_2 = linear lateral modes)
Geometry G (dict; camera units measured by roles A/C, the rest assumed):
  R_cam, t_cam      disc radius and thickness in camera units
  rho               brass density [kg/m^3] (assumed 8500)
  G_shear, E_young  steel wire moduli (assumed 79.3 GPa, 200 GPa)
  bend_ends         "clamped" (k_A = k_B = sqrt(E I_w T0), boundary-layer springs) | "free" (k = 0, ideal string) |
                    "rigid_disc" (k_A boundary layer at the clamp; disc joint rigid: k_B = I_perp (15)^2 g/ell)
  scale_mode        "swing" (ell free; set by the swing frequency + g) | "prompt" (R = R_prompt declared)
  R_prompt          0.05 m (prompt: "about ten centimetres across")
"""
from __future__ import annotations

import math

import numpy as np

import torsion_disc as td

G_STD = td.G_STD
RIGID_RATIO = 15.0
GEOM_DEFAULT = dict(unit="placeholder camera unit (1 = wire length at rest)",
                    R_cam=0.55, t_cam=0.033, rho=8500.0, G_shear=79.3e9, E_young=200e9,
                    bend_ends="clamped", tilt_mode="mode1", scale_mode="prompt", R_prompt=0.05,
                    source="PLACEHOLDER ratios from frame 0 by eye (disc ~540 px wide, wire ~490 px); replace with roles A/C measurement",
                    provenance={"R_cam": "placeholder", "t_cam": "placeholder", "rho": "assumed (brass 8400-8700)",
                                "G_shear": "assumed (steel 79.3 GPa)", "E_young": "assumed (steel 200 GPa)",
                                "R_prompt": "declared from the prompt (about ten centimetres across)"})


def dir_and_derivs(ax, ay):
    """unit vector e ~ (tan ax, tan ay, -1)/norm and its partial derivatives d e/d ax, d e/d ay (exact)."""
    v = np.array([math.tan(ax), math.tan(ay), -1.0]); nv = np.linalg.norm(v)
    e = v / nv
    Pm = (np.eye(3) - np.outer(e, e)) / nv
    return e, Pm @ np.array([1.0 / math.cos(ax) ** 2, 0.0, 0.0]), Pm @ np.array([0.0, 1.0 / math.cos(ay) ** 2, 0.0])


def state_from_angles(ell, h_c, psi, psid, phx, phy, phxd, phyd, thx, thy, thxd, thyd):
    """World-frame initial state (X, V, q, w) from wire angles, disc-normal angles, twist and their rates.
    Exact: |B| = ell, B . dB/dt = 0, psi(q) = psi, G(q) . w = psid, dn/dt = w x n."""
    s, s_a, s_b = dir_and_derivs(phx, phy)
    n, n_a, n_b = dir_and_derivs(thx, thy)          # disc DOWNWARD normal
    sd = s_a * phxd + s_b * phyd
    nd = n_a * thxd + n_b * thyd
    q = td.quat_from_tilt_twist(n, psi)
    w_perp = np.cross(n, nd)
    G = td.grad_psi(q)
    c = (psid - G @ w_perp) / (G @ n)
    w = w_perp + c * n
    r = -h_c * n                                    # B - X = R (0,0,h_c) = -h_c n
    B = ell * s; Bd = ell * sd
    X = B - r; V = Bd - np.cross(w, r)
    return [float(x) for x in X], [float(x) for x in V], [float(x) for x in q], [float(x) for x in w]


def physical(P, geom=None):
    """Geometry/scale/material chain -> dict of SI quantities (no initial state)."""
    g = dict(GEOM_DEFAULT); g.update(geom or {})
    if g["scale_mode"] == "prompt":
        lam = g["R_cam"] / g["R_prompt"]; ell = P["Lcam"] / lam
    else:
        ell = P["ell"]; lam = P["Lcam"] / ell
    R = g["R_cam"] / lam; t = g["t_cam"] / lam
    m = g["rho"] * math.pi * R * R * t
    I_z = 0.5 * m * R * R; I_perp = m * (3 * R * R + t * t) / 12.0
    kappa = I_z * P["om_t"] ** 2
    d_wire = (32.0 * ell * kappa / (math.pi * g["G_shear"])) ** 0.25
    EI = g["E_young"] * math.pi * d_wire ** 4 / 64.0
    T0 = m * G_STD
    kb = math.sqrt(EI * T0) if g["bend_ends"] in ("clamped", "rigid_disc") else 0.0
    kB = kb
    if g["bend_ends"] == "rigid_disc":
        # rigid wire-to-disc joint (lead decision 2026-10-26T03:42Z: the reference disc stays perpendicular to the wire):
        # stiff joint spring whose decoupled tilt frequency is RIGID_RATIO x the point-pendulum frequency sqrt(g/ell);
        # the joint then follows the wire to O((1/RIGID_RATIO)^2) ~ 4e-3 (FORMULATION s8); 15 keeps the stiff joint mode (~110 rad/s) resolved by h = 1/3072 s
        kB = I_perp * (g.get("rigid_ratio", RIGID_RATIO) ** 2) * G_STD / ell
    return dict(g=g, lam=lam, ell=ell, R=R, t=t, m=m, I_z=I_z, I_perp=I_perp, kappa=kappa, d_wire=d_wire, EI=EI,
                T0=T0, k_A=kb, k_B=kB, lambda_bl=(math.sqrt(EI / T0) if T0 > 0 else None),
                shear_strain_per_rad=0.5 * d_wire / ell)


def build_model(P, geom=None):
    ph = physical(P, geom); g = ph["g"]
    base = td.Model(m=ph["m"], R=ph["R"], t=ph["t"], ell=ph["ell"], kappa=ph["kappa"], k_A=ph["k_A"], k_B=ph["k_B"],
                    c_t=0.0, c_x=0.0, c_r=0.0, g=G_STD, X0=(0.0, 0.0, -ph["ell"] - 0.5 * ph["t"]), V0=(0.0, 0.0, 0.0),
                    q0=(1.0, 0.0, 0.0, 0.0), w0=(0.0, 0.0, 0.0))
    wl, Vm = td.lateral_modes(base)
    kappa1 = float(Vm[1, 0])
    if P.get("gamma_t") is not None:                 # fit parameterisation: gamma_t = c_t / (2 I_z) >= 0  (Q_t = om_t / (2 gamma_t))
        c_t = 2.0 * ph["I_z"] * P["gamma_t"]
    else:
        c_t = ph["I_z"] * P["om_t"] / P["Q_t"] if P.get("Q_t") else 0.0
    c_x = ph["m"] * wl[0] / P["Q_sw"] if P.get("Q_sw") else 0.0
    # tilt drag: referenced to the tilt (wobble) mode for an elastic joint; for the RIGID joint the disc only rocks with the
    # swing, so the reference frequency is the swing mode (otherwise the stiff joint frequency would turn Q_tilt into a
    # spuriously strong swing damping)
    w_ref = wl[0] if g["bend_ends"] == "rigid_disc" else wl[1]
    c_r = ph["I_perp"] * w_ref / P["Q_tilt"] if P.get("Q_tilt") else 0.0
    gp = lambda k: P.get(k, 0.0)
    if g["tilt_mode"] == "mode1":
        th = (kappa1 * gp("phx0"), kappa1 * gp("phy0"), kappa1 * gp("phxd0"), kappa1 * gp("phyd0"))
    else:
        th = (gp("thx0"), gp("thy0"), gp("thxd0"), gp("thyd0"))
    X0, V0, q0, w0 = state_from_angles(ph["ell"], 0.5 * ph["t"], P["psi0"], gp("psid0"), gp("phx0"), gp("phy0"),
                                       gp("phxd0"), gp("phyd0"), *th)
    M = td.replace(base, c_t=c_t, c_x=c_x, c_r=c_r, X0=tuple(X0), V0=tuple(V0), q0=tuple(q0), w0=tuple(w0))
    info = dict(ph, kappa1_tilt_over_swing=kappa1, omega_lateral=wl, theta0_used=th, c_t=c_t, c_x=c_x, c_r=c_r,
                mu=gp("mu"), Q_t=P.get("Q_t"), Q_sw=P.get("Q_sw"), Q_tilt=P.get("Q_tilt"))
    return M, info


def reequilibrate(M, psi0, psid0, phx0=0.0, phy0=0.0, phxd0=0.0, phyd0=0.0):
    """Re-equilibrated initial state for a parameter VARIANT M (P9 controls, clip-02 lesson): the variant's own equilibrium
    (|B| = its ell, hanging straight), the SAME twist amplitude/phase (psi0, psid0) and swing angles, tilt mode at rest
    (theta on the variant's own swing-mode shape).  Returns (Model, info)."""
    base = td.replace(M, c_t=0.0, c_x=0.0, c_r=0.0)
    wl, Vm = td.lateral_modes(base)
    k1 = float(Vm[1, 0])
    X0, V0, q0, w0 = state_from_angles(M.ell, M.h_c, psi0, psid0, phx0, phy0, phxd0, phyd0,
                                       k1 * phx0, k1 * phy0, k1 * phxd0, k1 * phyd0)
    return td.replace(M, X0=tuple(X0), V0=tuple(V0), q0=tuple(q0), w0=tuple(w0)), dict(kappa1=k1, omega_lateral=wl,
                                                                                         omega_torsion=math.sqrt(M.kappa / M.I_z))
