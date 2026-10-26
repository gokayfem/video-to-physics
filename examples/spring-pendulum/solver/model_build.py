"""Single source of the parameter -> Model mapping (used by fit_projected.py AND make_params.py, so the production model
is bit-identical to the fitted one).  physics/FORMULATION.md s6-s7.

Fit/primary parameters P (dict):
  ell   equilibrium |AH| [m] (model-conditioned scale), Lcam (same in camera units)
  bcam  hinge -> cylinder centre in camera units: ONLY used if geom b_mode == "fitted" (diagnostic); default b_mode
        "measured": b = b_cam / lam (role C measurement, per lead)
  rax   omega_axial / sqrt(g/ell) (ignored if geom k_mode == "coil_formula": k = G d^4/(8 D^3 N));
  pz    pivot depth below A in camera units (0 = physical hook bearing A; 0.225 W = reference's apparent pivot P_eff);  phi0, psi0, de0, phid0, psid0, ded0 (+ phiy0, psiy0, phidy0, psidy0) initial state
  gamma_sw, gamma_ax  damping rates [1/s];  beta static tilt [rad] (scene only, not in the solver state)
Geometry G (dict, measured image ratios to ell + assumptions):
  lengths in camera units (r_cam, h_cam, b_cam, e_cam, spring_d_cam, spring_D_cam), spring_turns, rho, rho_spring, gamma_rock,
  k_mode ("rax" | "coil_formula"), G_shear, axial_rest
"""
from __future__ import annotations

import math

import numpy as np

import spring_pendulum as sp

G_STD = sp.G_STD
# Geometry in CAMERA UNITS (W_in = S = clear span between the posts' inner faces; params/geometry.json ratios, role C, measured;
# camera.json unit).  Converted to metres with lam = Lcam / ell (camera units per metre, from the swing-frequency scale).
GEOM_DEFAULT = dict(unit="W_in (= S)",
                    r_cam=0.105786, h_cam=0.260333, b_cam=0.209616, e_cam=0.039725 - 0.008828,
                    spring_d_cam=2 * 0.004715, spring_D_cam=2 * 0.052591, spring_turns=21.0,
                    b_mode="measured", k_mode="rax", G_shear=79.3e9,
                    rho=7900.0, rho_spring=7850.0, gamma_rock=0.5,
                    source=("MEASURED by role C (params/geometry.json, scene geometry v003 = scene/NOTES.md Update 3, thicker eye wire; camera-of-record refit: weight_radius, weight_height, "
                            "derived.H_to_cylinder_centre, eye_ring_radius - eye_wire_radius, 2*spring_wire_radius, 2*coil_mean_radius (centreline), "
                            "active turns = 11 open + 10 tight; the 2 static hook turns are part of the fixed S-hook); "
                            "rho steel, G_shear, gamma_rock assumed"),
                    provenance={"r_cam": "measured", "h_cam": "measured", "b_cam": "measured", "e_cam": "measured",
                                "spring_d_cam": "measured (approx, 6.5 px wire)", "spring_D_cam": "measured", "spring_turns": "counted",
                                "rho": "assumed", "rho_spring": "assumed", "G_shear": "assumed (spring steel 79.3 GPa)", "gamma_rock": "assumed"})


def ic_from_planar(ell, b, phi, psi, de, phid, psid, ded, phi_y=0.0, psi_y=0.0, phid_y=0.0, psid_y=0.0):
    """Initial state from spring angle (phi in XZ, phi_y in YZ via direction ~ (tan phi, tan phi_y, -1)), body angles psi/psi_y,
    stretch offset de and their rates.  Exact closed-form derivatives."""
    L = ell + de

    def dirv(ax, ay):
        v = np.array([math.tan(ax), math.tan(ay), -1.0]); nv = np.linalg.norm(v)
        e = v / nv
        dv_dax = np.array([1.0 / math.cos(ax) ** 2, 0.0, 0.0]); dv_day = np.array([0.0, 1.0 / math.cos(ay) ** 2, 0.0])
        P = (np.eye(3) - np.outer(e, e)) / nv
        return e, P @ dv_dax, P @ dv_day

    eH, eHa, eHb = dirv(phi, phi_y); n, na, nb = dirv(psi, psi_y)
    deH = eHa * phid + eHb * phid_y; dn = na * psid + nb * psid_y
    H = L * eH; Hd = ded * eH + L * deH
    X = H + b * n; V = Hd + b * dn
    return [float(x) for x in X], [float(x) for x in V], [float(x) for x in n], [float(x) for x in dn]


def build_model(P, geom=None):
    g = dict(GEOM_DEFAULT); g.update(geom or {})
    ell = P["ell"]; lam = P["Lcam"] / ell
    b = (g["b_cam"] if g["b_mode"] == "measured" else P["bcam"]) / lam     # hinge -> CoM (cylinder centre)
    r = g["r_cam"] / lam; h = g["h_cam"] / lam
    m = g["rho"] * math.pi * r * r * h
    I_perp = m * (3 * r * r + h * h) / 12.0
    d = g["spring_d_cam"] / lam; D = g["spring_D_cam"] / lam
    m_s = g["rho_spring"] * (math.pi * d * d / 4) * (math.pi * D * g["spring_turns"])
    mu = m_s / 3
    if g["k_mode"] == "coil_formula":      # k = G d^4 / (8 D^3 N) from the measured wire/coil ratios, G assumed
        k = g["G_shear"] * d ** 4 / (8 * D ** 3 * g["spring_turns"])
        om_ax = math.sqrt(k / (m + mu))
    else:
        om_ax = P["rax"] * math.sqrt(G_STD / ell)
        k = (m + mu) * om_ax ** 2
    L0 = ell - (m + 0.5 * m_s) * G_STD / k
    c_x = 2 * m * P["gamma_sw"]; c_s_raw = 2 * (m + mu) * P["gamma_ax"] - c_x; c_s = max(0.0, c_s_raw)
    c_r = 2 * I_perp * g["gamma_rock"]
    de0, ded0 = P.get("de0", 0.0), P.get("ded0", 0.0)
    if g.get("axial_rest"):
        # axial mode at rest: spring length = the quasi-static value for the initial swing state (tension ~ W cos(phi0)
        # + centripetal), so no free axial oscillation is launched (FORMULATION s6; lead decision CHANGELOG)
        W = (m + 0.5 * m_s) * G_STD
        Tqs = W * math.cos(P["phi0"]) + (m + mu) * ell * P["phid0"] ** 2
        de0 = (Tqs - W) / k; ded0 = 0.0
    psi0, psid0 = P.get("psi0", 0.0), P.get("psid0", 0.0)
    kappa = None
    if g.get("rock_rest"):
        # rocking mode at rest: (psi, psid) on the linear swing-mode shape psi/phi = kappa (FORMULATION s4), so only the
        # swing mode is launched (lead decision: in-phase double-pendulum response, no free 4 Hz rocking)
        Mt = sp.Model(k=k, L0=L0, m=m, m_s=m_s, b=b, I_perp=I_perp, c_s=0, c_x=0, c_r=0, g=G_STD,
                      X0=(0, 0, -ell - b), V0=(0, 0, 0), n0=(0, 0, -1), u0=(0, 0, 0))
        _, Vm, _ = sp.lateral_mode_shapes(Mt)
        kappa = float(Vm[1, 0] / Vm[0, 0])
        psi0, psid0 = kappa * P["phi0"], kappa * P["phid0"]
    X0, V0, n0, u0 = ic_from_planar(ell, b, P["phi0"], psi0, de0, P["phid0"], psid0, ded0,
                                    P.get("phiy0", 0.0), P.get("psiy0", 0.0), P.get("phidy0", 0.0), P.get("psidy0", 0.0))
    M = sp.Model(k=k, L0=L0, m=m, m_s=m_s, b=b, I_perp=I_perp, c_s=c_s, c_x=c_x, c_r=c_r, g=G_STD,
                 X0=tuple(X0), V0=tuple(V0), n0=tuple(n0), u0=tuple(u0), body="rigid")
    info = dict(kappa_swing_mode=kappa, psi0_used=psi0, psid0_used=psid0, de0_used=de0, lam=lam, ell=ell, b=b, e=g["e_cam"] / lam, rax=om_ax / math.sqrt(G_STD / ell), pz=P.get("pz", 0.0),
                pivot_world=[P.get("px", 0.0) / lam, 0.0, -P.get("pz", 0.0) / lam], px=P.get("px", 0.0), r=r, h=h, m=m, I_perp=I_perp, m_s=m_s, spring_d=d, spring_D=D, k=k, L0=L0,
                omega_axial=om_ax, c_x=c_x, c_s=c_s, c_r=c_r, c_s_clipped=bool(c_s_raw < 0), geom=g)
    return M, info


def reequilibrate(M, phi0, phid0, phi_y0=0.0, phid_y0=0.0):
    """Re-equilibrated initial state for a parameter VARIANT M (P9 controls; CHANGELOG 2026-10-26T01:53:54Z):
    the variant's own static stretch (ell = L0 + (m + m_s/2) g / k), the SAME swing amplitude/phase (phi0, phid0 = spring
    angle about the pivot and its rate), axial mode at rest (quasi-static stretch for the initial swing state) and rocking
    mode at rest (psi on the variant's linear swing-mode shape).  Returns a new Model (state relative to the pivot)."""
    from dataclasses import replace
    W = (M.m + 0.5 * M.m_s) * M.g
    ell = M.L0 + W / M.k
    Mt = replace(M, c_s=0.0, c_x=0.0, c_r=0.0, X0=(0.0, 0.0, -ell - M.b), V0=(0.0, 0.0, 0.0), n0=(0.0, 0.0, -1.0), u0=(0.0, 0.0, 0.0))
    _, Vm, _ = sp.lateral_mode_shapes(Mt)
    kappa = float(Vm[1, 0] / Vm[0, 0])
    Tqs = W * math.cos(phi0) + (M.m + M.m_s / 3) * ell * phid0 ** 2
    de0 = (Tqs - W) / M.k
    X0, V0, n0, u0 = ic_from_planar(ell, M.b, phi0, kappa * phi0, de0, phid0, kappa * phid0, 0.0,
                                    phi_y0, kappa * phi_y0, phid_y0, kappa * phid_y0)
    return replace(M, X0=tuple(X0), V0=tuple(V0), n0=tuple(n0), u0=tuple(u0)), dict(ell_eq=ell, kappa=kappa, de0=de0)
