"""Independent reference (role E) for the ruler-buckling simulation.

Planar inextensible, unshearable Kirchhoff rod (elastica), clamped at s=0 (vertical), free at s=L,
self-weight q = rho*A*g per unit length, rigid eccentric tip body (platform + peg + n slotted weights)
whose total dead weight W acts at its CoM, offset (d lateral, e along the tip tangent) from the rod tip.

theta(s): angle of the tangent from +Z (vertical); y' = sin(theta), z' = cos(theta).
Moment balance (dead vertical loads):
    EI theta'' = -(W + q (L - s)) sin(theta),     theta(0) = 0,
    EI theta'(L) = W (e sin(theta_L) + d cos(theta_L))           (tip body moment).
Linearised (bifurcation, d = 0):  EI phi'' + (W + q(L-s)) phi = 0, phi(0)=0, EI phi'(L) = W e phi(L).

Written from first principles; does not use or read anything in solver/.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import ellipk, ellipe, jv
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[2]
G = 9.80665
RTOL, ATOL = 1e-13, 1e-15


@dataclass
class Rod:
    L: float = 0.268  # declared flexible length (jaw line -> tab bottom)
    b: float = 0.025
    h: float = 0.0005
    E: float = 193e9
    nu: float = 0.29
    rho: float = 7900.0
    g: float = G
    plate_modulus: bool = False  # True -> E/(1-nu^2) (anticlastic-suppressed, wide-plate bound)

    @property
    def A(self):
        return self.b * self.h

    @property
    def I(self):
        return self.b * self.h ** 3 / 12.0

    @property
    def EI(self):
        Eeff = self.E / (1 - self.nu ** 2) if self.plate_modulus else self.E
        return Eeff * self.I

    @property
    def q(self):  # self-weight per length [N/m]
        return self.rho * self.A * self.g

    @property
    def euler(self):  # pi^2 EI / (4 L^2)
        return math.pi ** 2 * self.EI / (4 * self.L ** 2)


@dataclass
class TipBody:
    """Rigid tip body: list of (name, mass kg, CoM lateral offset d [m], CoM height above tip e [m])."""
    parts: list = field(default_factory=list)

    @property
    def M(self):
        return sum(p[1] for p in self.parts)

    @property
    def e(self):
        return sum(p[1] * p[3] for p in self.parts) / self.M if self.M else 0.0

    @property
    def d(self):
        return sum(p[1] * p[2] for p in self.parts) / self.M if self.M else 0.0

    @property
    def J_c(self):  # in-plane rotary inertia about CoM, point-mass parts + thin-disc/rod terms neglected (<1e-3 effect on statics)
        return sum(p[1] * ((p[2] - self.d) ** 2 + (p[3] - self.e) ** 2) for p in self.parts)


# Declared configuration (lead message 2026-10-26, physics/FORMULATION.md geometry; masses re-derived here from geometry):
# vise jaw line -> flexible strip L_f = 0.268 m -> tab bottom O. Top 12 mm of the strip clamped between two symmetric Al tabs.
DECLARED = dict(L_f=0.268, tab_len=0.012, peg_h=0.040, peg_d=0.004, screws_kg=0.00048, m0_override=None)


def scene_tipbody(n_weights: int, sp: dict | None = None, include_tab_offset: bool = False, cfg: dict | None = None) -> TipBody:
    """Rigid tip body about O (strip mid-surface at the tab bottom, s = L_f): plate + 2 tabs + strip-in-tabs + screws + peg
    + n weights. Heights e measured along the tip tangent from O. Two symmetric tabs -> no lateral CoM offset (d = 0);
    include_tab_offset=True reproduces the old single-tab layout (d = -(h/2 + tab_t/2) for the tab mass)."""
    sp = sp or json.loads((ROOT / "config/scene_params.json").read_text())
    c = dict(DECLARED, **(cfg or {}))
    pl = sp["platform"]; ru = sp["ruler"]; sw = sp["slotted_weights"]
    h, b = ru["thickness_m"], ru["width_m"]
    rho_al, rho_ss = 2700.0, 7900.0
    px, py, pz = pl["plate_size_m"]
    tx, ty, tz = pl["tab_size_m"]
    lt = c["tab_len"]
    m_tab = rho_al * tx * ty * tz
    m_peg = rho_ss * math.pi * (c["peg_d"] / 2) ** 2 * c["peg_h"]
    m_strip_top = ru["material_fea"]["rho_kg_m3"] * b * h * lt
    parts = [("plate", rho_al * px * py * pz, 0.0, lt + pz / 2),
             ("tab_front", m_tab, -(h / 2 + ty / 2) if include_tab_offset else 0.0, tz / 2),
             ("strip_in_tabs", m_strip_top, 0.0, lt / 2),
             ("screws", c["screws_kg"], 0.0, lt - 0.006),
             ("peg", m_peg, 0.0, lt + pz + c["peg_h"] / 2)]
    if not include_tab_offset:
        parts.append(("tab_back", m_tab, 0.0, tz / 2))
    if c["m0_override"]:  # scale the fixed parts to a declared m0, keeping their CoM
        f = c["m0_override"] / sum(p[1] for p in parts)
        parts = [(nm, m * f, dd, ee) for nm, m, dd, ee in parts]
    t = sw["thickness_m"]
    for k in range(n_weights):
        parts.append((f"w{k+1}", sw["mass_each_kg"], 0.0, lt + pz + (k + 0.5) * t))
    return TipBody(parts)


# ---------------------------------------------------------------- nonlinear shooting
def _rhs(rod: Rod, W: float):
    EI, q, L = rod.EI, rod.q, rod.L

    def f(s, u):
        th, k, y, z = u
        return [k, -(W + q * (L - s)) * math.sin(th) / EI, math.sin(th), math.cos(th)]
    return f


def shoot(rod: Rod, W: float, c: float, n_out: int | None = None):
    """Integrate from the clamp with theta'(0)=c. Returns (residual, sol)."""
    s_eval = np.linspace(0, rod.L, n_out) if n_out else None
    sol = solve_ivp(_rhs(rod, W), (0, rod.L), [0.0, c, 0.0, 0.0], method="DOP853",
                    rtol=RTOL, atol=ATOL, t_eval=s_eval, dense_output=False)
    return sol


def tip_residual(rod: Rod, W: float, c: float, e: float, d: float):
    sol = shoot(rod, W, c)
    thL, kL = sol.y[0, -1], sol.y[1, -1]
    return rod.EI * kL - W * (e * math.sin(thL) + d * math.cos(thL))


def equilibrium(rod: Rod, W: float, e: float, d: float, c_hint: float | None = None, n_out=301,
                branch: str = "positive"):
    """Solve the BVP for theta'(0). branch='positive' picks the non-trivial root with c>0 (ideal case)
    or the root continuously connected to the imperfection direction (d != 0)."""
    f = lambda c: tip_residual(rod, W, c, e, d)
    c_scale = 2.0 * math.pi / rod.L  # theta'(0) bounded by ~ this for |theta| < pi
    if d != 0.0:
        sgn = -math.copysign(1.0, d)  # load CoM on -Y pushes the tip to -Y... resolved numerically below
        # scan both signs, take the root of smallest |theta_L| on the side favoured by the imperfection
        roots = []
        grid = np.linspace(-c_scale, c_scale, 801)
        vals = np.array([f(c) for c in grid])
        for i in range(len(grid) - 1):
            if np.sign(vals[i]) != np.sign(vals[i + 1]):
                roots.append(brentq(f, grid[i], grid[i + 1], xtol=1e-15, rtol=1e-14))
        # the stable (natural) path: same sign as the imperfection moment, largest |c| <= first beyond 0
        same = [r for r in roots if np.sign(r) == np.sign(d)]
        c = min(same, key=abs) if same else min(roots, key=abs)
        # in the post-buckled range the natural branch is the large-deflection root on the favoured side
        if len(same) > 1:
            c = max(same, key=abs)
    else:
        grid = np.linspace(1e-9 * c_scale, c_scale, 400)
        vals = np.array([f(cc) for cc in grid])
        idx = [i for i in range(len(grid) - 1) if np.sign(vals[i]) != np.sign(vals[i + 1])]
        if not idx:
            c = 0.0
        else:
            i = idx[0]
            c = brentq(f, grid[i], grid[i + 1], xtol=1e-16, rtol=1e-15)
    sol = shoot(rod, W, c, n_out)
    return dict(c=c, s=sol.t, theta=sol.y[0], kappa=sol.y[1], y=sol.y[2], z=sol.y[3],
                tip_y=sol.y[2, -1], tip_z=sol.y[3, -1], theta_L=sol.y[0, -1])


# ---------------------------------------------------------------- linear bifurcation (exact)
def lin_residual(rod: Rod, W: float, e: float):
    EI, q, L = rod.EI, rod.q, rod.L

    def f(s, u):
        return [u[1], -(W + q * (L - s)) * u[0] / EI]
    sol = solve_ivp(f, (0, L), [0.0, 1.0], method="DOP853", rtol=RTOL, atol=1e-18)
    phi, dphi = sol.y[:, -1]
    return EI * dphi - W * e * phi


def critical_load(rod: Rod, e: float = 0.0) -> float:
    """Lowest tip dead load W (N) at bifurcation (d = 0), self-weight included."""
    Wmax = 1.2 * rod.euler
    # phi(0)=0, phi'(0)=1 gives phi = sin-like; the residual starts positive (W->0) and changes sign at Wcr
    grid = np.linspace(1e-6 * Wmax, Wmax, 241)
    vals = [lin_residual(rod, W, e) for W in grid]
    for i in range(len(grid) - 1):
        if np.sign(vals[i]) != np.sign(vals[i + 1]):
            return brentq(lambda W: lin_residual(rod, W, e), grid[i], grid[i + 1], xtol=1e-15, rtol=1e-15)
    raise RuntimeError("no bifurcation below 1.2 Euler")


def critical_tip_mass_curve(rod: Rod, sp=None, n_max=12, include_tab=False, cfg=None):
    """For each weight count n: total tip weight W(n), its CoM height e(n) and Wcr(e(n))."""
    rows = []
    for n in range(0, n_max + 1):
        tb = scene_tipbody(n, sp, include_tab_offset=include_tab, cfg=cfg)
        W = tb.M * rod.g
        Wcr = critical_load(rod, tb.e)
        rows.append(dict(n=n, M_tip_kg=tb.M, W_N=W, e_m=tb.e, d_m=tb.d, Wcr_N=Wcr, ratio=W / Wcr))
    return rows


# ---------------------------------------------------------------- closed forms
def euler_closed(rod: Rod):
    return math.pi ** 2 * rod.EI / (4 * rod.L ** 2)


def eccentric_closed(rod: Rod, e: float):
    """q=0: tan(kL) = 1/(k e)  ->  W = EI k^2."""
    L = rod.L
    x = brentq(lambda x: math.cos(x) - x * (e / L) * math.sin(x), 1e-9, math.pi / 2, xtol=1e-16, rtol=1e-15)
    return rod.EI * (x / L) ** 2


def greenhill_qcr(rod: Rod):
    """Heavy clamped-free column: (q L^3/EI)_cr = (3/2 j)^2, j = first zero of J_{-1/3}."""
    j = brentq(lambda x: jv(-1.0 / 3.0, x), 1.0, 2.5, xtol=1e-16, rtol=1e-15)
    return (1.5 * j) ** 2 * rod.EI / rod.L ** 3


def elliptic_tip(rod: Rod, alpha: float):
    """Ideal elastica (q=0, e=d=0), tip angle alpha. Returns W, tip lateral y, tip height z."""
    p = math.sin(alpha / 2)
    m = p * p
    K, Ee = ellipk(m), ellipe(m)
    k = K / rod.L
    return rod.EI * k * k, 2 * p / k, (2 * Ee - K) / k


# ---------------------------------------------------------------- vibration (Hermite FE, independent)
def beam_fe(rod: Rod, ne: int, W: float = 0.0, M_tip=0.0, e=0.0, J_c=0.0, with_q=True):
    """Small lateral vibrations about the straight state with axial compression N(s)=W+q(L-s).
    Tip rigid body: mass M_tip, CoM at height e along the tangent, rotary inertia J_c about its CoM.
    Returns K, Kg, Mm (global, clamped dofs removed)."""
    L, EI, m = rod.L, rod.EI, rod.rho * rod.A
    q = rod.q if with_q else 0.0
    le = L / ne
    nd = 2 * (ne + 1)
    K = np.zeros((nd, nd)); Kg = np.zeros((nd, nd)); Mm = np.zeros((nd, nd))
    ke = EI / le ** 3 * np.array([[12, 6 * le, -12, 6 * le], [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
                                  [-12, -6 * le, 12, -6 * le], [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2]])
    me = m * le / 420 * np.array([[156, 22 * le, 54, -13 * le], [22 * le, 4 * le ** 2, 13 * le, -3 * le ** 2],
                                  [54, 13 * le, 156, -22 * le], [-13 * le, -3 * le ** 2, -22 * le, 4 * le ** 2]])
    # geometric stiffness with linearly varying N: Gauss quadrature of N * H' H'^T
    gp, gw = np.polynomial.legendre.leggauss(6)
    for i in range(ne):
        kg = np.zeros((4, 4))
        for xi, w in zip(gp, gw):
            x = (xi + 1) / 2
            s = (i + x) * le
            N = W + q * (L - s)
            dH = np.array([(-6 * x + 6 * x ** 2) / le, 1 - 4 * x + 3 * x ** 2, (6 * x - 6 * x ** 2) / le, -2 * x + 3 * x ** 2])
            kg += N * np.outer(dH, dH) * w * le / 2
        sl = slice(2 * i, 2 * i + 4)
        K[sl, sl] += ke; Mm[sl, sl] += me; Kg[sl, sl] += kg
    # tip body: CoM lateral = w_L + e th_L ; KE = 1/2 M (w+e th)^2 + 1/2 J th^2 ; load drop 1/2 W e th^2
    a, bdof = nd - 2, nd - 1
    T = np.array([1.0, e])
    Mm[np.ix_([a, bdof], [a, bdof])] += M_tip * np.outer(T, T)
    Mm[bdof, bdof] += J_c
    Kg[bdof, bdof] += W * e
    keep = np.arange(2, nd)
    # conditioning: rotation dofs scaled by le (theta = w~/le), then symmetric diagonal equilibration
    S = np.ones(nd); S[1::2] = 1.0 / le
    K, Kg, Mm = (S[:, None] * X * S[None, :] for X in (K, Kg, Mm))
    K, Kg, Mm = (X[np.ix_(keep, keep)] for X in (K, Kg, Mm))
    D = 1.0 / np.sqrt(np.diag(K))
    return tuple(D[:, None] * X * D[None, :] for X in (K, Kg, Mm))


def fe_frequency(rod: Rod, ne=30, **kw):
    K, Kg, Mm = beam_fe(rod, ne, **kw)
    w2 = eigh(K - Kg, Mm, eigvals_only=True, subset_by_index=[0, 0])[0]
    return math.copysign(math.sqrt(abs(w2)), w2) / (2 * math.pi)


def fe_critical(rod: Rod, ne=30, e=0.0):
    """Buckling eigenvalue from FE: (K - lambda Kg(W=1, incl. q as fixed)) -- here W unknown, q fixed:
    solve det(K - Kg(W)) = 0 by bisection on the lowest eigenvalue of K - Kg(W) w.r.t. identity."""
    def lam(W):
        K, Kg, Mm = beam_fe(rod, ne, W=W, e=e)
        return eigh(K - Kg, Mm, eigvals_only=True, subset_by_index=[0, 0])[0]
    return brentq(lam, 0.5 * rod.euler, 1.2 * rod.euler, xtol=1e-14, rtol=1e-14)


def tipmass_freq_closed(rod: Rod, M_tip: float):
    """Clamped-free Euler-Bernoulli beam, point tip mass (no rotary inertia, no gravity):
    1 + cos l cosh l + mu l (cos l sinh l - sin l cosh l) = 0, mu = M_tip/(m L)."""
    mu = M_tip / (rod.rho * rod.A * rod.L)
    f = lambda l: 1 + math.cos(l) * math.cosh(l) + mu * l * (math.cos(l) * math.sinh(l) - math.sin(l) * math.cosh(l))
    lam = brentq(f, 1e-6 if mu > 0 else 1.0, 1.875105 + 1e-6, xtol=1e-15, rtol=1e-15)
    return lam ** 2 * math.sqrt(rod.EI / (rod.rho * rod.A * rod.L ** 4)) / (2 * math.pi)
