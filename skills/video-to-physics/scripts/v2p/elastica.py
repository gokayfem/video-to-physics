"""Planar elastica: closed forms, a discrete Kirchhoff rod (statics + dynamics) and centreline -> strip grids.

Conventions: the rod is clamped at s = 0 pointing along +axial ("up"), angles theta(s) are measured from the
clamp direction towards +lateral. Loads are dead vertical (along -axial) forces; compression is positive.

Closed forms (benchmarks):  `euler_clamped_free`, `eccentric_critical_load` (kL tan kL = L/e),
`greenhill_critical_q` (heavy column), `heavy_column_critical_load` (shooting, self-weight + eccentric tip load),
`elliptic_tip` / `elliptic_alpha` (large-deflection clamped-free elastica with a tip load).

Discrete model (`DiscreteRod`): generalised coordinates = tangent angles theta_i at s_i = i h (i = 1..n, h = L/n,
theta_0 = 0 at the clamp). Kinematics: inextensible chain [clamp stub h/2 at angle 0, links of length h at
theta_1..theta_{n-1}, half link h/2 at theta_n], so theta_n is exactly the tip angle. Bending energy
U = sum_i EI/(2h) (theta_i - theta_{i-1} - (theta0_i - theta0_{i-1}))^2. Rod mass lumped to the link ends.
Second-order accurate in h (Euler, eccentric, Greenhill, post-buckled elastica). A rigid tip body (mass, CoM offset
e along the tip tangent, rotary inertia J) carries an extra dead load P (load parameter / time-dependent load).
Statics: Newton, linear buckling eigenproblem, tip-angle-controlled path following (passes the bifurcation by
starting on the buckling mode). Dynamics: exact point-mass Lagrangian + generalised-alpha (v2p.ode) with Rayleigh
damping C = alpha_M M + beta_K K_b and an energy ledger E - E0 - W_ext + W_diss.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import ode

G0 = 9.81


def _bisect(f: Callable, a: float, b: float, tol: float = 1e-15, it: int = 200):
    fa = f(a)
    for _ in range(it):
        m = 0.5 * (a + b)
        fm = f(m)
        if (fm > 0) == (fa > 0):
            a, fa = m, fm
        else:
            b = m
        if b - a <= tol * max(1.0, abs(m)):
            break
    return 0.5 * (a + b)


def _root(f: Callable, a: float, b: float):
    try:
        from scipy.optimize import brentq
        return brentq(f, a, b, xtol=1e-15, rtol=1e-15, maxiter=200)
    except ImportError:
        return _bisect(f, a, b)


# ============================================================================ closed forms
def euler_clamped_free(EI: float, L: float) -> float:
    """Euler critical tip load of a clamped-free column: pi^2 EI / (4 L^2)."""
    return math.pi**2 * EI / (4.0 * L**2)


def eccentric_critical_load(EI: float, L: float, e: float) -> float:
    """Clamped-free column, weightless, dead load applied at a rigid offset e along the tip tangent:
    kL tan(kL) = L/e, P = EI k^2 (e = 0 -> Euler)."""
    if e <= 0:
        return euler_clamped_free(EI, L)
    x = _root(lambda x: math.cos(x) - x * (e / L) * math.sin(x), 1e-12, math.pi / 2)
    return EI * (x / L) ** 2


def greenhill_critical_q(EI: float, L: float) -> float:
    """Self-weight per length q_cr of a heavy clamped-free column: (q L^3/EI)_cr = (3 j/2)^2, j = 1st zero of J_-1/3."""
    try:
        from scipy.special import jv
        j = _root(lambda x: jv(-1.0 / 3.0, x), 1.0, 2.5)
        c = (1.5 * j) ** 2
    except ImportError:
        c = 7.837347
    return c * EI / L**3


def _lin_shoot_residual(EI, L, q, e, P, n_steps=400):
    f = lambda s, y: np.array([y[1], -(P + q * (L - s)) * y[0] / EI])
    _, y = ode.rk4(f, np.array([0.0, 1.0]), 0.0, L, L / n_steps)
    phi, dphi = y[-1]
    return EI * dphi - P * e * phi


def heavy_column_critical_load(EI: float, L: float, q: float = 0.0, e: float = 0.0, n_steps: int = 400) -> float:
    """Critical dead tip load (applied at offset e along the tip tangent) of a clamped-free column with self-weight q
    per length, by shooting on EI phi'' + (P + q (L - s)) phi = 0, phi(0) = 0, EI phi'(L) = P e phi(L).
    Negative if self-weight alone exceeds Greenhill."""
    Pe = euler_clamped_free(EI, L)
    r = lambda P: _lin_shoot_residual(EI, L, q, e, P, n_steps)
    grid = np.linspace(-q * L, 1.05 * Pe, 64)
    vals = [r(P) for P in grid]
    for i in range(len(grid) - 1):
        if np.sign(vals[i]) != np.sign(vals[i + 1]):
            return _root(r, grid[i], grid[i + 1])
    raise RuntimeError("no bifurcation found in [-qL, 1.05 P_euler]")


def elliptic_tip(EI: float, L: float, alpha: float):
    """Inextensible clamped-free elastica, weightless, vertical dead tip load, tip angle alpha (rad, 0 < alpha < pi).
    Returns (P, tip_lateral, tip_axial):  p = sin(alpha/2), k = K(p^2)/L, P = EI k^2, lateral 2p/k, axial (2E - K)/k."""
    p = math.sin(0.5 * alpha)
    K, E = ode.ellipke(p * p)
    k = K / L
    return EI * k * k, 2.0 * p / k, (2.0 * E - K) / k


def elliptic_alpha(EI: float, L: float, P: float) -> float:
    """Tip angle of the post-buckled elastica for tip load P (> Euler)."""
    if P <= euler_clamped_free(EI, L):
        return 0.0
    return _root(lambda a: elliptic_tip(EI, L, a)[0] - P, 1e-9, math.pi - 1e-9)


# ============================================================================ discrete rod
@dataclass
class TipBody:
    """Rigid body at the free end: mass (kg), CoM offset along the tip tangent (m), rotary inertia about CoM (kg m^2)."""
    mass: float = 0.0
    offset: float = 0.0
    inertia: float = 0.0


class DiscreteRod:
    """Discrete planar Kirchhoff rod clamped at s = 0 (see module docstring).

    EI (N m^2), L flexible length (m), n links, mass_per_length rho*A (kg/m), self_weight toggles its gravity,
    tip: TipBody, theta0: stress-free angles at s_i (n,) or callable s -> angle (imperfection / natural shape).
    Load parameter P: extra dead load (N, along -axial) at the tip-body CoM (tip node if offset = 0)."""

    def __init__(self, EI: float, L: float, n: int, mass_per_length: float = 0.0, g: float = G0,
                 self_weight: bool = True, tip: TipBody | None = None, theta0=None):
        self.EI, self.L, self.n, self.g = float(EI), float(L), int(n), float(g)
        self.h = h = self.L / n
        self.tip = tip or TipBody()
        self.s_nodes = np.arange(1, n + 1) * h                # theta_i = tangent angle at s_i = i h
        self.k = np.full(n, self.EI / h)
        self.D = np.eye(n) - np.eye(n, k=-1)
        self.Kb = self.D.T @ (self.k[:, None] * self.D)
        # chain: rigid clamp stub (h/2, angle 0), links 1..n-1 of length h at theta_1..theta_{n-1}, a half link h/2
        # at theta_n, the tip body (offset e) at theta_n; rod mass lumped to the chain nodes (link ends)
        self.l = np.append(np.full(n, h), self.tip.offset)
        self.l[n - 1] = 0.5 * h
        seg = np.append(np.full(n, h), 0.0)
        seg[n - 1] = 0.5 * h
        m = 0.5 * mass_per_length * seg[:n]
        m[:-1] += 0.5 * mass_per_length * seg[1:n]
        self.m_node = m                                       # mass at the end of chain link i
        self.m_grav = m if self_weight else np.zeros(n)
        if theta0 is None:
            th0, base = np.zeros(n), 0.0
        elif callable(theta0):
            th0, base = np.asarray(theta0(self.s_nodes), float), float(theta0(0.0))
        else:
            th0, base = np.asarray(theta0, float), 0.0
        self.theta0 = th0
        self.kappa0 = self.D @ th0
        self.kappa0[0] -= base
        self.c = np.zeros(n)                                  # tip-body angle = c . theta = theta_n
        self.c[-1] = 1.0

    # ---------------------------------------------------------------- kinematics (links + rigid tip link)
    def _links(self, th):
        return np.append(th, self.c @ th), self.l

    def _fold(self, v):
        """T^T v: (n+1) extended-link vector -> (n) generalised coordinates."""
        return v[:-1] + v[-1] * self.c

    def _fold2(self, A):
        """T^T A T for an (n+1, n+1) extended-link matrix."""
        out = A[:-1, :-1] + np.outer(A[:-1, -1], self.c) + np.outer(self.c, A[-1, :-1])
        return out + A[-1, -1] * np.outer(self.c, self.c)

    def _weights(self, P):
        w = np.append(self.g * self.m_grav, self.g * self.tip.mass + P)
        return np.cumsum(w[::-1])[::-1]

    def positions(self, th):
        """Chain node positions (n+2, 2) as (lateral, axial): clamp, s = h/2, 3h/2, ..., L - h/2, L; and tip-body CoM (2,)."""
        phi, l = self._links(th)
        x = np.concatenate([[0.0, 0.0], np.cumsum(l * np.sin(phi))])
        z = np.concatenate([[0.0, 0.5 * self.h], 0.5 * self.h + np.cumsum(l * np.cos(phi))])
        return np.stack([x[:-1], z[:-1]], 1), np.array([x[-1], z[-1]])

    def tip_axial_jacobian(self, th):
        phi, l = self._links(th)
        return self._fold(-l * np.sin(phi))

    # ---------------------------------------------------------------- energies and derivatives
    def U(self, th):
        r = self.D @ th - self.kappa0
        return 0.5 * float(np.sum(self.k * r * r))

    def grad_U(self, th):
        return self.D.T @ (self.k * (self.D @ th - self.kappa0))

    def V(self, th, P=0.0):
        """Gravity potential of nodes + tip body, plus P times tip-CoM height."""
        phi, l = self._links(th)
        z = np.cumsum(l * np.cos(phi))
        return float(self.g * np.sum(self.m_grav * z[:-1]) + (self.g * self.tip.mass + P) * z[-1])

    def grad_V(self, th, P=0.0):
        phi, l = self._links(th)
        return self._fold(-l * np.sin(phi) * self._weights(P))

    def hess_V(self, th, P=0.0):
        phi, l = self._links(th)
        return self._fold2(np.diag(-l * np.cos(phi) * self._weights(P)))

    def residual(self, th, P=0.0):
        return self.grad_U(th) + self.grad_V(th, P)

    def jacobian(self, th, P=0.0):
        return self.Kb + self.hess_V(th, P)

    def dresidual_dP(self, th):
        phi, l = self._links(th)
        return self._fold(-l * np.sin(phi))

    def mass_matrix(self, th):
        phi, l = self._links(th)
        mi = np.append(self.m_node, self.tip.mass)
        S = np.cumsum(mi[::-1])[::-1]
        idx = np.arange(len(phi))
        Mx = np.outer(l, l) * np.cos(phi[:, None] - phi[None, :]) * S[np.maximum.outer(idx, idx)]
        return self._fold2(Mx) + self.tip.inertia * np.outer(self.c, self.c)

    def gyroscopic(self, th, w):
        """Velocity-squared (centripetal) generalised force term of the chain Lagrangian."""
        phi, l = self._links(th)
        mi = np.append(self.m_node, self.tip.mass)
        S = np.cumsum(mi[::-1])[::-1]
        idx = np.arange(len(phi))
        Cx = np.outer(l, l) * np.sin(phi[:, None] - phi[None, :]) * S[np.maximum.outer(idx, idx)]
        return self._fold(Cx @ np.append(w, self.c @ w) ** 2)


# ============================================================================ statics
def solve_static(rod: DiscreteRod, P: float, theta=None, tol: float = 1e-11, max_iter: int = 60):
    """Newton equilibrium at load P from guess theta. Returns (theta, converged)."""
    th = np.zeros(rod.n) if theta is None else np.array(theta, float)
    scale = max(1.0, rod.EI / rod.L)
    for _ in range(max_iter):
        r = rod.residual(th, P)
        if np.linalg.norm(r) < tol * scale:
            return th, True
        th = th - np.linalg.solve(rod.jacobian(th, P), r)
    return th, bool(np.linalg.norm(rod.residual(th, P)) < 1e-8 * scale)


def critical_load(rod: DiscreteRod):
    """Linear buckling about the straight state: smallest P with K_b + H_V(0, P) singular.
    Returns (P_cr, mode) with mode[-1] = 1. P_cr is the extra load on top of the tip-body weight
    (negative when gravity alone already buckles the rod)."""
    z = np.zeros(rod.n)
    A = rod.Kb + rod.hess_V(z, 0.0)
    B = -(rod.hess_V(z, 1.0) - rod.hess_V(z, 0.0))                 # symmetric positive definite
    Li = np.linalg.inv(np.linalg.cholesky(B))
    lam, Y = np.linalg.eigh(Li @ A @ Li.T)
    mode = Li.T @ Y[:, 0]
    return float(lam[0]), mode / mode[-1]


def equilibrium_path(rod: DiscreteRod, tip_angles, P0: float | None = None, theta0=None,
                     tol: float = 1e-11, max_iter: int = 60):
    """Static path with the tip-link angle as control (unknowns theta, P; constraint theta_n = alpha).
    Works through the bifurcation of the perfect rod and around limit points in P. Without guesses the path
    starts on the buckling mode at P_cr (branch switch onto the post-buckled branch).
    Returns dict P (k,), theta (k, n), tip (k, 2) [tip-node lateral, axial], tip_com (k, 2)."""
    tip_angles = np.asarray(tip_angles, float)
    if theta0 is None or P0 is None:
        Pc, mode = critical_load(rod)
        th, P = mode * tip_angles[0], Pc
    else:
        th, P = np.array(theta0, float), float(P0)
    n = rod.n
    scale = max(1.0, rod.EI / rod.L)
    out = {"P": [], "theta": [], "tip": [], "tip_com": []}
    for al in tip_angles:
        for _ in range(max_iter):
            R = rod.residual(th, P)
            F = np.append(R, th[-1] - al)
            if np.linalg.norm(R) < tol * scale and abs(F[-1]) < 1e-13:
                break
            J = np.zeros((n + 1, n + 1))
            J[:n, :n] = rod.jacobian(th, P)
            J[:n, n] = rod.dresidual_dP(th)
            J[n, n - 1] = 1.0
            d = np.linalg.solve(J, -F)
            th, P = th + d[:n], P + d[n]
        else:
            raise RuntimeError(f"tip-angle control did not converge at alpha={al:g}")
        nodes, com = rod.positions(th)
        for k, v in (("P", P), ("theta", th.copy()), ("tip", nodes[-1]), ("tip_com", com)):
            out[k].append(v)
    return {k: np.array(v) for k, v in out.items()}


def solve_on_path(rod: DiscreteRod, path: dict, P: float):
    """Equilibrium at load P on a computed path branch (Newton from the nearest path point)."""
    i = int(np.argmin(np.abs(path["P"] - P)))
    th, ok = solve_static(rod, P, path["theta"][i])
    if not ok:
        raise RuntimeError(f"no equilibrium near the path at P={P:g}")
    return th


# ============================================================================ dynamics
def simulate(rod: DiscreteRod, t_end: float, dt: float, load: Callable = lambda t: 0.0, theta_init=None,
             omega_init=None, rho_inf: float = 0.9, alpha_M: float = 0.0, beta_K: float = 0.0, out_times=None):
    """Generalised-alpha dynamics with time-dependent tip load P(t) and Rayleigh damping alpha_M M + beta_K K_b.
    Returns dict with t, theta, omega (every step), energy ledger (T, U, V, W_ext, W_diss, residual = E - E0 -
    W_ext + W_diss) and, if out_times is given, Hermite-resampled theta_out/omega_out."""
    n = rod.n
    th0 = np.zeros(n) if theta_init is None else np.array(theta_init, float)
    w0 = np.zeros(n) if omega_init is None else np.array(omega_init, float)

    def damping_matrix(q):
        return alpha_M * rod.mass_matrix(q) + beta_K * rod.Kb

    def force(q, w, t):
        return rod.gyroscopic(q, w) + rod.residual(q, load(t)) + damping_matrix(q) @ w

    r = ode.generalized_alpha(rod.mass_matrix, force, th0, w0, dt, int(round(t_end / dt)), rho_inf=rho_inf,
                              stiffness=lambda q, w, t: rod.jacobian(q, load(t)),
                              damping=lambda q, w, t: damping_matrix(q))
    t, Q, W = r["t"], r["q"], r["v"]
    T = np.array([0.5 * w @ rod.mass_matrix(q) @ w for q, w in zip(Q, W)])
    U = np.array([rod.U(q) for q in Q])
    V = np.array([rod.V(q, 0.0) for q in Q])
    z_tip = np.array([rod.positions(q)[1][1] for q in Q])
    P = np.array([load(tt) for tt in t])
    pd = np.array([w @ damping_matrix(q) @ w for q, w in zip(Q, W)])
    W_ext = np.concatenate([[0.0], np.cumsum(-0.5 * (P[1:] + P[:-1]) * np.diff(z_tip))])
    W_diss = np.concatenate([[0.0], np.cumsum(0.5 * (pd[1:] + pd[:-1]) * np.diff(t))])
    E = T + U + V
    out = dict(t=t, theta=Q, omega=W, T=T, U=U, V=V, W_ext=W_ext, W_diss=W_diss, residual=E - E[0] - W_ext + W_diss,
               iters=r["iters"])
    if out_times is not None:
        out["t_out"] = np.asarray(out_times, float)
        out["theta_out"], out["omega_out"] = ode.hermite_resample(t, Q, W, out["t_out"])
    return out


# ============================================================================ geometry -> SPEC deformation grid
def centreline_from_angles(s, theta):
    """Integrate a centreline from tangent angles theta(s) (theta from +axial towards +lateral), clamp at the origin.
    s (m,), theta (..., m) -> (..., m, 2) (lateral, axial), trapezoidal in s."""
    s = np.asarray(s, float)
    th = np.asarray(theta, float)
    ds = np.diff(s)
    dx = 0.5 * (np.sin(th[..., 1:]) + np.sin(th[..., :-1])) * ds
    dz = 0.5 * (np.cos(th[..., 1:]) + np.cos(th[..., :-1])) * ds
    zero = np.zeros(th.shape[:-1] + (1,))
    return np.stack([np.concatenate([zero, np.cumsum(dx, -1)], -1), np.concatenate([zero, np.cumsum(dz, -1)], -1)], -1)


def rod_centrelines(rod: DiscreteRod, thetas, rigid_extension: float = 0.0):
    """Node centrelines (N, n+1(+1), 2) for a history of link angles; optionally append a rigid straight piece of
    length rigid_extension along the last link (e.g. a clamped-in tip part of a physical strip)."""
    out = []
    for th in np.atleast_2d(thetas):
        nodes, _ = rod.positions(th)
        if rigid_extension > 0:
            nodes = np.vstack([nodes, nodes[-1] + rigid_extension * np.array([math.sin(th[-1]), math.cos(th[-1])])])
        out.append(nodes)
    return np.array(out)


def centreline_to_grid(centrelines, width: float, cols: int = 5, rows: int | None = None, origin=(0.0, 0.0, 0.0),
                       lateral_axis=(0.0, 1.0, 0.0), axial_axis=(0.0, 0.0, 1.0), width_axis=None):
    """Planar centrelines (N, m, 2) [(lateral, axial) from the clamp] -> strip mid-surface positions
    (N, rows, cols, 3) in world metres (SPEC deformation npz grid). Rows follow the arc (resampled uniformly in
    arc length when rows is given), cols span the width centred on the centreline along width_axis
    (default lateral x axial)."""
    C = np.asarray(centrelines, float)
    if C.ndim == 2:
        C = C[None]
    ax_l, ax_a = np.asarray(lateral_axis, float), np.asarray(axial_axis, float)
    ax_w = np.cross(ax_l, ax_a) if width_axis is None else np.asarray(width_axis, float)
    if rows is not None:
        seg = np.linalg.norm(np.diff(C, axis=1), axis=-1)
        s = np.concatenate([np.zeros((len(C), 1)), np.cumsum(seg, 1)], 1)
        su = np.linspace(0, 1, rows)
        C = np.array([np.stack([np.interp(su * si[-1], si, c[:, k]) for k in (0, 1)], -1) for c, si in zip(C, s)])
    wv = np.linspace(-0.5 * width, 0.5 * width, cols)
    P = (np.asarray(origin, float) + C[..., 0, None] * ax_l + C[..., 1, None] * ax_a)[:, :, None, :]
    return P + wv[None, None, :, None] * ax_w


def write_deformation_npz(path, positions, t, **extra):
    """Write the SPEC deformation npz: positions (N, rows, cols, 3) world metres, t (N,) seconds, plus extras
    (e.g. thickness=..., per-sample arrays)."""
    positions = np.asarray(positions, float)
    t = np.asarray(t, float)
    assert positions.ndim == 4 and positions.shape[-1] == 3 and len(t) == len(positions)
    np.savez_compressed(path, positions=positions, t=t, **extra)
