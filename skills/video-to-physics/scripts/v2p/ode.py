"""Time integrators and closed-form ODE references.

* `rk4` - classical fixed-step Runge-Kutta for y' = f(t, y).
* `generalized_alpha` - Chung-Hulbert generalized-alpha for  M(q) q'' + F(q, q', t) = 0  with Newton iterations
  (F = internal + damping - external forces, so a linear system is F = C v + K q - f(t)).
* `hermite_resample` - cubic Hermite interpolation of a (q, v) history onto arbitrary output times.
* closed forms used as benchmarks: damped linear oscillator, large-amplitude pendulum period (complete elliptic K).
numpy only; scipy is used for the elliptic integrals when present (AGM fallback otherwise).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np


# ----------------------------------------------------------------------------- explicit
def rk4_step(f: Callable, t: float, y, dt: float):
    """One classical RK4 step of y' = f(t, y)."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4(f: Callable, y0, t0: float, t1: float, dt: float):
    """Integrate y' = f(t, y) from t0 to t1 with fixed steps (last step shortened to land on t1).
    Returns (t (n,), y (n, *y0.shape))."""
    y = np.asarray(y0, float)
    n = max(1, int(math.ceil((t1 - t0) / dt - 1e-9)))
    ts = np.minimum(t0 + dt * np.arange(n + 1), t1)
    ys = [y]
    for k in range(n):
        y = rk4_step(f, ts[k], y, ts[k + 1] - ts[k])
        ys.append(y)
    return ts, np.array(ys)


# ----------------------------------------------------------------------------- generalized-alpha
def gen_alpha_params(rho_inf: float):
    """Chung-Hulbert parameters (alpha_m, alpha_f, gamma, beta) for spectral radius rho_inf in [0, 1]."""
    am = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0)
    af = rho_inf / (rho_inf + 1.0)
    gamma = 0.5 - am + af
    beta = 0.25 * (1.0 - am + af) ** 2
    return am, af, gamma, beta


def _fd_jac(fun, x, eps=1e-7):
    f0 = fun(x)
    J = np.empty((len(f0), len(x)))
    for i in range(len(x)):
        h = eps * max(1.0, abs(x[i]))
        xp = x.copy()
        xp[i] += h
        J[:, i] = (fun(xp) - f0) / h
    return J


def generalized_alpha(mass, force: Callable, q0, v0, dt: float, n_steps: int, t0: float = 0.0,
                      rho_inf: float = 0.9, stiffness: Callable | None = None, damping: Callable | None = None,
                      tol: float = 1e-10, max_iter: int = 30):
    """Integrate M(q) a + force(q, v, t) = 0 (Chung & Hulbert 1993).

    mass: (n, n) array or callable q -> (n, n).  force(q, v, t) -> (n,).
    stiffness(q, v, t) = d force/dq and damping(q, v, t) = d force/dv (finite differences when None).
    Balance is enforced at q_{n+1-af}, v_{n+1-af}, a_{n+1-am}, t_{n+1-af}. dM/dq is left out of the Newton
    tangent (quasi-Newton; converges for smooth problems).  rho_inf = 1: no numerical dissipation.
    Returns dict t, q, v, a (n_steps+1 rows) and iters (n_steps,).
    """
    am, af, gm, bt = gen_alpha_params(rho_inf)
    Mf = mass if callable(mass) else (lambda q, _M=np.asarray(mass, float): _M)
    Kf = stiffness or (lambda q, v, t: _fd_jac(lambda x: force(x, v, t), q))
    Cf = damping or (lambda q, v, t: _fd_jac(lambda x: force(q, x, t), v))
    q = np.array(q0, float)
    v = np.array(v0, float)
    t = float(t0)
    a = np.linalg.solve(Mf(q), -force(q, v, t))
    out = {"t": [t], "q": [q.copy()], "v": [v.copy()], "a": [a.copy()], "iters": []}
    for _ in range(n_steps):
        tm = t + (1.0 - af) * dt
        q1 = q + dt * v + 0.5 * dt * dt * a
        it = 0
        for it in range(1, max_iter + 1):
            a1 = (q1 - q - dt * v) / (bt * dt * dt) - (0.5 / bt - 1.0) * a
            v1 = v + dt * ((1.0 - gm) * a + gm * a1)
            qm = (1.0 - af) * q1 + af * q
            vm = (1.0 - af) * v1 + af * v
            M = Mf(qm)
            R = M @ ((1.0 - am) * a1 + am * a) + force(qm, vm, tm)
            J = (1.0 - am) / (bt * dt * dt) * M + (1.0 - af) * (gm / (bt * dt) * Cf(qm, vm, tm) + Kf(qm, vm, tm))
            d = np.linalg.solve(J, -R)
            q1 = q1 + d
            if np.linalg.norm(d) <= tol * max(1.0, np.linalg.norm(q1)):
                break
        a = (q1 - q - dt * v) / (bt * dt * dt) - (0.5 / bt - 1.0) * a
        v = v + dt * ((1.0 - gm) * out["a"][-1] + gm * a)
        q, t = q1, t + dt
        for k, val in (("t", t), ("q", q.copy()), ("v", v.copy()), ("a", a.copy()), ("iters", it)):
            out[k].append(val)
    return {k: np.array(val) for k, val in out.items()}


def linear_force(C, K, f: Callable | None = None):
    """force(q, v, t) = C v + K q - f(t) plus its constant tangents, for `generalized_alpha`."""
    C = np.atleast_2d(np.asarray(C, float))
    K = np.atleast_2d(np.asarray(K, float))
    fext = f or (lambda t: np.zeros(K.shape[0]))
    return (lambda q, v, t: C @ v + K @ q - fext(t)), (lambda q, v, t: K), (lambda q, v, t: C)


def hermite_resample(t, q, v, t_out):
    """Cubic Hermite interpolation of positions q (n, ...) with velocities v onto t_out; also linear v."""
    t = np.asarray(t, float)
    t_out = np.atleast_1d(np.asarray(t_out, float))
    k = np.clip(np.searchsorted(t, t_out, side="right") - 1, 0, len(t) - 2)
    h = t[k + 1] - t[k]
    s = ((t_out - t[k]) / h).reshape((-1,) + (1,) * (q.ndim - 1))
    hh = h.reshape(s.shape)
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    qi = h00 * q[k] + h10 * hh * v[k] + h01 * q[k + 1] + h11 * hh * v[k + 1]
    vi = (1 - s) * v[k] + s * v[k + 1]
    return qi, vi


# ----------------------------------------------------------------------------- closed forms
def ellipke_agm(m: float):
    """Complete elliptic integrals K(m), E(m) by the arithmetic-geometric mean (scipy-free fallback)."""
    a, b, c2sum, p = 1.0, math.sqrt(1.0 - m), 0.5 * m, 0.5
    while abs(a - b) > 1e-16 * a:
        c = 0.5 * (a - b)
        a, b = 0.5 * (a + b), math.sqrt(a * b)
        p *= 2.0
        c2sum += p * c * c
    K = math.pi / (2.0 * a)
    return K, K * (1.0 - c2sum)


def ellipke(m: float):
    """Complete elliptic integrals K(m), E(m) (parameter m = k^2); scipy when present, AGM otherwise."""
    try:
        from scipy.special import ellipk, ellipe
    except ImportError:
        return ellipke_agm(m)
    return float(ellipk(m)), float(ellipe(m))


def damped_oscillator(t, x0: float, v0: float, omega_n: float, zeta: float):
    """Exact x(t) of x'' + 2 zeta wn x' + wn^2 x = 0 (any zeta >= 0, zeta != 1)."""
    t = np.asarray(t, float)
    if zeta < 1.0:
        wd = omega_n * math.sqrt(1.0 - zeta**2)
        B = (v0 + zeta * omega_n * x0) / wd
        return np.exp(-zeta * omega_n * t) * (x0 * np.cos(wd * t) + B * np.sin(wd * t))
    r = omega_n * math.sqrt(zeta**2 - 1.0)
    l1, l2 = -zeta * omega_n + r, -zeta * omega_n - r
    c2 = (v0 - l1 * x0) / (l2 - l1)
    return (x0 - c2) * np.exp(l1 * t) + c2 * np.exp(l2 * t)


def pendulum_period(theta0: float, length: float, g: float = 9.81):
    """Period of a simple pendulum released from rest at amplitude theta0 (rad): 4 sqrt(L/g) K(sin^2(theta0/2))."""
    return 4.0 * math.sqrt(length / g) * ellipke(math.sin(0.5 * theta0) ** 2)[0]
