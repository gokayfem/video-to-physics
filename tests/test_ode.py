"""Tests for v2p.ode: RK4 order, generalized-alpha vs closed forms (damped oscillator, large-amplitude pendulum)."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills" / "video-to-physics" / "scripts"))
from v2p import ode  # noqa: E402


def _zero_up_crossings(t, x):
    i = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
    return t[i] - x[i] * (t[i + 1] - t[i]) / (x[i + 1] - x[i])


def test_rk4_order4():
    f = lambda t, y: y * np.cos(t)
    errs = []
    for dt in (0.2, 0.1, 0.05):
        _, y = ode.rk4(f, np.array([1.0]), 0.0, 3.0, dt)
        errs.append(abs(y[-1, 0] - math.exp(math.sin(3.0))))
    p = math.log(errs[1] / errs[2]) / math.log(2.0)
    assert 3.8 < p < 4.2, p


def test_gen_alpha_linear_energy_conserved_rho1():
    wn = 2 * math.pi
    force, K, C = ode.linear_force([[0.0]], [[wn**2]])
    r = ode.generalized_alpha([[1.0]], force, [1.0], [0.0], dt=0.02, n_steps=500, rho_inf=1.0, stiffness=K, damping=C)
    E = 0.5 * r["v"][:, 0] ** 2 + 0.5 * wn**2 * r["q"][:, 0] ** 2
    assert np.max(np.abs(E / E[0] - 1)) < 1e-10


def test_gen_alpha_damped_oscillator_decay():
    wn, zeta = 2 * math.pi, 0.05
    force, K, C = ode.linear_force([[2 * zeta * wn]], [[wn**2]])
    dt = 1.0 / 400
    r = ode.generalized_alpha([[1.0]], force, [1.0], [0.0], dt=dt, n_steps=2000, rho_inf=1.0, stiffness=K, damping=C)
    x_ex = ode.damped_oscillator(r["t"], 1.0, 0.0, wn, zeta)
    assert np.max(np.abs(r["q"][:, 0] - x_ex)) < 2e-3
    # decay rate from successive positive peaks: ln(x_k/x_{k+1}) / T_d = zeta wn
    x = r["q"][:, 0]
    pk = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
    rate = np.polyfit(r["t"][pk], np.log(x[pk]), 1)[0]
    assert abs(-rate / (zeta * wn) - 1) < 0.01, rate


def test_pendulum_large_amplitude_period():
    g, L, th0 = 9.81, 1.0, 2.5
    T_ex = ode.pendulum_period(th0, L, g)
    # RK4
    t, y = ode.rk4(lambda t, y: np.array([y[1], -g / L * math.sin(y[0])]), np.array([th0, 0.0]), 0.0, 3.2 * T_ex, T_ex / 2000)
    tc = _zero_up_crossings(t, y[:, 1])
    assert abs(np.mean(np.diff(tc)) / T_ex - 1) < 1e-6
    # generalized-alpha (nonlinear stiffness, analytic tangent)
    force = lambda q, v, t: g / L * np.sin(q)
    r = ode.generalized_alpha([[1.0]], force, [th0], [0.0], dt=T_ex / 2000, n_steps=6400, rho_inf=1.0,
                              stiffness=lambda q, v, t: np.array([[g / L * math.cos(q[0])]]),
                              damping=lambda q, v, t: np.zeros((1, 1)))
    tc = _zero_up_crossings(r["t"], r["v"][:, 0])
    assert abs(np.mean(np.diff(tc)) / T_ex - 1) < 1e-4
    E = 0.5 * r["v"][:, 0] ** 2 + g / L * (1 - np.cos(r["q"][:, 0]))
    assert np.max(np.abs(E / E[0] - 1)) < 1e-4


def test_elliptic_fallback_matches_scipy():
    K, E = ode.ellipke_agm(0.7)
    assert abs(K - 2.0753631352924691) < 1e-12 and abs(E - 1.2416705679458233) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
