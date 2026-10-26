"""Tests for v2p.checks on synthetic data: convergence order/GCI, energy balance, periodicity, Strouhal,
divergence, tracer budget, dye-advection scale and the negative-control harness."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills" / "video-to-physics" / "scripts"))
from v2p import checks as C  # noqa: E402


def test_observed_order_and_richardson():
    f = lambda h: 1.0 + 0.3 * h**2 + 0.05 * h**3
    hs = (0.4, 0.2, 0.1)
    p = C.observed_order(f(0.1), f(0.2), f(0.4), r=2.0)
    assert abs(p - 2) < 0.15, p
    assert abs(C.richardson(f(0.1), f(0.2), 2.0, 2.0) - 1.0) < 1e-4
    # non-constant refinement ratio
    f1 = lambda h: 2.0 - 0.7 * h**2
    p = C.observed_order(f1(0.1), f1(0.15), f1(0.3), r=1.5, r32=2.0)
    assert abs(p - 2) < 1e-6, p
    assert C.check_convergence([f(h) for h in hs])["passed"]


def test_gci():
    f = lambda h: 1.0 + 0.3 * h**2
    g = C.gci(f(0.1), f(0.2), f(0.4), r=2.0)
    assert abs(g["p"] - 2) < 1e-9 and abs(g["f_ext"] - 1.0) < 1e-12
    e21 = abs(f(0.2) - f(0.1)) / f(0.1)
    assert abs(g["gci_fine"] - 1.25 * e21 / 3.0) < 1e-15
    assert abs(g["asymptotic_ratio"] - 1.0) < 0.01


def test_energy_balance():
    t = np.linspace(0, 5, 2001)
    c, k = 0.3, 4.0
    x = np.exp(-0.5 * c * t) * np.cos(math.sqrt(k - c * c / 4) * t)
    v = np.gradient(x, t, edge_order=2)
    E = 0.5 * v**2 + 0.5 * k * x**2
    assert C.energy_balance(t, E, P_diss=c * v**2, tol=1e-3)["passed"]
    assert not C.energy_balance(t, E, P_diss=0.5 * c * v**2, tol=1e-3)["passed"]


def test_periodicity_per_band():
    T = 0.8
    t = np.arange(0, 8 * T, T / 40)
    x = np.linspace(0, 1, 50)
    y = np.linspace(-0.2, 0.2, 9)
    ph = 2 * np.pi * (x[None, None, :] - t[:, None, None] / T)
    u = np.sin(ph) * np.ones((1, len(y), 1))
    v = 0.5 * np.cos(ph) * np.ones((1, len(y), 1))
    bands = {"near": (0, 0.3), "far": (0.3, 1.0)}
    edges = C.period_edges(t, u[:, 4, 0])
    per = C.band_fluctuation_energy(t, x, u, v, bands, edges)
    assert C.periodicity_per_band(per, tol=0.01)["passed"]
    grow = np.exp(0.05 * t)[:, None, None]
    per = C.band_fluctuation_energy(t, x, u * grow, v * grow, bands, edges)
    assert not C.periodicity_per_band(per, tol=0.01)["passed"]


def test_strouhal_noisy_sine():
    rng = np.random.default_rng(1)
    f, D, U = 1.37, 0.01, 0.06
    t = np.arange(0, 6.3, 1 / 60)
    s = 0.8 * np.sin(2 * np.pi * f * t + 0.4) + 0.2 + 0.15 * rng.standard_normal(len(t))
    st = C.strouhal(t, s, D, U)
    assert abs(st["St_fit"] / (f * D / U) - 1) < 0.005, st
    assert abs(st["St_fft"] / (f * D / U) - 1) < 0.02 and abs(st["St_zero"] / (f * D / U) - 1) < 0.05


def test_divergence_ratio():
    x = np.linspace(0, 1, 81)
    y = np.linspace(0, 1, 61)
    X, Y = np.meshgrid(x, y)
    psi_u = np.pi * np.sin(np.pi * X) * np.cos(np.pi * Y)       # u = dpsi/dy, v = -dpsi/dx, psi = sin sin
    psi_v = -np.pi * np.cos(np.pi * X) * np.sin(np.pi * Y)
    assert C.divergence_ratio(x, y, psi_u, psi_v) < 0.01
    assert C.divergence_ratio(x, y, X, Y + 0.1 * np.sin(5 * X)) > 1.0
    assert not C.check_divergence(x, y, psi_u + 3.0 * X, psi_v)["passed"]


def test_tracer_mass_budget():
    t = np.linspace(0, 10, 101)
    out = np.where(t > 4, 0.3, 0.0)
    mass = 1.0 * t - C.cumulative_trapezoid(t, out)
    assert C.tracer_mass_budget(t, mass, outflux=out, tol=0.01)["passed"]
    assert C.tracer_mass_budget(t, mass, outflux=out, injected=t, tol=0.01)["passed"]
    assert not C.tracer_mass_budget(t, 0.9 * mass, outflux=out, injected=t, tol=0.01)["passed"]


def test_dye_advection_scale():
    xs = np.linspace(0, 1, 200)
    ys = np.linspace(0, 0.5, 100)
    X, Y = np.meshgrid(xs, ys)
    U = 0.2
    blob = lambda x0: np.exp(-((X - x0) ** 2 + (Y - 0.25) ** 2) / (2 * 0.04**2))
    vel = C.grid_velocity([0.0, 1.0], xs, ys, np.full((2, 100, 200), U), np.zeros((2, 100, 200)))
    r = C.check_dye_advection(blob(0.3), blob(0.3 + U * 1.0), xs, ys, vel, 0.0, 1.0)
    assert r["passed"] and abs(r["value"]["best_scale"] - 1) < 0.03, r["value"]
    assert not C.check_dye_advection(blob(0.3), blob(0.3 + 1.4 * U), xs, ys, vel, 0.0, 1.0)["passed"]


def test_negative_control_catches_10pct_frequency_error():
    t = np.arange(0, 10, 0.01)
    s = np.sin(2 * np.pi * 2.0 * t) + 0.05 * np.random.default_rng(0).standard_normal(len(t))
    check = lambda t, s: C.check_frequency(t, s, f_ref=2.0, tol=0.02)
    res = C.run_negative_controls(check, dict(t=t, s=s), [("scale", dict(key="t", factor=1.1)),
                                                           ("scale", dict(key="t", factor=1.005), False)])
    assert all(r["passed"] for r in res), res
    # a harness that would wrongly accept: a perturbation too small to matter must not count as caught
    r = C.negative_control(check, dict(t=t, s=s), "scale", key="t", factor=1.001)
    assert not r["passed"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
