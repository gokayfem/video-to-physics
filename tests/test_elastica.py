"""Tests for v2p.elastica: closed forms vs the discrete rod (statics, convergence order), dynamics energy/damping,
centreline -> SPEC deformation grid."""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills" / "video-to-physics" / "scripts"))
from v2p import elastica as E  # noqa: E402

EI, L = 2.0, 0.3


def _order(v, exact, r=2.0):
    e = [abs(x - exact) for x in v]
    return math.log(e[-2] / e[-1]) / math.log(r)


def test_euler_critical_load_and_order():
    Pe = E.euler_clamped_free(EI, L)
    vals = [E.critical_load(E.DiscreteRod(EI, L, n))[0] for n in (20, 40, 80)]
    assert abs(vals[-1] / Pe - 1) < 1e-3, vals[-1] / Pe - 1
    p_obs = math.log(abs(vals[0] - vals[1]) / abs(vals[1] - vals[2])) / math.log(2)
    assert abs(p_obs - 2) < 0.05, p_obs


def test_eccentric_closed_form():
    e = 0.05
    Pa = E.eccentric_critical_load(EI, L, e)
    kL = math.sqrt(Pa / EI) * L
    assert abs(kL * math.tan(kL) - L / e) < 1e-9
    vals = [E.critical_load(E.DiscreteRod(EI, L, n, tip=E.TipBody(0.0, e)))[0] for n in (20, 40, 80)]
    assert abs(vals[-1] / Pa - 1) < 1e-3 and abs(_order(vals, Pa) - 2) < 0.1


def test_greenhill_and_heavy_column():
    q = E.greenhill_critical_q(EI, L)
    assert abs(q * L**3 / EI - 7.8373) < 1e-3
    rod = E.DiscreteRod(EI, L, 80, mass_per_length=q / E.G0)
    assert abs(E.critical_load(rod)[0]) < 1e-3 * E.euler_clamped_free(EI, L)
    # self-weight + eccentric tip body: discrete eigenvalue vs independent shooting reference
    tip = E.TipBody(mass=0.5, offset=0.02)
    rod = E.DiscreteRod(EI, L, 100, mass_per_length=0.4 * q / E.G0, tip=tip)
    W_disc = E.critical_load(rod)[0] + tip.mass * E.G0
    W_ref = E.heavy_column_critical_load(EI, L, 0.4 * q, e=0.02)
    assert abs(W_disc / W_ref - 1) < 2e-4, W_disc / W_ref - 1


def test_elliptic_elastica_tip_deflection():
    Pe = E.euler_clamped_free(EI, L)
    rod = E.DiscreteRod(EI, L, 60)
    path = E.equilibrium_path(rod, np.linspace(0.02, 2.2, 56))
    assert abs(path["P"][0] / Pe - 1) < 1e-3                     # branch switch at the bifurcation
    for f in (1.1, 1.6, 2.5):
        P = f * Pe
        th = E.solve_on_path(rod, path, P)
        nodes, _ = rod.positions(th)
        _, lat, ax = E.elliptic_tip(EI, L, E.elliptic_alpha(EI, L, P))
        assert abs(nodes[-1, 0] / lat - 1) < 5e-3, (f, nodes[-1, 0] / lat - 1)
        assert abs(nodes[-1, 1] - ax) < 5e-3 * L


def test_imperfect_path_with_self_weight_and_tip_body():
    L_ = L
    th0 = lambda s: 0.002 * math.pi / (2 * L_) * np.sin(math.pi * s / (2 * L_))
    rod = E.DiscreteRod(EI, L, 40, mass_per_length=0.2, tip=E.TipBody(0.1, 0.02), theta0=th0)
    Pc, _ = E.critical_load(rod)
    th, ok = E.solve_static(rod, 0.0, rod.theta0)
    assert ok
    path = E.equilibrium_path(rod, np.linspace(th[-1], 1.0, 40), P0=0.0, theta0=th)
    assert np.all(np.diff(path["P"]) > 0) and path["P"][5] < Pc < path["P"][-1]


def test_dynamics_energy_and_damping():
    rod = E.DiscreteRod(EI, L, 12, mass_per_length=0.1, tip=E.TipBody(0.05, 0.02, 1e-5))
    Pc, mode = E.critical_load(rod)
    # undamped, rho_inf = 1, large amplitude: energy ledger closes
    r = E.simulate(rod, 0.3, 2e-4, load=lambda t: 0.5 * Pc, theta_init=0.4 * mode, rho_inf=1.0)
    assert np.max(np.abs(r["residual"])) < 1e-5 * np.max(r["U"])
    # time-dependent load + Rayleigh damping: E - E0 - W_ext + W_diss stays ~ 0
    r = E.simulate(rod, 0.3, 2e-4, load=lambda t: 0.5 * Pc * (1 + math.sin(20 * t)), theta_init=0.4 * mode,
                   rho_inf=0.8, alpha_M=3.0, beta_K=1e-4)
    assert np.max(np.abs(r["residual"])) < 1e-3 * (r["W_ext"][-1] + r["W_diss"][-1])
    # small-amplitude, mass-proportional damping: every mode decays as exp(-alpha_M t / 2)
    aM = 4.0
    z = np.zeros(rod.n)
    w2, V = np.linalg.eig(np.linalg.solve(rod.mass_matrix(z), rod.jacobian(z, 0.0)))
    m1 = np.real(V[:, np.argmin(np.real(w2))])
    r = E.simulate(rod, 1.0, 5e-4, load=lambda t: 0.0, theta_init=1e-4 * m1 / m1[-1], rho_inf=1.0, alpha_M=aM)
    x = r["theta"][:, -1]
    pk = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
    rate = -np.polyfit(r["t"][pk], np.log(x[pk]), 1)[0]
    assert abs(rate / (0.5 * aM) - 1) < 0.02, rate


def test_centreline_to_grid_npz():
    s = np.linspace(0, L, 31)
    theta = np.array([0.0 * s, 0.5 * s / L])                   # straight, then a uniform-curvature arc
    C = E.centreline_from_angles(s, theta)
    assert np.allclose(C[0, -1], [0, L])
    R = L / 0.5
    assert abs(C[1, -1, 0] - R * (1 - math.cos(0.5))) < 1e-4 and abs(C[1, -1, 1] - R * math.sin(0.5)) < 1e-4
    G = E.centreline_to_grid(C, width=0.02, cols=3, rows=11, origin=(1.0, 2.0, 0.5))
    assert G.shape == (2, 11, 3, 3)
    assert np.allclose(G[0, :, 1, 2], 0.5 + np.linspace(0, L, 11)) and np.allclose(G[0, 0, :, 0], [0.99, 1.0, 1.01])
    with tempfile.TemporaryDirectory() as d:
        E.write_deformation_npz(Path(d) / "def.npz", G, np.array([0.0, 0.1]), thickness=0.001)
        z = np.load(Path(d) / "def.npz")
        assert z["positions"].shape == (2, 11, 3, 3) and float(z["thickness"]) == 0.001


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
