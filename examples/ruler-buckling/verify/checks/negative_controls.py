"""Negative controls for check_production.py on synthetic inputs (built from the role-E reference, never from solver/).
Run:  <venv-python> verify/checks/negative_controls.py
Writes verify/out/synthetic/*.npz|json, verify/out/negative_controls.json, verify/visuals/v001_negative_controls.png.
Each case runs check_production.py as a subprocess; a control PASSES when the expected checks fail (or pass, for the good cases)."""
import json, math, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid

HERE = Path(__file__).resolve()
V = HERE.parents[1]; ROOT = HERE.parents[2]
sys.path.insert(0, str(V / "ref"))
import elastica as E
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PY = sys.executable
SYN = V / "out/synthetic"; SYN.mkdir(parents=True, exist_ok=True)
sp = json.loads((ROOT / "config/scene_params.json").read_text())
ru = sp["ruler"]
Z0, SCL, LT = ru["bottom_z_m"], ru["clamped_length_m"], E.DECLARED["tab_len"]
NW = 11
XW = np.linspace(-ru["width_m"] / 2, ru["width_m"] / 2, NW)
FPS_SUB = 120.0


def strip_nodes(y_f, z_f, thL, Lf):
    """Full-strip centreline (301 nodes at ~1 mm): clamp 20 mm vertical, flexible nodes (given), rigid 12 mm tab."""
    nc = int(round(SCL / 0.001)); nt = int(round(LT / 0.001))
    zc = np.linspace(0, SCL, nc + 1)[:-1]
    cl = np.c_[np.zeros(nc), zc]
    fl = np.c_[y_f, SCL + z_f]
    u = np.linspace(0, LT, nt + 1)[1:]
    tb = np.c_[y_f[-1] + u * math.sin(thL), SCL + z_f[-1] + u * math.cos(thL)]
    yz = np.vstack([cl, fl, tb])
    P = np.zeros((NW, len(yz), 3))
    P[:, :, 0] = XW[:, None]; P[:, :, 1] = yz[None, :, 0]; P[:, :, 2] = Z0 + yz[None, :, 1]
    return P.reshape(-1, 3)


def static_case(rod_gen, stretch=0.0, hold=80):
    Ps, Ws = [], []
    nf = 268  # same node count; a wrong length shows up as wrong spacing
    for n in range(13):
        tb = E.scene_tipbody(n)
        W = tb.M * rod_gen.g
        sol = E.equilibrium(rod_gen, W, tb.e, 0.0, n_out=nf + 1)
        y, z = sol["y"], sol["z"]
        if stretch and abs(sol["tip_y"]) > 1e-3:
            y, z = y * (1 + stretch), z * (1 + stretch)
        P = strip_nodes(y, z, sol["theta_L"], rod_gen.L)
        Ps += [P] * hold; Ws += [W] * hold
    t = np.arange(len(Ps)) / FPS_SUB
    return np.array(Ps), np.array(Ws), t


def dynamic_case(rod, n=5, a0=0.35, Q=50.0, sign=+1.0, T=10.0, sub=16):
    """1-DOF Ritz model theta(s,t) = a(t) sin(pi s / 2L) with exact nonlinear kinematics; Lagrange EOM with
    viscous damping c a_dot^2 >= 0 (sign=-1 injects energy). Energy-consistent by construction."""
    L, EI, q, m = rod.L, rod.EI, rod.q, rod.rho * rod.A
    tb = E.scene_tipbody(n); W = tb.M * rod.g; M, e, J = tb.M, tb.e, tb.J_c
    sf = np.linspace(0, L, 268 * sub + 1); psi = np.sin(math.pi * sf / (2 * L)); dpsi = math.pi / (2 * L) * np.cos(math.pi * sf / (2 * L))

    def geom(a):
        th = a * psi
        y = cumulative_trapezoid(np.sin(th), sf, initial=0); z = cumulative_trapezoid(np.cos(th), sf, initial=0)
        ya = cumulative_trapezoid(np.cos(th) * psi, sf, initial=0); za = cumulative_trapezoid(-np.sin(th) * psi, sf, initial=0)
        thL = th[-1]
        yc_a = ya[-1] + e * math.cos(thL); zc_a = za[-1] - e * math.sin(thL)
        Ma = m * np.trapezoid(ya ** 2 + za ** 2, sf) + M * (yc_a ** 2 + zc_a ** 2) + J * 1.0
        Vp = 0.5 * EI * a * a * np.trapezoid(dpsi ** 2, sf) + q * np.trapezoid(z, sf) + W * (z[-1] + e * math.cos(thL))
        return Ma, Vp, y, z, thL
    h = 1e-6
    K0 = (geom(h)[1] - 2 * geom(0)[1] + geom(-h)[1]) / h ** 2
    w0 = math.sqrt(max(K0, 1e-12) / geom(0)[0])
    c = sign * geom(0)[0] * w0 / Q

    def rhs(t, u):
        a, ad = u
        Ma = geom(a)[0]
        dM = (geom(a + h)[0] - geom(a - h)[0]) / (2 * h)
        dV = (geom(a + h)[1] - geom(a - h)[1]) / (2 * h)
        return [ad, (-0.5 * dM * ad * ad - dV - c * ad) / Ma]
    tt = np.arange(0, T + 1e-9, 1 / FPS_SUB)
    sol = solve_ivp(rhs, (0, T), [a0, 0.0], t_eval=tt, rtol=1e-10, atol=1e-12, method="DOP853")
    idx = np.arange(0, len(sf), sub)
    Ps = []
    for a in sol.y[0]:
        _, _, y, z, thL = geom(a)
        Ps.append(strip_nodes(y[idx], z[idx], thL, L))
    return np.array(Ps), np.full(len(tt), W), tt, w0 / (2 * math.pi)


def write(name, P, W, t, pcr):
    Pg = P.reshape(len(P), NW, -1, 3)[:, NW // 2]
    np.savez(SYN / f"{name}.npz", centreline_y=Pg[..., 1], centreline_z=Pg[..., 2], positions=P.astype(np.float32), t=t, tip_load_N=W)
    (SYN / f"{name}.json").write_text(json.dumps({"P_cr_N": pcr, "_doc": "synthetic role-E control"}))


def run(name, skip=""):
    out = V / f"out/synthetic/{name}_report.json"
    cmd = [PY, str(HERE.parent / "check_production.py"), "--npz", str(SYN / f"{name}.npz"), "--meta", str(SYN / f"{name}.json"),
           "--out", str(out)] + (["--skip", skip] if skip else [])
    subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return json.loads(out.read_text())


if __name__ == "__main__":
    rod = E.Rod(g=9.81)  # same g as params/physics.json read by the checker
    ref = json.loads((V / "out/reference.json").read_text())
    pcr = ref["scene"]["variants"]["peg40_from_geometry(m0=13.94g)"]["Wcr_N"]
    cases = {}
    P, W, t = static_case(rod); write("good_static", P, W, t, pcr)
    P, W, t = static_case(E.Rod(g=9.81, E=rod.E * 1.01)); write("bad_E_plus1pct", P, W, t, pcr * 1.01)
    P, W, t = static_case(E.Rod(g=9.81, L=rod.L * 1.01)); write("bad_length_plus1pct", P, W, t, pcr / 1.01 ** 2)
    P, W, t = static_case(rod, stretch=2e-5); write("bad_stretch_2e-5", P, W, t, pcr)
    f0 = None
    if not (SYN / "good_dynamic_Q50.npz").exists():
        P, W, t, f0 = dynamic_case(rod, sign=+1); write("good_dynamic_Q50", P, W, t, pcr)
    if not (SYN / "bad_energy_injection.npz").exists():
        P, W, t, f0 = dynamic_case(rod, sign=-1, Q=200.0, T=4.0, sub=2); write("bad_energy_injection", P, W, t, pcr)
    expect = {  # name: (skip, checks that must FAIL, checks that must PASS)
        "good_static": ("energy", [], ["C1_pcr", "C1b_n_buckle", "C2_postbuckling", "C3_equilibrium", "C4_inextensible", "C5_clamp"]),
        "bad_E_plus1pct": ("energy", ["C1_pcr", "C2_postbuckling", "C3_equilibrium"], ["C4_inextensible"]),
        "bad_length_plus1pct": ("energy", ["C4_inextensible", "C2_postbuckling"], []),
        "bad_stretch_2e-5": ("energy", ["C4_inextensible"], ["C5_clamp"]),
        "good_dynamic_Q50": ("", [], ["C4_inextensible", "C5_clamp", "C6_energy"]),
        "bad_energy_injection": ("", ["C6_energy"], ["C4_inextensible"]),
    }
    summary = {}
    for name, (skip, must_fail, must_pass) in expect.items():
        r = run(name, skip)
        got = {k: c.get("passed") for k, c in r["checks"].items()}
        ok = all(got.get(k) is False for k in must_fail) and all(got.get(k) is True for k in must_pass)
        summary[name] = dict(control_ok=ok, verdicts=got, must_fail=must_fail, must_pass=must_pass,
                             numbers=dict(len_err=r["checks"]["C4_inextensible"]["max_abs_length_err_m"],
                                          post=r["checks"].get("C2_postbuckling", {}).get("worst_rel"),
                                          equil=r["checks"].get("C3_equilibrium", {}).get("worst"),
                                          pcr=r["checks"].get("C1_pcr", {}).get("rel_vs_refs"),
                                          energy=({k: v for k, v in r["checks"]["C6_energy"].items() if k != "series"}
                                                  if "C6_energy" in r["checks"] else None)))
        summary[name]["_series"] = r["checks"].get("C6_energy", {}).get("series")
    (V / "out/negative_controls.json").write_text(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "_series"}
                                                              for k, v in summary.items()}, indent=1, default=float))
    # ---- visual
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
    a = ax[0]
    names = list(expect); checks = ["C1_pcr", "C1b_n_buckle", "C2_postbuckling", "C3_equilibrium", "C4_inextensible", "C5_clamp", "C6_energy"]
    M = np.full((len(names), len(checks)), np.nan)
    for i, nm in enumerate(names):
        for j, ckn in enumerate(checks):
            v = summary[nm]["verdicts"].get(ckn)
            M[i, j] = np.nan if v is None else (1.0 if v else 0.0)
    a.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i, nm in enumerate(names):
        for j, ckn in enumerate(checks):
            v = summary[nm]["verdicts"].get(ckn)
            a.text(j, i, "-" if v is None else ("pass" if v else "FAIL"), ha="center", va="center", fontsize=8)
    a.set_xticks(range(len(checks)), [c.replace("_", "\n", 1) for c in checks], fontsize=7)
    a.set_yticks(range(len(names)), [f"{nm}  [{'OK' if summary[nm]['control_ok'] else 'NOT DETECTED'}]" for nm in names], fontsize=8)
    a.set_title("v001 | negative controls (synthetic) | all detected: %s" % all(s["control_ok"] for s in summary.values()))
    a = ax[1]
    for nm, col in (("good_dynamic_Q50", "C2"), ("bad_energy_injection", "C3")):
        s = summary[nm]["_series"]
        if s:
            a.plot(s["t"], 1e3 * np.array(s["D"]), color=col, label=f"{nm}: dissipated D(t)")
            a.plot(s["t"], 1e3 * np.array(s["U"]), color=col, ls=":", lw=0.8, label=f"{nm}: strain U(t)")
    a.axhline(0, color="k", lw=0.6)
    a.set_xlabel("t [s]"); a.set_ylabel("mJ")
    a.set_title(f"v001 | energy balance on 1-DOF Ritz trajectories (n=5, a0=0.35 rad)\nD = work - dU - dV_self - dT must be >=0 and non-decreasing")
    a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(V / "visuals/v001_negative_controls.png", dpi=110)
    print(json.dumps({k: dict(ok=v["control_ok"], verdicts=v["verdicts"], numbers=v["numbers"]) for k, v in summary.items()}, indent=1, default=float))
