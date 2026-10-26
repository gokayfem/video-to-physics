#!/usr/bin/env python
"""Verifier visuals (clip 02).  v001 = kit self-test on synthetic inputs (no production result existed).
  .venv/bin/python verify/make_visuals.py --version v001 --status "kit self-test, synthetic inputs only"
Reads verify/out/ref_solver_selftest.json, verify/out/negative_controls_{physics,blend}.json, verify/out/negctl/P1_*.json."""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out"); VIS = os.path.join(HERE, "visuals")
C1, C2, C3, C4, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 1.6, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def footer(fig, ver, status, what):
    fig.text(0.01, 0.005, f"clip 02 vertical spring | verifier (D) | {ver} | status: {status} | {what}", fontsize=7.5, color=MUTED)


def refsolver(ver, status):
    R = json.load(open(os.path.join(OUT, "ref_solver_selftest.json"))); T = R["tests"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.4)); ax = ax.ravel()
    p = R["_plot_axial"]; t = np.array(p["t"])
    ax[0].plot(t, np.array(p["err"]) / p["a"], color=C1)
    ax[0].axhline(1e-7, color=C2, ls="--", lw=1); ax[0].text(0.2, 1.3e-7, "P3 tolerance 1e-7", color=C2, fontsize=8)
    ax[0].set_yscale("log"); ax[0].set_ylim(1e-17, 1e-5)
    ax[0].set_title("B2 axial oscillation: |z_GL4 - z_exact| / amplitude"); ax[0].set_xlabel("t [s]")
    b4 = R["_plot_B4"]; am = np.array(b4["amps"]); er = np.array(b4["errs"])
    ax[1].loglog(am, er, "o-", color=C1, ms=6, label="GL4 swing frequency vs sqrt(W/(M_t l_eq))")
    ax[1].loglog(am, er[-1] * (am / am[-1]) ** 2, "--", color=MUTED, lw=1, label="theta^2 slope")
    ax[1].axhline(1e-7, color=C2, ls="--", lw=1)
    ax[1].set_title("B4 small-angle swing: rel. frequency error -> 0 as theta^2"); ax[1].set_xlabel("release angle [rad]"); ax[1].legend(fontsize=8, frameon=False)
    names, vals = [], []
    for k, v in T.items():
        for f in ("max_dev_over_ell_eq", "rel", "max_err_over_amp", "max_err_over_rho"):
            if f in v and isinstance(v[f], float) and k.startswith(("B1", "B2", "B3")):
                names.append(k.replace("_undamped", " und.").replace("_damped", " damp.")); vals.append(max(v[f], 1e-17)); break
    for r in T["B5_rigid_modes"]["modes"]:
        names.append(f"B5_rigid_mode{r['mode']} (freq)"); vals.append(max(r["rel_err"], 1e-17))
    y = np.arange(len(names))
    ax[2].barh(y, vals, color=C1, height=0.6); ax[2].set_xscale("log"); ax[2].set_xlim(1e-18, 1e-5)
    ax[2].axvline(1e-7, color=C2, ls="--", lw=1); ax[2].set_yticks(y); ax[2].set_yticklabels(names, fontsize=7)
    ax[2].set_title("B1-B5 relative error (tol 1e-7, dashed)")
    e = R["_plot_energy"]
    for nm, col in (("point", C1), ("rigid", C3)):
        ax[3].plot(e[nm]["t"], np.abs(e[nm]["dE"]) + 1e-18, color=col, label=f"{nm}: |E-E0|/E_osc (undamped)")
        ax[3].plot(e[nm + "_damped"]["t"], np.abs(e[nm + "_damped"]["balance"]) + 1e-18, color=col, ls=":", label=f"{nm}: |E+W_diss-E0|/E_osc (damped)")
    ax[3].plot(e["point"]["t"], np.abs(e["point"]["dLz"]) + 1e-18, color=C4, lw=1, label="point: |dLz|/L_ref")
    ax[3].axhline(1e-8, color=C2, ls="--", lw=1); ax[3].set_yscale("log"); ax[3].set_ylim(1e-18, 1e-6)
    ax[3].set_title("B6/B7 energy, energy balance, L_z (P4 tol 1e-8 / 1e-7)"); ax[3].set_xlabel("t [s]"); ax[3].legend(fontsize=7, frameon=False)
    ap = R["_plot_autoparam"]
    ax[4].semilogy(ap["env_t"], ap["env"], "o", ms=3, color=C1, label="GL4: swing amplitude per swing period")
    ax[4].semilogy(ap["env_t"], ap["pred"], color=C2, lw=1.2, label=f"theory th0*sqrt(cosh 2*sigma*t), sigma=3/4*eps*w_s")
    rows = T["B8_autoparametric"]["rows"]
    ax[4].set_title("B8 2:1 autoparametric growth, eps=%.2f (rel err: %s)" % (ap["eps"], ", ".join(f"{r['rel_err']:.1e}" for r in rows)), fontsize=8.5)
    ax[4].set_xlabel("t [s]"); ax[4].set_ylabel("swing amplitude [rad]"); ax[4].legend(fontsize=7, frameon=False)
    ex = R["_plot_exchange"]
    ax[5].plot(ex["t"], ex["swing_frac"], color=C1, lw=1.0, label="swing share of oscillation energy")
    ax[5].plot(ex["t"], 1 - np.array(ex["swing_frac"]), color=C3, lw=1.0, label="axial share")
    ax[5].set_title(f"B8 energy exchange, eps={ex['eps']}, seed {ex['seed']} rad"); ax[5].set_xlabel("t [s]"); ax[5].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Verifier reference solver (GL4, order 8) vs analytic limits  -  self-test {R['status']}", color=INK, fontsize=11)
    footer(fig, ver, status, "verify/ref_solver.py --selftest; no production data")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97)); fig.savefig(os.path.join(VIS, f"{ver}_refsolver_validation.png"), dpi=130); plt.close(fig)


def negctl(ver, status):
    P = json.load(open(os.path.join(OUT, "negative_controls_physics.json"))); Bj = json.load(open(os.path.join(OUT, "negative_controls_blend.json")))
    rows = []  # (label, metric/tolerance, expected_fail)
    tol = dict(P2=1e-6, P4b=1e-7, P4c=1e-8, P7=None, P8ic=1e-9, P8s=1e-6, P8t=1e-12, P8l=1e-9)
    for k, v in P.items():
        if not isinstance(v, dict) or "key" not in v or "summary" not in v:
            continue
        kk = v["key"]; fail = bool(v["must_fail"])
        cand = [("P2 pos", kk["P2_max_pos_m"] / 1e-6), ("P4 balance", (kk["P4_balance"] or 0) / 1e-7), ("P4 cons. drift", (kk["P4_cons_drift"] or 0) / 1e-8),
                ("P7 T_min<0", 10.0 if min(kk["P7_Tmin"], kk["P7_Tcol_min"] if kk["P7_Tcol_min"] is not None else 1) < 0 else 0.0),
                ("P8 IC", kk["P8_dE0"] / 1e-9), ("P8 time scale", kk["P8_scale"] / 1e-6), ("P8 t grid", kk["P8_t"] / 1e-12),
                ("P8 ell col", (kk["P8_ell"] or 0) / 1e-9)]
        m = max(cand, key=lambda c: c[1])
        rows.append((f"physics: {k}", max(m[1], 1e-6), fail, m[0], v["as_expected"]))
    for k, v in Bj.items():
        if not isinstance(v, dict) or "key" not in v:
            continue
        kk = v["key"]; fail = bool(v["must_fail"])
        cand = [("P5 top gap", kk["top_gap_m"] / 1e-5), ("P5 anchor", (kk["anchor_drift_m"] or 0) / 1e-6), ("P5 coils", 10.0 if not kk["coil_counts_constant"] else 0.0),
                ("P6 eye", kk["eye_vs_ref_m"] / 1e-5), ("P6 rot", (kk["rot_vs_ref_rad"] or 0) / 1e-5), ("P8b fps", 10.0 if kk["fps"] != 24 else 0.0)]
        m = max(cand, key=lambda c: c[1])
        rows.append((f"blend: {k}", max(m[1], 1e-6), fail, m[0], v["as_expected"]))
    for k in ("bench_good_fakeRK4", "bench_bad_ignores_k", "bench_rigid_good_fakeRK4", "P1_good_fakeRK4", "P1_bad_injected_Oh2", "P9_good_fakeRK4", "P9_bad_ignores_k"):
        v = P[k]; exp = v["expected"]; st = v.get("P3") or v.get("P1") or v.get("P9")
        rows.append((f"{'P3' if 'bench' in k else k[:2]}: {k}", 10.0 if st == "FAIL" else 0.1, exp == "FAIL", "verdict", v["as_expected"]))
    for k in ("P10_good_vs_rebuild", "P10_good_vs_keys_pos_err", "P10_good_vs_coil_change"):
        v = Bj[k]; rows.append((f"P10: {k}", 10.0 if v["P10"] == "FAIL" else 0.1, v["expected"] == "FAIL", "verdict", v["as_expected"]))
    fig, ax = plt.subplots(figsize=(13, 0.27 * len(rows) + 1.6))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, val, fail, met, ok) in zip(y, rows):
        val = min(val, 1e6)
        ax.barh(yi, val, color=C2 if fail else C1, height=0.62)
        ax.text(min(max(val, 1e-6) * 1.3, 2e6), yi, f"{met}  {'detected' if fail and ok else ('passes' if ok else 'UNEXPECTED')}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5); ax.set_xscale("log"); ax.set_xlim(1e-6, 1e9)
    ax.axvline(1, color=INK, lw=1); ax.text(1.1, y[0] + 0.7, "tolerance (metric / tol = 1)", fontsize=8)
    ax.set_xlabel("worst metric / its acceptance tolerance (verdict-only rows: 10 = FAIL, 0.1 = PASS)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C1, label="good control (must PASS)"), Patch(color=C2, label="injected fault (must FAIL)")], loc="lower right", fontsize=8, frameon=False)
    allok = P["ALL_AS_EXPECTED"] and Bj["ALL_AS_EXPECTED"]
    ax.set_title(f"Negative controls: every injected fault detected, every good control passes -> {'ALL AS EXPECTED' if allok else 'NOT ALL AS EXPECTED'}")
    footer(fig, ver, status, "synthetic trajectories + verifier-built test .blends")
    fig.tight_layout(rect=(0.02, 0.02, 1, 1)); fig.savefig(os.path.join(VIS, f"{ver}_negative_controls.png"), dpi=130); plt.close(fig)


def convergence(ver, status):
    R = json.load(open(os.path.join(OUT, "ref_solver_selftest.json")))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    o = R["_plot_order"]; h = np.array(o["h"]); e = np.array(o["err"])
    ax[0].loglog(h, e, "o-", color=C1, ms=6, label="verifier GL4 vs GBS (k=4000 N/m stiff case)")
    ax[0].loglog(h, e[0] * (h / h[0]) ** 8, "--", color=MUTED, lw=1, label="slope 8")
    ax[0].set_xlabel("step h [s]"); ax[0].set_ylabel("max |position error| [m]")
    ax[0].set_title("Verifier GL4 order: observed %s" % ", ".join(f"{p:.2f}" for p in R["tests"]["GL4_order"]["observed_order"]))
    ax[0].legend(fontsize=8, frameon=False)
    for name, col in (("P1_good_fakeRK4", C1), ("P1_bad_injected_Oh2", C2)):
        p = json.load(open(os.path.join(OUT, "negctl", f"{name}.json")))["P1"]
        v = np.array(p["values"]); d = np.array(p["pos_successive_max_diffs_m"])
        ax[1].loglog(v[1:], d, "o-", color=col, ms=6, label=f"{name}: p_obs={p['p_obs_last']:.2f} -> {p['status']}")
    ax[1].axhline(1e-7, color=INK, ls="--", lw=1); ax[1].text(v[-1], 1.4e-7, "P1 tol 1e-7", fontsize=8)
    ax[1].set_xlabel("finer step of each pair h [s]"); ax[1].set_ylabel("max |E_h - E_h/2| [m]")
    ax[1].set_title("P1 check exercised on stand-in production (RK4, order 4 declared)"); ax[1].legend(fontsize=8, frameon=False)
    footer(fig, ver, status, "stand-in solver = verify/test/fake_production_solver.py (NOT production)")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(VIS, f"{ver}_convergence.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--version", default="v001"); ap.add_argument("--status", default="kit self-test, synthetic inputs only")
    a = ap.parse_args(); os.makedirs(VIS, exist_ok=True)
    refsolver(a.version, a.status); negctl(a.version, a.status); convergence(a.version, a.status)
    print("visuals written to", VIS)
