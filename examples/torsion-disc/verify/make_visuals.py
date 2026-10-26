#!/usr/bin/env python
"""Verifier visuals (clip 03).  v001 = kit self-test (synthetic inputs + stand-in production; no checkpoint existed).
  .venv/bin/python verify/make_visuals.py --version v001 --status "kit self-test, synthetic inputs only"
Reads verify/out/ref_solver_selftest.json, verify/out/negative_controls_{physics,blend,p1_p3_p9}.json, verify/out/negctl/P1_*.json."""
import argparse, json, math, os
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
    fig.text(0.01, 0.005, f"clip 03 torsion disc | verifier (D) | {ver} | status: {status} | {what}", fontsize=7.5, color=MUTED)


def tolline(ax, y, txt, x=0.02):
    ax.axhline(y, color=C2, ls="--", lw=1)
    ax.text(x, y * 1.6, txt, color=C2, fontsize=7.5, transform=ax.get_yaxis_transform())


def refsolver(ver, status):
    R = json.load(open(os.path.join(OUT, "ref_solver_selftest.json"))); T, PL = R["tests"], R["plots"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.6)); ax = ax.ravel()
    p = PL["torsion_undamped"]; t = np.array(p["t"]); pd = PL["torsion_damped"]
    ax[0].plot(t, p["psi"], color=C1, label="GL4, undamped")
    ax[0].plot(np.array(pd["t"]), pd["psi"], color=C4, lw=1.2, label="GL4, damped (gamma exaggerated)")
    ax[0].plot(t[::6], np.array(p["exact"])[::6], "o", ms=3, mfc="none", color=INK, label="exact closed form")
    ax[0].set_title("B2/B3 torsion, 1 rad (57 deg) release: psi(t)"); ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("psi [rad]")
    ax[0].legend(fontsize=7.5, frameon=False, loc="upper right")
    ax[1].semilogy(t, np.maximum(np.array(p["err"]) / p["amp"], 1e-18), color=C1, label="undamped")
    ax[1].semilogy(np.array(pd["t"]), np.maximum(np.array(pd["err"]) / pd["amp"], 1e-18), color=C4, lw=1.1, label="damped")
    tolline(ax[1], 1e-7, "P3 tolerance 1e-7"); ax[1].set_ylim(1e-18, 1e-5)
    ax[1].set_title("|psi_GL4 - psi_exact| / amplitude (exact for any amplitude)"); ax[1].set_xlabel("t [s]"); ax[1].legend(fontsize=7.5, frameon=False)
    b4 = PL["B4"]; am = np.array(b4["amps"]); er = np.maximum(np.array(b4["errs"]), 1e-16)
    ax[2].loglog(am, er, "o-", color=C1, ms=6, label="fitted swing freq vs sqrt(M g l / (I + M l^2))")
    ax[2].loglog(am, er[0] * (am / am[0]) ** 2, "--", color=MUTED, lw=1, label="theta^2 slope")
    tolline(ax[2], 1e-7, "1e-7")
    ax[2].set_title("B4 small-angle swing: error ~ theta^2"); ax[2].set_xlabel("release angle [rad]"); ax[2].set_ylabel("rel. frequency error")
    ax[2].legend(fontsize=7.5, frameon=False)
    c = PL["conical"]
    ax[3].plot(c["x"], c["y"], color=C1, lw=1.2); ax[3].set_aspect("equal")
    ax[3].set_title("B5 conical orbit (30 deg, kappa = 0): CoM path, 10 s"); ax[3].set_xlabel("x [m]"); ax[3].set_ylabel("y [m]")
    ins = ax[3].inset_axes([0.62, 0.62, 0.36, 0.34]); ins.semilogy(c["t"], np.maximum(c["err"], 1e-18), color=C2, lw=1)
    ins.set_title("|r - r_exact| / rho", fontsize=7); ins.tick_params(labelsize=6); ins.set_ylim(1e-17, 1e-12)
    for i, (k, v) in enumerate(PL["energy"].items()):
        if k == "damped":
            continue
        ax[4].semilogy(v["t"], np.maximum(np.abs(v["dE"]), 1e-18), lw=1, color=[C1, C4, C3, C2, MUTED, INK][i % 6], label=k)
    tolline(ax[4], 1e-8, "P4 tolerance 1e-8"); ax[4].set_ylim(1e-17, 1e-6)
    ax[4].set_title("B6 energy drift |E - E0| / E_osc, undamped 3-D runs"); ax[4].set_xlabel("t [s]"); ax[4].legend(fontsize=6.5, frameon=False, loc="upper left", ncol=2)
    names, vals, tols = [], [], []
    for k, v in T.items():
        for f, tol in (("max_err_over_amp", 1e-7), ("rel_err", 1e-7), ("max_err_over_rho", 1e-7), ("max_rel_drift", 1e-8), ("max_rel_residual", 1e-7),
                       ("twist_rel_err", 1e-7), ("max_dev", 1e-7), ("max_psi_or_y", 1e-7)):
            if f in v and isinstance(v[f], float):
                names.append(k); vals.append(max(v[f], 1e-17)); tols.append(tol); break
    y = np.arange(len(names))
    ax[5].barh(y, np.log10(vals) + 17, left=-17, color=[C1 if a < b else C2 for a, b in zip(vals, tols)], height=0.7)
    ax[5].set_yticks(y); ax[5].set_yticklabels([n[:40] for n in names], fontsize=5.2); ax[5].invert_yaxis()
    ax[5].axvline(-7, color=C2, ls="--", lw=1); ax[5].set_xlabel("log10(metric)   (tolerance line at 1e-7; all bars left of it)")
    ax[5].set_title(f"all {len(T)} self-tests: {R['status']}")
    fig.suptitle("Verifier reference solver (GL4, order 8, exact quaternion norm) vs analytic limits", fontsize=12, x=0.01, ha="left")
    footer(fig, ver, status, "ref_solver.py --selftest")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(os.path.join(VIS, f"{ver}_refsolver_validation.png"), dpi=130); plt.close(fig)


def convergence(ver, status):
    R = json.load(open(os.path.join(OUT, "ref_solver_selftest.json"))); o = R["plots"]["order"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    h = np.array(o["h"]); e = np.array(o["err"])
    ax[0].loglog(h, e, "o-", color=C1, ms=7, label="GL4 vs GBS (H = 1/1536 s)")
    ax[0].loglog(h, e[-1] * (h / h[-1]) ** 8, "--", color=MUTED, lw=1, label="slope 8")
    p = R["tests"]["GL4_order"]
    ax[0].set_title(f"verifier GL4 order: observed {p['p01']:.2f}, {p['p12']:.2f} (theory 8)"); ax[0].set_xlabel("step h [s]"); ax[0].set_ylabel("max state error")
    ax[0].legend(frameon=False, fontsize=8)
    for f, col, lab in (("good", C1, "stand-in RK4 (good)"), ("h2", C2, "stand-in with injected O(h^2) error")):
        fp = os.path.join(OUT, "negctl", f"P1_{f}.json")
        if not os.path.exists(fp):
            continue
        P1 = json.load(open(fp))["P1"]
        vv = np.array(P1["values"]); dd = np.array(P1["combined_successive_max_diffs"])
        ax[1].loglog(vv[1:], dd, "o-", color=col, ms=6, label=f"{lab}: p_obs = {P1['p_obs_last']:.2f} -> {P1['status']}")
    ax[1].axhline(1e-7, color=C2, ls="--", lw=1); ax[1].text(ax[1].get_xlim()[0] * 1.1 if False else 7e-4, 1.5e-7, "P1 tolerance 1e-7", color=C2, fontsize=7.5)
    ax[1].set_title("P1 checker on stand-in production (run via documented command)"); ax[1].set_xlabel("finer step of each pair h [s]")
    ax[1].set_ylabel("max |X_h - X_h/2| (m or rad)"); ax[1].legend(frameon=False, fontsize=8)
    footer(fig, ver, status, "check_convergence.py (stand-in), ref_solver selftest")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(VIS, f"{ver}_convergence.png"), dpi=130); plt.close(fig)


def negctl(ver, status):
    ph = json.load(open(os.path.join(OUT, "negative_controls_physics.json")))
    bl = json.load(open(os.path.join(OUT, "negative_controls_blend.json")))
    pp = json.load(open(os.path.join(OUT, "negative_controls_p1_p3_p9.json")))
    rows = []
    for k, v in ph.items():
        if k == "ALL_AS_EXPECTED":
            continue
        s = v["summary"]; must = v["must_fail"]
        rows.append(("physics", k, "good -> all PASS" if not must else "must FAIL " + ",".join(must),
                     " ".join(f"{p}:{s[p][0]}" for p in ("P2", "P4", "P7", "P8")), v["as_expected"]))
    for k, v in bl.items():
        if k == "ALL_AS_EXPECTED":
            continue
        if "summary" in v:
            must = v["must_fail"]; s = v["summary"]
            rows.append(("blend", k, "informational" if must is None else ("good -> all PASS" if not must else "must FAIL " + ",".join(must)),
                         " ".join(f"{p}:{s[p][0]}" for p in ("P5", "P6", "P8_blend")), v["as_expected"]))
        else:
            rows.append(("P10", k, f"expect {v['expected']}", f"P10:{v['P10']}", v["as_expected"]))
    for k, v in pp.items():
        if k == "ALL_AS_EXPECTED":
            continue
        rows.append((k.split("_")[0], k, f"expect {v['expected']}", f"{k.split('_')[0]}:{v['status']}", v["as_expected"]))
    fig, ax = plt.subplots(figsize=(13, 0.3 * len(rows) + 1.6))
    ax.axis("off")
    cols = ["group", "case", "expectation", "verdicts (P=PASS F=FAIL N=NOT RUN)", "as expected"]
    xs = [0.0, 0.08, 0.36, 0.58, 0.9]
    for x, c in zip(xs, cols):
        ax.text(x, 1.0, c, fontsize=9, fontweight="bold", transform=ax.transAxes, va="top")
    for i, r in enumerate(rows):
        y = 1.0 - (i + 1.3) / (len(rows) + 1.5)
        for x, c in zip(xs[:4], r[:4]):
            ax.text(x, y, c, fontsize=8, transform=ax.transAxes, va="top", color=INK, family="monospace" if x == xs[3] else None)
        ax.text(xs[4], y, "YES" if r[4] else "NO", fontsize=8.5, fontweight="bold", transform=ax.transAxes, va="top", color=C1 if r[4] else C2)
    allok = ph["ALL_AS_EXPECTED"] and bl["ALL_AS_EXPECTED"] and pp["ALL_AS_EXPECTED"]
    ax.set_title(f"Negative and good controls: {sum(r[4] for r in rows)}/{len(rows)} as expected -> {'ALL DETECTED' if allok else 'MISSED SOME'}",
                 loc="left", fontsize=12)
    footer(fig, ver, status, "test/run_negative_controls_{physics,blend,p1_p3_p9}.py")
    fig.savefig(os.path.join(VIS, f"{ver}_negative_controls.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
    # metric-vs-tolerance chart for the physics + blend injected faults
    items = []
    for k, v in ph.items():
        if k == "ALL_AS_EXPECTED" or "key" not in v:
            continue
        kk = v["key"]
        items.append((f"phys {k}", max(kk["P2_C_m"] / 1e-6, kk["P2_att_rad"] / 1e-6, (kk["P2_psi_rad"] or 0) / 1e-6, kk["P4_balance"] / 1e-7,
                                       max(kk["P4_fwd_dE"], 0) / 1e-12, (kk["P4_cons_drift"] or 0) / 1e-8, kk["P8_ic"] / 1e-9, kk["P8_qnorm"] / 1e-12,
                                       kk["P8_scale"] / 1e-6 if kk["P8_scale"] > 1e-9 else 0, kk["P8_wire"] / 1e-9, kk["P8_grid"] / 1e-12,
                                       (1e3 if (kk["P7_min_T"] is not None and kk["P7_min_T"] < 0) else 0)), bool(v["must_fail"])))
    for k, v in bl.items():
        if not isinstance(v, dict) or "key" not in v:
            continue
        kk = v["key"]
        p6f = json.load(open(os.path.join(OUT, "negctl", f"blend_{k}.json")))["P6_blend_vs_solver"]
        items.append((f"blend {k}", max(p6f.get("interp_probe_C_max_m", 0) / 1e-5, p6f.get("interp_probe_rotation_max_rad", 0) / 1e-5, kk["C_vs_ref_m"] / 1e-5, kk["rot_vs_ref_rad"] / 1e-5, kk["psi_vs_ref_rad"] / 1e-5, kk["bottom_gap_m"] / 1e-5,
                                        (kk["anchor_drift_m"] or 0) / 1e-6, kk["mark_dev_m"] / 1e-6, 1e3 if kk["fps"] != 24 else 0), bool(v["must_fail"])))
    fig, ax = plt.subplots(figsize=(11, 0.28 * len(items) + 1.5))
    y = np.arange(len(items)); val = np.array([max(i[1], 1e-6) for i in items])
    ax.barh(y, np.log10(val) + 6, left=-6, color=[C2 if i[2] else C1 for i in items], height=0.65)
    ax.axvline(0, color=INK, lw=1); ax.text(0.05, -0.9, "tolerance", fontsize=7.5)
    ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("log10( worst metric / its tolerance )   blue = good control (must stay < 0), orange = injected fault (must be > 0)")
    ax.set_title("Negative controls: worst metric relative to tolerance (P2/P4/P7/P8 physics; P5/P6/P8b blend)", loc="left")
    footer(fig, ver, status, "out/negative_controls_*.json")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(VIS, f"{ver}_negative_controls_margins.png"), dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--version", default="v001"); ap.add_argument("--status", default="kit self-test, synthetic inputs only")
    a = ap.parse_args()
    os.makedirs(VIS, exist_ok=True)
    refsolver(a.version, a.status); convergence(a.version, a.status); negctl(a.version, a.status)
    print("visuals written to", VIS)


if __name__ == "__main__":
    main()
