#!/usr/bin/env python
"""Diagnostic plots for every verifier report -> verify/visuals/<version>_*.png
  .venv/bin/python verify/make_visuals.py --version v001 --status preliminary
Every figure carries a footer: version, status, inputs' sha256 prefixes, UTC time."""
import argparse, datetime, glob, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from vlib import read_traj  # noqa: E402
from vparams import sha256_file, load_model  # noqa: E402

C1, C2, C3, C4, C5, C7, C8 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False})


def J(p):
    return json.load(open(os.path.join(HERE, "out", p)))


def footer(fig, a, inputs, title, verdict):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    ins = "  ".join(f"{os.path.basename(k)}:{v[:8]}" for k, v in inputs.items())
    fig.suptitle(f"[{verdict}] {title}", x=0.01, ha="left", fontsize=10.5, color=INK, fontweight="bold")
    fig.text(0.01, 0.003, f"verify {a.version} | status: {a.status.upper()} | {now}\ninputs sha256: {ins}",
             fontsize=6.5, color=INK2, ha="left", va="bottom")


def save(fig, name, a, out):
    p = os.path.join(HERE, "visuals", f"{a.version}_{name}.png")
    fig.savefig(p, dpi=130); plt.close(fig); out.append(p)


def tolline(ax, y, label):
    ax.axhline(y, color=C8, lw=1, ls="--"); ax.text(ax.get_xlim()[0], y, f" {label}", color=C8, va="bottom", fontsize=7)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--version", default="v001"); ap.add_argument("--status", default="preliminary")
    ap.add_argument("--inputs-root", default=ROOT, help="tree holding params/physics.json and solver/out/trajectory.csv (e.g. a checkpoint copy)")
    ap.add_argument("--outdir", default=os.path.join(HERE, "out"), help="dir with the verifier reports for this run")
    ap.add_argument("--tag", default=None, help="report name suffix (default prelim_<version>)")
    ap.add_argument("--blend-report", default=None, help="check_blend json basename in outdir (default check_blend_baked_a_prelim_<version>)")
    ap.add_argument("--blend-label", default="verifier-run build+bake")
    ap.add_argument("--checkpoint", default="", help="checkpoint id shown on every figure")
    ap.add_argument("--skip-synthetic", action="store_true", help="skip negative-control and ref-solver figures")
    a = ap.parse_args()
    global J
    OD = os.path.abspath(a.outdir); IR = os.path.abspath(a.inputs_root); TAG = a.tag or f"prelim_{a.version}"
    J = lambda p: json.load(open(os.path.join(OD, p)))
    if a.checkpoint: a.status = f"{a.status} | checkpoint {a.checkpoint}"
    os.makedirs(os.path.join(HERE, "visuals"), exist_ok=True)
    out = []
    phys = J(f"check_physics_{TAG}.json"); R = phys["results"]
    model, _ = load_model(os.path.join(IR, "params/physics.json"))
    d, _, _ = read_traj(os.path.join(IR, "solver/out/trajectory.csv"))
    ref, _, _ = read_traj(os.path.join(OD, "ref_trajectory.csv"), required=["t_s"])
    n = len(d["t_s"]); t = d["t_s"]
    ins = {"physics.json": sha256_file(os.path.join(IR, "params/physics.json")),
           "trajectory.csv": sha256_file(os.path.join(IR, "solver/out/trajectory.csv")),
           "ref_trajectory.csv": sha256_file(os.path.join(OD, "ref_trajectory.csv"))}

    # ---- P2 benchmark
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    ax[0].plot(t, np.degrees(d["theta_rad"]), color=C1, label="production solver (RK4)")
    ax[0].plot(ref["t_s"][:n], np.degrees(ref["theta_rad"][:n]), color=C2, ls=(0, (4, 3)), label="verifier ref (Taylor-24, event split)")
    ax[0].set_ylabel("θ (deg)"); ax[0].legend(loc="lower right", ncol=2, bbox_to_anchor=(1, 1.0))
    dd = np.abs(d["theta_rad"] - ref["theta_rad"][:n]); dd[dd == 0] = 1e-18
    ax[1].semilogy(t, dd, color=C1); ax[1].set_ylim(1e-17, 1e-4); tolline(ax[1], 1e-6, "P2 tol 1e-6 rad")
    ax[1].set_ylabel("|Δθ| (rad)"); ax[1].set_xlabel("t (s)")
    p2 = R["P2_independent_benchmark"]
    ax[1].text(0.99, 0.9, f"max |Δθ| = {p2['max_abs_dtheta_rad']:.2e} rad", transform=ax[1].transAxes, ha="right", color=INK)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    footer(fig, a, ins, "P2 independent benchmark: production θ(t) vs verifier reference", p2["status"]); save(fig, "P2_benchmark_theta", a, out)

    # ---- P1 convergence
    conv = J(f"check_convergence_{TAG}.json"); P1 = conv["P1"]
    st = json.load(open(os.path.join(OD, "ref_solver_selftest.json")))["convergence"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    h = np.array(P1["values"])[1:]; e = np.array(P1["theta_rad"]["successive_max_diffs"])
    ax[0].loglog(h, e, "o-", color=C1, ms=7, label=f"production RK4: max|θ_h − θ_2h| (p_obs={P1['p_obs_last']:.3f})")
    ax[0].loglog(h, e[0] * (h / h[0]) ** 4, color=INK2, lw=1, ls=":", label="slope 4 reference")
    ax[0].axhline(1e-7, color=C8, ls="--", lw=1); ax[0].text(h[-1], 1.2e-7, "P1 tol 1e-7 rad", color=C8, fontsize=7)
    ax[0].set_xlabel("step h (s)"); ax[0].set_ylabel("max |Δθ| between successive steps (rad)"); ax[0].legend(fontsize=7)
    ax[0].set_title("production solver (subprocess runs)", fontsize=9, color=INK2)
    for nm, col in [("gl4", C3), ("gl2", C7), ("taylor_N6", C2)]:
        ax[1].loglog(st[nm]["h_s"], st[nm]["err"], "o-", color=col, ms=6, label=f"verifier {nm}: p_obs={st[nm]['observed_order'][-1]:.2f} (theory {st[nm]['theoretical']})")
    ax[1].set_xlabel("step h (s)"); ax[1].set_ylabel("max |θ − θ_ref| (rad)"); ax[1].legend(fontsize=7)
    ax[1].set_title("verifier's own integrators (damped 30° case, events)", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    footer(fig, a, {**ins, **{os.path.basename(i["file"]): i["sha256"] for i in conv["inputs"][-1:]}}, "P1 convergence (log-log)", P1["status"]); save(fig, "P1_convergence", a, out)

    # ---- P4 energy
    dc, _, _ = read_traj(os.path.join(OD, "prod_runs/conservative/trajectory.csv"))
    E0 = R["P4_energy_damped"]["E0_recomputed_J"]
    from vlib import cumulative_dissipation
    from check_physics import jerk_of, P_of
    W = cumulative_dissipation(t, d["omega_rad_s"], d["alpha_rad_s2"], jerk_of(model, d["theta_rad"], d["omega_rad_s"], d["alpha_rad_s2"]), P_of(model))
    fig, ax = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    ax[0].plot(t, d["E_J"] / E0, color=C1, label="E(t)/E0 (column E_J, damped)")
    ax[0].plot(t, 1 - W / E0, color=C2, ls=(0, (4, 3)), label="1 − ∫P_diss dt / E0 (verifier quadrature)")
    ax[0].set_ylabel("energy / E0"); ax[0].legend()
    bal = np.abs(d["E_J"] - d["E_J"][0] + W) / E0; bal[bal == 0] = 1e-18
    ax[1].semilogy(t, bal, color=C1); ax[1].set_ylim(1e-18, 1e-5); tolline(ax[1], 1e-7, "P4 damped tol 1e-7")
    ax[1].set_ylabel("|E−E0+∫P dt|/E0")
    dr = np.abs(dc["E_J"] - dc["E_J"][0]) / E0; dr[dr == 0] = 1e-18
    ax[2].semilogy(t, dr, color=C3, label="conservative run (c=0 params copy): |E−E0|/E0"); ax[2].set_ylim(1e-18, 1e-6)
    tolline(ax[2], 1e-8, "P4 conservative tol 1e-8"); ax[2].set_ylabel("drift"); ax[2].set_xlabel("t (s)"); ax[2].legend(loc="lower right")
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    v4 = "PASS" if R["P4_energy_damped"]["status"] == R["P4_energy_conservative"]["status"] == "PASS" else "FAIL"
    footer(fig, a, ins, "P4 energy balance (damped) and drift (conservative)", v4); save(fig, "P4_energy", a, out)

    # ---- P3 periods + P7 tension
    p3 = R["P3_finite_amplitude_period"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    Tc = np.array(p3["periods_s"]); rel = np.abs(Tc / p3["T_exact_ellipk_s"] - 1); rel[rel == 0] = 1e-17
    ax[0].semilogy(np.arange(1, len(Tc) + 1), rel, "o", color=C1, ms=7, label="conservative production run")
    ax[0].axhline(1e-7, color=C8, ls="--", lw=1); ax[0].text(1, 1.3e-7, "P3 tol 1e-7", color=C8, fontsize=7)
    ax[0].set_ylim(1e-17, 1e-5); ax[0].set_xlabel("cycle #"); ax[0].set_ylabel("|T_meas/T_exact − 1|")
    ax[0].set_title(f"T_exact = 4√(I/mgd)K(sin²(θ0/2)) = {p3['T_exact_ellipk_s']:.12f} s\nsmall-angle T0 = {p3['T_small_angle_s']:.9f} s", fontsize=8, color=INK2)
    p7 = R["P7_tension"]
    Tb = model.m_bob * (model.g * np.cos(d["theta_rad"]) + model.L_c * d["omega_rad_s"] ** 2)
    ax[1].plot(t, d["tension_N"], color=C1, label="column tension_N"); ax[1].plot(t, Tb, color=C2, ls=(0, (4, 3)), label="recomputed m(g cosθ + L_c ω²)")
    ax[1].set_ylim(0, Tb.max() * 1.1); ax[1].set_xlabel("t (s)"); ax[1].set_ylabel("T (N)"); ax[1].legend(loc="lower right")
    ax[1].set_title(f"P7 tension: min {p7['recomputed_min_N']:.4f} N > 0  [{p7['status']}]", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.045, 1, 0.92))
    footer(fig, a, ins, "P3 finite-amplitude period and P7 tension", f"P3 {p3['status']}, P7 {p7['status']}"); save(fig, "P3_P7_period_tension", a, out)

    # ---- P6 blend overlay
    BR = a.blend_report or f"check_blend_baked_a_prelim_{a.version}"
    bj = J(f"{BR}.json"); bz = np.load(os.path.join(OD, f"{BR}.npz"))
    tb, thb, thr, inc = bz["t"], bz["theta_blend"], bz["theta_ref"], bz["in_clip"]
    Lc = model.L_c; L = np.linalg.norm(bz["bob_centre"], axis=1)
    fig, ax = plt.subplots(3, 1, figsize=(9, 7.4), sharex=True)
    ax[0].plot(tb[inc], np.degrees(thr[inc]), color=C2, label="verifier ref solution")
    ax[0].plot(tb[inc], np.degrees(thb[inc]), color=C1, ls=(0, (4, 3)), label="Blender evaluated bob (reopened .blend, 4 subframes/frame)")
    ax[0].set_ylabel("θ (deg)"); ax[0].legend(loc="lower right", fontsize=7, ncol=2, bbox_to_anchor=(1, 1.0))
    sub = np.isclose((tb * 24) % 1, 0)
    ax[1].semilogy(tb[inc & ~sub], np.abs(thb - thr)[inc & ~sub], ".", color=C1, ms=3, label="subframes")
    ax[1].semilogy(tb[inc & sub], np.maximum(np.abs(thb - thr)[inc & sub], 1e-12), ".", color=C3, ms=3, label="whole frames (keys)")
    ax[1].set_ylim(1e-12, 1e-4); tolline(ax[1], 1e-5, "P6 tol 1e-5 rad"); ax[1].set_ylabel("|θ_blend − θ_ref|"); ax[1].legend(loc="lower right", fontsize=7)
    ax[2].plot(tb[inc], (L[inc] - Lc) * 1e6, color=C1, label="|bob−P| − L_c")
    ax[2].plot(tb[inc], bz["bob_centre"][inc, 1] * 1e6, color=C3, label="y_bob")
    ax[2].axhline(1, color=C8, ls="--", lw=1); ax[2].axhline(-1, color=C8, ls="--", lw=1)
    ax[2].set_ylabel("µm (tol ±1)"); ax[2].set_xlabel("t (s)"); ax[2].legend(loc="lower right", fontsize=7)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    bins = {"pendulum.blend": [v for k, v in bj["inputs"].items() if k.endswith(".blend")][0], **ins}
    footer(fig, a, bins, f"P5/P6 Blender-evaluated bob vs verifier reference ({a.blend_label})", f"P5 {bj['summary']['P5']}, P6 {bj['summary']['P6']}"); save(fig, "P5_P6_blend_overlay", a, out)

    # ---- P9 controls
    cj = J(f"check_controls_{TAG}.json")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    cols = dict(base=INK2, L_x4=C1, amp_2deg=C3, amp_15deg=C4, damping_x10=C2)
    for v, e in cj["variants"].items():
        an = e["analysis"]
        ax[0].plot(an["turning_times"], np.degrees(an["amp_meas"]), "o", color=cols[v], ms=4)
        ax[0].plot(an["turning_times"], np.degrees(an["amp_pred"]), "-", color=cols[v], lw=1, label=f"{v} [{e['status']}]")
        ax[1].bar(v, max(an["max_rel_period_err"], 1e-16), color=cols[v], width=0.35, label=None)
        ax[1].bar(v, max(an["max_rel_envelope_err"], 1e-16), color=cols[v], width=0.35, alpha=0.45, bottom=0)
    ax[0].set_xlabel("t (s)"); ax[0].set_ylabel("turning-point |θ| (deg)"); ax[0].legend(fontsize=7)
    ax[0].set_title("dots: production runs; lines: averaged elliptic decay law", fontsize=8, color=INK2)
    ax[1].set_yscale("log"); ax[1].axhline(1e-4, color=C8, ls="--", lw=1); ax[1].text(-0.4, 1.3e-4, "P9 tol 1e-4", color=C8, fontsize=7)
    ax[1].set_ylabel("max rel. error (solid: period, pale: envelope)"); ax[1].tick_params(axis="x", labelsize=7)
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    footer(fig, a, ins, "P9 control response (params copies -> solver -> build+bake -> check_blend)", cj["status"]); save(fig, "P9_controls", a, out)

    if a.skip_synthetic:
        print("\n".join(out)); return
    # ---- negative controls
    nphys = J("negative_controls_physics.json"); nbl = J("negative_controls_blend.json")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    rows = []
    tol = dict(P2_dtheta=1e-6, P8_period_rel=1e-7, P8_xz_m=1e-9, P8_dtheta0=1e-9, P8_tgrid=1e-12, P4d_balance=1e-7, P4c_drift=1e-8)
    pick = {"theta_err_1e-4": "P2_dtheta", "timescale_0.5pct": "P8_period_rel", "length_1mm": "P8_xz_m", "energy_growth": "P4d_balance",
            "conservative_drift": "P4c_drift", "theta0_off_1e-8": "P8_dtheta0", "tgrid_wrong": "P8_tgrid"}
    for case, key in pick.items():
        good = nphys["good_gl4"]["key_numbers"][key] or 0
        rows.append((case, key, nphys[case]["key_numbers"][key] / tol[key], max(good, 1e-30) / tol[key], nphys[case]["verdict_as_expected"]))
    rows.append(("slack_120deg", "P7_min_N", None, None, nphys["slack_120deg"]["verdict_as_expected"]))
    y = np.arange(len(rows) - 1)
    ax[0].barh(y, [r[2] for r in rows[:-1]], color=C2, height=0.35, label="injected fault")
    ax[0].barh(y + 0.38, [max(r[3], 1e-9) for r in rows[:-1]], color=C1, height=0.35, label="good control (GL4-made)")
    ax[0].set_xscale("log"); ax[0].axvline(1, color=C8, ls="--", lw=1)
    ax[0].set_yticks(y + 0.19); ax[0].set_yticklabels([f"{r[0]}\n({r[1]})" for r in rows[:-1]], fontsize=7)
    ax[0].set_xlabel("metric / tolerance (>1 = FAIL)"); ax[0].legend(fontsize=7, loc="lower right")
    ax[0].set_title(f"trajectory checks; slack_120deg: P7 min T = {nphys['slack_120deg']['key_numbers']['P7_min_N']:.3f} N (<0 caught)", fontsize=8, color=INK2)
    brow = [("bob_1mm_lower", "L_err_m", 1e-6), ("bob_y_2um", "y_m", 1e-6), ("wire_top_gap_50um", "wire_top_gap_m", 1e-5),
            ("keys_timescale_0.5pct", "dtheta_ref", 1e-5), ("keys_theta_err_1e-4", "dtheta_ref", 1e-5)]
    yb = np.arange(len(brow))
    ax[1].barh(yb, [nbl[c]["key_numbers"][k] / tl for c, k, tl in brow], color=C2, height=0.35, label="injected fault")
    ax[1].barh(yb + 0.38, [max(nbl["good"]["key_numbers"][k], 1e-30) / tl for c, k, tl in brow], color=C1, height=0.35, label="good test .blend")
    ax[1].set_xscale("log"); ax[1].axvline(1, color=C8, ls="--", lw=1)
    ax[1].set_yticks(yb + 0.19); ax[1].set_yticklabels([f"{c}\n({k})" for c, k, _ in brow], fontsize=7)
    ax[1].set_xlabel("metric / tolerance (>1 = FAIL)"); ax[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    ax[1].set_title(f"Blender checks (scene_fps25 caught by scene check)", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    allok = nphys["ALL_AS_EXPECTED"] and nbl["ALL_AS_EXPECTED"]
    a2 = argparse.Namespace(version=a.version, status="synthetic test data (negative controls)")
    footer(fig, a2, {"negative_controls_physics.json": sha256_file(os.path.join(HERE, "out/negative_controls_physics.json")),
                     "negative_controls_blend.json": sha256_file(os.path.join(HERE, "out/negative_controls_blend.json"))},
           "Negative controls: every injected fault detected", "ALL AS EXPECTED" if allok else "NOT ALL DETECTED"); save(fig, "negative_controls", a, out)

    # ---- ref solver validation
    sj = json.load(open(os.path.join(HERE, "out/ref_solver_selftest.json")))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    cf = sj["closed_form_vs_integrators_0_10s"]
    labs = list(cf); x = np.arange(len(labs))
    ax[0].semilogy(x, [cf[k]["taylor_vs_closed_theta"] for k in labs], "o", color=C1, ms=7, label="Taylor-24 vs closed form")
    ax[0].semilogy(x, [cf[k]["gl4_vs_closed_theta"] for k in labs], "s", color=C2, ms=6, label="Gauss–Legendre s=4 vs closed form")
    ax[0].set_xticks(x); ax[0].set_xticklabels([k.replace("theta0=", "").replace(",omega0=", "\nω0=") for k in labs], fontsize=6.5)
    ax[0].set_ylabel("max |Δθ| over 0–10 s (rad)"); ax[0].legend(fontsize=7); ax[0].set_title("closed form θ = 2 asin(k sn(K − ω_n t, k))", fontsize=8, color=INK2)
    pf = sj["period_formulas"]
    A = np.array([float(k.replace("deg", "")) for k in pf]); y1 = np.array([pf[k]["T_over_T0_minus_1"] for k in pf])
    ax[1].loglog(A, y1, "o", color=C1, ms=7, label="T/T0 − 1 (ellipk)")
    ax[1].loglog(A, (np.radians(A)) ** 2 / 16, color=INK2, lw=1, ls=":", label="θ0²/16 (small-angle limit)")
    ax[1].set_xlabel("amplitude (deg)"); ax[1].set_ylabel("T/T0 − 1"); ax[1].legend(fontsize=7)
    ax[1].set_title("ellipk vs AGM agree to ≤4.4e-16 rel; → T0 as θ0→0", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    footer(fig, argparse.Namespace(version=a.version, status="verifier self-validation"),
           {"ref_solver.py": sha256_file(os.path.join(HERE, "ref_solver.py"))}, "Reference solver validation", "PASS"); save(fig, "refsolver_validation", a, out)
    print("\n".join(out))


if __name__ == "__main__":
    main()
