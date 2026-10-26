#!/usr/bin/env python
"""Verifier visuals for a checkpoint audit (clip 03).
  .venv/bin/python verify/make_visuals_ck.py --ck ck001 --version v002 --status "final for ck001"
Reads verify/out/{check_*_<ck>.json, ref_<ck>.csv, check_repeat_<ck>.json}, the checkpoint trajectory."""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); OUT = os.path.join(HERE, "out"); VIS = os.path.join(HERE, "visuals")
sys.path.insert(0, HERE)
from tio import read_traj, read_csv  # noqa: E402
from tmodel import quat_to_R, rot_angle  # noqa: E402
C1, C2, C3, C4, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 1.4, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def J(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def footer(fig, ver, status, what):
    fig.text(0.01, 0.005, f"clip 03 torsion disc | verifier (D) | {ver} | {status} | {what}", fontsize=7.5, color=MUTED)


def tol(ax, y, txt):
    ax.axhline(y, color=C2, ls="--", lw=1); ax.text(0.01, y * 1.5, txt, color=C2, fontsize=7.5, transform=ax.get_yaxis_transform())


def floor(x):
    return np.maximum(np.abs(np.asarray(x, float)), 1e-18)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ck", default="ck001"); ap.add_argument("--version", default="v002"); ap.add_argument("--status", default="final for ck001")
    a = ap.parse_args(); ck, ver, st = a.ck, a.version, a.status
    CK = os.path.join(ROOT, "checkpoints", ck)
    ph, cv, ctl, bl, pr = J(f"check_physics_{ck}.json"), J(f"check_convergence_{ck}.json"), J(f"check_controls_{ck}.json"), J(f"check_blend_{ck}.json"), J(f"check_projected_{ck}.json")
    rp, mk = J(f"check_repeat_{ck}.json"), J(f"check_mark_shader_{ck}.json")
    dp, _ = read_traj(os.path.join(CK, "solver", "out", "trajectory.csv")); dr, _ = read_csv(os.path.join(OUT, f"ref_{ck}.csv"))
    n = len(dp["t_s"]); t = dp["t_s"]
    # ---------------- fig 1: P2 / P3 / P1 / P4 / P7
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.4)); ax = ax.ravel()
    dC = np.linalg.norm(np.stack([dp[k] - dr[k][:n] for k in ("x_m", "y_m", "z_m")], -1), axis=1)
    qp = np.stack([dp[k] for k in ("qw", "qx", "qy", "qz")], -1); qr = np.stack([dr[k][:n] for k in ("qw", "qx", "qy", "qz")], -1)
    dA = rot_angle(quat_to_R(qr), quat_to_R(qp)); dpsi = np.abs(dp["psi_rad"] - dr["psi_rad"][:n])
    ax[0].semilogy(t, floor(dC), color=C1, label="|C_prod - C_ref| [m]"); ax[0].semilogy(t, floor(dA), color=C4, label="attitude angle [rad]")
    ax[0].semilogy(t, floor(dpsi), color=C3, lw=1, label="|psi_prod - psi_ref| [rad]"); tol(ax[0], 1e-6, "P2 tolerance 1e-6")
    ax[0].set_ylim(1e-17, 1e-4); ax[0].set_title(f"P2 production RK4 vs verifier GL4: {ph['summary']['P2']}"); ax[0].set_xlabel("t [s]"); ax[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    P3 = ph["P3_analytic_limit"]; names = [k for k in P3 if isinstance(P3[k], dict)]
    vals = [max(P3[k].get("max_err_over_amp", P3[k].get("rel_omega_err")) or 1e-17, 1e-17) for k in names]
    vals2 = [max(P3[k].get("rel_omega_err", 1e-17), 1e-17) for k in names]
    y = np.arange(len(names))
    ax[1].barh(y - 0.18, np.log10(vals) + 17, left=-17, height=0.34, color=C1, label="max |x - x_exact| / amp (B2/B3); freq err (B4)")
    ax[1].barh(y + 0.18, np.log10(vals2) + 17, left=-17, height=0.34, color=C4, label="fitted omega rel. error")
    ax[1].axvline(-7, color=C2, ls="--", lw=1); ax[1].set_yticks(y); ax[1].set_yticklabels(names, fontsize=7.5); ax[1].invert_yaxis()
    ax[1].set_xlabel("log10(error)   tolerance 1e-7"); ax[1].set_title(f"P3 production runs vs closed forms: {P3['status']}"); ax[1].legend(fontsize=7, frameon=False, loc="lower left")
    P1 = cv["P1"]; vv = np.array(P1["values"]); dd = np.array(P1["combined_successive_max_diffs"])
    ax[2].loglog(vv[1:], dd, "o-", color=C1, ms=6, label=f"production RK4, p_obs = {P1['p_obs_last']:.3f}")
    ax[2].loglog(vv[1:], dd[-1] * (vv[1:] / vv[-1]) ** 4, "--", color=MUTED, lw=1, label="slope 4")
    tol(ax[2], 1e-7, "P1 tolerance 1e-7"); ax[2].set_title(f"P1 convergence (checkpoint run.py --h): {P1['status']}"); ax[2].set_xlabel("finer step h [s]")
    ax[2].set_ylabel("max |X_h - X_h/2| (m / rad)"); ax[2].legend(fontsize=7.5, frameon=False)
    for key, col, lab in (("P4_energy_conservative", C1, "conservative variant |E - E0| / E_osc"), ("P4_energy_production_run", C4, "damped run |E + W - E0| / E_osc")):
        s = ph[key]["_series"]; ax[3].semilogy(s["t"], floor(s["dE"] if "conservative" in key else s["bal"]), color=col, label=lab)
    tol(ax[3], 1e-8, "P4 cons. 1e-8"); ax[3].axhline(1e-7, color=C2, ls=":", lw=1)
    ax[3].set_ylim(1e-17, 1e-5); ax[3].set_title(f"P4 energy (my recomputation of production states): {ph['summary']['P4']}"); ax[3].set_xlabel("t [s]"); ax[3].legend(fontsize=7.5, frameon=False, loc="lower right")
    s = ph["P4_energy_production_run"]["_series"]
    ax[4].plot(s["t"], 1 + np.array(s["dE"]), color=C4); ax[4].set_title("damped run: (E - E_static)/E_osc (Q_t = 1000 declared)"); ax[4].set_xlabel("t [s]")
    ax[5].plot(t, dp["tension_N"], color=C1); ax[5].set_title(f"P7 wire tension (column = recomputed to {ph['P7_forces']['column_vs_recomputed_max_rel']:.1e} Mg): {ph['summary']['P7']}")
    ax[5].set_xlabel("t [s]"); ax[5].set_ylabel("T [N]  (must be >= 0)")
    import hashlib
    sh = lambda f: hashlib.sha256(open(os.path.join(CK, f), "rb").read()).hexdigest()[:8]
    fig.suptitle(f"{ck}: physics audit P1-P4, P7 (checkpoint copies: params {sh('params/physics.json')}..., trajectory {sh('solver/out/trajectory.csv')}...)", x=0.01, ha="left", fontsize=12)
    footer(fig, ver, st, "check_physics.py, check_convergence.py"); fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(os.path.join(VIS, f"{ver}_{ck}_P1_P4_P7.png"), dpi=130); plt.close(fig)
    # ---------------- fig 2: P5 / P6 blend + P9
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.0)); ax = ax.ravel()
    S = bl["_series"]
    ax[0].semilogy(S["t"], floor(S["dC"]), color=C1, label="|C_blend - C_ref| [m]"); ax[0].semilogy(S["t"], floor(S["dR"]), color=C4, label="rotation angle [rad]")
    ax[0].semilogy(S["t"], floor(S["dpsi"]), color=C3, lw=1, label="|psi_blend - psi_ref| [rad]"); tol(ax[0], 1e-5, "P6 tolerance 1e-5")
    ax[0].set_ylim(1e-12, 1e-3); ax[0].set_title(f"P6 .blend vs verifier ref: {bl['summary']['P6']}"); ax[0].set_xlabel("t [s]")
    ax[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    ax[1].semilogy(S["t"], floor(S["gtop"]), color=C1, label="wire top end - pivot"); ax[1].semilogy(S["t"], floor(S["gbot"]), color=C4, label="wire bottom end - B")
    tol(ax[1], 1e-5, "P5 attachment 1e-5"); ax[1].set_ylim(1e-13, 1e-3); ax[1].set_title(f"P5 wire endpoints (evaluated GN mesh): {bl['summary']['P5']}"); ax[1].set_xlabel("t [s]")
    ax[1].legend(fontsize=7.5, frameon=False, loc="lower right")
    p5, p6 = bl["P5_constraints_saved_blend"], bl["P6_blend_vs_solver"]
    rows = [("anchor (TOR_Anchor) vs pivot_world_m", p5["anchor_world_vs_expected_max_m"], 1e-6), ("anchor drift", p5["anchor_drift_from_frame0_max_m"], 1e-6),
            ("static objects drift", p5["static_objects_max_translation_drift_m"], 1e-6), ("wire length variation", p5["wire_length_variation_m"], 1e-5),
            ("|B - pivot| - ell", p5["attach_to_anchor_vs_L_max_m"], 1e-5), ("disc mesh in own frame", p5["disc_mesh_in_own_frame_max_dev_m"], 1e-6),
            ("C vs ref", p6["C_vs_ref_max_m"], 1e-5), ("rotation vs ref [rad]", p6["rotation_vs_ref_max_rad"], 1e-5), ("psi vs ref [rad]", p6["psi_vs_ref_max_rad"], 1e-5),
            ("between-key probe C", p6["interp_probe_C_max_m"], 1e-5), ("between-key probe rotation", p6["interp_probe_rotation_max_rad"], 1e-5),
            ("frame-0 pose vs params IC", bl["P8_blend_part"]["disc_C_frame0_vs_IC_m"], 1e-6)]
    y = np.arange(len(rows)); v = np.array([max(r[1], 1e-17) / r[2] for r in rows])
    ax[2].barh(y, np.log10(v) + 12, left=-12, color=[C1 if x < 1 else C2 for x in v], height=0.6); ax[2].axvline(0, color=INK, lw=1)
    ax[2].set_yticks(y); ax[2].set_yticklabels([r[0] for r in rows], fontsize=7.5); ax[2].invert_yaxis(); ax[2].set_xlabel("log10(metric / tolerance)")
    ax[2].set_title(f"P5/P6/P8b metrics; mark shader rigid: {mk['status'] if mk else 'n/a'}")
    V = ctl["variants"]; names = list(V)
    ax[3].bar(np.arange(len(names)), [np.log10(max(V[k]["rel_omega_err"], 1e-17)) for k in names], color=C1, width=0.6, label="|omega_fit/omega_formula - 1|")
    ax[3].bar(np.arange(len(names)) + 0.3, [np.log10(max(V[k].get("rel_gamma_err") or 1e-17, 1e-17)) for k in names], color=C4, width=0.3, label="|gamma_fit/gamma - 1|")
    ax[3].axhline(-4, color=C2, ls="--", lw=1); ax[3].set_xticks(np.arange(len(names))); ax[3].set_xticklabels(names, fontsize=8)
    ax[3].set_ylabel("log10 rel. error (tolerance 1e-4)"); ax[3].set_title(f"P9 controls via make_variant (re-equilibrated): {ctl['status']}"); ax[3].legend(fontsize=7, frameon=False)
    rat = ctl.get("ratios", {})
    txt = "\n".join(f"{k}: T_var/T_base = {v['period_ratio_var_over_base']:.6f}" for k, v in rat.items() if k != "L_plus50")
    ax[4].axis("off"); ax[4].text(0, 0.95, "P9 period ratios (twist regime; L_plus50: swing regime)\n" + txt +
                                  "\nexpected: kappa x4 -> 0.5, I x4 -> 2, damping x10 -> ~1\nL x1.5: swing omega 6.873 -> " +
                                  f"{V['L_plus50']['omega_expected']:.4f} rad/s (formula) / {V['L_plus50']['omega_fit']:.4f} (fit)\n\nfull-IC variant runs vs verifier GL4 (max |dC|):\n" +
                                  "\n".join(f"  {k}: {V[k]['full_ic_vs_verifier']['C_max_m']:.1e} m, {V[k]['full_ic_vs_verifier']['attitude_max_rad']:.1e} rad" for k in names),
                                  va="top", fontsize=8.5, family="monospace")
    ax[5].axis("off")
    rpst = rp["status"] if rp else "pending"
    ax[5].text(0, 0.95, f"P10 two clean rebuilds\n(staged checkpoint inputs,\nfresh dirs): {rpst}\nkeys + evaluated geometry\nbitwise identical; also\nrebuild = checkpoint .blend\n" + ("\n".join(f"  {k}: {v}" for k, v in (rp or {}).get("summary", {}).items())), va="top", fontsize=9, family="monospace")
    fig.suptitle(f"{ck}: saved torsion.blend (reopened, frames + 4 subframes + between-key probes), P9 controls, P10", x=0.01, ha="left", fontsize=12)
    footer(fig, ver, st, "check_blend.py, check_mark_shader.py, check_controls.py, compare_dumps.py"); fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(os.path.join(VIS, f"{ver}_{ck}_P5_P6_P9_P10.png"), dpi=130); plt.close(fig)
    # ---------------- fig 3: projected residuals
    R = pr["results"]["verifier_ref"]["_series"]; tt = np.array(R["t"])
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for a_ in ax:
        a_.axvspan(0, 2.5, color="#e8f0fb", lw=0)
    ax[1].axvspan(2.5, 7.5, color="#eef5ea", lw=0)
    ax[0].plot(tt, R["du"], color=C1, label="u model - u track"); ax[0].plot(tt, R["dv"], color=C4, label="v model - v track")
    c = pr["comparison"]
    ax[0].set_ylabel("px"); ax[0].set_title(f"B (top-face centre) through camera/project.py: anchor 0-2.5 s RMS {c['B']['mine']:.4f} px (claim {c['B']['claim']:.4f}); "
                                            f"after {c['B_after']['mine']:.2f} px -> R2 {pr['R2_B_status']} (target 2 px)", fontsize=9.5, loc="left")
    ax[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    ax[1].plot(tt, R["dpsi"], color=C1); ax[1].axhline(0, color=MUTED, lw=0.6)
    ax[1].set_ylabel("deg"); ax[1].set_title(f"twist: line azimuth track - model (mod 180): 0-7.5 s RMS {c['psi']['mine']:.4f} deg (claim {c['psi']['claim']:.4f}); "
                                             f"3 sigma = {pr['R2_psi_target_deg']['three_sigma']:.2f} deg -> {pr['R2_psi_status']}; after {c['psi_after']['mine']:.2f} deg", fontsize=9.5, loc="left")
    ax[2].plot(tt, R["nrm"], color=C1); ax[2].set_ylabel("deg"); ax[2].set_xlabel("t [s]")
    ax[2].set_title(f"tilt: angle(model normal, measured normal): 0-2.5 s RMS {c['nrm']['mine']:.4f} deg (claim {c['nrm']['claim']:.4f})", fontsize=9.5, loc="left")
    fig.suptitle(f"{ck}: independent recompute of projected residuals (my GL4 trajectory, measurement/ files, camera of record) -> claims {c['status']}",
                 x=0.01, ha="left", fontsize=12)
    footer(fig, ver, st, "check_projected.py; shaded = per-mode anchor windows (B/tilt 0-2.5 s, twist 0-7.5 s)"); fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(os.path.join(VIS, f"{ver}_{ck}_projected_residuals.png"), dpi=130); plt.close(fig)
    # ---------------- fig 4: scorecard
    rows = [("P1 convergence", cv["P1"]["status"], f"finest pair {cv['P1']['e_last_m']:.1e}; order {cv['P1']['p_obs_last']:.3f} (theory 4)"),
            ("P2 independent benchmark", ph["summary"]["P2"], f"C {ph['P2_independent_benchmark']['C_max_m']:.1e} m, att {ph['P2_independent_benchmark']['attitude_max_rad']:.1e} rad, psi {ph['P2_independent_benchmark']['psi_max_rad']:.1e} rad"),
            ("P3 analytic limits", P3["status"], "pure twist 0.5 rad undamped/damped, 1e-5 rad lowest swing mode"),
            ("P4 energy", ph["summary"]["P4"], f"cons. drift {ph['P4_energy_conservative']['max_rel_drift_E']:.1e}; damped balance {ph['P4_energy_production_run']['max_rel_balance_residual']:.1e}"),
            ("P5 saved-blend constraints", bl["summary"]["P5"], f"wire ends <= {max(p5['wire_top_end_gap_max_m'], p5['wire_bottom_end_gap_max_m']):.1e} m; anchor {p5['anchor_world_vs_expected_max_m']:.1e} m; mark shader {mk['status'] if mk else '-'}"),
            ("P6 blend vs solver", bl["summary"]["P6"], f"C {p6['C_vs_ref_max_m']:.1e} m, rot {p6['rotation_vs_ref_max_rad']:.1e} rad, probe {p6['interp_probe_rotation_max_rad']:.1e} rad"),
            ("P7 forces", ph["summary"]["P7"], f"tension {ph['P7_forces']['recomputed_tension_min_N']:.3f}-{ph['P7_forces']['recomputed_tension_max_N']:.3f} N"),
            ("P8 IC / time", ph["summary"]["P8"] + " / " + bl["summary"]["P8_blend"], f"IC dev {max(ph['P8_initial_conditions_time']['ic_deviation'].values()):.1e}; time scale |s-1| {ph['P8_initial_conditions_time']['time_scale_estimate']['rel']:.1e}"),
            ("P9 control response", ctl["status"], "kappa x4, m x4 (I x4), ell x1.5, damping x10 via make_variant"),
            ("P10 repeatability", rpst, "two clean rebuilds, bitwise dumps"),
            ("R2 anchor B (fit claim recomputed)", pr["R2_B_status"], f"{c['B']['mine']:.3f} px <= 2 px (claim reproduced)"),
            ("R2 anchor twist (fit claim recomputed)", "MISS" if "MISS" in pr["R2_psi_status"] else pr["R2_psi_status"], f"{c['psi']['mine']:.3f} deg > 3 sigma {pr['R2_psi_target_deg']['three_sigma']:.2f} deg (claim reproduced; disclosed by dynamics)")]
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(rows) + 1.2)); ax.axis("off")
    for i, (a1, b1, c1) in enumerate(rows):
        yy = 1 - (i + 1) / (len(rows) + 1)
        ax.text(0.0, yy, a1, fontsize=9.5, transform=ax.transAxes); ax.text(0.30, yy, b1, fontsize=9.5, fontweight="bold", transform=ax.transAxes,
                                                                          color=C1 if b1.startswith("PASS") and "FAIL" not in b1 else C2)
        ax.text(0.44, yy, c1, fontsize=8.5, transform=ax.transAxes, color=INK)
    ax.set_title(f"{ck} verifier scorecard (acceptance v1, no tolerance relaxed)", loc="left", fontsize=12)
    footer(fig, ver, st, "verify/out/*_" + ck + ".json")
    fig.savefig(os.path.join(VIS, f"{ver}_{ck}_scorecard.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
    print("written", VIS)


if __name__ == "__main__":
    main()
