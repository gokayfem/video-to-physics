#!/usr/bin/env python
"""Checkpoint-audit visuals (clip 02).  .venv/bin/python verify/make_visuals_ck.py --ck ck001 --version v002 --status "final for ck001"
Reads verify/out/<ck>/ reports only."""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from vio import read_csv  # noqa: E402
C1, C2, C3, C4, INK, MUT = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.color": "#e4e3df", "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "lines.linewidth": 1.5})


def foot(fig, a, what):
    fig.text(0.01, 0.005, f"clip 02 vertical spring | verifier (D) | {a.version} | {a.ck} | status: {a.status} | {what}", fontsize=7.5, color=MUT)


def tol(ax, y, txt):
    ax.axhline(y, color=C2, ls="--", lw=1); ax.text(ax.get_xlim()[0], y * 1.4, txt, color=C2, fontsize=8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ck", default="ck001"); ap.add_argument("--version", default="v002"); ap.add_argument("--status", default="final for ck001")
    a = ap.parse_args()
    O = os.path.join(HERE, "out", a.ck); V = os.path.join(HERE, "visuals"); C = os.path.join(O, "ckcopy")
    ph = json.load(open(os.path.join(O, f"check_physics_{a.ck}.json"))); ph3 = json.load(open(os.path.join(O, f"check_physics_{a.ck}_3d.json")))
    prod, _ = read_csv(os.path.join(C, "solver", "out", "trajectory.csv")); ref, _ = read_csv(os.path.join(O, "ref_trajectory.csv"))
    n = len(prod["t_s"])
    # ---------------- P2 / P3
    fig, ax = plt.subplots(2, 2, figsize=(14, 8)); ax = ax.ravel()
    ax[0].plot(ref["t_s"][:n], ref["x_m"][:n] * 1e3, color=C1, lw=3, alpha=0.45, label="verifier GL4: H_x")
    ax[0].plot(prod["t_s"], prod["H_x_m"] * 1e3, color=INK, lw=0.9, label="production RK4: H_x")
    ax[0].plot(ref["t_s"][:n], (ref["z_m"][:n] + 0.4167) * 1e3 * 20, color=C3, lw=3, alpha=0.45, label="verifier: (H_z + 0.4167) x20")
    ax[0].plot(prod["t_s"], (prod["H_z_m"] + 0.4167) * 1e3 * 20, color=INK, lw=0.9, ls=":", label="production: same")
    ax[0].set_title("Hinge H(t): production vs independent solver (overlaid)"); ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("mm"); ax[0].legend(fontsize=7, frameon=False, ncol=2)
    p2 = ph["P2_independent_benchmark"]; s = p2["series"]
    ax[1].semilogy(s["t"], np.maximum(s["dpos"], 1e-18), color=C1, label=f"nominal (damped): max {p2['max_pos_diff_m']:.1e} m")
    s3 = ph3["P2_independent_benchmark"]["series"]
    ax[1].semilogy(s3["t"], np.maximum(s3["dpos"], 1e-18), color=C4, label=f"extra 3-D undamped variant: max {ph3['P2_independent_benchmark']['max_pos_diff_m']:.1e} m")
    ax[1].set_ylim(1e-16, 1e-4); tol(ax[1], 1e-6, "P2 tolerance 1e-6 m")
    ax[1].set_title(f"P2 |H_prod - H_verifier| (attitude max {p2['max_attitude_diff_rad']:.1e} rad) -> {p2['status']}"); ax[1].set_xlabel("t [s]"); ax[1].legend(fontsize=7.5, frameon=False)
    p3 = ph["P3_analytic_limit"]; names, vals = [], []
    for k, v in p3.items():
        if isinstance(v, dict):
            val = v.get("max_err_over_amp", v.get("max_dev_over_ell_eq", v.get("rel_omega_err")))
            names.append(f"{k} ({'pos/amp' if 'max_err_over_amp' in v else ('dev/l_eq' if 'max_dev_over_ell_eq' in v else 'rel freq')})"); vals.append(max(val, 1e-17))
            if "rel_omega_err" in v and "max_err_over_amp" in v:
                names.append(f"{k} (rel freq)"); vals.append(max(v["rel_omega_err"], 1e-17))
    y = np.arange(len(names)); ax[2].barh(y, vals, color=C1, height=0.55); ax[2].set_xscale("log"); ax[2].set_xlim(1e-18, 1e-5)
    ax[2].axvline(1e-7, color=C2, ls="--"); ax[2].set_yticks(y); ax[2].set_yticklabels(names, fontsize=8)
    ax[2].set_title(f"P3 production runs vs closed forms (B1 rest, B2 axial exact, B5 rigid modes; tol 1e-7) -> {p3['status']}")
    e = p3["B2_axial"]["series"]
    ax[3].plot(e["t"], np.array(e["err"]), color=C1); ax[3].set_title("P3 B2: (z_prod - z_exact)/amplitude, damped axial release (a = 2% l_eq)")
    ax[3].set_xlabel("t [s]")
    foot(fig, a, "production = <ck>/solver/run.py run as subprocess; verifier = verify/ref_solver.py (Gauss-Legendre s=4)")
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(os.path.join(V, f"{a.version}_P2_P3_benchmark.png"), dpi=130); plt.close(fig)
    # ---------------- P1 / P4 / P7
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    cf = json.load(open(os.path.join(O, f"check_convergence_{a.ck}_full.json")))["P1"]; cj = json.load(open(os.path.join(O, f"check_convergence_{a.ck}.json")))["P1"]
    v = np.array(cf["values"]); d = np.array(cf["pos_successive_max_diffs_m"])
    ax[0].loglog(v[1:], d, "o-", color=C1, ms=6, label="max |H_h - H_2h| (production --h)")
    ax[0].loglog(v[1:], d[-1] * (v[1:] / v[-1]) ** 4, "--", color=MUT, lw=1, label="slope 4")
    ax[0].axhline(1e-7, color=C2, ls="--", lw=1)
    ax[0].set_title(f"P1: orders {', '.join(f'{p:.3f}' for p in cf['observed_order'])}; finest pair {cj['e_last_m']:.1e} m -> {cj['status']}", fontsize=9)
    ax[0].set_xlabel("finer step of pair [s]"); ax[0].legend(fontsize=7.5, frameon=False)
    p4 = ph["P4_energy"]; s4 = p4["series"]; cr = p4["conservative_run"]
    ax[1].semilogy(s4["t"], np.abs(s4["balance"]) + 1e-18, color=C1, label=f"damped: |E + W_diss - E0|/E_osc (max {p4['balance_recomputed_E_max_over_E0']:.1e})")
    ax[1].semilogy(cr["series"]["t"], np.abs(cr["series"]["drift"]) + 1e-18, color=C3, label=f"undamped run: |E - E0|/E_osc (max {cr['max_drift_over_E0']:.1e})")
    c3 = ph3["P4_energy"]["conservative_run"]
    ax[1].semilogy(c3["series"]["t"], np.abs(c3["series"]["dLz"]) + 1e-18, color=C4, lw=1, label=f"3-D undamped: |dLz|/L_ref (max {c3['max_Lz_drift_over_Lref']:.1e})")
    ax[1].set_ylim(1e-18, 1e-5); ax[1].axhline(1e-7, color=C2, ls="--", lw=1); ax[1].axhline(1e-8, color=C2, ls=":", lw=1)
    ax[1].set_title(f"P4 energy (tol 1e-7 balance, 1e-8 drift) -> {p4['status']}"); ax[1].set_xlabel("t [s]"); ax[1].legend(fontsize=7, frameon=False)
    p7 = ph["P7_forces"]
    ax[2].plot(p7["series"]["t"], p7["series"]["T"], color=C1); ax[2].axhline(0, color=C2, ls="--", lw=1)
    ax[2].set_ylim(-5, 60); ax[2].set_title(f"P7 spring tension k(|PH|-L0): min {p7['min_tension_recomputed_N']:.2f} N (>= 0) -> {p7['status']}")
    ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("N")
    foot(fig, a, "P4 balance: verifier GL4 propagation of P_diss from every production state; drift/Lz from production runs with damping zeroed")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(V, f"{a.version}_P1_P4_P7.png"), dpi=130); plt.close(fig)
    # ---------------- P5 / P6
    bl = json.load(open(os.path.join(O, f"check_blend_{a.ck}.json"))); dz = np.load(os.path.join(O, f"check_blend_{a.ck}.npz"))
    t = dz["t"]; inc = dz["in_clip"]; err = np.linalg.norm(dz["eye"] - dz["eye_ref"], axis=1)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    sub = (np.round(t * 96) % 4) != 0
    ax[0].semilogy(t[inc & ~sub], err[inc & ~sub] + 1e-12, ".", ms=3, color=C1, label="whole frames")
    ax[0].semilogy(t[inc & sub], err[inc & sub] + 1e-12, ".", ms=2, color=C2, label="subframes .25/.5/.75")
    ax[0].axhline(1e-5, color=INK, ls="--", lw=1); ax[0].set_ylim(1e-11, 1e-3)
    p6 = bl["P6_blend_vs_solver"]
    ax[0].set_title(f"P6 hinge (blend) vs verifier: max {p6['eye_vs_ref_max_m']:.1e} m, rot {p6['weight_rotation_vs_ref_max_rad']:.1e} rad -> {p6['status']}", fontsize=8.5)
    ax[0].set_xlabel("t [s]"); ax[0].legend(fontsize=7.5, frameon=False)
    p5 = bl["P5_constraints_saved_blend"]
    items = [("anchor drift", p5["anchor_drift_from_frame0_max_m"], 1e-6), ("static objects drift", p5["static_objects_max_translation_drift_m"], 1e-6),
             ("static spring parts drift", p5["static_spring_vertices_max_drift_m"], 1e-6), ("top loop contact gap at A", p5["top_loop_contact_gap_max_m"], 1e-5),
             ("lower loop contact gap at H", p5["lower_loop_contact_gap_max_m"], 1e-5), ("H out of lower-loop plane", p5["hinge_out_of_lower_loop_plane_max_m"], 1e-5),
             ("H drift in lower-loop frame", p5["hinge_in_lower_loop_frame_drift_max_m"], 1e-5), ("contact sliding along loop", p5["lower_contact_sliding_along_loop_m"], 1e-5),
             ("coil-to-P_eff distance variation", p5["moving_coil_centreline_to_P_eff_variation_m"], 1e-5), ("HookLower vs weight origin", p5["hook_lower_vs_weight_origin_max_m"], 1e-5),
             ("coil winding variation [turns]", p5["coil_winding_variation_turns"], 1e-3)]
    y = np.arange(len(items))[::-1]
    ax[1].barh(y, [max(v / tl, 1e-9) for _, v, tl in items], color=C1, height=0.55); ax[1].set_xscale("log"); ax[1].set_xlim(1e-9, 10); ax[1].axvline(1, color=C2, ls="--")
    ax[1].set_yticks(y); ax[1].set_yticklabels([f"{n_} ({v:.1e})" for n_, v, _ in items], fontsize=7.5)
    ax[1].set_title(f"P5 constraints / tolerance (frames + 4 subframes) -> {p5['status']}", fontsize=9)
    wd = dz["winding"]; ax[2].plot(t[inc], wd[inc] - 21, color=C1, label="coil winding - 21 turns (centreline)")
    cc = dz["coil_counts"]; ax[2].plot(t[inc], cc[inc].mean(1) - 21.25, color=C3, label="half-plane crossings mean - 21.25 (mesh)")
    ax[2].set_ylim(-0.5, 0.5); ax[2].legend(fontsize=7.5, frameon=False)
    ax[2].set_title(f"coils constant: {p5['coil_counts_constant']}; topology {p5['topology_frame0_v_e_f']} constant: {p5['topology_constant']}", fontsize=8)
    ax[2].set_xlabel("t [s]")
    foot(fig, a, "reopened checkpoint spring.blend, evaluated depsgraph geometry (GN attributes vsp_part/vsp_cl/vsp_w)")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(V, f"{a.version}_P5_P6_blend.png"), dpi=130); plt.close(fig)
    # ---------------- P9 + P10 + negative controls on the production blend
    p9 = json.load(open(os.path.join(O, f"check_controls_{a.ck}.json")))
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.2))
    rows = []
    for vn, ent in p9["variants"].items():
        for k, r in ent["runs"].items():
            if k == "nominal_ic_diagnostic":
                continue
            if k == "production_ic":
                val = r.get("max_pos_diff_vs_verifier_m"); val = 1e3 if (val is None or isinstance(val, str)) else val
                lab = "re-equilibrated variant (make_variant)" if r.get("variant_cmd") else "production ICs"
                rows.append((f"{vn}: {lab} vs verifier [m] (tol 1e-6)", val / 1e-6, r["status"]))
            else:
                rows.append((f"{vn}: {k} rel freq (tol 1e-4)", r["rel_omega"] / 1e-4, r["status"]))
                if r.get("rel_gamma") is not None:
                    rows.append((f"{vn}: {k} rel decay rate", r["rel_gamma"] / 1e-4, r["status"]))
    y = np.arange(len(rows))[::-1]
    ax[0].barh(y, [max(min(v, 1e4), 1e-9) for _, v, _ in rows], color=[C1 if s_ == "PASS" else C2 for *_, s_ in rows], height=0.6)
    ax[0].set_xscale("log"); ax[0].set_xlim(1e-9, 1e5); ax[0].axvline(1, color=INK, ls="--"); ax[0].set_yticks(y); ax[0].set_yticklabels([r[0] for r in rows], fontsize=7)
    ax[0].set_title(f"P9 controls (production solver + build/bake): metric / tolerance -> {p9['status']} (orange = FAIL)", fontsize=9)
    neg = []
    for f in ("hook_gap_50um", "anchor_drift_2um", "weight_rot_1e-4", "coil_extra_turn"):
        r = json.load(open(os.path.join(O, "negctl", f"blend_{f}.json")))["summary"]; neg.append((f"{a.ck} blend + {f}", r["P5"], r["P6"]))
    p10 = {k: json.load(open(os.path.join(O, "p10", f"compare_{k}.json")))["status"] for k in ("ck_vs_A", "ck_vs_B", "A_vs_B", "ck_vs_coilfault")}
    ax[1].axis("off")
    txt = ["Injected faults in COPIES of the checkpoint spring.blend (must FAIL):"] + [f"   {n_}:  P5 {a5}  P6 {a6}" for n_, a5, a6 in neg]
    txt += ["", "P10 repeatability (two verifier clean rebuilds from staged inputs):"] + [f"   {k}: {v}" for k, v in p10.items()]
    txt += ["   (ck_vs_coilfault is a negative control: must FAIL)"]
    diag = [(vn, e["runs"]["nominal_ic_diagnostic"]) for vn, e in p9["variants"].items() if "nominal_ic_diagnostic" in e["runs"]]
    if diag:
        txt += ["", "P9 diagnostic: variants started from the NOMINAL state (not judged):"]
        txt += [f"   {vn}: rc {d['rc']} {','.join(d['raised']) or 'no error'}; verifier min T(0-0.5 s) {d['verifier_min_tension_first_0p5s_N']:.1f} N" for vn, d in diag]
    rc = [(vn, e["runs"]["production_ic"].get("ic_rule_check")) for vn, e in p9["variants"].items() if e["runs"].get("production_ic", {}).get("ic_rule_check")]
    if rc:
        txt += ["", "Re-equilibrated ICs vs the CHANGELOG rule (verifier check):"]
        txt += [f"   {vn}: consistent={c['consistent']}  psi/phi vs kappa {c['rel_kappa']:.0e}  T0 {c['tension0_N']:.1f} N" for vn, c in rc]
    ax[1].text(0.0, 1.0, "\n".join(txt), va="top", family="monospace", fontsize=8)
    foot(fig, a, "P9 variants are edited COPIES of params run through <ck>/solver/run.py and <ck>/scene/build.py --bake --look")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(V, f"{a.version}_P9_P10_negctl.png"), dpi=130); plt.close(fig)
    print("visuals written")


if __name__ == "__main__":
    main()
