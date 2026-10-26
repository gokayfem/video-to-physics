#!/usr/bin/env python
"""Independent recomputation of the projected bob residuals of a checkpoint (verifier-owned).

Uses ONLY data: <ck>/solver/out/trajectory.csv (theta at rows 4n = frame n), <ck>/params/physics.json (L_c),
<ck>/params/camera.json through <ck>/camera/project.py (camera of record, per-frame drift camera),
<ck>/measurement/bob_track.csv (method A u_px, v_px). Nothing from solver/ code is imported.
Compares with <ck>/solver/out/fit_projected/fit_projected.json metrics (claims under test).

  .venv/bin/python verify/audit_fit_residuals.py --ck verify/out/ck001/ckcopy --out verify/out/ck001/fit_audit_ck001.json \
      --fig verify/visuals/v003_fit_residual_map.png --label "ck001" --hashes "..."
"""
import argparse, csv, hashlib, importlib.util, json, math, os
import numpy as np


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--fig"); ap.add_argument("--label", default=""); ap.add_argument("--caption", default="")
    ap.add_argument("--fit-json", default="solver/out/fit_projected/fit_projected.json", help="claims under test, relative to --ck")
    ap.add_argument("--anchor", type=float, default=3.6, help="anchor window end t_a (s), window t <= t_a")
    ap.add_argument("--no-tilt", action="store_true", help="ignore physics level_tilt_rad (diagnostic)")
    a = ap.parse_args()
    ck = os.path.abspath(a.ck)
    f_traj = f"{ck}/solver/out/trajectory.csv"; f_phys = f"{ck}/params/physics.json"; f_cam = f"{ck}/params/camera.json"
    f_proj = f"{ck}/camera/project.py"; f_track = f"{ck}/measurement/bob_track.csv"; f_fit = f"{ck}/{a.fit_json}"
    inputs = {os.path.relpath(p, ck): sha(p) for p in (f_traj, f_phys, f_cam, f_proj, f_track, f_fit)}
    spec = importlib.util.spec_from_file_location("ck_project", f_proj); prj = importlib.util.module_from_spec(spec); spec.loader.exec_module(prj)
    P = json.load(open(f_phys)); Lc = P["L_c"]["value"]
    T = np.genfromtxt(f_traj, delimiter=",", names=True)
    assert np.all(np.abs(T["t_s"][::4] - np.arange(241) / 24) < 1e-12)
    th = T["theta_rad"][::4]
    rows = list(csv.DictReader(open(f_track)))
    n = np.array([int(r["frame"]) for r in rows]); um = np.array([float(r["u_px"]) for r in rows]); vm = np.array([float(r["v_px"]) for r in rows])
    su = np.array([float(r["sigma_u_px"]) for r in rows]); sv = np.array([float(r["sigma_v_px"]) for r in rows])
    assert np.array_equal(n, np.arange(241))
    X = np.stack([Lc * np.sin(th), np.zeros_like(th), -Lc * np.cos(th)], 1)   # world, true gravity
    tilt = 0.0 if (a.no_tilt or "level_tilt_rad" not in P) else P["level_tilt_rad"]["value"]
    # static scene (stand+cameras) rotated by +tilt about +Y through P: R_tilt = RotY(+tilt); stand-frame point = R_tilt^T X
    ct, st_ = math.cos(tilt), math.sin(tilt); Rt = np.array([[ct, 0, st_], [0, 1, 0], [-st_, 0, ct]])
    X = X @ Rt   # row-vector form of R_tilt^T x
    cam_Leff = json.load(open(f_cam))["scale"]["L_eff_m"]; phys_Leff = P["L_eff_m"]["value"]; gauge = cam_Leff / phys_Leff
    X = X * gauge   # metres of physics gauge -> camera.json metres (same L_eff units)
    res = {}
    for kind in ("reference", "locked"):
        uv = np.array([np.array(prj.project(X[i], prj.load_camera(frame=i, kind=kind, params_path=f_cam))).ravel() for i in range(241)])
        du, dv = uv[:, 0] - um, uv[:, 1] - vm
        r = np.hypot(du, dv)
        t = n / 24.0
        def rms(m): return float(np.sqrt(np.mean(r[m] ** 2)))
        def rmsuv(m): return [float(np.sqrt(np.mean(du[m] ** 2))), float(np.sqrt(np.mean(dv[m] ** 2)))]
        fit, hold, anc = t < 7.5, t >= 7.5, t <= a.anchor + 1e-9
        m4 = (n % 4 == 0)
        res[kind] = dict(rms_fit_px=rms(fit), rms_holdout_px=rms(hold), rms_fit_u_v_px=rmsuv(fit), rms_holdout_u_v_px=rmsuv(hold),
                         rms_anchor_px=rms(anc), anchor_t_s=a.anchor, rms_anchor_u_v_px=rmsuv(anc), rms_all_px=rms(t >= 0), max_abs_px=float(max(np.abs(du).max(), np.abs(dv).max())),
                         max_abs_px_definition="max over frames of max(|du|,|dv|) (matches fit_projected.json usage)", max_euclid_px=float(r.max()),
                         frame_of_max=int(np.argmax(r)), rms_fit_mod4_px=rms(fit & m4), rms_holdout_mod4_px=rms(hold & m4),
                         mean_du_fit_px=float(du[fit].mean()), mean_dv_fit_px=float(dv[fit].mean()),
                         mean_du_anchor_px=float(du[anc].mean()), mean_dv_anchor_px=float(dv[anc].mean()),
                         rms_anchor_after_removing_mean_px=float(np.sqrt(np.mean((du[anc]-du[anc].mean())**2+(dv[anc]-dv[anc].mean())**2))))
        if kind == "reference":
            series = dict(t=t, um=um, vm=vm, u=uv[:, 0], v=uv[:, 1], du=du, dv=dv, r=r)
    claims = json.load(open(f_fit))
    cm = claims["metrics"]
    cmp = {}
    for k in ("rms_fit_px", "rms_holdout_px", "max_abs_px"):
        mine = res["reference"][k]; theirs = cm[k]
        cmp[k] = dict(verifier=mine, claimed=theirs, abs_diff=abs(mine - theirs), rel_diff=abs(mine - theirs) / theirs)
    for k, kk in (("rms_fit_px", "rms_fit_px"), ("rms_holdout_px", "rms_holdout_px")):
        c2 = cm.get("contrast_locked_camera_no_drift_model", {}).get(kk)
        if c2 is not None:
            cmp["locked_" + k] = dict(verifier=res["locked"][k], claimed=c2, abs_diff=abs(res["locked"][k] - c2))
    if "rms_anchor_px" in cm:
        cmp["rms_anchor_px"] = dict(verifier=res["reference"]["rms_anchor_px"], claimed=cm["rms_anchor_px"], abs_diff=abs(res["reference"]["rms_anchor_px"] - cm["rms_anchor_px"]))
    if "rms_anchor_px" in cm.get("contrast_locked_camera_no_drift_model", {}):
        cmp["locked_rms_anchor_px"] = dict(verifier=res["locked"]["rms_anchor_px"], claimed=cm["contrast_locked_camera_no_drift_model"]["rms_anchor_px"], abs_diff=abs(res["locked"]["rms_anchor_px"] - cm["contrast_locked_camera_no_drift_model"]["rms_anchor_px"]))
    agree = all(v["abs_diff"] < 0.05 for v in cmp.values())
    sig = float(np.median(np.hypot(su, sv)))
    # internal derived-parameter consistency (formulas declared in physics.json itself)
    r_b = P["bob_radius"]["value"]; m = P["bob_mass"]["value"]; ratio = P["ratio_Lc_over_r"]["value"]; g = P["g"]["value"]
    Ieq = m * Lc ** 2 + 0.4 * m * r_b ** 2; w0 = math.sqrt(m * g * Lc / P["I_P"]["value"])
    derived = dict(
        L_c_vs_L_eq_formula=abs(Lc - P["L_eq"]["value"] / (1 + 0.4 / ratio ** 2)),
        r_vs_Lc_over_ratio=abs(r_b - Lc / ratio),
        m_vs_rho_volume_rel=abs(m / (4 / 3 * math.pi * r_b ** 3 * P["rho_bob"]["value"]) - 1),
        I_P_vs_formula_rel=abs(P["I_P"]["value"] / Ieq - 1),
        L_eq_vs_I_over_mLc_rel=abs(P["L_eq"]["value"] / (Ieq / (m * Lc)) - 1),
        c_vs_I_w0_over_Q_rel=abs(P["damping"]["c"]["value"] / (P["I_P"]["value"] * w0 / P["damping"]["Q"]["value"]) - 1),
        L_eff_m_vs_Lc_over_k_rel=abs(P["L_eff_m"]["value"] / (Lc / P["L_c_in_L_eff_units"]["value"]) - 1),
        fit_params_equal_physics=dict(L_eq=claims["params"]["L_eq"] == P["L_eq"]["value"], theta0=claims["params"]["theta0"] == P["theta_init"]["value"],
                                      omega0=claims["params"]["omega0"] == P["omega_init"]["value"], k=claims["params"]["k"] == P["L_c_in_L_eff_units"]["value"]),
        camera_json_sha256_used_by_fit=claims["inputs"].get("camera_json_sha256"), camera_json_sha256_in_checkpoint=inputs["params/camera.json"],
        camera_json_L_eff_m=json.load(open(f_cam))["scale"]["L_eff_m"], physics_L_eff_m=P["L_eff_m"]["value"])
    out = dict(label=a.label, inputs_sha256=inputs, tilt_applied_rad=tilt, gauge_cam_over_phys_Leff=gauge, camera_L_eff_m=cam_Leff, physics_L_eff_m=phys_Leff, method="theta at trajectory rows 4n; bob = L_c(sin,0,-cos); project.load_camera(frame=n, kind) per frame; residual = |model - track(method A)|",
               fit_window_claimed=claims.get("fit_window"), holdout_claimed=claims.get("holdout"),
               verifier=res, comparison_with_fit_projected_json=cmp, claims_reproduced_within_0p05px=agree,
               reference_fidelity=dict(median_sigma_px=sig, R2_target_px=max(2, 3 * sig), R3_target_px=max(3, 3 * sig),
                                       R2_pass=res["reference"]["rms_fit_px"] <= max(2, 3 * sig), R3_rms_pass=res["reference"]["rms_holdout_px"] <= max(3, 3 * sig),
                                       note="R2/R3 are reference-fidelity targets, reported not judged as physics (acceptance.json); expected FAIL per CHANGELOG 22:33:36Z"),
               derived_param_consistency=derived)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(dict(cmp=cmp, agree=agree, anchor=res["reference"]["rms_anchor_px"]), indent=1))
    if a.fig:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        s = series
        fig, ax = plt.subplots(2, 2, figsize=(15, 9.5))
        A = ax[0, 0]; A.plot(s["t"], s["um"], ".", ms=3, c="k", label="tracked u (reference clip, method A)"); A.plot(s["t"], s["u"], "-", c="C3", lw=1.2, label=f"{a.label} model projected (camera of record)")
        A.axvspan(0, a.anchor, color="C2", alpha=.12, label=f"anchor window 0-{a.anchor} s (user decision)"); A.axvline(7.5, ls="--", c="gray"); A.text(7.55, s["um"].min(), "fit | hold-out (t=7.5 s)", fontsize=8)
        A.set_xlabel("t [s]"); A.set_ylabel("u [px]"); A.legend(fontsize=8, loc="upper left"); A.set_title("bob centre u(t): tracked vs projected model")
        B = ax[0, 1]; B.plot(s["t"], s["du"], c="C0", label="du = model - track"); B.plot(s["t"], s["dv"], c="C1", label="dv"); B.plot(s["t"], s["r"], c="k", lw=.8, label="|r|")
        B.axvspan(0, a.anchor, color="C2", alpha=.12); B.axvline(7.5, ls="--", c="gray"); B.axhline(0, c="gray", lw=.5); B.set_xlabel("t [s]"); B.set_ylabel("px"); B.legend(fontsize=8)
        rr = res["reference"]
        B.set_title(f"residuals: RMS fit(t<7.5) {rr['rms_fit_px']:.3f} px, hold-out {rr['rms_holdout_px']:.3f} px, anchor(<={a.anchor}) {rr['rms_anchor_px']:.3f} px")
        C = ax[1, 0]; sc = C.scatter(s["um"], s["vm"], c=s["t"], s=8, cmap="viridis", label="tracked")
        C.quiver(s["um"], s["vm"], s["du"], s["dv"], s["t"], angles="xy", scale_units="xy", scale=0.25, cmap="viridis", width=.002)
        C.invert_yaxis(); C.set_aspect("equal", "datalim"); C.set_xlabel("u [px]"); C.set_ylabel("v [px] (down)"); plt.colorbar(sc, ax=C, label="t [s]")
        C.set_title("image-space residual map: arrows = model - track, x4 exaggerated")
        D = ax[1, 1]; D.axis("off")
        lines = ["Independent recomputation vs fit_projected.json (claims)", ""]
        for k, v in cmp.items():
            lines.append(f"{k:24s} verifier {v['verifier']:8.4f}  claimed {v['claimed']:8.4f}  |diff| {v['abs_diff']:.2e}")
        lines += ["", f"claims reproduced (<0.05 px): {agree}",
                  f"anchor window 0-{a.anchor} s RMS (ref cam): {rr['rms_anchor_px']:.3f} px  (u,v {rr['rms_anchor_u_v_px'][0]:.2f}, {rr['rms_anchor_u_v_px'][1]:.2f})",
                  f"R2 target {max(2, 3*sig):.1f} px -> {'PASS' if out['reference_fidelity']['R2_pass'] else 'FAIL'};  R3 target {max(3, 3*sig):.1f} px -> {'PASS' if out['reference_fidelity']['R3_rms_pass'] else 'FAIL'}",
                  "(reference-fidelity, reported; not physics P-items)", "",
                  f"fit window claimed: {claims.get('fit_window')}", f"tilt applied {tilt:+.6f} rad; gauge cam/phys L_eff {gauge:.6f}",
                  f"camera.json used by fit {derived['camera_json_sha256_used_by_fit'][:12]} vs ck {inputs['params/camera.json'][:12]}", "",
                  "inputs sha256:"] + [f"  {k}: {v[:12]}" for k, v in inputs.items()]
        D.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.5)
        import textwrap; fig.suptitle("\n".join(textwrap.wrap(a.caption or a.label, 170)), fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(a.fig, dpi=110)


if __name__ == "__main__":
    main()
