#!/usr/bin/env python
"""Independent recomputation of the clip-02 projected residuals (verifier-owned; never imports solver/).

Model points from the VERIFIER reference trajectory (verify/ref_solver.py at the checkpoint params):
  ring = eye centre = H + e n,   wctr = cylinder centre = H + b_c n      (n = body axis hinge->CoM = R(q) (0,0,-1))
  e = geometry_m.H_to_eye_centre, b_c = geometry_m.H_to_cylinder_centre (metres), camera units = metres * camera_units_per_m
Camera of record: <ck>/camera/project.py load_camera(frame=n, kind='reference') on <ck>/params/camera.json (camera role's code).
Tracks: the raw measurement files (<ck>/measurement/ring_track.csv, weight_track.csv) - not the solver's tracks_for_fit copy
(which is compared byte-for-byte in value as a side check).
RMS definition (reproduces the fit's per-point u/v split): sqrt(mean over frames and points of du^2 + dv^2).

  .venv/bin/python verify/audit_fit_residuals.py --ck verify/out/ck001/ckcopy --ref verify/out/ck001/ref_trajectory.csv \
      --fit-json solver/out/fit/prod_Ta7.5/fit_projected.json --anchor 7.5 --out verify/out/ck001/fit_audit_ck001.json \
      --fig verify/visuals/v002_fit_residual_map.png --label "v002 | ck001 | final for ck001"
"""
import argparse, csv, importlib.util, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from vio import read_csv, dump_json  # noqa: E402
from vparams import sha256_file, get_path  # noqa: E402
from vmodel import quat_to_R  # noqa: E402


def load_cols(path):
    with open(path) as f:
        r = csv.DictReader(f); rows = list(r)
    return {k: np.array([float(x[k]) for x in rows]) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--fit-json", default="solver/out/fit/prod_Ta7.5/fit_projected.json"); ap.add_argument("--anchor", type=float, default=7.5)
    ap.add_argument("--out", required=True); ap.add_argument("--fig"); ap.add_argument("--label", default="")
    a = ap.parse_args()
    ck = os.path.abspath(a.ck)
    spec = importlib.util.spec_from_file_location("ck_project", os.path.join(ck, "camera", "project.py"))
    proj = importlib.util.module_from_spec(spec); spec.loader.exec_module(proj)
    P = json.load(open(os.path.join(ck, "params", "physics.json")))
    lam = float(get_path(P, "/camera_scale/camera_units_per_m"))
    e = float(get_path(P, "/geometry_m/H_to_eye_centre")); bc = float(get_path(P, "/geometry_m/H_to_cylinder_centre"))
    ref, _ = read_csv(a.ref)
    idx = np.arange(241) * 4
    H = np.stack([ref["x_m"], ref["y_m"], ref["z_m"]], -1)[idx]
    n = quat_to_R(np.stack([ref["qw"], ref["qx"], ref["qy"], ref["qz"]], -1)[idx]) @ np.array([0.0, 0.0, -1.0])
    pts = {"ring": (H + e * n) * lam, "wctr": (H + bc * n) * lam}
    ring = load_cols(os.path.join(ck, "measurement", "ring_track.csv")); wt = load_cols(os.path.join(ck, "measurement", "weight_track.csv"))
    tr = {"ring": (ring["u_px"], ring["v_px"]), "wctr": (wt["u_px"], wt["v_px"])}
    sig = {"ring": (ring["sigma_u_px"], ring["sigma_v_px"]), "wctr": (wt["sigma_u_px"], wt["sigma_v_px"])}
    t = np.arange(241) / 24.0
    res = {}
    for name, X in pts.items():
        uv = np.array([proj.project(X[i], proj.load_camera(frame=i, kind="reference", params_path=os.path.join(ck, "params", "camera.json"))) for i in range(241)])
        res[name] = (uv[:, 0] - tr[name][0], uv[:, 1] - tr[name][1], uv)
    anc = t <= a.anchor + 1e-9
    def rms(mask, names=("ring", "wctr")):
        return float(math.sqrt(np.mean(np.concatenate([res[k][0][mask] ** 2 + res[k][1][mask] ** 2 for k in names]))))
    m = dict(rms_anchor_px=rms(anc), rms_after_anchor_px=rms(~anc), per_point={}, n_anchor_frames=int(anc.sum()))
    for k in res:
        m["per_point"][k] = dict(rms_anchor_px=rms(anc, (k,)), rms_anchor_u_v_px=[float(np.sqrt(np.mean(res[k][0][anc] ** 2))), float(np.sqrt(np.mean(res[k][1][anc] ** 2)))],
                                 rms_after_px=rms(~anc, (k,)), mean_du_anchor_px=float(np.mean(res[k][0][anc])), max_abs_px=float(np.max(np.hypot(res[k][0], res[k][1]))))
    mod4 = (np.arange(241) % 4 == 0)
    m["supplementary_n_mod4_eq0"] = dict(rms_anchor_px=rms(anc & mod4), rms_after_px=rms(~anc & mod4))
    med_sig = float(np.median(np.concatenate([np.hypot(*sig[k]) / math.sqrt(2) for k in sig])))   # per point-frame rms of (sigma_u, sigma_v), all frames
    m["median_sigma_px"] = med_sig; m["median_sigma_definition"] = "median over all frames and both points of sqrt((sigma_u^2+sigma_v^2)/2)"
    m["info_median_sigma_components_px"] = float(np.median(np.concatenate([np.concatenate(sig[k]) for k in sig])))
    m["R2_anchor_target_px"] = max(2.0, 3 * med_sig); m["R2_anchor_pass"] = m["rms_anchor_px"] <= m["R2_anchor_target_px"]
    F = json.load(open(os.path.join(ck, a.fit_json)))
    claim = F["metrics"]
    cmp_ = dict(rms_anchor_px=[claim["rms_anchor_px"], m["rms_anchor_px"]], rms_after_anchor_px=[claim["rms_after_anchor_px"], m["rms_after_anchor_px"]],
                ring_anchor=[claim["per_point"]["ring"]["rms_anchor_px"], m["per_point"]["ring"]["rms_anchor_px"]],
                wctr_anchor=[claim["per_point"]["wctr"]["rms_anchor_px"], m["per_point"]["wctr"]["rms_anchor_px"]],
                median_sigma_px=[claim["median_sigma_px"], med_sig], R2_target=[claim["R2_anchor_target_px"], m["R2_anchor_target_px"]],
                n_mod4_anchor=[claim["supplementary_n_mod4_eq0"]["rms_anchor_px"], m["supplementary_n_mod4_eq0"]["rms_anchor_px"]])
    maxdiff = max(abs(v[0] - v[1]) for v in cmp_.values())
    # side check: solver's tracks_for_fit == raw measurement tracks
    tf = load_cols(os.path.join(ck, "solver", "out", "tracks_for_fit.csv"))
    side = float(max(np.max(np.abs(tf["ring_u_px"] - ring["u_px"])), np.max(np.abs(tf["ring_v_px"] - ring["v_px"])),
                     np.max(np.abs(tf["wctr_u_px"] - wt["u_px"])), np.max(np.abs(tf["wctr_v_px"] - wt["v_px"]))))
    rep = dict(label=a.label, inputs={p: sha256_file(os.path.join(ck, p)) for p in ("params/physics.json", "params/camera.json", "camera/project.py", "camera/drift.csv",
                                                                                  "measurement/ring_track.csv", "measurement/weight_track.csv", a.fit_json)},
               ref=a.ref, ref_sha256=sha256_file(a.ref), gauge=dict(camera_units_per_m=lam, e_m=e, b_c_m=bc), anchor_s=a.anchor,
               metrics_recomputed=m, claims_vs_recomputed=cmp_, max_abs_claim_minus_recomputed_px=maxdiff,
               claims_reproduced=maxdiff < 1e-3, tracks_for_fit_vs_measurement_max_px=side,
               rms_definition="sqrt(mean over frames and both points of du^2 + dv^2)")
    rep["series"] = {k: dict(du=res[k][0], dv=res[k][1]) for k in res}
    dump_json(rep, a.out)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("series", "inputs")}, indent=1, default=float)[:2500])
    if a.fig:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        C1, C2, MUT = "#2a78d6", "#eb6834", "#52514e"
        fig, ax = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
        for i, k in enumerate(("ring", "wctr")):
            ax[i].plot(t, res[k][0], color=C1, lw=1.4, label="du (model - track)")
            ax[i].plot(t, res[k][1], color=C2, lw=1.4, label="dv")
            ax[i].fill_between(t, -3 * sig[k][0], 3 * sig[k][0], color="#d8d7d2", alpha=0.6, lw=0, label="+-3 sigma_u (track)")
            ax[i].axvline(a.anchor, color="k", lw=1); ax[i].text(a.anchor + 0.05, ax[i].get_ylim()[1] * 0.85, "anchor window end 7.5 s", fontsize=8)
            pp = m["per_point"][k]
            ax[i].set_title(f"{'ring (eye centre)' if k == 'ring' else 'weight centre'}: anchor RMS {pp['rms_anchor_px']:.2f} px (u {pp['rms_anchor_u_v_px'][0]:.2f} / v {pp['rms_anchor_u_v_px'][1]:.2f}), after {pp['rms_after_px']:.2f} px")
            ax[i].set_ylabel("pixel residual"); ax[i].grid(True, color="#e4e3df"); ax[i].legend(fontsize=8, frameon=False, loc="lower left")
        ax[1].set_xlabel("t [s]")
        fig.suptitle(f"Independent residual recomputation (verifier ref solver + camera/project.py + measurement tracks): anchor {m['rms_anchor_px']:.4f} px "
                     f"vs claimed {claim['rms_anchor_px']:.4f} px; R2 target {m['R2_anchor_target_px']:.2f} px -> {'PASS' if m['R2_anchor_pass'] else 'FAIL'}", fontsize=10)
        fig.text(0.01, 0.005, f"clip 02 vertical spring | verifier (D) | {a.label}", fontsize=7.5, color=MUT)
        fig.tight_layout(rect=(0, 0.02, 1, 0.96)); fig.savefig(a.fig, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
