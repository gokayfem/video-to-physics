#!/usr/bin/env python
"""Independent recomputation of the projected anchor-window residuals (R2 claims) for clip 03. Verifier-owned; never imports
solver/. Model motion = MY reference trajectory (verify/ref_solver.py at the checkpoint params); camera = camera of record
(camera/project.py load_camera(frame=n)); observations = measurement/ files directly (not solver/out/tracks_for_fit.csv).

Channels (as described in solver/NOTES.md 'Interface (measurement -> fit)'):
  B    : top-face centre B (world m -> camera units x camera_units_per_m) projected vs disc_track.csv (u_Q_px, v_Q_px)
  psi  : world azimuth of the engraved line, R (cos mu, sin mu, 0) projected on the horizontal, vs twist_track.csv psi_deg (mod 180)
  nrm  : disc up-normal a = R e_z vs disc_pose_camcentred.csv (nx, ny, nz)
Windows (CHANGELOG lead decision): B and nrm 0-2.5 s, psi 0-7.5 s (per-mode); reported after the window too.
RMS definitions reported both ways: 2-D (sqrt mean |d|^2) and per-component (sqrt mean of the components' squares).

  .venv/bin/python verify/check_projected.py --ck checkpoints/ck001 --ref verify/out/ref_ck001.csv [--claims <fit_summary.json>] --out X.json
"""
import argparse, csv, json, math, os, sys, importlib.util
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from tio import read_csv, dump_json  # noqa: E402
from tparams import sha256_file  # noqa: E402
from tmodel import quat_to_R  # noqa: E402


def load(path):
    with open(path) as f:
        r = csv.DictReader(f)
        rows = list(r)
    return {k: np.array([float(x[k]) for x in rows]) for k in rows[0]}


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--trajectory")
    ap.add_argument("--claims"); ap.add_argument("--b-track", default="measurement/disc_track.csv"); ap.add_argument("--b-cols", default="u_Q_px,v_Q_px"); ap.add_argument("--tB", type=float, default=2.5); ap.add_argument("--tpsi", type=float, default=7.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    CK = os.path.abspath(a.ck)
    spec = importlib.util.spec_from_file_location("ck_project", os.path.join(CK, "camera", "project.py"))
    proj = importlib.util.module_from_spec(spec); spec.loader.exec_module(proj)
    camjson = os.path.join(CK, "params", "camera.json")
    P = json.load(open(os.path.join(CK, "params", "physics.json")))
    k = float(P["camera_scale"]["camera_units_per_m"]); mu = float(P["mark"]["mu_body_rad"]["value"])
    rep = dict(inputs={}, camera_units_per_m=k, mu_body_rad=mu, windows=dict(B=a.tB, nrm=a.tB, psi=a.tpsi))
    dt = load(os.path.join(CK, a.b_track)); bu, bv = a.b_cols.split(","); rep["B_track"] = [a.b_track, a.b_cols]
    tw = load(os.path.join(CK, "measurement", "twist_track.csv"))
    dpz = load(os.path.join(CK, "measurement", "disc_pose_camcentred.csv"))
    for f in (a.b_track, "measurement/twist_track.csv", "measurement/disc_pose_camcentred.csv", "params/camera.json",
              "camera/project.py", "camera/drift.csv", "params/physics.json"):
        rep["inputs"][f] = sha256_file(os.path.join(CK, f))
    rep["inputs"][a.ref] = sha256_file(a.ref)
    res = {}
    for label, path in (("verifier_ref", a.ref), ("production_trajectory", a.trajectory)):
        if not path:
            continue
        d, _ = read_csv(path)
        n = dt["frame"].astype(int); t = n / 24.0
        row = np.round(t * 96).astype(int)
        B = np.stack([d["px_m"] if "px_m" in d else d["B_x_m"], d["py_m"] if "py_m" in d else d["B_y_m"], d["pz_m"] if "pz_m" in d else d["B_z_m"]], -1)[row] * k
        q = np.stack([d[c] for c in (("qw", "qx", "qy", "qz") if "qw" in d else ("q_w", "q_x", "q_y", "q_z"))], -1)[row]
        R = quat_to_R(q)
        uv = np.array([np.array(proj.project(B[i][None], proj.load_camera(frame=int(n[i]), params_path=camjson))).reshape(-1) for i in range(len(n))])
        du = uv[:, 0] - dt[bu]; dv = uv[:, 1] - dt[bv]
        mvec = np.einsum("nij,j->ni", R, np.array([math.cos(mu), math.sin(mu), 0.0]))
        az = np.degrees(np.arctan2(mvec[:, 1], mvec[:, 0]))
        ip = tw["frame"].astype(int)
        dpsi = (tw["psi_deg"] - az[np.searchsorted(n, ip)] + 90.0) % 180.0 - 90.0
        a_up = R[:, :, 2]
        nm = np.stack([dpz["nx"], dpz["ny"], dpz["nz"]], -1); nm /= np.linalg.norm(nm, axis=1, keepdims=True)
        ina = np.degrees(np.arccos(np.clip(np.sum(a_up[np.searchsorted(n, dpz["frame"].astype(int))] * nm, 1), -1, 1)))
        dnx = np.degrees(a_up[:, 0] - nm[:, 0]); dny = np.degrees(a_up[:, 1] - nm[:, 1])
        wB = t <= a.tB + 1e-9; wP = ip / 24.0 <= a.tpsi + 1e-9
        res[label] = dict(
            B_rms_2d_px_anchor=rms(np.hypot(du, dv)[wB]), B_rms_percomp_px_anchor=rms(np.r_[du[wB], dv[wB]]),
            B_rms_2d_px_after=rms(np.hypot(du, dv)[~wB]), B_max_px_anchor=float(np.hypot(du, dv)[wB].max()),
            psi_rms_deg_anchor=rms(dpsi[wP]), psi_rms_deg_after=rms(dpsi[~wP]), psi_end_deg=float(dpsi[-1]), psi_mean_deg_anchor=float(np.mean(dpsi[wP])),
            nrm_angle_rms_deg_anchor=rms(ina[wB]), nrm_percomp_rms_deg_anchor=rms(np.r_[dnx[wB], dny[wB]]), nrm_angle_rms_deg_after=rms(ina[~wB]),
            n_frames=int(len(n)), n_anchor_B=int(wB.sum()), n_anchor_psi=int(wP.sum()))
        res[label]["_series"] = dict(t=t.tolist(), du=du.tolist(), dv=dv.tolist(), dpsi=dpsi.tolist(), nrm=ina.tolist())
    rep["results"] = res
    if a.claims:
        C = json.load(open(a.claims))["metrics"]["per_channel"]
        rep["claims"] = dict(B_rms_anchor_px=C["pt:B"]["rms_anchor_px"], B_rms_after_px=C["pt:B"]["rms_after_px"], psi_rms_anchor_deg=C["psi"]["rms_anchor_deg"],
                             psi_rms_after_deg=C["psi"]["rms_after_deg"], nrm_rms_anchor_deg=C["nrm"]["rms_anchor_deg"], nrm_rms_after_deg=C["nrm"]["rms_after_deg"])
        rep["inputs"][a.claims] = sha256_file(a.claims)
        v = res["verifier_ref"]; c = rep["claims"]
        best_B = min((abs(v[x] - c["B_rms_anchor_px"]), x) for x in ("B_rms_2d_px_anchor", "B_rms_percomp_px_anchor"))
        best_n = min((abs(v[x] - c["nrm_rms_anchor_deg"]), x) for x in ("nrm_angle_rms_deg_anchor", "nrm_percomp_rms_deg_anchor"))
        rep["comparison"] = dict(B=dict(claim=c["B_rms_anchor_px"], mine=v[best_B[1]], definition_matched=best_B[1], abs_diff=best_B[0]),
                                 psi=dict(claim=c["psi_rms_anchor_deg"], mine=v["psi_rms_deg_anchor"], abs_diff=abs(v["psi_rms_deg_anchor"] - c["psi_rms_anchor_deg"])),
                                 nrm=dict(claim=c["nrm_rms_anchor_deg"], mine=v[best_n[1]], definition_matched=best_n[1], abs_diff=best_n[0]),
                                 B_after=dict(claim=c["B_rms_after_px"], mine=v["B_rms_2d_px_after"]), psi_after=dict(claim=c["psi_rms_after_deg"], mine=v["psi_rms_deg_after"]))
        rep["comparison"]["criterion"] = "claims reproduced if |mine - claim| <= 0.05 px / 0.05 deg (independent pipeline; same inputs)"
        rep["comparison"]["status"] = "REPRODUCED" if (best_B[0] <= 0.05 and rep["comparison"]["psi"]["abs_diff"] <= 0.05 and best_n[0] <= 0.05) else "DIFFERS"
        rep["R2_anchor_B_target_px"] = 2.0
        rep["R2_B_status"] = "PASS" if v["B_rms_2d_px_anchor"] <= 2.0 and v["B_rms_percomp_px_anchor"] <= 2.0 else "FAIL"
        sig = float(np.median(tw["sigma_psi_deg"]))
        rep["R2_psi_target_deg"] = dict(three_sigma=3 * sig, median_sigma_deg=sig)
        rep["R2_psi_status"] = "PASS" if v["psi_rms_deg_anchor"] <= 3 * sig else "MISS (rms > 3 sigma)"
    dump_json(rep, a.out)
    print(json.dumps({k2: {kk: vv for kk, vv in v2.items() if kk != "_series"} for k2, v2 in res.items()}, indent=1))
    if "comparison" in rep:
        print(json.dumps(rep["comparison"], indent=1), rep["R2_B_status"], rep["R2_psi_status"])


if __name__ == "__main__":
    main()
