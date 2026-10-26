"""Assemble solver/out/tracks_for_fit.csv (fit_projected channels) from measurement/ products (role A).

  B (pt:B)  image of the disc SILHOUETTE top-face centre P (lead VR03-05: P is disc-attached; Q, the mark rotation centre, is
            world-fixed): measurement/disc_P_track.csv u_P_px, v_P_px; sigma_px from the per-axis effective pose sd (sd_eff)
            through the camera-of-record projection Jacobian (camera/project.py), floor 0.5 px
  Bw        world P (D units, A origin): Px_D, Py_D, Pz_D with sd_eff_x/y/z (smoothed camera of record)
  psi       measurement/twist_track.csv psi_deg (world azimuth of the line, +X toward +Y), sigma_psi_deg
  nrm       disc up-normal world x/y: sin(tilt_about_Y), -sin(tilt_about_X) (disc_track.csv); sigma from
            disc_pose_camcentred.csv max(sd_a, sd_b, 0.004 rad floor)
"""
import csv, math, sys, importlib.util
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
rd = lambda p: list(csv.DictReader(open(ROOT / "measurement" / p)))
D = rd("disc_P_track.csv"); Tw = rd("twist_track.csv"); Pc = rd("disc_pose_camcentred.csv")
NRM_FLOOR = 0.004   # rad: AI-systematic floor (the per-frame formal pose sd ~4e-4 rad is 10-20x below the tilt's sine-fit residual
                    # 0.45-1.6 deg, measurement/analysis.json sway.tilt_*); declared, so the tilt does not swamp the twist/sway channels
B_FLOOR = 0.5       # px
spec = importlib.util.spec_from_file_location("cam_project", ROOT / "camera" / "project.py")
pj = importlib.util.module_from_spec(spec); spec.loader.exec_module(pj)


def px_sigma(n, Q, sd):
    """sigma_u, sigma_v from the camera-of-record projection Jacobian and the per-axis pose sd (diagonal), floor 0.5 px."""
    c = pj.load_camera(frame=n); J = np.zeros((2, 3)); u0 = np.array(pj.project(Q, c))
    for i in range(3):
        dQ = np.zeros(3); dQ[i] = 1e-6; J[:, i] = (np.array(pj.project(Q + dQ, c)) - u0) / 1e-6
    C = J @ np.diag(np.asarray(sd) ** 2) @ J.T
    return max(B_FLOOR, math.sqrt(C[0, 0])), max(B_FLOOR, math.sqrt(C[1, 1]))
out = HERE / "out" / "tracks_for_fit.csv"; out.parent.mkdir(exist_ok=True)
with open(out, "w") as f:
    f.write("frame,t_s,B_u_px,B_v_px,B_sigma_u_px,B_sigma_v_px,Bw_x,Bw_y,Bw_z,Bw_sigma_x,Bw_sigma_y,Bw_sigma_z,psi_deg,psi_sigma_deg,nrm_x,nrm_y,nrm_sigma\n")
    for d, tw, pc in zip(D, Tw, Pc):
        assert d["frame"] == tw["frame"] == pc["frame"]
        Q = np.array([float(d["Px_D"]), float(d["Py_D"]), float(d["Pz_D"])])
        su, sv = px_sigma(int(d["frame"]), Q, [float(d["sd_eff_x"]), float(d["sd_eff_y"]), float(d["sd_eff_z"])])
        nx = math.sin(math.radians(float(d["tilt_about_Y_deg"]))); ny = -math.sin(math.radians(float(d["tilt_about_X_deg"])))   # P-track pose
        sn = max(float(pc["sd_a"]), float(pc["sd_b"]), NRM_FLOOR)
        f.write(",".join(str(x) for x in (d["frame"], d["t_s"], d["u_P_px"], d["v_P_px"], su, sv, d["Px_D"], d["Py_D"], d["Pz_D"],
                                         d["sd_eff_x"], d["sd_eff_y"], d["sd_eff_z"], tw["psi_deg"], tw["sigma_psi_deg"], nx, ny, sn)) + "\n")
print(out)
