"""Stage 8: tracks -> 3D (world units W_in, camera of record per frame: camera/project.py).
Assumption under test: the ring centre (spring lower attachment), the weight centre and the spring axis lie in the
vertical plane Y = 0 through A (the frame-front-parallel plane containing the hook). Test: depth from apparent size
(weight silhouette width ~ 1/depth; ring radius ~ 1/depth) regressed on lateral position X and on the quadrature
(velocity) term -> swing-plane yaw and any elliptical (conical) out-of-plane component.
Writes measurement/tracks_3d.csv and measurement/tracks_3d.json."""
import numpy as np, os, sys, json, csv
sys.path.insert(0, os.path.dirname(__file__))
from common import *
ROOT_ = ROOT; sys.path.insert(0, os.path.join(ROOT_, "camera"))
import project as PJ
f = lambda T, k: np.array([float(r[k]) for r in T])
WT = list(csv.DictReader(open(os.path.join(MEAS, "weight_track.csv")))); RT = list(csv.DictReader(open(os.path.join(MEAS, "ring_track.csv"))))
ST = list(csv.DictReader(open(os.path.join(MEAS, "spring_track.csv")))); AN = json.load(open(os.path.join(MEAS, "analysis.json")))
t = np.arange(N) / FPS; cams = [PJ.load_camera(frame=n) for n in range(N)]
def onplane(u, v): return np.array([PJ.intersect_plane(u[n], v[n], cams[n], [0, 1, 0], 0.0) for n in range(N)])
Pw = onplane(f(WT, "u_px"), f(WT, "v_px")); Pr = onplane(f(RT, "u_px"), f(RT, "v_px"))
thW = np.radians(f(WT, "tilt_deg")); hl = 60.0
top = onplane(f(WT, "u_px") - hl * np.sin(thW), f(WT, "v_px") - hl * np.cos(thW)); bot = onplane(f(WT, "u_px") + hl * np.sin(thW), f(WT, "v_px") + hl * np.cos(thW))
tilt3 = np.degrees(np.arctan2(bot[:, 0] - top[:, 0], -(bot[:, 2] - top[:, 2])))   # weight axis angle from -Z toward +X (in plane Y=0)
Pst = onplane(f(ST, "axis_u_at_top"), f(ST, "v_top")); Psb = onplane(f(ST, "axis_u_at_bot"), f(ST, "v_bot"))
P = json.load(open(os.path.join(ROOT_, "params/camera.json"))); Peff = np.array(P["key_points"]["P_eff_world_on_plane_Y0"])
d = Pr - Peff; theta = np.degrees(np.arctan2(d[:, 0], -d[:, 2])); rr = np.linalg.norm(d, axis=1)
dW = Pw - Peff; thetaW = np.degrees(np.arctan2(dW[:, 0], -dW[:, 2])); rW = np.linalg.norm(dW, axis=1)
spring_tilt3 = np.degrees(np.arctan2(Psb[:, 0] - Pst[:, 0], -(Psb[:, 2] - Pst[:, 2]))); Lsp = np.linalg.norm(Psb - Pst, axis=1)
# ---- depth from apparent size
def size_test(width_px, Pplane, name):
    fn = np.array([c["f_px"] for c in cams]); dep_plane = np.array([PJ.depth(Pplane[n], cams[n]) for n in range(N)])
    # apparent size s = f D / depth  ->  depth_meas = K f / s with K = D (fitted so that mean depth equals plane depth)
    K = np.median(dep_plane * width_px / fn); dep_meas = K * fn / width_px
    dz = dep_meas - dep_plane                                     # depth excess vs the plane hypothesis
    X = Pplane[:, 0] - Pplane[:, 0].mean(); Xd = np.gradient(X, 1 / FPS) / (2 * np.pi / 1.3007)
    A = np.c_[np.ones(N), X, Xd, t - 5]; co, res, *_ = np.linalg.lstsq(A, dz, rcond=None); r = dz - A @ co
    cov = np.linalg.inv(A.T @ A) * np.var(r); sd = np.sqrt(np.diag(cov))
    # depth excess along the view direction ~ Y (camera looks along +Y within 7 deg)
    return {"signal": name, "K_size_units": float(K), "depth_noise_rms_Win": float(r.std()), "dY_dX": float(co[1]), "dY_dX_sd": float(sd[1]),
            "swing_plane_yaw_deg": float(np.degrees(np.arctan(co[1]))), "swing_plane_yaw_sd_deg": float(np.degrees(sd[1])),
            "quadrature_Y_amp_Win": float(co[2]), "quadrature_sd": float(sd[2]), "lateral_amp_Win": float(np.std(X) * np.sqrt(2)),
            "depth_trend_per_s_Win": float(co[3]), "depth_trend_sd": float(sd[3])}
szW = size_test(f(WT, "width_px"), Pw, "weight silhouette width"); szR = size_test(f(RT, "radius_px"), Pr, "ring outer radius")
out = dict(assumption="ring, weight centre and spring axis in the vertical plane Y=0 through A (frame-front-parallel)",
           size_tests=[szW, szR],
           P_eff_world=Peff.tolist(), theta_ring_deg=dict(min=float(theta.min()), max=float(theta.max()), mean=float(theta.mean())),
           r_ring_Win=dict(mean=float(rr.mean()), sd=float(rr.std())), r_weight_Win=dict(mean=float(rW.mean()), sd=float(rW.std())),
           ring_to_weight_centre_Win=float(np.mean(np.linalg.norm(Pw - Pr, axis=1))), ring_to_weight_centre_sd=float(np.std(np.linalg.norm(Pw - Pr, axis=1))),
           A_to_Peff_Win=float(np.linalg.norm(Peff)), spring_body_len_Win=dict(mean=float(Lsp.mean()), sd=float(Lsp.std())),
           weight_size_Win=dict(diameter=float(np.median(f(WT, "width_px") * np.array([PJ.depth(Pw[n], cams[n]) for n in range(N)]) / np.array([c["f_px"] for c in cams]))),
                                silhouette_length=float(np.median(f(WT, "length_px") * np.array([PJ.depth(Pw[n], cams[n]) for n in range(N)]) / np.array([c["f_px"] for c in cams])))),
           ring_outer_radius_Win=float(np.median(f(RT, "radius_px") * np.array([PJ.depth(Pr[n], cams[n]) for n in range(N)]) / np.array([c["f_px"] for c in cams]))))
with open(os.path.join(MEAS, "tracks_3d.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["frame", "t_s", "ring_X", "ring_Y", "ring_Z", "weight_X", "weight_Y", "weight_Z", "weight_tilt_inplane_deg", "theta_ring_about_Peff_deg",
                                    "r_ring_Peff", "theta_weight_about_Peff_deg", "r_weight_Peff", "spring_top_X", "spring_top_Z", "spring_bot_X", "spring_bot_Z", "spring_tilt_inplane_deg", "spring_body_len"])
    for n in range(N):
        w.writerow([n, "%.6f" % t[n]] + ["%.6f" % x for x in (*Pr[n], *Pw[n], tilt3[n], theta[n], rr[n], thetaW[n], rW[n], Pst[n, 0], Pst[n, 2], Psb[n, 0], Psb[n, 2], spring_tilt3[n], Lsp[n])])
json.dump(out, open(os.path.join(MEAS, "tracks_3d.json"), "w"), indent=1); print(json.dumps(out, indent=1))
