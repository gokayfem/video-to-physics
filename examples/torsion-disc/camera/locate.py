"""Place the frame-0 camera in the world (clip 03) - bootstrap for camera/pose_bundle.py.
Gauge: D = disc diameter = 1 (declared; the disc silhouette fit uses top-face radius 0.5).
Frame 0 only: disc pose from the silhouette (measurement/scripts/s3_disc_pose.fit, thickness 0.03 D) through the frame-0
camera (constant-f VP calibration, camera-centred axes), plus the engraved-line rotation centre offset
(measurement/cache/apex_offset.npy). A (origin) = point on the frame-0 ray through the wire-top pixel (wire leaves the clamp,
measurement/wire_track.csv) that lies vertically above the frame-0 disc rotation centre (LSQ in x,y) [asm: wire vertical at
t = 0; tested later against the bundle-triangulated disc track]. C0_world = -A_camcentred.
Initial 3D lines for the bundle [inf]: frame-0 back-projection planes intersected with assumed planes (crossbar and post
front faces flush at y = yf, crossbar square section; post inner-back silhouette at yf + crossbar depth).
Writes camera/out/locate.json"""
import numpy as np, json, csv, pathlib, sys, os
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "measurement/scripts"))
os.environ["NMAX"] = "1"
import camfr; assert camfr.MODE == "camcentred", "remove camera/out/camera_poses.json to re-bootstrap"
sys.argv = ["x"]; import s3_disc_pose as S3
apex = np.load(ROOT / "measurement/cache/apex_offset.npy")
x, sd, rms, med, m = S3.fit(0, 0.03, S3.init(0)); nn = S3.normal(x[3], x[4]); ex, ey = S3.basis(nn); Q0 = x[:3] + apex[0] * ex + apex[1] * ey
W = list(csv.DictReader(open(ROOT / "measurement/wire_track.csv")))
k0 = camfr.cam(0); top = np.array([float(W[0]["u_top_px"]), float(W[0]["v_top_px"])]); d = camfr.ray(top, k0)[0]
s = (d[:2] @ Q0[:2]) / (d[:2] @ d[:2]); A = s * d; miss = A[:2] - Q0[:2]; C = -A
T = json.load(open(ROOT / "measurement/cache/edges_track.json"))[0]
def plane(name):   # back-projection plane of the frame-0 image line: normal m (world), through C
    n_, c_ = np.array(T[name]["n"]), T[name]["c"]; l = np.r_[n_, c_]; K = np.array([[k0["f"], 0, k0["cx"]], [0, k0["f"], k0["cy"]], [0, 0, 1]])
    mm = k0["R"].T @ K.T @ l; return mm / np.linalg.norm(mm)
def line_on(name, fixed_axis, value):  # intersect back-projection plane with plane {X[fixed_axis] = value}; returns point, dir
    mm = plane(name); e = np.eye(3)[fixed_axis]; dvec = np.cross(mm, e); dvec /= np.linalg.norm(dvec)
    # point: solve mm.(X - C) = 0, X[fixed] = value, and minimise |X| along dvec
    Aeq = np.array([mm, e, dvec]); b = np.array([mm @ C, value, 0.0]); return np.linalg.solve(Aeq, b), dvec
# crossbar front face y = yf (square section: depth = height)
def zfront(yf):
    pt, _ = line_on("xbar_top_b", 1, yf); pb, _ = line_on("xbar_bot_b", 1, yf); return pt[2], pb[2]
yf = -0.05
for _ in range(30):
    zt, zb = zfront(yf); yf = -(zt - zb) / 2
zt, zb = zfront(yf); depth = zt - zb
init = {}
for k in ("xbar_top_a", "xbar_top_b", "xbar_bot_a", "xbar_bot_b", "zL_outer", "zL_frontinner", "zR_front", "zR_outer"):
    p, dv = line_on(k, 1, yf); init[k] = p.tolist()
p, dv = line_on("zL_back", 1, yf + depth); init["zL_back"] = p.tolist()
geo = {"crossbar_front_face_y": yf, "crossbar_top_z": zt, "crossbar_bottom_z": zb, "crossbar_depth_assumed_eq_height": depth,
       "post_front_corner_x": {k: init[k][0] for k in ("zL_outer", "zL_frontinner", "zR_front", "zR_outer")}}
out = dict(unit="D = disc diameter = 1", frame=0, A_camcentred=A.tolist(), C0_world=C.tolist(), distance_C0_to_A=float(np.linalg.norm(A)),
           disc0_rotation_centre_world=(Q0 - A).tolist(), disc0_fit=dict(x=x.tolist(), sd=sd.tolist(), rms_px=rms), A_horizontal_miss=miss.tolist(),
           wire_length_A_to_disc0_top=float(A[2] - Q0[2]), lines_init_world=init, frame_geometry_inferred_frame0=geo, apex_offset_disc_plane=apex.tolist())
json.dump(out, open(ROOT / "camera/out/locate.json", "w"), indent=1); print(json.dumps(out, indent=1)[:1500])
