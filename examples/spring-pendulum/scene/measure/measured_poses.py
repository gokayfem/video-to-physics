"""Per-frame TEST poses of the weight from the reference tracks, through the camera of record (scene role).
H = eye inner top (ring track centre + (R_e - r_e) towards P_eff), body axis = H -> weight silhouette centre
(weight_track u,v), both on the plane y = 0. Rotation about +Y: beta = atan2(-d_x, -d_z) (R_y(beta)(0,0,-1) = d).
Used ONLY to pose unbaked preview/visual builds; never baked. -> scene/measure/measured_poses.json (units W_in)."""
import json, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "camera"))
import project as PJ
G = json.loads((ROOT / "params/geometry.json").read_text())["ratios"]
Re, re_ = G["eye_ring_radius"]["value"], G["eye_wire_radius"]["value"]
P = np.array(json.loads((ROOT / "params/camera.json").read_text())["key_points"]["P_eff_world_on_plane_Y0"])
Rg = np.genfromtxt(ROOT / "measurement/ring_track.csv", delimiter=",", names=True)
Wt = np.genfromtxt(ROOT / "measurement/weight_track.csv", delimiter=",", names=True)


def p3(u, v, cam):
    o, w = PJ.ray(u, v, cam)
    return o + (-o[1] / w[1]) * w


out = {}
for i in range(len(Rg)):
    n = int(Rg["frame"][i]); cam = PJ.load_camera(frame=n)
    E = p3(Rg["u_px"][i], Rg["v_px"][i], cam)
    H = E + (Re - re_) * (P - E) / np.linalg.norm(P - E)
    j = np.where(Wt["frame"] == n)[0][0]
    Wc = p3(Wt["u_px"][j], Wt["v_px"][j], cam)
    d = (Wc - H) / np.linalg.norm(Wc - H)
    out[str(n)] = {"H_Win": H.round(6).tolist(), "beta_rad": float(np.arctan2(-d[0], -d[2]))}
(HERE / "measured_poses.json").write_text(json.dumps({"method": __doc__, "frames": out}, indent=0))
for n in ("0", "120", "240"):
    print(n, out[n], "deg", np.degrees(out[n]["beta_rad"]))
