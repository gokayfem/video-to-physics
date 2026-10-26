"""Spring layout through the camera of record (scene role): measurement/spring_track.csv + ring_track.csv per frame ->
3D points on the plane y = 0 (load_camera(frame=n)). Writes scene/measure/spring_layout.json.
Per frame: coil top T (axis_u_at_top, v_top), coil bottom B, dense (tight) section start D = T + s_dense_start along the
image axis, eye centre E (ring track), hinge H = E + (R_e - r_e) * unit(T - E) (eye inner top, approximately along
the spring axis). P_eff from camera.json key_points."""
import json, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "camera"))
import project as PJ
S = np.genfromtxt(ROOT / "measurement/spring_track.csv", delimiter=",", names=True)
Rg = np.genfromtxt(ROOT / "measurement/ring_track.csv", delimiter=",", names=True)
G = json.loads((ROOT / "params/geometry.json").read_text())["ratios"]
Re, re_ = G["eye_ring_radius"]["value"], G["eye_wire_radius"]["value"]
CJ = json.loads((ROOT / "params/camera.json").read_text())
P = np.array(CJ["key_points"]["P_eff_world_on_plane_Y0"])


def p3(u, v, cam):
    o, w = PJ.ray(u, v, cam)
    return o + (-o[1] / w[1]) * w


rows = []
for i in range(len(S)):
    n = int(S["frame"][i]); cam = PJ.load_camera(frame=n)
    T = p3(S["axis_u_at_top"][i], S["v_top"][i], cam)
    B = p3(S["axis_u_at_bot"][i], S["v_bot"][i], cam)
    d = np.array([S["axis_u_at_bot"][i] - S["axis_u_at_top"][i], S["v_bot"][i] - S["v_top"][i]]); d /= np.linalg.norm(d)
    s0 = S["s_dense_start_px"][i]
    Dn = p3(S["axis_u_at_top"][i] + s0 * d[0], S["v_top"][i] + s0 * d[1], cam) if np.isfinite(s0) else np.full(3, np.nan)
    j = np.where(Rg["frame"] == n)[0][0]
    E = p3(Rg["u_px"][j], Rg["v_px"][j], cam)
    up = (T - E) / np.linalg.norm(T - E)
    H = E + (Re - re_) * up
    rows.append(dict(frame=n, T=T, B=B, D=Dn, H=H, pitch_px=S["open_pitch_px"][i], ngaps=S["n_open_gaps"][i]))
f = lambda k: np.array([r[k] for r in rows])
T, B, D, H = f("T"), f("B"), f("D"), f("H")
ax = (H - P) / np.linalg.norm(H - P, axis=1)[:, None]
L = np.linalg.norm(H - P, axis=1)
along = lambda X: np.einsum("ij,ij->i", X - P, ax)
sT, sD, sB = along(T), along(D), along(B)
ok = np.isfinite(sD)
out = {"P_eff_Win": P.tolist(), "n_frames": len(rows),
       "PH_length_Win": {"mean": float(L.mean()), "sd": float(L.std()), "f000": float(L[0]), "f120": float(L[120]), "f240": float(L[240])},
       "coil_top_along_axis_from_P_Win": {"mean": float(sT.mean()), "sd": float(sT.std())},
       "open_length_Win": {"mean": float((sD - sT)[ok].mean()), "sd": float((sD - sT)[ok].std()), "n": int(ok.sum())},
       "tight_length_Win": {"mean": float((sB - sD)[ok].mean()), "sd": float((sB - sD)[ok].std())},
       "bottom_hardware_Win": {"mean": float((L - sB).mean()), "sd": float((L - sB).std())},
       "open_pitch_px_median": float(np.nanmedian(f("pitch_px"))), "open_gaps_median": float(np.nanmedian(f("ngaps"))),
       "H_xz_Win": {k: H[n][[0, 2]].round(5).tolist() for k, n in (("f000", 0), ("f120", 120), ("f240", 240))},
       "method": __doc__}
(HERE / "spring_layout.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1)[:1600])
