"""Reference-measured TEST poses of the disc for look/blockout renders (scene role C; NOT physics).
For each analysed frame (scene/measure/mark_analysis.json): the two half-line directions of the engraved mark at
r = 0.7 in the unwarped ellipse frame -> image points -> back-projected through the camera of record
(params/camera.json via camera/project.py; else the scene role's approximate camera) onto the disc top plane z = -L (D units) -> world twist psi of the line
(mod 180 deg, the mark is a diameter), with the disc offset from fit_geometry.json at its fitted frames.
Writes scene/measure/measured_poses.json {frame: {psi_deg, offset_D}}.
.venv/bin/python scene/measure/measured_poses.py
"""
import json, pathlib, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def approx_cam(G, n):
    fr = G["camera_approx"]["frames"]
    keys = sorted(int(k) for k in fr)
    n = min(max(n, keys[0]), keys[-1])
    i = max(j for j in range(len(keys) - 1) if keys[j] <= n)
    k0, k1 = keys[i], keys[i + 1]
    s = (n - k0) / (k1 - k0)
    sl = Slerp([0, 1], Rot.from_matrix([fr[str(k0)]["R_world_to_cv"], fr[str(k1)]["R_world_to_cv"]]))
    R = sl([s]).as_matrix()[0]
    C = (1 - s) * np.array(fr[str(k0)]["C_D"]) + s * np.array(fr[str(k1)]["C_D"])
    f = (1 - s) * fr[str(k0)]["f_px"] + s * fr[str(k1)]["f_px"]
    return R, C, f


sys.path.insert(0, str(ROOT / "camera"))
try:
    import project as PJ  # noqa: E402
except Exception:
    PJ = None


def main():
    G = json.loads((ROOT / "params/geometry.json").read_text())
    L = G["ratios"]["wire_length_A_to_disc_top"]["value"]
    MA = json.loads((HERE / "mark_analysis.json").read_text())["rows"]
    cr = HERE / "fit_geometry_camref.json"
    offs = json.loads(cr.read_text())["disc_offset_D"] if cr.exists() and (ROOT / "params/camera.json").exists() else \
        {k: v["disc_offset_D"] for k, v in G["camera_approx"]["frames"].items()}
    out = {}
    for r in MA:
        n = r["frame"]
        e = r["ellipse"]
        th = np.radians(e["theta_deg"])
        if (ROOT / "params/camera.json").exists():                 # camera of record (unit D)
            c = PJ.load_camera(frame=n)
            R, C, f = np.asarray(c["R"]), np.asarray(c["C"]), c["f_px"]
        else:
            R, C, f = approx_cam(G, n)
        pts = []
        for a in r["half_angles_deg"]:
            a = np.radians(a)
            X, Y = 0.7 * np.cos(a) * e["a"], 0.7 * np.sin(a) * e["b"]
            u = e["centre"][0] + X * np.cos(th) - Y * np.sin(th)
            v = e["centre"][1] + X * np.sin(th) + Y * np.cos(th)
            d = R.T @ np.array([(u - 960) / f, (v - 544) / f, 1.0])
            s = (-L - C[2]) / d[2]
            pts.append(C + s * d)
        dv = pts[0] - pts[1]
        psi = np.degrees(np.arctan2(dv[1], dv[0]))
        psi = (psi + 90) % 180 - 90
        off = offs.get(str(n), [0.0, 0.0])
        out[str(n)] = {"psi_deg": round(float(psi), 2), "offset_D": off, "kink_deg": r["kink_deg"]}
    (HERE / "measured_poses.json").write_text(json.dumps({"note": __doc__.strip().splitlines()[0], "frames": out}, indent=1))
    print({k: v["psi_deg"] for k, v in out.items() if int(k) % 20 == 0})


if __name__ == "__main__":
    main()
