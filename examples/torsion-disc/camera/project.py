"""Pure-numpy projection utility, clip 03 (torsion disc). Adapted from clip-02/01 camera/project.py.

Conventions
-----------
* World (config/coordinates.md): origin A = where the torsion wire leaves the black clamp, X along the crossbar toward
  image right, Y horizontal away from the camera, Z up.
  Length unit: camera.json "scale.unit" (D = disc diameter = 1); metres = units x scale.metres_per_unit
  (model-conditioned; None until the dynamics fit sets it). project()/ray() work in whatever unit C is given in.
* Pixels: CONTRACT convention, origin at the top-left corner of the top-left pixel, pixel centres at +0.5, 1920x1088.
* Camera dict: R (3x3 world->OpenCV camera: x right, y down, z forward), C (centre), f_px, cx, cy, type 'PERSP'|'ORTHO'
  (ORTHO: ortho_scale = visible width in world units, u = cx + (W/ortho_scale) Xc, v = cy + (W/ortho_scale) Yc).

Getting cameras
---------------
    cam = load_camera(frame=n)          # camera of record for frame n (t = n/24 s): measured per-frame ORBIT pose
                                        # (camera/drift.csv: C_n, R_n, f); n may be fractional
    cam = load_camera(kind="locked")    # frame-0 calibration held for all frames (drift-free diagnostic)
    cam = load_camera(kind="top_ortho") / load_camera(kind="overview")   # locked diagnostic cameras
    blender_camera_settings(cam)        # Blender camera (location, rotation matrix, lens/ortho_scale, shift)
Functions: project(points, cam) -> (u, v); ray(u, v, cam) -> (origin, unit dir); intersect_plane(u, v, cam, n, d);
depth(points, cam). Vectorised; no Blender / OpenCV imports.
"""
import json, pathlib, csv
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
W, H = 1920, 1088

def project(points, cam):
    X = np.asarray(points, float); Xc = (X - np.asarray(cam["C"], float)) @ np.asarray(cam["R"], float).T
    if cam.get("type", "PERSP") == "ORTHO":
        k = W / cam["ortho_scale"]; return cam["cx"] + k * Xc[..., 0], cam["cy"] + k * Xc[..., 1]
    z = Xc[..., 2]
    if np.any(z <= 0): raise ValueError("point behind camera")
    return cam["cx"] + cam["f_px"] * Xc[..., 0] / z, cam["cy"] + cam["f_px"] * Xc[..., 1] / z

def depth(points, cam):
    X = np.asarray(points, float); return ((X - np.asarray(cam["C"], float)) @ np.asarray(cam["R"], float).T)[..., 2]

def ray(u, v, cam):
    u = np.asarray(u, float); v = np.asarray(v, float)
    dc = np.stack([(u - cam["cx"]) / cam["f_px"], (v - cam["cy"]) / cam["f_px"], np.ones_like(u)], -1)
    dw = dc @ np.asarray(cam["R"], float); dw /= np.linalg.norm(dw, axis=-1, keepdims=True)
    return np.asarray(cam["C"], float), dw

def intersect_plane(u, v, cam, n, d):
    o, w = ray(u, v, cam); n = np.asarray(n, float); s = -(o @ n + d) / (w @ n); return o + s[..., None] * w

_POSES = None
def _poses():
    global _POSES
    if _POSES is None:
        rows = list(csv.DictReader(open(ROOT / "camera/drift.csv")))
        _POSES = dict(C=np.array([[float(r[k]) for k in ("Cx", "Cy", "Cz")] for r in rows]),
                      rv=np.array([[float(r[k]) for k in ("rotvec_x", "rotvec_y", "rotvec_z")] for r in rows]),
                      f=np.array([float(r["f_px"]) for r in rows]), cx=np.array([float(r["cx_px"]) for r in rows]), cy=np.array([float(r["cy_px"]) for r in rows]))
    return _POSES

def _rotvec_to_R(rv):
    th = np.linalg.norm(rv)
    if th < 1e-15: return np.eye(3)
    k = rv / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

def _R_to_rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(c)
    if th < 1e-12: return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th)); return v * th

def load_camera(frame=0, kind="reference", params_path=None, unit="model"):
    """kind: 'reference' (per-frame camera of record: the AI camera ORBITS, so every frame has its own position C_n and
    rotation R_n; constant focal length), 'locked' (frame-0 camera for all frames), 'top_ortho', 'overview'.
    Fractional frames: C and f interpolated linearly, R by geodesic (slerp) interpolation.
    unit: 'model' (D units) or 'm' (metres; requires scale.metres_per_unit in camera.json)."""
    P = json.load(open(params_path or ROOT / "params/camera.json"))
    k = 1.0
    if unit == "m":
        k = P["scale"]["metres_per_unit"]
        if k is None: raise ValueError("metres_per_unit not set yet (model-conditioned by dynamics)")
    if kind in ("top_ortho", "overview"):
        c = P["locked_cameras"][kind]["cv"]
        cam = {"R": np.array(c["R_world_to_cv"]), "C": np.array(c["C"]) * k, "cx": W / 2, "cy": H / 2, "type": c["type"], "kind": kind}
        if c["type"] == "ORTHO": cam["ortho_scale"] = c["ortho_scale"] * k
        else: cam["f_px"] = c["f_px"]
        return cam
    D = _poses()
    if kind == "locked": frame = 0
    n0 = int(np.clip(np.floor(frame), 0, len(D["f"]) - 1)); n1 = min(n0 + 1, len(D["f"]) - 1); a = float(frame) - n0
    R0 = _rotvec_to_R(D["rv"][n0]); R1 = _rotvec_to_R(D["rv"][n1]); R = _rotvec_to_R(a * _R_to_rotvec(R1 @ R0.T)) @ R0
    C = ((1 - a) * D["C"][n0] + a * D["C"][n1]) * k
    return {"R": R, "C": C, "f_px": (1 - a) * D["f"][n0] + a * D["f"][n1], "cx": (1 - a) * D["cx"][n0] + a * D["cx"][n1], "cy": (1 - a) * D["cy"][n0] + a * D["cy"][n1],
            "frame": frame, "kind": kind, "type": "PERSP"}

def blender_camera_settings(cam, sensor_width_mm=36.0):
    """Equivalent Blender camera (sensor_fit HORIZONTAL, 1920x1088, pixel aspect 1): location, 3x3 world rotation
    (Blender camera looks along local -Z, +Y up), lens_mm or ortho_scale, shift_x/y."""
    R = np.asarray(cam["R"], float); Rbl = R.T @ np.diag([1.0, -1.0, -1.0])
    out = {"location": list(map(float, cam["C"])), "rotation_matrix": Rbl.tolist(), "sensor_width_mm": sensor_width_mm, "sensor_fit": "HORIZONTAL",
           "shift_x": float(-(cam["cx"] - W / 2) / W), "shift_y": float((cam["cy"] - H / 2) / W), "type": cam.get("type", "PERSP")}
    if out["type"] == "ORTHO": out["ortho_scale"] = float(cam["ortho_scale"])
    else: out["lens_mm"] = float(cam["f_px"] * sensor_width_mm / W)
    return out

if __name__ == "__main__":
    cam = load_camera(0); print("A projects (frame 0) to", project(np.zeros(3), cam))
