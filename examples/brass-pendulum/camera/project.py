"""Pure-numpy pinhole projection utility for clip 01 (camera role).

Conventions
-----------
* World: CONTRACT frame (metres, Z up, origin = pivot P, X in the swing plane toward image right,
  Y away from the camera side).
* Pixels: CONTRACT convention -- origin at the top-left corner of the top-left pixel, u right,
  v down, pixel centres at integer + 0.5, full resolution 1920 x 1088.
* Camera: ideal pinhole, square pixels, no lens distortion.  A camera is a dict with
      R      3x3 world->camera rotation, OpenCV camera axes (x right, y down, z forward)
      C      camera centre in world metres (3,)
      f_px   focal length in pixels
      cx, cy principal point in pixels (contract convention)
  u = cx + f_px * Xc/Zc,  v = cy + f_px * Yc/Zc,  with Xc = R (X - C).

Getting cameras
---------------
    cam = load_camera()            # reference-matched camera at reference frame 120 (static part)
    cam = load_camera(frame=n)     # reference-matched camera for frame n (includes the measured
                                   # zoom/pan drift, camera/drift.csv; t = n/24 s)
    cam = load_camera(kind="locked")  # drift-free camera (frame-120 intrinsics for every frame)
`n` may be fractional (linear interpolation between frames) for sub-frame evaluation.

Functions: project(points_world_m, cam) -> (u, v); ray(u, v, cam) -> (origin, unit_dir);
intersect_plane(u, v, cam, n, d) for the plane n.X + d = 0 (e.g. swing plane Y=0: n=(0,1,0), d=0);
depth(points, cam).  All vectorised; no Blender / OpenCV imports.
"""
import json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
W, H = 1920, 1088


def project(points_world_m, cam):
    """points (..., 3) world metres -> (u, v) arrays in contract pixel coordinates."""
    X = np.asarray(points_world_m, float)
    Xc = (X - np.asarray(cam["C"], float)) @ np.asarray(cam["R"], float).T
    z = Xc[..., 2]
    if np.any(z <= 0):
        raise ValueError("point behind camera")
    u = cam["cx"] + cam["f_px"] * Xc[..., 0] / z
    v = cam["cy"] + cam["f_px"] * Xc[..., 1] / z
    return u, v


def depth(points_world_m, cam):
    """camera-frame depth Zc (m) of world points."""
    X = np.asarray(points_world_m, float)
    return ((X - np.asarray(cam["C"], float)) @ np.asarray(cam["R"], float).T)[..., 2]


def ray(u, v, cam):
    """inverse projection: pixel(s) -> (origin (3,), unit direction (..., 3)) in world metres."""
    u = np.asarray(u, float); v = np.asarray(v, float)
    dc = np.stack([(u - cam["cx"]) / cam["f_px"], (v - cam["cy"]) / cam["f_px"], np.ones_like(u)], -1)
    dw = dc @ np.asarray(cam["R"], float)          # R^T dc  (row-vector form)
    dw /= np.linalg.norm(dw, axis=-1, keepdims=True)
    return np.asarray(cam["C"], float), dw


def intersect_plane(u, v, cam, n, d):
    """world point where the pixel ray meets the plane n.X + d = 0."""
    o, w = ray(u, v, cam)
    n = np.asarray(n, float)
    s = -(o @ n + d) / (w @ n)
    return o + s[..., None] * w


def _rz_cv(rho):
    c, s = np.cos(rho), np.sin(rho)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def load_camera(frame=120, kind="reference", params_path=None):
    """Build a camera dict from params/camera.json.
    kind='reference': reference-matched camera incl. per-frame drift (zoom/pan/roll in image space,
                      expressed as per-frame focal length, principal point and roll).
    kind='locked'   : drift-free (frame-120 calibration for all frames)."""
    P = json.load(open(params_path or ROOT / "params/camera.json"))
    ref = P["reference_camera"]
    R0 = np.array(ref["extrinsics"]["R_world_to_cv"], float)
    C = np.array(ref["extrinsics"]["C_world_m"], float)
    f0 = ref["intrinsics"]["f_px"]; c0 = np.array([ref["intrinsics"]["cx_px"], ref["intrinsics"]["cy_px"]])
    if kind == "locked":
        return {"R": R0, "C": C, "f_px": f0, "cx": c0[0], "cy": c0[1], "frame": frame, "kind": kind}
    d = P["drift"]["model"]                     # smooth polynomial in t of image similarity vs frame 120
    t = float(frame) / 24.0
    s = np.polyval(d["scale_poly"], t); rho = np.polyval(d["roll_poly"], t)
    tx = np.polyval(d["tx_poly"], t); ty = np.polyval(d["ty_poly"], t)
    c, sn = np.cos(rho), np.sin(rho)
    cn = s * np.array([[c, -sn], [sn, c]]) @ c0 + np.array([tx, ty])
    return {"R": _rz_cv(rho) @ R0, "C": C, "f_px": s * f0, "cx": cn[0], "cy": cn[1], "frame": frame, "kind": kind}


def blender_camera_settings(cam, sensor_width_mm=36.0):
    """Equivalent Blender camera (sensor_fit HORIZONTAL, 1920x1088, pixel aspect 1):
    location, 3x3 world rotation (Blender camera looks along local -Z, +Y up), lens_mm, shift_x/y."""
    R = np.asarray(cam["R"], float)
    Rbl = R.T @ np.diag([1.0, -1.0, -1.0])          # columns: camera local x, y, z in world
    return {"location": list(map(float, cam["C"])), "rotation_matrix": Rbl.tolist(),
            "lens_mm": float(cam["f_px"] * sensor_width_mm / W), "sensor_width_mm": sensor_width_mm,
            "sensor_fit": "HORIZONTAL",
            "shift_x": float(-(cam["cx"] - W / 2) / W), "shift_y": float((cam["cy"] - H / 2) / W)}


if __name__ == "__main__":
    cam = load_camera()
    u, v = project(np.array([[0, 0, 0.0]]), cam)
    print("pivot projects to", u, v)
