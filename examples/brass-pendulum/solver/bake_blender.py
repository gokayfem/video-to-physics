"""Bake the validated solver trajectory onto PND_Swing as cubic-Hermite Bezier keyframes.

blender -b <file.blend> --python solver/bake_blender.py -- --trajectory solver/out/trajectory.csv \
        [--out baked.blend | --save] [--report solver/out/bake_check.json] [--bob-object NAME] \
        [--check-only] [--fix-scene-timing]

Convention (CONTRACT): PND_Swing is an empty at P; children hang along its local -Z.  A rotation by
phi about world +Y maps (0,0,-L) to (-L sin phi, 0, -L cos phi), so to obtain the bob centre
P + L_c (sin th, 0, -cos th) we key  rotation_euler[1] = phi = -theta   (rotation_mode XYZ, X=Z=0).
This sign is verified numerically from the evaluated world matrix after the bake.

Hermite construction: key at frame n has value v_n = -theta_n, slope dv/dframe = -omega_n / 24.
Handles are placed at +/- 1/3 frame with value offset +/- slope/3, handle type FREE.  With equal
1/3 time spacing the Bezier x(u) is linear in u, so the segment is exactly the cubic Hermite
interpolant of (theta, omega) at the two ends (error estimate ~5e-7 rad at 24 fps for this motion).

Reads only the trajectory (and trajectory_meta.json next to it); no physics in here.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

FPS = 24
SUB = 4  # trajectory samples per frame (1/96 s)


def sha256_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_traj(path):
    with open(path) as f:
        r = csv.reader(f)
        head = next(r)
        rows = np.array([[float(x) for x in row] for row in r])
    col = {h: i for i, h in enumerate(head)}
    t = rows[:, col["t_s"]]
    k = np.round(t * FPS * SUB).astype(int)
    if np.any(np.abs(t * FPS * SUB - k) > 1e-9) or np.any(np.diff(k) != 1) or k[0] != 0:
        raise SystemExit("trajectory must be sampled at exactly k/96 s from t = 0")
    return t, rows[:, col["theta_rad"]], rows[:, col["omega_rad_s"]], rows, col


def get_fcurve(obj):
    from bpy_extras import anim_utils
    ad = obj.animation_data or obj.animation_data_create()
    name = "PND_Swing_bake"
    act = bpy.data.actions.get(name)
    if act is not None:
        bpy.data.actions.remove(act)
    act = bpy.data.actions.new(name)
    ad.action = act
    if ad.action_slot is None:
        slot = act.slots.new(id_type="OBJECT", name=obj.name)
        ad.action_slot = slot
    cb = anim_utils.action_ensure_channelbag_for_slot(act, ad.action_slot)
    fc = cb.fcurves.new("rotation_euler", index=1)
    return act, fc


def bake(obj, theta, omega, n_frames):
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = (0.0, 0.0, 0.0)
    act, fc = get_fcurve(obj)
    fc.auto_smoothing = "NONE"
    fc.extrapolation = "CONSTANT"
    kps = fc.keyframe_points
    kps.add(n_frames)
    frames = np.arange(n_frames, dtype=float)
    th = theta[::SUB][:n_frames]
    om = omega[::SUB][:n_frames]
    v = -th
    slope = -om / FPS           # d(value)/d(frame)
    for i, kp in enumerate(kps):
        kp.interpolation = "BEZIER"
        kp.handle_left_type = "FREE"
        kp.handle_right_type = "FREE"
    co = np.column_stack([frames, v]).ravel()
    hl = np.column_stack([frames - 1.0 / 3.0, v - slope / 3.0]).ravel()
    hr = np.column_stack([frames + 1.0 / 3.0, v + slope / 3.0]).ravel()
    kps.foreach_set("co", co.astype(np.float32))
    kps.foreach_set("handle_left", hl.astype(np.float32))
    kps.foreach_set("handle_right", hr.astype(np.float32))
    # read back: Blender must not have recomputed handles
    got = {}
    for prop, want in (("co", co), ("handle_left", hl), ("handle_right", hr)):
        buf = np.empty(2 * n_frames, dtype=np.float32)
        kps.foreach_get(prop, buf)
        got[prop] = buf
        err = float(np.max(np.abs(buf - want.astype(np.float32))))   # must be bit-identical
        if err != 0.0:
            raise SystemExit(f"keyframe {prop} readback mismatch {err}")
    h = hashlib.sha256()
    for prop in ("co", "handle_left", "handle_right"):
        h.update(got[prop].tobytes())
    return act, fc, h.hexdigest()


def check(scene, obj, t, theta, n_frames, L_c, bob=None, subframes=(0.0, 0.25, 0.5, 0.75)):
    """Developer check: evaluated world transform at frames and subframes vs trajectory."""
    worst = {"angle_err_rad": 0.0}
    errs = []
    max_y = 0.0; max_piv = 0.0; max_len = 0.0; max_bob_pos = 0.0
    for n in range(n_frames):
        for s in subframes:
            if n == n_frames - 1 and s > 0:
                continue
            scene.frame_set(n, subframe=s)
            mw = obj.matrix_world
            d = mw.to_3x3() @ Vector((0.0, 0.0, -1.0))
            th_eval = math.atan2(d.x, -d.z)
            k = SUB * n + int(round(s * SUB))
            e = th_eval - theta[k]
            errs.append(e)
            max_y = max(max_y, abs(d.y) * L_c)
            max_piv = max(max_piv, mw.translation.length)
            if bob is not None:
                pb = bob.matrix_world.translation
                want = Vector((L_c * math.sin(theta[k]), 0.0, -L_c * math.cos(theta[k])))
                max_bob_pos = max(max_bob_pos, (pb - want).length)
                max_len = max(max_len, abs((pb - mw.translation).length - L_c))
            if abs(e) > worst["angle_err_rad"]:
                worst = {"angle_err_rad": abs(e), "frame": n, "subframe": s}
    errs = np.array(errs)
    scene.frame_set(0)
    on_frames = np.array(errs[::len(subframes)])
    return {"n_samples": int(len(errs)), "subframes": list(subframes),
            "max_abs_angle_err_rad": float(np.max(np.abs(errs))),
            "rms_angle_err_rad": float(np.sqrt(np.mean(errs ** 2))),
            "max_abs_angle_err_on_frames_rad": float(np.max(np.abs(on_frames))),
            "worst": worst, "max_out_of_plane_m": max_y, "max_pivot_drift_m": max_piv,
            "bob_object": bob.name if bob is not None else None,
            "max_bob_position_err_m": max_bob_pos if bob is not None else None,
            "max_bob_length_err_m": max_len if bob is not None else None,
            "euler_sign_verified": bool(np.max(np.abs(errs)) < 1e-4)}


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--object", default="PND_Swing")
    ap.add_argument("--bob-object", default=None)
    ap.add_argument("--out", default=None, help="save baked file here")
    ap.add_argument("--save", action="store_true", help="save in place")
    ap.add_argument("--report", default=None)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--fix-scene-timing", action="store_true")
    a = ap.parse_args(argv)

    scene = bpy.context.scene
    timing = (scene.render.fps, scene.render.fps_base, scene.frame_start, scene.frame_end)
    if timing != (24, 1.0, 0, 240):
        if a.fix_scene_timing:
            scene.render.fps, scene.render.fps_base, scene.frame_start, scene.frame_end = 24, 1.0, 0, 240
        else:
            raise SystemExit(f"scene timing {timing} != (24, 1.0, 0, 240); use --fix-scene-timing")
    obj = bpy.data.objects.get(a.object)
    if obj is None:
        raise SystemExit(f"object {a.object!r} not found")
    if obj.parent is not None:
        raise SystemExit(f"{a.object} must not be parented (world rotation about +Y required)")
    bob = bpy.data.objects.get(a.bob_object) if a.bob_object else None

    t, theta, omega, rows, col = load_traj(a.trajectory)
    n_frames = 241
    if len(t) < SUB * (n_frames - 1) + 1:
        raise SystemExit("trajectory shorter than 10 s")
    meta_path = a.meta or os.path.join(os.path.dirname(a.trajectory), "trajectory_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    traj_sha = sha256_file(a.trajectory)
    if meta.get("trajectory_sha256") not in (None, traj_sha):
        raise SystemExit("trajectory sha256 does not match trajectory_meta.json")
    L_c = float(meta.get("derived_model", {}).get("L_c")) if meta else \
        float(math.hypot(rows[0, col["x_m"]], rows[0, col["z_m"]]))

    report = {"blend": bpy.data.filepath, "object": a.object, "trajectory": a.trajectory,
              "trajectory_sha256": traj_sha, "params_sha256": meta.get("params_sha256"),
              "solver_source_sha256": meta.get("solver_source_sha256"), "L_c_m": L_c,
              "convention": "rotation_euler[1] = -theta (XYZ), Bezier FREE handles at +/-1/3 frame, value +/- (-omega/24)/3"}
    if not a.check_only:
        act, fc, kh = bake(obj, theta, omega, n_frames)
        obj["sim01_params_sha256"] = meta.get("params_sha256") or ""
        obj["sim01_trajectory_sha256"] = traj_sha
        obj["sim01_solver_source_sha256"] = meta.get("solver_source_sha256") or ""
        obj["sim01_bake"] = ("BAKED from solver/out/trajectory.csv by solver/bake_blender.py: "
                             "rotation_euler[1] = -theta, cubic Hermite Bezier keys on frames 0..240")
        obj["sim01_bake_sha256"] = sha256_file(os.path.abspath(__file__))
        report["keyframe_arrays_sha256"] = kh
        report["action"] = act.name
    report["check"] = check(scene, obj, t, theta, n_frames, L_c, bob)
    scene.frame_set(0)
    d0 = obj.matrix_world.to_3x3() @ Vector((0, 0, -1))
    report["theta0_blend_err_rad"] = abs(math.atan2(d0.x, -d0.z) - theta[0])
    report["pass_P6_1e-5"] = report["check"]["max_abs_angle_err_rad"] < 1e-5
    print("BAKE_REPORT " + json.dumps(report))
    if a.report:
        with open(a.report, "w") as f:
            json.dump(report, f, indent=2)
    if a.out:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out))
    elif a.save:
        bpy.ops.wm.save_mainfile()
    if not report["pass_P6_1e-5"]:
        sys.exit(2)


main()
