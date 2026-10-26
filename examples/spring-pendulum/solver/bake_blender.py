"""Bake the validated solver trajectory (clip 02) onto the rig as cubic-Hermite Bezier keyframes on every frame.

blender -b <file.blend> --python solver/bake_blender.py -- --trajectory solver/out/trajectory.csv \
        [--out baked.blend | --save] [--report solver/out/bake_check.json] [--check-only] [--fix-scene-timing]

Rig contract (solver/NOTES.md):
  VSP_Anchor     empty at A = world origin, NOT animated (checked)
  VSP_HookLower  empty, location keyed = H(t) (lower hook = hinge), no parent
  VSP_Weight     object with origin at the hinge H, local -Z = body axis n (hinge -> CoM), no parent;
                 location keyed = H(t), rotation_mode 'XYZ', rotation_euler keyed from the solver quaternion
                 (columns eul_*_rad, unwrapped; rates euld_*_rad_s computed analytically from the angular velocity).
Hermite construction (per channel): keys on EVERY FRAME and, by default (--keys-per-frame 4), also on every 1/96 s
trajectory sample (frames n + 0, .25, .5, .75), value v, slope dv/dframe = (dv/dt)/24; FREE handles at +-1/(3*KPF) frame
with value +- slope/(3*KPF) -> each segment is exactly the cubic Hermite interpolant of (value, rate) at both ends.
Why 4 keys per frame: the weight's rocking mode (~5 Hz, FORMULATION s4) makes frame-only Hermite keys miss P6 (1e-5) by
up to ~1.6e-4 rad between frames; with 1/96 s keys the interpolation error is ~1e-7.
Reads only the trajectory and its meta (no physics here).  Check: evaluated world transforms at frames and 4 subframes vs
the trajectory (positions in m, body axis in rad), anchor drift, spring endpoint attachment.
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
SUB = 4


def sha256_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_traj(path):
    with open(path) as f:
        r = csv.reader(f); head = next(r); rows = np.array([[float(x) for x in row] for row in r])
    col = {h: i for i, h in enumerate(head)}
    t = rows[:, col["t_s"]]
    k = np.round(t * FPS * SUB).astype(int)
    if np.any(np.abs(t * FPS * SUB - k) > 1e-9) or np.any(np.diff(k) != 1) or k[0] != 0:
        raise SystemExit("trajectory must be sampled at exactly k/96 s from t = 0")
    return {h: rows[:, i] for h, i in col.items()}


def new_action(obj, name):
    from bpy_extras import anim_utils
    ad = obj.animation_data or obj.animation_data_create()
    act = bpy.data.actions.get(name)
    if act is not None:
        bpy.data.actions.remove(act)
    act = bpy.data.actions.new(name); ad.action = act
    if ad.action_slot is None:
        ad.action_slot = act.slots.new(id_type="OBJECT", name=obj.name)
    cb = anim_utils.action_ensure_channelbag_for_slot(act, ad.action_slot)
    return act, cb


KPF = 4   # keys per frame (set in main): 1 = keys on frames only; 4 = keys on every 1/96 s trajectory sample


def key_channel(cb, path, index, val, rate, n_frames):
    fc = cb.fcurves.new(path, index=index)
    fc.auto_smoothing = "NONE"; fc.extrapolation = "CONSTANT"
    step = SUB // KPF
    nk = (n_frames - 1) * KPF + 1
    kps = fc.keyframe_points; kps.add(nk)
    fr = np.arange(nk, dtype=float) / KPF
    v = val[::step][:nk]; sl = rate[::step][:nk] / FPS
    dh = 1.0 / (3 * KPF)
    for kp in kps:
        kp.interpolation = "BEZIER"; kp.handle_left_type = "FREE"; kp.handle_right_type = "FREE"
    co = np.column_stack([fr, v]).ravel(); hl = np.column_stack([fr - dh, v - sl * dh]).ravel()
    hr = np.column_stack([fr + dh, v + sl * dh]).ravel()
    n_frames = nk
    kps.foreach_set("co", co.astype(np.float32)); kps.foreach_set("handle_left", hl.astype(np.float32))
    kps.foreach_set("handle_right", hr.astype(np.float32))
    h = hashlib.sha256()
    for prop, want in (("co", co), ("handle_left", hl), ("handle_right", hr)):
        buf = np.empty(2 * n_frames, dtype=np.float32); kps.foreach_get(prop, buf)
        if float(np.max(np.abs(buf - want.astype(np.float32)))) != 0.0:
            raise SystemExit(f"keyframe {prop} readback mismatch on {path}[{index}]")
        h.update(buf.tobytes())
    return h.hexdigest()


def bake(T, hook, weight, n_frames):
    hashes = {}
    for obj, nm in ((hook, "VSP_HookLower_bake"), (weight, "VSP_Weight_bake")):
        obj.rotation_mode = "XYZ"; obj.location = (0, 0, 0); obj.rotation_euler = (0, 0, 0)
        act, cb = new_action(obj, nm)
        for i, ax in enumerate("xyz"):
            hashes[f"{obj.name}.location[{i}]"] = key_channel(cb, "location", i, T[f"H_{ax}_m"], T[f"Hdot_{ax}_m_s"], n_frames)
        if obj is weight:
            for i, ax in enumerate("xyz"):
                hashes[f"{obj.name}.rotation_euler[{i}]"] = key_channel(cb, "rotation_euler", i, T[f"eul_{ax}_rad"], T[f"euld_{ax}_rad_s"], n_frames)
    return hashes


def check(scene, T, anchor, hook, weight, b, n_frames, subframes=(0.0, 0.25, 0.5, 0.75)):
    """subframes must be multiples of 1/4 (trajectory samples); see check_midpoints for in-between times."""
    e_hook = e_wpos = e_n = e_com = e_twist = e_anchor = 0.0; worst = {}
    A0 = anchor.matrix_world.translation.copy() if anchor else Vector((0, 0, 0))
    for n in range(n_frames):
        for s in subframes:
            if n == n_frames - 1 and s > 0:
                continue
            scene.frame_set(n, subframe=s)
            k = SUB * n + int(round(s * SUB))
            H = np.array([T["H_x_m"][k], T["H_y_m"][k], T["H_z_m"][k]])
            nn = np.array([T["n_x"][k], T["n_y"][k], T["n_z"][k]])
            X = np.array([T["X_x_m"][k], T["X_y_m"][k], T["X_z_m"][k]])
            hp = np.array(hook.matrix_world.translation); wm = weight.matrix_world
            wp = np.array(wm.translation); nb = np.array(wm.to_3x3() @ Vector((0, 0, -1)))
            com = wp + b * nb
            q = (T["q_w"][k], T["q_x"][k], T["q_y"][k], T["q_z"][k])
            from mathutils import Quaternion
            Rq = np.array(Quaternion(q).to_matrix()); Rb = np.array(wm.to_3x3())
            vals = {"hook": np.max(np.abs(hp - H)), "wpos": np.max(np.abs(wp - H)), "n": np.max(np.abs(nb - nn)),
                    "com": np.max(np.abs(com - X)), "rot": np.max(np.abs(Rb - Rq))}
            if anchor:
                e_anchor = max(e_anchor, (anchor.matrix_world.translation - A0).length + A0.length)
            for key, v in vals.items():
                if v > worst.get(key, (0,))[0]:
                    worst[key] = (float(v), n, s)
            e_hook = max(e_hook, vals["hook"]); e_wpos = max(e_wpos, vals["wpos"]); e_n = max(e_n, vals["n"])
            e_com = max(e_com, vals["com"]); e_twist = max(e_twist, vals["rot"])
    scene.frame_set(0)
    return {"subframes": list(subframes), "max_hook_pos_err_m": e_hook, "max_weight_origin_err_m": e_wpos,
            "max_body_axis_err": e_n, "max_com_err_m": e_com, "max_rotation_matrix_err": e_twist,
            "max_anchor_offset_m": e_anchor, "worst": worst,
            "pass_P6_1e-5": bool(max(e_hook, e_wpos, e_n, e_com, e_twist) < 1e-5),
            "pass_P5_anchor_1e-6": bool(e_anchor < 1e-6)}


def check_midpoints(scene, T, hook, weight, b, n_frames):
    """Between trajectory samples (subframe offsets 1/8, 3/8, 5/8, 7/8): compare with the cubic Hermite interpolant of the
    1/96 s samples of the SOLVER values (an O(h^4) reference, not the solver itself; its own error is reported too)."""
    from mathutils import Quaternion
    worst = 0.0; worst_n = 0.0
    for n in range(n_frames - 1):
        for s8 in (0.125, 0.375, 0.625, 0.875):
            scene.frame_set(n, subframe=s8)
            k = SUB * n + int(s8 * SUB)
            w = s8 * SUB - int(s8 * SUB); dt = 1.0 / (FPS * SUB)
            def herm(c, cd):
                p0, p1, m0, m1 = T[c][k], T[c][k + 1], T[cd][k] * dt, T[cd][k + 1] * dt
                return (2*w**3 - 3*w**2 + 1) * p0 + (w**3 - 2*w**2 + w) * m0 + (-2*w**3 + 3*w**2) * p1 + (w**3 - w**2) * m1
            H = np.array([herm(f"H_{a}_m", f"Hdot_{a}_m_s") for a in "xyz"])
            nn = np.array([herm(f"n_{a}", f"ndot_{a}") for a in "xyz"])
            worst = max(worst, float(np.max(np.abs(np.array(hook.matrix_world.translation) - H))))
            worst_n = max(worst_n, float(np.max(np.abs(np.array(weight.matrix_world.to_3x3() @ Vector((0, 0, -1))) - nn))))
    scene.frame_set(0)
    return {"max_hook_err_vs_hermite_ref_m": worst, "max_axis_err_vs_hermite_ref": worst_n}


def main():
    global KPF
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", required=True); ap.add_argument("--meta", default=None)
    ap.add_argument("--out", default=None); ap.add_argument("--save", action="store_true")
    ap.add_argument("--report", default=None); ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--fix-scene-timing", action="store_true")
    ap.add_argument("--keys-per-frame", type=int, choices=[1, 2, 4], default=4)
    ap.add_argument("--create-test-rig", action="store_true", help="create bare VSP_* empties (developer test only)")
    a = ap.parse_args(argv)
    scene = bpy.context.scene
    timing = (scene.render.fps, scene.render.fps_base, scene.frame_start, scene.frame_end)
    if timing != (24, 1.0, 0, 240):
        if a.fix_scene_timing or a.create_test_rig:
            scene.render.fps, scene.render.fps_base, scene.frame_start, scene.frame_end = 24, 1.0, 0, 240
        else:
            raise SystemExit(f"scene timing {timing} != (24, 1.0, 0, 240); use --fix-scene-timing")
    if a.create_test_rig:
        for nm in ("VSP_Anchor", "VSP_HookLower", "VSP_Weight"):
            if nm not in bpy.data.objects:
                o = bpy.data.objects.new(nm, None); scene.collection.objects.link(o)
    anchor = bpy.data.objects.get("VSP_Anchor"); hook = bpy.data.objects.get("VSP_HookLower"); weight = bpy.data.objects.get("VSP_Weight")
    if hook is None or weight is None:
        raise SystemExit("VSP_HookLower and VSP_Weight are required")
    for o in (hook, weight):
        if o.parent is not None:
            raise SystemExit(f"{o.name} must not be parented (world transforms are keyed)")
    KPF = a.keys_per_frame
    T = load_traj(a.trajectory)
    n_frames = 241
    meta_path = a.meta or os.path.join(os.path.dirname(a.trajectory), "trajectory_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    traj_sha = sha256_file(a.trajectory)
    if meta.get("trajectory_sha256") not in (None, traj_sha):
        raise SystemExit("trajectory sha256 does not match trajectory_meta.json")
    b = float(meta["derived_model"]["b"])
    rep = {"blend": bpy.data.filepath, "trajectory": a.trajectory, "trajectory_sha256": traj_sha,
           "params_sha256": meta.get("params_sha256"), "solver_source_sha256": meta.get("solver_source_sha256"), "b_m": b,
           "convention": "VSP_HookLower.location = H; VSP_Weight.location = H, rotation_euler XYZ = solver quaternion; FREE Bezier handles +-1/3 frame"}
    if not a.check_only:
        rep["keyframe_arrays_sha256"] = bake(T, hook, weight, n_frames)
        for o in (hook, weight):
            o["vsp_params_sha256"] = meta.get("params_sha256") or ""
            o["vsp_trajectory_sha256"] = traj_sha
            o["vsp_solver_source_sha256"] = meta.get("solver_source_sha256") or ""
            o["vsp_bake"] = "BAKED from solver/out/trajectory.csv by solver/bake_blender.py (cubic Hermite Bezier keys, frames 0..240)"
            o["vsp_bake_sha256"] = sha256_file(os.path.abspath(__file__))
    rep["keys_per_frame"] = KPF
    rep["check"] = check(scene, T, anchor, hook, weight, b, n_frames)
    rep["check_midpoints"] = check_midpoints(scene, T, hook, weight, b, n_frames)
    print("BAKE_REPORT " + json.dumps(rep, default=float))
    if a.report:
        with open(a.report, "w") as f:
            json.dump(rep, f, indent=2, default=float)
    if a.out:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out))
    elif a.save:
        bpy.ops.wm.save_mainfile()
    if not rep["check"]["pass_P6_1e-5"]:
        sys.exit(2)


main()
