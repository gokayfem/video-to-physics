"""Bake the validated solver trajectory (clip 03, torsion disc) onto the rig as cubic-Hermite Bezier keyframes.

blender -b <file.blend> --python solver/bake_blender.py -- --trajectory solver/out/trajectory.csv \
        [--out baked.blend | --save] [--report solver/out/bake_check.json] [--check-only] [--fix-scene-timing]
        [--keys-per-frame 4] [--create-test-rig]

Rig contract (solver/NOTES.md "Bake contract"; scene role builds the objects):
  TOR_Anchor    empty at the suspension point = params pivot_world_m (A + fitted horizontal offset), NOT animated (checked)
  TOR_Disc      disc object, origin = B = centre of the TOP face (wire attachment), local +Z = disc up-normal,
                local +X = zero-twist reference; the engraved mark lies at local azimuth mu (params mark.mu_body_rad);
                no parent; location keyed = B(t); rotation_mode 'QUATERNION' (scene/NOTES.md contract), rotation_quaternion
                keyed from q_* with slopes qdot_* (Blender normalises the quaternion on evaluation)
  TOR_Wire      scene GN wire TOR_Anchor -> TOR_Disc origin (not keyed); optional TOR_WireEnd empty is keyed = B(t) if present
Hermite construction (per channel): keys on EVERY 1/96 s trajectory sample by default (--keys-per-frame 4: frames n + 0, .25,
.5, .75), value v, slope dv/dframe = (dv/dt)/24, FREE handles at +-1/(3*KPF) frame -> each segment is exactly the cubic Hermite
interpolant of (value, rate) at both ends.  4 keys/frame: the tilt mode (~14 rad/s) and twist rates need sub-frame keys for P6.
Reads only the trajectory and its meta (no physics here).  Check: evaluated world transforms at frames and 4 subframes vs the
trajectory (positions in m, rotation-matrix entries), anchor drift, wire-end attachment (|TOR_WireEnd - TOR_Disc origin|),
wire length |TOR_WireEnd - TOR_Anchor| = ell.
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Quaternion, Vector

FPS = 24
SUB = 4
KPF = 4
PIVOT = np.zeros(3)


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
    kps.foreach_set("co", co.astype(np.float32)); kps.foreach_set("handle_left", hl.astype(np.float32))
    kps.foreach_set("handle_right", hr.astype(np.float32))
    h = hashlib.sha256()
    for prop, want in (("co", co), ("handle_left", hl), ("handle_right", hr)):
        buf = np.empty(2 * nk, dtype=np.float32); kps.foreach_get(prop, buf)
        if float(np.max(np.abs(buf - want.astype(np.float32)))) != 0.0:
            raise SystemExit(f"keyframe {prop} readback mismatch on {path}[{index}]")
        h.update(buf.tobytes())
    return h.hexdigest()


def bake(T, disc, wend, n_frames):
    hashes = {}
    objs = [(disc, "TOR_Disc_bake")] + ([(wend, "TOR_WireEnd_bake")] if wend is not None else [])
    for obj, nm in objs:
        obj.location = (0, 0, 0)
        act, cb = new_action(obj, nm)
        for i, ax in enumerate("xyz"):
            hashes[f"{obj.name}.location[{i}]"] = key_channel(cb, "location", i, T[f"B_{ax}_m"], T[f"Bdot_{ax}_m_s"], n_frames)
        if obj is disc:
            obj.rotation_mode = "QUATERNION"; obj.rotation_quaternion = (1, 0, 0, 0)
            for i, ax in enumerate("wxyz"):
                hashes[f"{obj.name}.rotation_quaternion[{i}]"] = key_channel(cb, "rotation_quaternion", i, T[f"q_{ax}"], T[f"qdot_{ax}_1_s"], n_frames)
    return hashes


def check(scene, T, anchor, disc, wend, ell, h_c, n_frames, subframes=(0.0, 0.25, 0.5, 0.75)):
    worst = {}
    scene.frame_set(0)                     # evaluate the depsgraph first (matrix_world of a just-moved test-rig anchor is stale)
    A0 = anchor.matrix_world.translation.copy() if anchor else Vector((0, 0, 0))
    e_anchor = 0.0
    for n in range(n_frames):
        for s in subframes:
            if n == n_frames - 1 and s > 0:
                continue
            scene.frame_set(n, subframe=s)
            k = SUB * n + int(round(s * SUB))
            B = np.array([T["B_x_m"][k], T["B_y_m"][k], T["B_z_m"][k]])
            X = np.array([T["X_x_m"][k], T["X_y_m"][k], T["X_z_m"][k]])
            Rq = np.array(Quaternion((T["q_w"][k], T["q_x"][k], T["q_y"][k], T["q_z"][k])).to_matrix())
            dm = disc.matrix_world; Rb = np.array(dm.to_3x3()); dp = np.array(dm.translation)
            wp = np.array(wend.matrix_world.translation) if wend is not None else dp
            com = dp + Rb @ np.array([0.0, 0.0, -h_c])
            Ap = np.array(anchor.matrix_world.translation) if anchor else PIVOT
            vals = {"disc_origin": np.max(np.abs(dp - B)), "wire_end": np.max(np.abs(wp - B)), "rot": np.max(np.abs(Rb - Rq)),
                    "com": np.max(np.abs(com - X)), "attach": float(np.linalg.norm(wp - dp)),
                    "wire_len": abs(float(np.linalg.norm(wp - Ap)) - ell)}
            if anchor:
                e_anchor = max(e_anchor, (anchor.matrix_world.translation - A0).length + float(np.linalg.norm(np.array(A0) - PIVOT)))   # static AND at the pivot
            for key, v in vals.items():
                if v > worst.get(key, (0,))[0]:
                    worst[key] = (float(v), n, s)
    scene.frame_set(0)
    mx = lambda k: worst.get(k, (0.0,))[0]
    return {"subframes": list(subframes), "max_disc_origin_err_m": mx("disc_origin"), "max_wire_end_err_m": mx("wire_end"),
            "max_rotation_matrix_err": mx("rot"), "max_com_err_m": mx("com"), "max_attachment_gap_m": mx("attach"),
            "max_wire_length_err_m": mx("wire_len"), "max_anchor_offset_m": e_anchor, "worst": worst,
            "pass_P6_1e-5": bool(max(mx("disc_origin"), mx("wire_end"), mx("rot"), mx("com")) < 1e-5),
            "pass_P5_anchor_1e-6": bool(e_anchor < 1e-6),
            "pass_P5_attach_1e-6": bool(mx("attach") < 1e-6 and mx("wire_len") < 1e-6)}


def check_midpoints(scene, T, disc, n_frames):
    """Between trajectory samples (1/8 .. 7/8 of a sample interval): disc origin vs the cubic Hermite interpolant of the
    1/96 s solver samples (an O(h^4) reference), and the rotation vs the normalised quaternion Hermite interpolant."""
    worst = 0.0; worst_r = 0.0
    for n in range(n_frames - 1):
        for s8 in (0.125, 0.375, 0.625, 0.875):
            scene.frame_set(n, subframe=s8)
            k = SUB * n + int(s8 * SUB)
            w = s8 * SUB - int(s8 * SUB); dt = 1.0 / (FPS * SUB)

            def herm(p0, p1, m0, m1):
                return (2*w**3 - 3*w**2 + 1) * p0 + (w**3 - 2*w**2 + w) * m0 + (-2*w**3 + 3*w**2) * p1 + (w**3 - w**2) * m1
            B = np.array([herm(T[f"B_{a}_m"][k], T[f"B_{a}_m"][k + 1], T[f"Bdot_{a}_m_s"][k] * dt, T[f"Bdot_{a}_m_s"][k + 1] * dt) for a in "xyz"])
            worst = max(worst, float(np.max(np.abs(np.array(disc.matrix_world.translation) - B))))
            qh = [herm(T[f"q_{a}"][k], T[f"q_{a}"][k + 1], T[f"qdot_{a}_1_s"][k] * dt, T[f"qdot_{a}_1_s"][k + 1] * dt) for a in "wxyz"]
            Re = np.array(Quaternion(qh).normalized().to_matrix())
            worst_r = max(worst_r, float(np.max(np.abs(np.array(disc.matrix_world.to_3x3()) - Re))))
    scene.frame_set(0)
    return {"max_disc_origin_err_vs_hermite_ref_m": worst, "max_rot_err_vs_quaternion_hermite_ref": worst_r}


def main():
    global KPF
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", required=True); ap.add_argument("--meta", default=None)
    ap.add_argument("--out", default=None); ap.add_argument("--save", action="store_true")
    ap.add_argument("--report", default=None); ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--fix-scene-timing", action="store_true")
    ap.add_argument("--keys-per-frame", type=int, choices=[1, 2, 4], default=4)
    ap.add_argument("--create-test-rig", action="store_true", help="create bare TOR_* empties (developer test only)")
    a = ap.parse_args(argv)
    scene = bpy.context.scene
    timing = (scene.render.fps, scene.render.fps_base, scene.frame_start, scene.frame_end)
    if timing != (24, 1.0, 0, 240):
        if a.fix_scene_timing or a.create_test_rig:
            scene.render.fps, scene.render.fps_base, scene.frame_start, scene.frame_end = 24, 1.0, 0, 240
        else:
            raise SystemExit(f"scene timing {timing} != (24, 1.0, 0, 240); use --fix-scene-timing")
    if a.create_test_rig:
        for nm in ("TOR_Anchor", "TOR_Disc"):
            if nm not in bpy.data.objects:
                o = bpy.data.objects.new(nm, None); scene.collection.objects.link(o)
    anchor = bpy.data.objects.get("TOR_Anchor"); disc = bpy.data.objects.get("TOR_Disc"); wend = bpy.data.objects.get("TOR_WireEnd")
    if disc is None:
        raise SystemExit("TOR_Disc is required")
    for o in [x for x in (disc, wend) if x is not None]:
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
    ell = float(meta["derived_model"]["ell"]); h_c = float(meta["derived_model"]["h_c"])
    global PIVOT
    PIVOT = np.array(meta.get("rig", {}).get("pivot_world_m", [0.0, 0.0, 0.0]), float)
    if a.create_test_rig:
        bpy.data.objects["TOR_Anchor"].location = PIVOT.tolist()
    if anchor is not None and float(np.max(np.abs(np.array(anchor.matrix_world.translation) - PIVOT))) > 1e-6:
        print(f"WARNING: TOR_Anchor at {list(anchor.matrix_world.translation)} != pivot_world_m {PIVOT.tolist()} (scene must place the wire top at the pivot)")
    rep = {"blend": bpy.data.filepath, "trajectory": a.trajectory, "trajectory_sha256": traj_sha,
           "params_sha256": meta.get("params_sha256"), "solver_source_sha256": meta.get("solver_source_sha256"), "ell_m": ell, "h_c_m": h_c,
           "convention": "TOR_Disc.location = B (top-face centre), rotation_quaternion = solver q (wxyz); optional TOR_WireEnd.location = B; FREE Bezier handles"}
    if not a.check_only:
        rep["keyframe_arrays_sha256"] = bake(T, disc, wend, n_frames)
        for o in [x for x in (disc, wend) if x is not None]:
            o["tor_params_sha256"] = meta.get("params_sha256") or ""
            o["tor_trajectory_sha256"] = traj_sha
            o["tor_solver_source_sha256"] = meta.get("solver_source_sha256") or ""
            o["tor_bake"] = "BAKED from solver/out/trajectory.csv by solver/bake_blender.py (cubic Hermite Bezier keys, frames 0..240)"
            o["tor_bake_sha256"] = sha256_file(os.path.abspath(__file__))
    rep["keys_per_frame"] = KPF
    rep["check"] = check(scene, T, anchor, disc, wend, ell, h_c, n_frames)
    rep["check_midpoints"] = check_midpoints(scene, T, disc, n_frames)
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
