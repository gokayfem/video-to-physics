"""Static-scene tilt audit inside Blender (verifier-owned).
  blender -b F.blend --python verify/check_tilt.py -- --params physics.json --trajectory traj.csv --out R.json
Checks: PND_Swing unparented, untilted (world rotation only about Y, = -theta, translation 0) and gravity-referenced
(bob world angle from -Z_world equals solver theta); tilt parent empty at P with rotation (0, +level_tilt, 0), not animated;
which objects are (not) under it; stand up-axis angle in world; bob rest direction expressed in the stand frame."""
import bpy, sys, json, math, argparse, hashlib
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser(); ap.add_argument("--params", required=True); ap.add_argument("--trajectory", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--tilt-name", default="PND_StaticTilt"); a = ap.parse_args(argv)
P = json.load(open(a.params)); tilt = P["level_tilt_rad"]["value"]; Lc = P["L_c"]["value"]
T = np.genfromtxt(a.trajectory, delimiter=",", names=True)
sc = bpy.context.scene; ob = bpy.data.objects
sw = ob["PND_Swing"]; bob = ob["PND_Bob"]; ti = ob.get(a.tilt_name)
def anc(o):
    r = []
    while o.parent: o = o.parent; r.append(o.name)
    return r
rep = dict(blend=bpy.data.filepath, blend_sha256=hashlib.sha256(open(bpy.data.filepath, "rb").read()).hexdigest(),
           params_sha256=hashlib.sha256(open(a.params, "rb").read()).hexdigest(), level_tilt_rad=tilt)
rep["swing"] = dict(parent=sw.parent.name if sw.parent else None, constraints=[c.type for c in sw.constraints], ancestors=anc(sw))
rep["tilt_empty_found"] = ti is not None
if ti:
    rep["tilt_empty"] = dict(type=ti.type, parent=ti.parent.name if ti.parent else None, rotation_mode=ti.rotation_mode,
                             rotation_euler=list(ti.rotation_euler), location=list(ti.location), animated=bool(ti.animation_data and ti.animation_data.action),
                             constraints=[c.type for c in ti.constraints])
under, notunder = [], []
for o in ob:
    if o == ti: continue
    (under if (ti and ti.name in anc(o)) else notunder).append(f"{o.name}:{o.type}")
rep["objects_under_tilt"] = sorted(under); rep["objects_not_under_tilt"] = sorted(notunder)
rep["cameras_under_tilt"] = {o.name: bool(ti and ti.name in anc(o)) for o in ob if o.type == "CAMERA"}
rep["swing_under_tilt"] = bool(ti and ti.name in anc(sw))
# per-frame checks
maxrot_offaxis = maxloc = maxdth = maxdth_tilt = 0.0; stand_up = []
rows = T["t_s"]
for n in range(0, 241):
    sc.frame_set(n)
    M = sw.matrix_world
    R = np.array(M.to_3x3()); loc = np.array(M.translation)
    th = T["theta_rad"][4 * n]
    # expected world rotation of swing = rotation about +Y by -theta (bob at L(sin th,0,-cos th))
    c, s = math.cos(-th), math.sin(-th); Rexp = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    maxrot_offaxis = max(maxrot_offaxis, float(np.abs(R - Rexp).max())); maxloc = max(maxloc, float(np.abs(loc).max()))
    bw = np.array(bob.matrix_world.translation)
    maxdth = max(maxdth, abs(math.atan2(bw[0], -bw[2]) - th))
    if ti:
        Mt = np.array(ti.matrix_world)
        up = Mt[:3, :3] @ np.array([0, 0, 1.0]); stand_up.append(math.atan2(up[0], up[2]))
        bl = np.linalg.inv(Mt) @ np.append(bw, 1)
        maxdth_tilt = max(maxdth_tilt, abs(math.atan2(bl[0], -bl[2]) - (th + tilt)))
rep["swing_world_rotation_vs_RotY(-theta)_max_abs"] = maxrot_offaxis
rep["swing_world_translation_max_abs_m"] = maxloc
rep["bob_angle_from_minusZ_world_vs_theta_max_rad"] = maxdth
if ti:
    Mt = np.array(ti.matrix_world)
    rest_w = np.array([0, 0, -Lc, 1.0]); rl = np.linalg.inv(Mt) @ rest_w
    rep["stand_up_axis_angle_in_world_rad"] = dict(min=min(stand_up), max=max(stand_up), expected=tilt, max_err=max(abs(x - tilt) for x in stand_up))
    rep["tilt_empty_world_origin_m"] = list(Mt[:3, 3])
    rep["bob_rest_direction_in_stand_frame_rad"] = math.atan2(rl[0], -rl[2])
    rep["bob_rest_minus_stand_down_in_world_angle_rad"] = 0.0 - min(stand_up)
    rep["bob_in_stand_frame_equals_theta_plus_tilt_max_err_rad"] = maxdth_tilt
    ok = (rep["swing"]["parent"] is None and not rep["swing_under_tilt"] and maxrot_offaxis < 1e-6 and maxloc < 1e-9 and maxdth < 1e-6
          and abs(rep["bob_rest_direction_in_stand_frame_rad"] - tilt) < 1e-6 and rep["stand_up_axis_angle_in_world_rad"]["max_err"] < 1e-6
          and not rep["tilt_empty"]["animated"] and all(rep["cameras_under_tilt"].values()) and np.abs(Mt[:3, 3]).max() < 1e-9)
else:
    ok = False
rep["status"] = "PASS" if ok else "FAIL"
json.dump(rep, open(a.out, "w"), indent=2, default=float)
print("CHECK_TILT", rep["status"])
