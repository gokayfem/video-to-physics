"""Numerical rig check of a saved clip-02 .blend (scene role). Light Blender work (no rendering).
v002 rig: STATIC hook (top loop on A, S-bend, 2 hook turns) down to the hinge P_eff; the spring body P_eff -> H rotates
about P_eff (ramped over the bend length). Test poses are |P-H| x tilt about P x out-of-plane x weight rock.

blender -b scene/out/spring.blend --python scene/check_rig.py -- [--out scene/out/check_rig.json] [--frames]
For each TEST pose of VSP_Weight (H location + rotation; any baked action is detached in memory only, the file is
not saved) the evaluated VSP_Spring mesh is read with its point attributes vsp_part / vsp_cl (centreline) and:
  top     top-loop inner top on A:      max_cl(part 0) . u  == r_h ;  flat top segment centred on the axis
  bottom  lower-loop inner bottom on H: min_cl(part 5) . u  == -r_h (relative to H); bottom segment centred
  eye     eye ring inner top == H (weight origin)
  coils   winding of parts 2+3 about the axis == N_open + N_tight turns; no wire overlap (centreline points more
          than 0.75 turn apart are >= 2 r_w apart)
  links   lower loop vs eye ring: centreline distance >= r_h + r_e (touching at H, no interpenetration)
  clear   no spring/weight/eye vertex inside the crossbar (except the top loop, which wraps it), posts or plate;
          no spring vertex inside the weight cylinder; top loop centreline >= r_h from the bar box
u = unit vector from H to A. --frames additionally evaluates the baked frames 0..240 (if the file is baked).
PASS if every tolerance holds for every pose (1e-6 m attachments; 1e-5 m clearances).
"""
import argparse, json, math, pathlib, sys
import bpy
import numpy as np
from mathutils import Euler, Vector

ROOT = pathlib.Path(__file__).resolve().parents[1]
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "scene" / "out" / "check_rig.json"))
ap.add_argument("--frames", action="store_true")
a = ap.parse_args(argv)
sc = bpy.context.scene
rep_build = json.loads(pathlib.Path(bpy.data.filepath).with_suffix(".build.json").read_text())
D = rep_build["dims_m"]
S = rep_build["S_m"]
spring, weight, eye = bpy.data.objects["VSP_Spring"], bpy.data.objects["VSP_Weight"], bpy.data.objects["VSP_WeightEye"]
mod = [m for m in spring.modifiers if m.type == "NODES"][0]


sys.path.insert(0, str(ROOT / "scene" / "vendor"))
from nodekit import inspect as nkinspect  # noqa: E402
MODV = {n: nkinspect._slot(mod, i).value for n, i in nkinspect.modifier_inputs(mod).items()}   # exact (get_modifier_inputs rounds)


def modval(name):
    return MODV[name]


r_h, r_w, R_c = modval("Hook Wire Radius"), modval("Wire Radius"), modval("Coil Radius")
N = int(modval("Open Coils")) + int(modval("Tight Coils"))
RES = int(modval("Resolution"))
R_e, r_e = eye["vsp_ring_radius_m"], eye["vsp_wire_radius_m"]
Rw, hw = weight["vsp_cylinder_radius_m"], weight["vsp_cylinder_height_m"]
c_local = Vector(weight["vsp_cylinder_centre_local_m"])
Phinge = np.array([D["hinge_P_eff_x"], 0.0, D["hinge_P_eff_z"]])
N = N + int(modval("Hook Turns"))

# static boxes (world): posts, bar, plate (rounded corners ignored: boxes are conservative)
xc, half, pw, pd = D["frame_centre_x"], S / 2, D["post_width_x"], D["post_depth_y"]
boxes = {
    "post_L": ((xc - half - pw, -pd / 2, D["base_top_z"]), (xc - half, pd / 2, D["post_top_z"])),
    "post_R": ((xc + half, -pd / 2, D["base_top_z"]), (xc + half + pw, pd / 2, D["post_top_z"])),
    "bar": ((xc - half, -D["bar_depth_y"] / 2, -D["bar_height_z"]), (xc + half, D["bar_depth_y"] / 2, 0.0)),
    "plate": ((xc - D["plate_width_x"] / 2, D["plate_centre_y"] - D["plate_depth_y"] / 2, D["base_top_z"] - D["plate_thickness"]),
              (xc + D["plate_width_x"] / 2, D["plate_centre_y"] + D["plate_depth_y"] / 2, D["base_top_z"])),
}


def inside(P, box, tol=1e-5):
    lo, hi = np.array(box[0]) + tol, np.array(box[1]) - tol
    return np.all((P > lo) & (P < hi), axis=1)


def box_dist(P, box):
    lo, hi = np.array(box[0]), np.array(box[1])
    d = np.maximum(np.maximum(lo - P, P - hi), 0.0)
    return np.linalg.norm(d, axis=1)


def evaluated(ob, attrs=()):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    m = ev.to_mesh()
    co = np.empty(len(m.vertices) * 3)
    m.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    Mw = np.array(ob.matrix_world)
    co = co @ Mw[:3, :3].T + Mw[:3, 3]
    out = {"co": co}
    for name, kind in attrs:
        at = m.attributes[name]
        if kind == "INT":
            v = np.empty(len(at.data), dtype=np.int32); at.data.foreach_get("value", v)
        else:
            v = np.empty(len(at.data) * 3); at.data.foreach_get("vector", v); v = v.reshape(-1, 3)
        out[name] = v
    ev.to_mesh_clear()
    return out


def weight_samples():
    """Dense analytic samples of the weight cylinder surface (rings every 0.5 mm, 180 around), world coordinates."""
    Mw = np.array(weight.matrix_world)
    th = np.linspace(0, 2 * math.pi, 180, endpoint=False)
    zs = np.arange(-hw / 2, hw / 2 + 1e-9, 0.0005)
    side = np.array([(Rw * math.cos(t), Rw * math.sin(t), z) for z in zs for t in th])
    rr = np.arange(0.0, Rw + 1e-9, 0.0005)
    caps = np.array([(r * math.cos(t), r * math.sin(t), z) for z in (-hw / 2, hw / 2) for r in rr for t in th])
    P = np.vstack([side, caps]) + np.array(c_local)
    return P @ Mw[:3, :3].T + Mw[:3, 3]


r_l = modval("Lower Loop Wire Radius")
LINK_TOL = (r_l + r_e) * (1 - math.cos(math.radians(8.0))) + 1e-5
# m: analytic bound of the fixed-hinge simplification. The physics hinge H is a FIXED body point; when the body rocks by
# up to 8 deg relative to the spring axis a real eye would roll along the loop wire, while rotating rigidly about H
# overlaps the two wire surfaces by <= (r_h + r_e)(1 - cos 8 deg) (0.08 mm at S_m 0.417 with the v003 wire radii; 1 px ~ 0.6 mm).


def check_pose(label):
    bpy.context.view_layer.update()
    H = np.array(weight.matrix_world.translation)
    sp = evaluated(spring, (("vsp_part", "INT"), ("vsp_cl", "VEC"), ("vsp_cl_rest", "VEC")))
    part, cl, clr, co = sp["vsp_part"], sp["vsp_cl"], sp["vsp_cl_rest"], sp["co"]
    u = (Phinge - H) / np.linalg.norm(Phinge - H)              # moving-body axis, from H towards P_eff
    res = {"pose": label, "H_m": H.round(6).tolist(), "PH_m": float(np.linalg.norm(Phinge - H))}
    # static parts (top loop, S-bend, hook turns) must not move: vsp_cl == vsp_cl_rest
    stat = np.isin(part, (0, 1, 6))
    res["static_parts_max_move_m"] = float(np.abs(cl[stat] - clr[stat]).max())
    # top loop on A (static, vertical)
    c0 = np.unique(cl[part == 0], axis=0)
    p0 = c0[:, 2]
    res["top_inner_top_err_m"] = float(abs(p0.max() - r_h))
    seg = c0[p0 > p0.max() - 1e-7]
    res["top_segment_lateral_m"] = float(np.linalg.norm(seg.mean(0)[:2]))
    # lower loop on H
    c5 = np.unique(cl[part == 5], axis=0)
    p5 = (c5 - H) @ u
    res["bottom_inner_bottom_err_m"] = float(abs(p5.min() + r_l))
    segb = c5[p5 < p5.min() + 1e-7]
    q = segb.mean(0) - H
    res["bottom_segment_lateral_m"] = float(np.linalg.norm(q - (q @ u) * u))
    # eye inner top at H
    Me = np.array(eye.matrix_world)
    eye_c = Me[:3, 3]
    ez = Me[:3, 2] / np.linalg.norm(Me[:3, 2])
    res["eye_inner_top_err_m"] = float(np.linalg.norm(eye_c + (R_e - r_e) * ez - H))
    # moving part is a rigid rotation about P where the bend weight is 1 (Kabsch on fully rotated points)
    mv = np.isin(part, (3, 4, 5))
    X, Y = clr[mv] - Phinge, cl[mv] - Phinge
    U_, _, Vt = np.linalg.svd(X.T @ Y)
    Rk = (U_ @ np.diag([1, 1, np.sign(np.linalg.det(U_ @ Vt))]) @ Vt).T
    res["moving_rigid_resid_m"] = float(np.abs(X @ Rk.T - Y).max())
    # coils: winding about the rest axis (vertical through P) for hook + open + tight turns, in generation order
    cmask = np.isin(part, (6, 2, 3))
    cr, cw = clr[cmask], cl[cmask]
    key = np.unique(np.c_[cr, cw].round(12), axis=0)
    order = np.argsort(-key[:, 2], kind="stable")
    key = key[order]
    cr, cw = key[:, :3], key[:, 3:]
    ang = np.unwrap(np.arctan2(cr[:, 1] - Phinge[1], cr[:, 0] - Phinge[0]))
    res["coil_turns"] = float(abs(ang[-1] - ang[0]) / (2 * math.pi))
    # wire overlap in the bent configuration: pairs more than 0.75 turn apart (rest-frame angle)
    dmat = np.linalg.norm(cw[:, None, :] - cw[None, :, :], axis=2)
    far = np.abs(ang[:, None] - ang[None, :]) > 1.5 * math.pi
    res["coil_min_centreline_gap_m"] = float(dmat[far].min())
    res["coil_min_gap_over_2rw"] = res["coil_min_centreline_gap_m"] / (2 * r_w)
    # links: lower loop vs eye centreline circle
    th = np.linspace(0, 2 * math.pi, 7201)
    ex_e, ez_e = Me[:3, 0] / np.linalg.norm(Me[:3, 0]), ez
    circ = eye_c + R_e * (np.cos(th)[:, None] * ex_e + np.sin(th)[:, None] * ez_e)
    # densify the loop polyline: segments from every point to its two nearest neighbours, 41 samples each
    dd = np.linalg.norm(c5[:, None, :] - c5[None, :, :], axis=2)
    np.fill_diagonal(dd, np.inf)
    nb = np.argsort(dd, axis=1)[:, :2]
    tt = np.linspace(0, 1, 41)[:, None]
    dense = np.vstack([c5[i] + tt * (c5[j] - c5[i]) for i in range(len(c5)) for j in nb[i]] +
                      [segb[i] + np.linspace(0, 1, 401)[:, None] * (segb[j] - segb[i])
                       for i in range(len(segb)) for j in range(i + 1, len(segb))])
    dl = np.linalg.norm(dense[:, None, :] - circ[None, :, :], axis=2).min()
    res["loop_eye_min_centreline_dist_m"] = float(dl)
    res["loop_eye_dist_minus_contact_m"] = float(dl - (r_l + r_e))
    # clearances
    wt = weight_samples()
    ey_ = evaluated(eye)["co"]
    clear = {}
    for bn, bx in boxes.items():
        sel = co if bn != "bar" else co[part != 0]
        clear[f"spring_in_{bn}"] = int(inside(sel, bx).sum())
        clear[f"weight_in_{bn}"] = int(inside(wt, bx).sum())
        clear[f"eye_in_{bn}"] = int(inside(ey_, bx).sum())
        clear[f"min_dist_weight_{bn}_m"] = float(box_dist(wt, bx).min())
    clear["top_loop_cl_dist_to_bar_minus_rh_m"] = float(box_dist(c0, boxes["bar"]).min() - r_h)
    # spring vertices inside the weight cylinder (weight frame)
    Mw = np.array(weight.matrix_world)
    loc = (co - Mw[:3, 3]) @ Mw[:3, :3]
    rr = np.hypot(loc[:, 0] - c_local.x, loc[:, 1] - c_local.y)
    zz = loc[:, 2] - c_local.z
    clear["spring_in_weight"] = int(((rr < Rw - 1e-5) & (np.abs(zz) < hw / 2 - 1e-5)).sum())
    res["clearances"] = clear
    tol = 1e-6
    ok = (res["static_parts_max_move_m"] < tol and res["moving_rigid_resid_m"] < tol and res["top_inner_top_err_m"] < tol and res["top_segment_lateral_m"] < tol and res["bottom_inner_bottom_err_m"] < tol
          and res["bottom_segment_lateral_m"] < tol and res["eye_inner_top_err_m"] < tol and abs(res["coil_turns"] - N) < 1e-3
          and res["coil_min_gap_over_2rw"] >= 1.0 - 1e-6 and res["loop_eye_dist_minus_contact_m"] > -LINK_TOL
          and clear["top_loop_cl_dist_to_bar_minus_rh_m"] > -1e-5
          and all(v == 0 for k, v in clear.items() if k.startswith(("spring_in", "weight_in", "eye_in"))))
    res["pass"] = bool(ok)
    return res


# detach any baked action in memory (file is never saved)
action = weight.animation_data.action if weight.animation_data else None
results = []
if action:
    weight.animation_data.action = None
poses = []
for L_S in (0.75, D["spring_length_ref_PH"] / S, 0.82):
    for tx in (-12.0, 0.0, 12.0):
        for ty in (-2.0, 0.0, 2.0):
            for wr in (-8.0, 0.0, 8.0):
                poses.append((L_S, tx, ty, wr))   # |P-H| in W_in, tilt about P
for (L_S, tx, ty, wr) in poses:
    L = L_S * S
    # spring direction: tilt tx about Y (in the image plane), ty about X (towards/away from the camera)
    d = Vector((0, 0, -1))
    d.rotate(Euler((math.radians(ty), math.radians(-tx), 0.0), "XYZ"))
    weight.location = Vector(Phinge) + d * L
    # body along the spring axis plus a relative rock wr about the same axes
    weight.rotation_euler = (math.radians(ty + 0.5 * wr), math.radians(-tx + wr), 0.0)
    r = check_pose({"PH_Win": round(L_S, 5), "tilt_x_deg": tx, "tilt_y_deg": ty, "weight_rock_deg": wr})
    results.append(r)
frames = []
if a.frames and action:
    weight.animation_data.action = action
    for f in range(0, 241, 1):
        sc.frame_set(f)
        r = check_pose({"frame": f})
        frames.append(r)
keys = ["static_parts_max_move_m", "moving_rigid_resid_m", "top_inner_top_err_m", "top_segment_lateral_m", "bottom_inner_bottom_err_m", "bottom_segment_lateral_m",
        "eye_inner_top_err_m", "loop_eye_dist_minus_contact_m"]
allr = results + frames
summary = {k: float(max(abs(r[k]) for r in allr)) for k in keys}
summary["coil_turns_min_max"] = [min(r["coil_turns"] for r in allr), max(r["coil_turns"] for r in allr)]
summary["coil_min_gap_over_2rw_min"] = min(r["coil_min_gap_over_2rw"] for r in allr)
summary["loop_eye_dist_minus_contact_min_m"] = min(r["loop_eye_dist_minus_contact_m"] for r in allr)
summary["top_loop_to_bar_min_m"] = min(r["clearances"]["top_loop_cl_dist_to_bar_minus_rh_m"] for r in allr)
summary["min_weight_plate_m"] = min(r["clearances"]["min_dist_weight_plate_m"] for r in allr)
summary["min_weight_posts_m"] = min(min(r["clearances"]["min_dist_weight_post_L_m"], r["clearances"]["min_dist_weight_post_R_m"]) for r in allr)
summary["intersections_total"] = int(sum(v for r in allr for k, v in r["clearances"].items() if k.startswith(("spring_in", "weight_in", "eye_in"))))
summary["static_parts_max_move_m"] = max(r["static_parts_max_move_m"] for r in allr)
summary["moving_rigid_resid_max_m"] = max(r["moving_rigid_resid_m"] for r in allr)
summary["n_poses"], summary["n_frames"] = len(results), len(frames)
summary["failed"] = [r["pose"] for r in allr if not r["pass"]]
out = {"blend": pathlib.Path(bpy.data.filepath).name, "S_m": S, "r_h": r_h, "r_w": r_w, "coil_radius": R_c, "N_turns": N,
       "eye_R": R_e, "eye_r": r_e, "summary": summary, "pass": not summary["failed"], "poses": results, "frames": frames,
       "method": __doc__}
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
print("CHECK_RIG", json.dumps({"pass": out["pass"], **{k: v for k, v in summary.items() if k != "failed"},
                               "n_failed": len(summary["failed"]), "first_failed": summary["failed"][:3]}))
