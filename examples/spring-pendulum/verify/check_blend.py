"""P5, P6, P8 (blend part), P10 dump on a SAVED, reopened .blend, evaluated through the depsgraph (clip 02).

  blender -b <file.blend> --python verify/check_blend.py -- \
      --params params/physics.json --ref verify/out/ref_trajectory.csv [--trajectory solver/out/trajectory.csv] \
      --out verify/out/check_blend.json [--dump verify/out/check_blend.npz] \
      [--weight-name VSP_Weight] [--eye-name VSP_Weight_Eye | --eye-local x,y,z] [--spring-name VSP_Spring] [--anchor-name VSP_Anchor] \
      [--weight-origin com|eye|none] [--orientation rigid|fixed|spring_axis|none] [--spring-axis-local 0,0,1] \
      [--top-end-offset 0] [--bottom-end-offset 0] [--model-json M.json] [--label TEXT]

Sampling: frames 0..240 x subframes 0, .25, .5, .75 (t = (n+s)/24 -> ref row 4(n+s)); samples with t > 10 s are reported,
not judged.  Everything is measured from evaluated geometry / evaluated world matrices, never custom properties.
Spring: end points = centroid of the evaluated spring vertices at the extreme projections on the A->eye axis;
coil count = number of connected wire crossings (mesh edges) through 8 half-planes about that axis.
Runs with Blender's bundled numpy (vparams/vmodel are numpy-only).
"""
import bpy, sys, os, json, math, datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vparams import load_model, sha256_file  # noqa: E402
from vmodel import quat_to_R  # noqa: E402

TOL = dict(P5_anchor=1e-6, P5_static=1e-6, P5_attach=1e-5, P6_pos=1e-5, P6_rad=1e-5, P8_blend=1e-6)
SUBFRAMES = (0.0, 0.25, 0.5, 0.75)
NPLANES = 8


def args():
    import argparse
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True); ap.add_argument("--model-json")
    ap.add_argument("--ref", required=True); ap.add_argument("--trajectory")
    ap.add_argument("--out", required=True); ap.add_argument("--dump")
    ap.add_argument("--weight-name"); ap.add_argument("--eye-name"); ap.add_argument("--eye-local")
    ap.add_argument("--spring-name"); ap.add_argument("--anchor-name")
    ap.add_argument("--weight-origin", default="com", choices=["com", "eye", "none"])
    ap.add_argument("--orientation", default="auto", choices=["auto", "rigid", "fixed", "spring_axis", "none"])
    ap.add_argument("--spring-axis-local", default="0,0,1")
    ap.add_argument("--top-end-offset", type=float, default=0.0); ap.add_argument("--bottom-end-offset", type=float, default=0.0)
    ap.add_argument("--exclude-static", default="", help="comma list of object names not required to be static")
    ap.add_argument("--anchor-world", help="x,y,z expected world position of the anchor object (default: model A)")
    ap.add_argument("--vsp", action="store_true", help="production spring: use GN point attributes vsp_part/vsp_cl/vsp_w")
    ap.add_argument("--vsp-top-part", type=int, default=0); ap.add_argument("--vsp-lower-part", type=int, default=5)
    ap.add_argument("--vsp-coil-parts", default="2,3"); ap.add_argument("--vsp-static-parts", default="0,1,6")
    ap.add_argument("--vsp-first-moving-part", type=int, default=2)
    ap.add_argument("--hook-lower-name", default="VSP_HookLower")
    ap.add_argument("--frames", default="0,240"); ap.add_argument("--label", default="")
    return ap.parse_args(argv)


def read_csv(path):
    with open(path) as f:
        head = f.readline().strip().split(",")
        A = np.array([[float(x) for x in l.split(",")] for l in f if l.strip()])
    return {h: A[:, i] for i, h in enumerate(head)}


def find_obj(explicit, words, types=("MESH", "CURVE", "EMPTY"), kind=""):
    if explicit:
        o = bpy.data.objects.get(explicit)
        if o is None:
            raise SystemExit(f"{kind} object {explicit!r} not found")
        return o, "explicit"
    hits = [o for o in bpy.data.objects if any(w in o.name.lower() for w in words) and o.type in types]
    if not hits:
        return None, "not found"
    hits.sort(key=lambda o: (-len(getattr(o.data, "vertices", []) or []) if o.data is not None and hasattr(o.data, "vertices") else 0, o.name))
    return hits[0], f"name match {words}" + (f"; candidates {[h.name for h in hits]}" if len(hits) > 1 else "")


def descendants(o):
    out = []
    for c in o.children:
        out.append(c); out.extend(descendants(c))
    return out


def eval_mesh(obj, dg):
    oe = obj.evaluated_get(dg)
    me = oe.to_mesh()
    n = len(me.vertices)
    co = np.empty(n * 3, np.float32); me.vertices.foreach_get("co", co)
    ne = len(me.edges)
    ed = np.empty(ne * 2, np.int32); me.edges.foreach_get("vertices", ed)
    nf = len(me.polygons)
    oe.to_mesh_clear()
    M = np.array(oe.matrix_world, dtype=np.float64)
    V = co.reshape(n, 3).astype(np.float64) @ M[:3, :3].T + M[:3, 3]
    return V, ed.reshape(ne, 2), (n, ne, nf), M


def components(nv, edges, mask):
    """number of connected components among vertices in mask using edges with both ends in mask."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return 0
    e = edges[mask[edges[:, 0]] & mask[edges[:, 1]]]
    lab = np.arange(nv)
    while True:
        old = lab.copy()
        m = np.minimum(lab[e[:, 0]], lab[e[:, 1]])
        np.minimum.at(lab, e[:, 0], m); np.minimum.at(lab, e[:, 1], m)
        lab = lab[lab]  # pointer jumping
        if np.array_equal(lab, old):
            break
    return int(len(np.unique(lab[idx])))


def spring_analysis(V, E, A, eye, top_off, bot_off, mask=None):
    u = eye - A; L = np.linalg.norm(u); u = u / L
    p = (V - A) @ u
    span = p.max() - p.min()
    eps = 1e-7 + 1e-5 * span
    top = V[p <= p.min() + eps].mean(0); bot = V[p >= p.max() - eps].mean(0)
    # coil band + half-plane crossings
    e1 = np.array([1.0, 0, 0]) - u[0] * u
    if np.linalg.norm(e1) < 0.3:
        e1 = np.array([0, 1.0, 0]) - u[1] * u
    e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
    r = (V - A) - np.outer(p, u)
    rho = np.linalg.norm(r, axis=1)
    phi = np.arctan2(r @ e2, r @ e1)
    big = rho > 0.25 * rho.max()
    R = float(np.median(rho[big])) if big.any() else 0.0
    band = (rho > 0.5 * R) & (rho < 1.5 * R)
    if mask is not None:
        band &= mask
        R = float(np.median(rho[mask]))
    eb = E[band[E[:, 0]] & band[E[:, 1]]]
    dphi = np.abs(np.angle(np.exp(1j * (phi[eb[:, 0]] - phi[eb[:, 1]])))) if len(eb) else np.array([0.1])
    half = 1.5 * float(np.max(dphi)) + 1e-3
    counts = []
    for k in range(NPLANES):
        ang = 2 * math.pi * k / NPLANES
        dd = np.abs(np.angle(np.exp(1j * (phi - ang))))
        counts.append(components(len(V), E, band & (dd < half)))
    return dict(top=top, bot=bot, gap_top=float(np.linalg.norm(top - (A + top_off * u))),
                gap_bot=float(np.linalg.norm(bot - (eye + bot_off * u))), coil_counts=counts, coil_radius=R,
                slab_half_angle=half, axis=u, length=L)


def attr(me, name, kind):
    a = me.attributes[name]; n = len(a.data)
    if kind == "vec":
        arr = np.empty(n * 3, np.float32); a.data.foreach_get("vector", arr); return arr.reshape(n, 3).astype(np.float64)
    arr = np.empty(n, np.float32 if kind == "float" else np.int32); a.data.foreach_get("value", arr); return arr


def eval_spring_vsp(obj, dg):
    oe = obj.evaluated_get(dg); me = oe.to_mesh()
    n = len(me.vertices)
    co = np.empty(n * 3, np.float32); me.vertices.foreach_get("co", co)
    ed = np.empty(len(me.edges) * 2, np.int32); me.edges.foreach_get("vertices", ed)
    part = attr(me, "vsp_part", "int"); cl = attr(me, "vsp_cl", "vec"); w = attr(me, "vsp_w", "float")
    topo = (n, len(me.edges), len(me.polygons))
    oe.to_mesh_clear()
    M = np.array(oe.matrix_world, dtype=np.float64)
    tr = lambda X: X @ M[:3, :3].T + M[:3, 3]
    return tr(co.reshape(n, 3).astype(np.float64)), ed.reshape(-1, 2), part, tr(cl), w, topo


def unique_seq(P):
    keep = np.r_[True, np.any(P[1:] != P[:-1], axis=1)]
    return P[keep]


def dist_to_centreline(cl_pts, target):
    """distance from target to the centreline polyline (index order). Straight GN segments carry only end rings, so a
    polyline is exact there; on arcs the chord sagitta is reported via the local spacing."""
    a, b = cl_pts[:-1], cl_pts[1:]; ab = b - a
    t = np.clip(np.einsum("ij,ij->i", target - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300), 0, 1)
    q = a + ab * t[:, None]; d = np.linalg.norm(q - target, axis=1); i = int(np.argmin(d))
    seg = np.linalg.norm(ab, axis=1); s_along = float(np.sum(seg[:i]) + t[i] * seg[i])   # arclength of the contact point
    return float(d[i]), s_along, i


def loop_frame_coords(cl_pts, target, ref_axes=None):
    """target expressed in the loop's own frame (centroid + PCA axes, signs aligned to ref_axes)."""
    c = cl_pts.mean(0); _, _, Vt = np.linalg.svd(cl_pts - c)
    if ref_axes is not None:
        Vt = Vt * np.sign(np.sum(Vt * ref_axes, axis=1))[:, None]
    return (target - c) @ Vt.T, Vt


def winding_turns(cl_pts, A, H):
    u = H - A; u = u / np.linalg.norm(u)
    e1 = np.array([1.0, 0, 0]) - u[0] * u; e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
    r = cl_pts - A; ph = np.unwrap(np.arctan2(r @ e2, r @ e1))
    return float(abs(ph[-1] - ph[0]) / (2 * math.pi))


def fcurves_of(obj):
    ad = obj.animation_data
    info = dict(has_animation_data=ad is not None)
    arrays = {}
    if ad is None:
        return info, arrays
    info["nla_strips"] = [dict(name=s.name, scale=s.scale, repeat=s.repeat) for t in ad.nla_tracks for s in t.strips]
    info["drivers"] = [(d.data_path, d.array_index) for d in ad.drivers]
    act = ad.action
    info["action"] = act.name if act else None
    fcs = []
    if act:
        try:
            from bpy_extras import anim_utils
            cb = anim_utils.action_get_channelbag_for_slot(act, ad.action_slot)
            fcs = list(cb.fcurves) if cb else []
        except Exception as ex:
            fcs = list(getattr(act, "fcurves", [])); info["fcurve_api_note"] = str(ex)
    info["fcurves"] = []
    for fc in fcs:
        kp = fc.keyframe_points; n = len(kp)
        co = np.empty(2 * n, np.float32); kp.foreach_get("co", co)
        hl = np.empty(2 * n, np.float32); kp.foreach_get("handle_left", hl)
        hr = np.empty(2 * n, np.float32); kp.foreach_get("handle_right", hr)
        key = f"{obj.name}|{fc.data_path}[{fc.array_index}]"
        arrays[key] = dict(co=co.reshape(n, 2), hl=hl.reshape(n, 2), hr=hr.reshape(n, 2),
                           interp=np.array([k.interpolation for k in kp]), htypes=np.array([(k.handle_left_type, k.handle_right_type) for k in kp]))
        info["fcurves"].append(dict(channel=key, n_keys=n, interpolations=sorted(set(k.interpolation for k in kp)),
                                    extrapolation=fc.extrapolation, modifiers=[m.type for m in fc.modifiers], mute=fc.mute))
    return info, arrays


def main():
    a = args()
    sc = bpy.context.scene
    blend = bpy.data.filepath
    model, _ = load_model(a.params, a.model_json)
    ref = read_csv(a.ref)
    traj = read_csv(a.trajectory) if a.trajectory else None
    rep = dict(label=a.label, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               blender=bpy.app.version_string, inputs={blend: sha256_file(blend), a.params: sha256_file(a.params), a.ref: sha256_file(a.ref)},
               verifier_code_sha256=sha256_file(os.path.abspath(__file__)), tolerances=TOL, blend_is_dirty_after_load=bool(bpy.data.is_dirty),
               model=model.as_dict())
    if a.trajectory:
        rep["inputs"][a.trajectory] = sha256_file(a.trajectory)
    scene = dict(fps=sc.render.fps, fps_base=sc.render.fps_base, frame_start=sc.frame_start, frame_end=sc.frame_end,
                 frame_map_old=sc.render.frame_map_old, frame_map_new=sc.render.frame_map_new, unit_scale=sc.unit_settings.scale_length)
    scene["ok"] = (scene["fps"] == 24 and scene["fps_base"] == 1.0 and scene["frame_start"] == 0 and scene["frame_end"] == 240
                   and scene["frame_map_old"] == scene["frame_map_new"] and scene["unit_scale"] == 1.0)
    rep["scene"] = scene
    weight, how_w = find_obj(a.weight_name, ["weight", "bob", "mass"], ("MESH", "EMPTY", "CURVE"), "weight")
    spring, how_s = find_obj(a.spring_name, ["spring", "coil", "helix"], ("MESH", "CURVE"), "spring")
    anchor, how_a = find_obj(a.anchor_name, ["anchor"], ("MESH", "EMPTY", "CURVE"), "anchor")
    eye = None; how_e = "eye-local" if a.eye_local else "not found"
    if not a.eye_local:
        eye, how_e = find_obj(a.eye_name, ["eye", "hook_point", "attach"], ("EMPTY", "MESH"), "eye")
    rep["objects"] = dict(weight=[weight.name if weight else None, how_w], spring=[spring.name if spring else None, how_s],
                          anchor=[anchor.name if anchor else None, how_a], eye=[eye.name if eye else a.eye_local, how_e])
    if weight is None or spring is None or (eye is None and not a.eye_local):
        rep["fatal"] = "weight/spring/eye not found"; json.dump(rep, open(a.out, "w"), indent=2); print("CHECK_BLEND_SUMMARY", "fatal"); return
    eye_local = np.array([float(x) for x in a.eye_local.split(",")]) if a.eye_local else None
    orient = a.orientation if a.orientation != "auto" else ("rigid" if model.body == "rigid" else "none")
    rep["orientation_rule"] = orient
    moving = set([weight.name, spring.name] + [o.name for o in descendants(weight)] + [o.name for o in descendants(spring)])
    moving |= set(x for x in a.exclude_static.split(",") if x)
    statics = [o for o in bpy.data.objects if o.name not in moving and o.type not in ("CAMERA",)]
    rep["static_objects_checked"] = [o.name for o in statics]
    rep["animated_objects"] = [o.name for o in bpy.data.objects if o.animation_data and (o.animation_data.action or len(o.animation_data.drivers) or len(o.animation_data.nla_tracks))]
    rep["spring_modifiers_constraints"] = dict(modifiers=[(m.name, m.type) for m in spring.modifiers], constraints=[(c.name, c.type) for c in spring.constraints])
    f0, f1 = (int(x) for x in a.frames.split(","))
    S = []
    static0 = None; vsp0 = None
    anchor_world = np.array([float(x) for x in a.anchor_world.split(",")]) if a.anchor_world else model.A
    hook = bpy.data.objects.get(a.hook_lower_name) if a.vsp else None
    for n in range(f0, f1 + 1):
        for s in SUBFRAMES:
            sc.frame_set(n, subframe=s)
            dg = bpy.context.evaluated_depsgraph_get()
            Mw = np.array(weight.evaluated_get(dg).matrix_world, dtype=np.float64)
            Ew = (Mw @ np.r_[eye_local, 1.0])[:3] if eye_local is not None else np.array(eye.evaluated_get(dg).matrix_world, dtype=np.float64)[:3, 3]
            if a.vsp:
                Vs, Es, part, cl, vw, topo = eval_spring_vsp(spring, dg)
                coil_mask = np.isin(part, [int(x) for x in a.vsp_coil_parts.split(",")])
                sp = spring_analysis(Vs, Es, model.A, Ew, a.top_end_offset, a.bottom_end_offset, mask=coil_mask)
                rw_top = float(np.median(np.linalg.norm(Vs[part == a.vsp_top_part] - cl[part == a.vsp_top_part], axis=1)))
                rw_low = float(np.median(np.linalg.norm(Vs[part == a.vsp_lower_part] - cl[part == a.vsp_lower_part], axis=1)))
                dt_, st_, _ = dist_to_centreline(unique_seq(cl[part == a.vsp_top_part]), anchor_world)
                ulow = unique_seq(cl[part == a.vsp_lower_part])
                db_, sb_, _ = dist_to_centreline(ulow, Ew)
                hloc, axes = loop_frame_coords(ulow, Ew, None if vsp0 is None else vsp0["axes"])
                stv = Vs[np.isin(part, [int(x) for x in a.vsp_static_parts.split(",")])]
                w0v = Vs[vw == 0.0]
                clm = unique_seq(cl[part == a.vsp_first_moving_part])
                dP_, _, _ = dist_to_centreline(clm, model.A)
                if vsp0 is None:
                    vsp0 = dict(static=stv.copy(), w0=w0v.copy(), first=clm[0].copy(), axes=axes, hloc=hloc)
                sp.update(gap_top=abs(dt_ - rw_top), gap_bot=abs(db_ - rw_low), contact_top_signed=dt_ - rw_top, contact_bot_signed=db_ - rw_low,
                          rw_top=rw_top, rw_low=rw_low, s_contact_top=st_, s_contact_low=sb_, h_in_loop_drift=float(np.linalg.norm(hloc - vsp0["hloc"])), h_out_of_loop_plane=float(abs(hloc[2])),
                          loop_planarity=float(np.max(np.abs((ulow - ulow.mean(0)) @ axes[2]))),
                          static_spring_drift=float(np.max(np.linalg.norm(stv - vsp0["static"], axis=1))) if stv.shape == vsp0["static"].shape else float("inf"),
                          w0_vertex_motion=float(np.max(np.linalg.norm(w0v - vsp0["w0"], axis=1))) if w0v.shape == vsp0["w0"].shape else float("inf"),
                          moving_centreline_to_P=dP_, moving_first_cl_drift=float(np.linalg.norm(clm[0] - vsp0["first"])),
                          winding=winding_turns(unique_seq(cl[coil_mask]), model.A, Ew))
                if hook is not None:
                    sp["hook_vs_weight_origin"] = float(np.linalg.norm(np.array(hook.evaluated_get(dg).matrix_world, dtype=np.float64)[:3, 3] - Mw[:3, 3]))
            else:
                Vs, Es, topo, _ = eval_mesh(spring, dg)
                sp = spring_analysis(Vs, Es, model.A, Ew, a.top_end_offset, a.bottom_end_offset)
            stat = np.array([np.array(o.evaluated_get(dg).matrix_world, dtype=np.float64) for o in statics]) if statics else np.zeros((0, 4, 4))
            if static0 is None:
                static0 = stat
            anc = np.array(anchor.evaluated_get(dg).matrix_world, dtype=np.float64)[:3, 3] if anchor is not None else None
            S.append(dict(n=n, s=s, Mw=Mw, E=Ew, sp=sp, topo=topo, static_drift=float(np.max(np.abs(stat[:, :3, 3] - static0[:, :3, 3]))) if len(stat) else 0.0,
                          static_rot_drift=float(np.max(np.abs(stat[:, :3, :3] - static0[:, :3, :3]))) if len(stat) else 0.0, anchor=anc))
    tt = np.array([(x["n"] + x["s"]) / 24.0 for x in S]); idx = np.array([int(round(4 * (x["n"] + x["s"]))) for x in S])
    inclip = tt <= 10.0 + 1e-12
    Eb = np.array([x["E"] for x in S]); Mw = np.array([x["Mw"] for x in S])
    Rb = Mw[:, :3, :3] / np.linalg.norm(Mw[:, :3, :3], axis=1, keepdims=True)
    tops = np.array([x["sp"]["top"] for x in S]); bots = np.array([x["sp"]["bot"] for x in S])
    gtop = np.array([x["sp"]["gap_top"] for x in S]); gbot = np.array([x["sp"]["gap_bot"] for x in S])
    coils = np.array([x["sp"]["coil_counts"] for x in S]); topo = np.array([x["topo"] for x in S])
    # ---------------- P5
    p5 = dict(n_samples=len(S), n_judged=int(inclip.sum()))
    if anchor is not None:
        AN = np.array([x["anchor"] for x in S])
        p5["anchor_world_vs_expected_max_m"] = float(np.max(np.linalg.norm(AN - anchor_world, axis=1))); p5["anchor_expected"] = anchor_world.tolist()
        p5["anchor_drift_from_frame0_max_m"] = float(np.max(np.linalg.norm(AN - AN[0], axis=1)))
    else:
        p5["anchor"] = "no anchor object; anchor judged via spring top end only"
    p5["static_objects_max_translation_drift_m"] = max(x["static_drift"] for x in S)
    p5["static_objects_max_matrix_drift"] = max(x["static_rot_drift"] for x in S)
    p5["spring_top_end_gap_max_m"] = float(gtop[inclip].max()); p5["spring_bottom_end_gap_max_m"] = float(gbot[inclip].max())
    p5["spring_top_end_world_variation_m"] = float(np.max(np.linalg.norm(tops - tops[0], axis=1)[inclip]))
    p5["top_end_offset_m"] = a.top_end_offset; p5["bottom_end_offset_m"] = a.bottom_end_offset
    p5["coil_counts_frame0"] = coils[0].tolist()
    p5["coil_counts_constant"] = bool(np.all(coils[inclip] == coils[0]))
    p5["coil_count_turns_estimate"] = float(np.mean(coils[0]))
    bad = np.where(np.any(coils != coils[0], axis=1))[0]
    p5["coil_count_first_change_t"] = float(tt[bad[0]]) if len(bad) else None
    p5["topology_constant"] = bool(np.all(topo == topo[0])); p5["topology_frame0_v_e_f"] = topo[0].tolist()
    p5["coil_radius_m_min_max"] = [float(min(x["sp"]["coil_radius"] for x in S)), float(max(x["sp"]["coil_radius"] for x in S))]
    p5["spring_length_min_max_m"] = [float(min(x["sp"]["length"] for x in S)), float(max(x["sp"]["length"] for x in S))]
    ok5 = (p5.get("anchor_world_vs_expected_max_m", 0) < TOL["P5_anchor"] and p5.get("anchor_drift_from_frame0_max_m", 0) < TOL["P5_anchor"]
           and p5["static_objects_max_translation_drift_m"] < TOL["P5_static"] and p5["spring_top_end_gap_max_m"] < TOL["P5_attach"]
           and p5["spring_bottom_end_gap_max_m"] < TOL["P5_attach"] and p5["coil_counts_constant"] and p5["topology_constant"])
    if a.vsp:
        jj = np.where(inclip)[0]
        g = lambda key: np.array([S[i]["sp"][key] for i in jj])
        p5["method_spring_ends"] = ("GN point attributes: contact gap = |distance(target, loop centreline polyline vsp_cl)"
                                    " - wire radius|; top loop (part %d) vs anchor %s, lower loop (part %d) vs hinge H" % (a.vsp_top_part, anchor_world.tolist(), a.vsp_lower_part))
        p5["top_loop_contact_gap_max_m"] = float(g("gap_top").max()); p5["lower_loop_contact_gap_max_m"] = float(g("gap_bot").max())
        p5["lower_loop_contact_signed_min_max_m"] = [float(g("contact_bot_signed").min()), float(g("contact_bot_signed").max())]
        p5["top_loop_contact_signed"] = float(g("contact_top_signed")[0])
        p5["lower_contact_sliding_along_loop_m"] = float(np.ptp(g("s_contact_low")))
        p5["hinge_in_lower_loop_frame_drift_max_m"] = float(g("h_in_loop_drift").max())
        p5["hinge_out_of_lower_loop_plane_max_m"] = float(g("h_out_of_loop_plane").max())
        p5["info_lower_loop_centreline_planarity_m"] = float(g("loop_planarity").max()); p5["top_contact_sliding_along_loop_m"] = float(np.ptp(g("s_contact_top")))
        p5["wire_radius_top_lower_m"] = [S[0]["sp"]["rw_top"], S[0]["sp"]["rw_low"]]
        p5["static_spring_parts"] = a.vsp_static_parts
        p5["static_spring_vertices_max_drift_m"] = float(g("static_spring_drift").max())
        p5["info_unrotated_vsp_w0_vertices_max_motion_m"] = float(g("w0_vertex_motion").max())
        p5["info_note_w0"] = "vsp_w == 0 marks 'not rotated', not 'static': coil vertices below P translate with the per-turn extension"
        dp = g("moving_centreline_to_P")
        p5["moving_coil_centreline_to_P_eff_min_max_m"] = [float(dp.min()), float(dp.max())]
        p5["moving_coil_centreline_to_P_eff_variation_m"] = float(dp.max() - dp.min())
        p5["note_P_eff"] = "P_eff is the coil axis point of the top coil, so the centreline stays one coil radius from it; judged: variation (no sliding)"
        p5["moving_first_centreline_point_drift_max_m"] = float(g("moving_first_cl_drift").max())
        wt = g("winding"); p5["coil_winding_turns_min_max"] = [float(wt.min()), float(wt.max())]
        p5["coil_winding_variation_turns"] = float(wt.max() - wt.min())
        if hook is not None:
            p5["hook_lower_vs_weight_origin_max_m"] = float(g("hook_vs_weight_origin").max())
        p5["note_generic_end_centroids"] = "spring_top/bottom_end_gap_* use extreme-projection centroids (generic test-rig rule); not judged in --vsp mode"
        ok5 = (p5.get("anchor_world_vs_expected_max_m", 0) < TOL["P5_anchor"] and p5.get("anchor_drift_from_frame0_max_m", 0) < TOL["P5_anchor"]
               and p5["static_objects_max_translation_drift_m"] < TOL["P5_static"] and p5["static_spring_vertices_max_drift_m"] < TOL["P5_static"]
               and p5["top_loop_contact_gap_max_m"] < TOL["P5_attach"] and p5["lower_loop_contact_gap_max_m"] < TOL["P5_attach"]
               and p5["moving_coil_centreline_to_P_eff_variation_m"] < TOL["P5_attach"]
               and p5["lower_contact_sliding_along_loop_m"] < TOL["P5_attach"] and p5["hinge_in_lower_loop_frame_drift_max_m"] < TOL["P5_attach"]
               and p5["hinge_out_of_lower_loop_plane_max_m"] < TOL["P5_attach"] and p5["top_contact_sliding_along_loop_m"] < TOL["P5_attach"] and p5["moving_first_centreline_point_drift_max_m"] < TOL["P5_attach"]
               and p5["coil_counts_constant"] and p5["topology_constant"] and p5["coil_winding_variation_turns"] < 1e-3
               and p5.get("hook_lower_vs_weight_origin_max_m", 0) < TOL["P5_attach"])
    p5["status"] = "PASS" if ok5 else "FAIL"
    rep["P5_constraints_saved_blend"] = p5
    # ---------------- P6
    Eref = np.stack([ref["x_m"], ref["y_m"], ref["z_m"]], -1)[idx]
    Cref = np.stack([ref["xc_m"], ref["yc_m"], ref["zc_m"]], -1)[idx]
    qref = np.stack([ref["qw"], ref["qx"], ref["qy"], ref["qz"]], -1)[idx]
    Rref = quat_to_R(qref)
    dE = np.linalg.norm(Eb - Eref, axis=1)
    p6 = dict(max_abs_t_ref_mismatch=float(np.max(np.abs(ref["t_s"][idx] - tt))),
              eye_vs_ref_max_m=float(dE[inclip].max()), eye_vs_ref_t_at_max=float(tt[inclip][np.argmax(dE[inclip])]),
              eye_vs_ref_on_frames_max_m=float(dE[inclip & (np.array([x["s"] for x in S]) == 0)].max()),
              info_after_clip_end_max_m=float(dE[~inclip].max()) if (~inclip).any() else None)
    ok6 = p6["eye_vs_ref_max_m"] < TOL["P6_pos"] and p6["max_abs_t_ref_mismatch"] < 1e-12 and scene["ok"]
    if a.weight_origin != "none":
        tgt = Cref if a.weight_origin == "com" else Eref
        dO = np.linalg.norm(Mw[:, :3, 3] - tgt, axis=1)
        p6[f"weight_origin_vs_ref_{a.weight_origin}_max_m"] = float(dO[inclip].max())
        ok6 = ok6 and p6[f"weight_origin_vs_ref_{a.weight_origin}_max_m"] < TOL["P6_pos"]
    def ang(Ra, Rb_):
        D = np.einsum("nji,njk->nik", Ra, Rb_)
        return np.arccos(np.clip((np.trace(D, axis1=1, axis2=2) - 1) / 2, -1, 1))
    if orient == "rigid":
        Rbind = Rref[0].T @ Rb[0]
        p6["weight_rotation_vs_ref_max_rad"] = float(ang(Rref @ Rbind, Rb)[inclip].max())
        p6["weight_rotation_frame0_vs_q0_rad"] = float(ang(quat_to_R(model.q0)[None], Rb[:1] @ Rbind.T[None])[0])
        ok6 = ok6 and p6["weight_rotation_vs_ref_max_rad"] < TOL["P6_rad"]
    elif orient == "fixed":
        p6["weight_rotation_variation_max_rad"] = float(ang(np.repeat(Rb[:1], len(Rb), 0), Rb)[inclip].max())
        ok6 = ok6 and p6["weight_rotation_variation_max_rad"] < TOL["P6_rad"]
    elif orient == "spring_axis":
        al = np.array([float(x) for x in a.spring_axis_local.split(",")]); al /= np.linalg.norm(al)
        w_ax = Rb @ al
        u = (model.A - Eref); u /= np.linalg.norm(u, axis=1, keepdims=True)
        p6["weight_axis_vs_spring_axis_max_rad"] = float(np.max(np.arccos(np.clip(np.sum(w_ax * u, 1), -1, 1))[inclip]))
        ok6 = ok6 and p6["weight_axis_vs_spring_axis_max_rad"] < TOL["P6_rad"]
    if traj is not None and all(c in traj for c in ("x_m", "y_m", "z_m")):
        tr = traj["t_s"]; it = np.round(tt * 96).astype(int)
        ok_t = (it < len(tr))
        ok_t[ok_t] &= np.abs(tr[it[ok_t]] - tt[ok_t]) < 1e-9
        if ok_t.any():
            Et = np.stack([traj["x_m"], traj["y_m"], traj["z_m"]], -1)[it[ok_t]]
            p6["eye_vs_trajectory_max_m"] = float(np.linalg.norm(Eb[ok_t] - Et, axis=1).max()); p6["trajectory_samples"] = int(ok_t.sum())
            ok6 = ok6 and p6["eye_vs_trajectory_max_m"] < TOL["P6_pos"]
    p6["status"] = "PASS" if ok6 else "FAIL"
    rep["P6_blend_vs_solver"] = p6
    # ---------------- keyframe audit + P8 blend
    farr = {}
    rep["animation"] = {}
    for o in bpy.data.objects:
        info, arr = fcurves_of(o)
        if info.get("has_animation_data"):
            rep["animation"][o.name] = info; farr.update(arr)
    audit = {}
    for k, arr in farr.items():
        co = arr["co"].astype(np.float64)
        ent = dict(n_keys=len(co), integer_frames_0_240=bool(len(co) == 241 and np.all(co[:, 0] == np.arange(241))))
        if k.startswith(weight.name + "|location[") and len(co):
            j = int(k[-2]); ii = np.round(4 * co[:, 0]).astype(int); okk = ii < len(ref["t_s"])
            tgt = (Cref if a.weight_origin == "com" else Eref)
            col = ["xc_m", "yc_m", "zc_m"][j] if a.weight_origin == "com" else ["x_m", "y_m", "z_m"][j]
            ent["max_key_vs_ref_m"] = float(np.max(np.abs(co[okk, 1] - ref[col][ii[okk]])))
            dx = arr["hr"][:, 0].astype(np.float64) - co[:, 0]
            slope = np.where(dx != 0, (arr["hr"][:, 1] - co[:, 1]) / np.where(dx != 0, dx, 1), np.nan)
            ent["handle_x_offsets_frames"] = [float(np.min(dx)), float(np.max(dx))]
            ent["info_handle_slope_per_frame_range"] = [float(np.nanmin(slope)), float(np.nanmax(slope))]
        audit[k] = ent
    rep["keyframe_audit"] = audit
    rep["motion_is_baked_keyframes"] = any(e["n_keys"] > 0 for k, e in audit.items() if k.startswith(weight.name + "|"))
    p8 = dict(scene=scene, eye_frame0_vs_E0_param_m=float(np.linalg.norm(Eb[0] - model.E0)))
    p8["status"] = "PASS" if (scene["ok"] and p8["eye_frame0_vs_E0_param_m"] < TOL["P8_blend"]) else "FAIL"
    rep["P8_blend_part"] = p8
    rep["summary"] = dict(P5=p5["status"], P6=p6["status"], P8_blend=p8["status"], motion_is_baked_keyframes=rep["motion_is_baked_keyframes"])
    if a.dump:
        flat = {}
        for k, arr in farr.items():
            for f in ("co", "hl", "hr", "interp", "htypes"):
                flat[f"{k}|{f}"] = arr[f]
        np.savez(a.dump, winding=np.array([x["sp"].get("winding", np.nan) for x in S]), t=tt, in_clip=inclip, eye=Eb, weight_matrix=Mw, spring_top=tops, spring_bottom=bots, coil_counts=coils, topology=topo,
                 eye_ref=Eref, **flat)
        rep["dump"] = a.dump
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("CHECK_BLEND_SUMMARY", json.dumps(rep["summary"]))


main()
