"""P5, P6, P8 (blend part), P10 dump on a SAVED, reopened .blend, evaluated through the depsgraph (clip 03 torsion disc).

  blender -b <file.blend> --python verify/check_blend.py -- \
      --params params/physics.json [--model-json M.json] --ref verify/out/ref_ck001.csv [--trajectory solver/out/trajectory.csv] \
      --out verify/out/check_blend_ck001.json [--dump verify/out/check_blend_ck001.npz] \
      [--disc-name TOR_Disc] [--wire-name TOR_Wire] [--anchor-name TOR_Clamp] [--mark-name TOR_Mark] \
      [--com-local 0,0,0] [--attach-local x,y,z] [--axes-map I|R9] [--absolute-axes] [--top-end-offset 0] [--bottom-end-offset 0] \
      [--exclude-static a,b] [--frames 0,240] [--label TEXT]

Sampling: frames 0..240 x subframes 0, .25, .5, .75 (t = (n+s)/24 -> ref row 4(n+s)); samples with t > 10 s are reported, not
judged. Everything is measured from evaluated geometry / evaluated world matrices, never custom properties.
Disc pose: world matrix of the disc object (scale removed); C = M com_local; P_attach = M attach_local (default p_b + com_local);
body axes = R_obj @ axes_map.  Wire ends: centroid of evaluated wire vertices at the extreme projections on the A->P axis.
Mark rigidity: evaluated mark vertices expressed in the disc local frame must be constant (and the disc mesh itself).
Runs with Blender's bundled numpy (tmodel/tparams are numpy-only).
"""
import bpy, sys, os, json, math, datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tparams import load_model, sha256_file  # noqa: E402
from tmodel import quat_to_R, rot_angle, twist_of  # noqa: E402

TOL = dict(P5_anchor=1e-6, P5_static=1e-6, P5_attach=1e-5, P5_mark=1e-6, P6_pos=1e-5, P6_rad=1e-5, P8_blend=1e-6)
SUBFRAMES = (0.0, 0.25, 0.5, 0.75)


def args():
    import argparse
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True); ap.add_argument("--model-json")
    ap.add_argument("--ref", required=True); ap.add_argument("--trajectory")
    ap.add_argument("--out", required=True); ap.add_argument("--dump")
    ap.add_argument("--disc-name"); ap.add_argument("--wire-name"); ap.add_argument("--anchor-name"); ap.add_argument("--mark-name")
    ap.add_argument("--com-local", default="0,0,0"); ap.add_argument("--attach-local")
    ap.add_argument("--axes-map", default="I", help="'I' or 9 comma values (row-major) R_map: body axes = R_obj @ R_map")
    ap.add_argument("--absolute-axes", action="store_true", help="judge absolute attitude (bind = I) instead of frame-0 bind")
    ap.add_argument("--top-end-offset", type=float, default=0.0); ap.add_argument("--bottom-end-offset", type=float, default=0.0)
    ap.add_argument("--exclude-static", default=""); ap.add_argument("--anchor-world")
    ap.add_argument("--frames", default="0,240"); ap.add_argument("--label", default="")
    return ap.parse_args(argv)


def read_csv(path):
    with open(path) as f:
        head = f.readline().strip().split(",")
        A = np.array([[float(x) for x in l.split(",")] for l in f if l.strip()])
    return {h: A[:, i] for i, h in enumerate(head)}


def find_obj(explicit, words, types=("MESH", "CURVE", "EMPTY"), kind="", exclude=()):
    if explicit:
        o = bpy.data.objects.get(explicit)
        if o is None:
            raise SystemExit(f"{kind} object {explicit!r} not found")
        return o, "explicit"
    hits = [o for o in bpy.data.objects if any(w in o.name.lower() for w in words) and o.type in types and o.name not in exclude]
    if not hits:
        return None, "not found"
    hits.sort(key=lambda o: (len(o.name), o.name))
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
    topo = (n, len(me.edges), len(me.polygons))
    oe.to_mesh_clear()
    M = np.array(oe.matrix_world, dtype=np.float64)
    V = co.reshape(n, 3).astype(np.float64) @ M[:3, :3].T + M[:3, 3]
    return V, topo, M


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
        fr = co.reshape(n, 2)[:, 0].astype(np.float64)
        info["fcurves"].append(dict(channel=key, n_keys=n, interpolations=sorted(set(k.interpolation for k in kp)),
                                    key_spacing_frames=sorted(set(np.round(np.diff(fr), 6).tolist()))[:5] if n > 1 else [],
                                    extrapolation=fc.extrapolation, modifiers=[m.type for m in fc.modifiers], mute=fc.mute))
    return info, arrays


def main():
    a = args()
    sc = bpy.context.scene
    blend = bpy.data.filepath
    model, _ = load_model(a.params, a.model_json)
    ref = read_csv(a.ref)
    traj = read_csv(a.trajectory) if a.trajectory else None
    com_local = np.array([float(x) for x in a.com_local.split(",")])
    attach_local = np.array([float(x) for x in a.attach_local.split(",")]) if a.attach_local else com_local + model.p_b
    Rmap = np.eye(3) if a.axes_map == "I" else np.array([float(x) for x in a.axes_map.split(",")]).reshape(3, 3)
    rep = dict(label=a.label, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               blender=bpy.app.version_string, inputs={blend: sha256_file(blend), a.params: sha256_file(a.params), a.ref: sha256_file(a.ref)},
               verifier_code_sha256=sha256_file(os.path.abspath(__file__)), tolerances=TOL, blend_is_dirty_after_load=bool(bpy.data.is_dirty),
               model=model.as_dict(), com_local=com_local.tolist(), attach_local=attach_local.tolist(), axes_map=Rmap.tolist())
    if a.trajectory:
        rep["inputs"][a.trajectory] = sha256_file(a.trajectory)
    scene = dict(fps=sc.render.fps, fps_base=sc.render.fps_base, frame_start=sc.frame_start, frame_end=sc.frame_end,
                 frame_map_old=sc.render.frame_map_old, frame_map_new=sc.render.frame_map_new, unit_scale=sc.unit_settings.scale_length)
    scene["ok"] = (scene["fps"] == 24 and scene["fps_base"] == 1.0 and scene["frame_start"] == 0 and scene["frame_end"] == 240
                   and scene["frame_map_old"] == scene["frame_map_new"] and scene["unit_scale"] == 1.0)
    rep["scene"] = scene
    disc, how_d = find_obj(a.disc_name, ["disc"], ("MESH", "EMPTY"), "disc")
    wire, how_w = find_obj(a.wire_name, ["wire"], ("MESH", "CURVE"), "wire")
    anchor, how_a = find_obj(a.anchor_name, ["clamp", "anchor"], ("MESH", "EMPTY", "CURVE"), "anchor")
    mark, how_m = find_obj(a.mark_name, ["mark", "index", "line"], ("MESH", "CURVE"), "mark", exclude=(disc.name if disc else "",))
    rep["objects"] = dict(disc=[disc.name if disc else None, how_d], wire=[wire.name if wire else None, how_w],
                          anchor=[anchor.name if anchor else None, how_a], mark=[mark.name if mark else None, how_m])
    if disc is None or wire is None:
        rep["fatal"] = "disc/wire not found"; json.dump(rep, open(a.out, "w"), indent=2); print("CHECK_BLEND_SUMMARY", "fatal"); return
    moving = set([disc.name, wire.name] + [o.name for o in descendants(disc)] + [o.name for o in descendants(wire)])
    if mark is not None:
        moving |= {mark.name} | {o.name for o in descendants(mark)}
    moving |= set(x for x in a.exclude_static.split(",") if x)
    statics = [o for o in bpy.data.objects if o.name not in moving and o.type not in ("CAMERA",)]
    rep["static_objects_checked"] = [o.name for o in statics]
    rep["animated_objects"] = [o.name for o in bpy.data.objects if o.animation_data and (o.animation_data.action or len(o.animation_data.drivers) or len(o.animation_data.nla_tracks))]
    rep["wire_modifiers_constraints"] = dict(modifiers=[(m.name, m.type) for m in wire.modifiers], constraints=[(c.name, c.type) for c in wire.constraints])
    rep["mark_relation"] = None if mark is None else dict(parent=mark.parent.name if mark.parent else None, parent_type=mark.parent_type)
    anchor_world = np.array([float(x) for x in a.anchor_world.split(",")]) if a.anchor_world else model.A
    f0, f1 = (int(x) for x in a.frames.split(","))
    S = []
    static0 = None; mark0 = None; disc0 = None
    for n in range(f0, f1 + 1):
        for s in SUBFRAMES:
            sc.frame_set(n, subframe=s)
            dg = bpy.context.evaluated_depsgraph_get()
            Md = np.array(disc.evaluated_get(dg).matrix_world, dtype=np.float64)
            sca = np.linalg.norm(Md[:3, :3], axis=0)
            Rd = (Md[:3, :3] / sca) @ Rmap
            C = (Md @ np.r_[com_local, 1.0])[:3]; Pb = (Md @ np.r_[attach_local, 1.0])[:3]
            Vw, topo_w, _ = eval_mesh(wire, dg)
            ax = Pb - anchor_world; L = np.linalg.norm(ax); ax = ax / L
            p = (Vw - anchor_world) @ ax
            span = p.max() - p.min(); eps = 1e-7 + 1e-5 * span
            top = Vw[p <= p.min() + eps].mean(0); bot = Vw[p >= p.max() - eps].mean(0)
            perp = np.linalg.norm((Vw - anchor_world) - np.outer(p, ax), axis=1)
            Minv = np.linalg.inv(Md)
            if mark is not None:
                Vm, topo_m, Mm = eval_mesh(mark, dg)
                mloc = Vm @ Minv[:3, :3].T + Minv[:3, 3]
                mrel = Minv @ Mm
            else:
                mloc, topo_m, mrel = np.zeros((0, 3)), (0, 0, 0), np.eye(4)
            if disc.type == "MESH":
                Vd, topo_d, _ = eval_mesh(disc, dg)
                dloc = Vd @ Minv[:3, :3].T + Minv[:3, 3]
            else:
                dloc, topo_d = np.zeros((0, 3)), (0, 0, 0)
            stat = np.array([np.array(o.evaluated_get(dg).matrix_world, dtype=np.float64) for o in statics]) if statics else np.zeros((0, 4, 4))
            if static0 is None:
                static0, mark0, disc0 = stat, (mloc.copy(), mrel.copy()), dloc.copy()
            anc = np.array(anchor.evaluated_get(dg).matrix_world, dtype=np.float64)[:3, 3] if anchor is not None else None
            S.append(dict(n=n, s=s, Md=Md, Rd=Rd, C=C, P=Pb, top=top, bot=bot, wire_perp_max=float(perp.max()),
                          wire_len=float(np.linalg.norm(bot - top)), topo=tuple(topo_w) + tuple(topo_m) + tuple(topo_d),
                          mark_dev=float(np.max(np.linalg.norm(mloc - mark0[0], axis=1))) if (len(mloc) and mloc.shape == mark0[0].shape) else (0.0 if not len(mloc) else float("inf")),
                          mark_rel_dev=float(np.max(np.abs(mrel - mark0[1]))), mark_local=mloc,
                          disc_mesh_dev=float(np.max(np.linalg.norm(dloc - disc0, axis=1))) if (len(dloc) and dloc.shape == disc0.shape) else 0.0,
                          static_drift=float(np.max(np.abs(stat[:, :3, 3] - static0[:, :3, 3]))) if len(stat) else 0.0,
                          static_rot_drift=float(np.max(np.abs(stat[:, :3, :3] - static0[:, :3, :3]))) if len(stat) else 0.0, anchor=anc,
                          scale=sca))
    tt = np.array([(x["n"] + x["s"]) / 24.0 for x in S]); idx = np.array([int(round(4 * (x["n"] + x["s"]))) for x in S])
    inclip = tt <= 10.0 + 1e-12
    Cb = np.array([x["C"] for x in S]); Pb = np.array([x["P"] for x in S]); Rb = np.array([x["Rd"] for x in S]); Md = np.array([x["Md"] for x in S])
    tops = np.array([x["top"] for x in S]); bots = np.array([x["bot"] for x in S]); topo = np.array([x["topo"] for x in S])
    # ---------------- P5
    p5 = dict(n_samples=len(S), n_judged=int(inclip.sum()))
    if anchor is not None:
        AN = np.array([x["anchor"] for x in S])
        p5["anchor_world_vs_expected_max_m"] = float(np.max(np.linalg.norm(AN - anchor_world, axis=1))); p5["anchor_expected"] = anchor_world.tolist()
        p5["anchor_drift_from_frame0_max_m"] = float(np.max(np.linalg.norm(AN - AN[0], axis=1)))
    else:
        p5["anchor"] = "no anchor object; anchor judged via the wire top end only"
    p5["static_objects_max_translation_drift_m"] = max(x["static_drift"] for x in S)
    p5["static_objects_max_matrix_drift"] = max(x["static_rot_drift"] for x in S)
    gtop = np.linalg.norm(tops - (anchor_world + a.top_end_offset * (Pb - anchor_world) / np.linalg.norm(Pb - anchor_world, axis=1, keepdims=True)), axis=1)
    gbot = np.linalg.norm(bots - (Pb - a.bottom_end_offset * (Pb - anchor_world) / np.linalg.norm(Pb - anchor_world, axis=1, keepdims=True)), axis=1)
    p5["wire_top_end_gap_max_m"] = float(gtop[inclip].max()); p5["wire_bottom_end_gap_max_m"] = float(gbot[inclip].max())
    p5["wire_top_end_world_variation_m"] = float(np.max(np.linalg.norm(tops - tops[0], axis=1)[inclip]))
    wl = np.array([x["wire_len"] for x in S])
    p5["wire_length_min_max_m"] = [float(wl[inclip].min()), float(wl[inclip].max())]
    p5["wire_length_variation_m"] = float(np.ptp(wl[inclip]))
    p5["attach_to_anchor_vs_L_max_m"] = float(np.max(np.abs(np.linalg.norm(Pb - anchor_world, axis=1) - model.L)[inclip]))
    p5["info_wire_max_perp_distance_m"] = float(max(x["wire_perp_max"] for x in S))
    p5["topology_constant"] = bool(np.all(topo == topo[0])); p5["topology_frame0_wire_mark_disc"] = topo[0].tolist()
    p5["mark_vertices_in_disc_frame_max_dev_m"] = float(max(x["mark_dev"] for x, k in zip(S, inclip) if k))
    p5["mark_matrix_rel_disc_max_dev"] = float(max(x["mark_rel_dev"] for x, k in zip(S, inclip) if k))
    p5["disc_mesh_in_own_frame_max_dev_m"] = float(max(x["disc_mesh_dev"] for x, k in zip(S, inclip) if k))
    sc_all = np.array([x["scale"] for x in S]); p5["disc_object_scale_min_max"] = [float(sc_all.min()), float(sc_all.max())]
    p5["mark"] = "mark object" if mark is not None else "NO mark object found (mark may be disc geometry/material; then rigid with the disc mesh)"
    ok5 = (p5.get("anchor_world_vs_expected_max_m", 0) < TOL["P5_anchor"] and p5.get("anchor_drift_from_frame0_max_m", 0) < TOL["P5_anchor"]
           and p5["static_objects_max_translation_drift_m"] < TOL["P5_static"] and p5["wire_top_end_gap_max_m"] < TOL["P5_attach"]
           and p5["wire_bottom_end_gap_max_m"] < TOL["P5_attach"] and p5["wire_length_variation_m"] < TOL["P5_attach"]
           and p5["attach_to_anchor_vs_L_max_m"] < TOL["P5_attach"] and p5["topology_constant"]
           and p5["mark_vertices_in_disc_frame_max_dev_m"] < TOL["P5_mark"] and p5["mark_matrix_rel_disc_max_dev"] < TOL["P5_mark"]
           and p5["disc_mesh_in_own_frame_max_dev_m"] < TOL["P5_mark"] and np.ptp(sc_all) < 1e-6)
    p5["status"] = "PASS" if ok5 else "FAIL"
    rep["P5_constraints_saved_blend"] = p5
    # ---------------- P6
    Cref = np.stack([ref["x_m"], ref["y_m"], ref["z_m"]], -1)[idx]
    Pref = np.stack([ref["px_m"], ref["py_m"], ref["pz_m"]], -1)[idx]
    Rref = quat_to_R(np.stack([ref["qw"], ref["qx"], ref["qy"], ref["qz"]], -1)[idx])
    dC = np.linalg.norm(Cb - Cref, axis=1); dP = np.linalg.norm(Pb - Pref, axis=1)
    Rbind = np.eye(3) if a.absolute_axes else Rref[0].T @ Rb[0]
    dR = rot_angle(Rref @ Rbind, Rb)
    dR_abs = rot_angle(Rref, Rb)
    u_b = (anchor_world - Pb) / np.linalg.norm(anchor_world - Pb, axis=1, keepdims=True)
    psi_b = twist_of(model.twist, u_b, Rb[:, :, 2], Rb[:, :, 0])
    dpsi = np.abs(np.angle(np.exp(1j * (psi_b - ref["psi_rad"][idx]))))
    sub0 = np.array([x["s"] for x in S]) == 0
    p6 = dict(max_abs_t_ref_mismatch=float(np.max(np.abs(ref["t_s"][idx] - tt))),
              C_vs_ref_max_m=float(dC[inclip].max()), C_vs_ref_t_at_max=float(tt[inclip][np.argmax(dC[inclip])]),
              C_vs_ref_on_frames_max_m=float(dC[inclip & sub0].max()), attach_vs_ref_max_m=float(dP[inclip].max()),
              rotation_vs_ref_max_rad=float(dR[inclip].max()), rotation_vs_ref_t_at_max=float(tt[inclip][np.argmax(dR[inclip])]),
              rotation_bind=("absolute (I)" if a.absolute_axes else "frame-0 bind R_ref(0)^T R_blend(0)"), info_bind_angle_rad=float(rot_angle(np.eye(3), Rbind)),
              info_rotation_absolute_max_rad=float(dR_abs[inclip].max()),
              psi_vs_ref_max_rad=float(dpsi[inclip].max()), psi_convention=model.twist,
              info_after_clip_end_C_max_m=float(dC[~inclip].max()) if (~inclip).any() else None)
    ok6 = (p6["C_vs_ref_max_m"] < TOL["P6_pos"] and p6["attach_vs_ref_max_m"] < TOL["P6_pos"] and p6["rotation_vs_ref_max_rad"] < TOL["P6_rad"]
           and p6["psi_vs_ref_max_rad"] < TOL["P6_rad"] and p6["max_abs_t_ref_mismatch"] < 1e-12 and scene["ok"])
    if traj is not None and all(c in traj for c in ("x_m", "y_m", "z_m")):
        tr = traj["t_s"]; it = np.round(tt * 96).astype(int)
        ok_t = (it < len(tr)) & inclip
        ok_t[ok_t] &= np.abs(tr[it[ok_t]] - tt[ok_t]) < 1e-9
        if ok_t.any():
            Ct = np.stack([traj["x_m"], traj["y_m"], traj["z_m"]], -1)[it[ok_t]]
            p6["C_vs_trajectory_max_m"] = float(np.linalg.norm(Cb[ok_t] - Ct, axis=1).max()); p6["trajectory_samples"] = int(ok_t.sum())
            ok6 = ok6 and p6["C_vs_trajectory_max_m"] < TOL["P6_pos"]
    # interpolation probe between 1/96 s samples (catches stepped/linear keys that coincide with the sample grid):
    # every 3rd frame at subframes .125/.375/.625/.875 vs the ref, cubic-Hermite interpolated (C with v_C; q with qdot from w)
    PR = []
    for n in range(f0, min(f1, 240), 3):
        for s in (0.125, 0.375, 0.625, 0.875):
            sc.frame_set(n, subframe=s)
            dg = bpy.context.evaluated_depsgraph_get()
            M_ = np.array(disc.evaluated_get(dg).matrix_world, dtype=np.float64)
            PR.append(((n + s) / 24.0, (M_ @ np.r_[com_local, 1.0])[:3], (M_[:3, :3] / np.linalg.norm(M_[:3, :3], axis=0)) @ Rmap))
    if PR:
        tp = np.array([x[0] for x in PR]); i0 = np.floor(tp * 96 + 1e-9).astype(int); h = 1 / 96.0; sfr = (tp - i0 * h) / h
        h00 = 2 * sfr ** 3 - 3 * sfr ** 2 + 1; h10 = sfr ** 3 - 2 * sfr ** 2 + sfr; h01 = -2 * sfr ** 3 + 3 * sfr ** 2; h11 = sfr ** 3 - sfr ** 2
        def herm(X, Vv):
            return h00[:, None] * X[i0] + h10[:, None] * h * Vv[i0] + h01[:, None] * X[i0 + 1] + h11[:, None] * h * Vv[i0 + 1]
        Xc = np.stack([ref["x_m"], ref["y_m"], ref["z_m"]], -1); Vc = np.stack([ref["vx_m_s"], ref["vy_m_s"], ref["vz_m_s"]], -1)
        Q = np.stack([ref["qw"], ref["qx"], ref["qy"], ref["qz"]], -1); W = np.stack([ref["wx"], ref["wy"], ref["wz"]], -1)
        from tmodel import quat_mul
        Qd = 0.5 * quat_mul(np.concatenate([np.zeros((len(Q), 1)), W], -1), Q)
        Ci = herm(Xc, Vc); Qi = herm(Q, Qd); Ri = quat_to_R(Qi / np.linalg.norm(Qi, axis=1, keepdims=True))
        p6["interp_probe_n"] = len(PR)
        p6["interp_probe_C_max_m"] = float(np.max(np.linalg.norm(np.array([x[1] for x in PR]) - Ci, axis=1)))
        p6["interp_probe_rotation_max_rad"] = float(np.max(rot_angle(Ri @ Rbind, np.array([x[2] for x in PR]))))
        p6["interp_probe_note"] = "added check (not a relaxation): between-key samples, same 1e-5 tolerances; ref Hermite error ~1e-8"
        ok6 = ok6 and p6["interp_probe_C_max_m"] < TOL["P6_pos"] and p6["interp_probe_rotation_max_rad"] < TOL["P6_rad"]
    p6["status"] = "PASS" if ok6 else "FAIL"
    rep["P6_blend_vs_solver"] = p6
    # ---------------- keyframe audit + P8 blend
    farr = {}
    rep["animation"] = {}
    for o in bpy.data.objects:
        info, arr = fcurves_of(o)
        if info.get("has_animation_data"):
            rep["animation"][o.name] = info; farr.update(arr)
    if getattr(wire.data, "shape_keys", None) is not None:
        info, arr = fcurves_of(wire.data.shape_keys); rep["animation"]["wire_shape_keys"] = info; farr.update(arr)
    rep["motion_is_baked_keyframes"] = any(k.startswith(disc.name + "|") for k in farr)
    O0 = model.observables(model.initial_state()[None])
    C0 = np.array([O0["x_m"][0], O0["y_m"][0], O0["z_m"][0]])
    p8 = dict(scene=scene, disc_C_frame0_vs_IC_m=float(np.linalg.norm(Cb[0] - C0)),
              disc_attitude_frame0_vs_q0_rad=float(rot_angle(quat_to_R(model.q0) @ Rbind, Rb[0])))
    p8["status"] = "PASS" if (scene["ok"] and p8["disc_C_frame0_vs_IC_m"] < TOL["P8_blend"] and p8["disc_attitude_frame0_vs_q0_rad"] < TOL["P8_blend"]) else "FAIL"
    rep["P8_blend_part"] = p8
    rep["summary"] = dict(P5=p5["status"], P6=p6["status"], P8_blend=p8["status"], motion_is_baked_keyframes=rep["motion_is_baked_keyframes"])
    rep["_series"] = dict(t=tt[::4].tolist(), dC=dC[::4].tolist(), dR=dR[::4].tolist(), dpsi=dpsi[::4].tolist(), gtop=gtop[::4].tolist(), gbot=gbot[::4].tolist())
    if a.dump:
        flat = {}
        for k, arr in farr.items():
            for f in ("co", "hl", "hr", "interp", "htypes"):
                flat[f"{k}|{f}"] = arr[f]
        ml = [x["mark_local"] for x in S]
        np.savez(a.dump, t=tt, in_clip=inclip, disc_matrix=Md, attach=Pb, wire_top=tops, wire_bottom=bots,
                 mark_local=np.array(ml) if len(ml[0]) else np.zeros((len(S), 0, 3)), topology=topo,
                 anchor=np.array([x["anchor"] if x["anchor"] is not None else np.full(3, np.nan) for x in S]), **flat)
        rep["dump"] = a.dump
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("CHECK_BLEND_SUMMARY", json.dumps(rep["summary"]))


main()
