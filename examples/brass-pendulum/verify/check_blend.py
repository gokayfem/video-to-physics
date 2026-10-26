"""P5, P6 (+ P8 blend part, P10 dump) on a SAVED .blend, evaluated through the depsgraph.

  blender -b <file.blend> --python verify/check_blend.py -- \
      --params params/physics.json --ref verify/out/ref_trajectory.csv \
      [--trajectory solver/out/trajectory.csv] --out verify/out/check_blend.json [--dump verify/out/blend_dump.npz] \
      [--bob-name NAME] [--wire-name NAME] [--pin-name NAME] [--wire-top-target x,y,z] [--model-json override.json]

The ref CSV is produced by verify/ref_solver.py at 1/96 s from t=0 to >= 240.75/24 s, so frame n,
subframe s maps to row 4n + 4s exactly. Bob centre = least-squares sphere fit to the evaluated bob
mesh vertices in world space (fallback: world bounding-box centre); never a custom property.
Runs with Blender's bundled numpy only (imports verify/vparams.py which is stdlib-only).
"""
import bpy, sys, os, json, math, hashlib, datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vparams import load_model, sha256_file  # noqa: E402

TOL = dict(P5_pivot=1e-6, P5_L=1e-6, P5_y=1e-6, P5_wire=1e-5, P6=1e-5, P8_blend=1e-6)
SUBFRAMES = (0.0, 0.25, 0.5, 0.75)


def args():
    import argparse
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--model-json")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--trajectory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump")
    ap.add_argument("--swing-name", default="PND_Swing")
    ap.add_argument("--bob-name"); ap.add_argument("--wire-name"); ap.add_argument("--pin-name")
    ap.add_argument("--wire-top-target", default="0,0,0")
    ap.add_argument("--frames", default="0,240")
    ap.add_argument("--label", default="")
    return ap.parse_args(argv)


def read_csv(path):
    with open(path) as f:
        head = f.readline().strip().split(",")
        A = np.array([[float(x) for x in l.split(",")] for l in f if l.strip()])
    return {h: A[:, i] for i, h in enumerate(head)}


def descendants(o):
    out = []
    for c in o.children:
        out.append(c); out.extend(descendants(c))
    return out


def pick(cands, key_words, explicit, kind):
    if explicit:
        o = bpy.data.objects.get(explicit)
        if o is None:
            raise SystemExit(f"{kind} object {explicit!r} not found")
        return o, "explicit"
    hits = [o for o in cands if any(k in o.name.lower() for k in key_words) and o.type in ("MESH", "CURVE")]
    if len(hits) == 1:
        return hits[0], f"name match {key_words}"
    if len(hits) > 1:
        # prefer mesh with most vertices
        hits.sort(key=lambda o: -len(getattr(o.data, "vertices", [])))
        return hits[0], f"name match {key_words}; ambiguous {[h.name for h in hits]} -> took first by vertex count"
    return None, "not found"


def world_verts(obj, dg):
    oe = obj.evaluated_get(dg)
    me = oe.to_mesh()
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    oe.to_mesh_clear()
    co = co.reshape(n, 3).astype(np.float64)
    M = np.array(oe.matrix_world, dtype=np.float64)
    return co @ M[:3, :3].T + M[:3, 3], M


def sphere_fit(P):
    c0 = P.mean(axis=0)
    Q = P - c0
    A = np.c_[2 * Q, np.ones(len(Q))]
    b = (Q * Q).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = math.sqrt(sol[3] + c @ c)
    res = np.sqrt(((Q - c) ** 2).sum(1)) - r
    return c + c0, r, float(np.sqrt((res ** 2).mean()))


def fcurve_info(obj):
    info = dict(has_animation_data=obj.animation_data is not None)
    ad = obj.animation_data
    if ad is None:
        return info, {}
    info["nla_tracks"] = len(ad.nla_tracks)
    info["nla_strips"] = [dict(name=s.name, scale=s.scale, repeat=s.repeat, frame_start=s.frame_start)
                          for t in ad.nla_tracks for s in t.strips]
    info["drivers"] = [(d.data_path, d.array_index) for d in ad.drivers]
    act = ad.action
    info["action"] = act.name if act else None
    fcs = []
    if act:
        try:
            from bpy_extras import anim_utils
            cb = anim_utils.action_get_channelbag_for_slot(act, ad.action_slot)
            fcs = list(cb.fcurves) if cb else []
        except Exception as e:  # legacy API fallback
            fcs = list(getattr(act, "fcurves", []))
            info["fcurve_api_note"] = str(e)
    info["fcurves"] = []
    arrays = {}
    for fc in fcs:
        kp = fc.keyframe_points
        n = len(kp)
        co = np.empty(2 * n, np.float32); kp.foreach_get("co", co)
        hl = np.empty(2 * n, np.float32); kp.foreach_get("handle_left", hl)
        hr = np.empty(2 * n, np.float32); kp.foreach_get("handle_right", hr)
        key = f"{fc.data_path}[{fc.array_index}]"
        arrays[key] = dict(co=co.reshape(n, 2), hl=hl.reshape(n, 2), hr=hr.reshape(n, 2),
                           interp=[k.interpolation for k in kp],
                           htypes=[(k.handle_left_type, k.handle_right_type) for k in kp])
        info["fcurves"].append(dict(channel=key, n_keys=n, interpolations=sorted(set(k.interpolation for k in kp)),
                                    handle_types=sorted(set(k.handle_left_type for k in kp) | set(k.handle_right_type for k in kp)),
                                    extrapolation=fc.extrapolation, modifiers=[m.type for m in fc.modifiers],
                                    mute=fc.mute, frame_range=[float(co[0]) if n else None, float(co[-2]) if n else None]))
    return info, arrays


def main():
    a = args()
    t_start = datetime.datetime.now(datetime.timezone.utc)
    sc = bpy.context.scene
    blend = bpy.data.filepath
    model, Pjson = load_model(a.params, a.model_json)
    from vparams import find as _find
    r_param, _ = _find(Pjson, "r_bob")
    ref = read_csv(a.ref)
    traj = read_csv(a.trajectory) if a.trajectory else None
    rep = dict(label=a.label, created_utc=t_start.isoformat(timespec="seconds"), blender=bpy.app.version_string,
               inputs={blend: sha256_file(blend), a.params: sha256_file(a.params), a.ref: sha256_file(a.ref)},
               verifier_code_sha256=sha256_file(os.path.abspath(__file__)), tolerances=TOL,
               blend_is_dirty_after_load=bool(bpy.data.is_dirty))
    if a.trajectory:
        rep["inputs"][a.trajectory] = sha256_file(a.trajectory)
    # ref must have been made at these params
    meta_p = a.ref + ".meta.json"
    if os.path.exists(meta_p):
        rm = json.load(open(meta_p))
        rep["ref_params_sha256_matches"] = rm.get("provenance", {}).get("params_sha256") == sha256_file(a.params)
    # ------------------------------------------------ scene / time checks
    scene = dict(fps=sc.render.fps, fps_base=sc.render.fps_base, frame_start=sc.frame_start, frame_end=sc.frame_end,
                 frame_map_old=sc.render.frame_map_old, frame_map_new=sc.render.frame_map_new,
                 unit_scale=sc.unit_settings.scale_length, gravity=list(sc.gravity))
    scene_ok = (scene["fps"] == 24 and scene["fps_base"] == 1.0 and scene["frame_start"] == 0 and scene["frame_end"] == 240
                and scene["frame_map_old"] == scene["frame_map_new"] and scene["unit_scale"] == 1.0)
    scene["ok"] = scene_ok
    rep["scene"] = scene
    # ------------------------------------------------ objects
    sw = bpy.data.objects.get(a.swing_name)
    if sw is None:
        rep["fatal"] = f"{a.swing_name} not found"; json.dump(rep, open(a.out, "w"), indent=2); return
    desc = descendants(sw)
    bob, bob_how = pick(desc, ["bob"], a.bob_name, "bob")
    wire, wire_how = pick(desc, ["wire", "suspension", "string", "thread", "rod"], a.wire_name, "wire")
    pin, pin_how = pick(list(bpy.data.objects), ["pin"], a.pin_name, "pin")
    rep["objects"] = dict(swing=dict(name=sw.name, type=sw.type, rotation_mode=sw.rotation_mode, parent=sw.parent.name if sw.parent else None),
                          bob=dict(name=bob.name if bob else None, how=bob_how, is_descendant_of_swing=bob in desc if bob else False),
                          wire=dict(name=wire.name if wire else None, how=wire_how),
                          pin=dict(name=pin.name if pin else None, how=pin_how),
                          swing_children=[c.name for c in desc])
    if bob is None:
        rep["fatal"] = "bob object not found"; json.dump(rep, open(a.out, "w"), indent=2); return
    finfo, farrays = fcurve_info(sw)
    rep["animation"] = finfo
    rep["motion_is_baked_keyframes"] = any(f["n_keys"] > 0 for f in finfo.get("fcurves", []))
    # other animated objects in scene (should be none besides the rig)
    rep["other_animated_objects"] = [o.name for o in bpy.data.objects if o.animation_data and o != sw and
                                     (o.animation_data.action or o.animation_data.drivers or len(o.animation_data.nla_tracks))]
    # ------------------------------------------------ evaluation loop
    f0, f1 = (int(x) for x in a.frames.split(","))
    top_target = np.array([float(x) for x in a.wire_top_target.split(",")])
    rows = []
    for n in range(f0, f1 + 1):
        for s in SUBFRAMES:
            sc.frame_set(n, subframe=s)
            dg = bpy.context.evaluated_depsgraph_get()
            Vb, Mb = world_verts(bob, dg)
            c, r, rms = sphere_fit(Vb)
            bb = 0.5 * (Vb.min(0) + Vb.max(0))
            Msw = np.array(sw.evaluated_get(dg).matrix_world, dtype=np.float64)
            piv = Msw[:3, 3].copy()
            row = dict(n=n, s=s, c=c, r=r, rms=rms, bb=bb, piv=piv)
            if pin is not None:
                row["pin"] = np.array(pin.evaluated_get(dg).matrix_world, dtype=np.float64)[:3, 3]
            if wire is not None:
                Vw, _ = world_verts(wire, dg)
                u = c / np.linalg.norm(c)
                pr = Vw @ u
                span = pr.max() - pr.min()
                eps = 1e-6 + 1e-4 * span
                top = Vw[pr <= pr.min() + eps].mean(0)
                bot = Vw[pr >= pr.max() - eps].mean(0)
                Minv = np.linalg.inv(Msw)
                row.update(wtop=top, wbot=bot, wtop_loc=(Minv @ np.r_[top, 1])[:3], wbot_loc=(Minv @ np.r_[bot, 1])[:3])
            rows.append(row)
    C = np.array([r_["c"] for r_ in rows]); BB = np.array([r_["bb"] for r_ in rows])
    PIV = np.array([r_["piv"] for r_ in rows])
    tt = np.array([(r_["n"] + r_["s"]) / 24.0 for r_ in rows])
    idx = np.array([int(round(4 * (r_["n"] + r_["s"]))) for r_ in rows])
    fit_rms_max = max(r_["rms"] for r_ in rows)
    radius = float(np.median([r_["r"] for r_ in rows]))
    sphere_ok = fit_rms_max < 1e-3 * radius
    centre = C if sphere_ok else BB
    rep["bob_centre_method"] = "least-squares sphere fit to evaluated world vertices" if sphere_ok else \
        "world bounding-box centre (sphere fit residual too large)"
    rep["bob_sphere_fit"] = dict(radius_m=radius, max_rms_residual_m=fit_rms_max, max_sphere_vs_bbox_centre_m=float(np.max(np.linalg.norm(C - BB, axis=1))))
    rep["bob_sphere_fit"]["r_bob_param_m"] = r_param
    rep["bob_sphere_fit"]["info_rel_radius_blend_vs_params"] = (radius / r_param - 1) if r_param else None
    P = np.zeros(3)  # contract: pivot at the origin
    L = np.linalg.norm(centre - P, axis=1)
    th_b = np.arctan2(centre[:, 0], -centre[:, 2])
    # ---------------- P5
    p5 = dict(n_samples=len(rows),
              pivot_drift_swing_origin_m=float(np.max(np.linalg.norm(PIV - P, axis=1))),
              L_c_param_m=model.L_c, L_min_m=float(L.min()), L_max_m=float(L.max()),
              max_abs_L_minus_Lc_m=float(np.max(np.abs(L - model.L_c))) if model.L_c else None,
              max_abs_y_bob_m=float(np.max(np.abs(centre[:, 1]))),
              L_variation_m=float(L.max() - L.min()))
    if pin is not None:
        PIN = np.array([r_["pin"] for r_ in rows])
        p5["pin_drift_from_frame0_m"] = float(np.max(np.linalg.norm(PIN - PIN[0], axis=1)))
        p5["pin_origin_world_frame0"] = PIN[0].tolist()
    ok5 = p5["pivot_drift_swing_origin_m"] < TOL["P5_pivot"] and p5.get("pin_drift_from_frame0_m", 0) < TOL["P5_pivot"] \
        and p5["max_abs_y_bob_m"] < TOL["P5_y"] and (p5["max_abs_L_minus_Lc_m"] is not None and p5["max_abs_L_minus_Lc_m"] < TOL["P5_L"])
    if wire is not None:
        WT = np.array([r_["wtop"] for r_ in rows]); WB = np.array([r_["wbot"] for r_ in rows])
        WTL = np.array([r_["wtop_loc"] for r_ in rows]); WBL = np.array([r_["wbot_loc"] for r_ in rows])
        gap_top = np.linalg.norm(WT - top_target, axis=1)
        gap_bot = np.abs(np.linalg.norm(WB - centre, axis=1) - radius) if sphere_ok else None
        p5.update(wire_top_gap_max_m=float(gap_top.max()), wire_top_target=top_target.tolist(),
                  wire_bottom_gap_to_bob_surface_max_m=float(gap_bot.max()) if gap_bot is not None else None,
                  wire_top_local_variation_m=float(np.max(np.linalg.norm(WTL - WTL[0], axis=1))),
                  wire_bottom_local_variation_m=float(np.max(np.linalg.norm(WBL - WBL[0], axis=1))),
                  wire_endpoint_method="end-ring centroids of evaluated wire vertices projected on pivot->bob axis")
        ok5 = ok5 and gap_top.max() < TOL["P5_wire"] and (gap_bot is not None and gap_bot.max() < TOL["P5_wire"]) \
            and p5["wire_top_local_variation_m"] < TOL["P5_wire"] and p5["wire_bottom_local_variation_m"] < TOL["P5_wire"]
    else:
        p5["wire"] = "NOT FOUND -> wire endpoint check not run"
        ok5 = False
    p5["status"] = "PASS" if ok5 else "FAIL"
    rep["P5_constraints_saved_blend"] = p5
    # ---------------- P6
    th_ref = ref["theta_rad"][idx]
    t_ref = ref["t_s"][idx]
    inclip = tt <= 10.0 + 1e-12  # frame 240 subframes > 0 lie after the clip end (t > 10 s): reported, not judged
    err = np.abs(th_b - th_ref)
    p6 = dict(max_abs_t_ref_mismatch=float(np.max(np.abs(t_ref - tt))),
              n_samples_judged=int(inclip.sum()),
              max_abs_dtheta_vs_verifier_ref_rad=float(np.max(err[inclip])),
              t_at_max=float(tt[inclip][int(np.argmax(err[inclip]))]),
              rms_dtheta_vs_ref_rad=float(np.sqrt(np.mean(err[inclip] ** 2))),
              info_after_clip_end_max_dtheta_rad=float(np.max(err[~inclip])) if (~inclip).any() else None,
              info_after_clip_end_note="t > 10 s (frame 240 subframes) depends on F-curve extrapolation; outside the clip, not judged")
    if traj is not None:
        inr = idx < len(traj["t_s"])
        p6["max_abs_dtheta_vs_trajectory_rad"] = float(np.max(np.abs(th_b[inr] - traj["theta_rad"][idx[inr]])))
        p6["trajectory_samples_compared"] = int(inr.sum())
    on_frame = np.array([r_["s"] == 0.0 for r_ in rows])
    p6["max_abs_dtheta_on_frames_rad"] = float(np.max(err[on_frame & inclip]))
    p6["max_abs_dtheta_subframes_rad"] = float(np.max(err[~on_frame & inclip]))
    ok6 = p6["max_abs_dtheta_vs_verifier_ref_rad"] < TOL["P6"] and p6["max_abs_t_ref_mismatch"] < 1e-12 and scene_ok \
        and p6.get("max_abs_dtheta_vs_trajectory_rad", 0.0) < TOL["P6"]
    p6["status"] = "PASS" if ok6 else "FAIL"
    rep["P6_blend_vs_solver"] = p6
    # ---------------- keyframe audit (+P8 blend)
    audit = {}
    rot = [k for k in farrays if k.startswith("rotation")]
    sign = float(np.sign(np.sum(th_b * th_ref))) or 1.0
    for k, arr in farrays.items():
        co, hr, hl = arr["co"].astype(np.float64), arr["hr"].astype(np.float64), arr["hl"].astype(np.float64)
        ent = dict(n_keys=len(co), frames_are_integers_0_240=bool(len(co) == 241 and np.all(co[:, 0] == np.arange(241))))
        if k.startswith("rotation_euler[1]") and len(co):
            fr = co[:, 0]
            ii = np.round(4 * fr).astype(int)
            ok_i = ii < len(ref["theta_rad"])
            # Euler sign from geometry: value = s_e * theta
            s_e = float(np.sign(np.dot(co[ok_i, 1], ref["theta_rad"][ii[ok_i]]))) or 1.0
            ent["euler_sign_value_equals"] = f"{'+' if s_e > 0 else '-'}theta"
            ent["max_abs_key_value_vs_ref_rad"] = float(np.max(np.abs(s_e * co[ok_i, 1] - ref["theta_rad"][ii[ok_i]])))
            if traj is not None:
                ok_t = ii < len(traj["theta_rad"])
                ent["max_abs_key_value_vs_trajectory_rad"] = float(np.max(np.abs(s_e * co[ok_t, 1] - traj["theta_rad"][ii[ok_t]])))
            dxr = hr[:, 0] - co[:, 0]; dxl = co[:, 0] - hl[:, 0]
            slope_r = np.where(dxr != 0, (hr[:, 1] - co[:, 1]) / np.where(dxr != 0, dxr, 1), np.nan)
            om_ref = ref["omega_rad_s"][ii[ok_i]] / 24.0
            ent["max_abs_handle_slope_vs_omega_over_24"] = float(np.nanmax(np.abs(s_e * slope_r[ok_i] - om_ref)))
            ent["handle_x_offsets_frames"] = [float(np.min(dxl)), float(np.max(dxr))]
            ent["theta_key0_vs_theta0_param"] = float(abs(s_e * co[0, 1] - model.theta0)) if fr[0] == 0 else None
        audit[k] = ent
    rep["keyframe_audit"] = audit
    i00 = 0
    p8b = dict(theta_blend_frame0_rad=float(th_b[i00]), theta0_param=model.theta0,
               abs_diff=float(abs(th_b[i00] - model.theta0)), scene=scene)
    p8b["status"] = "PASS" if (p8b["abs_diff"] < TOL["P8_blend"] and scene_ok) else "FAIL"
    rep["P8_blend_part"] = p8b
    rep["summary"] = dict(P5=p5["status"], P6=p6["status"], P8_blend=p8b["status"],
                          motion_is_baked_keyframes=rep["motion_is_baked_keyframes"])
    if a.dump:
        flat = {}
        for k, arr in farrays.items():
            flat[f"{k}|co"] = arr["co"]; flat[f"{k}|hl"] = arr["hl"]; flat[f"{k}|hr"] = arr["hr"]
            flat[f"{k}|interp"] = np.array(arr["interp"]); flat[f"{k}|htypes"] = np.array(arr["htypes"])
        np.savez(a.dump, t=tt, theta_ref=th_ref, in_clip=inclip, bob_centre=centre, sphere_centre=C, bbox_centre=BB, pivot=PIV, theta_blend=th_b, **flat)
        rep["dump"] = a.dump
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("CHECK_BLEND_SUMMARY", json.dumps(rep["summary"]))


main()
