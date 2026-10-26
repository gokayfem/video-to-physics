"""Checks for scene/out/vortex_anim.blend -> scene/out/vortex_anim_checks.json, plus labelled visuals.
usage: python scene/anim/checks.py   (after build_anim.py runs: prev_ (960x544 32 spp, frames 0,1,60,120,240),
       full_test_/full_nodye_/full_ffdye_ (1920x1088, frame 0, params samples))"""
import json, os, sys, subprocess, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom
ROOT = geom.ROOT; OUT = os.path.join(ROOT, "scene", "out"); RD = os.path.join(OUT, "anim_renders")
VIS = os.path.join(ROOT, "scene", "visuals"); os.makedirs(VIS, exist_ok=True)
PY = sys.executable; P = geom.P; res = {"note": "TEST dye sequence (scene/anim/test_dye, synthetic kinematics) - not the solver output"}
st = json.load(open(os.path.join(OUT, "vortex_anim_state.json")))
meta = json.load(open(os.path.join(st["dye_dir"], "meta.json")))
rd = lambda n: cv2.imread(os.path.join(RD, n)).astype(np.float32)

# 1 camera identity ------------------------------------------------------------------------------
cam = st["camera"]; W, H = cam["res"]; Mw = np.array(cam["matrix_world"]); f = cam["lens_mm"] / cam["sensor_width_mm"] * W
Ra = np.diag([1, -1, -1]) @ Mw[:3, :3].T; ta = -Ra @ Mw[:3, 3]; Ka = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
pts = np.array([v["X"] for v in geom.C["projection_test_anchor_points"].values()] + [[0.1, 0.03, 0.013], [-0.2, -0.05, 0], [0.25, 0.05, 0.03]])
d = np.linalg.norm(geom.proj(pts, Ka, Ra, ta) - geom.proj(pts), axis=1)
res["camera_identical"] = {"n_points": len(pts), "max_reproj_px": float(d.max()), "tol_px": 0.05, "pass": bool(d.max() <= 0.05),
                           "dof_anim": cam["dof"], "dof_ff": geom.C["intrinsics"]["depth_of_field"], "shift": cam["shift"]}

# 2 texture -> world mapping via 3 marker discs (render with dye - render without dye, frame 0, full res)
A, B = rd("full_test_f000.png"), rd("full_nodye_f000.png")
diff = np.abs(A - B).sum(2)
mk = []
for (cc, rr), (xw, yw) in zip(meta["markers_px_col_row"], meta["markers_world_xy"]):
    Xw = [xw, yw, P["dye"]["z_sheet"]]; pr = geom.proj_refr([Xw])[0]; pp = geom.proj([Xw])[0]
    u0, v0 = int(round(pr[0])), int(round(pr[1])); r = 22
    win = diff[v0 - r:v0 + r + 1, u0 - r:u0 + r + 1]; m = win > 0.5 * win.max()
    vv, uu = np.nonzero(m); w = win[m]
    meas = [u0 - r + (uu * w).sum() / w.sum() + 0.5, v0 - r + (vv * w).sum() / w.sum() + 0.5]   # pixel-centre convention
    e = float(np.hypot(meas[0] - pr[0], meas[1] - pr[1]))
    mk.append({"tex_px_col_row": [cc, rr], "world_xyz": Xw, "pred_refraction_uv": [round(float(x), 2) for x in pr],
               "pred_pinhole_uv_no_refraction": [round(float(x), 2) for x in pp], "measured_uv": [round(float(x), 2) for x in meas],
               "err_px": round(e, 3), "pinhole_err_px": round(float(np.hypot(meas[0] - pp[0], meas[1] - pp[1])), 2), "peak_diff_8bit_sum": float(win.max())})
res["texture_to_world_markers"] = {"markers": mk, "max_err_px": max(m_["err_px"] for m_ in mk), "tol_px": 1.0,
                                   "pass": bool(max(m_["err_px"] for m_ in mk) <= 1.0),
                                   "method": "centroid of |render(dye) - render(no dye)| in a 45 px window; prediction = flat-surface Snell (n=1.333, z=0.023) + camera_ff pinhole"}

# 3 dye only inside the channel ------------------------------------------------------------------
hw = P["channel"]["inner_width"] / 2; sb = st["dye_sheet_bounds"]
geo_ok = sb["y"][0] >= -hw and sb["y"][1] <= hw and P["water"]["surface_z"] > sb["z"][0] > 0
def poly_mask(z, shape=(H, W), dil=6):
    xs = np.linspace(-0.30, 0.30, 400)
    edge = np.concatenate([np.c_[xs, np.full_like(xs, -hw), np.full_like(xs, z)], np.c_[xs[::-1], np.full_like(xs, hw), np.full_like(xs, z)]])
    uv = geom.proj_refr(edge) if z < geom.ZS else geom.proj(edge)
    m = np.zeros(shape, np.uint8); cv2.fillPoly(m, [np.round(uv * 8).astype(np.int32)], 1, shift=3)
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1, 2 * dil + 1))) > 0
inch = poly_mask(P["dye"]["z_sheet"]) | poly_mask(0.0)
sig = diff > 3 * 8   # > 8/255 per channel on average
frac_water = float((sig & ~inch).sum() / max(sig.sum(), 1))
t_w = P["channel"]["wall_thickness"]; zt = P["channel"]["wall_height"] + P["channel"]["wall_z_gap"]
corners = np.array([[x, y, z] for x in (-0.30, 0.30) for y in (-hw - t_w, hw + t_w) for z in (0.0, zt)])
box = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(box, cv2.convexHull(np.round(geom.proj(corners)).astype(np.int32)), 1)
box = cv2.dilate(box, np.ones((13, 13), np.uint8)) > 0
frac_out = float((sig & ~box).sum() / max(sig.sum(), 1))
res["dye_only_inside_channel"] = {"sheet_bounds_m": sb, "channel_inner_half_width_m": hw, "geometry_pass": bool(geo_ok),
    "render_changed_px(>8/255)": int(sig.sum()), "fraction_outside_channel_incl_walls": frac_out,
    "fraction_outside_water_footprint_(=reflections_in_acrylic_walls)": frac_water,
    "pass": bool(geo_ok and frac_out < 0.001),
    "note": "changed px = |render(TEST dye) - render(no dye)| frame 0; channel incl. walls = projected hull of the channel box (walls outer faces, z 0..wall top) +6 px; water footprint = refracted projection of the channel interior at sheet height and floor +6 px"}
cv2.imwrite(os.path.join(OUT, "anim_check_dye_diff.png"), np.clip(diff, 0, 255).astype(np.uint8))

# 4 frame-index sync -------------------------------------------------------------------------------
bc = meta["barcode"]; sync = {}
for fr in (0, 1, 60, 120, 240):
    im = rd(f"prev_f{fr:03d}.png"); s = im.shape[1] / W
    def samp(x, y):
        u, v = geom.proj_refr([[x, y, P["dye"]["z_sheet"]]])[0] * s
        u, v = int(u), int(v); p = im[v - 1:v + 2, u - 1:u + 2].reshape(-1, 3).mean(0)
        return (p[1] + p[2]) / 2 - p[0]          # yellowness (BGR)
    sc_ = np.array([samp(x, bc["y"]) - samp(x, bc["y"] - 0.006) for x in bc["cells_x"]])
    thr = 0.5 * min(sc_[0], sc_[-1]); bits = (sc_[1:-1] > thr).astype(int)
    dec = int("".join(map(str, bits)), 2)
    sync[fr] = {"texture_file_resolved_by_blender": st["texture_file_for_scene_frame"].get(str(fr)), "barcode_decoded_from_render": dec,
                "guard_contrast": [round(float(sc_[0]), 1), round(float(sc_[-1]), 1)], "pass": bool(dec == fr and st["texture_file_for_scene_frame"].get(str(fr)) == f"f{fr:04d}.png")}
res["frame_index_sync"] = {"frames": sync, "pass": all(v["pass"] for v in sync.values()),
                           "image_user": "frame_start=0, frame_offset=-1, frame_duration=241, auto_refresh"}

# 5 frame-0 look vs firstframe/frame0.png -------------------------------------------------------
F0 = cv2.imread(os.path.join(ROOT, "firstframe", "frame0.png")).astype(np.float32)
def post(name):
    o = os.path.join(RD, "post_" + name); subprocess.run([PY, os.path.join(ROOT, "scene", "post.py"), os.path.join(RD, name), o], check=True)
    return cv2.imread(o).astype(np.float32)
At = post("full_test_f000.png")
# dye area = refracted footprint of (frame-0 look dye alpha > 0.01) U (test dye sum > 0.01) at sheet height + on the floor (shadows)
dye_ff = cv2.imread(os.path.join(OUT, "dye_ff.png"), 0).astype(np.float32) / 255
ny, nx = dye_ff.shape; rff = P["dye"]["texture_res_m_per_px"]
jj, ii = np.nonzero(dye_ff[::5, ::5] > 0.01); xf = -0.15 + (ii * 5 + 0.5) * rff; yf = (-hw + 0.0005) + (jj * 5 + 0.5) * rff
t0 = cv2.imread(os.path.join(st["dye_dir"], "f0000.png"), -1).astype(np.float32).sum(2) / 65535
r2, c2 = np.nonzero(t0[::2, ::2] > 0.01); xt, yt = geom.px_to_world(c2 * 2, r2 * 2)
XY = np.r_[np.c_[xf, yf], np.c_[xt, yt]]
dm = np.zeros((H, W), np.uint8)
for z, dil in ((P["dye"]["z_sheet"], 12), (0.0, 30)):
    uv = np.round(geom.proj_refr(np.c_[XY, np.full(len(XY), z)])).astype(int); ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    mm = np.zeros((H, W), np.uint8); mm[uv[ok, 1], uv[ok, 0]] = 1
    dm |= cv2.dilate(mm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1, 2 * dil + 1)))
dm = dm > 0
mad_out = float(np.abs(At - F0)[~dm].mean()); mad_all = float(np.abs(At - F0).mean())
r5 = {"mad_outside_dye_area_8bit": round(mad_out, 3), "tol_8bit": 2.0, "pass": bool(mad_out <= 2.0), "mad_whole_image_8bit": round(mad_all, 3),
      "dye_area_fraction_of_image": round(float(dm.mean()), 4), "p99_absdiff_outside": float(np.percentile(np.abs(At - F0)[~dm], 99)),
      "dye_area": "refracted footprint of frame-0 look dye + TEST f0000 dye at z=0.013 (dilated 12 px) and on the floor z=0 (dilated 30 px, shadows)"}
if os.path.exists(os.path.join(RD, "full_ffdye_f000.png")):
    Ac = post("full_ffdye_f000.png"); r5["identity_with_frame0_dye_textures_mad_whole_image_8bit"] = round(float(np.abs(Ac - F0).mean()), 3)
res["frame0_vs_firstframe"] = r5
vis = np.clip(np.abs(At - F0).mean(2) * 8, 0, 255).astype(np.uint8); vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR); vis[dm] = (vis[dm] * 0.4 + np.array([60, 0, 0])).astype(np.uint8)
cv2.imwrite(os.path.join(OUT, "anim_check_frame0_absdiff_x8.png"), vis)

res["all_pass"] = all(v.get("pass", True) for v in res.values() if isinstance(v, dict))
json.dump(res, open(os.path.join(OUT, "vortex_anim_checks.json"), "w"), indent=1)

# visuals --------------------------------------------------------------------------------------
tiles = []
for fr in (0, 60, 120, 240):
    src = f"prev_f{fr:03d}.png"; im = post(src).astype(np.uint8)
    lab = f"v001 | f{fr:03d} | t={fr / 24:.2f}s | TEST dye"
    cv2.rectangle(im, (0, 0), (430, 34), (0, 0, 0), -1); cv2.putText(im, lab, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(VIS, f"v001_f{fr:03d}_TEST.png"), im); tiles.append(im)
sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
cv2.imwrite(os.path.join(VIS, "v001_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in res.items()}, indent=0))
