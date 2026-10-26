"""Preview render of a saved clip-02 .blend through its scene camera (scene role). Run under tools/with_render_lock.sh.

blender -b scene/out/x.blend --python scene/preview_render.py -- --frame 0 --out x.png
    [--pose-measured]        pose VSP_Weight at the reference's measured H for this frame (geometry.json derived
                             AH_frames_xz_S x S_m), body axis along A->H  (TEST pose, only when the file is not baked)
    [--pose "Hx,Hy,Hz,rx,ry,rz"]  explicit TEST pose (m, deg)
    [--keep-look]            keep the file's engine / samples / colour management (look.py)
    [--border u0,v0,u1,v1]   full-res region at 100 % (<= 960x544 px)
    [--percent 50]           default 960x544
Writes <out> and <out>.json (pose used, camera, frame).
"""
import argparse, json, math, pathlib, sys
import bpy
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--frame", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--pose", default=None)
ap.add_argument("--pose-measured", action="store_true")
ap.add_argument("--keep-look", action="store_true")
ap.add_argument("--engine", default="BLENDER_EEVEE")
ap.add_argument("--border", default=None)
ap.add_argument("--percent", type=int, default=50)
ap.add_argument("--samples", type=int, default=None)
a = ap.parse_args(argv)
ROOT = pathlib.Path(__file__).resolve().parents[1]
sc = bpy.context.scene
if bpy.context.window:
    bpy.context.window.scene = sc
sc.frame_set(a.frame)
w = bpy.data.objects["VSP_Weight"]
baked = bool(w.animation_data and w.animation_data.action)
pose = None
if (a.pose or a.pose_measured) and not baked:
    if a.pose:
        v = [float(x) for x in a.pose.split(",")]
        loc, rot = v[:3], v[3:6]
    else:
        rep = json.loads(pathlib.Path(bpy.data.filepath).with_suffix(".build.json").read_text())
        Sm = rep["S_m"]
        mp = json.loads((ROOT / "scene" / "measure" / "measured_poses.json").read_text())["frames"][str(a.frame)]
        loc = [x * Sm for x in mp["H_Win"]]
        rot = [0.0, math.degrees(mp["beta_rad"]), 0.0]
    w.location = loc
    w.rotation_euler = [math.radians(x) for x in rot]
    pose = {"H_m": loc, "rotation_euler_deg": rot, "status": "TEST pose: reference-measured H and weight axis (scene/measure/measured_poses.json)"}
    bpy.context.view_layer.update()
if not a.keep_look:
    sc.render.engine = a.engine
    if a.engine == "BLENDER_EEVEE":
        sc.eevee.taa_render_samples = 16
    sc.view_settings.view_transform = "AgX"
if a.samples:
    if sc.render.engine == "CYCLES":
        sc.cycles.samples = a.samples
    else:
        sc.eevee.taa_render_samples = a.samples
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 1920, 1088, a.percent
sc.render.use_border = sc.render.use_crop_to_border = False
if a.border:
    u0, v0, u1, v1 = [float(x) for x in a.border.split(",")]
    assert (u1 - u0) * (v1 - v0) <= 960 * 544, "keep border renders small"
    sc.render.resolution_percentage = 100
    sc.render.use_border, sc.render.use_crop_to_border = True, True
    sc.render.border_min_x, sc.render.border_max_x = u0 / 1920, u1 / 1920
    sc.render.border_min_y, sc.render.border_max_y = 1 - v1 / 1088, 1 - v0 / 1088
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_depth = "8"
sc.render.filepath = str(pathlib.Path(a.out).resolve())
bpy.ops.render.render(write_still=True, scene=sc.name)
info = {"frame": a.frame, "camera": sc.camera.name, "baked": baked, "pose": pose, "engine": sc.render.engine,
        "view_transform": sc.view_settings.view_transform, "look": sc.view_settings.look, "border": a.border,
        "camera_disclosure": sc.camera.get("vsp_disclosure", "")}
pathlib.Path(a.out + ".json").write_text(json.dumps(info, indent=1))
print("VSP_PREVIEW", json.dumps(info))
