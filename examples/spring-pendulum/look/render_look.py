"""Look preview renders of a built clip-02 .blend (scene role C). Run under tools/with_render_lock.sh.

blender -b look/out/test.blend --python look/render_look.py -- --tag v001 [--frame 0] [--crops] [--percent 50]
    [--engine EEVEE|CYCLES] [--samples N]
Poses VSP_Weight at the reference's measured H for that frame (geometry.json derived.AH_frames_xz_S x S_m; body along
A->H) when the file is not baked (TEST pose, labelled). Writes look/out/<tag>_full.png, <tag>_crop_<name>.png (100 %
border renders, boxes = look/metrics.py CROPS) and <tag>_render.json. Uses the file's own look settings.
"""
import argparse, json, math, pathlib, sys
import bpy

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--frame", type=int, default=0)
ap.add_argument("--engine", default="EEVEE")
ap.add_argument("--percent", type=int, default=50)
ap.add_argument("--samples", type=int, default=None)
ap.add_argument("--crops", action="store_true")
ap.add_argument("--only", default=None, help="comma list of crop names")
ap.add_argument("--outdir", default=str(HERE / "out"))
a = ap.parse_args(argv)
CROPS = {"post": (480, 150, 680, 650), "top": (860, 20, 1100, 260), "spring": (840, 200, 1060, 700),
         "weight": (740, 690, 1060, 1000), "base": (440, 940, 1040, 1088), "table": (1450, 840, 1920, 1088)}
out = pathlib.Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
sc = bpy.context.scene
if bpy.context.window:
    bpy.context.window.scene = sc
sc.frame_set(a.frame)
w = bpy.data.objects["VSP_Weight"]
baked = bool(w.animation_data and w.animation_data.action)
pose = None
if not baked:
    rep = json.loads(pathlib.Path(bpy.data.filepath).with_suffix(".build.json").read_text())
    mp = json.loads((ROOT / "scene" / "measure" / "measured_poses.json").read_text())["frames"][str(a.frame)]
    Sm = rep["S_m"]
    w.location = [x * Sm for x in mp["H_Win"]]
    w.rotation_euler = (0.0, mp["beta_rad"], 0.0)
    pose = {"H_m": list(w.location), "rotation_euler_deg": [0.0, math.degrees(mp["beta_rad"]), 0.0],
            "status": "TEST pose: reference-measured H and weight axis (scene/measure/measured_poses.json)"}
    bpy.context.view_layer.update()
sc.render.engine = "BLENDER_EEVEE" if a.engine.upper().startswith("EEVEE") else "CYCLES"
if a.samples:
    if sc.render.engine == "CYCLES":
        sc.cycles.samples = a.samples
    else:
        sc.eevee.taa_render_samples = a.samples
if sc.render.engine == "CYCLES":
    sc.cycles.device = "CPU"
sc.render.resolution_x, sc.render.resolution_y = 1920, 1088
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_depth = "8"
info = {"tag": a.tag, "frame": a.frame, "engine": sc.render.engine, "camera": sc.camera.name, "baked": baked, "pose": pose,
        "view_transform": sc.view_settings.view_transform, "look": sc.view_settings.look, "files": {}}


def render(path, border=None, percent=100):
    sc.render.resolution_percentage = percent
    sc.render.use_border = border is not None
    sc.render.use_crop_to_border = border is not None
    if border:
        u0, v0, u1, v1 = border
        sc.render.border_min_x, sc.render.border_max_x = u0 / 1920, u1 / 1920
        sc.render.border_min_y, sc.render.border_max_y = 1 - v1 / 1088, 1 - v0 / 1088
    sc.render.filepath = str(path)
    bpy.ops.render.render(write_still=True, scene=sc.name)


f = out / f"{a.tag}_full.png"
render(f, None, a.percent)
info["files"]["full"] = str(f)
if a.crops:
    names = a.only.split(",") if a.only else list(CROPS)
    for k in names:
        box = CROPS[k]
        f = out / f"{a.tag}_crop_{k}.png"
        render(f, box, 100)
        info["files"][k] = str(f)
        info.setdefault("crop_boxes", {})[k] = box
(out / f"{a.tag}_render.json").write_text(json.dumps(info, indent=1))
print("VSP_LOOK_RENDER", json.dumps(info)[:300])
