"""Render a look preview of a built pendulum .blend (Look role). Run under tools/with_render_lock.sh.

blender -b look/out/test.blend --python look/render_look.py -- --tag v001 [--theta-deg=-4.62] [--engine EEVEE|CYCLES]
    [--percent 50] [--crops] [--samples N]
Writes look/out/<tag>_full.png (+ <tag>_crop_<name>.png at 100 % via border render) and <tag>_render.json with the
bob's projected pixel centre/radius (full-res px, contract convention) for the metrics.
--theta-deg poses PND_Swing; if the file is baked the action is detached in memory only (file not saved), and the
still is labelled as posed.
"""
import argparse, json, math, pathlib, sys
import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

HERE = pathlib.Path(__file__).resolve().parent
CROPS = {  # full-res px boxes (u0, v0, u1, v1), each <= 960x544 px
    "post": (680, 240, 860, 600), "arm_pin": (780, 40, 1140, 200), "base": (480, 860, 1440, 1088),
    "bob": (950, 740, 1130, 920)}

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--frame", type=int, default=0)
ap.add_argument("--theta-deg", type=float, default=None)
ap.add_argument("--engine", default="EEVEE")
ap.add_argument("--percent", type=int, default=50)
ap.add_argument("--samples", type=int, default=None)
ap.add_argument("--crops", action="store_true")
ap.add_argument("--view", default=None, help="experiment: override view transform (e.g. Standard, AgX, Filmic)")
ap.add_argument("--look", default=None, help="experiment: override look (e.g. None, 'AgX - Medium High Contrast')")
ap.add_argument("--exposure", type=float, default=None)
ap.add_argument("--outdir", default=str(HERE / "out"))
a = ap.parse_args(argv)
out = pathlib.Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
sc = bpy.context.scene
if bpy.context.window:
    bpy.context.window.scene = sc
sc.frame_set(a.frame)
sw = bpy.data.objects["PND_Swing"]
posed = False
if a.theta_deg is not None:
    if sw.animation_data and sw.animation_data.action:
        sw.animation_data.action = None
    sw.rotation_euler[1] = -math.radians(a.theta_deg)
    posed = True
bpy.context.view_layer.update()
if a.view:
    sc.view_settings.view_transform = a.view
if a.look:
    sc.view_settings.look = a.look
if a.exposure is not None:
    sc.view_settings.exposure = a.exposure
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

bob = bpy.data.objects["PND_Bob"]
c = bob.matrix_world.translation
cam = sc.camera
p = world_to_camera_view(sc, cam, c)
fpx = cam.data.lens / cam.data.sensor_width * 1920
dist = (cam.matrix_world.inverted() @ c).z * -1
r_px = fpx * bob["pnd_radius_m"] / dist
info = {"tag": a.tag, "frame": a.frame, "posed_theta_deg": a.theta_deg if posed else None, "engine": sc.render.engine,
        "camera": cam.name, "bob_px": [p.x * 1920, (1 - p.y) * 1088, r_px], "view_transform": sc.view_settings.view_transform,
        "look": sc.view_settings.look, "exposure": sc.view_settings.exposure, "files": {}}


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


bu, bv, br = info["bob_px"]
h = max(90, int(1.6 * br))
CROPS["bob"] = (int(bu - h), int(bv - h), int(bu + h), int(bv + h))
f = out / f"{a.tag}_full.png"
render(f, None, a.percent)
info["files"]["full"] = str(f)
if a.crops:
    for k, box in CROPS.items():
        f = out / f"{a.tag}_crop_{k}.png"
        render(f, box, 100)
        info["files"][k] = str(f)
        info.setdefault("crop_boxes", {})[k] = box
(out / f"{a.tag}_render.json").write_text(json.dumps(info, indent=1))
print("PND_LOOK_RENDER", json.dumps(info))
