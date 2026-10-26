"""Motion-blur calibration renders (scene role; run under tools/with_render_lock.sh).
blender -b look/out/v010.blend --python scene/measure/mb_tune_render.py -- --out scene/out/mb
The weight (frame-120 measured TEST pose) is keyed LINEAR in x so that it moves at a known image speed through frame 10;
full-res border crops of frame 10 are rendered for several shutters (CENTER). Writes crops + mb_render.json."""
import argparse, json, math, pathlib, sys
import bpy
from bpy_extras.object_utils import world_to_camera_view as w2c
from mathutils import Vector
ROOT = pathlib.Path(__file__).resolve().parents[2]
argv = sys.argv[sys.argv.index("--") + 1:]
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
ap.add_argument("--speeds", default="0,8,16,24"); ap.add_argument("--shutters", default="0,0.1,0.2,0.35,0.5")
a = ap.parse_args(argv)
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
sc = bpy.context.scene
rep = json.loads(pathlib.Path(bpy.data.filepath).with_suffix(".build.json").read_text()); Sm = rep["S_m"]
mp = json.loads((ROOT / "scene/measure/measured_poses.json").read_text())["frames"]["120"]
w = bpy.data.objects["VSP_Weight"]
if w.animation_data:
    w.animation_data_clear()
H0 = Vector([x * Sm for x in mp["H_Win"]]); beta = mp["beta_rad"]
w.rotation_euler = (0.0, beta, 0.0)
cam = sc.camera
sc.frame_set(10)
def px(P):
    q = w2c(sc, cam, P); return Vector((q.x * 1920, (1 - q.y) * 1088))
dpx = (px(H0 + Vector((0.01, 0, 0))) - px(H0)).length / 0.01          # px per metre in x at H
info = {"pose": mp, "px_per_m_x": dpx, "renders": []}
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 1920, 1088, 100
sc.render.image_settings.file_format = "PNG"
for spd in [float(x) for x in a.speeds.split(",")]:
    vx = spd / dpx                                                      # metres per frame
    if w.animation_data:
        w.animation_data_clear()
    for f in (8, 12):
        w.location = H0 + Vector(((f - 10) * vx, 0, 0)); w.keyframe_insert("location", frame=f)
    for fc in (w.animation_data.action.layers[0].strips[0].channelbags[0].fcurves if hasattr(w.animation_data.action, "layers") and len(w.animation_data.action.layers) else w.animation_data.action.fcurves):
        for k in fc.keyframe_points:
            k.interpolation = "LINEAR"
    sc.frame_set(10)
    c = px(w.matrix_world @ Vector((0, 0, -0.06 * Sm / 0.3 * 0.0 - rep["dims_m"]["H_to_cylinder_centre_used"])))
    box = (int(c.x) - 140, int(c.y) - 120, int(c.x) + 140, int(c.y) + 120)
    sc.render.use_border = sc.render.use_crop_to_border = True
    sc.render.border_min_x, sc.render.border_max_x = box[0] / 1920, box[2] / 1920
    sc.render.border_min_y, sc.render.border_max_y = 1 - box[3] / 1088, 1 - box[1] / 1088
    for sh in [float(x) for x in a.shutters.split(",")]:
        if spd == 0 and sh > 0:
            continue
        sc.render.use_motion_blur = sh > 0
        sc.render.motion_blur_shutter = max(sh, 0.01); sc.render.motion_blur_position = "CENTER"
        f = out / f"mb_v{spd:g}_s{sh:g}.png"
        sc.render.filepath = str(f)
        bpy.ops.render.render(write_still=True)
        info["renders"].append({"file": f.name, "speed_px_per_frame": spd, "shutter": sh, "box": box, "centre_px": [c.x, c.y]})
(out / "mb_render.json").write_text(json.dumps(info, indent=1))
print("MB_DONE", len(info["renders"]))
