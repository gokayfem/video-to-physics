"""Look/blockout preview renders of a built clip-03 .blend (scene role C). Run under tools/with_render_lock.sh.

blender -b X.blend --python look/render_look.py -- --tag v001 [--frame 0] [--crops] [--percent 50] [--samples N]
    [--camera TOR_Cam_Reference] [--no-pose] [--outdir look/out]
Unbaked files: TOR_Disc is posed at the reference-measured TEST pose for that frame (scene/measure/measured_poses.json:
twist psi of the engraved line, disc offset where fitted), labelled in <tag>_render.json. Writes <tag>_full.png
(percent of 1920x1088), <tag>_crop_<name>.png (100 % border renders of CROPS, boxes in that frame's pixels) and
<tag>_render.json.
"""
import argparse, json, math, pathlib, sys
import bpy
from mathutils import Matrix

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--frame", type=int, default=0)
ap.add_argument("--percent", type=int, default=50)
ap.add_argument("--samples", type=int, default=None)
ap.add_argument("--crops", action="store_true")
ap.add_argument("--only", default=None)
ap.add_argument("--camera", default="TOR_Cam_Reference")
ap.add_argument("--no-pose", action="store_true")
ap.add_argument("--outdir", default=str(HERE / "out"))
ap.add_argument("--auto-crops", action="store_true", help="crop boxes from projecting fixed 3D points through camera/project.py at --frame")
a = ap.parse_args(argv)
CROPS = {"clamp": (860, 0, 1100, 240), "disc": (660, 540, 1300, 820), "wood": (380, 0, 640, 260)}
out = pathlib.Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
if a.auto_crops:
    sys.path.insert(0, str(ROOT / "camera"))
    import numpy as np
    import project as PJ
    cam = PJ.load_camera(frame=a.frame)
    G = json.loads((ROOT / "params/geometry.json").read_text())["ratios"]
    L_ = G["wire_length_A_to_disc_top"]["value"]
    xl = G["frame_centre_x"]["value"] - G["span_inner"]["value"] / 2
    zt = G["post_top_z"]["value"]

    def box(P, w, h):
        u, v = PJ.project(np.array(P, float), cam)
        u0 = int(min(max(round(float(u) - w / 2), 0), 1920 - w)); v0 = int(min(max(round(float(v) - h / 2), 0), 1088 - h))
        return (u0, v0, u0 + w, v0 + h)
    CROPS = {"clamp": box((0.0, 0.0, 0.14), 240, 240), "disc": box((0.0, 0.0, -L_ - 0.02), 640, 280),
             "wood": box((xl, -0.1, zt - 0.2), 260, 260)}
sc = bpy.context.scene
sc.frame_set(a.frame)
sc.camera = bpy.data.objects[a.camera]
disc = bpy.data.objects["TOR_Disc"]
baked = bool(disc.animation_data and disc.animation_data.action)
pose = None
if not baked and not a.no_pose:
    rep = json.loads(pathlib.Path(bpy.data.filepath).with_suffix(".build.json").read_text())
    Dm = rep["D_m"]
    mp = json.loads((ROOT / "scene/measure/measured_poses.json").read_text())["frames"]
    keys = sorted(int(k) for k in mp)
    k = min(keys, key=lambda q: abs(q - a.frame))
    P = mp[str(k)]
    L = rep["dims_m"]["wire_length_A_to_disc_top"]
    mu = float(disc.get("tor_mark_azimuth_rad", 0.0))
    disc.location = (P["offset_D"][0] * Dm, P["offset_D"][1] * Dm, -L)
    disc.rotation_mode = "QUATERNION"
    disc.rotation_quaternion = Matrix.Rotation(math.radians(P["psi_deg"]) - mu, 3, "Z").to_quaternion()
    pose = {"frame_used": k, "psi_deg": P["psi_deg"], "offset_D": P["offset_D"],
            "status": "TEST pose: reference-measured mark angle (scene/measure/measured_poses.json), not physics"}
    bpy.context.view_layer.update()
sc.render.engine = "BLENDER_EEVEE"
if a.samples:
    sc.eevee.taa_render_samples = a.samples
sc.render.resolution_x, sc.render.resolution_y = 1920, 1088
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_depth = "8"
info = {"tag": a.tag, "frame": a.frame, "camera": sc.camera.name, "baked": baked, "pose": pose,
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
    for k in (a.only.split(",") if a.only else list(CROPS)):
        f = out / f"{a.tag}_crop_{k}.png"
        render(f, CROPS[k], 100)
        info["files"][k] = str(f)
        info.setdefault("crop_boxes", {})[k] = CROPS[k]
(out / f"{a.tag}_render.json").write_text(json.dumps(info, indent=1))
print("TOR_LOOK_RENDER", json.dumps(info)[:300])
