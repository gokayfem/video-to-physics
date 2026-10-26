"""Review renders of a (baked) clip-02 .blend at given frames: full frame at 50 % + full-res close-up crops (scene role).
Run under tools/with_render_lock.sh.
blender -b X.blend --python look/render_review.py -- --tag ck001 --frames 0,120,240 --outdir look/out/review
Crops per frame: hook+spring (fixed), walnut post (fixed), weight (centred on the reference weight track at that frame)."""
import argparse, json, pathlib, sys
import bpy
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
argv = sys.argv[sys.argv.index("--") + 1:]
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--frames", default="0,120,240")
ap.add_argument("--no-crops", action="store_true")
ap.add_argument("--outdir", default=str(HERE / "out" / "review"))
a = ap.parse_args(argv)
out = pathlib.Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
W = np.genfromtxt(ROOT / "measurement/weight_track.csv", delimiter=",", names=True)
sc = bpy.context.scene
sc.render.resolution_x, sc.render.resolution_y = 1920, 1088
sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_depth = "8"
info = {"tag": a.tag, "blend": pathlib.Path(bpy.data.filepath).name, "engine": sc.render.engine, "view_transform": sc.view_settings.view_transform,
        "look": sc.view_settings.look, "motion_blur": sc.render.use_motion_blur, "frames": {}}


def render(path, border=None, percent=100):
    sc.render.resolution_percentage = percent
    sc.render.use_border = sc.render.use_crop_to_border = border is not None
    if border:
        u0, v0, u1, v1 = border
        sc.render.border_min_x, sc.render.border_max_x = u0 / 1920, u1 / 1920
        sc.render.border_min_y, sc.render.border_max_y = 1 - v1 / 1088, 1 - v0 / 1088
    sc.render.filepath = str(path)
    bpy.ops.render.render(write_still=True, scene=sc.name)


for fs in a.frames.split(","):
    n = int(fs); sc.frame_set(n)
    u, v = W["u_px"][n], W["v_px"][n]
    crops = {"hook_spring": (820, 20, 1100, 560), "post": (470, 150, 690, 650),
             "weight": (int(u) - 170, int(v) - 200, int(u) + 170, int(v) + 130)}
    rec = {"files": {}, "crops": crops}
    f = out / f"{a.tag}_f{n:03d}_full.png"; render(f, None, 50); rec["files"]["full"] = str(f)
    for k, b in ({} if a.no_crops else crops).items():
        f = out / f"{a.tag}_f{n:03d}_{k}.png"; render(f, b, 100); rec["files"][k] = str(f)
    info["frames"][str(n)] = rec
(out / f"{a.tag}_review.json").write_text(json.dumps(info, indent=1))
print("VSP_REVIEW_RENDER", a.tag)
