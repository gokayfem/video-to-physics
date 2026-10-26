"""Motion-blur shutter tuning, render side (VR-05). Run inside Blender on the saved scene:
blender -b scene/out/pendulum.blend --python scene/measure/mb_tune_render.py -- --outdir scene/out/preview/mb --shutters 0,0.3,0.5,0.7
1) bob pixel track (u, v, r) for frames 0..240 through the scene camera (world_to_camera_view, contract pixels);
2) picks the 6 fastest, 4 slowest and every 6th frame in 1..96 (t <= 4 s, where the reference amplitude is still sane);
3) renders a full-resolution border crop around the bob at each frame for each shutter (frozen look kept).
Writes track.csv, crops/<frame>_<shutter>.png and crops.json (crop origins)."""
import sys, json, math, pathlib, argparse, bpy, numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view as w2c
argv = sys.argv[sys.argv.index("--") + 1:]
ap = argparse.ArgumentParser(); ap.add_argument("--outdir", required=True); ap.add_argument("--shutters", default="0,0.3,0.5,0.7")
a = ap.parse_args(argv)
out = pathlib.Path(a.outdir); (out / "crops").mkdir(parents=True, exist_ok=True)
sc = bpy.context.scene
if bpy.context.window: bpy.context.window.scene = sc
bob = bpy.data.objects["PND_Bob"]; r_m = float(bob["pnd_radius_m"])
rows = []
for n in range(241):
    sc.frame_set(n)
    c = bob.matrix_world.translation
    q = w2c(sc, sc.camera, c)
    cam = sc.camera.matrix_world.inverted() @ c
    f_px = sc.camera.data.lens / sc.camera.data.sensor_width * 1920
    rows.append((n, q.x * 1920, (1 - q.y) * 1088, f_px * r_m / -cam.z))
T = np.array(rows)
sp = np.abs(np.gradient(T[:, 1]))
np.savetxt(out / "track.csv", np.c_[T, sp], delimiter=",", header="frame,u_px,v_px,radius_px,speed_px_per_frame", comments="")
win = np.arange(1, 97)
fast = list(map(int, win[np.argsort(-sp[win])][:6])); slow = list(map(int, win[np.argsort(sp[win])][:4]))
extra = [n for n in range(3, 97, 6) if n not in fast + slow]          # spread over all speeds for a slope fit
crops = {}
sc.render.resolution_percentage = 100
sc.render.use_border = sc.render.use_crop_to_border = True
for n in sorted(fast + slow + extra):
    u, v, r = T[n, 1], T[n, 2], T[n, 3]
    u0, u1, v0, v1 = int(u - r - 40), int(u + r + 40), int(v - r - 20), int(v + r + 20)
    crops[n] = [u0, v0, u1, v1, "fast" if n in fast else ("slow" if n in slow else "mid"), float(sp[n])]
    sc.render.border_min_x, sc.render.border_max_x = u0 / 1920, u1 / 1920
    sc.render.border_min_y, sc.render.border_max_y = 1 - v1 / 1088, 1 - v0 / 1088
    for s in [float(x) for x in a.shutters.split(",")]:
        sc.render.use_motion_blur = s > 0
        sc.render.motion_blur_shutter = max(s, 0.01); sc.render.motion_blur_position = "CENTER"
        sc.frame_set(n)
        sc.render.filepath = str(out / "crops" / f"{n:03d}_{s:.2f}.png")
        bpy.ops.render.render(write_still=True, scene=sc.name)
json.dump({"crops": crops, "fast": fast, "slow": slow, "shutters": a.shutters}, open(out / "crops.json", "w"), indent=1)
print("MB_TUNE", fast, slow)
