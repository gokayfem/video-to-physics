"""Preview render of an opened .blend (PNG sequence f0000.png, f0001.png, ...).

usage: blender -b SCENE.blend --factory-startup --python render_preview.py -- --out DIR
           [--pct 50] [--engine EEVEE|WORKBENCH|CYCLES] [--samples 16] [--camera NAME]
           [--frames START:END | --frames 0,60,120] [--res WxH]
Defaults keep previews cheap (50 % resolution, 16 samples). Frame n is written as f%04d.png with its own
frame number, so previews line up with the reference at t = n / fps. Run under `v2p.py lock --`.
"""
import argparse
import os
import sys

import bpy

ENGINES = {"EEVEE": "BLENDER_EEVEE", "EEVEE_NEXT": "BLENDER_EEVEE_NEXT", "WORKBENCH": "BLENDER_WORKBENCH",
           "CYCLES": "CYCLES"}


def parse():
    ap = argparse.ArgumentParser(prog="render_preview.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pct", type=int, default=50)
    ap.add_argument("--engine", default="EEVEE")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--camera")
    ap.add_argument("--frames", help="START:END (inclusive) or a comma list; default scene range")
    ap.add_argument("--res", help="WxH (default: keep the scene's resolution)")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return ap.parse_args(argv)


def set_engine(scene, name, samples):
    eng = ENGINES.get(name.upper(), name)
    try:
        scene.render.engine = eng
    except TypeError:                       # engine id renamed between Blender versions
        scene.render.engine = "BLENDER_EEVEE_NEXT" if eng == "BLENDER_EEVEE" else "BLENDER_EEVEE"
    if scene.render.engine.startswith("BLENDER_EEVEE"):
        scene.eevee.taa_render_samples = samples
    elif scene.render.engine == "CYCLES":
        scene.cycles.samples = samples


def frame_list(scene, spec):
    if not spec:
        return list(range(scene.frame_start, scene.frame_end + 1))
    if ":" in spec:
        a, b = spec.split(":")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def main():
    a = parse()
    s = bpy.context.scene
    if a.camera:
        if a.camera not in bpy.data.objects:
            sys.exit(f"camera {a.camera!r} not found")
        s.camera = bpy.data.objects[a.camera]
    set_engine(s, a.engine, a.samples)
    if a.res:
        w, h = (int(v) for v in a.res.lower().split("x"))
        s.render.resolution_x, s.render.resolution_y = w, h
    s.render.resolution_percentage = a.pct
    s.render.image_settings.file_format = "PNG"
    os.makedirs(a.out, exist_ok=True)
    for f in frame_list(s, a.frames):
        s.frame_set(f)
        s.render.filepath = os.path.join(a.out, f"f{f:04d}.png")
        bpy.ops.render.render(write_still=True)
    print(f"rendered {a.out}")


main()
