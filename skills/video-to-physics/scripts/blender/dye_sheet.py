"""Dye sheet: a plane at height z textured with the tracer PNG sequence written by v2p.dye (Blender 5.2+).

  blender -b --factory-startup --python dye_sheet.py -- --dye-dir DIR --out OUT.blend [--blend IN.blend]
          [--z 0.0] [--extent x0,x1,y0,y1] [--palette "r,g,b;r,g,b;..."] [--kappa 3.0] [--peak-alpha 0.95]
          [--frame-offset 0] [--name V2P_DyeSheet] [--json REPORT.json]

Contract (meta.json from v2p.dye): 16-bit RGB, channel k = tracer 3g + k of group g (group 0 in DIR, group g in
DIR/g<g>), plane extent + px, pixel centres x_i = x0 + (i + 0.5) px, row 0 at +y. The sheet spans exactly the
extent with the world -> UV map u = (x - x0)/(x1 - x0), v = (y - y0)/(y1 - y0) (image row 0 sits at v = 1), images
are Non-Color, Linear interpolation, CLIP extension. Scene frame n shows f{n - frame_offset:04d}.png.
Shading: colour = sum_k c_k palette_k / sum_k c_k, alpha = peak_alpha (1 - exp(-kappa sum c)) / (1 - exp(-kappa))
(clamped to peak_alpha), mixed Transparent <-> Principled. The resolved frame -> file map is verified and reported.
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

DEFAULT_PALETTE = [(0.80, 0.05, 0.05), (0.05, 0.25, 0.85), (0.95, 0.65, 0.02), (0.05, 0.6, 0.15),
                   (0.55, 0.1, 0.7), (0.0, 0.0, 0.0)]


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--dye-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--blend", default=None)
    p.add_argument("--z", type=float, default=0.0)
    p.add_argument("--extent", default=None, help="x0,x1,y0,y1 (default: meta.json plane)")
    p.add_argument("--palette", default=None, help="linear RGB per tracer, 'r,g,b;r,g,b;...'")
    p.add_argument("--kappa", type=float, default=3.0)
    p.add_argument("--peak-alpha", type=float, default=0.95)
    p.add_argument("--frame-offset", type=int, default=0, help="scene frame n <-> f{n - offset}.png")
    p.add_argument("--name", default="V2P_DyeSheet")
    p.add_argument("--json", default=None)
    return p.parse_args(argv)


def _sequence(d):
    files = sorted(glob.glob(os.path.join(d, "f[0-9][0-9][0-9][0-9].png")))
    if not files:
        raise SystemExit(f"no fNNNN.png in {d}")
    nums = [int(re.findall(r"f(\d+)\.png$", f)[0]) for f in files]
    if nums != list(range(nums[0], nums[0] + len(nums))):
        raise SystemExit(f"non-contiguous sequence in {d}")
    return files, nums[0]


def main():
    import bpy
    a = _args()
    if a.blend:
        bpy.ops.wm.open_mainfile(filepath=a.blend)
    sc = bpy.context.scene
    d = Path(a.dye_dir).resolve()
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    if a.extent:
        x0, x1, y0, y1 = map(float, a.extent.split(","))
    else:
        pl = meta["plane"]
        x0, x1, y0, y1 = pl["x0"], pl["x1"], pl["y0"], pl["y1"]
    n_tr = len(meta.get("tracers", [])) or 3
    pal = ([tuple(map(float, c.split(","))) for c in a.palette.split(";")] if a.palette else DEFAULT_PALETTE)
    pal = (pal * (1 + n_tr // len(pal)))[:n_tr]
    groups = [d] + [d / f"g{g}" for g in range(1, (n_tr + 2) // 3)]

    me = bpy.data.meshes.new(a.name)
    verts = [(x0, y0, a.z), (x1, y0, a.z), (x1, y1, a.z), (x0, y1, a.z)]
    me.from_pydata(verts, [], [(0, 1, 2, 3)])
    me.update()
    uv = me.uv_layers.new(name="UVMap")
    for li, lp in enumerate(me.loops):
        x, y, _ = verts[lp.vertex_index]
        uv.data[li].uv = ((x - x0) / (x1 - x0), (y - y0) / (y1 - y0))
    ob = bpy.data.objects.new(a.name, me)
    sc.collection.objects.link(ob)

    m = bpy.data.materials.new(a.name + "_Mat")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    new, link = nt.nodes.new, nt.links.new

    def math_node(op, x, y):
        n = new("ShaderNodeMath")
        n.operation = op
        for i, v in enumerate((x, y)):
            link(v, n.inputs[i]) if not isinstance(v, (int, float)) else setattr(n.inputs[i], "default_value", v)
        return n.outputs[0]

    def vec_node(op, x, y=None, scale=None):
        n = new("ShaderNodeVectorMath")
        n.operation = op
        link(x, n.inputs[0]) if not isinstance(x, tuple) else setattr(n.inputs[0], "default_value", x)
        if y is not None:
            link(y, n.inputs[1])
        if scale is not None:
            link(scale, n.inputs["Scale"])
        return n.outputs[0]

    uvn = new("ShaderNodeUVMap")
    uvn.uv_map = "UVMap"
    csum, col, users, k = None, None, [], 0
    for g, gd in enumerate(groups):
        files, first = _sequence(gd)
        img = bpy.data.images.load(files[0], check_existing=False)
        img.name = f"{a.name}_g{g}"
        img.source = "SEQUENCE"
        img.colorspace_settings.name = "Non-Color"
        tx = new("ShaderNodeTexImage")
        tx.image, tx.interpolation, tx.extension = img, "Linear", "CLIP"
        iu = tx.image_user
        iu.frame_duration = len(files)
        iu.frame_start = a.frame_offset + first        # Blender: file number = clamp(n - start + 1, 1, dur) + offset
        iu.frame_offset = first - 1
        iu.use_auto_refresh, iu.use_cyclic = True, False
        users.append((tx, img, gd))
        link(uvn.outputs["UV"], tx.inputs["Vector"])
        sep = new("ShaderNodeSeparateColor")
        link(tx.outputs["Color"], sep.inputs[0])
        for ch in range(3):
            if k >= n_tr:
                break
            csum = sep.outputs[ch] if csum is None else math_node("ADD", csum, sep.outputs[ch])
            term = vec_node("SCALE", tuple(pal[k]), scale=sep.outputs[ch])
            col = term if col is None else vec_node("ADD", col, term)
            k += 1
    color = vec_node("SCALE", col, scale=math_node("DIVIDE", 1.0, math_node("MAXIMUM", csum, 1e-4)))
    expo = math_node("EXPONENT", math_node("MULTIPLY", csum, -a.kappa), 0.0)
    alpha = math_node("MULTIPLY", math_node("SUBTRACT", 1.0, expo), a.peak_alpha / (1.0 - math.exp(-a.kappa)))
    alpha = math_node("MINIMUM", alpha, a.peak_alpha)
    pb = new("ShaderNodeBsdfPrincipled")
    pb.inputs["Roughness"].default_value = 0.6
    link(color, pb.inputs["Base Color"])
    tr = new("ShaderNodeBsdfTransparent")
    mix = new("ShaderNodeMixShader")
    link(alpha, mix.inputs[0])
    link(tr.outputs[0], mix.inputs[1])
    link(pb.outputs[0], mix.inputs[2])
    out = new("ShaderNodeOutputMaterial")
    link(mix.outputs[0], out.inputs[0])
    me.materials.append(m)

    # verify the frame -> file mapping on the first group
    tx, img, gd = users[0]
    files, first = _sequence(gd)
    probe = sorted({a.frame_offset + first, a.frame_offset + first + len(files) // 2, a.frame_offset + first + len(files) - 1})
    resolved = {}
    for f in probe:                    # read the depsgraph-evaluated image user (originals are not updated in -b)
        sc.frame_set(f)
        ev = m.evaluated_get(bpy.context.evaluated_depsgraph_get())
        etx = ev.node_tree.nodes[tx.name]
        resolved[f] = os.path.basename(etx.image.filepath_from_user(image_user=etx.image_user))
    ok = all(resolved[f] == f"f{f - a.frame_offset:04d}.png" for f in probe)
    sc.frame_start = a.frame_offset + first
    sc.frame_end = a.frame_offset + first + len(files) - 1
    Path(a.out).resolve().parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(a.out).resolve()))
    rep = dict(object=ob.name, extent=[x0, x1, y0, y1], z=a.z, tracers=n_tr, groups=len(groups), kappa=a.kappa,
               frames=[sc.frame_start, sc.frame_end], resolved={str(k_): v for k_, v in resolved.items()}, mapping_ok=ok)
    print("V2P_DYE_SHEET " + json.dumps(rep))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
