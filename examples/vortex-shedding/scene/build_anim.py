"""Animated scene (frames 0-240 @ 24 fps): frame-0 scene of build.py + dye sheet driven by the solver's per-frame tracer images.

Usage (always under the render lock):
  ../_tools/with_render_lock.sh blender -b --factory-startup --python scene/build_anim.py -- \
      --dye-dir <dir with f0000.png..f0240.png> [--rebuild-base] [--render-frames 0,60,...] [--render-dir DIR]
      [--scale 0.5] [--samples 32] [--hide-dye] [--ff-dye] [--prefix NAME_] [--no-save]
Base scene: scene/out/vortex_ff.blend, i.e. exactly what scene/build.py builds for frame 0 (camera, look, lights, render
settings). --rebuild-base re-runs build.py first (in-process, same params; deterministic) instead of loading the saved file.
Dye-image contract (CONTRACT.md): 16-bit RGB, R,G,B = tracer 0,1,2 concentration 0..1; x [-0.30,0.30], y [-0.06,0.06],
0.25 mm/px, row 0 = +y edge, col 0 = -x edge.  Look mapping (same shader structure as build.py v3 frame-0 dye):
  colour = sum_k c_k * dye_palette_linear[k] / sum_k c_k ;  alpha = peak_alpha * (1 - exp(-kappa*sum c)) / (1 - exp(-kappa))
  (clamped 0.97; kappa = --dye-kappa, default 3 ~ frame-0's alpha ~ sqrt(sigma0/sigma) streak fall-off for c ~ sigma0/sigma).
Water micro-ripple: translates downstream at U = 0.015 m/s (look only; declared).
Writes scene/out/vortex_anim.blend (unless --no-save) and scene/out/vortex_anim_state.json (camera + resolved texture frames).
"""
import bpy, json, math, os, sys, runpy, hashlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "scene", "out")
P = json.load(open(os.path.join(ROOT, "config", "scene_params.json")))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def arg(name, default=None, cast=str):
    return cast(argv[argv.index(name) + 1]) if name in argv else default
flag = lambda n: n in argv
DYE_DIR = os.path.abspath(arg("--dye-dir"))
KAPPA = arg("--dye-kappa", 3.0, float)
FRAMES = [int(f) for f in arg("--render-frames", "").split(",") if f != ""]
RDIR = os.path.abspath(arg("--render-dir", os.path.join(OUT, "anim_renders")))
SCALE = arg("--scale", None, float); SAMPLES = arg("--samples", None, int); PREFIX = arg("--prefix", "")
NF, FPS, U = 241, 24, 0.015
assert os.path.isfile(os.path.join(DYE_DIR, "f0000.png")) and os.path.isfile(os.path.join(DYE_DIR, "f0240.png")), DYE_DIR

# ---------------------------------------------------------------- base = build.py frame-0 scene
BASE = os.path.join(OUT, "vortex_ff.blend")
if flag("--rebuild-base"):
    sys.argv = [sys.argv[0], "--"]                       # build.py without --render: rebuilds + saves vortex_ff.blend
    runpy.run_path(os.path.join(ROOT, "scene", "build.py"), run_name="__main__")
bpy.ops.wm.open_mainfile(filepath=BASE)
base_sha = hashlib.sha256(open(BASE, "rb").read()).hexdigest()
sc = bpy.context.scene
ch = P["channel"]; hw = ch["inner_width"] / 2; dy = P["dye"]; V = P["v3"]
assert P.get("style") == "cinematic34"

# Metal GPU devices are a preference (not stored in the .blend) -> re-enable as build.py does
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "METAL"; prefs.refresh_devices()
for d in prefs.devices: d.use = True
sc.cycles.device = "GPU"

# ---------------------------------------------------------------- dye sheet driven by the tracer image sequence
if not flag("--ff-dye"):
    old = bpy.data.objects["PND_DyeSheet"]; bpy.data.objects.remove(old)
    X0, X1, Y0, Y1 = -0.30, 0.30, -0.06, 0.06                  # texture extent (contract)
    ys0, ys1 = -hw + 0.0005, hw - 0.0005                         # sheet confined inside the channel walls (as build.py)
    me = bpy.data.meshes.new("PND_DyeSheetAnim")
    verts = [(X0, ys0, dy["z_sheet"]), (X1, ys0, dy["z_sheet"]), (X1, ys1, dy["z_sheet"]), (X0, ys1, dy["z_sheet"])]
    me.from_pydata(verts, [], [(0, 1, 2, 3)]); me.update()
    uvl = me.uv_layers.new(name="UVMap")
    for li, lp in enumerate(me.loops):                           # exact world->UV: u=(x-X0)/(X1-X0), v=(y-Y0)/(Y1-Y0)
        x, y, _ = verts[lp.vertex_index]; uvl.data[li].uv = ((x - X0) / (X1 - X0), (y - Y0) / (Y1 - Y0))
    ds = bpy.data.objects.new("PND_DyeSheetAnim", me); sc.collection.objects.link(ds)

    img = bpy.data.images.load(os.path.join(DYE_DIR, "f0000.png"), check_existing=False)
    img.name = "PND_DyeTracers"; img.source = "SEQUENCE"; img.colorspace_settings.name = "Non-Color"
    m = bpy.data.materials.new("PND_DyeAnim"); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    N = lambda t: nt.nodes.new(t); L = nt.links.new
    out = N("ShaderNodeOutputMaterial")
    uvn = N("ShaderNodeUVMap"); uvn.uv_map = "UVMap"
    tx = N("ShaderNodeTexImage"); tx.image = img; tx.interpolation = "Linear"; tx.extension = "CLIP"
    iu = tx.image_user; iu.frame_duration = NF; iu.frame_start = 0; iu.frame_offset = -1  # scene frame n -> f{n:04d}.png
    iu.use_auto_refresh = True; iu.use_cyclic = False
    L(uvn.outputs["UV"], tx.inputs["Vector"])
    sep = N("ShaderNodeSeparateColor"); L(tx.outputs["Color"], sep.inputs[0])
    def math_(op, a, b=None, bv=None):
        n = N("ShaderNodeMath"); n.operation = op
        L(a, n.inputs[0]) if not isinstance(a, float) else n.inputs[0].__setattr__("default_value", a)
        if b is not None: L(b, n.inputs[1])
        if bv is not None: n.inputs[1].default_value = bv
        return n.outputs[0]
    s01 = math_("ADD", sep.outputs[0], sep.outputs[1]); csum = math_("ADD", s01, sep.outputs[2])
    e = math_("EXPONENT", math_("MULTIPLY", csum, bv=-KAPPA))
    alpha = math_("MULTIPLY", math_("SUBTRACT", 1.0, e), bv=dy["peak_alpha"] / (1 - math.exp(-KAPPA)))
    alpha = math_("MINIMUM", alpha, bv=0.97)
    pal = V["dye_palette_linear"]
    acc = None
    for k in range(3):
        vs = N("ShaderNodeVectorMath"); vs.operation = "SCALE"; vs.inputs[0].default_value = pal[k]
        L(sep.outputs[k], vs.inputs["Scale"])
        if acc is None: acc = vs.outputs[0]
        else:
            va = N("ShaderNodeVectorMath"); va.operation = "ADD"; L(acc, va.inputs[0]); L(vs.outputs[0], va.inputs[1]); acc = va.outputs[0]
    inv = math_("DIVIDE", 1.0, math_("MAXIMUM", csum, bv=1e-4))
    vn = N("ShaderNodeVectorMath"); vn.operation = "SCALE"; L(acc, vn.inputs[0]); L(inv, vn.inputs["Scale"])
    col = vn.outputs[0]
    # same shader structure as build.py (v3): alpha-mix(transparent, mix(transparent(col), principled(col, rough 0.6)))
    tr = N("ShaderNodeBsdfTransparent"); tr.inputs[0].default_value = (1, 1, 1, 1)
    pb = N("ShaderNodeBsdfPrincipled"); pb.inputs["Roughness"].default_value = 0.6; L(col, pb.inputs["Base Color"])
    tt = N("ShaderNodeBsdfTransparent"); L(col, tt.inputs[0])
    mm = N("ShaderNodeMixShader"); mm.inputs[0].default_value = 1 - V["dye_shader_transparent_mix"]
    L(tt.outputs[0], mm.inputs[1]); L(pb.outputs[0], mm.inputs[2])
    m2 = N("ShaderNodeMixShader"); L(alpha, m2.inputs[0]); L(tr.outputs[0], m2.inputs[1]); L(mm.outputs[0], m2.inputs[2])
    L(m2.outputs[0], out.inputs[0])
    ds.data.materials.append(m)
    if flag("--hide-dye"):
        ds.hide_render = True
else:
    img = None
    if flag("--hide-dye"): bpy.data.objects["PND_DyeSheet"].hide_render = True

# ---------------------------------------------------------------- water micro-ripple translates at U (look only)
wm_ = bpy.data.materials["PND_Water"]; wnt = wm_.node_tree
nz = [n for n in wnt.nodes if n.type == "TEX_NOISE"][0]
src = nz.inputs["Vector"].links[0].from_socket
mp = wnt.nodes.new("ShaderNodeMapping"); mp.name = "RippleAdvect"
wnt.links.new(src, mp.inputs["Vector"]); wnt.links.new(mp.outputs[0], nz.inputs["Vector"])
loc = mp.inputs["Location"]
loc.default_value = (0, 0, 0); loc.keyframe_insert("default_value", frame=0)
loc.default_value = (-U * 240 / FPS, 0, 0); loc.keyframe_insert("default_value", frame=240)   # sample at p - U t
for fc in wnt.animation_data.action.fcurves if hasattr(wnt.animation_data.action, "fcurves") else []:
    for kp in fc.keyframe_points: kp.interpolation = "LINEAR"
    fc.extrapolation = "LINEAR"
try:   # Blender 4.4+/5.x layered actions
    for lay in wnt.animation_data.action.layers:
        for st in lay.strips:
            for cb in st.channelbags:
                for fc in cb.fcurves:
                    for kp in fc.keyframe_points: kp.interpolation = "LINEAR"
                    fc.extrapolation = "LINEAR"
except AttributeError:
    pass

# ---------------------------------------------------------------- timeline
sc.render.fps = FPS; sc.render.fps_base = 1.0
sc.frame_start, sc.frame_end = 0, NF - 1
sc.frame_set(0)

# ---------------------------------------------------------------- state export for the checks
co = sc.camera; cd = co.data
state = {"base_blend": BASE, "base_sha256": base_sha, "dye_dir": DYE_DIR, "kappa": KAPPA, "ff_dye": flag("--ff-dye"),
         "camera": {"matrix_world": [list(r) for r in co.matrix_world], "lens_mm": cd.lens, "sensor_width_mm": cd.sensor_width,
                    "sensor_fit": cd.sensor_fit, "shift": [cd.shift_x, cd.shift_y], "res": [sc.render.resolution_x, sc.render.resolution_y],
                    "dof": [cd.dof.use_dof, cd.dof.aperture_fstop, cd.dof.focus_distance]},
         "frames": [sc.frame_start, sc.frame_end], "fps": sc.render.fps}
if img is not None:
    resolved = {}
    for f in (0, 1, 60, 120, 240):
        sc.frame_set(f); bpy.context.view_layer.update()
        tx.image_user.frame_current = f
        resolved[f] = os.path.basename(img.filepath_from_user(image_user=tx.image_user))
    state["texture_file_for_scene_frame"] = resolved
    o = bpy.data.objects["PND_DyeSheetAnim"]; bb = [o.matrix_world @ v.co for v in o.data.vertices]
    state["dye_sheet_bounds"] = {"x": [min(p.x for p in bb), max(p.x for p in bb)], "y": [min(p.y for p in bb), max(p.y for p in bb)],
                                 "z": [min(p.z for p in bb), max(p.z for p in bb)]}
    sc.frame_set(0)
json.dump(state, open(os.path.join(OUT, ("vortex_anim_state.json" if not flag("--no-save") else PREFIX + "vortex_anim_state.json")), "w"), indent=1)
if not flag("--no-save"):
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "vortex_anim.blend"), compress=True, relative_remap=False)
print("BUILD_ANIM_OK", json.dumps(state.get("texture_file_for_scene_frame")))

# ---------------------------------------------------------------- optional renders (preview overrides applied after the save)
if SCALE is not None: sc.render.resolution_percentage = int(round(100 * SCALE))
if SAMPLES is not None: sc.cycles.samples = SAMPLES
os.makedirs(RDIR, exist_ok=True)
for f in FRAMES:
    sc.frame_set(f)
    sc.render.filepath = os.path.join(RDIR, f"{PREFIX}f{f:03d}.png")
    bpy.ops.render.render(write_still=True)
    print("RENDER_OK", sc.render.filepath)
