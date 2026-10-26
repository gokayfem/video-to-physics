"""Deterministic, params-driven first-frame scene: vortex shedding channel (batch-2 concept B).

Usage:
  blender -b --factory-startup --python scene/build.py -- [--render OUT.png] [--scale 0.5] [--samples N]
Reads config/scene_params.json, writes scene/out/vortex_ff.blend, scene/out/dye_ff.png,
config/camera_ff.json (analytic camera; render-based projection test is scene/check_camera.py).
"""
import bpy, json, math, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = json.load(open(os.path.join(ROOT, "config", "scene_params.json")))
OUT = os.path.join(ROOT, "scene", "out")
os.makedirs(OUT, exist_ok=True)

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def arg(name, default=None, cast=str):
    return cast(argv[argv.index(name) + 1]) if name in argv else default
RENDER_OUT = arg("--render")
SCALE = arg("--scale", 1.0, float)
SAMPLES = arg("--samples", P["render"]["samples"], int)

# ---------------------------------------------------------------- reset
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.unit_settings.system = "METRIC"; sc.unit_settings.scale_length = 1.0

def box(name, cx, cy, cz, sx, sy, sz, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, cz))
    o = bpy.context.object; o.name = name; o.scale = (sx, sy, sz)
    bpy.ops.object.transform_apply(scale=True)
    o.data.materials.append(mat); return o

def cyl(name, r, depth, loc, rot=(0, 0, 0), mat=None, verts=64):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=depth, location=loc, rotation=rot)
    o = bpy.context.object; o.name = name
    bpy.ops.object.shade_smooth()
    if mat: o.data.materials.append(mat)
    return o

def new_mat(name):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); return m, nt, out

def principled(name, color, rough, metal=0.0):
    m, nt, out = new_mat(name)
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = (*color, 1); b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    nt.links.new(b.outputs[0], out.inputs[0]); return m

def clear_dielectric(name, ior, rough=0.0, scuff=0.0, ripple=0.0):
    """Glass that is invisible to shadow rays (no caustics needed for light to reach the plate)."""
    m, nt, out = new_mat(name)
    g = nt.nodes.new("ShaderNodeBsdfGlass"); g.inputs["IOR"].default_value = ior
    g.inputs["Roughness"].default_value = rough; g.inputs["Color"].default_value = (1, 1, 1, 1)
    t = nt.nodes.new("ShaderNodeBsdfTransparent")
    lp = nt.nodes.new("ShaderNodeLightPath"); mx = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Shadow Ray"], mx.inputs[0])
    nt.links.new(g.outputs[0], mx.inputs[1]); nt.links.new(t.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs[0])
    if scuff or ripple:
        tcn = nt.nodes.new("ShaderNodeTexCoord")
    if scuff:   # faint scuffs: streaky roughness variation (stretched noise along x)
        mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (40, 900, 900)
        nz = nt.nodes.new("ShaderNodeTexNoise"); nz.inputs["Detail"].default_value = 6
        nt.links.new(tcn.outputs["Object"], mp.inputs[0]); nt.links.new(mp.outputs[0], nz.inputs["Vector"])
        mr = nt.nodes.new("ShaderNodeMapRange"); nt.links.new(nz.outputs["Fac"], mr.inputs[0])
        mr.inputs[1].default_value = 0.55; mr.inputs[2].default_value = 0.8
        mr.inputs[3].default_value = rough; mr.inputs[4].default_value = rough + scuff
        nt.links.new(mr.outputs[0], g.inputs["Roughness"])
    if ripple:  # very subtle micro-ripple on the water surface
        nz = nt.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 60; nz.inputs["Detail"].default_value = 3
        nt.links.new(tcn.outputs["Object"], nz.inputs["Vector"])
        bm = nt.nodes.new("ShaderNodeBump"); bm.inputs["Strength"].default_value = ripple; bm.inputs["Distance"].default_value = 0.0005
        nt.links.new(nz.outputs["Fac"], bm.inputs["Height"]); nt.links.new(bm.outputs[0], g.inputs["Normal"])
    return m

# ---------------------------------------------------------------- grid plate (z=0 top)
pl = P["plate"]
m, nt, out = new_mat("PND_GridPlate")
tc = nt.nodes.new("ShaderNodeTexCoord"); sep = nt.nodes.new("ShaderNodeSeparateXYZ")
nt.links.new(tc.outputs["Object"], sep.inputs[0])
def line_mask(comp, origin):
    # distance (m) to nearest grid line along one axis -> 1 inside line
    a = nt.nodes.new("ShaderNodeMath"); a.operation = "SUBTRACT"; nt.links.new(sep.outputs[comp], a.inputs[0]); a.inputs[1].default_value = origin
    w = nt.nodes.new("ShaderNodeMath"); w.operation = "PINGPONG"; nt.links.new(a.outputs[0], w.inputs[0]); w.inputs[1].default_value = pl["grid_pitch"] / 2
    # pingpong(x, p/2) = distance to nearest multiple of p
    lt = nt.nodes.new("ShaderNodeMath"); lt.operation = "LESS_THAN"; nt.links.new(w.outputs[0], lt.inputs[0]); lt.inputs[1].default_value = pl["grid_line_width"] / 2
    return lt
a0 = pl["albedo"]; tint = pl.get("tint", [1, 1, 1])
b = nt.nodes.new("ShaderNodeBsdfPrincipled"); b.inputs["Roughness"].default_value = 0.85
if pl.get("grid_enabled", True):
    lx = line_mask("X", pl["grid_origin"][0]); ly = line_mask("Y", pl["grid_origin"][1])
    mx_ = nt.nodes.new("ShaderNodeMath"); mx_.operation = "MAXIMUM"
    nt.links.new(lx.outputs[0], mx_.inputs[0]); nt.links.new(ly.outputs[0], mx_.inputs[1])
    mixc = nt.nodes.new("ShaderNodeMix"); mixc.data_type = "RGBA"
    nt.links.new(mx_.outputs[0], mixc.inputs["Factor"])
    a1 = pl["grid_ink_albedo"]
    mixc.inputs[6].default_value = (a0, a0, a0, 1); mixc.inputs[7].default_value = (a1, a1, a1, 1)
    nt.links.new(mixc.outputs[2], b.inputs["Base Color"])
else:
    # plain warm matte white with subtle real-world mottling + fine paper/paint tooth
    n1 = nt.nodes.new("ShaderNodeTexNoise"); n1.inputs["Scale"].default_value = 18.0; n1.inputs["Detail"].default_value = 4
    n2 = nt.nodes.new("ShaderNodeTexNoise"); n2.inputs["Scale"].default_value = 900.0; n2.inputs["Detail"].default_value = 2
    for n_ in (n1, n2): nt.links.new(tc.outputs["Object"], n_.inputs["Vector"])
    mr = nt.nodes.new("ShaderNodeMapRange"); nt.links.new(n1.outputs["Fac"], mr.inputs[0])
    mr.inputs[3].default_value = a0 * 0.96; mr.inputs[4].default_value = a0 * 1.03
    mr2 = nt.nodes.new("ShaderNodeMapRange"); nt.links.new(n2.outputs["Fac"], mr2.inputs[0])
    mr2.inputs[3].default_value = 0.985; mr2.inputs[4].default_value = 1.012
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"
    nt.links.new(mr.outputs[0], mul.inputs[0]); nt.links.new(mr2.outputs[0], mul.inputs[1])
    tn = nt.nodes.new("ShaderNodeMix"); tn.data_type = "RGBA"; tn.blend_type = "MULTIPLY"; tn.inputs["Factor"].default_value = 1
    nt.links.new(mul.outputs[0], tn.inputs[6]); tn.inputs[7].default_value = (*tint, 1)
    nt.links.new(tn.outputs[2], b.inputs["Base Color"])
    bm = nt.nodes.new("ShaderNodeBump"); bm.inputs["Strength"].default_value = 0.05; bm.inputs["Distance"].default_value = 0.0002
    nt.links.new(n2.outputs["Fac"], bm.inputs["Height"]); nt.links.new(bm.outputs[0], b.inputs["Normal"])
nt.links.new(b.outputs[0], out.inputs[0])
plate = box("PND_Plate", 0, 0, -pl["thickness"] / 2, pl["size_x"], pl["size_y"], pl["thickness"], m)
# object coords == world coords (box at origin x/y, scale applied) -> grid lines at world x,y = k*pitch

# ---------------------------------------------------------------- channel walls (acrylic)
ch = P["channel"]; wa = P["water"]
acryl = clear_dielectric("PND_Acrylic", 1.49, 0.0, scuff=P["look"]["acrylic_scuff_rough"])
hw = ch["inner_width"] / 2; t = ch["wall_thickness"]
for s in (-1, 1):
    box(f"PND_Wall_{'L' if s < 0 else 'R'}", 0, s * (hw + t / 2), ch["wall_height"] / 2,
        ch["length"], t, ch["wall_height"], acryl).location.z += ch["wall_z_gap"]  # avoid coplanar faces with the plate
    bv = bpy.context.object.modifiers.new("bevel", "BEVEL"); bv.width = P["look"]["acrylic_bevel"]; bv.segments = 3

# ---------------------------------------------------------------- water surface (single interface)
water = clear_dielectric("PND_Water", wa["ior"], 0.0, ripple=P["look"]["water_ripple_strength"])
bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, wa["surface_z"]))
w = bpy.context.object; w.name = "PND_WaterSurface"; w.scale = (ch["length"], ch["inner_width"] - 0.0001, 1)
bpy.ops.object.transform_apply(scale=True); w.data.materials.append(water)

# ---------------------------------------------------------------- cylinder
cy = P["cylinder"]
black = principled("PND_MatteBlack", (0.012, 0.012, 0.012), 0.75)
cyl("PND_Cylinder", cy["D"] / 2, cy["z_top"] - cy["z_bottom"],
    (cy["center"][0], cy["center"][1], (cy["z_top"] + cy["z_bottom"]) / 2), mat=black, verts=128)

# ---------------------------------------------------------------- injectors + rake
inj = P["injectors"]; rk = inj["rake_bar"]
steel = principled("PND_Steel", (0.62, 0.62, 0.63), 0.28, 1.0)
alu = principled("PND_RakeAlu", (0.55, 0.56, 0.58), 0.45, 1.0)
box("PND_RakeBar", rk["x_center"], 0, rk["z_bottom"] + rk["thickness"] / 2,
    rk["width_x"], ch["inner_width"] + 2 * t + 0.01, rk["thickness"], alu)
r = inj["needle_od"] / 2
for i, y in enumerate(inj["y"]):
    zt = inj["z_tip"]; zb = rk["z_bottom"]
    cyl(f"PND_NeedleV_{i}", r, zb - zt, (inj["x_vertical"], y, (zb + zt) / 2), mat=steel, verts=16)
    L = inj["x_tip"] - inj["x_vertical"]
    cyl(f"PND_NeedleH_{i}", r, L, (inj["x_vertical"] + L / 2, y, zt), rot=(0, math.pi / 2, 0), mat=steel, verts=16)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(inj["x_vertical"], y, zt), segments=16, ring_count=8)
    bpy.context.object.name = f"PND_NeedleBend_{i}"; bpy.context.object.data.materials.append(steel)

# ---------------------------------------------------------------- dye sheet (texture from streamline model)
dy = P["dye"]; U = 1.0; a = cy["D"] / 2; xc, yc = cy["center"]
x0, x1 = -ch["length"] / 4, ch["length"] / 4          # sheet x extent (covers frame)
y0_, y1_ = -hw + 0.0005, hw - 0.0005
res = dy["texture_res_m_per_px"]
nx = int(round((x1 - x0) / res)); ny = int(round((y1_ - y0_) / res))
X = x0 + (np.arange(nx) + 0.5) * res; Y = y0_ + (np.arange(ny) + 0.5) * res
XX, YY = np.meshgrid(X - xc, Y - yc)
R2 = np.maximum(XX ** 2 + YY ** 2, (a * 0.5) ** 2)
psi = YY * (1 - a * a / R2)
# |grad psi| (potential flow speed / U), clamped near stagnation
u = 1 - a * a * (XX ** 2 - YY ** 2) / R2 ** 2; v = -2 * a * a * XX * YY / R2 ** 2
spd = np.maximum(np.hypot(u, v), 0.35)
s_from_tip = np.maximum(XX + xc - inj["x_tip"], 0.0)
sigma = np.sqrt(dy["sigma0"] ** 2 + (dy["sigma_growth"] ** 2) * s_from_tip)  # diffusive widening ~ sqrt(s)
alpha = np.zeros_like(XX); sidx = np.zeros(XX.shape, np.int32)
for k_, y_i in enumerate(inj["y"]):
    d = (psi - (y_i - yc)) / spd
    a_k = dy["peak_alpha"] * (dy["sigma0"] / sigma) ** 0.5 * np.exp(-0.5 * (d / sigma) ** 2)
    sidx = np.where(a_k > alpha, k_, sidx); alpha = np.maximum(alpha, a_k)
x_front = xc + dy["front_x_rel_cyl_D"] * cy["D"]
start = 1 / (1 + np.exp(-(XX + xc - inj["x_tip"]) / 0.0002))                  # dye starts at needle tip
front = 1 / (1 + np.exp((XX + xc - x_front) / (dy["front_fade"] / 4)))         # soft dye front
alpha *= start * front
# natural unevenness: smooth random density field (fixed seed), stronger downstream
rng = np.random.default_rng(P["look"]["dye_seed"])
def smooth_field(cx_, cy_):
    g = rng.standard_normal((cy_, cx_))
    xi = np.linspace(0, cx_ - 1, nx); yi = np.linspace(0, cy_ - 1, ny)
    rows = np.array([np.interp(xi, np.arange(cx_), r) for r in g])
    return np.array([np.interp(yi, np.arange(cy_), c) for c in rows.T]).T
fld = 0.6 * smooth_field(30, 90) + 0.4 * smooth_field(10, 24)  # elongated along the flow
alpha *= np.clip(1 + P["look"]["dye_unevenness"] * fld, 0.35, 1.3)
alpha = np.clip(alpha, 0, 0.97)
alpha[(XX ** 2 + YY ** 2) < a * a] = 0.0
img = bpy.data.images.new("PND_DyeAlpha", nx, ny, alpha=True, float_buffer=False)
px = np.zeros((ny, nx, 4), np.float32); px[..., 0] = px[..., 1] = px[..., 2] = alpha; px[..., 3] = 1
img.pixels.foreach_set(px.ravel())
img.filepath_raw = os.path.join(OUT, "dye_ff.png"); img.file_format = "PNG"; img.save()
img.colorspace_settings.name = "Non-Color"; img.pack()

m, nt, out = new_mat("PND_Dye")
tx = nt.nodes.new("ShaderNodeTexImage"); tx.image = img; tx.interpolation = "Cubic"; tx.extension = "CLIP"
tr = nt.nodes.new("ShaderNodeBsdfTransparent"); tr.inputs[0].default_value = (1, 1, 1, 1)
tint = nt.nodes.new("ShaderNodeBsdfTransparent"); tint.inputs[0].default_value = (*[c * 1.5 for c in dy["color_rgb_linear"]], 1)
dif = nt.nodes.new("ShaderNodeBsdfDiffuse"); dif.inputs[0].default_value = (*dy["color_rgb_linear"], 1)
m1 = nt.nodes.new("ShaderNodeMixShader"); m1.inputs[0].default_value = 0.35
nt.links.new(tint.outputs[0], m1.inputs[1]); nt.links.new(dif.outputs[0], m1.inputs[2])
m2 = nt.nodes.new("ShaderNodeMixShader")
nt.links.new(tx.outputs["Color"], m2.inputs[0]); nt.links.new(tr.outputs[0], m2.inputs[1]); nt.links.new(m1.outputs[0], m2.inputs[2])
nt.links.new(m2.outputs[0], out.inputs[0])
bpy.ops.mesh.primitive_plane_add(size=1, location=((x0 + x1) / 2, (y0_ + y1_) / 2, dy["z_sheet"]))
ds = bpy.context.object; ds.name = "PND_DyeSheet"; ds.scale = (x1 - x0, y1_ - y0_, 1)
bpy.ops.object.transform_apply(scale=True); ds.data.materials.append(m)

CINE = P.get("style") == "cinematic34"
if CINE:
    V = P["v3"]; pal = np.array(V["dye_palette_linear"], np.float32)
    cols = pal[np.array(V["dye_streak_colors"])[sidx]]
    cimg = bpy.data.images.new("PND_DyeColor", nx, ny, alpha=False, float_buffer=False)
    cpx = np.ones((ny, nx, 4), np.float32); cpx[..., :3] = cols
    cimg.pixels.foreach_set(cpx.ravel()); cimg.filepath_raw = os.path.join(OUT, "dye_color_ff.png"); cimg.file_format = "PNG"; cimg.save()
    cimg.colorspace_settings.name = "Non-Color"; cimg.pack()
    ct = nt.nodes.new("ShaderNodeTexImage"); ct.image = cimg; ct.interpolation = "Closest"; ct.extension = "EXTEND"
    pb = nt.nodes.new("ShaderNodeBsdfPrincipled"); pb.inputs["Roughness"].default_value = 0.6
    nt.links.new(ct.outputs["Color"], pb.inputs["Base Color"])
    tt = nt.nodes.new("ShaderNodeBsdfTransparent"); nt.links.new(ct.outputs["Color"], tt.inputs[0])
    mm = nt.nodes.new("ShaderNodeMixShader"); mm.inputs[0].default_value = 1 - V["dye_shader_transparent_mix"]
    nt.links.new(tt.outputs[0], mm.inputs[1]); nt.links.new(pb.outputs[0], mm.inputs[2])
    nt.links.new(mm.outputs[0], m2.inputs[2])

# ---------------------------------------------------------------- lighting
lg = P["lighting"]
wd = bpy.data.worlds.new("PND_World"); sc.world = wd; wd.use_nodes = True
wd.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
wd.node_tree.nodes["Background"].inputs[1].default_value = lg["world_white"]
ld = bpy.data.lights.new("PND_Overhead", "AREA"); ld.shape = "SQUARE"; ld.size = lg["area_light"]["size"]
ld.energy = lg["area_light"]["power_W"]; ld.color = (1, 1, 1)
lo = bpy.data.objects.new("PND_Overhead", ld); lo.location = (0, 0, lg["area_light"]["z"]); sc.collection.objects.link(lo)
ld.cycles.is_caustics_light = False
lo.visible_glossy = False   # no mirror image of the light panel in the water surface (no glare)

# ---------------------------------------------------------------- photoreal extras
lk = P["look"]
# meniscus: thin water fillet along each inner wall face (refracts a faint line)
for s_ in (-1, 1):
    cyl(f"PND_Meniscus_{s_}", lk["meniscus_r"], ch["length"], (0, s_ * (hw - 0.00005), wa["surface_z"] - lk["meniscus_r"] * 0.3),
        rot=(0, math.pi / 2, 0), mat=water, verts=24)
# rake fixing screws (cap head + hex socket) where the bar rests on each wall
for s_ in (-1, 1):
    ys = s_ * (hw + t / 2); zt = rk["z_bottom"] + rk["thickness"]
    cyl(f"PND_ScrewHead_{s_}", 0.0022, 0.0018, (rk["x_center"], ys, zt + 0.0009), mat=steel, verts=48)
    cyl(f"PND_ScrewSocket_{s_}", 0.0012, 0.0006, (rk["x_center"], ys, zt + 0.00165), mat=black, verts=6)
# softbox seen only in reflections (water + acrylic), off to the upper right -> soft sheen, no hot spot
sbm, snt, sout = new_mat("PND_Softbox")
em = snt.nodes.new("ShaderNodeEmission"); em.inputs["Strength"].default_value = lk["softbox_strength"]
gr = snt.nodes.new("ShaderNodeTexGradient"); gr.gradient_type = "SPHERICAL"
stc = snt.nodes.new("ShaderNodeTexCoord"); snt.links.new(stc.outputs["Object"], gr.inputs[0])
smr = snt.nodes.new("ShaderNodeMapRange"); smr.interpolation_type = "SMOOTHSTEP"
smr.inputs[1].default_value = 0.5; smr.inputs[2].default_value = 1.0; smr.inputs[3].default_value = 0.0
smr.inputs[4].default_value = lk["softbox_strength"]
snt.links.new(gr.outputs["Fac"], smr.inputs[0]); snt.links.new(smr.outputs[0], em.inputs["Strength"])  # soft-edged panel
snt.links.new(em.outputs[0], sout.inputs[0])
bpy.ops.mesh.primitive_plane_add(size=1, location=tuple(lk["softbox_pos"]))
sb = bpy.context.object; sb.name = "PND_Softbox"; sb.scale = (*lk["softbox_size"], 1); sb.rotation_euler = (math.pi, 0, 0)
sb.data.materials.append(sbm)
sb.visible_camera = False; sb.visible_diffuse = False; sb.visible_shadow = False; sb.visible_transmission = False
# camera
cp = P["camera"]
cd = bpy.data.cameras.new("PND_Camera"); cd.lens = cp["lens_mm"]; cd.sensor_width = cp["sensor_width_mm"]
cd.sensor_fit = cp["sensor_fit"]; cd.clip_start = 0.05; cd.clip_end = 5
co = bpy.data.objects.new("PND_Camera", cd); co.location = cp["position"]; co.rotation_euler = (0, 0, 0)
co.lock_location = (True,) * 3; co.lock_rotation = (True,) * 3; co.lock_scale = (True,) * 3
sc.collection.objects.link(co); sc.camera = co

# ---------------------------------------------------------------- render settings
rp = P["render"]
sc.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "METAL"; prefs.refresh_devices()
for d in prefs.devices: d.use = True
sc.cycles.device = "GPU"
sc.cycles.samples = SAMPLES; sc.cycles.use_denoising = True; sc.cycles.denoiser = "OPENIMAGEDENOISE"
sc.cycles.seed = rp["seed"]; sc.cycles.use_adaptive_sampling = False
sc.cycles.caustics_reflective = False; sc.cycles.caustics_refractive = False
sc.cycles.max_bounces = 16; sc.cycles.transmission_bounces = 16; sc.cycles.transparent_max_bounces = 16
sc.cycles.glossy_bounces = 6; sc.cycles.diffuse_bounces = 3
sc.render.resolution_x, sc.render.resolution_y = rp["res"]; sc.render.resolution_percentage = int(round(100 * SCALE))
sc.render.pixel_aspect_x = sc.render.pixel_aspect_y = 1
sc.view_settings.view_transform = rp["view_transform"]; sc.view_settings.look = "None"
sc.view_settings.exposure = rp["exposure"]
sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGB"
sc.render.image_settings.color_depth = "8"
sc.frame_start = sc.frame_end = sc.frame_current = 0

# ---------------------------------------------------------------- v3 cinematic 3/4 overlay (look only; physics geometry untouched)
from mathutils import Vector
def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()
if CINE:
    # dark slate / deep-blue textured base with fake caustic web inside the channel
    m, nt, out = new_mat("PND_Slate")
    tco = nt.nodes.new("ShaderNodeTexCoord")
    n1 = nt.nodes.new("ShaderNodeTexNoise"); n1.inputs["Scale"].default_value = 25; n1.inputs["Detail"].default_value = 8
    n2 = nt.nodes.new("ShaderNodeTexNoise"); n2.inputs["Scale"].default_value = 400; n2.inputs["Detail"].default_value = 4
    for n_ in (n1, n2): nt.links.new(tco.outputs["Object"], n_.inputs["Vector"])
    cr = nt.nodes.new("ShaderNodeValToRGB"); nt.links.new(n1.outputs["Fac"], cr.inputs[0])
    cr.color_ramp.elements[0].color = (*V["slate_dark"], 1); cr.color_ramp.elements[1].color = (*V["slate_light"], 1)
    b = nt.nodes.new("ShaderNodeBsdfPrincipled"); nt.links.new(cr.outputs[0], b.inputs["Base Color"])
    rr = nt.nodes.new("ShaderNodeMapRange"); nt.links.new(n2.outputs["Fac"], rr.inputs[0])
    rr.inputs[3].default_value = 0.3; rr.inputs[4].default_value = 0.65; nt.links.new(rr.outputs[0], b.inputs["Roughness"])
    bm = nt.nodes.new("ShaderNodeBump"); bm.inputs["Strength"].default_value = 0.25; bm.inputs["Distance"].default_value = 0.0005
    hb = nt.nodes.new("ShaderNodeMath"); hb.operation = "ADD"
    nt.links.new(n1.outputs["Fac"], hb.inputs[0]); nt.links.new(n2.outputs["Fac"], hb.inputs[1])
    nt.links.new(hb.outputs[0], bm.inputs["Height"]); nt.links.new(bm.outputs[0], b.inputs["Normal"])
    # caustic web: voronoi distance-to-edge on noise-warped coords, masked to the channel floor
    nw = nt.nodes.new("ShaderNodeTexNoise"); nw.inputs["Scale"].default_value = 60; nt.links.new(tco.outputs["Object"], nw.inputs["Vector"])
    wm = nt.nodes.new("ShaderNodeMix"); wm.data_type = "VECTOR"; wm.inputs["Factor"].default_value = 0.012
    nt.links.new(tco.outputs["Object"], wm.inputs[4]); nt.links.new(nw.outputs["Color"], wm.inputs[5])
    vo = nt.nodes.new("ShaderNodeTexVoronoi"); vo.feature = "DISTANCE_TO_EDGE"; vo.inputs["Scale"].default_value = V["caustic_scale"]
    nt.links.new(wm.outputs[1], vo.inputs["Vector"])
    cm = nt.nodes.new("ShaderNodeMapRange"); cm.interpolation_type = "SMOOTHSTEP"
    cm.inputs[1].default_value = 0.0; cm.inputs[2].default_value = 0.035; cm.inputs[3].default_value = 1.0; cm.inputs[4].default_value = 0.0
    nt.links.new(vo.outputs["Distance"], cm.inputs[0])
    sp = nt.nodes.new("ShaderNodeSeparateXYZ"); nt.links.new(tco.outputs["Object"], sp.inputs[0])
    ab = nt.nodes.new("ShaderNodeMath"); ab.operation = "ABSOLUTE"; nt.links.new(sp.outputs["Y"], ab.inputs[0])
    lt = nt.nodes.new("ShaderNodeMath"); lt.operation = "LESS_THAN"; lt.inputs[1].default_value = hw
    nt.links.new(ab.outputs[0], lt.inputs[0])
    mk = nt.nodes.new("ShaderNodeMath"); mk.operation = "MULTIPLY"
    nt.links.new(cm.outputs[0], mk.inputs[0]); nt.links.new(lt.outputs[0], mk.inputs[1])
    ms = nt.nodes.new("ShaderNodeMath"); ms.operation = "MULTIPLY"; ms.inputs[1].default_value = V["caustic_strength"]
    nt.links.new(mk.outputs[0], ms.inputs[0])
    nt.links.new(ms.outputs[0], b.inputs["Emission Strength"]); b.inputs["Emission Color"].default_value = (*V["caustic_color"], 1)
    nt.links.new(b.outputs[0], out.inputs[0])
    pobj = bpy.data.objects["PND_Plate"]; pobj.data.materials.clear(); pobj.data.materials.append(m)
    # polished walls, livelier water ripple
    for nd in bpy.data.materials["PND_Water"].node_tree.nodes:
        if nd.type == "BUMP": nd.inputs["Strength"].default_value = V["water_ripple_strength"]
    # lighting: dark blue world, warm key, coloured practicals visible in reflections
    wd.node_tree.nodes["Background"].inputs[0].default_value = (*V["world_color"], 1)
    wd.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    bpy.data.objects.remove(lo)
    for L in V["lights"]:
        ldat = bpy.data.lights.new(L["name"], "AREA"); ldat.shape = "RECTANGLE"; ldat.size, ldat.size_y = L["size"]
        ldat.energy = L["power_W"]; ldat.color = L["color"]
        lob = bpy.data.objects.new(L["name"], ldat); lob.location = L["pos"]; sc.collection.objects.link(lob)
        look_at(lob, L.get("target", V["target"])); lob.visible_camera = False
    sb.location = V["softbox_pos"]; sb.scale = (*V["softbox_size"], 1); look_at(sb, V["target"])
    sb.rotation_euler.rotate_axis("X", math.pi)  # face the target (plane normal +Z)
    em.inputs["Color"].default_value = (*V["softbox_color"], 1); smr.inputs[4].default_value = V["softbox_strength"]
    # 3/4 camera with depth of field
    co.lock_location = (False,) * 3; co.lock_rotation = (False,) * 3
    cd.lens = V["lens_mm"]
    tgt = Vector(V["target"]); el, az = math.radians(V["elevation_deg"]), math.radians(V["azimuth_deg"])
    co.location = tgt + V["distance_m"] * Vector((-math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
    look_at(co, tgt)
    co.lock_location = (True,) * 3; co.lock_rotation = (True,) * 3
    cd.dof.use_dof = True; cd.dof.aperture_fstop = V["fstop"]
    focus = Vector((cy["center"][0], cy["center"][1], V["focus_z"]))
    cd.dof.focus_distance = (co.location - focus).length
    sc.view_settings.view_transform = V["view_transform"]; sc.view_settings.exposure = V["exposure"]

# ---------------------------------------------------------------- camera_ff.json (from the Blender camera of record)
bpy.context.view_layer.update()
W, H = rp["res"]; f_px = cd.lens / cd.sensor_width * W; cx, cyy = W / 2, H / 2
Mw = np.array(co.matrix_world); Rbc = Mw[:3, :3]; C = Mw[:3, 3]
R_wc = (np.diag([1, -1, -1]) @ Rbc.T)          # world -> OpenCV camera (x right, y down, z forward)
t_wc = -R_wc @ C
def proj(Pw):
    pc = R_wc @ np.asarray(Pw, float) + t_wc
    return [cx + f_px * pc[0] / pc[2], cyy + f_px * pc[1] / pc[2]], float(pc[2])
anchors = {"cylinder_base_centre": [cy["center"][0], cy["center"][1], 0.0],
           "cylinder_top_centre": [cy["center"][0], cy["center"][1], cy["z_top"]],
           **{f"needle_tip_{i}": [inj["x_tip"], y_, inj["z_tip"]] for i, y_ in enumerate(inj["y"])},
           **{f"wall_top_inner_{nm}_x{xx:+.2f}": [xx, sg * hw, ch["wall_height"] + ch["wall_z_gap"]]
              for nm, sg in (("near", -1), ("far", 1)) for xx in (-0.10, 0.0, 0.10, 0.15)}}
proj_pts = {k: {"X": v, "uv": [round(c, 2) for c in proj(v)[0]], "depth_m": round(proj(v)[1], 5)} for k, v in anchors.items()}
def local_scale(Pw):
    (u0, v0), z = proj(Pw); (u1, v1), _ = proj(np.add(Pw, [0.001, 0, 0])); (u2, v2), _ = proj(np.add(Pw, [0, 0.001, 0]))
    return {"px_per_mm_along_flow_X": round(math.hypot(u1 - u0, v1 - v0), 4), "px_per_mm_across_Y": round(math.hypot(u2 - u0, v2 - v0), 4)}
cam = {
    "model": "pinhole, no lens distortion; (0,0) = top-left corner of the image, pixel centres at +0.5; uv = K (R X + t)",
    "image_size": [W, H], "style": P.get("style", "topdown"),
    "intrinsics": {"fx_px": f_px, "fy_px": f_px, "cx_px": cx, "cy_px": cyy, "skew": 0.0, "lens_mm": cd.lens,
                   "sensor_width_mm": cd.sensor_width, "sensor_fit": cd.sensor_fit, "K": [[f_px, 0, cx], [0, f_px, cyy], [0, 0, 1]],
                   "depth_of_field": {"enabled": bool(cd.dof.use_dof), "fstop": cd.dof.aperture_fstop, "focus_distance_m": cd.dof.focus_distance}},
    "extrinsics": {"camera_position_world_m": [float(c) for c in C], "blender_rotation_euler_xyz_rad": list(co.rotation_euler),
                   "R_world_to_cv_camera": R_wc.tolist(), "t_world_to_cv_camera": t_wc.tolist(),
                   "elevation_deg_below_horizontal": P.get("v3", {}).get("elevation_deg") if CINE else 90.0,
                   "note": "World: Z up, flow +X, plate top z=0. OpenCV camera frame: x right, y down, z forward."},
    "scale_on_plate_z0 (local, perspective varies across the frame)": {k: local_scale(v) for k, v in
        {"at_cylinder_base": anchors["cylinder_base_centre"], "wake_x+0.05": [0.05, 0, 0], "wake_x+0.12": [0.12, 0, 0], "upstream_x-0.10": [-0.10, 0, 0]}.items()},
    "projection_test_anchor_points": proj_pts,
    "refraction_note": ("Everything below the water surface (z < %.3f m: plate/floor, dye sheet at z=%.3f, needle tips, cylinder below the surface) "
                        "is seen through a flat n=%.3f interface: at this oblique view the apparent position is displaced strongly (tens of px) "
                        "toward the camera/upwards in the image vs the pinhole projection above; use Snell refraction at the surface "
                        "(see scene/check_camera.py r_refr for the 1-D version) for any sub-surface point. Points above the surface "
                        "(cylinder top, wall tops, rake) follow the pinhole model directly.") % (wa["surface_z"], dy["z_sheet"], wa["ior"]),
}
json.dump(cam, open(os.path.join(ROOT, "config", "camera_ff.json"), "w"), indent=1)

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "vortex_ff.blend"), compress=True)
print("BUILD_OK", nx, ny)
if RENDER_OUT:
    sc.render.filepath = RENDER_OUT
    bpy.ops.render.render(write_still=True)
    print("RENDER_OK", RENDER_OUT)
