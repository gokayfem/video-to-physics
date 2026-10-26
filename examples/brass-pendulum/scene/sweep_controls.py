"""Control sanity sweep: change PND Stand macro controls on the saved file and render small Workbench stills.
blender -b scene/out/pendulum.blend --python scene/sweep_controls.py -- --outdir scene/out/sweep
Proves the controls drive the geometry (pivot, pin and pendulum stay put). File is not saved."""
import sys, pathlib, bpy
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "vendor"))   # vendored nodekit
from nodekit import inspect as nki
argv = sys.argv[sys.argv.index("--") + 1:]
out = pathlib.Path(argv[argv.index("--outdir") + 1]); out.mkdir(parents=True, exist_ok=True)
sc = bpy.context.scene
if bpy.context.window: bpy.context.window.scene = sc
mod = bpy.data.objects["PND_StandControls"].modifiers[0]
base = nki.get_modifier_inputs(mod)
sc.render.engine = "BLENDER_WORKBENCH"; sc.display.shading.light = "STUDIO"; sc.display.shading.color_type = "MATERIAL"
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 1920, 1088, 25
for name, ch in [("default", {}), ("post_height_x1.15", {"Post Height": base["Post Height"] * 1.15}),
                 ("arm_length_plus_0.06m", {"Arm Length": base["Arm Length"] + 0.06}),
                 ("base_wider_deeper", {"Base Width": base["Base Width"] * 1.25, "Base Depth": base["Base Depth"] * 1.3,
                                        "Base Edge Radius": base["Base Edge Radius"] * 2})]:
    nki.set_modifier_inputs(mod, {**base, **ch} if ch else {k: base[k] for k in ("Post Height",)})
    nki.set_modifier_inputs(mod, {k: v for k, v in {**base, **ch}.items() if not isinstance(v, str)})
    sc.render.filepath = str(out / f"{name}.png"); bpy.ops.render.render(write_still=True, scene=sc.name)
