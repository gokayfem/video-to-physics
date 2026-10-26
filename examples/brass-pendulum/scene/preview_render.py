"""Small EEVEE preview of a saved pendulum .blend from its scene camera (geometry-role approx camera by default).

blender -b scene/out/pendulum.blend --python scene/preview_render.py -- --frame 0 --theta-deg=-4.68 --out x.png
--theta-deg poses PND_Swing (rotation_euler[1] = -theta) when the file has no baked animation; omitted => as stored.
Resolution 960x544 (half of 1920x1088). Run under tools/with_render_lock.sh.
"""
import argparse, math, sys, bpy
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--frame", type=int, default=0)
ap.add_argument("--theta-deg", type=float, default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--motion-blur-shutter", type=float, default=None,
                help="TEST override: enable motion blur with this shutter (fraction of a frame), CENTER")
ap.add_argument("--keep-look", action="store_true", help="keep the file's engine/samples/colour management (look.py)")
ap.add_argument("--mask", action="store_true", help="render a white-on-black silhouette of base, post and arm")
ap.add_argument("--engine", default="BLENDER_EEVEE")
ap.add_argument("--border", default=None, help="u0,v0,u1,v1 in full-res pixels: render that region at 100%% (cropped)")
a = ap.parse_args(argv)
sc = bpy.context.scene
bpy.context.window.scene = sc if bpy.context.window else None
sc.frame_set(a.frame)
sw = bpy.data.objects["PND_Swing"]
if a.theta_deg is not None and not (sw.animation_data and sw.animation_data.action):
    sw.rotation_euler[1] = -math.radians(a.theta_deg)
ap2 = None
if "--mask" in argv:
    # silhouette of the walnut stand parts only: Workbench flat white on black
    keep = {"PND_Base", "PND_Post", "PND_Arm"}
    for ob in sc.objects:
        if ob.type in {"MESH", "CURVE"} and ob.name not in keep:
            ob.hide_render = True
    a.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "FLAT"
    sc.display.shading.color_type = "SINGLE"
    sc.display.shading.single_color = (1, 1, 1)
    sc.display.shading.background_type = "VIEWPORT"
    sc.display.shading.background_color = (0, 0, 0)
    sc.render.film_transparent = False
    sc.display_settings.display_device = "sRGB"
    sc.view_settings.view_transform = "Standard"
if not a.keep_look or a.mask:
    sc.render.engine = a.engine
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 1920, 1088, 50
if a.border:
    u0, v0, u1, v1 = [float(x) for x in a.border.split(",")]
    assert (u1 - u0) * (v1 - v0) <= 960 * 544, "keep border renders small"
    sc.render.resolution_percentage = 100
    sc.render.use_border, sc.render.use_crop_to_border = True, True
    sc.render.border_min_x, sc.render.border_max_x = u0 / 1920, u1 / 1920
    sc.render.border_min_y, sc.render.border_max_y = 1 - v1 / 1088, 1 - v0 / 1088
if a.engine == "BLENDER_EEVEE" and not a.keep_look:
    sc.eevee.taa_render_samples = 32
if not a.mask and not a.keep_look:
    sc.view_settings.view_transform = "AgX"
if a.motion_blur_shutter is not None:
    sc.render.use_motion_blur = a.motion_blur_shutter > 0
    sc.render.motion_blur_shutter = max(a.motion_blur_shutter, 0.01)
    sc.render.motion_blur_position = "CENTER"
sc.render.image_settings.file_format = "PNG"
sc.render.filepath = a.out
bpy.ops.render.render(write_still=True, scene=sc.name)
