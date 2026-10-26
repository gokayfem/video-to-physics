"""Lead preview render: blender -b <blend> --python review/render_preview.py -- --out DIR [--pct 50] [--engine EEVEE|WORKBENCH]"""
import argparse, sys, bpy
a = argparse.ArgumentParser()
a.add_argument('--out', required=True); a.add_argument('--pct', type=int, default=50)
a.add_argument('--engine', default='BLENDER_EEVEE'); a.add_argument('--samples', type=int, default=16)
a.add_argument('--camera', default=None)
a = a.parse_args(sys.argv[sys.argv.index('--') + 1:])
s = bpy.context.scene
s.camera = bpy.data.objects[a.camera] if a.camera else next((o for o in bpy.data.objects if o.type == 'CAMERA' and o.name.endswith('_Cam_Reference')), s.camera)
s.render.engine = a.engine
if a.engine.startswith('BLENDER_EEVEE'):
    s.eevee.taa_render_samples = a.samples
elif a.engine == 'CYCLES':
    s.cycles.samples = a.samples; s.cycles.use_denoising = True
s.render.resolution_x, s.render.resolution_y, s.render.resolution_percentage = 1920, 1088, a.pct
s.render.image_settings.file_format = 'PNG'
s.render.filepath = a.out.rstrip('/') + '/f'
s.frame_start, s.frame_end = 0, 240
bpy.ops.render.render(animation=True)
