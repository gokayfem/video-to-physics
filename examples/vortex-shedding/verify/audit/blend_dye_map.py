"""Role E: inspect dye image-sequence wiring in a .blend (run: blender -b X.blend --python blend_dye_map.py -- out.json)"""
import bpy, json, sys, os
out = sys.argv[sys.argv.index('--') + 1]
sc = bpy.context.scene
R = dict(blend=bpy.data.filepath, scene=sc.name, frame_start=sc.frame_start, frame_end=sc.frame_end, fps=sc.render.fps / sc.render.fps_base, users=[])
def users():
    for mat in bpy.data.materials:
        if mat.node_tree:
            for n in mat.node_tree.nodes:
                if n.type == 'TEX_IMAGE' and n.image and n.image.source == 'SEQUENCE':
                    yield mat.name, n
for mname, n in users():
    iu = n.image_user; img = n.image
    d = dict(material=mname, node=n.name, image=img.name, filepath=img.filepath, abs=bpy.path.abspath(img.filepath),
             colorspace=img.colorspace_settings.name, sfra=iu.frame_start, offset=iu.frame_offset, frames=iu.frame_duration,
             cyclic=iu.use_cyclic, auto_refresh=iu.use_auto_refresh, probes={})
    for f in sorted(set([sc.frame_start, sc.frame_start + 1, sc.frame_start + 120, sc.frame_end])):
        sc.frame_set(f)
        try:
            d['probes'][f] = dict(file=os.path.basename(img.filepath_from_user(image_user=iu)), frame_current=iu.frame_current)
        except Exception as e:
            d['probes'][f] = dict(error=str(e))
        # Blender formula (BKE_image_user_frame_get): file# = f - sfra + 1 + offset (clamped to [.., frames] when not cyclic)
        d['probes'][f]['formula_file_number'] = min(max(f - iu.frame_start + 1, 0), iu.frame_duration) + iu.frame_offset
    R['users'].append(d)
json.dump(R, open(out, 'w'), indent=1)
print(json.dumps(R, indent=1)[:3000])
