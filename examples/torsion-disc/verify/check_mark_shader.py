"""Mark rigidity when the mark is a SHADER on the disc (clip 03 ck001): reopened .blend, verifier-owned.
  blender -b X.blend --python verify/check_mark_shader.py -- --params P.json --out J.json [--disc-name TOR_Disc]
PASS iff: every Texture Coordinate node feeding the material uses the 'Object' output of the disc itself (object = None), every
Attribute node is OBJECT-type and reads a custom property that is NOT animated (no F-curve / driver), the azimuth property equals
params mark.mu_body_rad (1e-12), and no world/camera-space coordinate (Generated from another object, Window, Camera...) drives
the mark geometry. Disc mesh rigidity in its own frame is judged by check_blend.py (P5)."""
import bpy, sys, json, os
argv = sys.argv[sys.argv.index("--") + 1:]
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--params", required=True); ap.add_argument("--out", required=True); ap.add_argument("--disc-name", default="TOR_Disc")
a = ap.parse_args(argv)
P = json.load(open(a.params))
mu = P.get("mark", {}).get("mu_body_rad", {}); mu = mu.get("value") if isinstance(mu, dict) else mu
d = bpy.data.objects[a.disc_name]
rep = dict(blend=bpy.data.filepath, disc=d.name, materials=[], params_mu=mu)
ok = True
for m in d.data.materials:
    nt = m.node_tree; ent = dict(name=m.name, texcoord=[], attributes=[], other_coord_nodes=[])
    for n in nt.nodes:
        if n.type == "TEX_COORD":
            outs = [l.from_socket.name for l in nt.links if l.from_node == n]
            ent["texcoord"].append(dict(node=n.name, object=n.object.name if n.object else None, outputs_used=outs))
            ok &= (n.object is None or n.object == d) and all(o == "Object" for o in outs)
        if n.type == "ATTRIBUTE":
            used = any(l.from_node == n for l in nt.links)
            ent["attributes"].append(dict(name=n.attribute_name, type=n.attribute_type, used=used, value=d.get(n.attribute_name)))
            if used:
                ok &= n.attribute_type == "OBJECT" and n.attribute_name in d.keys()
        if n.type in ("UVMAP", "TEX_IMAGE") and any(l.from_node == n for l in nt.links):
            ent["other_coord_nodes"].append(n.type)
    rep["materials"].append(ent)
ad = d.animation_data
anim_props = []
if ad:
    anim_props = [dr.data_path for dr in ad.drivers if dr.data_path.startswith('["')]
    if ad.action:
        from bpy_extras import anim_utils
        cb = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
        anim_props += [f.data_path for f in cb.fcurves if f.data_path.startswith('["')]
rep["animated_custom_properties"] = anim_props
ok &= not anim_props
az = d.get("tor_mark_azimuth_rad")
rep["blend_mark_azimuth_rad"] = az
rep["azimuth_vs_params"] = None if (az is None or mu is None) else abs(az - mu)
ok &= rep["azimuth_vs_params"] is not None and rep["azimuth_vs_params"] < 1e-12
rep["status"] = "PASS" if ok else "FAIL"
json.dump(rep, open(a.out, "w"), indent=2, default=str)
print("CHECK_MARK_SHADER", rep["status"])
