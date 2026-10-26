"""Look (materials, textures, lighting, set) for clip 02 (vertical spring) — scene role C.

Called by scene/build.py --look as apply_look(scene, params). Every number comes from params/look.json (or env
VSP_LOOK_JSON for experiments). Deterministic: procedural textures with fixed seeds stored in look.json.
Lengths in look.json are in S (clear span between the posts; params["S_m"] metres); light energies are W at
S_ref_m and scale with (S_m / S_ref_m)^2.

Materials are edited IN PLACE (builder names):
  VSP_Walnut        procedural walnut (clip-01 v011 shader: fibre streaks + figure bands + pores + domain warp),
                    per-object grain direction via object custom properties vsp_grain_axis / vsp_end_axis /
                    vsp_wood_seed / vsp_wood_tone / vsp_wood_freq (posts: grain along Z)
  VSP_BlackSteel    satin black powder-coat/blackened steel (crossbar, base plate) with a faint orange-peel bump
  VSP_ScrewSteel    black screw heads
  VSP_SpringSteel   spring wire: bright drawn steel, anisotropic along the wire (tangent from the curve's UV-less
                    geometry is not available, so isotropic satin + fine roughness noise)
  VSP_BrushedSteel  weight: lathe-brushed steel: anisotropic, tangent = radial about the object Z axis (brushing runs
                    around the circumference: vertical highlight bands on the side, concentric rings on the top),
                    fine circumferential streaks in roughness/bump
  VSP_EyeSteel      polished steel ring
Set (collection VSP_Look): VSP_Look_Table (light wood, top at the plate bottom, back edge at y_wall),
  VSP_Look_Backdrop (vertical wall at the table's back edge), lights (VSP_Look_*), world, colour management.
"""
import json
import math
import os
import pathlib

import bpy
from mathutils import Vector

LOOK_COLL = "VSP_Look"


# ---------------------------------------------------------------- small node helpers (from clip-01 scene/look.py)
class NT:
    def __init__(self, mat):
        self.nt = mat.node_tree
        self.nt.nodes.clear()
        self.x = 0

    def n(self, kind, label=None, **props):
        node = self.nt.nodes.new(kind)
        for k, v in props.items():
            setattr(node, k, v)
        if label:
            node.label = label
        node.location = (self.x, 0)
        self.x += 40
        return node

    @staticmethod
    def sock(sockets, name):
        for s in sockets:
            if s.enabled and (s.name == name or s.identifier == name):
                return s
        raise KeyError(name)

    def link(self, a, b):
        self.nt.links.new(a, b)

    def inp(self, node, name, value):
        s = self.sock(node.inputs, name)
        if hasattr(value, "is_output") or hasattr(value, "links"):
            self.link(value, s)
        else:
            s.default_value = value
        return s

    def out(self, node, name=None):
        return node.outputs[0] if name is None else self.sock(node.outputs, name)

    # math
    def math(self, op, a, b=None, clamp=False):
        m = self.n("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if hasattr(v, "is_output"):
                self.link(v, m.inputs[i])
            else:
                m.inputs[i].default_value = v
        return m.outputs[0]

    def vmath(self, op, a, b=None, scale=None):
        m = self.n("ShaderNodeVectorMath", operation=op)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if hasattr(v, "is_output"):
                self.link(v, m.inputs[i])
            else:
                m.inputs[i].default_value = v
        if scale is not None:
            s = self.sock(m.inputs, "Scale")
            if hasattr(scale, "is_output"):
                self.link(scale, s)
            else:
                s.default_value = scale
        return m

    def combine(self, x, y, z):
        c = self.n("ShaderNodeCombineXYZ")
        for i, v in enumerate((x, y, z)):
            if hasattr(v, "is_output"):
                self.link(v, c.inputs[i])
            else:
                c.inputs[i].default_value = v
        return c.outputs[0]

    def noise(self, vec, scale=1.0, detail=2.0, rough=0.5, distortion=0.0, dims="3D", w=None):
        t = self.n("ShaderNodeTexNoise", noise_dimensions=dims)
        self.link(vec, self.sock(t.inputs, "Vector"))
        self.sock(t.inputs, "Scale").default_value = scale
        self.sock(t.inputs, "Detail").default_value = detail
        self.sock(t.inputs, "Roughness").default_value = rough
        self.sock(t.inputs, "Distortion").default_value = distortion
        return self.sock(t.outputs, "Factor") if any(s.name == "Factor" for s in t.outputs) else t.outputs["Fac"]

    def maprange(self, v, a0, a1, b0=0.0, b1=1.0, interp="LINEAR", clamp=True):
        m = self.n("ShaderNodeMapRange", interpolation_type=interp, clamp=clamp)
        self.link(v, m.inputs["Value"])
        m.inputs["From Min"].default_value, m.inputs["From Max"].default_value = a0, a1
        m.inputs["To Min"].default_value, m.inputs["To Max"].default_value = b0, b1
        return self.sock(m.outputs, "Result")

    def mixrgb(self, fac, a, b, blend="MIX"):
        m = self.n("ShaderNodeMix", data_type="RGBA", blend_type=blend)
        for name, v in (("Factor", fac), ("A", a), ("B", b)):
            s = self.sock(m.inputs, name)
            if hasattr(v, "is_output"):
                self.link(v, s)
            else:
                s.default_value = v if name == "Factor" else (*v, 1.0) if len(v) == 3 else v
        return self.sock(m.outputs, "Result")

    def attr_obj(self, name, out="Vector"):
        a = self.n("ShaderNodeAttribute", attribute_type="OBJECT", attribute_name=name)
        return self.sock(a.outputs, out)




def _principled(T):
    b = T.n("ShaderNodeBsdfPrincipled")
    o = T.n("ShaderNodeOutputMaterial")
    T.link(b.outputs[0], o.inputs["Surface"])
    return b


def _bump(T, b, height, strength, distance):
    bump = T.n("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = distance
    T.link(height, bump.inputs["Height"])
    T.link(bump.outputs["Normal"], b.inputs["Normal"])
    return bump


# ---------------------------------------------------------------- walnut (clip-01 v011 shader, unit = S)
def _walnut(mat, W, S):
    T = NT(mat)
    b = _principled(T)
    tc = T.n("ShaderNodeTexCoord")
    fscale = T.attr_obj("vsp_wood_freq", "Fac")
    P = T.vmath("SCALE", tc.outputs["Object"], scale=T.math("DIVIDE", fscale, S)).outputs["Vector"]
    geo = T.n("ShaderNodeNewGeometry")
    N = geo.outputs["Normal"]
    A = T.attr_obj("vsp_grain_axis")
    E = T.attr_obj("vsp_end_axis")
    seed = T.attr_obj("vsp_wood_seed", "Fac")
    tone = T.attr_obj("vsp_wood_tone", "Fac")
    C = T.vmath("CROSS_PRODUCT", N, A).outputs["Vector"]
    Clen = T.vmath("LENGTH", C).outputs["Value"]
    s_side = T.math("DIVIDE", T.vmath("DOT_PRODUCT", P, C).outputs["Value"], T.math("MAXIMUM", Clen, 0.05))
    s_end = T.vmath("DOT_PRODUCT", P, E).outputs["Value"]
    ndota = T.math("ABSOLUTE", T.vmath("DOT_PRODUCT", N, A).outputs["Value"])
    w_end = T.maprange(ndota, 0.6, 0.85, interp="SMOOTHSTEP")
    s = T.math("ADD", T.math("MULTIPLY", s_side, T.math("SUBTRACT", 1.0, w_end)), T.math("MULTIPLY", s_end, w_end))
    t = T.vmath("DOT_PRODUCT", P, A).outputs["Value"]
    wv = T.combine(T.math("MULTIPLY", s, W["warp_freq"] * 0.5), T.math("MULTIPLY", t, W["warp_freq"]), seed)
    warp = T.math("MULTIPLY", T.math("SUBTRACT", T.noise(wv, 1.0, 3.0, 0.5), 0.5), W["warp_amp"] * 2.0)
    sw = T.math("ADD", s, warp)
    v1 = T.combine(T.math("MULTIPLY", sw, W["streak_freq_across"]), T.math("MULTIPLY", t, W["streak_freq_along"]),
                   T.math("ADD", seed, 3.3))
    streak = T.noise(v1, 1.0, W["streak_detail"], 0.55)
    streak = T.maprange(streak, 0.5 - 0.25 / W["streak_contrast"], 0.5 + 0.25 / W["streak_contrast"])
    v2 = T.combine(T.math("MULTIPLY", sw, W["band_freq_across"]), T.math("MULTIPLY", t, W["band_freq_along"]),
                   T.math("ADD", seed, 7.7))
    band = T.maprange(T.noise(v2, 1.0, 2.0, 0.5), 0.3, 0.7)
    fig = T.math("ADD", T.math("MULTIPLY", streak, 1.0 - W["band_weight"]), T.math("MULTIPLY", band, W["band_weight"]))
    if W.get("cathedral_amount", 0.0) > 0:
        # flat-sawn 'cathedral' arcs: stripes of (s + k * |t - t0|^...) approximated by a sine of s warped by t
        arc = T.math("ADD", sw, T.math("MULTIPLY", T.math("SINE", T.math("MULTIPLY", t, W["cathedral_freq_along"])), W["cathedral_bend"]))
        v5 = T.combine(T.math("MULTIPLY", arc, W["cathedral_freq_across"]), 0.0, T.math("ADD", seed, 2.2))
        cath = T.maprange(T.noise(v5, 1.0, 1.0, 0.5), 0.35, 0.65)
        fig = T.math("ADD", T.math("MULTIPLY", fig, 1.0 - W["cathedral_amount"]), T.math("MULTIPLY", cath, W["cathedral_amount"]))
    if W.get("mottle_amount", 0.0) > 0:
        v4 = T.vmath("SCALE", P, scale=W["mottle_freq"]).outputs["Vector"]
        mot = T.math("SUBTRACT", T.noise(v4, 1.0, 3.0, 0.6), 0.5)
        fig = T.math("ADD", fig, T.math("MULTIPLY", mot, W["mottle_amount"] * 2.0), clamp=True)
    v3 = T.combine(T.math("MULTIPLY", sw, W["pore_freq_across"]), T.math("MULTIPLY", t, W["pore_freq_along"]),
                   T.math("ADD", seed, 11.1))
    pore = T.maprange(T.noise(v3, 1.0, 2.0, 0.6), W["pore_threshold"], W["pore_threshold"] + 0.08, interp="SMOOTHSTEP")
    pore = T.math("MULTIPLY", pore, W["pore_amount"])
    ec = W.get("end_contrast", 1.0)
    if ec != 1.0:
        fig_end = T.math("ADD", 0.5, T.math("MULTIPLY", T.math("SUBTRACT", fig, 0.5), ec))
        fig = T.math("ADD", T.math("MULTIPLY", fig, T.math("SUBTRACT", 1.0, w_end)), T.math("MULTIPLY", fig_end, w_end))
    ramp = T.n("ShaderNodeValToRGB")
    T.link(fig, ramp.inputs["Fac"])
    el = ramp.color_ramp.elements
    el[0].position, el[0].color = 0.0, (*W["color_dark"], 1.0)
    el[1].position, el[1].color = 1.0, (*W["color_light"], 1.0)
    mid = el.new(0.5)
    mid.color = (*W["color_mid"], 1.0)
    col = ramp.outputs["Color"]
    col = T.mixrgb(pore, col, W["pore_color"])
    endk = T.math("ADD", 1.0, T.math("MULTIPLY", w_end, W["end_grain_darken"] - 1.0))
    k = T.math("MULTIPLY", endk, tone)
    col = T.mixrgb(1.0, col, T.combine(k, k, k), blend="MULTIPLY")
    T.link(col, b.inputs["Base Color"])
    rough = T.math("ADD", W["roughness"], T.math("MULTIPLY", pore, W["roughness_pore"] - W["roughness"]))
    T.link(rough, b.inputs["Roughness"])
    b.inputs["Specular IOR Level"].default_value = W["specular_ior_level"]
    b.inputs["Coat Weight"].default_value = W["coat_weight"]
    b.inputs["Coat Roughness"].default_value = W["coat_roughness"]
    _bump(T, b, T.math("SUBTRACT", fig, T.math("MULTIPLY", pore, 0.6)), W["bump_strength"], W["bump_distance"] * S)
    mat.diffuse_color = (*W["color_mid"], 1.0)
    mat["vsp_look"] = "procedural walnut (clip-01 v011 shader), see scene/look.py and params/look.json walnut"


# ---------------------------------------------------------------- metals
def _satin(mat, P, S):
    """Satin/black steel: base colour, metallic, roughness with low-frequency variation, orange-peel bump."""
    T = NT(mat)
    b = _principled(T)
    b.inputs["Base Color"].default_value = (*P["base_color"], 1.0)
    b.inputs["Metallic"].default_value = P["metallic"]
    b.inputs["Specular IOR Level"].default_value = P.get("specular_ior_level", 0.5)
    if "specular_tint" in P:
        b.inputs["Specular Tint"].default_value = (*P["specular_tint"], 1.0)
    tc = T.n("ShaderNodeTexCoord")
    pos = T.vmath("SCALE", tc.outputs["Object"], scale=1.0 / S).outputs["Vector"]
    rn = T.noise(T.vmath("SCALE", pos, scale=P.get("rough_noise_freq", 8.0)).outputs["Vector"], 1.0, 3.0, 0.5)
    rough = T.maprange(rn, 0.3, 0.7, P["roughness"] - P.get("rough_noise_amp", 0.0), P["roughness"] + P.get("rough_noise_amp", 0.0))
    T.link(rough, b.inputs["Roughness"])
    if P.get("coat_weight", 0.0) > 0:
        b.inputs["Coat Weight"].default_value = P["coat_weight"]
        b.inputs["Coat Roughness"].default_value = P.get("coat_roughness", 0.2)
    if P.get("bump_strength", 0.0) > 0:
        pn = T.noise(T.vmath("SCALE", pos, scale=P.get("bump_freq", 400.0)).outputs["Vector"], 1.0, 2.0, 0.5)
        _bump(T, b, pn, P["bump_strength"], P.get("bump_distance", 0.0002) * S)
    mat.diffuse_color = (*P["base_color"], 1.0)
    mat["vsp_look"] = json.dumps(P, sort_keys=True)


def _brushed(mat, P, S):
    """Lathe-brushed steel: tangent radial about object Z (the weight's axis), streaks around the circumference."""
    T = NT(mat)
    b = _principled(T)
    b.inputs["Base Color"].default_value = (*P["base_color"], 1.0)
    b.inputs["Metallic"].default_value = P["metallic"]
    tan = T.n("ShaderNodeTangent", direction_type="RADIAL", axis="Z")
    T.link(tan.outputs["Tangent"], b.inputs["Tangent"])
    b.inputs["Anisotropic"].default_value = P["anisotropic"]
    tc = T.n("ShaderNodeTexCoord")
    sep = T.n("ShaderNodeSeparateXYZ")
    T.link(tc.outputs["Object"], sep.inputs[0])
    # cylindrical coordinates: radius r, height z (streaks = rings of constant z on the side, constant r on the top)
    r = T.math("SQRT", T.math("ADD", T.math("MULTIPLY", sep.outputs["X"], sep.outputs["X"]),
                              T.math("MULTIPLY", sep.outputs["Y"], sep.outputs["Y"])))
    ang = T.math("ARCTAN2", sep.outputs["Y"], sep.outputs["X"])
    zz = T.math("DIVIDE", sep.outputs["Z"], S)
    rr = T.math("DIVIDE", r, S)
    side = T.combine(T.math("MULTIPLY", zz, P["streak_freq"]), T.math("MULTIPLY", ang, P["streak_freq_along"]), 0.0)
    top = T.combine(T.math("MULTIPLY", rr, P["streak_freq"]), T.math("MULTIPLY", ang, P["streak_freq_along"]), 5.0)
    geo = T.n("ShaderNodeNewGeometry")
    nz = T.n("ShaderNodeSeparateXYZ")
    T.link(geo.outputs["Normal"], nz.inputs[0])
    # object-space normal is not available directly; use |N.z_world| ~ facing up/down for the top face (weight is upright
    # within ~10 deg); the streak pattern differs only in which coordinate is used
    wt = T.maprange(T.math("ABSOLUTE", nz.outputs["Z"]), 0.7, 0.9, interp="SMOOTHSTEP")
    s_side = T.noise(side, 1.0, P["streak_detail"], 0.6)
    s_top = T.noise(top, 1.0, P["streak_detail"], 0.6)
    streak = T.math("ADD", T.math("MULTIPLY", s_side, T.math("SUBTRACT", 1.0, wt)), T.math("MULTIPLY", s_top, wt))
    rough = T.maprange(streak, 0.35, 0.65, P["roughness"] - P["roughness_streak_amp"], P["roughness"] + P["roughness_streak_amp"])
    T.link(rough, b.inputs["Roughness"])
    colk = T.maprange(streak, 0.3, 0.7, 1.0 - P["color_streak_amp"], 1.0 + P["color_streak_amp"])
    col = T.mixrgb(1.0, P["base_color"], T.combine(colk, colk, colk), blend="MULTIPLY")
    T.link(col, b.inputs["Base Color"])
    _bump(T, b, streak, P["bump_strength"], P["bump_distance"] * S)
    mat.diffuse_color = (*P["base_color"], 1.0)
    mat["vsp_look"] = json.dumps(P, sort_keys=True)


def _plain_metal(mat, P):
    T = NT(mat)
    b = _principled(T)
    b.inputs["Base Color"].default_value = (*P["base_color"], 1.0)
    b.inputs["Metallic"].default_value = P["metallic"]
    b.inputs["Roughness"].default_value = P["roughness"]
    if "anisotropic" in P:
        b.inputs["Anisotropic"].default_value = P["anisotropic"]
    mat.diffuse_color = (*P["base_color"], 1.0)
    mat["vsp_look"] = json.dumps(P, sort_keys=True)


# ---------------------------------------------------------------- table wood (light, grain along X)
def _tablewood(mat, P, S):
    T = NT(mat)
    b = _principled(T)
    tc = T.n("ShaderNodeTexCoord")
    sep = T.n("ShaderNodeSeparateXYZ")
    T.link(tc.outputs["Object"], sep.inputs[0])
    ax_along, ax_across = ("Y", "X") if P.get("grain_axis", "X") == "Y" else ("X", "Y")
    x = T.math("DIVIDE", sep.outputs[ax_along], S)       # along the grain
    y = T.math("DIVIDE", sep.outputs[ax_across], S)      # across the grain
    wv = T.combine(T.math("MULTIPLY", x, P["warp_freq"]), T.math("MULTIPLY", y, P["warp_freq"] * 2.0), 1.3)
    warp = T.math("MULTIPLY", T.math("SUBTRACT", T.noise(wv, 1.0, 2.0, 0.5), 0.5), P["warp_amp"] * 2.0)
    yw = T.math("ADD", y, warp)
    v1 = T.combine(T.math("MULTIPLY", x, P["streak_freq_along"]), T.math("MULTIPLY", yw, P["streak_freq_across"]), 4.4)
    streak = T.maprange(T.noise(v1, 1.0, P["streak_detail"], 0.55), 0.5 - 0.25 / P["streak_contrast"], 0.5 + 0.25 / P["streak_contrast"])
    v2 = T.combine(T.math("MULTIPLY", x, P["band_freq_along"]), T.math("MULTIPLY", yw, P["band_freq_across"]), 8.8)
    band = T.maprange(T.noise(v2, 1.0, 2.0, 0.5), 0.3, 0.7)
    fig = T.math("ADD", T.math("MULTIPLY", streak, 1.0 - P["band_weight"]), T.math("MULTIPLY", band, P["band_weight"]))
    ramp = T.n("ShaderNodeValToRGB")
    T.link(fig, ramp.inputs["Fac"])
    el = ramp.color_ramp.elements
    el[0].color, el[1].color = (*P["color_dark"], 1.0), (*P["color_light"], 1.0)
    T.link(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = P["roughness"]
    b.inputs["Specular IOR Level"].default_value = P.get("specular_ior_level", 0.5)
    _bump(T, b, fig, P["bump_strength"], P["bump_distance"] * S)
    mat.diffuse_color = (*P["color_light"], 1.0)
    mat["vsp_look"] = "procedural light table wood, grain axis from look.json set.table.grain_axis"


def _backdrop(mat, P):
    T = NT(mat)
    b = _principled(T)
    b.inputs["Base Color"].default_value = (*P["color"], 1.0)
    b.inputs["Roughness"].default_value = P["roughness"]
    b.inputs["Specular IOR Level"].default_value = P.get("specular_ior_level", 0.2)
    mat.diffuse_color = (*P["color"], 1.0)


# ---------------------------------------------------------------- set, lights, world
def _coll(sc):
    c = bpy.data.collections.get(LOOK_COLL) or bpy.data.collections.new(LOOK_COLL)
    if c.name not in sc.collection.children:
        sc.collection.children.link(c)
    return c


def _quad(name, verts, coll, mat):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], [tuple(range(len(verts)))])
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def _set(sc, Sd, D, coll, report):
    S = D["S"]
    zt = D["base_top_z"] - D["plate_thickness"]
    yw = Sd.get("table_back_edge_y_S", D["table_back_edge_y"] / S) * S
    hw, hh, front = Sd["half_width_S"] * S, Sd["wall_height_S"] * S, Sd["table_front_S"] * S
    tmat = bpy.data.materials.get("VSP_Look_TableWood") or bpy.data.materials.new("VSP_Look_TableWood")
    _tablewood(tmat, Sd["table"], S)
    wmat = bpy.data.materials.get("VSP_Look_BackdropMat") or bpy.data.materials.new("VSP_Look_BackdropMat")
    _backdrop(wmat, Sd["backdrop"])
    table = _quad("VSP_Look_Table", [(-hw, -front, zt), (hw, -front, zt), (hw, yw, zt), (-hw, yw, zt)], coll, tmat)
    wall = _quad("VSP_Look_Backdrop", [(-hw, yw, zt), (hw, yw, zt), (hw, yw, zt + hh), (-hw, yw, zt + hh)], coll, wmat)
    if Sd.get("table_visible_glossy") is False:           # VR-05: the weight mirrored the lit table as a beige lobe
        table.visible_glossy = False
    report["set"] = {"table_z_m": zt, "wall_y_m": yw, "table_visible_glossy": table.visible_glossy}
    return table, wall


def _sph(target, az, el, dist):
    a, e = math.radians(az), math.radians(el)
    return target + dist * Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))


def _aim(ob, target):
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = (target - ob.location).to_track_quat("-Z", "Y")


def _lights(sc, L, D, coll, set_objs, report, k_energy):
    S = D["S"]
    target = Vector(L["target_S"]) * S
    rep = {}
    flagged = bpy.data.collections.new("VSP_Look_FlagBackdrop")
    flagged.objects.link(set_objs[1])
    flagged.collection_objects[0].light_linking.link_state = "EXCLUDE"
    for key in sorted(k for k in L if isinstance(L[k], dict) and L[k].get("type") == "AREA"):
        P = L[key]
        name = "VSP_Look_" + "".join(w.capitalize() for w in key.split("_"))
        ld = bpy.data.lights.new(name, "AREA")
        ld.shape = P.get("shape", "DISK")
        if ld.shape in ("RECTANGLE", "ELLIPSE"):
            ld.size, ld.size_y = P["size_S"] * S, P.get("size_y_S", P["size_S"]) * S
        else:
            ld.size = P["size_S"] * S
        ld.energy = P["energy_W"] * k_energy
        ld.color = P["color"]
        ob = bpy.data.objects.new(name, ld)
        coll.objects.link(ob)
        tgt = Vector(P["aim_S"]) * S if "aim_S" in P else target
        if "location_S" in P:
            ob.location = Vector(P["location_S"]) * S
        else:
            ob.location = _sph(tgt, P["azimuth_deg"], P["elevation_deg"], P["distance_S"] * S)
        _aim(ob, tgt)
        if P.get("flag_off_backdrop"):
            ob.light_linking.receiver_collection = flagged
        if P.get("flag_off_weight"):                   # VR-05: keep the big key/fill out of the weight's reflections
            fw = bpy.data.collections.get("VSP_Look_FlagBackdropWeight")
            if fw is None:
                fw = bpy.data.collections.new("VSP_Look_FlagBackdropWeight")
                for on in [set_objs[1].name] + list(L.get("weight_objects", ["VSP_Weight", "VSP_WeightEye"])):
                    if on in bpy.data.objects:
                        fw.objects.link(bpy.data.objects[on])
                for co in fw.collection_objects:
                    co.light_linking.link_state = "EXCLUDE"
            ob.light_linking.receiver_collection = fw
        if P.get("receivers"):                         # light-linked to named objects only (e.g. a strip for the weight)
            rc2 = bpy.data.collections.new(name + "_Receivers")
            for on in P["receivers"]:
                if on in bpy.data.objects:
                    rc2.objects.link(bpy.data.objects[on])
            ob.light_linking.receiver_collection = rc2
        if P.get("set_only"):
            rc = bpy.data.collections.get("VSP_Look_SetReceivers")
            if rc is None:
                rc = bpy.data.collections.new("VSP_Look_SetReceivers")
                for s_ in set_objs:
                    rc.objects.link(s_)
            ob.light_linking.receiver_collection = rc
            ld.use_shadow = P.get("shadow", False)
        rep[name] = {"location_m": list(ob.location), "energy_W": ld.energy, "size_m": ld.size}
    report["lights"] = rep


def _world(sc, Wd):
    w = bpy.data.worlds.get("VSP_Look_World") or bpy.data.worlds.new("VSP_Look_World")
    bg = w.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (*Wd["color"], 1.0)
    bg.inputs["Strength"].default_value = Wd["strength"]
    w.color = Wd["color"]
    sc.world = w


def _render(sc, R):
    vs = sc.view_settings
    vs.view_transform = R["view_transform"]
    vs.look = R["look"]
    vs.exposure, vs.gamma = R["exposure"], R["gamma"]
    sc.render.engine = "BLENDER_EEVEE"
    sc.render.filepath = R.get("output_path", "//render/")
    ee = sc.eevee
    for k, v in R["eevee"].items():
        if hasattr(ee, k):
            try:
                setattr(ee, k, v)
            except Exception:
                pass
    cy = R["cycles"]
    c = sc.cycles
    c.samples = cy["samples"]
    c.use_denoising = cy["use_denoising"]
    c.max_bounces = cy["max_bounces"]
    c.glossy_bounces = cy["glossy_bounces"]
    c.device = cy["device"]


def load(root):
    p = os.environ.get("VSP_LOOK_JSON") or str(pathlib.Path(root) / "params" / "look.json")
    return p, json.loads(pathlib.Path(p).read_text())


def apply_look(scene, params):
    root = params["root"]
    path, LK = load(root)
    D = params["dims_m"]
    S = params["S_m"]
    k_energy = (S / LK["S_ref_m"]) ** 2
    report = {"look_json": pathlib.Path(path).name, "version": LK["version"]}
    M = params["materials"]
    W = LK["walnut"]
    _walnut(bpy.data.materials[M["walnut"]], W, S)
    _satin(bpy.data.materials[M["blacksteel"]], LK["black_steel"], S)
    if "barsteel" in M:
        _satin(bpy.data.materials[M["barsteel"]], LK.get("crossbar_steel", LK["black_steel"]), S)
    _satin(bpy.data.materials[M["screw"]], LK["screw_steel"], S)
    _plain_metal(bpy.data.materials[M["spring"]], LK["spring_steel"])
    _brushed(bpy.data.materials[M["weight"]], LK["brushed_steel"], S)
    _plain_metal(bpy.data.materials[M["eye"]], LK["eye_steel"])
    ob = bpy.data.objects[params["objects"]["posts"]]
    ob["vsp_grain_axis"] = [float(x) for x in W["grain_axis"]]
    ob["vsp_end_axis"] = [float(x) for x in W["end_axis"]]
    ob["vsp_wood_seed"] = float(W["seed"])
    ob["vsp_wood_tone"] = float(W["tone"])
    ob["vsp_wood_freq"] = float(W["freq_scale"])
    coll = _coll(scene)
    table, wall = _set(scene, LK["set"], D, coll, report)
    _lights(scene, LK["lights"], D, coll, [table, wall], report, k_energy)
    _world(scene, LK["world"])
    for pname, pr in sorted(LK.get("probes", {}).items()):
        if not pr.get("enabled", True):
            continue
        lp = bpy.data.lightprobes.new(pname, "SPHERE")
        lp.influence_distance = pr["influence_S"] * S
        if hasattr(lp, "clip_start"):
            lp.clip_start = pr.get("clip_start_S", 0.05) * S
        po = bpy.data.objects.new(pname, lp)
        coll.objects.link(po)
        po.location = Vector(pr["location_S"]) * S
    _render(scene, LK["render"])
    scene["vsp_look_report"] = json.dumps(report, default=list)
    print("VSP_LOOK", json.dumps(report, default=list)[:400])
    return report
