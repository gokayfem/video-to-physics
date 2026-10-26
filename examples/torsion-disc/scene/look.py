"""Look (materials, textures, lighting, set) for clip 03 (torsion disc) — scene role C.

Called by scene/build.py --look as apply_look(scene, params). Every number comes from params/look.json (or env
TOR_LOOK_JSON for experiments). Deterministic: procedural textures with fixed seeds stored in look.json.
Lengths in look.json are in D (disc diameter; params["D_m"] metres); light energies are W at D_ref_m and scale with
(D_m / D_ref_m)^2. Helpers (NT, walnut, satin, plain metal) are the clip-01/02 shaders.

Materials are edited IN PLACE (builder names):
  TOR_Walnut      procedural walnut (clip-01/02 shader: fibre streaks + figure bands + cathedral arcs + pores + warp);
                  per-object grain via custom properties tor_grain_axis / tor_end_axis / tor_wood_seed / tor_wood_tone /
                  tor_wood_freq (posts: Z, crossbar: X, feet: Y)
  TOR_BlackSteel  clamp: satin black-oxide steel with faint orange-peel bump
  TOR_ScrewSteel  screws and collet: polished steel
  TOR_WireSteel   wire: drawn steel
  TOR_Brass       disc: lathe/turned brass. Anisotropic GGX with the tangent CIRCUMFERENTIAL about the disc's local Z
                  (Tangent node RADIAL, axis Z, object space) -> the radial highlight 'cones' of a turned face; fine
                  concentric streaks (noise of the local radius) in roughness/colour/bump; the engraved diameter: a
                  groove of width tor_mark_width_m along local direction tor_mark_azimuth_rad, top face only, dark core +
                  bump walls. Everything is in the disc's OBJECT space, so texture and mark rotate with the disc.
Set (collection TOR_Look): TOR_Look_Table (lavender-grey matte, top at the feet bottom), TOR_Look_Backdrop (wall at
  set.wall_y_D), optional cove; lights TOR_Look_* (receivers / exclusions by light linking), world, colour management.
"""
import json
import math
import os
import pathlib

import bpy
from mathutils import Vector

LOOK_COLL = "TOR_Look"


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
    fscale = T.attr_obj("tor_wood_freq", "Fac")
    P = T.vmath("SCALE", tc.outputs["Object"], scale=T.math("DIVIDE", fscale, S)).outputs["Vector"]
    geo = T.n("ShaderNodeNewGeometry")
    N = geo.outputs["Normal"]
    A = T.attr_obj("tor_grain_axis")
    E = T.attr_obj("tor_end_axis")
    seed = T.attr_obj("tor_wood_seed", "Fac")
    tone = T.attr_obj("tor_wood_tone", "Fac")
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
    mat["tor_look"] = "procedural walnut (clip-01 v011 shader), see scene/look.py and params/look.json walnut"


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
    mat["tor_look"] = json.dumps(P, sort_keys=True)


def _screw_metal(mat, P, D):
    """Polished screw steel; the front button screw (object y < front_y_D, i.e. in front of the clamp's front plate) gets
    its own darker, rougher values (VR03-06: a dark socket-head screw, not a chrome dome)."""
    T = NT(mat)
    b = _principled(T)
    b.inputs["Metallic"].default_value = P["metallic"]
    tc = T.n("ShaderNodeTexCoord")
    sp = T.n("ShaderNodeSeparateXYZ")
    T.link(tc.outputs["Object"], sp.inputs[0])
    fr = T.math("LESS_THAN", sp.outputs["Y"], P["front_y_D"] * D)
    col = T.mixrgb(fr, P["base_color"], P["front_base_color"])
    T.link(col, b.inputs["Base Color"])
    T.link(T.math("ADD", P["roughness"], T.math("MULTIPLY", fr, P["front_roughness"] - P["roughness"])), b.inputs["Roughness"])
    mat.diffuse_color = (*P["base_color"], 1.0)
    mat["tor_look"] = json.dumps(P, sort_keys=True)


def _plain_metal(mat, P):
    T = NT(mat)
    b = _principled(T)
    b.inputs["Base Color"].default_value = (*P["base_color"], 1.0)
    b.inputs["Metallic"].default_value = P["metallic"]
    b.inputs["Roughness"].default_value = P["roughness"]
    if "anisotropic" in P:
        b.inputs["Anisotropic"].default_value = P["anisotropic"]
    mat.diffuse_color = (*P["base_color"], 1.0)
    mat["tor_look"] = json.dumps(P, sort_keys=True)


# ---------------------------------------------------------------- brass disc
def _brass(mat, P, D):
    T = NT(mat)
    b = _principled(T)
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Specular IOR Level"].default_value = P.get("specular_ior_level", 0.5)
    tan = T.n("ShaderNodeTangent", direction_type="RADIAL", axis="Z")
    T.link(tan.outputs["Tangent"], b.inputs["Tangent"])
    b.inputs["Anisotropic"].default_value = P["anisotropic"]
    tc = T.n("ShaderNodeTexCoord")
    sep = T.n("ShaderNodeSeparateXYZ")
    T.link(tc.outputs["Object"], sep.inputs[0])
    x, y, z = sep.outputs["X"], sep.outputs["Y"], sep.outputs["Z"]
    r = T.math("SQRT", T.math("ADD", T.math("MULTIPLY", x, x), T.math("MULTIPLY", y, y)))
    rD = T.math("DIVIDE", r, D)
    ang = T.math("ARCTAN2", y, x)
    # concentric turning streaks: noise of the radius (+ a little angle so rings are not perfect)
    v1 = T.combine(T.math("MULTIPLY", rD, P["streak_freq"]), T.math("MULTIPLY", ang, P["streak_freq_along"]), 3.1)
    streak = T.noise(v1, 1.0, P["streak_detail"], 0.6)
    RS = P.get("radial_streaks")
    if RS:                                                     # fine radial texture of the turned face (rotates with the disc)
        vr = T.combine(T.math("MULTIPLY", ang, RS["freq"] / (2 * math.pi)), T.math("MULTIPLY", rD, RS.get("freq_r", 2.0)), 7.7)
        rstreak = T.noise(vr, 1.0, RS.get("detail", 6.0), 0.6)
    # top face (local z ~ 0) vs rim/bottom
    top = T.maprange(z, -0.25 * P["top_band_m_scale"] * D, -0.05 * P["top_band_m_scale"] * D, 0.0, 1.0)
    # engraved diameter: across-line coordinate v = -x sin(mu) + y cos(mu)
    mu = T.attr_obj("tor_mark_azimuth_rad", "Fac")
    w = T.attr_obj("tor_mark_width_m", "Fac")
    v = T.math("ADD", T.math("MULTIPLY", T.math("MULTIPLY", x, -1.0), T.math("SINE", mu)), T.math("MULTIPLY", y, T.math("COSINE", mu)))
    av = T.math("ABSOLUTE", v)
    hw = T.math("MULTIPLY", w, 0.5)
    # groove profile: 1 in the core, smooth V walls out to half width (+ AA margin)
    core = T.math("SUBTRACT", 1.0, T.math("DIVIDE", av, T.math("MAXIMUM", hw, 1e-7)), clamp=True)
    groove = T.math("MULTIPLY", T.maprange(core, 0.0, P["mark_wall"], 0.0, 1.0, interp="SMOOTHSTEP"), top)
    inside = T.math("LESS_THAN", rD, 0.5 - P["mark_end_inset"])
    groove = T.math("MULTIPLY", groove, inside)
    # base colour: brass, streak modulation, darker groove
    colk = T.maprange(streak, 0.3, 0.7, 1.0 - P["color_streak_amp"], 1.0 + P["color_streak_amp"])
    col = T.mixrgb(1.0, P["base_color"], T.combine(colk, colk, colk), blend="MULTIPLY")
    VL = P.get("view_lobes", {})
    if VL.get("enabled"):
        # EEVEE has no anisotropic GGX: emulate the turned-face 'cones' (bright where the radial direction lines up with
        # the view azimuth, i.e. along the image minor axis), computed in the disc's object space so it is invariant
        # under twist exactly like a real turned finish; Cycles would get the same from the anisotropic BSDF.
        geo = T.n("ShaderNodeNewGeometry")
        vt = T.n("ShaderNodeVectorTransform", vector_type="VECTOR", convert_from="WORLD", convert_to="OBJECT")
        T.link(geo.outputs["Incoming"], vt.inputs["Vector"])
        si = T.n("ShaderNodeSeparateXYZ")
        T.link(vt.outputs["Vector"], si.inputs[0])
        ixy = T.vmath("NORMALIZE", T.combine(si.outputs["X"], si.outputs["Y"], 0.0)).outputs["Vector"]
        rxy = T.vmath("NORMALIZE", T.combine(x, y, 0.0)).outputs["Vector"]
        c = T.math("ABSOLUTE", T.vmath("DOT_PRODUCT", ixy, rxy).outputs["Value"])
        if "edge_deg" in VL:                                   # flat-topped cone with soft edges (reference look)
            e0, e1 = VL["edge_deg"]
            lobe = T.maprange(c, math.cos(math.radians(e1)), math.cos(math.radians(e0)), 0.0, 1.0, interp="SMOOTHSTEP")
        else:
            lobe = T.math("POWER", c, VL["power"])
        # soft centre (the cones meet in a small bright/dark star at the centre)
        k = T.math("ADD", VL["dark"], T.math("MULTIPLY", lobe, VL["bright"] - VL["dark"]))
        k = T.math("ADD", T.math("MULTIPLY", k, top), T.math("SUBTRACT", 1.0, top))
        col = T.mixrgb(1.0, col, T.combine(k, k, k), blend="MULTIPLY")
    if RS:
        kr = T.maprange(rstreak, 0.3, 0.7, 1.0 - RS["amp"], 1.0 + RS["amp"])
        kr = T.math("ADD", T.math("MULTIPLY", T.math("SUBTRACT", kr, 1.0), top), 1.0)
        col = T.mixrgb(1.0, col, T.combine(kr, kr, kr), blend="MULTIPLY")
    col = T.mixrgb(T.math("MULTIPLY", groove, P["mark_darkness"]), col, P["mark_color"])
    rim_col = P.get("rim_color", P["base_color"])
    col = T.mixrgb(T.math("SUBTRACT", 1.0, top), col, rim_col)
    T.link(col, b.inputs["Base Color"])
    rough_top = T.maprange(streak, 0.35, 0.65, P["roughness"] - P["roughness_streak_amp"], P["roughness"] + P["roughness_streak_amp"])
    rough = T.math("ADD", T.math("MULTIPLY", rough_top, top), T.math("MULTIPLY", P["rim_roughness"], T.math("SUBTRACT", 1.0, top)))
    rough = T.math("ADD", rough, T.math("MULTIPLY", groove, P["mark_roughness"] - P["roughness"]))
    T.link(rough, b.inputs["Roughness"])
    aniso = T.math("MULTIPLY", P["anisotropic"], top)
    T.link(aniso, b.inputs["Anisotropic"])
    if "metallic_top" in P:                                    # a little diffuse brass on the top face (warm cone gaps)
        T.link(T.math("ADD", T.math("MULTIPLY", P["metallic_top"], top), T.math("SUBTRACT", 1.0, top)), b.inputs["Metallic"])
    height = T.math("SUBTRACT", T.math("MULTIPLY", streak, P["bump_streak"]), T.math("MULTIPLY", groove, P.get("mark_bump", 1.0)))
    _bump(T, b, height, P["bump_strength"], P["mark_depth_D"] * D)
    mat.diffuse_color = (*P["base_color"], 1.0)
    mat["tor_look"] = json.dumps(P, sort_keys=True)


# ---------------------------------------------------------------- set
def _matte(mat, P, D):
    """Matte paper/fabric with a faint large-scale mottle and fine grain (table and backdrop)."""
    T = NT(mat)
    b = _principled(T)
    tc = T.n("ShaderNodeTexCoord")
    pos = T.vmath("SCALE", tc.outputs["Object"], scale=1.0 / D).outputs["Vector"]
    mot = T.noise(T.vmath("SCALE", pos, scale=P.get("mottle_freq", 0.5)).outputs["Vector"], 1.0, 2.0, 0.5)
    k = T.maprange(mot, 0.3, 0.7, 1.0 - P.get("mottle_amp", 0.0), 1.0 + P.get("mottle_amp", 0.0))
    col = T.mixrgb(1.0, P["color"], T.combine(k, k, k), blend="MULTIPLY")
    sp = T.n("ShaderNodeSeparateXYZ")
    T.link(pos, sp.inputs[0])
    for ax in ("Y", "Z"):                                      # optional linear albedo ramps (world D units; the set
        rp = P.get(f"ramp_{ax.lower()}")                       # objects sit at the origin, so object = world coords)
        if rp:
            kr = T.maprange(sp.outputs[ax], rp[0], rp[1], rp[2], rp[3], interp="SMOOTHSTEP")
            col = T.mixrgb(1.0, col, T.combine(kr, kr, kr), blend="MULTIPLY")
    T.link(col, b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = P["roughness"]
    b.inputs["Specular IOR Level"].default_value = P.get("specular_ior_level", 0.3)
    if P.get("bump_strength", 0.0) > 0:
        g = T.noise(T.vmath("SCALE", pos, scale=P.get("grain_freq", 300.0)).outputs["Vector"], 1.0, 2.0, 0.5)
        _bump(T, b, g, P["bump_strength"], P.get("bump_distance_D", 0.002) * D)
    mat.diffuse_color = (*P["color"], 1.0)
    mat["tor_look"] = json.dumps(P, sort_keys=True)


def _coll(sc):
    c = bpy.data.collections.get(LOOK_COLL) or bpy.data.collections.new(LOOK_COLL)
    if c.name not in sc.collection.children:
        sc.collection.children.link(c)
    return c


def _mesh_obj(name, verts, faces, coll, mat, smooth=False):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def _set(sc, Sd, Dd, coll, report):
    """Table plane at the feet bottom, wall at y_wall with a cylindrical cove of radius cove_D joining them."""
    D = Dd["D"]
    zt = Dd["table_z"]
    yw = Sd["wall_y_D"] * D
    hw, hh, front = Sd["half_width_D"] * D, Sd["wall_height_D"] * D, Sd["table_front_D"] * D
    rc = Sd.get("cove_D", 0.0) * D
    tmat = bpy.data.materials.get("TOR_Look_TableMat") or bpy.data.materials.new("TOR_Look_TableMat")
    _matte(tmat, Sd["table"], D)
    wmat = bpy.data.materials.get("TOR_Look_BackdropMat") or bpy.data.materials.new("TOR_Look_BackdropMat")
    _matte(wmat, Sd["backdrop"], D)
    table = _mesh_obj("TOR_Look_Table", [(-hw, -front, zt), (hw, -front, zt), (hw, yw - rc, zt), (-hw, yw - rc, zt)],
                      [(0, 1, 2, 3)], coll, tmat)
    wall = _mesh_obj("TOR_Look_Backdrop", [(-hw, yw, zt + rc), (hw, yw, zt + rc), (hw, yw, zt + hh), (-hw, yw, zt + hh)],
                     [(0, 1, 2, 3)], coll, wmat)
    objs = [table, wall]
    if rc > 0:
        n = 16
        verts, faces = [], []
        for i in range(n + 1):
            a = -math.pi / 2 + (math.pi / 2) * i / n                # a + 90 deg: 0 at the table, 90 at the wall
            yy, zz = yw - rc + rc * math.sin(a + math.pi / 2), zt + rc - rc * math.cos(a + math.pi / 2)
            verts += [(-hw, yy, zz), (hw, yy, zz)]
        for i in range(n):
            faces.append((2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2))
        cove = _mesh_obj("TOR_Look_Cove", verts, faces, coll, tmat if Sd.get("cove_material", "table") == "table" else wmat, smooth=True)
        objs.append(cove)
    for o in objs:
        if Sd.get("set_visible_glossy") is False:
            o.visible_glossy = False
    report["set"] = {"table_z_m": zt, "wall_y_m": yw, "cove_m": rc}
    return objs


def _sph(target, az, el, dist):
    a, e = math.radians(az), math.radians(el)
    return target + dist * Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))


def _aim(ob, target):
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = (target - ob.location).to_track_quat("-Z", "Y")


def _link_coll(name, obj_names, state=None):
    c = bpy.data.collections.new(name)
    for on in obj_names:
        if on in bpy.data.objects:
            c.objects.link(bpy.data.objects[on])
    if state:
        for co in c.collection_objects:
            co.light_linking.link_state = state
    return c


def _lights(sc, L, Dd, coll, report, k_energy):
    D = Dd["D"]
    target = Vector(L["target_D"]) * D
    rep = {}
    for key in sorted(k for k in L if isinstance(L[k], dict) and L[k].get("type") in ("AREA", "SPOT", "POINT")):
        P = L[key]
        name = "TOR_Look_" + "".join(w.capitalize() for w in key.split("_"))
        ld = bpy.data.lights.new(name, P["type"])
        if P["type"] == "AREA":
            ld.shape = P.get("shape", "DISK")
            if ld.shape in ("RECTANGLE", "ELLIPSE"):
                ld.size, ld.size_y = P["size_D"] * D, P.get("size_y_D", P["size_D"]) * D
            else:
                ld.size = P["size_D"] * D
            if "spread_deg" in P:
                ld.spread = math.radians(P["spread_deg"])
        elif P["type"] == "SPOT":
            ld.spot_size = math.radians(P.get("spot_deg", 45.0))
            ld.spot_blend = P.get("spot_blend", 0.5)
            ld.shadow_soft_size = P.get("radius_D", 0.2) * D
        else:
            ld.shadow_soft_size = P.get("radius_D", 0.2) * D
        ld.energy = P["energy_W"] * k_energy
        ld.color = P["color"]
        if P.get("shadow") is False:
            ld.use_shadow = False
        ob = bpy.data.objects.new(name, ld)
        coll.objects.link(ob)
        tgt = Vector(P["aim_D"]) * D if "aim_D" in P else target
        if "location_D" in P:
            ob.location = Vector(P["location_D"]) * D
        else:
            ob.location = _sph(tgt, P["azimuth_deg"], P["elevation_deg"], P["distance_D"] * D)
        _aim(ob, tgt)
        if P.get("follow_camera") and sc.camera is not None:
            # VR03-09: the reference wall gradient turns with the orbiting camera (AI lighting is camera-relative); a
            # wall-only light parented to TOR_Cam_Reference keeps its frame-0 placement relative to the camera
            sc.frame_set(0)
            ob.parent = sc.camera
            ob.matrix_parent_inverse = sc.camera.matrix_world.inverted()
            ob["tor_note"] = "parented to the scene camera (camera-relative wall wash, light-linked to the set only)"
        if P.get("receivers"):
            ob.light_linking.receiver_collection = _link_coll(name + "_Receivers", P["receivers"])
        elif P.get("exclude"):
            ob.light_linking.receiver_collection = _link_coll(name + "_Exclude", P["exclude"], "EXCLUDE")
        if P.get("shadow_exclude"):
            ob.light_linking.blocker_collection = _link_coll(name + "_NoShadow", P["shadow_exclude"], "EXCLUDE")
        rep[name] = {"location_m": [round(v, 6) for v in ob.location], "energy_W": ld.energy}
    report["lights"] = rep


def _world(sc, Wd):
    w = bpy.data.worlds.get("TOR_Look_World") or bpy.data.worlds.new("TOR_Look_World")
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
    p = os.environ.get("TOR_LOOK_JSON") or str(pathlib.Path(root) / "params" / "look.json")
    return p, json.loads(pathlib.Path(p).read_text())


def apply_look(scene, params):
    root = params["root"]
    path, LK = load(root)
    Dd = params["dims_m"]
    D = params["D_m"]
    k_energy = (D / LK["D_ref_m"]) ** 2
    report = {"look_json": pathlib.Path(path).name, "version": LK["version"]}
    M = params["materials"]
    W = LK["walnut"]
    _walnut(bpy.data.materials[M["walnut"]], W, D)
    _satin(bpy.data.materials[M["blacksteel"]], LK["black_steel"], D)
    if "front_y_D" in LK["screw_steel"]:
        _screw_metal(bpy.data.materials[M["screw"]], LK["screw_steel"], D)
    else:
        _plain_metal(bpy.data.materials[M["screw"]], LK["screw_steel"])
    if "collet" in M and "collet_steel" in LK:
        _plain_metal(bpy.data.materials[M["collet"]], LK["collet_steel"])
    _plain_metal(bpy.data.materials[M["wire"]], LK["wire_steel"])
    _brass(bpy.data.materials[M["brass"]], LK["brass"], D)
    O = params["objects"]
    for key, part in (("posts", "posts"), ("bar", "crossbar"), ("feet", "feet")):
        ob = bpy.data.objects[O[key]]
        Wp = W["parts"][part]
        ob["tor_grain_axis"] = [float(x) for x in Wp["grain_axis"]]
        ob["tor_end_axis"] = [float(x) for x in Wp["end_axis"]]
        ob["tor_wood_seed"] = float(Wp["seed"])
        ob["tor_wood_tone"] = float(Wp["tone"])
        ob["tor_wood_freq"] = float(Wp.get("freq_scale", 1.0))
    coll = _coll(scene)
    _set(scene, LK["set"], Dd, coll, report)
    _lights(scene, LK["lights"], Dd, coll, report, k_energy)
    _world(scene, LK["world"])
    for pname, pr in sorted(LK.get("probes", {}).items()):
        if not pr.get("enabled", True):
            continue
        lp = bpy.data.lightprobes.new(pname, "SPHERE")
        lp.influence_distance = pr["influence_D"] * D
        po = bpy.data.objects.new(pname, lp)
        coll.objects.link(po)
        po.location = Vector(pr["location_D"]) * D
    _render(scene, LK["render"])
    scene["tor_look_report"] = json.dumps(report, default=list)
    print("TOR_LOOK", json.dumps(report, default=list)[:300])
    return report
