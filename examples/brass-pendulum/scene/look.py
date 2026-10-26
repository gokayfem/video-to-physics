"""Look (materials, textures, lighting) for the brass-pendulum scene — owned by the Look role (CONTRACT v3).

Called by scene/build.py --look as apply_look(scene, params). Every number comes from params/look.json
(path: <root>/params/look.json, or env PND_LOOK_JSON for experiments). Deterministic: no randomness besides fixed
noise seeds stored in look.json.

What it does
* Edits the builder's materials IN PLACE (same names): PND_Walnut (procedural walnut), PND_Brass, PND_Steel,
  PND_WireSteel.
* Walnut: object coordinates (= world coordinates for the stand parts) scaled by 1/L_c. Each wood object carries
  custom properties read by Attribute(OBJECT) nodes:
    pnd_grain_axis  (vector) fibre direction of that part (base/arm: X, post: Z)
    pnd_end_axis    (vector) stripe direction used on end-grain faces
    pnd_wood_seed   (float)  noise offset so parts differ
    pnd_wood_tone   (float)  per-part brightness multiplier
  On side faces the across-grain coordinate is dot(P, N x A)/|N x A| (box projection that follows rounded edges),
  on end faces (|N.A| large) dot(P, E). Fibre streaks = noise stretched along A; broad figure bands = coarse
  stretched noise; pores = thresholded fine stretched noise (darker, rougher); domain warp for wavy lines.
* Set: PND_Look_Table (top plane at the base bottom) and PND_Look_Backdrop (vertical wall perpendicular to the
  camera's horizontal view direction; its crease is placed at a given image row) in collection PND_Look.
* Lights (PND_Look_Key / _RimRight / _FillFront area lights aimed at a target, PND_Look_BackdropSpot light-linked
  to the set only), world (dark), colour management, EEVEE and Cycles settings.
"""
import json
import math
import os
import pathlib

import bpy
from mathutils import Vector

LOOK_COLL = "PND_Look"


# ---------------------------------------------------------------- small node helpers
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


def _metal(mat, P):
    T = NT(mat)
    b = _principled(T)
    b.inputs["Base Color"].default_value = (*P["base_color"], 1.0)
    b.inputs["Metallic"].default_value = P["metallic"]
    b.inputs["Roughness"].default_value = P["roughness"]
    if "anisotropic" in P:
        b.inputs["Anisotropic"].default_value = P["anisotropic"]
    mat.diffuse_color = (*P["base_color"], 1.0)
    mat["pnd_look"] = json.dumps(P, sort_keys=True)


# ---------------------------------------------------------------- walnut
def _walnut(mat, W, Lc):
    T = NT(mat)
    b = _principled(T)
    tc = T.n("ShaderNodeTexCoord")
    fscale = T.attr_obj("pnd_wood_freq", "Fac")                                         # per-part grain scale
    P = T.vmath("SCALE", tc.outputs["Object"], scale=T.math("DIVIDE", fscale, Lc)).outputs["Vector"]  # coords in L_c
    geo = T.n("ShaderNodeNewGeometry")
    N = geo.outputs["Normal"]                       # world normal (stand objects are unrotated: world = object)
    A = T.attr_obj("pnd_grain_axis")
    E = T.attr_obj("pnd_end_axis")
    seed = T.attr_obj("pnd_wood_seed", "Fac")
    tone = T.attr_obj("pnd_wood_tone", "Fac")

    C = T.vmath("CROSS_PRODUCT", N, A).outputs["Vector"]
    Clen = T.vmath("LENGTH", C).outputs["Value"]
    s_side = T.math("DIVIDE", T.vmath("DOT_PRODUCT", P, C).outputs["Value"], T.math("MAXIMUM", Clen, 0.05))
    s_end = T.vmath("DOT_PRODUCT", P, E).outputs["Value"]
    ndota = T.math("ABSOLUTE", T.vmath("DOT_PRODUCT", N, A).outputs["Value"])
    w_end = T.maprange(ndota, 0.6, 0.85, interp="SMOOTHSTEP")
    s = T.math("ADD", T.math("MULTIPLY", s_side, T.math("SUBTRACT", 1.0, w_end)), T.math("MULTIPLY", s_end, w_end))
    t = T.vmath("DOT_PRODUCT", P, A).outputs["Value"]

    # domain warp: wavy lines
    wv = T.combine(T.math("MULTIPLY", s, W["warp_freq_per_Lc"] * 0.5), T.math("MULTIPLY", t, W["warp_freq_per_Lc"]), seed)
    warp = T.math("MULTIPLY", T.math("SUBTRACT", T.noise(wv, 1.0, 3.0, 0.5), 0.5), W["warp_amp_Lc"] * 2.0)
    sw = T.math("ADD", s, warp)

    # fibre streaks (fine, long)
    v1 = T.combine(T.math("MULTIPLY", sw, W["streak_freq_across_per_Lc"]), T.math("MULTIPLY", t, W["streak_freq_along_per_Lc"]),
                   T.math("ADD", seed, 3.3))
    streak = T.noise(v1, 1.0, W["streak_detail"], 0.55)
    streak = T.maprange(streak, 0.5 - 0.25 / W["streak_contrast"], 0.5 + 0.25 / W["streak_contrast"])
    # broad figure bands
    v2 = T.combine(T.math("MULTIPLY", sw, W["band_freq_across_per_Lc"]), T.math("MULTIPLY", t, W["band_freq_along_per_Lc"]),
                   T.math("ADD", seed, 7.7))
    band = T.maprange(T.noise(v2, 1.0, 2.0, 0.5), 0.3, 0.7)
    fig = T.math("ADD", T.math("MULTIPLY", streak, 1.0 - W["band_weight"]), T.math("MULTIPLY", band, W["band_weight"]))
    # isotropic fine mottle (breaks the too-perfect streak coherence; reference coherence ~0.86-0.89)
    if W.get("mottle_amount", 0.0) > 0:
        v4 = T.vmath("SCALE", P, scale=W["mottle_freq_per_Lc"]).outputs["Vector"]
        mot = T.math("SUBTRACT", T.noise(v4, 1.0, 3.0, 0.6), 0.5)
        fig = T.math("ADD", fig, T.math("MULTIPLY", mot, W["mottle_amount"] * 2.0), clamp=True)
    # pores (short dark dashes along the grain)
    v3 = T.combine(T.math("MULTIPLY", sw, W["pore_freq_across_per_Lc"]), T.math("MULTIPLY", t, W["pore_freq_along_per_Lc"]),
                   T.math("ADD", seed, 11.1))
    pore = T.maprange(T.noise(v3, 1.0, 2.0, 0.6), W["pore_threshold"], W["pore_threshold"] + 0.08, interp="SMOOTHSTEP")
    pore = T.math("MULTIPLY", pore, W["pore_amount"])

    # end-grain faces: reduced figure contrast (reference base end reads almost featureless)
    ec = W.get("end_contrast", 1.0)
    if ec != 1.0:
        fig_end = T.math("ADD", 0.5, T.math("MULTIPLY", T.math("SUBTRACT", fig, 0.5), ec))
        fig = T.math("ADD", T.math("MULTIPLY", fig, T.math("SUBTRACT", 1.0, w_end)), T.math("MULTIPLY", fig_end, w_end))
    # upward faces (base top): stronger figure contrast, as the reference's flat-sawn top reads
    tcon = W.get("top_contrast", 1.0)
    if tcon != 1.0:
        nz = T.n("ShaderNodeSeparateXYZ")
        T.link(N, nz.inputs[0])
        w_top = T.maprange(nz.outputs["Z"], 0.6, 0.9, interp="SMOOTHSTEP")
        k_top = T.math("ADD", 1.0, T.math("MULTIPLY", w_top, tcon - 1.0))
        fig = T.math("ADD", 0.5, T.math("MULTIPLY", T.math("SUBTRACT", fig, 0.5), k_top), clamp=True)
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

    bump = T.n("ShaderNodeBump")
    bump.inputs["Strength"].default_value = W["bump_strength"]
    bump.inputs["Distance"].default_value = W["bump_distance_Lc"] * Lc
    T.link(T.math("SUBTRACT", fig, T.math("MULTIPLY", pore, 0.6)), bump.inputs["Height"])
    T.link(bump.outputs["Normal"], b.inputs["Normal"])
    mat.diffuse_color = (*W["color_mid"], 1.0)
    mat["pnd_look"] = "procedural walnut, see scene/look.py and params/look.json walnut"


# ---------------------------------------------------------------- set, lights, world
def _coll(sc):
    c = bpy.data.collections.get(LOOK_COLL) or bpy.data.collections.new(LOOK_COLL)
    if c.name not in sc.collection.children:
        sc.collection.children.link(c)
    return c


def _plain_mat(name, color, rough):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    T = NT(m)
    b = _principled(T)
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    m.diffuse_color = (*color, 1.0)
    return m


def _quad(name, verts, coll, mat):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], [tuple(range(len(verts)))])
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def _cam_frame(sc):
    bpy.context.view_layer.update()      # the builder's camera matrix_world is stale until evaluated
    cam = sc.camera
    M = cam.matrix_world
    fwd = -(M.to_3x3() @ Vector((0, 0, 1)))
    W, H = sc.render.resolution_x, sc.render.resolution_y
    fpx = cam.data.lens / cam.data.sensor_width * W

    cx = W / 2 - cam.data.shift_x * max(W, H)       # principal point (Blender lens shift, sensor_fit HORIZONTAL)
    cy = H / 2 + cam.data.shift_y * max(W, H)

    def ray(u, v):
        return (M.to_3x3() @ Vector(((u - cx) / fpx, -(v - cy) / fpx, -1.0))).normalized()
    return M.translation.copy(), fwd, ray


def _set(sc, S, D, coll, report):
    Lc = D["L_c"]
    zt = -D["h_pivot"] - D["base_thickness"]
    o, fwd, ray = _cam_frame(sc)
    if S.get("wall_yaw_from_camera", True):
        h = Vector((fwd.x, fwd.y, 0)).normalized()
    else:
        a = math.radians(S["wall_yaw_deg"])
        h = Vector((-math.sin(a), math.cos(a), 0))
    side = Vector((h.y, -h.x, 0))                     # camera-right in the horizontal plane
    if S.get("crease_v_px") is not None:
        d = ray(sc.render.resolution_x / 2, S["crease_v_px"])
        tt = (zt - o.z) / d.z
        pc = o + tt * d
        dist = pc.dot(h)                              # distance of the crease from P along h
    else:
        dist = S["wall_distance_Lc"] * Lc
    hw, hh, front = S["wall_half_width_Lc"] * Lc, S["wall_height_Lc"] * Lc, S["table_extent_front_Lc"] * Lc
    c0 = h * dist
    p = lambda s_, f_, z_: tuple(c0 + side * s_ + h * f_ + Vector((0, 0, z_)))
    tmat = _plain_mat("PND_Look_TableMat", S["table_color"], S["table_roughness"])
    wmat = _plain_mat("PND_Look_BackdropMat", S["backdrop_color"], S["backdrop_roughness"])
    table = _quad("PND_Look_Table", [p(-hw, -front, zt), p(hw, -front, zt), p(hw, 0, zt), p(-hw, 0, zt)], coll, tmat)
    wall = _quad("PND_Look_Backdrop", [p(-hw, 0, zt), p(hw, 0, zt), p(hw, 0, zt + hh), p(-hw, 0, zt + hh)], coll, wmat)
    report["set"] = {"table_z_m": zt, "wall_dir_xy": [h.x, h.y], "wall_distance_m": dist, "wall_distance_Lc": dist / Lc}
    return table, wall, (c0, h, zt)


def _sph(target, az, el, dist):
    a, e = math.radians(az), math.radians(el)
    return target + dist * Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))


def _aim(ob, target):
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = (target - ob.location).to_track_quat("-Z", "Y")


def _lights(sc, L, D, coll, set_objs, wallinfo, report, k_energy):
    Lc = D["L_c"]
    target = Vector(L["target_Lc"]) * Lc
    rep = {}
    # "flag": a receiver collection that EXCLUDES the backdrop wall (every other object stays lit), like a
    # flag/gobo keeping the subject lights off the sweep so the backdrop pool comes from the backdrop spot.
    flagged = bpy.data.collections.new("PND_Look_FlagBackdrop")
    flagged.objects.link(set_objs[1])
    flagged.collection_objects[0].light_linking.link_state = "EXCLUDE"
    for key, name in (("key", "PND_Look_Key"), ("softbox", "PND_Look_Softbox"), ("rim_right", "PND_Look_RimRight"),
                      ("fill_front", "PND_Look_FillFront")):
        if key not in L:
            continue
        P = L[key]
        ld = bpy.data.lights.new(name, "AREA")
        ld.shape = P.get("shape", "DISK")
        ld.size = P["size_Lc"] * Lc
        ld.energy = P["energy_W"] * k_energy
        ld.color = P["color"]
        ob = bpy.data.objects.new(name, ld)
        coll.objects.link(ob)
        ob.location = _sph(target, P["azimuth_deg"], P["elevation_deg"], P["distance_Lc"] * Lc)
        _aim(ob, target)
        if P.get("flag_off_backdrop") and flagged is not None:
            ob.light_linking.receiver_collection = flagged
        rep[name] = {"location_m": list(ob.location), "energy_W": ld.energy, "size_m": ld.size,
                     "flag_off_backdrop": bool(P.get("flag_off_backdrop"))}
    o, fwd, ray = _cam_frame(sc)
    c0, h, zt = wallinfo
    side = Vector((h.y, -h.x, 0))
    rc = None
    for key in sorted(k for k in L if k.startswith("backdrop_")):
        P = L[key]
        name = "PND_Look_" + "".join(w.capitalize() for w in key.split("_"))
        if P.get("type", "SPOT") == "AREA":
            ld = bpy.data.lights.new(name, "AREA")
            ld.shape, ld.size = "DISK", P["size_Lc"] * Lc
        else:
            ld = bpy.data.lights.new(name, "SPOT")
            ld.spot_size, ld.spot_blend = math.radians(P["spot_size_deg"]), P["spot_blend"]
            ld.shadow_soft_size = P["radius_Lc"] * Lc
        ld.energy = P["energy_W"] * k_energy
        ld.color = P["color"]
        ob = bpy.data.objects.new(name, ld)
        coll.objects.link(ob)
        d = ray(*P["aim_px_on_wall"])           # the wall point seen at the given pixel
        aim = o + ((c0.dot(h) - o.dot(h)) / d.dot(h)) * d
        if "standoff_Lc" in P:                  # facing the wall from a distance in front of it (soft pool)
            ob.location = aim - h * (P["standoff_Lc"] * Lc)
        elif "location_from_camera_Lc" in P:      # [right, up, back] relative to the scene camera, horizontal frame
            r_, u_, b_ = P["location_from_camera_Lc"]
            ob.location = o + side * (r_ * Lc) + Vector((0, 0, u_ * Lc)) - h * (b_ * Lc)
        else:
            ob.location = Vector(P["location_Lc"]) * Lc
        _aim(ob, aim)
        if P.get("link_to_set_only", True):
            if rc is None:
                rc = bpy.data.collections.new("PND_Look_SetReceivers")
                for s_ in set_objs:
                    rc.objects.link(s_)
            ob.light_linking.receiver_collection = rc
            ld.use_shadow = False
        rep[name] = {"location_m": list(ob.location), "aim_m": list(aim), "energy_W": ld.energy}
    report["lights"] = rep


def _world(sc, Wd):
    w = bpy.data.worlds.get("PND_Look_World") or bpy.data.worlds.new("PND_Look_World")
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
    sc.render.filepath = R.get("output_path", "//render/")   # relative to the .blend (delivery finding F5)
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
    p = os.environ.get("PND_LOOK_JSON") or str(pathlib.Path(root) / "params" / "look.json")
    return p, json.loads(pathlib.Path(p).read_text())


def apply_look(scene, params):
    root = params["root"]
    path, LK = load(root)
    D = params["dims_m"]
    Lc = params["L_c_m"]
    k_energy = (Lc / LK["L_ref_m"]) ** 2
    report = {"look_json": path, "version": LK["version"]}
    M = params["materials"]
    W = LK["walnut"]
    _walnut(bpy.data.materials[M["walnut"]], W, Lc)
    _metal(bpy.data.materials[M["brass"]], LK["brass"])
    _metal(bpy.data.materials[M["steel"]], LK["steel"])
    _metal(bpy.data.materials[M["wire"]], LK["wire"])
    for name in ("PND_Base", "PND_Post", "PND_Arm"):
        ob = bpy.data.objects[name]
        ob["pnd_grain_axis"] = [float(x) for x in W["grain_axis_per_object"][name]]
        ob["pnd_end_axis"] = [float(x) for x in W["end_stripe_axis_per_object"][name]]
        ob["pnd_wood_seed"] = float(W["seed_per_object"][name])
        ob["pnd_wood_tone"] = float(W["tone_per_object"][name])
        ob["pnd_wood_freq"] = float(W["freq_scale_per_object"][name])
    coll = _coll(scene)
    table, wall, wallinfo = _set(scene, LK["set"], D, coll, report)
    _lights(scene, LK["lights"], D, coll, [table, wall], wallinfo, report, k_energy)
    _world(scene, LK["world"])
    for pname, pr in sorted(LK.get("probes", {}).items()):
        if not pr.get("enabled", True):
            continue
        lp = bpy.data.lightprobes.new(pname, "SPHERE")
        lp.influence_distance = pr["influence_Lc"] * Lc
        if hasattr(lp, "clip_start"):
            lp.clip_start = pr.get("clip_start_Lc", 0.12) * Lc
        po = bpy.data.objects.new(pname, lp)
        coll.objects.link(po)
        po.location = Vector(pr["location_Lc"]) * Lc    # EEVEE only; Cycles ignores probes
        report.setdefault("probes", {})[pname] = {"location_m": list(po.location), "influence_m": lp.influence_distance}
    _render(scene, LK["render"])
    scene["pnd_look_report"] = json.dumps(report, default=list)
    print("PND_LOOK", json.dumps(report, default=list))
    return report
