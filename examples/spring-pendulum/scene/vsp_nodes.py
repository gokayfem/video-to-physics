"""Geometry Nodes groups for clip 02 (vertical spring), built with the vendored nodekit.

Groups
  VSP Rounded Block  box with separately rounded vertical corners, top edges and bottom edges (from clip 01)
  VSP Frame          (modifier) two walnut posts + black-steel crossbar + black-steel base plate + 4 screw heads,
                     laid out around A (origin, bar top under the spring). Face attribute `vsp_part`
                     (0 posts, 1 crossbar, 2 base plate, 3 screws). Nothing reads time.
  VSP Frame Part     (modifier) one part of a VSP Frame object, picked by `vsp_part`
  VSP Spring         (modifier) the whole spring wire: top loop around the crossbar, top transition, open and
                     tight coil sections, bottom transition and the lower loop that holds the weight's eye.
                     Driven ONLY by the position H of the `Lower End` object relative to the spring object (A =
                     the spring object's origin): the spring axis is A->H, the coil COUNT is fixed, every coil
                     takes the same extension (uniform stiffness per turn), tight coils never go below solid
                     pitch. Rigid end hardware rotates with the axis. Point attributes on the evaluated mesh:
                     vsp_part (INT: 0 top loop, 1 top transition, 2 open coils, 3 tight coils, 4 bottom
                     transition, 5 lower loop) and vsp_cl (centreline point of each ring, world = object space).
All lengths are metres in object space (= world space; the objects are unrotated at the origin).
"""
import math
import pathlib
import sys

VENDOR = str(pathlib.Path(__file__).resolve().parent / "vendor")
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

from nodekit.graph import Graph, once  # noqa: E402

PART_IDS = {"posts": 0, "bar": 1, "plate": 2, "screws": 3}
SPRING_PARTS = {"top_loop": 0, "top_transition": 1, "coils_open": 2, "coils_tight": 3, "bottom_transition": 4,
                "lower_loop": 5}


def _xyz(g, x, y, z, label=None):
    return g.combine(x, y, z, label=label)


def _cmp(g, op, a, b, dtype="FLOAT"):
    return g.node("FunctionNodeCompare", props={"data_type": dtype, "operation": op}, A=a, B=b)["Result"]


@once
def rounded_block():
    g = Graph("VSP Rounded Block", description="Box of a given size and centre whose vertical corner edges, top "
              "edges and bottom edges are rounded with separate radii (circular bevel profiles)")
    size = g.input("Size", "VECTOR", (1.0, 1.0, 1.0), desc="Full box size along X, Y, Z (m)", subtype="XYZ")
    centre = g.input("Centre", "VECTOR", (0.0, 0.0, 0.0), desc="Box centre in object space (m)", subtype="TRANSLATION")
    rc = g.input("Corner Radius", "FLOAT", 0.0, min=0.0, max=1.0, subtype="DISTANCE",
                 desc="Radius of the four edges parallel to Z (plan-view corner rounding)")
    rt = g.input("Top Edge Radius", "FLOAT", 0.0, min=0.0, max=1.0, subtype="DISTANCE",
                 desc="Radius of the edges around the top face")
    rb = g.input("Bottom Edge Radius", "FLOAT", 0.0, min=0.0, max=1.0, subtype="DISTANCE",
                 desc="Radius of the edges around the bottom face")
    seg = g.input("Segments", "INT", 6, min=1, max=32, desc="Bevel segments per rounded edge")

    def bevel(mesh, sel, radius):
        b = g.node("GeometryNodeMeshBevel", Mesh=mesh, Selection=sel, Segments=seg,
                   **{"Start Left Offset": radius, "Start Right Offset": radius,
                      "End Left Offset": radius, "End Right Offset": radius})
        on = _cmp(g, "GREATER_THAN", radius, 1e-7)
        return g.node("GeometryNodeSwitch", props={"input_type": "GEOMETRY"}, Switch=on,
                      **{"False": mesh, "True": b["Mesh"]})

    with g.frame("Box at the origin; half height for edge selection"):
        cube = g.node("GeometryNodeMeshCube", Size=size)
        hz = g.math("MULTIPLY", g.separate(size)["Z"], 0.5)
        ev = g.node("GeometryNodeInputMeshEdgeVertices")
        z1, z2 = g.separate(ev["Position 1"])["Z"], g.separate(ev["Position 2"])["Z"]
    with g.frame("Round the vertical edges (edges whose ends differ in Z)"):
        dz = g.math("ABSOLUTE", g.math("SUBTRACT", z1, z2))
        m1 = bevel(cube["Mesh"], _cmp(g, "GREATER_THAN", dz, 1e-6), rc)
    with g.frame("Round the top edges (both ends on the top plane)"):
        m2 = bevel(m1, _cmp(g, "GREATER_THAN", g.math("MINIMUM", z1, z2), g.math("SUBTRACT", hz, 1e-6)), rt)
    with g.frame("Round the bottom edges (both ends on the bottom plane)"):
        m3 = bevel(m2, _cmp(g, "LESS_THAN", g.math("MAXIMUM", z1, z2),
                            g.math("ADD", g.math("MULTIPLY", hz, -1.0), 1e-6)), rb)
    with g.frame("Smooth shading with sharp edges above 30 degrees, then move to Centre"):
        sm = g.node("GeometryNodeSetShadeSmooth", props={"domain": "FACE"}, Geometry=m3, **{"Shade Smooth": True})
        ang = g.node("GeometryNodeInputMeshEdgeAngle")
        sm2 = g.node("GeometryNodeSetShadeSmooth", props={"domain": "EDGE"}, Geometry=sm["Geometry"],
                     Selection=_cmp(g, "GREATER_THAN", ang["Unsigned Angle"], math.radians(30.0)),
                     **{"Shade Smooth": False})
        out = g.node("GeometryNodeTransform", Geometry=sm2["Geometry"], Translation=centre)
    g.output("Mesh", "GEOMETRY", out["Geometry"], desc="The rounded box")
    return g.finish()


def _tag(g, geo, part, name="vsp_part", domain="FACE"):
    st = g.node("GeometryNodeStoreNamedAttribute", props={"data_type": "INT", "domain": domain},
                Geometry=geo, Name=name, Value=part)
    return st["Geometry"]


@once
def frame():
    rb = rounded_block()
    g = Graph("VSP Frame", description="Rectangular spring frame: two walnut posts standing on a black-steel base "
              "plate, joined near the top by a black-steel crossbar. A (origin) is the crossbar top under the "
              "spring's top loop; everything is laid out around it, so the spring never moves when the frame "
              "controls change.", modifier=True)
    span = g.input("Span", "FLOAT", 0.25, min=0.02, max=5.0, subtype="DISTANCE", panel="Primary",
                   desc="Clear span between the posts' inner faces (the geometry unit S)")
    pw = g.input("Post Width", "FLOAT", 0.044, min=0.005, max=1.0, subtype="DISTANCE", panel="Primary",
                 desc="Post section along X")
    pd = g.input("Post Depth", "FLOAT", 0.04, min=0.005, max=1.0, subtype="DISTANCE", panel="Primary",
                 desc="Post section along Y")
    ptz = g.input("Post Top Z", "FLOAT", 0.003, min=-1.0, max=2.0, subtype="DISTANCE", panel="Primary",
                  desc="Post top relative to A (the crossbar top)")
    btz = g.input("Base Top Z", "FLOAT", -0.357, min=-5.0, max=0.0, subtype="DISTANCE", panel="Primary",
                  desc="Base-plate top relative to A (negative)")
    bh = g.input("Bar Height", "FLOAT", 0.0206, min=0.002, max=0.5, subtype="DISTANCE", panel="Primary",
                 desc="Crossbar section height; its top is at z = 0")
    bd = g.input("Bar Depth", "FLOAT", 0.0103, min=0.002, max=0.5, subtype="DISTANCE", panel="Primary",
                 desc="Crossbar section depth (Y), centred on y = 0")
    xc = g.input("Frame Centre X", "FLOAT", 0.0, min=-1.0, max=1.0, subtype="DISTANCE", panel="Placement",
                 desc="X of the frame's centre relative to A")
    plw = g.input("Plate Width", "FLOAT", 0.346, min=0.02, max=5.0, subtype="DISTANCE", panel="Placement",
                  desc="Base plate size along X (centred on the frame)")
    pldp = g.input("Plate Depth", "FLOAT", 0.104, min=0.02, max=5.0, subtype="DISTANCE", panel="Placement",
                   desc="Base plate size along Y")
    plcy = g.input("Plate Centre Y", "FLOAT", -0.008, min=-2.0, max=2.0, subtype="DISTANCE", panel="Placement",
                   desc="Base plate centre in Y (negative = towards the camera)")
    plt = g.input("Plate Thickness", "FLOAT", 0.0065, min=0.001, max=0.5, subtype="DISTANCE", panel="Placement",
                  desc="Base plate thickness (below Base Top Z)")
    plcr = g.input("Plate Corner Radius", "FLOAT", 0.008, min=0.0, max=0.5, subtype="DISTANCE", panel="Detail",
                   desc="Plan-view corner rounding of the plate")
    pler = g.input("Plate Edge Radius", "FLOAT", 0.0015, min=0.0, max=0.1, subtype="DISTANCE", panel="Detail",
                   desc="Rounding of the plate's top edges")
    wer = g.input("Wood Edge Radius", "FLOAT", 0.0011, min=0.0, max=0.05, subtype="DISTANCE", panel="Detail",
                  desc="Edge softening of the posts")
    sr = g.input("Screw Radius", "FLOAT", 0.005, min=0.0, max=0.1, subtype="DISTANCE", panel="Detail",
                 desc="Screw head radius (0 hides the screws)")
    sh = g.input("Screw Height", "FLOAT", 0.0023, min=0.0, max=0.05, subtype="DISTANCE", panel="Detail",
                 desc="Screw head dome height above the plate")
    six = g.input("Screw Inset X", "FLOAT", 0.012, min=0.0, max=1.0, subtype="DISTANCE", panel="Detail",
                  desc="Screw centre to the plate's side edge")
    siy = g.input("Screw Inset Y", "FLOAT", 0.019, min=0.0, max=1.0, subtype="DISTANCE", panel="Detail",
                  desc="Screw centre to the plate's front/back edge")
    seg = g.input("Bevel Segments", "INT", 5, min=1, max=24, panel="Detail", desc="Segments per rounded edge")
    g._panels["Detail"].default_closed = True

    with g.frame("Layout: inner faces at Frame Centre X +- Span/2"):
        half = g.math("MULTIPLY", span, 0.5)
        pcx_off = g.math("ADD", half, g.math("MULTIPLY", pw, 0.5), label="post centre offset")
        sink = 0.0005
        ph = g.math("ADD", g.math("SUBTRACT", ptz, btz), sink, label="post height (+ sink)")
        pcz = g.math("SUBTRACT", g.math("MULTIPLY", g.math("ADD", ptz, btz), 0.5), sink / 2)
    with g.frame("Posts: two rounded walnut columns"):
        posts = g.node("GeometryNodeJoinGeometry")
        for sgn in (-1.0, 1.0):
            px = g.math("ADD", xc, g.math("MULTIPLY", pcx_off, sgn))
            blk = g.group(rb, "Post", Size=_xyz(g, pw, pd, ph), Centre=_xyz(g, px, 0.0, pcz),
                          **{"Corner Radius": wer, "Top Edge Radius": wer, "Bottom Edge Radius": 0.0, "Segments": seg})
            g.connect(blk["Mesh"], posts.i("Geometry"))
        posts_t = _tag(g, posts["Geometry"], PART_IDS["posts"])
    with g.frame("Crossbar: flat steel bar between the inner faces (ends hidden 1 mm inside the posts)"):
        bar = g.node("GeometryNodeMeshCube", Size=_xyz(g, g.math("ADD", span, 0.002), bd, bh))
        bar_x = g.node("GeometryNodeTransform", Geometry=bar["Mesh"],
                       Translation=_xyz(g, xc, 0.0, g.math("MULTIPLY", bh, -0.5)))
        bar_t = _tag(g, bar_x["Geometry"], PART_IDS["bar"])
    with g.frame("Base plate: rounded slab under Base Top Z"):
        plate = g.group(rb, "Plate", Size=_xyz(g, plw, pldp, plt),
                        Centre=_xyz(g, xc, plcy, g.math("SUBTRACT", btz, g.math("MULTIPLY", plt, 0.5))),
                        **{"Corner Radius": plcr, "Top Edge Radius": pler, "Bottom Edge Radius": g.math("MULTIPLY", pler, 0.5),
                           "Segments": seg})
        plate_t = _tag(g, plate["Mesh"], PART_IDS["plate"])
    with g.frame("Screws: four flattened domes on the plate corners"):
        dome = g.node("GeometryNodeMeshUVSphere", Segments=32, Rings=12, Radius=1.0)
        dsm = g.node("GeometryNodeSetShadeSmooth", props={"domain": "FACE"}, Geometry=dome["Mesh"], **{"Shade Smooth": True})
        screws = g.node("GeometryNodeJoinGeometry")
        sxo = g.math("SUBTRACT", g.math("MULTIPLY", plw, 0.5), six)
        syo = g.math("SUBTRACT", g.math("MULTIPLY", pldp, 0.5), siy)
        for a_ in (-1.0, 1.0):
            for b_ in (-1.0, 1.0):
                t = g.node("GeometryNodeTransform", Geometry=dsm["Geometry"],
                           Translation=_xyz(g, g.math("ADD", xc, g.math("MULTIPLY", sxo, a_)),
                                            g.math("ADD", plcy, g.math("MULTIPLY", syo, b_)), btz),
                           Scale=_xyz(g, sr, sr, sh))
                g.connect(t["Geometry"], screws.i("Geometry"))
        show = _cmp(g, "GREATER_THAN", g.math("MINIMUM", sr, sh), 1e-7)
        screws_s = g.node("GeometryNodeSwitch", props={"input_type": "GEOMETRY"}, Switch=show,
                          **{"True": screws["Geometry"]})
        screws_t = _tag(g, screws_s["Output"], PART_IDS["screws"])
    with g.frame("Join all parts"):
        join = g.node("GeometryNodeJoinGeometry")
        for geo in (posts_t, bar_t, plate_t, screws_t):
            g.connect(geo, join.i("Geometry"))
    g.output("Geometry", "GEOMETRY", join["Geometry"], desc="Frame mesh with face attribute vsp_part")
    return g.finish()


@once
def frame_part():
    g = Graph("VSP Frame Part", description="Takes the evaluated mesh of a VSP Frame object and keeps one part "
              "(vsp_part: 0 posts, 1 crossbar, 2 base plate, 3 screws), so each part is its own object with its "
              "own material while all controls stay on the one VSP Frame modifier", modifier=True)
    obj = g.input("Frame", "OBJECT", desc="Object carrying the VSP Frame modifier")
    part = g.input("Part", "INT", 0, min=0, max=3, desc="0 posts, 1 crossbar, 2 base plate, 3 screws")
    mat = g.input("Material", "MATERIAL", desc="Material of this part")
    with g.frame("Read the frame and keep the faces of one part"):
        info = g.node("GeometryNodeObjectInfo", props={"transform_space": "ORIGINAL"}, Object=obj)
        attr = g.node("GeometryNodeInputNamedAttribute", props={"data_type": "INT"}, Name="vsp_part")
        sep = g.node("GeometryNodeSeparateGeometry", props={"domain": "FACE"}, Geometry=info["Geometry"],
                     Selection=_cmp(g, "EQUAL", attr["Attribute"], part, "INT"))
        sm = g.node("GeometryNodeSetMaterial", Geometry=sep["Selection"], Material=mat)
    g.output("Geometry", "GEOMETRY", sm["Geometry"], desc="Faces of the chosen part")
    return g.finish()


def _fillet_rect(g, width, height, radius, count):
    """Rounded rectangle (poly fillet) in the XY plane, centred at the origin."""
    q = g.node("GeometryNodeCurvePrimitiveQuadrilateral", props={"mode": "RECTANGLE"}, Width=width, Height=height)
    f = g.node("GeometryNodeFilletCurve", Curve=q["Curve"], Radius=radius)
    if hasattr(f.b, "mode"):
        f.b.mode = "POLY"
        g.connect(count, f.i("Count"))
    else:                                                     # Blender 5.x: mode is a menu socket
        try:
            f.i("Mode").default_value = "Poly"
        except Exception:
            pass
        g.connect(count, f.i("Count"))
    try:
        f.i("Limit Radius").default_value = True
    except Exception:
        pass
    return f["Curve"]


@once
def spring():
    g = Graph("VSP Spring", description="Helical extension spring. STATIC part: top loop around the crossbar (inner top on "
              "the bar at A = origin), S-bend and the hook turns ending at the hinge P (= P_eff, the top coil). MOVING part: "
              "open turns, tight turns, bottom transition and the lower loop whose inner bottom is at H (origin of the Lower "
              "End object). The moving part is built hanging straight down from P and rotated about P so that P->H is its "
              "axis; the rotation ramps in smoothly over the Bend Length below P (the spring bends at its top coil while the "
              "hook stays fixed). Fixed coil counts; every moving turn takes the same extension; tight turns stop at solid "
              "pitch.", modifier=True)
    low = g.input("Lower End", "OBJECT", desc="Object whose origin is H (the lower loop's inner bottom)")
    hinge = g.input("Hinge", "VECTOR", (0.0, 0.0, -0.06), subtype="TRANSLATION", desc="P_eff: the spring's swing hinge (top coil)")
    rc = g.input("Coil Radius", "FLOAT", 0.0135, min=0.001, max=0.5, subtype="DISTANCE", panel="Coils", desc="Mean coil radius")
    rw = g.input("Wire Radius", "FLOAT", 0.0012, min=0.0001, max=0.05, subtype="DISTANCE", panel="Coils", desc="Spring wire radius")
    nh = g.input("Hook Turns", "INT", 2, min=0, max=20, panel="Coils", desc="Static close turns above the hinge")
    ph = g.input("Hook Turn Pitch", "FLOAT", 0.003, min=0.0, max=0.5, subtype="DISTANCE", panel="Coils", desc="Pitch of the static hook turns")
    no = g.input("Open Coils", "INT", 11, min=1, max=200, panel="Coils", desc="Open (active) turns below the hinge")
    nt = g.input("Tight Coils", "INT", 10, min=0, max=200, panel="Coils", desc="Tight turns at the bottom")
    lo = g.input("Open Length Ref", "FLOAT", 0.14, min=0.0, max=5.0, subtype="DISTANCE", panel="Coils",
                 desc="Open section length (from the hinge) in the reference configuration")
    lt = g.input("Tight Length Ref", "FLOAT", 0.06, min=0.0, max=5.0, subtype="DISTANCE", panel="Coils",
                 desc="Tight section length in the reference configuration")
    bend = g.input("Bend Length", "FLOAT", 0.013, min=0.0, max=1.0, subtype="DISTANCE", panel="Coils",
                   desc="Length below the hinge over which the rotation ramps in (smoothstep)")
    res = g.input("Resolution", "INT", 32, min=6, max=256, panel="Coils", desc="Curve points per turn")
    prof = g.input("Profile Resolution", "INT", 12, min=4, max=64, panel="Coils", desc="Vertices around the wire")
    rh = g.input("Hook Wire Radius", "FLOAT", 0.0015, min=0.0001, max=0.05, subtype="DISTANCE", panel="Ends", desc="End-loop wire radius")
    botlen = g.input("Bottom Hardware Length", "FLOAT", 0.03, min=0.0, max=2.0, subtype="DISTANCE", panel="Ends",
                     desc="Distance from the last coil to H along the axis")
    tlh = g.input("Top Loop Inner Height", "FLOAT", 0.021, min=0.001, max=0.5, subtype="DISTANCE", panel="Ends", desc="Top loop inner height")
    tlw = g.input("Top Loop Inner Width", "FLOAT", 0.013, min=0.001, max=0.5, subtype="DISTANCE", panel="Ends", desc="Top loop inner width (Y)")
    tlr = g.input("Top Loop Corner Radius", "FLOAT", 0.0015, min=0.0, max=0.5, subtype="DISTANCE", panel="Ends", desc="Top loop inner corner radius")
    llw = g.input("Lower Loop Inner Width", "FLOAT", 0.011, min=0.001, max=0.5, subtype="DISTANCE", panel="Ends", desc="Lower loop inner width (Y)")
    llr = g.input("Lower Loop Corner Radius", "FLOAT", 0.003, min=0.0, max=0.5, subtype="DISTANCE", panel="Ends", desc="Lower loop inner corner radius")
    rl = g.input("Lower Loop Wire Radius", "FLOAT", 0.0012, min=0.0001, max=0.05, subtype="DISTANCE", panel="Ends",
                 desc="Wire radius of the bottom transition and lower loop (the spring wire; VR-18)")
    lls = g.input("Lower Loop Gap", "FLOAT", 0.005, min=0.0, max=0.5, subtype="DISTANCE", panel="Ends", desc="Gap between the last coil and the lower loop top")
    mat = g.input("Material", "MATERIAL", desc="Spring steel material")

    def store(geo, name, val, dtype="INT"):
        return g.node("GeometryNodeStoreNamedAttribute", props={"data_type": dtype, "domain": "POINT"},
                      Geometry=geo, Name=name, Value=val)["Geometry"]

    sp = g.separate(hinge)
    px, pz = sp["X"], sp["Z"]
    with g.frame("Axis: P = hinge, H from the Lower End object; L = |H - P|; rotation taking -Z onto (H - P)"):
        info = g.node("GeometryNodeObjectInfo", props={"transform_space": "RELATIVE"}, Object=low)
        dvec = g.vmath("SUBTRACT", info["Location"], hinge)["Vector"]
        L = g.vmath("LENGTH", dvec)["Value"]
        dn = g.vmath("NORMALIZE", dvec)["Vector"]
        cr = g.vmath("CROSS_PRODUCT", (0.0, 0.0, -1.0), dn)["Vector"]
        crl = g.vmath("LENGTH", cr)["Value"]
        axis = g.node("GeometryNodeSwitch", props={"input_type": "VECTOR"}, Switch=_cmp(g, "GREATER_THAN", crl, 1e-9),
                      **{"False": (1.0, 0.0, 0.0), "True": g.vmath("NORMALIZE", cr)["Vector"]})["Output"]
        # atan2(|a x b|, a . b): well conditioned at small angles (acos loses ~1e-5 m in float32 near vertical)
        ang = g.math("ARCTAN2", crl, g.vmath("DOT_PRODUCT", (0.0, 0.0, -1.0), dn)["Value"], label="swing angle")
    with g.frame("Pitches of the moving turns"):
        ntot = g.math("ADD", no, nt)
        lb = g.math("SUBTRACT", L, botlen, label="coil body length")
        dlt = g.math("DIVIDE", g.math("SUBTRACT", lb, g.math("ADD", lo, lt)), ntot, label="extension per turn")
        pt = g.math("MAXIMUM", g.math("ADD", g.math("DIVIDE", lt, g.math("MAXIMUM", nt, 1)), dlt), g.math("MULTIPLY", rw, 2.0), label="tight pitch")
        po = g.math("DIVIDE", g.math("SUBTRACT", lb, g.math("MULTIPLY", nt, pt)), no, label="open pitch")
        ho, ht = g.math("MULTIPLY", no, po), g.math("MULTIPLY", nt, pt)
        hh = g.math("MULTIPLY", nh, ph)
        z1 = g.math("SUBTRACT", pz, ho, label="open/tight boundary z")
        zce = g.math("SUBTRACT", z1, ht, label="last coil z")
        zH = g.math("SUBTRACT", pz, L, label="H z (rest frame)")
    flip = (math.pi, 0.0, 0.0)

    def spiral(turns, height, ztop, part):
        s = g.node("GeometryNodeCurveSpiral", Resolution=res, Rotations=turns, **{"Start Radius": rc, "End Radius": rc, "Height": height})
        x = g.node("GeometryNodeTransform", Geometry=s["Curve"], Rotation=flip, Translation=_xyz(g, px, 0.0, ztop))
        return store(x["Geometry"], "vsp_part", part)
    with g.frame("Coils: static hook turns above P, then open and tight turns below P (rest frame, hanging straight)"):
        c_h = spiral(nh, hh, g.math("ADD", pz, hh), 6)
        c_o = spiral(no, ho, pz, 2)
        c_t = spiral(nt, ht, z1, 3)
    with g.frame("Top loop on the bar at A (static)"):
        tw = g.math("ADD", tlw, g.math("MULTIPLY", rh, 2.0))
        th = g.math("ADD", tlh, g.math("MULTIPLY", rh, 2.0))
        rect = _fillet_rect(g, tw, th, g.math("ADD", tlr, rh), 8)
        tl = g.node("GeometryNodeTransform", Geometry=rect, Rotation=(math.pi / 2, 0.0, math.pi / 2),
                    Translation=_xyz(g, 0.0, 0.0, g.math("MULTIPLY", tlh, -0.5)))
        tl_t = store(tl["Geometry"], "vsp_part", 0)
        tl_bot = g.math("SUBTRACT", g.math("MULTIPLY", tlh, -1.0), rh)
    with g.frame("Top transition (static): shank down from the loop, S-bend into the first hook turn"):
        zh0 = g.math("ADD", pz, hh)
        gap_t = g.math("SUBTRACT", tl_bot, zh0)
        b1 = g.node("GeometryNodeCurvePrimitiveBezierSegment", props={"mode": "POSITION"}, Resolution=24,
                    Start=_xyz(g, 0.0, 0.0, tl_bot),
                    **{"Start Handle": _xyz(g, 0.0, 0.0, g.math("SUBTRACT", tl_bot, g.math("MULTIPLY", gap_t, 0.6))),
                       "End Handle": _xyz(g, g.math("ADD", px, g.math("MULTIPLY", rc, 1.6)), g.math("MULTIPLY", rc, 0.9),
                                          g.math("ADD", zh0, g.math("MULTIPLY", gap_t, 0.35))),
                       "End": _xyz(g, g.math("ADD", px, rc), 0.0, zh0)})
        b1_t = store(b1["Curve"], "vsp_part", 1)
    with g.frame("Bottom transition and lower loop (moving; inner bottom at H)"):
        ll_top = g.math("SUBTRACT", zce, lls)
        tan_t = g.math("DIVIDE", pt, 2 * math.pi)
        b2 = g.node("GeometryNodeCurvePrimitiveBezierSegment", props={"mode": "POSITION"}, Resolution=16,
                    Start=_xyz(g, g.math("ADD", px, rc), 0.0, zce),
                    **{"Start Handle": _xyz(g, g.math("ADD", px, rc), g.math("MULTIPLY", rc, -1.0), g.math("SUBTRACT", zce, tan_t)),
                       "End Handle": _xyz(g, px, 0.0, g.math("ADD", ll_top, g.math("MULTIPLY", rc, 0.8))),
                       "End": _xyz(g, px, 0.0, ll_top)})
        b2_t = store(b2["Curve"], "vsp_part", 4)
        lw = g.math("ADD", llw, g.math("MULTIPLY", rl, 2.0))
        lbot = g.math("SUBTRACT", zH, rl)
        rect2 = _fillet_rect(g, lw, g.math("SUBTRACT", ll_top, lbot), g.math("ADD", llr, rl), 8)
        ll = g.node("GeometryNodeTransform", Geometry=rect2, Rotation=(math.pi / 2, 0.0, math.pi / 2),
                    Translation=_xyz(g, px, 0.0, g.math("MULTIPLY", g.math("ADD", ll_top, lbot), 0.5)))
        ll_t = store(ll["Geometry"], "vsp_part", 5)

    def bend_about_hinge(geo):
        """rest position -> P + R(axis, w * angle) (rest - P), w = smoothstep((pz - z) / Bend Length); records vsp_cl_rest/vsp_w/vsp_cl."""
        pos = g.node("GeometryNodeInputPosition")["Position"]
        geo = store(geo, "vsp_cl_rest", pos, "FLOAT_VECTOR")
        zz = g.separate(pos)["Z"]
        t = g.math("DIVIDE", g.math("SUBTRACT", pz, zz), g.math("MAXIMUM", bend, 1e-9))
        w = g.map_range(t, 0.0, 1.0, 0.0, 1.0, clamp=True, interp="SMOOTHSTEP")["Result"]
        rot = g.node("FunctionNodeAxisAngleToRotation", Axis=axis, Angle=g.math("MULTIPLY", ang, w))["Rotation"]
        rv = g.node("FunctionNodeRotateVector", Vector=g.vmath("SUBTRACT", pos, hinge)["Vector"], Rotation=rot)["Vector"]
        newp = g.vmath("ADD", rv, hinge)["Vector"]
        geo = store(geo, "vsp_w", w, "FLOAT")
        geo = g.node("GeometryNodeSetPosition", Geometry=geo, Position=newp)["Geometry"]
        return store(geo, "vsp_cl", g.node("GeometryNodeInputPosition")["Position"], "FLOAT_VECTOR")
    with g.frame("Bend/rotate about P (static parts have w = 0), sweep the wire"):
        coils = g.node("GeometryNodeJoinGeometry")
        for geo in (c_t, c_o, c_h):
            g.connect(geo, coils.i("Geometry"))
        hooks = g.node("GeometryNodeJoinGeometry")
        for geo in (b1_t, tl_t):
            g.connect(geo, hooks.i("Geometry"))
        lower = g.node("GeometryNodeJoinGeometry")
        for geo in (ll_t, b2_t):
            g.connect(geo, lower.i("Geometry"))
        outs = []
        for geo, radius in ((coils["Geometry"], rw), (hooks["Geometry"], rh), (lower["Geometry"], rl)):
            cl = bend_about_hinge(geo)
            circ = g.node("GeometryNodeCurvePrimitiveCircle", props={"mode": "RADIUS"}, Resolution=prof, Radius=radius)
            outs.append(g.node("GeometryNodeCurveToMesh", Curve=cl, **{"Profile Curve": circ["Curve"]})["Mesh"])
        join = g.node("GeometryNodeJoinGeometry")
        for o in outs:
            g.connect(o, join.i("Geometry"))
        sm = g.node("GeometryNodeSetShadeSmooth", props={"domain": "FACE"}, Geometry=join["Geometry"], **{"Shade Smooth": True})
        sm2 = g.node("GeometryNodeSetMaterial", Geometry=sm["Geometry"], Material=mat)
    g.output("Geometry", "GEOMETRY", sm2["Geometry"], desc="Spring mesh (vsp_part, vsp_cl, vsp_cl_rest, vsp_w point attributes)")
    return g.finish()
