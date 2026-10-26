"""Geometry Nodes groups for clip 03 (torsion disc), built with the vendored nodekit (scene/vendor/nodekit).

Groups
  TOR Rounded Block  box with separately rounded vertical corners, top edges and bottom edges (clip-01/02 group)
  TOR Frame          (modifier) the static stand around A (origin = wire exit at the clamp collet bottom): two walnut
                     posts on two walnut feet (T feet along Y), a walnut crossbar between the posts' inner faces (top
                     flush with the post tops), and the black-steel clamp (U bracket over the bar: front, top and back
                     plates), two cap screws on top, a button screw on the front plate and the collet the wire leaves
                     from. Face attribute `tor_part` (0 posts, 1 crossbar, 2 feet, 3 clamp, 4 screws, 5 collet, 6 hex sockets).
                     Nothing reads time.
  TOR Frame Part     (modifier) one part of a TOR Frame object, picked by `tor_part`
  TOR Wire           (modifier) the torsion wire: a straight tube between the origins of the Top and Bottom objects
                     (A = TOR_Anchor, B = TOR_Disc origin = top-face centre), evaluated in the wire object's space
                     (the wire object sits unrotated at the world origin). Point attribute tor_end (0 top ring, 1 bottom
                     ring, 2 interior) for check_rig. Twist is not drawn on the wire (it is shown by the disc).
All lengths are metres in object space.
"""
import math
import pathlib
import sys

VENDOR = str(pathlib.Path(__file__).resolve().parent / "vendor")
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

from nodekit.graph import Graph, once  # noqa: E402

PART_IDS = {"posts": 0, "bar": 1, "feet": 2, "clamp": 3, "screws": 4, "collet": 5, "socket": 6}


def _xyz(g, x, y, z, label=None):
    return g.combine(x, y, z, label=label)


def _cmp(g, op, a, b, dtype="FLOAT"):
    return g.node("FunctionNodeCompare", props={"data_type": dtype, "operation": op}, A=a, B=b)["Result"]


@once
def rounded_block():
    g = Graph("TOR Rounded Block", description="Box of a given size and centre whose vertical corner edges, top "
              "edges and bottom edges are rounded with separate radii (circular bevel profiles)")
    size = g.input("Size", "VECTOR", (1.0, 1.0, 1.0), desc="Full box size along X, Y, Z (m)", subtype="XYZ")
    centre = g.input("Centre", "VECTOR", (0.0, 0.0, 0.0), desc="Box centre in object space (m)", subtype="TRANSLATION")
    rc = g.input("Corner Radius", "FLOAT", 0.0, min=0.0, max=1.0, subtype="DISTANCE",
                 desc="Radius of the four edges parallel to Z (plan-view corner rounding)")
    rt = g.input("Top Edge Radius", "FLOAT", 0.0, min=0.0, max=1.0, subtype="DISTANCE",
                 desc="Radius of the edges around the top face")
    rb = g.input("Bottom Edge Radius", "FLOAT", 0.0, min=0.0, max=1.0, subtype="DISTANCE",
                 desc="Radius of the edges around the bottom face")
    seg = g.input("Segments", "INT", 4, min=1, max=32, desc="Bevel segments per rounded edge")

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


def _tag(g, geo, part, name="tor_part", domain="FACE"):
    st = g.node("GeometryNodeStoreNamedAttribute", props={"data_type": "INT", "domain": domain},
                Geometry=geo, Name=name, Value=part)
    return st["Geometry"]


def _cyl(g, radius, depth, centre, rot=(0.0, 0.0, 0.0), verts=48, fill="NGON", cap_round=None):
    """Cylinder along local Z (rotated by rot) centred at centre; smooth sides."""
    c = g.node("GeometryNodeMeshCylinder", props={"fill_type": fill}, Vertices=verts, Radius=radius, Depth=depth)
    sm = g.node("GeometryNodeSetShadeSmooth", props={"domain": "FACE"}, Geometry=c["Mesh"], **{"Shade Smooth": True})
    ang = g.node("GeometryNodeInputMeshEdgeAngle")
    sm2 = g.node("GeometryNodeSetShadeSmooth", props={"domain": "EDGE"}, Geometry=sm["Geometry"],
                 Selection=_cmp(g, "GREATER_THAN", ang["Unsigned Angle"], math.radians(30.0)), **{"Shade Smooth": False})
    return g.node("GeometryNodeTransform", Geometry=sm2["Geometry"], Rotation=rot, Translation=centre)["Geometry"]


@once
def frame():
    rb = rounded_block()
    g = Graph("TOR Frame", description="Torsion-disc stand around A (origin = wire exit at the clamp collet bottom): "
              "walnut posts on T feet, walnut crossbar between the posts (tops flush), black-steel U clamp over the "
              "bar centre with two top cap screws, a front button screw and the collet the wire leaves from. The wire "
              "and disc never move when these controls change (they hang from A).", modifier=True)
    P = "Primary"
    span = g.input("Span", "FLOAT", 0.17, min=0.02, max=5.0, subtype="DISTANCE", panel=P, desc="Clear span between the posts' inner faces")
    xc = g.input("Frame Centre X", "FLOAT", 0.0, min=-1.0, max=1.0, subtype="DISTANCE", panel=P, desc="Frame centre X relative to A")
    pw = g.input("Post Width", "FLOAT", 0.02, min=0.002, max=1.0, subtype="DISTANCE", panel=P, desc="Post section along X")
    pd = g.input("Post Depth", "FLOAT", 0.015, min=0.002, max=1.0, subtype="DISTANCE", panel=P, desc="Post section along Y")
    ptz = g.input("Post Top Z", "FLOAT", 0.026, min=-1.0, max=2.0, subtype="DISTANCE", panel=P, desc="Post (and crossbar) top relative to A")
    bh = g.input("Bar Height", "FLOAT", 0.021, min=0.002, max=0.5, subtype="DISTANCE", panel=P, desc="Crossbar section height")
    bd = g.input("Bar Depth", "FLOAT", 0.015, min=0.002, max=0.5, subtype="DISTANCE", panel=P, desc="Crossbar section depth (Y), centred on y = 0")
    F = "Feet"
    ftz = g.input("Foot Top Z", "FLOAT", -0.15, min=-5.0, max=0.0, subtype="DISTANCE", panel=F, desc="Foot top (post bottom) relative to A")
    fh = g.input("Foot Height", "FLOAT", 0.015, min=0.001, max=0.5, subtype="DISTANCE", panel=F, desc="Foot height (table at Foot Top Z - Foot Height)")
    fw = g.input("Foot Width", "FLOAT", 0.022, min=0.002, max=0.5, subtype="DISTANCE", panel=F, desc="Foot width along X, centred on the post")
    ffy = g.input("Foot Front Y", "FLOAT", -0.078, min=-2.0, max=0.0, subtype="DISTANCE", panel=F, desc="Foot front end Y (towards the camera)")
    fby = g.input("Foot Back Y", "FLOAT", 0.065, min=0.0, max=2.0, subtype="DISTANCE", panel=F, desc="Foot back end Y")
    C = "Clamp"
    cw = g.input("Clamp Width", "FLOAT", 0.0226, min=0.002, max=0.5, subtype="DISTANCE", panel=C, desc="Clamp size along X")
    cd = g.input("Clamp Depth", "FLOAT", 0.030, min=0.002, max=0.5, subtype="DISTANCE", panel=C, desc="Clamp outer size along Y")
    ctz = g.input("Clamp Top Z", "FLOAT", 0.029, min=-0.5, max=0.5, subtype="DISTANCE", panel=C, desc="Clamp top (outer) relative to A")
    cbz = g.input("Clamp Bottom Z", "FLOAT", 0.004, min=0.0, max=0.5, subtype="DISTANCE", panel=C, desc="Clamp plates' bottom edge (= collet top)")
    sh = g.input("Clamp Sheet", "FLOAT", 0.0014, min=0.0002, max=0.05, subtype="DISTANCE", panel=C, desc="Sheet thickness of the bracket")
    cbr = g.input("Clamp Bend Radius", "FLOAT", 0.002, min=0.0, max=0.05, subtype="DISTANCE", panel=C, desc="Outer radius of the top bends")
    tsr = g.input("Top Screw Radius", "FLOAT", 0.0036, min=0.0, max=0.05, subtype="DISTANCE", panel=C, desc="Top cap-screw head radius")
    tsh = g.input("Top Screw Height", "FLOAT", 0.003, min=0.0, max=0.05, subtype="DISTANCE", panel=C, desc="Top cap-screw head height")
    tsy = g.input("Top Screw Y", "FLOAT", 0.007, min=0.0, max=0.5, subtype="DISTANCE", panel=C, desc="|Y| of the two top screws (one in front, one behind)")
    fsr = g.input("Front Screw Radius", "FLOAT", 0.0029, min=0.0, max=0.05, subtype="DISTANCE", panel=C, desc="Front button-screw head radius")
    fsz = g.input("Front Screw Z", "FLOAT", 0.0042, min=0.0, max=0.5, subtype="DISTANCE", panel=C, desc="Front screw centre above the clamp bottom")
    fsh = g.input("Front Screw Height", "FLOAT", 0.0015, min=0.0, max=0.05, subtype="DISTANCE", panel=C, desc="Front screw head height")
    wsr = g.input("Washer Radius", "FLOAT", 0.0036, min=0.0, max=0.05, subtype="DISTANCE", panel=C, desc="Front screw washer radius")
    wsh = g.input("Washer Height", "FLOAT", 0.0006, min=0.0, max=0.05, subtype="DISTANCE", panel=C, desc="Front screw washer thickness")
    skr = g.input("Socket Radius", "FLOAT", 0.0009, min=0.0, max=0.05, subtype="DISTANCE", panel=C, desc="Hex socket circumradius")
    clr = g.input("Collet Radius", "FLOAT", 0.0037, min=0.0, max=0.05, subtype="DISTANCE", panel=C, desc="Collet radius (wire exit at its bottom = A)")
    D = "Detail"
    wer = g.input("Wood Edge Radius", "FLOAT", 0.0006, min=0.0, max=0.05, subtype="DISTANCE", panel=D, desc="Edge softening of the walnut parts")
    seg = g.input("Bevel Segments", "INT", 3, min=1, max=24, panel=D, desc="Segments per rounded edge")
    g._panels[D].default_closed = True

    with g.frame("Layout: inner faces at Frame Centre X +- Span/2"):
        half = g.math("MULTIPLY", span, 0.5)
        pcx_off = g.math("ADD", half, g.math("MULTIPLY", pw, 0.5), label="post centre offset")
        sink = 0.0002
        ph = g.math("ADD", g.math("SUBTRACT", ptz, ftz), sink, label="post height (+ sink into the foot)")
        pcz = g.math("SUBTRACT", g.math("MULTIPLY", g.math("ADD", ptz, ftz), 0.5), sink / 2)
        fl = g.math("SUBTRACT", fby, ffy, label="foot length")
        fcy = g.math("MULTIPLY", g.math("ADD", fby, ffy), 0.5)
        fcz = g.math("SUBTRACT", ftz, g.math("MULTIPLY", fh, 0.5))
    with g.frame("Posts and feet"):
        posts, feet = g.node("GeometryNodeJoinGeometry"), g.node("GeometryNodeJoinGeometry")
        for sgn in (-1.0, 1.0):
            px = g.math("ADD", xc, g.math("MULTIPLY", pcx_off, sgn))
            blk = g.group(rb, "Post", Size=_xyz(g, pw, pd, ph), Centre=_xyz(g, px, 0.0, pcz),
                          **{"Corner Radius": wer, "Top Edge Radius": wer, "Bottom Edge Radius": 0.0, "Segments": seg})
            g.connect(blk["Mesh"], posts.i("Geometry"))
            ft = g.group(rb, "Foot", Size=_xyz(g, fw, fl, fh), Centre=_xyz(g, px, fcy, fcz),
                         **{"Corner Radius": wer, "Top Edge Radius": wer, "Bottom Edge Radius": 0.0, "Segments": seg})
            g.connect(ft["Mesh"], feet.i("Geometry"))
        posts_t = _tag(g, posts["Geometry"], PART_IDS["posts"])
        feet_t = _tag(g, feet["Geometry"], PART_IDS["feet"])
    with g.frame("Crossbar between the inner faces (ends 0.2 mm inside the posts), top at Post Top Z"):
        bar = g.group(rb, "Bar", Size=_xyz(g, g.math("ADD", span, 0.0004), bd, bh),
                      Centre=_xyz(g, xc, 0.0, g.math("SUBTRACT", ptz, g.math("MULTIPLY", bh, 0.5))),
                      **{"Corner Radius": 0.0, "Top Edge Radius": wer, "Bottom Edge Radius": wer, "Segments": seg})
        bar_t = _tag(g, bar["Mesh"], PART_IDS["bar"])
    with g.frame("Clamp: U bracket (top plate with rounded bends + front and back plates)"):
        ch = g.math("SUBTRACT", ctz, cbz, label="clamp height")
        top = g.group(rb, "Clamp top", Size=_xyz(g, cw, cd, g.math("MULTIPLY", cbr, 2.0)),
                      Centre=_xyz(g, 0.0, 0.0, g.math("SUBTRACT", ctz, cbr)),
                      **{"Corner Radius": 0.0, "Top Edge Radius": cbr, "Bottom Edge Radius": 0.0, "Segments": seg})
        clamp = g.node("GeometryNodeJoinGeometry")
        g.connect(top["Mesh"], clamp.i("Geometry"))
        plate_h = g.math("SUBTRACT", ch, cbr)
        for sgn in (-1.0, 1.0):
            pyc = g.math("MULTIPLY", g.math("SUBTRACT", g.math("MULTIPLY", cd, 0.5), g.math("MULTIPLY", sh, 0.5)), sgn)
            pl = g.group(rb, "Clamp plate", Size=_xyz(g, cw, sh, plate_h),
                         Centre=_xyz(g, 0.0, pyc, g.math("ADD", cbz, g.math("MULTIPLY", plate_h, 0.5))),
                         **{"Corner Radius": g.math("MULTIPLY", sh, 0.3), "Top Edge Radius": 0.0,
                            "Bottom Edge Radius": g.math("MULTIPLY", sh, 0.3), "Segments": 2})
            g.connect(pl["Mesh"], clamp.i("Geometry"))
        clamp_t = _tag(g, clamp["Geometry"], PART_IDS["clamp"])
    with g.frame("Screws: two cap screws on top, one button screw on the front plate"):
        screws = g.node("GeometryNodeJoinGeometry")
        for sgn in (-1.0, 1.0):
            s = _cyl(g, tsr, tsh, _xyz(g, 0.0, g.math("MULTIPLY", tsy, sgn), g.math("ADD", ctz, g.math("MULTIPLY", tsh, 0.5))))
            g.connect(s, screws.i("Geometry"))
        # front screw: washer (thin disc) + button head (squashed dome) on the front plate; hex socket recessed in the head
        fy0 = g.math("MULTIPLY", cd, -0.5)                       # front plate outer face
        fz = g.math("ADD", cbz, fsz)
        ws = _cyl(g, wsr, wsh, _xyz(g, 0.0, g.math("SUBTRACT", fy0, g.math("MULTIPLY", wsh, 0.5)), fz), rot=(math.pi / 2, 0.0, 0.0), verts=48)
        g.connect(ws, screws.i("Geometry"))
        dome = g.node("GeometryNodeMeshUVSphere", Segments=48, Rings=16, Radius=1.0)
        dsm = g.node("GeometryNodeSetShadeSmooth", props={"domain": "FACE"}, Geometry=dome["Mesh"], **{"Shade Smooth": True})
        head = g.node("GeometryNodeTransform", Geometry=dsm["Geometry"], Rotation=(math.pi / 2, 0.0, 0.0),
                      Translation=_xyz(g, 0.0, g.math("SUBTRACT", fy0, wsh), fz), Scale=_xyz(g, fsr, fsr, fsh))
        g.connect(head["Geometry"], screws.i("Geometry"))
        screws_t = _tag(g, screws["Geometry"], PART_IDS["screws"])
        # hex socket: a dark hexagonal prism sunk into the dome (its front face just proud of the dome's apex region)
        sock = _cyl(g, skr, g.math("MULTIPLY", fsh, 0.6), _xyz(g, 0.0, g.math("SUBTRACT", g.math("SUBTRACT", fy0, wsh),
                                                                              g.math("MULTIPLY", fsh, 0.72)), fz),
                    rot=(math.pi / 2, 0.0, math.pi / 6), verts=6)
        sock_t = _tag(g, sock, PART_IDS["socket"])
    with g.frame("Collet: from A (z = 0) up to the clamp bottom"):
        col = _cyl(g, clr, g.math("ADD", cbz, 0.0002), _xyz(g, 0.0, 0.0, g.math("MULTIPLY", g.math("ADD", cbz, 0.0002), 0.5)))
        col_t = _tag(g, col, PART_IDS["collet"])
    with g.frame("Join all parts"):
        join = g.node("GeometryNodeJoinGeometry")
        for geo in (posts_t, bar_t, feet_t, clamp_t, screws_t, col_t, sock_t):
            g.connect(geo, join.i("Geometry"))
    g.output("Geometry", "GEOMETRY", join["Geometry"], desc="Stand mesh with face attribute tor_part")
    return g.finish()


@once
def frame_part():
    g = Graph("TOR Frame Part", description="Takes the evaluated mesh of a TOR Frame object and keeps one part "
              "(tor_part: 0 posts, 1 crossbar, 2 feet, 3 clamp, 4 screws, 5 collet, 6 hex sockets), so each part is its own object "
              "with its own material while all controls stay on the one TOR Frame modifier", modifier=True)
    obj = g.input("Frame", "OBJECT", desc="Object carrying the TOR Frame modifier")
    part = g.input("Part", "INT", 0, min=0, max=6, desc="0 posts, 1 crossbar, 2 feet, 3 clamp, 4 screws, 5 collet, 6 hex sockets")
    mat = g.input("Material", "MATERIAL", desc="Material of this part")
    with g.frame("Read the frame and keep the faces of one part"):
        info = g.node("GeometryNodeObjectInfo", props={"transform_space": "ORIGINAL"}, Object=obj)
        attr = g.node("GeometryNodeInputNamedAttribute", props={"data_type": "INT"}, Name="tor_part")
        sep = g.node("GeometryNodeSeparateGeometry", props={"domain": "FACE"}, Geometry=info["Geometry"],
                     Selection=_cmp(g, "EQUAL", attr["Attribute"], part, "INT"))
        sm = g.node("GeometryNodeSetMaterial", Geometry=sep["Selection"], Material=mat)
    g.output("Geometry", "GEOMETRY", sm["Geometry"], desc="Faces of the chosen part")
    return g.finish()


@once
def wire():
    g = Graph("TOR Wire", description="Torsion wire: a straight tube from the origin of Top (A, TOR_Anchor) to the "
              "origin of Bottom (B, the disc's top-face centre = TOR_Disc origin), in the wire object's space. Both "
              "end rings sit exactly on A and B for any disc pose; the wire has a fixed radius and no drawn twist "
              "(twist is shown by the disc).", modifier=True)
    top = g.input("Top", "OBJECT", desc="Object whose origin is A (fixed clamp exit)")
    bot = g.input("Bottom", "OBJECT", desc="Object whose origin is B (disc attachment)")
    r = g.input("Radius", "FLOAT", 0.00014, min=0.00001, max=0.01, subtype="DISTANCE", desc="Wire radius")
    n = g.input("Segments", "INT", 16, min=2, max=256, desc="Points along the wire")
    prof = g.input("Profile Resolution", "INT", 8, min=3, max=64, desc="Vertices around the wire")
    mat = g.input("Material", "MATERIAL", desc="Wire steel material")
    with g.frame("Endpoints from the two objects (relative to this object)"):
        a = g.node("GeometryNodeObjectInfo", props={"transform_space": "RELATIVE"}, Object=top)["Location"]
        b = g.node("GeometryNodeObjectInfo", props={"transform_space": "RELATIVE"}, Object=bot)["Location"]
        line = g.node("GeometryNodeCurvePrimitiveLine", props={"mode": "POINTS"}, Start=a, End=b)
        rs = g.node("GeometryNodeResampleCurve", Curve=line["Curve"], Count=n)
        idx = g.node("GeometryNodeInputIndex")["Index"]
        last = g.math("SUBTRACT", n, 1)
        endid = g.node("GeometryNodeSwitch", props={"input_type": "INT"}, Switch=_cmp(g, "EQUAL", idx, 0, "INT"),
                       **{"False": g.node("GeometryNodeSwitch", props={"input_type": "INT"},
                                          Switch=_cmp(g, "EQUAL", idx, last, "INT"), **{"False": 2, "True": 1})["Output"],
                          "True": 0})["Output"]
        st = g.node("GeometryNodeStoreNamedAttribute", props={"data_type": "INT", "domain": "POINT"},
                    Geometry=rs["Curve"], Name="tor_end", Value=endid)
    with g.frame("Sweep a circle (no caps: the ends are hidden in the collet and on the disc)"):
        circ = g.node("GeometryNodeCurvePrimitiveCircle", props={"mode": "RADIUS"}, Resolution=prof, Radius=r)
        mesh = g.node("GeometryNodeCurveToMesh", Curve=st["Geometry"], **{"Profile Curve": circ["Curve"]})
        sm = g.node("GeometryNodeSetShadeSmooth", props={"domain": "FACE"}, Geometry=mesh["Mesh"], **{"Shade Smooth": True})
        sm2 = g.node("GeometryNodeSetMaterial", Geometry=sm["Geometry"], Material=mat)
    g.output("Geometry", "GEOMETRY", sm2["Geometry"], desc="Wire mesh (point attribute tor_end)")
    return g.finish()
