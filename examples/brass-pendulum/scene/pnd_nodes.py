"""Geometry Nodes groups for the brass-pendulum stand, pin and wire loop (static parts only).

Built with nodekit (procedural-blender-nodes skill). Nothing here depends on time or on the swing animation:
the pendulum motion lives only on the PND_Swing empty (keyframes baked by solver/bake_blender.py).

Groups
  PND Rounded Block  box with separately rounded vertical corners, top edges and bottom edges
  PND Stand          (modifier) walnut base + post + arm + steel post bolt from a few macro controls;
                     stores face attribute `pnd_part` (0 base, 1 post, 2 arm, 3 bolt)
  PND Stand Part     (modifier) one part of a PND Stand object, picked by `pnd_part`
  PND Pin            (modifier) polished steel pin along +X: shaft from the arm end face, rounded head
  PND Wire Loop      (modifier) wire ring around the pin whose lowest centreline point is the pivot P
All lengths are metres in world space (pivot P at the origin, X right in the swing plane, Z up).
"""
import math
import sys

import pathlib

# nodekit is vendored (unmodified, MIT) in scene/vendor/ -- see scene/vendor/VENDORED.json for source commit and hashes
VENDOR = str(pathlib.Path(__file__).resolve().parent / "vendor")
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

from nodekit.graph import Graph, once  # noqa: E402

PART_IDS = {"base": 0, "post": 1, "arm": 2, "bolt": 3}


def _xyz(g, x, y, z, label=None):
    return g.combine(x, y, z, label=label)


@once
def rounded_block():
    g = Graph("PND Rounded Block", description="Box of a given size and centre whose vertical corner edges, top "
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
        on = g.node("FunctionNodeCompare", props={"data_type": "FLOAT", "operation": "GREATER_THAN"},
                    A=radius, B=1e-7)
        return g.node("GeometryNodeSwitch", props={"input_type": "GEOMETRY"}, Switch=on["Result"],
                      **{"False": mesh, "True": b["Mesh"]})

    with g.frame("Box at the origin; half height for edge selection"):
        cube = g.node("GeometryNodeMeshCube", Size=size)
        hz = g.math("MULTIPLY", g.separate(size)["Z"], 0.5)
        ev = g.node("GeometryNodeInputMeshEdgeVertices")
        z1, z2 = g.separate(ev["Position 1"])["Z"], g.separate(ev["Position 2"])["Z"]
    with g.frame("Round the vertical edges (edges whose ends differ in Z)"):
        dz = g.math("ABSOLUTE", g.math("SUBTRACT", z1, z2))
        vsel = g.node("FunctionNodeCompare", props={"data_type": "FLOAT", "operation": "GREATER_THAN"}, A=dz, B=1e-6)
        m1 = bevel(cube["Mesh"], vsel["Result"], rc)
    with g.frame("Round the top edges (both ends on the top plane)"):
        tsel = g.node("FunctionNodeCompare", props={"data_type": "FLOAT", "operation": "GREATER_THAN"},
                      A=g.math("MINIMUM", z1, z2), B=g.math("SUBTRACT", hz, 1e-6))
        m2 = bevel(m1, tsel["Result"], rt)
    with g.frame("Round the bottom edges (both ends on the bottom plane)"):
        bsel = g.node("FunctionNodeCompare", props={"data_type": "FLOAT", "operation": "LESS_THAN"},
                      A=g.math("MAXIMUM", z1, z2), B=g.math("ADD", g.math("MULTIPLY", hz, -1.0), 1e-6))
        m3 = bevel(m2, bsel["Result"], rb)
    with g.frame("Smooth shading with sharp edges above 30 degrees, then move to Centre"):
        sm = g.node("GeometryNodeSetShadeSmooth", props={"domain": "FACE"}, Geometry=m3, **{"Shade Smooth": True})
        ang = g.node("GeometryNodeInputMeshEdgeAngle")
        sharp = g.node("FunctionNodeCompare", props={"data_type": "FLOAT", "operation": "GREATER_THAN"},
                       A=ang["Unsigned Angle"], B=math.radians(30.0))
        sm2 = g.node("GeometryNodeSetShadeSmooth", props={"domain": "EDGE"}, Geometry=sm["Geometry"],
                     Selection=sharp["Result"], **{"Shade Smooth": False})
        out = g.node("GeometryNodeTransform", Geometry=sm2["Geometry"], Translation=centre)
    g.output("Mesh", "GEOMETRY", out["Geometry"], desc="The rounded box")
    return g.finish()


def _tag(g, geo, part):
    st = g.node("GeometryNodeStoreNamedAttribute", props={"data_type": "INT", "domain": "FACE"},
                Geometry=geo, Name="pnd_part", Value=part)
    return st["Geometry"]


@once
def stand():
    rb = rounded_block()
    g = Graph("PND Stand", description="Walnut pendulum stand (base slab, square post, horizontal arm, steel bolt) "
              "laid out around the pivot P at the origin. The arm end face stays at Arm End X when other controls "
              "change, so the pin and the pendulum never move; the post and base follow the arm.", modifier=True)
    # Primary
    post_h = g.input("Post Height", "FLOAT", 0.49, min=0.05, max=5.0, subtype="DISTANCE", panel="Primary",
                     desc="Post top above the base top surface")
    arm_len = g.input("Arm Length", "FLOAT", 0.17, min=0.02, max=5.0, subtype="DISTANCE", panel="Primary",
                      desc="Visible arm length from the post's +X face to the arm end face")
    bw = g.input("Base Width", "FLOAT", 0.5, min=0.05, max=5.0, subtype="DISTANCE", panel="Primary",
                 desc="Base size along X (swing plane direction)")
    bd = g.input("Base Depth", "FLOAT", 0.23, min=0.05, max=5.0, subtype="DISTANCE", panel="Primary",
                 desc="Base size along Y (towards / away from the camera)")
    bt = g.input("Base Thickness", "FLOAT", 0.065, min=0.005, max=1.0, subtype="DISTANCE", panel="Primary",
                 desc="Base slab thickness")
    btr = g.input("Base Edge Radius", "FLOAT", 0.0085, min=0.0, max=0.2, subtype="DISTANCE", panel="Primary",
                  desc="Rounding of the base's top edges")
    wer = g.input("Wood Edge Radius", "FLOAT", 0.0023, min=0.0, max=0.05, subtype="DISTANCE", panel="Primary",
                  desc="Edge softening of the post and arm")
    # Placement
    hp = g.input("Pivot Height", "FLOAT", 0.44, min=0.01, max=5.0, subtype="DISTANCE", panel="Placement",
                 desc="Height of the pivot P above the base top (base top is at z = -Pivot Height)")
    axe = g.input("Arm End X", "FLOAT", -0.027, min=-1.0, max=1.0, subtype="DISTANCE", panel="Placement",
                  desc="X of the arm end face relative to the pivot (negative: the face is left of P)")
    acz = g.input("Arm Centre Z", "FLOAT", 0.0125, min=-1.0, max=1.0, subtype="DISTANCE", panel="Placement",
                  desc="Height of the arm centre line (= pin axis) above P")
    ah = g.input("Arm Height", "FLOAT", 0.045, min=0.005, max=1.0, subtype="DISTANCE", panel="Placement",
                 desc="Arm section height (Z)")
    ad = g.input("Arm Depth", "FLOAT", 0.031, min=0.005, max=1.0, subtype="DISTANCE", panel="Placement",
                 desc="Arm section depth (Y); the arm is centred on the swing plane y = 0")
    psx = g.input("Post Size X", "FLOAT", 0.068, min=0.005, max=1.0, subtype="DISTANCE", panel="Placement",
                  desc="Post section along X")
    psy = g.input("Post Size Y", "FLOAT", 0.055, min=0.005, max=1.0, subtype="DISTANCE", panel="Placement",
                  desc="Post section along Y")
    pcy = g.input("Post Centre Y", "FLOAT", 0.0, min=-1.0, max=1.0, subtype="DISTANCE", panel="Placement",
                  desc="Post centre offset in Y")
    bml = g.input("Base Left Margin", "FLOAT", 0.044, min=0.0, max=2.0, subtype="DISTANCE", panel="Placement",
                  desc="Distance from the base's left (-X) edge to the post's left face")
    bcy = g.input("Base Centre Y", "FLOAT", -0.036, min=-2.0, max=2.0, subtype="DISTANCE", panel="Placement",
                  desc="Base centre offset in Y (negative: towards the camera)")
    bcr = g.input("Base Corner Radius", "FLOAT", 0.0116, min=0.0, max=0.5, subtype="DISTANCE", panel="Placement",
                  desc="Plan-view rounding of the base's vertical corners")
    bbr = g.input("Base Bottom Edge Radius", "FLOAT", 0.003, min=0.0, max=0.2, subtype="DISTANCE",
                  panel="Placement", desc="Rounding of the base's bottom edges")
    # Detail
    blr = g.input("Bolt Radius", "FLOAT", 0.008, min=0.0, max=0.2, subtype="DISTANCE", panel="Detail",
                  desc="Radius of the steel bolt cap on the post's -X face (0 hides it)")
    blp = g.input("Bolt Protrusion", "FLOAT", 0.007, min=0.0, max=0.2, subtype="DISTANCE", panel="Detail",
                  desc="How far the bolt cap stands out of the post")
    seg = g.input("Bevel Segments", "INT", 6, min=1, max=24, panel="Detail", desc="Segments per rounded edge")
    g._panels["Detail"].default_closed = True

    with g.frame("Layout: post centre X from the arm end, base top, post and base X spans"):
        half_psx = g.math("MULTIPLY", psx, 0.5)
        post_right = g.math("SUBTRACT", axe, arm_len, label="post +X face")
        pcx = g.math("SUBTRACT", post_right, half_psx, label="post centre X")
        post_left = g.math("SUBTRACT", pcx, half_psx, label="post -X face")
        ztop = g.math("MULTIPLY", hp, -1.0, label="base top z")
        bleft = g.math("SUBTRACT", post_left, bml, label="base -X edge")
        bcx = g.math("ADD", bleft, g.math("MULTIPLY", bw, 0.5), label="base centre X")
    with g.frame("Base: rounded slab under the base top plane"):
        bcz = g.math("SUBTRACT", ztop, g.math("MULTIPLY", bt, 0.5))
        base = g.group(rb, "Base", Size=_xyz(g, bw, bd, bt), Centre=_xyz(g, bcx, bcy, bcz),
                       **{"Corner Radius": bcr, "Top Edge Radius": btr, "Bottom Edge Radius": bbr, "Segments": seg})
        base_t = _tag(g, base["Mesh"], PART_IDS["base"])
    with g.frame("Post: square column standing on the base top (sunk 1 mm into it)"):
        sink = 0.001
        pz = g.math("ADD", ztop, g.math("MULTIPLY", post_h, 0.5))
        post = g.group(rb, "Post", Size=_xyz(g, psx, psy, g.math("ADD", post_h, sink)),
                       Centre=_xyz(g, pcx, pcy, g.math("SUBTRACT", pz, sink / 2)),
                       **{"Corner Radius": wer, "Top Edge Radius": wer, "Bottom Edge Radius": 0.0, "Segments": seg})
        post_t = _tag(g, post["Mesh"], PART_IDS["post"])
    with g.frame("Arm: from the post centre (hidden inside the post) to the arm end face"):
        alen = g.math("SUBTRACT", axe, pcx)
        acx = g.math("MULTIPLY", g.math("ADD", axe, pcx), 0.5)
        # model the arm with its long axis on Z, round it, then lay it along +X
        arm = g.group(rb, "Arm", Size=_xyz(g, ah, ad, alen), Centre=_xyz(g, 0.0, 0.0, 0.0),
                      **{"Corner Radius": wer, "Top Edge Radius": wer, "Bottom Edge Radius": wer, "Segments": seg})
        arm_x = g.node("GeometryNodeTransform", Geometry=arm["Mesh"], Rotation=(0.0, math.pi / 2, 0.0),
                       Translation=_xyz(g, acx, 0.0, acz))
        arm_t = _tag(g, arm_x["Geometry"], PART_IDS["arm"])
    with g.frame("Bolt: flattened steel dome on the post's -X face at arm height"):
        dome = g.node("GeometryNodeMeshUVSphere", Segments=32, Rings=16, Radius=1.0)
        sm = g.node("GeometryNodeSetShadeSmooth", props={"domain": "FACE"}, Geometry=dome["Mesh"],
                    **{"Shade Smooth": True})
        bolt = g.node("GeometryNodeTransform", Geometry=sm["Geometry"], Translation=_xyz(g, post_left, 0.0, acz),
                      Scale=_xyz(g, blp, blr, blr))
        show = g.node("FunctionNodeCompare", props={"data_type": "FLOAT", "operation": "GREATER_THAN"},
                      A=g.math("MINIMUM", blr, blp), B=1e-7)
        bolt_s = g.node("GeometryNodeSwitch", props={"input_type": "GEOMETRY"}, Switch=show["Result"],
                        **{"True": bolt["Geometry"]})
        bolt_t = _tag(g, bolt_s["Output"], PART_IDS["bolt"])
    with g.frame("Join all parts"):
        join = g.node("GeometryNodeJoinGeometry")
        for geo in (base_t, post_t, arm_t, bolt_t):
            g.connect(geo, join.i("Geometry"))
    g.output("Geometry", "GEOMETRY", join["Geometry"], desc="Stand mesh with face attribute pnd_part")
    return g.finish()


@once
def stand_part():
    g = Graph("PND Stand Part", description="Takes the evaluated mesh of a PND Stand object and keeps one part "
              "(pnd_part: 0 base, 1 post, 2 arm, 3 bolt), so each part is its own object with its own material "
              "while all controls stay on the one PND Stand modifier", modifier=True)
    obj = g.input("Stand", "OBJECT", desc="Object carrying the PND Stand modifier")
    part = g.input("Part", "INT", 0, min=0, max=3, desc="0 base, 1 post, 2 arm, 3 bolt")
    mat = g.input("Material", "MATERIAL", desc="Material of this part")
    with g.frame("Read the stand and keep the faces of one part"):
        info = g.node("GeometryNodeObjectInfo", props={"transform_space": "ORIGINAL"}, Object=obj)
        attr = g.node("GeometryNodeInputNamedAttribute", props={"data_type": "INT"}, Name="pnd_part")
        eq = g.node("FunctionNodeCompare", props={"data_type": "INT", "operation": "EQUAL"},
                    A=attr["Attribute"], B=part)
        sep = g.node("GeometryNodeSeparateGeometry", props={"domain": "FACE"}, Geometry=info["Geometry"],
                     Selection=eq["Result"])
        sm = g.node("GeometryNodeSetMaterial", Geometry=sep["Selection"], Material=mat)
    g.output("Geometry", "GEOMETRY", sm["Geometry"], desc="Faces of the chosen part")
    return g.finish()


@once
def pin():
    g = Graph("PND Pin", description="Polished steel pin along +X: a shaft from inside the arm end face to a "
              "rounded head; the wire loop hangs on the shaft against the head's shoulder", modifier=True)
    sr = g.input("Shaft Radius", "FLOAT", 0.0085, min=0.0005, max=0.2, subtype="DISTANCE", desc="Shaft radius")
    sx = g.input("Shaft Start X", "FLOAT", -0.03, min=-1.0, max=1.0, subtype="DISTANCE",
                 desc="Shaft start X (set a little inside the arm, behind its end face)")
    hx = g.input("Head Start X", "FLOAT", 0.0015, min=-1.0, max=1.0, subtype="DISTANCE",
                 desc="X of the head's shoulder")
    hr = g.input("Head Radius", "FLOAT", 0.0105, min=0.0005, max=0.2, subtype="DISTANCE", desc="Head radius")
    tx = g.input("Tip X", "FLOAT", 0.023, min=-1.0, max=1.0, subtype="DISTANCE", desc="X of the rounded tip")
    az = g.input("Axis Z", "FLOAT", 0.0125, min=-1.0, max=1.0, subtype="DISTANCE", desc="Height of the pin axis")
    nv = g.input("Resolution", "INT", 48, min=8, max=256, desc="Vertices around the pin")
    mat = g.input("Material", "MATERIAL", desc="Pin material (polished steel)")
    rot = (0.0, math.pi / 2, 0.0)        # cylinder axis Z -> X
    with g.frame("Shaft: cylinder from Shaft Start X to the head"):
        sl = g.math("SUBTRACT", hx, sx)
        shaft = g.node("GeometryNodeMeshCylinder", Vertices=nv, Radius=sr, Depth=sl)
        shaft_x = g.node("GeometryNodeTransform", Geometry=shaft["Mesh"], Rotation=rot,
                         Translation=_xyz(g, g.math("MULTIPLY", g.math("ADD", hx, sx), 0.5), 0.0, az))
    with g.frame("Head: short cylinder plus a hemispherical dome ending at Tip X"):
        dome_c = g.math("SUBTRACT", tx, hr)
        hl = g.math("MAXIMUM", g.math("SUBTRACT", dome_c, hx), 1e-5)
        head = g.node("GeometryNodeMeshCylinder", Vertices=nv, Radius=hr, Depth=hl)
        head_x = g.node("GeometryNodeTransform", Geometry=head["Mesh"], Rotation=rot,
                        Translation=_xyz(g, g.math("MULTIPLY", g.math("ADD", hx, dome_c), 0.5), 0.0, az))
        dome = g.node("GeometryNodeMeshUVSphere", Segments=nv, Rings=24, Radius=hr)
        dome_x = g.node("GeometryNodeTransform", Geometry=dome["Mesh"], Rotation=rot,
                        Translation=_xyz(g, dome_c, 0.0, az))
    with g.frame("Join; smooth shading with sharp rims"):
        join = g.node("GeometryNodeJoinGeometry")
        for geo in (shaft_x["Geometry"], head_x["Geometry"], dome_x["Geometry"]):
            g.connect(geo, join.i("Geometry"))
        sm = g.node("GeometryNodeSetShadeSmooth", props={"domain": "FACE"}, Geometry=join["Geometry"],
                    **{"Shade Smooth": True})
        ang = g.node("GeometryNodeInputMeshEdgeAngle")
        sharp = g.node("FunctionNodeCompare", props={"data_type": "FLOAT", "operation": "GREATER_THAN"},
                       A=ang["Unsigned Angle"], B=math.radians(45.0))
        sm2 = g.node("GeometryNodeSetShadeSmooth", props={"domain": "EDGE"}, Geometry=sm["Geometry"],
                     Selection=sharp["Result"], **{"Shade Smooth": False})
        sm3 = g.node("GeometryNodeSetMaterial", Geometry=sm2["Geometry"], Material=mat)
    g.output("Geometry", "GEOMETRY", sm3["Geometry"], desc="Pin mesh")
    return g.finish()


@once
def wire_loop():
    g = Graph("PND Wire Loop", description="Tight wire ring around the pin (ring plane YZ through x = 0, concentric "
              "with the pin axis) closed by a short twisted V that ends at the pivot P = origin, where the hanging "
              "wire leaves it (VR-11)", modifier=True)
    lr = g.input("Loop Radius", "FLOAT", 0.011, min=0.0005, max=0.2, subtype="DISTANCE",
                 desc="Centreline radius of the ring (tight: pin head radius + wire radius)")
    az = g.input("Axis Z", "FLOAT", 0.015, min=0.0, max=0.5, subtype="DISTANCE",
                 desc="Height of the pin axis (ring centre) above P; must exceed Loop Radius")
    wr = g.input("Wire Radius", "FLOAT", 0.0007, min=0.00005, max=0.02, subtype="DISTANCE", desc="Wire radius")
    ta = g.input("Tail Angle", "FLOAT", math.radians(25.0), min=0.0, max=1.2, subtype="ANGLE",
                 desc="Half-angle from the ring bottom of the two points where the twisted V leaves the ring")
    res = g.input("Resolution", "INT", 48, min=8, max=256, desc="Segments around the ring")
    mat = g.input("Material", "MATERIAL", desc="Wire material")
    rot = (0.0, math.pi / 2, 0.0)
    with g.frame("Ring in the YZ plane centred on the pin axis"):
        circ = g.node("GeometryNodeCurvePrimitiveCircle", props={"mode": "RADIUS"}, Resolution=res, Radius=lr)
        ring = g.node("GeometryNodeTransform", Geometry=circ["Curve"], Rotation=rot, Translation=_xyz(g, 0.0, 0.0, az))
    with g.frame("Twisted V: two short wires from the ring (at +-Tail Angle from its bottom) to P"):
        sy = g.math("MULTIPLY", lr, g.math("SINE", ta))
        zc = g.math("SUBTRACT", az, g.math("MULTIPLY", lr, g.math("COSINE", ta)))
        l1 = g.node("GeometryNodeCurvePrimitiveLine", Start=_xyz(g, 0.0, sy, zc), End=(0.0, 0.0, 0.0))
        l2 = g.node("GeometryNodeCurvePrimitiveLine", Start=_xyz(g, 0.0, g.math("MULTIPLY", sy, -1.0), zc),
                    End=(0.0, 0.0, 0.0))
        cj = g.node("GeometryNodeJoinGeometry")
        for c in (ring["Geometry"], l1["Curve"], l2["Curve"]):
            g.connect(c, cj.i("Geometry"))
    with g.frame("Sweep a round wire profile; smooth; material"):
        prof = g.node("GeometryNodeCurvePrimitiveCircle", props={"mode": "RADIUS"}, Resolution=12, Radius=wr)
        tube = g.node("GeometryNodeCurveToMesh", Curve=cj["Geometry"], **{"Profile Curve": prof["Curve"], "Fill Caps": True})
        sm = g.node("GeometryNodeSetShadeSmooth", props={"domain": "FACE"}, Geometry=tube["Mesh"],
                    **{"Shade Smooth": True})
        sm2 = g.node("GeometryNodeSetMaterial", Geometry=sm["Geometry"], Material=mat)
    g.output("Geometry", "GEOMETRY", sm2["Geometry"], desc="Loop mesh")
    return g.finish()
