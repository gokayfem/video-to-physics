"""Rig check for clip 03 (scene role C).

blender -b scene/out/torsion.blend --python scene/check_rig.py [-- --frames] [--out scene/out/check_rig.json]
TEST poses of TOR_Disc (twist psi in {-40, -10, 0, 10, 40} deg x B offset {0, +-3 mm X, +-3 mm Y, -2 mm Z} x tilt
{0, (+2, 0), (0, -2)} deg; B = rest point + offset) — or, with --frames, the baked animation at every frame + 3 subframes.
Per pose, on the evaluated depsgraph:
  * wire ends: centroid of the evaluated TOR_Wire ring tor_end = 0 vs TOR_Anchor world origin, ring 1 vs TOR_Disc origin;
    TOR_WireEnd world location vs TOR_Disc origin; centreline straightness (max distance of ring centroids from A-B)
  * intersections: BVH overlap of TOR_Disc with every stand part and with TOR_Wire (excluding the wire's last ring at B,
    which sits on the top face by design); TOR_Wire with stand parts except the collet (the wire's top ring sits on the
    collet bottom at A by design); disc-to-table clearance (lowest disc vertex z - table z)
  * the anchor never moves from its built position (custom property tor_pivot_m); the disc mesh is rigid (vertex set equals the rest mesh transformed by matrix_world)
"""
import json, math, pathlib, sys
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

HERE = pathlib.Path(bpy.data.filepath).resolve().parent
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
FRAMES = "--frames" in argv
OUT = argv[argv.index("--out") + 1] if "--out" in argv else str(pathlib.Path(__file__).resolve().parent / "out" / ("check_rig_frames.json" if FRAMES else "check_rig.json"))
TOL = 1e-6
sc = bpy.context.scene
A_ob, disc, wend, wire = (bpy.data.objects[n] for n in ("TOR_Anchor", "TOR_Disc", "TOR_WireEnd", "TOR_Wire"))
STAND = [n for n in ("TOR_Posts", "TOR_Crossbar", "TOR_Feet", "TOR_Clamp", "TOR_Screws", "TOR_Collet", "TOR_ScrewSockets") if n in bpy.data.objects]
table = bpy.data.objects.get("TOR_Look_Table")
rest_loc = disc.location.copy()
disc.rotation_mode = "QUATERNION"
rest_rot = disc.rotation_quaternion.copy()
PIVOT = np.array(A_ob.get("tor_pivot_m", [0.0, 0.0, 0.0]), float)     # intended (built) suspension point


def ev_mesh(ob, dg):
    e = ob.evaluated_get(dg)
    m = e.to_mesh()
    n = len(m.vertices)
    co = np.empty(n * 3); m.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    M = np.array(e.matrix_world)
    W = co @ M[:3, :3].T + M[:3, 3]
    attr = None
    if "tor_end" in m.attributes:
        a = np.empty(n, dtype=np.int32); m.attributes["tor_end"].data.foreach_get("value", a); attr = a
    e.to_mesh_clear()
    return W, attr


def bvh(ob, dg):
    """World-space BVH (BVHTree.FromObject works in object-local coordinates)."""
    e = ob.evaluated_get(dg)
    m = e.to_mesh()
    M = e.matrix_world
    verts = [M @ v.co for v in m.vertices]
    polys = [tuple(p.vertices) for p in m.polygons]
    e.to_mesh_clear()
    return BVHTree.FromPolygons(verts, polys, epsilon=0.0)


def check(dg, label):
    r = {"pose": label}
    A = np.array(A_ob.matrix_world.translation)
    B = np.array(disc.matrix_world.translation)
    Wv, end = ev_mesh(wire, dg)
    c0, c1 = Wv[end == 0].mean(0), Wv[end == 1].mean(0)
    r["wire_top_err_m"] = float(np.linalg.norm(c0 - A))
    r["wire_bottom_err_m"] = float(np.linalg.norm(c1 - B))
    r["wireend_err_m"] = float(np.linalg.norm(np.array(wend.matrix_world.translation) - B))
    r["anchor_drift_m"] = float(np.linalg.norm(A - PIVOT))
    # straightness: every vertex lies at distance ~radius from the A-B line
    u = (B - A) / np.linalg.norm(B - A)
    d = Wv - A
    dist = np.linalg.norm(d - np.outer(d @ u, u), axis=1)
    r["wire_dist_from_axis_m"] = [float(dist.min()), float(dist.max())]
    r["wire_length_m"] = float(np.linalg.norm(B - A))
    # rigidity of the disc mesh
    Dv, _ = ev_mesh(disc, dg)
    M = np.array(disc.matrix_world)
    loc = (Dv - M[:3, 3]) @ M[:3, :3]                         # back to local (orthonormal, unit scale)
    r["disc_local_max_dev_m"] = float(np.abs(loc - REST_LOCAL).max())
    # intersections
    bd = bvh(disc, dg)
    hits = {}
    for n in STAND:
        ov = bd.overlap(bvh(bpy.data.objects[n], dg))
        if ov:
            hits["disc~" + n] = len(ov)
    bw = bvh(wire, dg)
    for n in STAND:
        if n == "TOR_Collet":
            continue
        ov = bw.overlap(bvh(bpy.data.objects[n], dg))
        if ov:
            hits["wire~" + n] = len(ov)
    # wire vs disc: only faces of the wire's lowest segment may touch the top face (ring at B by design)
    ovw = bw.overlap(bd)
    if ovw:
        e = wire.evaluated_get(dg).to_mesh()
        zmin_ok = []
        for fi, _ in ovw:
            p = e.polygons[fi]
            zs = [np.array(wire.matrix_world @ e.vertices[v].co) for v in p.vertices]
            zmin_ok.append(min(np.linalg.norm(z - B) for z in zs) < 0.06 * np.linalg.norm(B - A) + 1e-4)
        wire.evaluated_get(dg).to_mesh_clear()
        if not all(zmin_ok):
            hits["wire~TOR_Disc(away from B)"] = int(sum(1 for x in zmin_ok if not x))
        r["wire_disc_contact_faces_at_B"] = int(sum(zmin_ok))
    r["intersections"] = hits
    if table is not None:
        r["disc_table_clearance_m"] = float(Dv[:, 2].min() - table.matrix_world.translation.z - min(v.co.z for v in table.data.vertices))
    # wire must stay inside the collet bore region? (top ring centre at A; report the wire top z)
    col = bpy.data.objects.get("TOR_Collet")
    if col is not None:
        cv, _ = ev_mesh(col, dg)
        rc = float(np.max(np.hypot(cv[:, 0], cv[:, 1])))
        r["wire_top_inside_collet_footprint"] = bool(np.hypot(A[0], A[1]) + 0.0 < rc and abs(A[2] - cv[:, 2].min()) < 1e-4)
    r["pass"] = (r.get("wire_top_inside_collet_footprint", True) and r["wire_top_err_m"] < TOL and r["wire_bottom_err_m"] < TOL and r["wireend_err_m"] < TOL and r["anchor_drift_m"] < TOL
                 and r["disc_local_max_dev_m"] < 1e-6 and not hits and r.get("disc_table_clearance_m", 1) > 0)
    return r


dg = bpy.context.evaluated_depsgraph_get()
sc.frame_set(0)
# rest-local disc vertices (from the mesh data; the disc has no modifiers)
REST_LOCAL = np.array([v.co[:] for v in disc.data.vertices])
results = []
if FRAMES:
    for n in range(sc.frame_start, sc.frame_end + 1):
        for s in (0.0, 0.25, 0.5, 0.75):
            if n == sc.frame_end and s > 0:
                continue
            sc.frame_set(n, subframe=s)
            dg = bpy.context.evaluated_depsgraph_get()
            results.append(check(dg, f"frame {n + s}"))
else:
    if disc.animation_data:
        disc.animation_data.action = None
    for psi in (-40.0, -10.0, 0.0, 10.0, 40.0):
        for off in ((0, 0, 0), (3, 0, 0), (-3, 0, 0), (0, 3, 0), (0, -3, 0), (0, 0, -2)):
            for tilt in ((0, 0), (2, 0), (0, -2)):
                disc.location = rest_loc + Vector(off) / 1000.0
                Rm = Matrix.Rotation(math.radians(tilt[1]), 3, "Y") @ Matrix.Rotation(math.radians(tilt[0]), 3, "X") @ \
                    Matrix.Rotation(math.radians(psi), 3, "Z")
                disc.rotation_quaternion = Rm.to_quaternion()
                bpy.context.view_layer.update()
                dg = bpy.context.evaluated_depsgraph_get()
                results.append(check(dg, {"psi_deg": psi, "B_offset_mm": off, "tilt_xy_deg": tilt}))
    disc.location, disc.rotation_quaternion = rest_loc, rest_rot
summ = {"n_poses": len(results), "all_pass": all(r["pass"] for r in results),
        "max_wire_top_err_m": max(r["wire_top_err_m"] for r in results),
        "max_wire_bottom_err_m": max(r["wire_bottom_err_m"] for r in results),
        "max_wireend_err_m": max(r["wireend_err_m"] for r in results),
        "max_anchor_drift_m": max(r["anchor_drift_m"] for r in results),
        "max_disc_local_dev_m": max(r["disc_local_max_dev_m"] for r in results),
        "wire_axis_distance_range_m": [min(r["wire_dist_from_axis_m"][0] for r in results), max(r["wire_dist_from_axis_m"][1] for r in results)],
        "intersections": sorted({k for r in results for k in r["intersections"]}),
        "min_disc_table_clearance_m": min(r.get("disc_table_clearance_m", 9) for r in results),
        "wire_length_range_m": [min(r["wire_length_m"] for r in results), max(r["wire_length_m"] for r in results)]}
pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(OUT).write_text(json.dumps({"file": bpy.data.filepath, "mode": "frames" if FRAMES else "test poses", "summary": summ,
                                         "poses": results}, indent=1, default=str))
print("TOR_CHECK_RIG", json.dumps(summ))
