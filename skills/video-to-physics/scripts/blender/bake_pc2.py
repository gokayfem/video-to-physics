"""Bake a SPEC deformation npz into a PC2 point cache driven by a Mesh Cache modifier (Blender 5.2+).

  blender -b --factory-startup --python bake_pc2.py -- --npz DEF.npz --fps 24 --out OUT.blend
          [--blend IN.blend] [--target OBJECT] [--name V2P_Strip] [--thickness 0.0005] [--pc2 PATH]
          [--match-tol 1e-4] [--json REPORT.json]

npz: positions (N, rows, cols, 3) world metres, t (N,) seconds, uniformly sampled (subframes allowed:
samples per frame spf = 1 / (dt * fps), passed to the modifier as frame_scale). Scene frame n <-> t = n / fps.
Without --target a grid strip object is generated from the mid-surface (two layers offset by +-thickness/2 along
the surface normal when thickness > 0, closed with side walls). With --target the existing object's vertices are
matched (nearest, world space, sample 0) to the generated vertex layout; the map is saved next to the PC2.
Run as a script; the helpers (strip_vertices, strip_faces, write_pc2, read_pc2, expected_positions) are reused by
check_bake.py.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ pure helpers (numpy only)
def strip_vertices(positions, thickness=0.0):
    """Mid-surface grid (N, rows, cols, 3) -> vertex positions (N, V, 3). Layout: [layer][row][col], layer 0 at
    +thickness/2 along n = d/drow x d/dcol, layer 1 at -thickness/2 (single layer when thickness == 0)."""
    P = np.asarray(positions, float)
    N, R, C, _ = P.shape
    if thickness <= 0:
        return P.reshape(N, R * C, 3)
    d_r = np.gradient(P, axis=1)
    d_c = np.gradient(P, axis=2)
    n = np.cross(d_r, d_c)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    off = 0.5 * thickness * n
    return np.concatenate([(P + off).reshape(N, R * C, 3), (P - off).reshape(N, R * C, 3)], axis=1)


def strip_faces(rows, cols, layered):
    """Quad faces for `strip_vertices`: top (and reversed bottom + side walls when layered)."""
    idx = lambda L, r, c: L * rows * cols + r * cols + c
    faces = [(idx(0, r, c), idx(0, r, c + 1), idx(0, r + 1, c + 1), idx(0, r + 1, c))
             for r in range(rows - 1) for c in range(cols - 1)]
    if not layered:
        return faces
    faces += [(idx(1, r, c), idx(1, r + 1, c), idx(1, r + 1, c + 1), idx(1, r, c + 1))
              for r in range(rows - 1) for c in range(cols - 1)]
    ring = ([(0, c) for c in range(cols)] + [(r, cols - 1) for r in range(1, rows)]
            + [(rows - 1, c) for c in range(cols - 2, -1, -1)] + [(r, 0) for r in range(rows - 2, 0, -1)])
    for (r0, c0), (r1, c1) in zip(ring, ring[1:] + ring[:1]):
        faces.append((idx(0, r1, c1), idx(0, r0, c0), idx(1, r0, c0), idx(1, r1, c1)))
    return faces


def write_pc2(path, pos_local, samples_per_frame, start=0.0):
    """PC2: header 'POINTCACHE2\\0', version 1, n_points, start frame, sample rate (frames per sample), n_samples;
    then float32 xyz per point per sample."""
    pos = np.ascontiguousarray(pos_local, "<f4")
    n, v, _ = pos.shape
    with open(path, "wb") as f:
        f.write(b"POINTCACHE2\0" + struct.pack("<iiffi", 1, v, float(start), 1.0 / samples_per_frame, n))
        f.write(pos.tobytes())


def read_pc2(path):
    b = Path(path).read_bytes()
    _, v, start, rate, n = struct.unpack("<iiffi", b[12:32])
    return np.frombuffer(b[32:], "<f4").reshape(n, v, 3), start, rate


def sample_timing(t, fps):
    """(frame of sample 0, samples per frame) for a uniformly sampled t."""
    t = np.asarray(t, float)
    dt = np.diff(t)
    if len(t) > 1 and np.max(np.abs(dt - dt[0])) > 1e-6 * dt[0]:
        raise ValueError("npz t must be uniformly sampled")
    spf = 1.0 / (dt[0] * fps) if len(t) > 1 else 1.0
    f0 = t[0] * fps
    spf = round(spf) if abs(spf - round(spf)) < 1e-9 * spf else spf
    f0 = round(f0) if abs(f0 - round(f0)) < 1e-9 * max(1.0, abs(f0)) else f0
    return float(f0), float(spf)


def expected_positions(npz_path, thickness, frame, fps, vmap=None):
    """World vertex positions the evaluated mesh must have at a (fractional) scene frame: linear interpolation of
    the sample sequence (exact at sample times)."""
    z = np.load(npz_path)
    V = strip_vertices(z["positions"], thickness)
    if vmap is not None:
        V = V[:, vmap]
    f0, spf = sample_timing(z["t"], fps)
    k = (frame - f0) * spf
    k0 = int(np.clip(np.floor(k + 1e-9), 0, len(V) - 1))
    k1 = min(k0 + 1, len(V) - 1)
    w = float(np.clip(k - k0, 0.0, 1.0))
    return (1 - w) * V[k0] + w * V[k1], k


# ------------------------------------------------------------------ Blender side
def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="npz deformation -> PC2 + Mesh Cache")
    p.add_argument("--npz", required=True)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--blend", default=None)
    p.add_argument("--target", default=None)
    p.add_argument("--name", default="V2P_Strip")
    p.add_argument("--thickness", type=float, default=None)
    p.add_argument("--pc2", default=None)
    p.add_argument("--match-tol", type=float, default=1e-4)
    p.add_argument("--json", default=None)
    return p.parse_args(argv)


def main():
    import bpy
    from mathutils import Matrix
    from mathutils.kdtree import KDTree

    a = _args()
    if a.blend:
        bpy.ops.wm.open_mainfile(filepath=a.blend)
    sc = bpy.context.scene
    z = np.load(a.npz)
    P, t = z["positions"], z["t"]
    fps = a.fps or (float(z["fps"]) if "fps" in z.files else float(sc.render.fps) / sc.render.fps_base)
    thick = a.thickness if a.thickness is not None else (float(z["thickness"]) if "thickness" in z.files else 0.0)
    N, R, C, _ = P.shape
    V = strip_vertices(P, thick)
    f0, spf = sample_timing(t, fps)
    out = Path(a.out).resolve()
    pc2 = Path(a.pc2).resolve() if a.pc2 else out.with_name(out.stem + f"_{a.target or a.name}.pc2")
    pc2.parent.mkdir(parents=True, exist_ok=True)
    vmap = None
    if a.target:
        ob = bpy.data.objects[a.target]
        mw = np.array(ob.matrix_world)
        nv = len(ob.data.vertices)
        co = np.empty(nv * 3)
        ob.data.vertices.foreach_get("co", co)
        world = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        kd = KDTree(V.shape[1])
        for i, p in enumerate(V[0]):
            kd.insert(p, i)
        kd.balance()
        found = [kd.find(p) for p in world]
        vmap = np.array([f[1] for f in found], int)
        dmax = max(f[2] for f in found)
        if dmax > a.match_tol:
            raise SystemExit(f"target vertices do not match the npz layout (max distance {dmax:.3g} m)")
        np.save(pc2.with_suffix(".map.npy"), vmap)
        inv = np.linalg.inv(mw)
        local = V[:, vmap] @ inv[:3, :3].T + inv[:3, 3]
    else:
        me = bpy.data.meshes.new(a.name)
        me.from_pydata([tuple(p) for p in V[0]], [], strip_faces(R, C, thick > 0))
        me.update()
        ob = bpy.data.objects.new(a.name, me)
        sc.collection.objects.link(ob)
        ob.matrix_world = Matrix.Identity(4)
        local = V
    write_pc2(pc2, local, spf)
    mc = ob.modifiers.new("V2P_MeshCache", "MESH_CACHE")
    mc.cache_format = "PC2"
    mc.filepath = str(pc2)
    mc.time_mode = "FRAME"
    mc.play_mode = "SCENE"
    mc.frame_start = f0 * spf          # Blender: cache sample = scene_frame * frame_scale - frame_start
    mc.frame_scale = spf
    mc.interpolation = "LINEAR"
    mc.deform_mode = "OVERWRITE"
    mc.forward_axis, mc.up_axis = "POS_Y", "POS_Z"
    ob["v2p_npz"] = str(Path(a.npz).resolve())
    ob["v2p_pc2"] = str(pc2)
    ob["v2p_thickness"] = thick
    ob["v2p_fps"] = fps
    ob["v2p_map"] = str(pc2.with_suffix(".map.npy")) if vmap is not None else ""
    sc.render.fps, sc.render.fps_base = int(round(fps)), round(fps) / fps
    last = f0 + (N - 1) / spf
    sc.frame_start, sc.frame_end = int(np.floor(f0)), int(np.floor(last + 1e-9))
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    rep = dict(object=ob.name, pc2=str(pc2), vertices=int(V.shape[1]), samples=int(N), samples_per_frame=spf,
               frame_start=f0, frame_end=last, thickness=thick, fps=fps, blend=str(out), mapped_target=bool(vmap is not None))
    print("V2P_BAKE " + json.dumps(rep))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
