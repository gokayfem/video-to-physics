"""Fallback runner for scene/build.py when the builder file is mid-edit and broken (Look role, read-only use).

blender -b --factory-startup --python look/build_shim.py -- <same args as scene/build.py>
Executes scene/build.py's source unchanged except for an in-memory rename of the local float `rel` (which, in
the 01:5x builder revision, shadows the module function rel() and raises TypeError). scene/build.py itself is not
modified. Use only while that bug is present; reported to the geometry role in look/NOTES.md.
"""
import pathlib
p = pathlib.Path(__file__).resolve().parents[1] / "scene" / "build.py"
src = p.read_text()
src = src.replace("    rel = (r_phys - r_geo) / r_geo", "    rel_r = (r_phys - r_geo) / r_geo")
src = src.replace("diff {rel:+.2%}", "diff {rel_r:+.2%}")
exec(compile(src, str(p), "exec"), {"__file__": str(p), "__name__": "__main__"})
