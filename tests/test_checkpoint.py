"""Checkpoint: frozen copy used by the build, outputs present, manifest verifies, never overwritten;
the real templates/build.json shape (freeze/copy/exclude/commands/outputs, {ck}) is honoured."""
import json
import sys
from pathlib import Path

from synth import tmpdir
from v2p.checkpoint import create, verify

TEMPLATES = Path(__file__).resolve().parents[1] / "skills" / "video-to-physics" / "templates"
PY = sys.executable
QUIET = dict(log=lambda *_: None)


def _case(build):
    c = tmpdir() / "case"
    (c / "config").mkdir(parents=True)
    (c / "params").mkdir()
    (c / "params" / "p.json").write_text('{"k": 1}')
    (c / "config" / "build.json").write_text(json.dumps(build))
    return c


def test_checkpoint_build_manifest_immutable():
    cmd = f"{PY} -c \"import os,shutil; os.makedirs('{{ck}}/out'); shutil.copy('{{ck}}/params/p.json', '{{ck}}/out/r.json')\""
    c = _case({"freeze": ["params"], "commands": [cmd], "outputs": ["out/r.json"]})
    create(c, "ck001", **QUIET)
    ck = c / "checkpoints" / "ck001"
    assert (ck / "READY").exists() and json.loads((ck / "out" / "r.json").read_text()) == {"k": 1}
    rels = {l.split("  ", 1)[1] for l in (ck / "MANIFEST.sha256").read_text().splitlines()}
    assert {"params/p.json", "out/r.json", "build.log", "checkpoint.json"} <= rels
    assert verify(ck) == []
    try:
        create(c, "ck001", **QUIET)
        raise AssertionError("existing checkpoint was overwritten")
    except FileExistsError:
        pass
    f = ck / "out" / "r.json"
    f.chmod(0o644)
    f.write_text('{"k": 2}')
    assert any("changed" in p for p in verify(ck))


def test_checkpoint_failure_leaves_no_ready():
    c = _case({"commands": ["exit 3"]})
    try:
        create(c, "ck002", **QUIET)
        raise AssertionError("failure not raised")
    except RuntimeError:
        pass
    ck = c / "checkpoints" / "ck002"
    assert (ck / "FAILED").exists() and not (ck / "READY").exists()
    c2 = _case({"commands": [], "outputs": ["nothing.bin"]})
    try:
        create(c2, "ck001", **QUIET)
        raise AssertionError("missing output not detected")
    except RuntimeError:
        assert not (c2 / "checkpoints" / "ck001" / "READY").exists()


def test_real_build_template():
    build = json.loads((TEMPLATES / "build.json").read_text())
    c = tmpdir() / "case"
    for rel in [e["from"] if isinstance(e, dict) else e for e in build["freeze"]]:
        (c / rel).parent.mkdir(parents=True, exist_ok=True)
        (c / rel).write_bytes(b"frozen")
    for rel in build["copy"]:
        p = c / rel
        if "." in Path(rel).name:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x")
        else:
            (p / "sub").mkdir(parents=True, exist_ok=True)
            (p / "keep.py").write_text("x")
            (p / "__pycache__").mkdir()
            (p / "__pycache__" / "k.pyc").write_text("x")
            (p / "a.blend1").write_text("x")
    (c / "scene" / "out").mkdir(parents=True, exist_ok=True)
    (c / "scene" / "out" / "render.png").write_text("x")
    # keep the template's freeze/copy/exclude/outputs, swap its Blender commands for synthetic ones
    writes = {o: f"{{ck}}/{o}" for o in build["outputs"]}
    build["commands"] = [f"{PY} -c \"import os; p='{dst}'; os.makedirs(os.path.dirname(p), exist_ok=True); "
                         f"open(p,'w').write(open('{{ck}}/{build['freeze'][0]['to']}').read())\""
                         for dst in writes.values()]
    (c / "config").mkdir(exist_ok=True)
    (c / "config" / "build.json").write_text(json.dumps(build))
    create(c, "ck001", **QUIET)
    ck = c / "checkpoints" / "ck001"
    for o in build["outputs"]:
        assert (ck / o).read_text() == "frozen"
    assert (ck / "scene" / "keep.py").exists()
    assert not (ck / "scene" / "out").exists() and not (ck / "scene" / "a.blend1").exists()
    assert not any(ck.rglob("__pycache__"))
    assert verify(ck) == []
