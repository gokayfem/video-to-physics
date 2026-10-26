"""`new`: scaffold a case directory from a video, an image, an idea (text or text file) or nothing.

Creates CONTRACT.md, STATUS.json, config/{case.json, acceptance.json, CHANGELOG.md, build.json},
reference/ (immutable copy of the input + SHA256 + meta.json), role dirs with visuals/ and PROGRESS.log,
review/{page.json, visuals/} and checkpoints/.

Templates are read from skills/video-to-physics/templates/ (CONTRACT.md, acceptance.json, page.json) relative
to this file, with minimal built-ins as fallback. Known `{{KEY}}` placeholders are filled: TITLE, CASE, DATE (=NOW),
INPUT_KIND (=KIND), INPUT, REFERENCE_FILE, REFERENCE, SHA256, SIZE, WIDTH, HEIGHT, FPS, FRAMES, LAST_FRAME,
DURATION_S, PREFIX, G, G_LABEL. Unknown placeholders stay visible for the lead to fill.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

from .video import IMAGE_EXTS, VIDEO_EXTS, probe, sha256_file

TEMPLATES = Path(__file__).resolve().parents[2] / "templates"
ROLES = ("measurement", "physics", "solver", "scene", "verify", "visual_review")
KINDS = ("ai-video", "footage", "image", "idea", "none")

BUILTIN_CONTRACT = """# {{TITLE}} — case contract (v1, lead only)

Case: `{{CASE}}` · input kind: **{{INPUT_KIND}}** · created {{DATE}}
Reference: `reference/{{REFERENCE_FILE}}` · sha256 `{{SHA256}}` · {{SIZE}} · {{FPS}} fps · {{FRAMES}} frames

## Goal
Keep the scene, replace the reference's wrong physics with a solved, benchmarked model; frame n <-> t = n / fps.

## Conventions
SI units, world Z up, gravity -Z (assumption unless measured). Pixel centres at +0.5. Every quantity is labelled
measured / assumed / fitted / model-conditioned. Generated footage is a visual reference, not a measurement.

## Roles and interfaces
measurement/ physics/ solver/ scene/ verify/ visual_review/ — each writes PROGRESS.log and visuals/.
Checkpoints: `v2p.py checkpoint CASE ckNNN` from config/build.json (immutable).
"""

BUILTIN_ACCEPTANCE = {
    "version": 1, "declared_utc": "{{DATE}}",
    "declared_before": "any measurement result, fit or render existed for this case",
    "note": "Tolerances are never relaxed after a failure; case-specific checks may be added before fitting.",
    "physics_validation": {
        "convergence": ">= 3 resolutions; observed order within 0.5 of theory",
        "independent_benchmark": "independently written verifier (no shared code) agrees with production",
        "analytic_limit": "closed-form result reproduced in its regime of validity",
        "energy": "conservative drift / dissipation balance within declared tolerance",
        "negative_controls": "deliberately wrong physics fails the checks"},
    "reference_fidelity": {"note": "reported separately; the reference is not ground truth"},
    "visual_review": {"side_by_side": "identical physical times, no retiming"},
}


def _fill(text: str, values: dict, as_json=False) -> str:
    for k, v in values.items():
        v = json.dumps(str(v))[1:-1] if as_json else str(v)     # keep JSON templates valid
        text = text.replace("{{" + k + "}}", v).replace(f"__{k}__", v)
    return text


def prefix_for(name: str) -> str:
    """Object-name prefix for Blender: CamelCase of the case name without a leading number, e.g. 'RulerBuckling'."""
    words = [w for w in re.split(r"[^A-Za-z0-9]+", name) if w and not w.isdigit()]
    return ("".join(w[:1].upper() + w[1:] for w in words) or "Case")[:20]


def _template(name: str, fallback: str) -> str:
    p = TEMPLATES / name
    return p.read_text() if p.exists() else fallback


def detect_kind(inp: str | None) -> str:
    if not inp:
        return "none"
    p = Path(inp)
    if p.is_file():
        ext = p.suffix.lower()
        if ext in VIDEO_EXTS:
            return "ai-video"
        if ext in IMAGE_EXTS:
            return "image"
        return "idea"
    return "idea"


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "case"


def new_case(case, inp=None, title=None, kind=None) -> dict:
    """Scaffold CASE. inp: path to a video/image/text file, or an idea as text. Returns case.json content."""
    case = Path(case)
    if case.exists() and any(case.iterdir()):
        raise FileExistsError(f"{case} exists and is not empty")
    kind = kind or detect_kind(inp)
    if kind not in KINDS:
        raise ValueError(f"--kind must be one of {KINDS}")
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    title = title or case.name.replace("-", " ").replace("_", " ").strip().title()

    for d in ROLES:
        (case / d / "visuals").mkdir(parents=True, exist_ok=True)
        (case / d / "PROGRESS.log").write_text(f"{now} | scaffolded | 0% | not started\n")
    for d in ("config", "reference", "review/visuals", "checkpoints"):
        (case / d).mkdir(parents=True, exist_ok=True)

    ref = {"kind": kind, "path": None, "sha256": None}
    src = Path(inp) if inp and Path(inp).is_file() else None
    if kind in ("ai-video", "footage", "image"):
        if src is None:
            raise FileNotFoundError(f"--kind {kind} needs an existing file, got {inp!r}")
        name = ("reference" if kind != "image" else "image") + src.suffix.lower()
        dst = case / "reference" / name
        shutil.copy2(src, dst)
        dst.chmod(0o444)
        ref.update(path=f"reference/{name}", source=str(src.resolve()), sha256=sha256_file(dst))
        if kind != "image":
            info = probe(dst)
            ref.update({k: info[k] for k in ("fps", "frames", "width", "height", "duration_s")})
        else:
            import cv2
            im = cv2.imread(str(dst))
            if im is not None:
                ref.update(width=im.shape[1], height=im.shape[0])
            if src.with_suffix(".txt").exists():
                shutil.copy2(src.with_suffix(".txt"), case / "reference" / "description.txt")
    elif kind == "idea":
        text = src.read_text() if src else (inp or "")
        (case / "reference" / "idea.md").write_text(f"# Idea\n\n{text.strip()}\n")
        ref.update(path="reference/idea.md", sha256=sha256_file(case / "reference" / "idea.md"))
        if src:
            ref["source"] = str(src.resolve())
    if ref.get("sha256"):
        (case / "reference" / "SHA256").write_text(f"{ref['sha256']}  {Path(ref['path']).name}\n")
    (case / "reference" / "meta.json").write_text(json.dumps(ref, indent=2))

    size = f"{ref['width']}x{ref['height']}" if ref.get("width") else "n/a"
    frames = ref.get("frames")
    values = {"TITLE": title, "CASE": case.name, "KIND": kind, "INPUT_KIND": kind, "INPUT": inp or "",
              "REFERENCE": ref.get("path") or "none", "REFERENCE_FILE": Path(ref["path"]).name if ref.get("path") else "none",
              "SHA256": ref.get("sha256") or "n/a", "FPS": ref.get("fps", "n/a"), "FRAMES": frames or "n/a",
              "LAST_FRAME": frames - 1 if frames else "n/a", "SIZE": size, "WIDTH": ref.get("width", "n/a"),
              "HEIGHT": ref.get("height", "n/a"), "DURATION_S": ref.get("duration_s", "n/a"), "NOW": now, "DATE": now,
              "PREFIX": prefix_for(case.name), "G": "9.80665", "G_LABEL": "assumed standard gravity"}
    (case / "CONTRACT.md").write_text(_fill(_template("CONTRACT.md", BUILTIN_CONTRACT), values))
    acc = _fill(_template("acceptance.json", json.dumps(BUILTIN_ACCEPTANCE, indent=2)), values, as_json=True)
    acc_obj = json.loads(acc)
    if isinstance(acc_obj, dict):
        acc_obj.setdefault("declared_utc", now)
    (case / "config" / "acceptance.json").write_text(json.dumps(acc_obj, indent=2) + "\n")
    page = _fill(_template("page.json", json.dumps({"title": title, "status": "building", "summary_md": "",
                                                    "items": []}, indent=2)), values, as_json=True)
    (case / "review" / "page.json").write_text(page)
    if (TEMPLATES / "REPORT.md").exists():
        (case / "review" / "REPORT.md").write_text(_fill((TEMPLATES / "REPORT.md").read_text(), values))

    case_cfg = {"title": title, "case": case.name, "kind": kind, "input": inp or None, "created_utc": now,
                "reference": ref, "fps": ref.get("fps"), "frames": ref.get("frames"),
                "size": [ref["width"], ref["height"]] if ref.get("width") else None,
                "world": {"units": "SI", "up": "+Z", "gravity_m_s2": [0, 0, -9.80665], "status": "assumed"},
                "generation": {"endpoint": None, "settings": {}}}
    (case / "config" / "case.json").write_text(json.dumps(case_cfg, indent=2) + "\n")
    (case / "config" / "build.json").write_text(json.dumps({"commands": [], "freeze": ["config"], "outputs": []},
                                                           indent=2) + "\n")
    (case / "config" / "CHANGELOG.md").write_text(f"# Changelog (lead only)\n\n## {now} — kickoff\n"
                                                  f"Case scaffolded from {kind} input.\n")
    (case / "STATUS.json").write_text(json.dumps({"case": case.name, "phase": "kickoff", "kind": kind,
                                                  "reference_sha256": ref.get("sha256"), "checkpoint": None,
                                                  "started_utc": now}, indent=2) + "\n")
    return case_cfg
