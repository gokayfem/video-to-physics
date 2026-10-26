"""Immutable checkpoints built from frozen inputs.

CASE/config/build.json (see templates/build.json):
    {"freeze":   ["params/p.json", {"from": "solver/out/x.npz", "to": "solver/out/x.npz"}, ...],
     "copy":     ["CONTRACT.md", "config", "scene", ...],          # source snapshot (exclude applies)
     "exclude":  ["__pycache__", "*.blend1", "scene/out", ...],     # glob on any path component or rel path
     "commands": ["python3 scene/build.py --out {ck}/scene.blend --data {ck}/solver/out/x.npz", ...],
     "outputs":  ["scene.blend", "checks/check.json"]}              # must exist in {ck} after the commands
Procedure for `checkpoint CASE NAME` (all paths relative to CASE, no '..'):
  1. refuse if CASE/checkpoints/NAME exists (checkpoints are never overwritten);
  2. copy `freeze` entries to {ck}/<to> FIRST and the `copy` snapshot next, so commands read frozen copies;
  3. run `commands` in order (shell, cwd = CASE). Placeholders {ck} (= {checkpoint}, {frozen}) and {case} are
     substituted and exported as V2P_CHECKPOINT / V2P_CASE. Output goes to {ck}/build.log. A failure writes
     FAILED and stops; READY is never written;
  4. every `outputs` path must exist in {ck} (an output that only exists in CASE is copied in);
  5. write checkpoint.json and MANIFEST.sha256 (paths relative to {ck}), make files read-only, write READY last.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def _safe_rel(rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"build.json paths must be relative to CASE without '..': {rel}")
    return p


def excluded(rel: str, patterns) -> bool:
    parts = Path(rel).parts
    for pat in patterns:
        if fnmatch.fnmatch(rel, pat) or rel == pat or rel.startswith(pat.rstrip("/") + "/"):
            return True
        if any(fnmatch.fnmatch(part, pat) for part in parts):
            return True
    return False


def _copy(case: Path, src_rel: str, dst: Path, exclude=()):
    src = case / _safe_rel(src_rel)
    if not src.exists():
        raise FileNotFoundError(f"declared path missing: {src}")
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    for f in sorted(src.rglob("*")):
        rel = f.relative_to(case).as_posix()
        if f.is_file() and not excluded(rel, exclude):
            out = dst / f.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)


def write_manifest(ck: Path) -> Path:
    files = sorted(p for p in ck.rglob("*") if p.is_file() and p.name not in ("MANIFEST.sha256", "READY"))
    out = ck / "MANIFEST.sha256"
    out.write_text("".join(f"{_sha(p)}  {p.relative_to(ck).as_posix()}\n" for p in files))
    return out


def verify(ck) -> list[str]:
    """Re-hash a checkpoint; returns a list of problems (empty = intact)."""
    ck = Path(ck)
    problems = [] if (ck / "READY").exists() else ["READY missing"]
    man = ck / "MANIFEST.sha256"
    if not man.exists():
        return problems + ["MANIFEST.sha256 missing"]
    listed = set()
    for line in man.read_text().splitlines():
        digest, rel = line.split("  ", 1)
        listed.add(rel)
        p = ck / rel
        if not p.exists():
            problems.append(f"missing {rel}")
        elif _sha(p) != digest:
            problems.append(f"changed {rel}")
    for p in ck.rglob("*"):
        rel = p.relative_to(ck).as_posix()
        if p.is_file() and p.name not in ("MANIFEST.sha256", "READY") and rel not in listed:
            problems.append(f"unlisted {rel}")
    return problems


def _readonly(ck: Path):
    for p in ck.rglob("*"):
        if p.is_file():
            p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _sub(s: str, subs: dict) -> str:
    for k, v in subs.items():          # plain replacement: other braces in shell commands stay untouched
        s = s.replace("{" + k + "}", v)
    return s


def create(case, name, log=print) -> dict:
    """Build checkpoint NAME for CASE (see module doc). Returns a summary dict; raises on failure."""
    case = Path(case).resolve()
    cfg_path = case / "config" / "build.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found (template: templates/build.json)")
    cfg = json.loads(cfg_path.read_text())
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"bad checkpoint name {name!r}")
    ck = case / "checkpoints" / name
    if ck.exists():
        raise FileExistsError(f"checkpoint {ck} already exists; checkpoints are immutable - choose a new name")
    ck.mkdir(parents=True)
    exclude = cfg.get("exclude", [])
    for entry in cfg.get("freeze", []):
        src, dst = (entry, entry) if isinstance(entry, str) else (entry["from"], entry.get("to", entry["from"]))
        _copy(case, src, ck / _safe_rel(dst))
    for rel in cfg.get("copy", []):
        _copy(case, rel, ck / _safe_rel(rel), exclude)

    subs = {"ck": str(ck), "checkpoint": str(ck), "frozen": str(ck), "case": str(case)}
    env = {**os.environ, "V2P_CASE": str(case), "V2P_CHECKPOINT": str(ck), "V2P_FROZEN": str(ck)}
    started = time.time()
    with open(ck / "build.log", "w") as logf:
        for k, cmd in enumerate(cfg.get("commands", [])):
            line = _sub(cmd, subs) if isinstance(cmd, str) else [_sub(c, subs) for c in cmd]
            logf.write(f"$ {line}\n")
            logf.flush()
            rc = subprocess.call(line, shell=isinstance(line, str), cwd=case, env=env, stdout=logf,
                                 stderr=subprocess.STDOUT)
            if rc != 0:
                (ck / "FAILED").write_text(f"command {k} exited with code {rc}: {line}\n")
                raise RuntimeError(f"checkpoint {name}: command {k} failed (exit {rc}); see {ck / 'build.log'}")
    missing = []
    for rel in cfg.get("outputs", []):
        if not (ck / _safe_rel(rel)).exists():
            if (case / _safe_rel(rel)).exists():
                _copy(case, rel, ck / _safe_rel(rel))
            else:
                missing.append(rel)
    if missing:
        (ck / "FAILED").write_text(f"declared outputs missing: {missing}\n")
        raise RuntimeError(f"checkpoint {name}: outputs missing after build: {missing}")
    info = {"name": name, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "build_s": round(time.time() - started, 2),
            **{k: cfg.get(k, []) for k in ("freeze", "copy", "exclude", "commands", "outputs")}}
    (ck / "checkpoint.json").write_text(json.dumps(info, indent=2))
    write_manifest(ck)
    _readonly(ck)
    (ck / "READY").write_text(info["utc"] + "\n")
    log(f"checkpoint {ck} READY")
    return {**info, "path": str(ck)}
