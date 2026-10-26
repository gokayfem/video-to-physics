"""Run-hygiene watchdog and the global render lock.

Watchdog (read-only, never kills anything) flags, under a ROOT directory of cases:
  * hand-written wait loops (`until ...; sleep` / `while ...; sleep`) running longer than loop_min without
    BOTH a deadline and a failure check (bounded ones are only noted);
  * long jobs (blender / python whose command line mentions ROOT) running longer than job_min;
  * a render lock held longer than job_min (or whose owner pid is gone);
  * idle roles: a role dir with a PROGRESS.log whose newest file is older than idle_min and whose last
    heartbeat does not say it finished (or 'not started', as scaffolded by `new`). Cases whose STATUS.json says complete/verified are skipped.
The watchdog never matches its own process or its ancestors (it is often launched from a shell loop).

Render lock: `with render_lock(path)` serialises heavy Blender work (one render at a time) via an atomic
mkdir; the owner file records pid + command; a lock whose owner pid is dead is reclaimed.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

WAIT_RE = re.compile(r"until\b.*\b(grep|sleep|test|\[)|while\b.*\bsleep")
BOUND_RE = (re.compile(r"date \+%s|SECONDS|timeout\b|deadline", re.I), re.compile(r"-ge|-gt|-lt|-le|\(\(.*[<>].*\)\)"),
            re.compile(r"Error|FAIL|exited|exit code|\$\?|Traceback", re.I))
FINISHED_RE = re.compile(r"100\s*%|complete|delivered|\bdone\b|finished", re.I)
FAILED_RE = re.compile(r"exited with code [1-9]|Error|Traceback")
DONE_STATUS_RE = re.compile(r'"(phase|status)"\s*:\s*"(complete|completed|verified|done|released)"', re.I)


def default_lock_path() -> Path:
    """Global render lock: $V2P_RENDER_LOCK or ~/.cache/video-to-physics/render.lock"""
    env = os.environ.get("V2P_RENDER_LOCK")
    return Path(env) if env else Path.home() / ".cache" / "video-to-physics" / "render.lock"


def etime_minutes(s: str) -> float:
    """ps etime '[[dd-]hh:]mm:ss' -> minutes."""
    days, rest = (s.split("-", 1) + [""])[:2] if "-" in s else ("0", s)
    parts = [int(x) for x in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(days) * 1440 + parts[0] * 60 + parts[1] + parts[2] / 60


def ps_snapshot() -> list[tuple[int, int, float, str]]:
    """[(pid, ppid, elapsed_min, command)] for every process (POSIX `ps`)."""
    out = subprocess.run(["ps", "-axo", "pid=,ppid=,etime=,command="], capture_output=True, text=True).stdout
    return parse_ps(out)


def parse_ps(text: str):
    rows = []
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s+(\d+)\s+(\S+)\s+(.*)", line)
        if m:
            rows.append((int(m.group(1)), int(m.group(2)), etime_minutes(m.group(3)), m.group(4)))
    return rows


def ancestors(pid: int, procs) -> set[int]:
    """pid plus its parent chain (so the watchdog never flags the loop that launched it)."""
    parent = {p: pp for p, pp, _, _ in procs}
    out, cur = set(), pid
    while cur and cur not in out:
        out.add(cur)
        cur = parent.get(cur, 0)
    return out


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def lock_owner(lock: Path) -> dict:
    try:
        return json.loads((lock / "owner.json").read_text())
    except Exception:
        return {}


def scan(root, loop_min=12.0, job_min=45.0, idle_min=25.0, procs=None, own_pid=None, lock=None, now=None):
    """Return (alerts, notes) lists of strings. `procs`/`own_pid`/`now` are injectable for tests."""
    root_abs = Path(root).absolute()
    root = root_abs.resolve()
    procs = ps_snapshot() if procs is None else procs
    own = ancestors(os.getpid() if own_pid is None else own_pid, procs)
    now = time.time() if now is None else now
    alerts, notes = [], []

    for pid, _, et, cmd in procs:
        if pid in own or cmd.startswith("ps -axo"):
            continue
        if WAIT_RE.search(cmd) and et > loop_min:
            if all(r.search(cmd) for r in BOUND_RE):
                notes.append(f"bounded wait pid {pid} running {et:.0f} min (has deadline + failure check)")
                continue
            info = ""
            target = re.search(r"grep -q\w*\s+(?:'[^']*'|\"[^\"]*\"|\S+)\s+(\S+)", cmd)
            if target and Path(target.group(1).strip("'\"")).is_file():
                tail = Path(target.group(1).strip("'\"")).read_text(errors="ignore")[-400:].strip()
                info = f" | waited-on file tail: {tail[-120:]!r}"
                if FAILED_RE.search(tail):
                    info += " | LIKELY STALE (target command failed)"
            alerts.append(f"ALERT wait-loop pid {pid} running {et:.0f} min without deadline+failure check{info}")
        elif re.search(r"\bblender\b|python", cmd, re.I) and (str(root) in cmd or str(root_abs) in cmd) and et > job_min:
            alerts.append(f"ALERT long job pid {pid} {et:.0f} min: {cmd[:160]}")

    lock = Path(lock) if lock else default_lock_path()
    if lock.exists():
        age = (now - lock.stat().st_mtime) / 60
        owner = lock_owner(lock)
        desc = f"pid {owner.get('pid', '?')}: {str(owner.get('cmd', '?'))[:120]}"
        if owner.get("pid") and not _pid_alive(int(owner["pid"])):
            alerts.append(f"ALERT render lock {lock} held by dead pid ({desc}); it will be reclaimed on next use")
        elif age > job_min:
            alerts.append(f"ALERT render lock held {age:.0f} min by {desc}")
        else:
            notes.append(f"render lock held {age:.0f} min by {desc}")

    for case in _cases(root):
        status = case / "STATUS.json"
        if status.exists() and DONE_STATUS_RE.search(status.read_text(errors="ignore")):
            continue
        for hb in sorted(case.glob("*/PROGRESS.log")) + sorted(case.glob("*/*/PROGRESS.log")):
            role = hb.parent
            newest = max((f.stat().st_mtime for f in role.rglob("*") if f.is_file() and "cache" not in f.parts),
                         default=hb.stat().st_mtime)
            idle = (now - newest) / 60
            lines = hb.read_text(errors="ignore").strip().splitlines()
            last = lines[-1][:120] if lines else "empty heartbeat"
            name = f"{role.relative_to(root) if role != root else role.name}"
            line = f"{name}: idle {idle:.0f} min | {last}"
            if idle > idle_min and not FINISHED_RE.search(last) and "not started" not in last:
                alerts.append("ALERT stale role " + line)
            else:
                notes.append(line)
    return alerts, notes


def _cases(root: Path):
    """ROOT itself if it is a case, else every direct/second-level subdir with STATUS.json or CONTRACT.md."""
    marks = ("STATUS.json", "CONTRACT.md")
    if any((root / m).exists() for m in marks):
        return [root]
    found = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if any((d / m).exists() for m in marks):
            found.append(d)
        else:
            found += [s for s in sorted(d.iterdir()) if s.is_dir() and any((s / m).exists() for m in marks)]
    return found


def report(alerts, notes) -> str:
    return "\n".join(alerts + ["  " + n for n in notes] + [f"watchdog: {len(alerts)} alert(s)"])


# ------------------------------------------------------------------ render lock
@contextlib.contextmanager
def render_lock(lock=None, cmd="", timeout_min=60.0, poll_s=5.0, log=print):
    """Hold the global render lock while the block runs. Bounded wait: raises TimeoutError after timeout_min."""
    lock = Path(lock) if lock else default_lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout_min * 60
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            owner = lock_owner(lock)
            if owner.get("pid") and not _pid_alive(int(owner["pid"])):
                log(f"[render-lock] reclaiming lock of dead pid {owner['pid']}")
                shutil.rmtree(lock, ignore_errors=True)
                continue
            if time.time() > deadline:
                raise TimeoutError(f"render lock {lock} still held after {timeout_min} min by {owner}")
            log(f"[render-lock] waiting (held by pid {owner.get('pid', '?')}: {str(owner.get('cmd', ''))[:80]})")
            time.sleep(poll_s)
    try:
        (lock / "owner.json").write_text(json.dumps({"pid": os.getpid(), "cmd": cmd,
                                                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
        yield lock
    finally:
        shutil.rmtree(lock, ignore_errors=True)


def run_locked(argv, lock=None, timeout_min=60.0) -> int:
    """Run argv under the render lock; returns its exit code."""
    with render_lock(lock, " ".join(argv), timeout_min):
        return subprocess.call(argv)
