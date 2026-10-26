"""Watchdog: stale PROGRESS.log flagged, own process (and its parent loop) ignored, bounded waits noted."""
import os
import time

from synth import tmpdir
from v2p.watchdog import etime_minutes, parse_ps, render_lock, scan


def _case(root, name):
    c = root / name
    (c / "solver").mkdir(parents=True)
    (c / "STATUS.json").write_text('{"phase": "building"}')
    return c


def test_etime():
    assert etime_minutes("05:30") == 5.5 and etime_minutes("1-02:00:00") == 1440 + 120


def test_stale_role_and_own_process():
    root = tmpdir()
    c = _case(root, "case-a")
    hb = c / "solver" / "PROGRESS.log"
    hb.write_text("2026-01-01T00:00Z | fitting | 40% | -\n")
    old = time.time() - 3600
    os.utime(hb, (old, old))
    fresh = _case(root, "case-b")
    (fresh / "solver" / "PROGRESS.log").write_text("x | running | 10% | -\n")
    done = _case(root, "case-c")
    (done / "solver" / "PROGRESS.log").write_text("x | delivered | 100% | -\n")
    os.utime(done / "solver" / "PROGRESS.log", (old, old))

    me, parent = 5000, 4000
    ps = (f"{parent} 1 30:00 zsh -c until grep -q DONE log; do sleep 5; done\n"
          f"{me} {parent} 30:00 python3 v2p.py watchdog {root} until x; do sleep 1; done\n"
          f"7777 1 40:00 bash -c until grep -q DONE /nonexistent/log; do sleep 5; done\n"
          f"8888 1 40:00 bash -c 'end=$(($(date +%s)+600)); until grep -q DONE l || [ $(date +%s) -ge $end ]; "
          f"do grep -q Error l && exit 1; sleep 5; done'\n"
          f"9999 1 50:00 blender -b {root}/case-a/scene/x.blend --python r.py\n")
    alerts, notes = scan(root, loop_min=12, job_min=45, idle_min=25, procs=parse_ps(ps), own_pid=me,
                         lock=root / "no.lock")
    text = "\n".join(alerts)
    assert any("stale role case-a/solver" in a for a in alerts), alerts
    assert "case-b" not in text and "case-c" not in text
    assert "pid 7777" in text and "pid 9999" in text
    assert f"pid {me}" not in text and f"pid {parent}" not in text      # self and launching loop ignored
    assert any("bounded wait pid 8888" in n for n in notes), notes


def test_real_ps_does_not_flag_self():
    root = tmpdir()
    alerts, _ = scan(root, loop_min=0, job_min=0, idle_min=0, lock=root / "no.lock")
    assert not any(f"pid {os.getpid()} " in a for a in alerts)


def test_render_lock_exclusive_and_reclaim():
    root = tmpdir()
    lock = root / "render.lock"
    with render_lock(lock, "job"):
        assert (lock / "owner.json").exists()
        try:
            with render_lock(lock, "second", timeout_min=0.0005, poll_s=0.01, log=lambda *_: None):
                raise AssertionError("lock was not exclusive")
        except TimeoutError:
            pass
    assert not lock.exists()
    lock.mkdir()
    (lock / "owner.json").write_text('{"pid": 999999999, "cmd": "dead"}')   # dead owner is reclaimed
    with render_lock(lock, "job", timeout_min=0.01, log=lambda *_: None):
        pass
