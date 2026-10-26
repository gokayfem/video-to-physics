#!/usr/bin/env python3
"""Minimal test runner (no pytest): imports every tests/test_*.py and runs its test_* functions.

usage: python3 tests/run_tests.py [-k SUBSTRING] [FILE ...]
A test passes if it returns, fails on any exception, and is skipped on unittest.SkipTest.
skills/video-to-physics/scripts is put on sys.path so tests can `import v2p...`.
"""
import argparse
import importlib.util
import sys
import time
import traceback
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "skills" / "video-to-physics" / "scripts"))
sys.path.insert(0, str(HERE))


def missing_dependencies():
    missing = []
    for mod, pkg in (("numpy", "numpy"), ("cv2", "opencv-python"), ("PIL", "Pillow")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    return missing


def main():
    missing = missing_dependencies()
    if missing:
        print(f"missing dependencies: {', '.join(missing)}\n  pip install -r requirements.txt", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="", help="only run tests whose name contains this")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    files = [Path(f).resolve() for f in a.files] or sorted(HERE.glob("test_*.py"))
    passed, failed, skipped = 0, [], 0
    for f in files:
        spec = importlib.util.spec_from_file_location(f.stem, f)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except unittest.SkipTest as e:
            print(f"SKIP {f.name}: {e}")
            skipped += 1
            continue
        except Exception:
            failed.append(f.name)
            print(f"FAIL {f.name} (import)\n{traceback.format_exc()}")
            continue
        for name in sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))):
            if a.k not in f"{f.stem}.{name}":
                continue
            t0 = time.time()
            try:
                getattr(mod, name)()
                passed += 1
                print(f"PASS {f.stem}.{name} ({time.time() - t0:.2f}s)")
            except unittest.SkipTest as e:
                skipped += 1
                print(f"SKIP {f.stem}.{name}: {e}")
            except Exception:
                failed.append(f"{f.stem}.{name}")
                print(f"FAIL {f.stem}.{name}\n{traceback.format_exc()}")
    print(f"\n{passed} passed, {len(failed)} failed, {skipped} skipped")
    if failed:
        print("failed: " + ", ".join(failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    sys.exit(main())
