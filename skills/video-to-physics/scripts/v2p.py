#!/usr/bin/env python3
"""Video 2 Physics command line.

    python3 skills/video-to-physics/scripts/v2p.py <command> [args]     (python3 v2p.py <command> -h for details)

Every command takes explicit paths; nothing is tied to a particular clip or phenomenon. Modules are imported
only by the command that needs them, so `-h` works with the standard library alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def _print(obj):
    print(json.dumps(obj, indent=2, default=str))


# ------------------------------------------------------------------ command implementations (lazy imports)
def cmd_new(a):
    from v2p.case import new_case
    cfg = new_case(a.case, a.input, a.title, a.kind)
    print(f"scaffolded {a.case} ({cfg['kind']}; reference sha256 {cfg['reference'].get('sha256')})")


def cmd_probe(a):
    from v2p.video import probe
    _print(probe(a.video))


def cmd_generate(a):
    from v2p.fal_gen import generate
    prompt = Path(a.prompt_file).read_text().strip() if a.prompt_file else a.prompt
    if not prompt:
        sys.exit("generate: --prompt or --prompt-file is required")
    settings = dict(p.split("=", 1) for p in a.param)
    for k, v in settings.items():           # numbers/booleans/json values in --param k=v
        try:
            settings[k] = json.loads(v)
        except json.JSONDecodeError:
            pass
    plan = generate(a.case, a.first_frame, prompt, n=a.n, seed=a.seed, endpoint=a.endpoint, duration=a.duration,
                    round_name=a.round, settings=settings, dry_run=a.dry_run, timeout_s=a.timeout_min * 60)
    print(f"{plan['n']} candidate(s), endpoint {plan['endpoint']}, seeds {plan['seeds']}"
          + (" (dry run, nothing submitted)" if a.dry_run else ""))


def cmd_screen(a):
    from v2p.fal_gen import screen
    screen(a.case, a.round, a.first_frame)


def cmd_stabilize(a):
    from v2p.stabilize import stabilize
    _print(stabilize(a.video, a.out, mask=a.mask, scale=a.scale, motion=a.motion, auto_mask=a.auto_mask))


def cmd_rectify(a):
    import numpy as np
    from v2p.camera import Camera
    from v2p.rectify import parse_plane, parse_refract, rectify_video
    ext = _floats(a.extent)
    if len(ext) != 4:
        sys.exit("--extent needs x0,x1,y0,y1")
    Hs = np.load(a.stab) if a.stab else None
    _print(rectify_video(a.video, Camera.from_json(a.camera), parse_plane(a.plane), ext, a.px / 1000.0, a.out,
                         refract=parse_refract(a.refract), homographies=Hs, max_frames=a.frames))


def cmd_sbs(a):
    from v2p.compose import sbs
    _print(sbs(a.ref, a.ours, a.out, label_ours=a.label, label_ref=a.ref_label, frames=a.frames, panel_w=a.panel_w,
               fps=a.fps))


def cmd_sheet(a):
    from v2p.compose import sheet
    _print(sheet(a.video, a.out, _floats(a.times) if a.times else None, a.label, cols=a.cols))


def cmd_page(a):
    from v2p.review_page import build
    _print(build(a.case))


def cmd_checkpoint(a):
    from v2p import checkpoint
    if a.verify:
        problems = checkpoint.verify(Path(a.case) / "checkpoints" / a.name)
        print("\n".join(problems) or "intact")
        sys.exit(1 if problems else 0)
    _print(checkpoint.create(a.case, a.name))


def cmd_watchdog(a):
    from v2p.watchdog import report, scan
    alerts, notes = scan(a.root, a.loop_min, a.job_min, a.idle_min, lock=a.lock)
    print(report(alerts, notes))
    sys.exit(1 if alerts else 0)


def cmd_lock(a):
    from v2p.watchdog import run_locked
    argv = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not argv:
        sys.exit("lock: give a command after --")
    sys.exit(run_locked(argv, a.lock, a.timeout_min))


def cmd_release(a):
    from v2p.release import render, thumbnail
    _print(render(a.config, a.out, draft=a.draft))
    if a.thumbnail:
        thumbnail(a.config, a.thumbnail, draft=a.draft)


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="v2p.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command")

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_, description=help_)
        p.set_defaults(fn=fn)
        return p

    p = add("new", cmd_new, "scaffold a case dir (contract, config, reference copy + sha256, role dirs)")
    p.add_argument("case", help="case directory to create")
    p.add_argument("--input", help="video, image, text file, or the idea as text (omit for --kind none)")
    p.add_argument("--title")
    p.add_argument("--kind", choices=["ai-video", "footage", "image", "idea", "none"],
                   help="default: detected from --input (video -> ai-video)")

    p = add("probe", cmd_probe, "fps, frame count, size, duration and sha256 of a video (json)")
    p.add_argument("video")

    p = add("generate", cmd_generate, "fal image-to-video candidates (request ids saved before polling; "
                                      "never resubmits an uncertain job; FAL_KEY from env)")
    p.add_argument("case")
    p.add_argument("--first-frame", required=True, help="PNG/JPG first frame")
    p.add_argument("--prompt")
    p.add_argument("--prompt-file")
    p.add_argument("--n", type=int, default=3, help="number of candidates (default 3)")
    p.add_argument("--seed", type=int, help="first seed (candidate k uses seed+k); default random, recorded")
    p.add_argument("--endpoint", help="fal endpoint id; default config/case.json generation.endpoint")
    p.add_argument("--duration", type=float, default=10, help="clip seconds (default 10)")
    p.add_argument("--param", action="append", default=[], metavar="K=V", help="extra endpoint input (repeatable)")
    p.add_argument("--round", default="", help="sub-directory of generation/ for this round")
    p.add_argument("--timeout-min", type=float, default=30, help="poll deadline per candidate (default 30)")
    p.add_argument("--dry-run", action="store_true", help="write request files only; no network, no key")

    p = add("screen", cmd_screen, "screen candidates: frame-0 diff, camera drift, motion timeline -> screen.json + sheet")
    p.add_argument("case")
    p.add_argument("--round", default="")
    p.add_argument("--first-frame", help="default generation/<round>/first_frame.*")

    p = add("stabilize", cmd_stabilize, "per-frame homography frame0->n on static regions + drift plot + json")
    p.add_argument("video", help="video file or image directory")
    p.add_argument("--out", required=True)
    p.add_argument("--mask", help="PNG, white = static region to align on")
    p.add_argument("--auto-mask", action="store_true", help="static mask from low temporal variance")
    p.add_argument("--scale", type=float, default=0.5, help="working scale for ECC (default 0.5)")
    p.add_argument("--motion", default="homography", choices=["translation", "euclidean", "affine", "homography"])

    p = add("rectify", cmd_rectify, "top-down rectification of a world plane (optional flat-interface refraction)")
    p.add_argument("video", help="video file or image directory")
    p.add_argument("--camera", required=True, help="camera json (data contract)")
    p.add_argument("--plane", required=True, help="z=H (metres)")
    p.add_argument("--refract", help="n=1.333,z=Z : flat interface at Z with index n above the plane")
    p.add_argument("--extent", required=True, help="x0,x1,y0,y1 in metres")
    p.add_argument("--px", type=float, required=True, help="output pixel size in millimetres")
    p.add_argument("--stab", help="H_frame0_to_n.npy from `stabilize` to follow camera motion")
    p.add_argument("--frames", type=int, help="limit number of frames")
    p.add_argument("--out", required=True)

    p = add("sbs", cmd_sbs, "reference | ours | 50/50 blend MP4 at true fps, burned labels, no retiming")
    p.add_argument("ref")
    p.add_argument("ours", help="mp4 or directory of PNG frames")
    p.add_argument("out")
    p.add_argument("--label", default="ours (physics)", help="label of the OURS panel")
    p.add_argument("--ref-label", default="reference")
    p.add_argument("--frames", type=int, help="default: min of both lengths")
    p.add_argument("--panel-w", type=int, default=960)
    p.add_argument("--fps", type=float, help="default: the reference's fps")

    p = add("sheet", cmd_sheet, "labelled contact sheet")
    p.add_argument("video")
    p.add_argument("out")
    p.add_argument("--times", help="comma-separated seconds (default 8 evenly spaced frames)")
    p.add_argument("--label")
    p.add_argument("--cols", type=int, default=4)

    p = add("page", cmd_page, "playable review page review/index.html from review/page.json")
    p.add_argument("case")

    p = add("checkpoint", cmd_checkpoint, "build an immutable checkpoint from config/build.json "
                                          "(frozen inputs, outputs, MANIFEST.sha256, READY)")
    p.add_argument("case")
    p.add_argument("name")
    p.add_argument("--verify", action="store_true", help="re-hash an existing checkpoint instead")

    p = add("watchdog", cmd_watchdog, "flag stale wait loops, long jobs, stuck render lock, idle roles (exit 1 on alert)")
    p.add_argument("root", help="a case dir or a dir of cases")
    p.add_argument("--loop-min", type=float, default=12)
    p.add_argument("--job-min", type=float, default=45)
    p.add_argument("--idle-min", type=float, default=25)
    p.add_argument("--lock", help="render lock path (default $V2P_RENDER_LOCK or ~/.cache/video-to-physics/render.lock)")

    p = add("lock", cmd_lock, "run a command under the global render lock (one Blender render at a time)")
    p.add_argument("--lock", help="lock path (default as for watchdog)")
    p.add_argument("--timeout-min", type=float, default=60, help="give up waiting after this long (default 60)")
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="-- CMD ARGS...")

    p = add("release", cmd_release, "release film from a JSON segment config (frame 0 = title-over-hero thumbnail)")
    p.add_argument("config")
    p.add_argument("out")
    p.add_argument("--draft", action="store_true", help="half resolution, fast encode")
    p.add_argument("--thumbnail", help="also write frame 0 as this PNG")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        a.fn(a)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, TimeoutError) as e:
        sys.exit(f"v2p {a.cmd}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
