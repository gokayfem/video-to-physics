# AGENTS.md

Guidance for coding agents (Claude Code, Codex, Cursor, Copilot, Gemini) working on this repository. To use the
skill rather than develop it, see the README.

## Layout

```text
skills/video-to-physics/        the skill (this is what gets installed)
  SKILL.md                      entry point: goal, input routing, phases, roles, hard rules, reference routing
  references/                   guides: workflow, inputs, physics routing, measurement, solvers, verification,
                                Blender integration, review and delivery, agents and guardrails, lessons
  templates/                    CONTRACT.md, acceptance.json, build.json, page.json, release.json, REPORT.md,
                                roles/ (paste-in briefs for the specialist roles)
  scripts/v2p.py                CLI: new | probe | generate | screen | stabilize | rectify | sbs | sheet | page |
                                checkpoint | watchdog | lock | release
  scripts/v2p/                  importable library (plain Python: numpy, opencv, Pillow; scipy optional)
    video, camera, stabilize, rectify, compose, review_page, watchdog, fal_gen, checkpoint, release, case
    checks, elastica, dye, ode     physics and verification
  scripts/blender/              run inside Blender 5.2+: render_preview, camera_from_json, bake_pc2, dye_sheet,
                                check_bake
examples/                       five worked cases with sources, results, images and video links
tests/                          synthetic-data tests: python3 tests/run_tests.py
gallery/                        README images
adapters/                       copy-in prompts for tools without skill support
.claude-plugin/                 Claude Code plugin + marketplace manifests
```

## Commands

```bash
python3 tests/run_tests.py                                   # all tests (Blender tests skip without Blender)
python3 skills/video-to-physics/scripts/v2p.py --help        # every command has --help
```

Blender 5.2+ runs as `blender -b --factory-startup --python <script> -- <args>`; renders go through
`v2p.py lock -- ...` so only one render runs at a time.

## Rules for changes

- Input-agnostic: no hardcoded clips, paths, scenes or phenomena in `skills/`. Case-specific code lives in
  `examples/<case>/`.
- Every new module gets a synthetic-data test in `tests/`; physics code is tested against a closed form or a
  published value with a stated tolerance.
- Keep the data contracts stable (camera json, `deformation.npz`, tracer PNG + `meta.json`); the references and the
  Blender scripts depend on them.
- No secrets in files or logs. `FAL_KEY` is read from the environment and never printed.
