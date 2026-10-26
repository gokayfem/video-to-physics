# Review and Delivery

Visual review checks what the numbers cannot: does the render look like the scene, do props behave, does anything
pop, slide or flicker, and does a clean rebuild give the same thing. Delivery packages the accepted checkpoint for
a human decision.

## Contents

- [Side-by-side and sheets](#side-by-side-and-sheets)
- [Visual review checklist](#visual-review-checklist)
- [Reproducibility review](#reproducibility-review)
- [Findings format](#findings-format)
- [Final render](#final-render)
- [Report structure](#report-structure)
- [Review page](#review-page)
- [Release film](#release-film)
- [Approval](#approval)

## Side-by-side and sheets

- `v2p.py sbs REF OURS OUT --label "ck002 | preview"` writes reference / ours / 50-50 blend at the true fps with
  burned labels. OURS can be an mp4 or a PNG directory. Nothing is retimed: frame n of both is t = n/fps.
- `v2p.py sheet VIDEO OUT --times 0,2.5,5,7.5,9.9 --label "..."` gives a labelled contact sheet; use the same times
  for reference and render.
- Label every image or video: version | frame or time | status.

## Visual review checklist

| Item | Look for |
|---|---|
| Static scene | silhouettes overlap (IoU ≥ 0.9 against the reference for static parts); no shake from per-frame camera noise |
| Motion | no popping, sliding attachments, shape changes, interpenetration or flicker in dense strips at turning points and contacts |
| Props and logistics | every moving prop has a cause; no teleports or appearances; counts consistent |
| Parts | every visible part present and readable; thin hardware widths as measured |
| Materials | region metrics and 2-D luma profiles; backdrop gradient direction; metal highlight bands |
| Motion blur | shutter as measured; none if the reference shows none |
| Fluids | dye sharpness, colour balance, fronts, film grain moving per frame |
| Honesty | divergences from the reference are physics corrections or declared deviations, not unexplained differences |
| Frame 0 | our frame 0 vs the reference frame 0 (and vs our first frame for `idea`) |

## Reproducibility review

- Rebuild from the checkpoint alone (its frozen inputs, its commands) into a scratch directory; compare state
  dumps, caches and evaluated transforms byte for byte.
- Verify `MANIFEST.sha256` from inside the checkpoint directory.
- Look for absolute paths into the working tree; a checkpoint must not depend on files outside it.

## Findings format

`visual_review/<ck>_review.json`:

```json
{"checkpoint": "ck002", "findings": [
  {"id": "F1", "severity": "major", "t_s": 0.0, "what": "dye ring at cylinder base in frame 0 not in solved flow",
   "evidence": "visual_review/visuals/ck002_f0_ring.png", "proposal": "declare as frame-0 deviation"}],
 "blockers": 0, "reproducible": true}
```

Severities: **blocker** (wrong physics visible, broken scene, not reproducible), **major** (clearly visible
mismatch or defect), **minor** (polish). The lead decides each major within one message: fix in a new checkpoint
or declare.

## Final render

- One full-resolution render per case, of the accepted checkpoint, after review fixes, under `v2p.py lock`.
- Same engine, samples and post as the last approved preview, at full resolution; per-frame grain seed.
- Stop a final render early if a defect appears; do not finish a render that will be replaced.
- Then the final side-by-side and contact sheet.

## Report structure

`REPORT.md` from `templates/REPORT.md`, about one page, visuals first:

1. Goal and status (checkpoint, audit result, awaiting approval). "Not claimed: real-world accuracy."
2. What the reference shows (measured): events, defects with numbers. "None of these behaviours were imitated."
3. Corrected physics: declared/measured inputs, assumed values, the model, what was anchored, the result.
4. Verification: solver checks, independent verifier numbers, negative controls, audit k/n PASS, what was not checked.
5. Scene and look: frame-0 comparison, our additions, declared deviations, scene checks, reproducibility.
6. Sensitive assumption and its effect.
7. Key files and the rebuild command.

## Review page

`review/page.json` (template in `templates/page.json`): `title`, `status`, `summary_md` (paragraphs separated by
blank lines), `items` (each with `path` relative to `review/`, `label`, `caption`). `v2p.py page CASE` writes
`review/index.html`. Order: final side-by-side, final render, solver plots, verifier audit, measurement and
divergence map, review findings.

## Release film

`v2p.py release CONFIG.json OUT.mp4 [--draft]` builds a film from segments (template: `templates/release.json`):
title over a hero split as frame 0, full clip, AI-vs-physics splits with caption pills, stat count-ups, card rows,
outro.

- **Frame 0 is a finished thumbnail**: title over the hero frame, fully composed. Never black, never a fade-in.
- Use only real outputs of the project; splits compare the reference and the render at the same t.
- Captions state measured facts ("AI: 3.5× time-compressed"), not claims of real-world accuracy.
- Render `--draft` first and look at a contact sheet of the film before the full render.

## Approval

Present the review page link, the final side-by-side and a short status. Stop and wait. Only the user's own message
in chat approves a case or starts the next one; record it in `config/CHANGELOG.md`.
