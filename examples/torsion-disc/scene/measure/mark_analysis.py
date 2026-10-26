"""Engraved-mark analysis on the reference (scene role C): is the mark a radius or a full diameter, is it straight or
kinked at the centre, and how wide is it (px)?

Per frame (every --step frames): disc silhouette (disc_silhouette.silhouette) -> ellipse fit to the UPPER silhouette
half (the top face's far rim; the near half includes the rim band) -> affine unwarp of the disc top face to a circle
(radius 1). In the unwarped disc, the dark line is found by a black-hat (luma) response sampled on rays from the
centre: for each of 720 ray angles the mean response over r in [0.15, 0.9] -> angular profile. The two strongest
peaks (>= 60 deg apart) give the two half-line angles; coverage = fraction of r samples along the peak ray above
threshold. Diameter <=> both halves (second half searched within +-40 deg of
the first + 180) have coverage >= 0.6 and peak >= 25 % of the first. Kink = 180 deg - (angle between the halves).
The turned-brass streaks are static in image space, so the per-(angle, r) median over all analysed frames is
subtracted before peak finding. Writes scene/measure/mark_analysis.json and mark_analysis.png.
.venv/bin/python scene/measure/mark_analysis.py [--step 4]
"""
import argparse, json, pathlib, sys
import numpy as np, cv2

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from disc_silhouette import silhouette  # noqa: E402


def analyse(n):
    f, pts = silhouette(n)
    cy = np.median(pts[:, 1])
    up = pts[pts[:, 1] < cy + 5]
    (ex, ey), (MA, ma), ang = cv2.fitEllipse(up.astype(np.float32))
    a, b = max(MA, ma) / 2, min(MA, ma) / 2
    th = np.radians(ang if MA >= ma else ang + 90)            # major-axis direction
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # affine unwarp of the top face to a circle of radius 300 px (disc point (x, y) -> image ellipse)
    Rr = 300.0
    M = np.array([[a / Rr * np.cos(th), -b / Rr * np.sin(th), 0], [a / Rr * np.sin(th), b / Rr * np.cos(th), 0]])
    M[:, 2] = [ex, ey] - M[:, :2] @ [Rr, Rr]
    un = cv2.warpAffine(g, M, (600, 600), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    bh = cv2.morphologyEx(un, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    angs = np.radians(np.arange(0, 360, 0.5))
    rs = np.linspace(0.15, 0.9, 60)
    U = Rr + Rr * rs[None, :] * np.cos(angs[:, None])
    V = Rr + Rr * rs[None, :] * np.sin(angs[:, None])
    R = cv2.remap(bh, U.astype(np.float32), V.astype(np.float32), cv2.INTER_LINEAR)
    analyse.last_unwarp = cv2.warpAffine(f, M, (600, 600), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return {"frame": n, "ellipse": {"centre": [ex, ey], "a": a, "b": b, "theta_deg": float(np.degrees(th))}}, R


def detect(meta, R, Rmed):
    """Peaks of the static-pattern-subtracted response (the turned-brass streaks are fixed in image space; the mark
    rotates): R - median over frames."""
    angs = np.arange(0, 360, 0.5)
    Rs = np.clip(R - Rmed, 0, None)
    prof = Rs.mean(1)
    prof_s = np.convolve(np.r_[prof[-4:], prof, prof[:4]], np.ones(5) / 5, "same")[4:-4]
    i1 = int(np.argmax(prof_s))
    d = np.abs(((np.arange(720) - i1 + 360) % 720) - 360)
    p2 = prof_s.copy(); p2[d < 280] = -1                       # opposite half: within +-40 deg of i1 + 180
    i2 = int(np.argmax(p2))
    base = np.median(prof_s)
    cov = []
    for i in (i1, i2):                                        # fraction of radii where the ray response is > 3x the median
        cov.append(float((Rs[i] > max(3 * np.median(Rs), 0.25 * Rs[i].max())).mean()))
    a1, a2 = angs[i1], angs[i2]
    sep = abs(((a2 - a1 + 180) % 360) - 180)
    r = dict(meta)
    r.update({"half_angles_deg": [float(a1), float(a2)], "coverage": cov, "peak": [float(prof_s[i1] - base), float(prof_s[i2] - base)],
              "separation_deg": float(sep), "kink_deg": float(180 - sep),
              "diameter": bool(min(cov) >= 0.6 and prof_s[i2] - base > 0.25 * (prof_s[i1] - base))})
    return r, prof_s - base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=4)
    a = ap.parse_args()
    rows, profs, tiles = [], [], []
    metas, Rs, unw = [], [], {}
    for n in range(0, 241, a.step):
        meta, R = analyse(n)
        metas.append(meta); Rs.append(R)
        if n % 20 == 0:
            unw[n] = analyse.last_unwarp
    Rmed = np.median(np.array(Rs), axis=0)
    for meta, R in zip(metas, Rs):
        n = meta["frame"]
        r, p = detect(meta, R, Rmed)
        rows.append(r); profs.append(p)
        if n % 20 == 0:
            t = cv2.resize(unw[n], (200, 200), interpolation=cv2.INTER_AREA)
            cv2.circle(t, (100, 100), 90, (0, 255, 255), 1)
            cv2.putText(t, f"f{n} k{r['kink_deg']:.0f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles.append(t)
        print(n, "diam" if r["diameter"] else "RADIUS?", "kink %.1f" % r["kink_deg"], "cov", np.round(r["coverage"], 2))
    rowimgs = [np.hstack(tiles[i:i + 7] + [np.zeros_like(tiles[0])] * (7 - len(tiles[i:i + 7]))) for i in range(0, len(tiles), 7)]
    cv2.imwrite(str(HERE / "mark_unwarped.jpg"), np.vstack(rowimgs))
    k = np.array([r["kink_deg"] for r in rows if r["diameter"]])
    summ = {"frames": len(rows), "diameter_frames": int(sum(r["diameter"] for r in rows)),
            "kink_deg_median": float(np.median(k)) if len(k) else None,
            "kink_deg_p90": float(np.percentile(np.abs(k), 90)) if len(k) else None,
            "kink_abs_gt_5deg_frames": int((np.abs(k) > 5).sum()),
            "note": "kink = 180 - angle between the two half-lines in the unwarped disc plane; the unwarp assumes a level disc "
                    "and an affine (weak-perspective) ellipse, error ~1-2 deg"}
    (HERE / "mark_analysis.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].imshow(np.array(profs), aspect="auto", extent=[0, 360, 240, 0], cmap="magma")
    ax[0].set_xlabel("angle in the disc plane (deg)"); ax[0].set_ylabel("frame"); ax[0].set_title("dark-line angular profile")
    fr = [r["frame"] for r in rows]
    ax[1].plot(fr, [r["kink_deg"] for r in rows], "o-", ms=3)
    ax[1].set_xlabel("frame"); ax[1].set_ylabel("kink (deg)"); ax[1].set_title("180 - angle between half-lines")
    ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(HERE / "mark_analysis.png", dpi=90)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
