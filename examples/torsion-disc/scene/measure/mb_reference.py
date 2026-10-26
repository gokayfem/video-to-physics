"""Motion-blur evidence from the reference (scene role C): width of the engraved mark (luma dip FWHM, px) vs its
local image speed (px/frame), at frames where the mark is near the image vertical (it then moves horizontally and its
width is not foreshortened). Speed per row = |u_line(n+1) - u_line(n-1)| / 2 at that row; width = FWHM of the dip
below the local background (median of +-12..20 px) after subtracting the radial-brushing gradient (linear fit).
A shutter s (frames) widens a w0 line to ~ sqrt(w0^2 + (s v)^2) (box blur convolved with the static profile, FWHM
approximation). Writes scene/measure/mb_reference.json.
.venv/bin/python scene/measure/mb_reference.py
"""
import json, pathlib
import numpy as np, cv2
from scipy.optimize import least_squares

HERE = pathlib.Path(__file__).resolve().parent
REF = HERE.parents[3] / "reference-batch/clips/03-torsion-disc.mp4"


def frames(ns):
    cap = cv2.VideoCapture(str(REF)); out = {}
    for n in ns:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n); ok, f = cap.read(); assert ok
        out[n] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(float)
    return out


def line_u(g, v, u_guess, half=14):
    row = g[v, int(u_guess) - half:int(u_guess) + half + 1]
    i = int(np.argmin(row))
    if 1 <= i < len(row) - 1:                                  # parabolic sub-pixel minimum
        a, b, c = row[i - 1], row[i], row[i + 1]
        d = (a - c) / (2 * (a - 2 * b + c)) if (a - 2 * b + c) != 0 else 0
    else:
        d = 0
    return int(u_guess) - half + i + d, row


def fwhm(row, i0):
    x = np.arange(len(row), dtype=float)
    bg_idx = np.r_[0:6, len(row) - 6:len(row)]
    p = np.polyfit(x[bg_idx], row[bg_idx], 1)
    r = np.polyval(p, x) - row                                 # dip depth (positive)
    i = int(round(i0)); i = min(max(i, 1), len(row) - 2)
    pk = r[i]
    if pk < 8:
        return None, pk
    h = pk / 2
    lft = i
    while lft > 0 and r[lft] > h: lft -= 1
    rgt = i
    while rgt < len(r) - 1 and r[rgt] > h: rgt += 1
    xl = lft + (h - r[lft]) / (r[lft + 1] - r[lft]) if r[lft + 1] != r[lft] else lft
    xr = rgt - 1 + (r[rgt - 1] - h) / (r[rgt - 1] - r[rgt]) if r[rgt - 1] != r[rgt] else rgt
    return xr - xl, pk


def main():
    rows = []
    for centre_frames, (uc, vc) in (((70, 71, 72, 73, 74), (995, 652)), ((174, 175, 176, 177, 178), (1010, 650))):
        F = frames(range(centre_frames[0] - 1, centre_frames[-1] + 2))
        for n in centre_frames:
            for v in list(range(vc - 50, vc - 12, 4)) + list(range(vc + 12, vc + 110, 4)):
                # follow the line from the centre outwards using the previous row's position
                u0, _ = line_u(F[n], vc + (12 if v > vc else -12), uc)
                u, row = line_u(F[n], v, u0 + (v - vc) * 0.0, half=30)
                up, _ = line_u(F[n - 1], v, u, half=12)
                un, _ = line_u(F[n + 1], v, u, half=12)
                speed = abs(un - up) / 2
                seg = F[n][v, int(round(u)) - 20:int(round(u)) + 21]
                w, depth = fwhm(seg, 20 + (u - round(u)))
                if w is not None and 1 < w < 25:
                    rows.append({"frame": n, "v": v, "u": round(u, 2), "speed_px_per_frame": round(speed, 3), "fwhm_px": round(w, 3),
                                 "depth": round(float(depth), 1)})
    sp = np.array([r["speed_px_per_frame"] for r in rows]); w = np.array([r["fwhm_px"] for r in rows])
    fit = least_squares(lambda p: np.sqrt(p[0] ** 2 + (p[1] * sp) ** 2) - w, [4.0, 0.3], loss="soft_l1", f_scale=1.0)
    w0, s = fit.x
    lin = np.polyfit(sp, w, 1)
    bins = [(0, 2), (2, 4), (4, 6), (6, 9), (9, 15)]
    tab = [{"speed_px": f"{a}-{b}", "n": int(((sp >= a) & (sp < b)).sum()),
            "fwhm_median": round(float(np.median(w[(sp >= a) & (sp < b)])), 2) if ((sp >= a) & (sp < b)).any() else None} for a, b in bins]
    out = {"method": __doc__.strip().splitlines()[0], "n": len(rows), "fit_quadrature": {"w0_px": round(float(w0), 3), "shutter_frames": round(float(abs(s)), 3)},
           "fit_linear": {"intercept_px": round(float(lin[1]), 3), "slope_px_per_px_speed": round(float(lin[0]), 3)},
           "binned": tab, "rows": rows}
    (HERE / "mb_reference.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
