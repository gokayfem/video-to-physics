"""Analyse the motion-blur calibration crops with the same effective-edge-width metric as mb_reference.py and propose a
shutter. -> scene/measure/mb_tune.json"""
import json, pathlib, math
import numpy as np, cv2
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
D = ROOT / "scene/out/mb"
I = json.loads((D / "mb_render.json").read_text())
tilt = -math.degrees(I["pose"]["beta_rad"])            # weight_track image-tilt convention (f120: track -6.9, beta +7.5)
REF = json.loads((HERE / "mb_reference.json").read_text())


def eff_width(g, v, t_deg):
    t = np.tan(np.radians(t_deg))
    band = np.zeros(g.shape[1], np.float32)
    for dy in range(-25, 25):
        band += np.interp(np.arange(g.shape[1]) + t * dy, np.arange(g.shape[1]), g[v + dy])
    band /= 50
    d = np.abs(np.diff(band)); mid = len(band) // 2
    ws = []
    for lo, hi in ((15, mid - 20), (mid + 20, len(band) - 16)):
        x0 = lo + int(np.argmax(d[lo:hi]))
        dd = d[x0 - 12:x0 + 12]
        ws.append(float(abs(band[x0 + 9] - band[x0 - 9]) / max(dd.max(), 1e-6)))
    return float(np.mean(ws))


rows = []
for r in I["renders"]:
    g = cv2.cvtColor(cv2.imread(str(D / r["file"])), cv2.COLOR_BGR2GRAY).astype(np.float32)
    rows.append({**r, "eff_width_px": eff_width(g, g.shape[0] // 2 + 20, tilt)})
ra, rb = REF["fit_effective_width"]["a_px"], REF["fit_effective_width"]["b"]
by_sh = {}
for r in rows:
    by_sh.setdefault(r["shutter"], []).append(r)
w0 = [r["eff_width_px"] for r in rows if r["speed_px_per_frame"] == 0][0]
summary = {}
for sh, rr in sorted(by_sh.items()):
    pts = [(0.0, w0)] + [(r["speed_px_per_frame"], r["eff_width_px"]) for r in rr if r["speed_px_per_frame"] > 0]
    sp, wd = np.array(pts).T
    b, a = np.polyfit(sp, wd, 1)
    summary[str(sh)] = {"slope": float(b), "intercept": float(a), "widths": wd.round(3).tolist(), "speeds": sp.tolist()}
shs = np.array(sorted(float(k) for k in summary)); slopes = np.array([summary[str(s) if str(s) in summary else repr(s)]["slope"] for s in shs])
prop = float(np.interp(rb, slopes, shs)) if slopes.min() <= rb <= slopes.max() else None
res = {"reference_fit": REF["fit_effective_width"], "render_static_width_px": w0, "render_by_shutter": summary,
       "proposal_shutter_slope_match": prop, "rows": rows,
       "note": "slope = d(effective edge width)/d(speed); proposal = shutter whose render slope equals the reference slope (linear interpolation)"}
(HERE / "mb_tune.json").write_text(json.dumps(res, indent=1))
print("ref slope", rb, "ref intercept", ra, "render static", w0)
for k, v in summary.items():
    print(k, round(v["slope"], 4), v["widths"])
print("PROPOSED shutter", prop)
