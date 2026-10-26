"""Motion-blur shutter tuning, analysis side (VR-05).
.venv/bin/python scene/measure/mb_tune_analyze.py --dir scene/out/preview/mb [--out scene/measure/mb_tune.json]
Edge width = the visual reviewer's metric (visual_review/scripts/blur.py, re-implemented here, not imported):
grey image, rows v-25..v+25 averaged, 25-px window centred on the left/right silhouette (u -/+ r), count of
|d profile| > 30 % of its max, mean of both sides. Reference: frames 1..96 (t <= 4 s) with the reviewer's reference
bob track (visual_review/out/bobtrack_ref.csv); linear fit width = a + b*speed. Render: crops from mb_tune_render.py."""
import json, argparse, pathlib, sys, cv2, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "visual_review/scripts"))
from vrlib import track_bob   # read-only use of the reviewer's bob tracker so render and reference are tracked alike
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="scene/out/preview/mb"); ap.add_argument("--out", default="scene/measure/mb_tune.json")
a = ap.parse_args(); D = pathlib.Path(a.dir)
def edge_w(g, u, v, r, ox=0, oy=0):
    ws = []
    for side in (-1, 1):
        band = g[int(v) - 25 - oy:int(v) + 25 - oy].mean(0)
        x0 = int(u + side * r) - ox
        prof = band[x0 - 12:x0 + 13]
        d = np.abs(np.diff(prof)); ws.append(float((d > 0.3 * d.max()).sum()))
    return float(np.mean(ws))
tr = np.genfromtxt(ROOT / "visual_review/out/bobtrack_ref.csv", delimiter=",", names=True)
sp_ref = np.abs(np.gradient(tr["u_px"]))
W, S = [], []
for n in range(1, 97):
    g = cv2.cvtColor(cv2.imread(str(ROOT / f"visual_review/cache/ref/f{n:04d}.png")), cv2.COLOR_BGR2GRAY).astype(np.float32)
    W.append(edge_w(g, tr["u_px"][n], tr["v_px"][n], tr["radius_px"][n])); S.append(sp_ref[n])
W, S = np.array(W), np.array(S)
b, a0 = np.polyfit(S, W, 1)
res = {"reference": {"frames": "1..96", "fit_width_px": f"{a0:.3f} + {b:.4f} * speed_px_per_frame",
                     "a": a0, "b": b, "resid_rms_px": float(np.std(W - (a0 + b * S))),
                     "slow_q25": float(W[S < np.percentile(S, 25)].mean()), "fast_q75": float(W[S > np.percentile(S, 75)].mean()),
                     "speed_q25": float(S[S < np.percentile(S, 25)].mean()), "speed_q75": float(S[S > np.percentile(S, 75)].mean())}}
C = json.load(open(D / "crops.json")); T = np.genfromtxt(D / "track.csv", delimiter=",", names=True)
shutters = sorted({p.stem.split("_")[1] for p in (D / "crops").glob("*.png")}, key=float)
res["render"] = {}
for s in shutters:
    rows = []
    for n, (u0, v0, u1, v1, kind, spd) in C["crops"].items():
        n = int(n)
        im = cv2.imread(str(D / "crops" / f"{n:03d}_{s}.png"))
        canvas = np.empty((1088, 1920, 3), np.uint8); canvas[:] = np.median(im[:5].reshape(-1, 3), 0)
        canvas[v0:v0 + im.shape[0], u0:u0 + im.shape[1]] = im
        tu, tv, tr_, *_ = track_bob(canvas, (T["u_px"][n], T["v_px"][n]))
        g = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float32)
        w = edge_w(g, tu, tv, tr_)
        rows.append({"frame": n, "kind": kind, "speed": spd, "width": w, "ref_at_speed": a0 + b * spd, "diff": w - (a0 + b * spd)})
    d = np.array([r["diff"] for r in rows]); fast = [r for r in rows if r["kind"] == "fast"]; slow = [r for r in rows if r["kind"] == "slow"]
    res["render"][s] = {"rows": rows, "mean_diff_px": float(d.mean()), "rms_diff_px": float(np.sqrt((d ** 2).mean())),
                        "fast_mean_width": float(np.mean([r["width"] for r in fast])), "slow_mean_width": float(np.mean([r["width"] for r in slow])),
                        "fast_mean_ref": float(np.mean([r["ref_at_speed"] for r in fast])), "slow_mean_ref": float(np.mean([r["ref_at_speed"] for r in slow])),
                        "fast_speed": float(np.mean([r["speed"] for r in fast]))}
# slope fit per shutter: width = a + b*speed over all sampled frames; match b to the reference slope (offset-free)
for sname in shutters:
    rr = res["render"][sname]["rows"]; bb, aa = np.polyfit([r["speed"] for r in rr], [r["width"] for r in rr], 1)
    res["render"][sname]["fit_a"], res["render"][sname]["fit_b"] = float(aa), float(bb)
sb = np.array([res["render"][x]["fit_b"] - b for x in shutters])
kb = np.where(np.diff(np.sign(sb)) != 0)[0]
res["slope_diff_by_shutter"] = dict(zip(shutters, sb.tolist()))
res["shutter_slope_match"] = float(np.array([float(x) for x in shutters])[kb[0]] - sb[kb[0]] * (float(shutters[kb[0] + 1]) - float(shutters[kb[0]])) / (sb[kb[0] + 1] - sb[kb[0]])) if len(kb) else None
# choose: minimise rms diff on fast frames by linear interpolation over shutter
sh = np.array([float(s) for s in shutters]); fd = np.array([res["render"][s]["fast_mean_width"] - res["render"][s]["fast_mean_ref"] for s in shutters])
res["fast_diff_by_shutter"] = dict(zip(shutters, fd.tolist()))
k = np.where(np.diff(np.sign(fd)) != 0)[0]
res["shutter_zero_crossing"] = float(sh[k[0]] - fd[k[0]] * (sh[k[0] + 1] - sh[k[0]]) / (fd[k[0] + 1] - fd[k[0]])) if len(k) else None
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res["reference"], indent=0))
for s in shutters:
    r = res["render"][s]; print(s, "fast", round(r["fast_mean_width"], 2), "ref", round(r["fast_mean_ref"], 2), "| slow", round(r["slow_mean_width"], 2), "ref", round(r["slow_mean_ref"], 2), "| rms", round(r["rms_diff_px"], 2), "speed", round(r["fast_speed"], 1))
print("zero crossing (fast-frame width)", res["shutter_zero_crossing"]); print("slope match", res["shutter_slope_match"], {k: round(res["render"][k]["fit_b"], 3) for k in shutters}, "ref b", round(b, 3))
