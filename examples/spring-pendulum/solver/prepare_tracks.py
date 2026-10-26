"""Merge measurement tracks into the fit input solver/out/tracks_for_fit.csv (no processing beyond column mapping).
ring = eye-ring centre (measurement/ring_track.csv, method A, sigma from A-vs-B), wctr = weight centre
(measurement/weight_track.csv, method B, sigma from B-vs-C; measurement NOTES: treat as +-3 px systematic -> sigma floor
--wctr-sigma-floor, default 1.5 px = measured median)."""
import argparse, csv, hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("--wctr-sigma-floor", type=float, default=0.0)
ap.add_argument("--out", default=str(ROOT / "solver/out/tracks_for_fit.csv")); a = ap.parse_args()
R = list(csv.DictReader(open(ROOT / "measurement/ring_track.csv"))); W = list(csv.DictReader(open(ROOT / "measurement/weight_track.csv")))
assert len(R) == len(W) == 241
with open(a.out, "w") as f:
    f.write("frame,t_s,ring_u_px,ring_v_px,ring_sigma_u_px,ring_sigma_v_px,wctr_u_px,wctr_v_px,wctr_sigma_u_px,wctr_sigma_v_px\n")
    for r, w in zip(R, W):
        assert r["frame"] == w["frame"]
        su = max(float(w["sigma_u_px"]), a.wctr_sigma_floor); sv = max(float(w["sigma_v_px"]), a.wctr_sigma_floor)
        f.write(f"{int(r['frame'])},{int(r['frame'])/24!r},{r['u_px']},{r['v_px']},{r['sigma_u_px']},{r['sigma_v_px']},{w['u_px']},{w['v_px']},{su!r},{sv!r}\n")
json.dump({"inputs": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ("measurement/ring_track.csv", "measurement/weight_track.csv")},
           "wctr_sigma_floor": a.wctr_sigma_floor}, open(a.out.replace(".csv", ".json"), "w"), indent=1)
print(a.out)
