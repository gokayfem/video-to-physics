"""Calibration uncertainty (clip 03): edge jackknife (drop each edge) and free principal point. Writes camera/out/calib_variants.json."""
import json, pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]; PY = sys.executable
E = json.load(open(ROOT / "camera/out/edges_f000.json")); res = {}
for k in E:
    subprocess.run([PY, str(ROOT / "camera/calibrate.py"), "--drop=" + k], check=True, capture_output=True)
    r = json.load(open(ROOT / f"camera/out/calib_result_drop_{k}.json")); res[k] = {x: r[x] for x in ("f_px", "view_pitch_deg", "view_yaw_deg_from_Y", "roll_deg", "edge_rms_px")}
subprocess.run([PY, str(ROOT / "camera/calibrate.py"), "--free-pp"], check=True, capture_output=True)
fp = json.load(open(ROOT / "camera/out/calib_result_freepp.json"))
subprocess.run([PY, str(ROOT / "camera/calibrate.py")], check=True, capture_output=True)
import numpy as np
f = np.array([v["f_px"] for v in res.values()]); p = [v["view_pitch_deg"] for v in res.values()]; y = [v["view_yaw_deg_from_Y"] for v in res.values()]
n = len(f); jk = float(np.sqrt((n - 1) / n * np.sum((f - f.mean()) ** 2)))
out = {"jackknife": res, "summary": {"f_sd": jk, "f_min": float(f.min()), "f_max": float(f.max()), "pitch_range": [min(p), max(p)], "yaw_range": [min(y), max(y)]},
       "free_pp": {k: fp[k] for k in ("f_px", "cx", "cy", "view_pitch_deg", "view_yaw_deg_from_Y", "edge_rms_px")}}
json.dump(out, open(ROOT / "camera/out/calib_variants.json", "w"), indent=1); print(json.dumps(out["summary"]), out["free_pp"])
