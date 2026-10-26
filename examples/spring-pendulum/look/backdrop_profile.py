"""Backdrop brightness profiles, reference vs render (VR-04). Oklab L of 9x9 px patches (960x544 image -> full-res coords)
at fixed backdrop-only sample points (frame-0 geometry: outside the posts, inside the opening clear of spring/weight).
.venv/bin/python look/backdrop_profile.py render.png [--frame 0] [--json out.json]"""
import argparse, json, pathlib, sys
import numpy as np, cv2
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics as MX
PTS = {"row300": [(u, 300) for u in (100, 250, 400, 680, 760, 1160, 1240, 1520, 1700, 1850)],
       "row500": [(u, 500) for u in (100, 250, 400, 680, 760, 1160, 1240, 1520, 1700, 1850)],
       "row700": [(u, 700) for u in (100, 250, 400, 680, 740, 1160, 1240, 1520, 1700, 1850)],
       "col700": [(700, v) for v in (150, 250, 350, 450, 550, 650, 750)],
       "col1200": [(1200, v) for v in (150, 250, 350, 450, 550, 650, 750)],
       "low_peak": [(1087, 746), (1150, 740), (1050, 780)]}


def L_at(img, pts):
    out = []
    for u, v in pts:
        p = img[v - 4:v + 5, u - 4:u + 5]
        out.append(float(MX.oklab(p)[..., 0].mean()))
    return np.array(out)


def profiles(ref, cur):
    res = {}
    for k, pts in PTS.items():
        r, c = L_at(ref, pts), L_at(cur, pts)
        res[k] = {"pts": pts, "ref": r.round(3).tolist(), "cur": c.round(3).tolist(), "dL": (c - r).round(3).tolist(),
                  "mae": round(float(np.abs(c - r).mean()), 4)}
    allr = np.concatenate([np.array(v["dL"]) for v in res.values()])
    res["summary"] = {"mae_all": round(float(np.abs(allr).mean()), 4), "max_abs": round(float(np.abs(allr).max()), 4)}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("render"); ap.add_argument("--frame", type=int, default=0); ap.add_argument("--json")
    a = ap.parse_args()
    ref = MX.ref_frame(a.frame)
    cur = cv2.resize(cv2.imread(a.render), (1920, 1088), interpolation=cv2.INTER_LINEAR)
    res = profiles(ref, cur)
    for k, v in res.items():
        if k != "summary":
            print(f"{k:9s} ref {v['ref']}\n          cur {v['cur']}  mae {v['mae']}")
    print(res["summary"])
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=1))
