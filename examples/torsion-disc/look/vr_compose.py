"""Visual-review response sheet (scene role C): REFERENCE | ck001 (review/renders/ck001_eevee50, 50 % preview) | NEW
(look/out/<tag>, baked physics, camera of record) at one frame, close-ups (clamp, disc, wood) as ref | ck001 | new,
and the VR03 numbers (look/metrics.py "review" block, region dL, backdrop 2D, disc ring profile).
.venv/bin/python look/vr_compose.py --tag n0 --frame 0 --version v010 --out scene/visuals/...jpg
"""
import argparse, json, pathlib
import numpy as np, cv2
import metrics as MX
from compose import bar, fitbox

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--frame", type=int, required=True)
ap.add_argument("--version", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--caption", default="")
ap.add_argument("--ck", default="ck001")
a = ap.parse_args()
ref = MX.ref_frame(a.frame)
ck = cv2.resize(cv2.imread(str(ROOT / f"review/renders/{a.ck}_eevee50/f{a.frame:04d}.png")), (1920, 1088), interpolation=cv2.INTER_CUBIC)
info = json.loads((HERE / "out" / f"{a.tag}_render.json").read_text())
cur = cv2.resize(cv2.imread(info["files"]["full"]), (1920, 1088), interpolation=cv2.INTER_CUBIC)
for k, b in info["crop_boxes"].items():
    cur[b[1]:b[3], b[0]:b[2]] = cv2.imread(info["files"][k])
m = json.loads((HERE / "out" / f"metrics_{a.tag}.json").read_text())
W = 1920; w3 = W // 3; h3 = int(w3 * 1088 / 1920)
row = np.hstack([fitbox(i, w3, h3) for i in (ref, ck, cur)])
lab = np.hstack([bar(w3, [(t, 0.55, (255, 255, 255))], 28) for t in (f"REFERENCE frame {a.frame}", f"{a.ck} (50 % EEVEE preview, upscaled)", f"NEW {a.version} (baked, camera of record)")])
tiles = []
for k in [q for q in ("clamp", "disc", "wood") if q in info["crop_boxes"]]:
    u0, v0, u1, v1 = info["crop_boxes"][k]
    trip = np.hstack([ref[v0:v1, u0:u1], ck[v0:v1, u0:u1], cur[v0:v1, u0:u1]])
    tiles.append(np.vstack([bar(w3, [(f"{k}: ref | {a.ck} | new (100 %; {a.ck} crop upscaled from 50 %)", 0.42, (220, 220, 220))], 26), fitbox(trip, w3, 260)]))
rv = m["review"]; r, c = rv["ref"], rv["cur"]
keys = ["horizon_v_px", "wall_left_L", "wall_mid_L", "wall_right_L", "wall_C", "wall_hue_deg", "brass_C", "brass_C_p05_p95",
        "fine_over_main_lobe_power", "brass_L", "mark_fwhm_px", "mark_dip"]
lines = [(f"{a.version} | frame {a.frame} | {a.caption}", 0.55, (255, 255, 255)),
         ("VR03 numbers ref -> new: " + "  ".join(f"{k} {r[k]} -> {c[k]}" for k in keys[:6] if k in r), 0.42, (200, 200, 120)),
         ("  " + "  ".join(f"{k} {r[k]} -> {c[k]}" for k in keys[6:] if k in r), 0.42, (200, 200, 120))]
b = m["backdrop2d"]; d = m.get("disc_profile", {}).get("r0.7", {})
s = f"backdrop 2D: MAE {b['mae']:.3f} bias {b['bias']:+.3f} corr {b['corr']:.2f}   disc ring r0.7: MAE {d.get('mae', float('nan')):.3f} corr {d.get('corr', float('nan')):.2f} std ref/new {d.get('std_ref', 0):.3f}/{d.get('std_cur', 0):.3f}"
if "regions" in m:
    s += "   regions dL: " + " ".join(f"{k} {v['dL']:+.3f}" for k, v in m["regions"].items() if k in ("foot_left", "crossbar", "post_left", "clamp", "backdrop_right", "table_left"))
lines.append((s, 0.42, (200, 200, 200)))
sheet = np.vstack([lab, row, np.hstack(tiles + [np.full((tiles[0].shape[0], W - len(tiles) * w3, 3), 24, np.uint8)]), bar(W, lines)])
cv2.imwrite(a.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]); print("wrote", a.out)
