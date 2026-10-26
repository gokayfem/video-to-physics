"""Deterministic ink masks (white = black ink) for the ruler face and the grid card.
Run with the numpy/opencv venv:  python scene/make_textures.py
Image v=0 (bottom row) maps to the object's low end (ruler bottom / card bottom); u=0 maps to -X.
"""
import json
import pathlib

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = json.loads((ROOT / "config/scene_params.json").read_text())
OUT = ROOT / "scene/tex"
OUT.mkdir(parents=True, exist_ok=True)


def ruler(ppm=40):  # px per mm
    r = P["ruler"]; m = r["markings"]
    W = int(round(r["width_m"] * 1e3 * ppm)); H = int(round(r["length_m"] * 1e3 * ppm))
    img = np.zeros((H, W), np.uint8)  # image coords: row 0 = ruler TOP (z = length), no flips (text must not mirror)
    lw = m["tick_line_width_m"] * 1e3 * ppm
    for mm in range(5, int(r["length_m"] * 1e3), 5):
        z = H - mm * ppm
        L = m["tick_len_10mm_m"] if mm % 10 == 0 else m["tick_len_5mm_m"]
        x1 = int(round(L * 1e3 * ppm))
        y0 = int(round(z - lw / 2)); y1 = int(round(z + lw / 2))
        img[max(y0, 0):y1 + 1, 0:x1] = 255
    # numbers (cm), rotated like a real ruler: text baseline along the length
    th = m["number_height_m"] * 1e3 * ppm
    for cm in range(1, m.get("numbers_max_cm", int(r["length_m"] * 100) - 1) + 1):
        s = str(cm)
        scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, int(th), 4)
        (tw, tht), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 4)
        tile = np.zeros((tht + 8, tw + 8), np.uint8)
        cv2.putText(tile, s, (4, tht + 4), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 4, cv2.LINE_AA)
        # rotate so text reads bottom->top along the upright ruler (like a real ruler stood on its 0 end)
        tile = cv2.rotate(tile, cv2.ROTATE_90_COUNTERCLOCKWISE)
        th_, tw_ = tile.shape
        cx = int(round((m["tick_len_10mm_m"] * 1e3 + 1.8) * ppm + tw_ / 2))
        cz = int(round(H - cm * 10 * ppm))
        y0 = cz - th_ // 2; x0 = cx - tw_ // 2
        if y0 < 0 or y0 + th_ > H:
            continue
        img[y0:y0 + th_, x0:x0 + tw_] = np.maximum(img[y0:y0 + th_, x0:x0 + tw_], tile)
    cv2.imwrite(str(OUT / "ruler_ink.png"), img)
    return W, H


def card(ppm=12):
    c = P["grid_card"]
    W = int(round(c["width_m"] * 1e3 * ppm)); H = int(round(c["height_m"] * 1e3 * ppm))
    img = np.zeros((H, W), np.uint8)
    x0m = c["x_center_m"] - c["width_m"] / 2; z0m = c["bottom_z_m"]
    pitch = c["grid_pitch_m"]; mg = c["margin_m"]
    xs = np.arange(np.ceil((x0m + mg) / pitch - 1e-9), np.floor((x0m + c["width_m"] - mg) / pitch + 1e-9) + 1) * pitch
    zs = np.arange(np.ceil((z0m + mg) / pitch - 1e-9), np.floor((z0m + c["height_m"] - mg) / pitch + 1e-9) + 1) * pitch
    xa, xb = (xs[0] - x0m) * 1e3 * ppm, (xs[-1] - x0m) * 1e3 * ppm
    za, zb = (zs[0] - z0m) * 1e3 * ppm, (zs[-1] - z0m) * 1e3 * ppm
    for i, x in enumerate(xs):
        major = int(round(x / pitch)) % c["major_every"] == 0
        lw = (c["major_line_width_m"] if major else c["line_width_m"]) * 1e3 * ppm
        u = (x - x0m) * 1e3 * ppm
        img[int(round(za - lw / 2)):int(round(zb + lw / 2)) + 1, int(round(u - lw / 2)):int(round(u + lw / 2)) + 1] = 255
    for z in zs:
        major = int(round(z / pitch)) % c["major_every"] == 0
        lw = (c["major_line_width_m"] if major else c["line_width_m"]) * 1e3 * ppm
        v = (z - z0m) * 1e3 * ppm
        img[int(round(v - lw / 2)):int(round(v + lw / 2)) + 1, int(round(xa - lw / 2)):int(round(xb + lw / 2)) + 1] = 255
    cv2.imwrite(str(OUT / "card_ink.png"), np.flipud(img))
    return dict(x_lines_m=[round(float(v), 4) for v in xs], z_lines_m=[round(float(v), 4) for v in zs])


if __name__ == "__main__":
    print("ruler", ruler())
    g = card()
    (OUT / "card_lines.json").write_text(json.dumps(g))
    print("card x", g["x_lines_m"][0], g["x_lines_m"][-1], "z", g["z_lines_m"][0], g["z_lines_m"][-1])
