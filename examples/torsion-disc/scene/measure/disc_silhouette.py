"""Disc (brass) silhouette per reference frame (scene role C). Oklab b* + chroma threshold inside an ROI,
largest component, boundary points. Writes scene/measure/disc_sil_fNNN.json and a debug overlay.
.venv/bin/python scene/measure/disc_silhouette.py 0 120 240"""
import sys, json, pathlib, cv2, numpy as np
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = ROOT.parents[1] / "reference-batch/clips/03-torsion-disc.mp4"
sys.path.insert(0, str(ROOT / "look"))


def frame(n):
    cap = cv2.VideoCapture(str(REF)); cap.set(cv2.CAP_PROP_POS_FRAMES, n); ok, f = cap.read(); assert ok
    return f


def oklab(bgr):
    c = bgr[..., ::-1].astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return np.cbrt(lin @ M1.T) @ M2.T


def silhouette(n, roi=(600, 520, 1360, 860)):
    f = frame(n)
    u0, v0, u1, v1 = roi
    c = f[v0:v1, u0:u1]
    L = oklab(c)
    hue = np.degrees(np.arctan2(L[..., 2], L[..., 1]))
    C = np.hypot(L[..., 1], L[..., 2])
    m = ((C > 0.025) & (hue > 45) & (hue < 115)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    k, lab, st, _ = cv2.connectedComponentsWithStats(m)
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    m = (lab == i).astype(np.uint8)
    # fill holes (the dark engraved line, wire)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cs, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    pts = cnt[:, 0, :].astype(float) + [u0 + 0.5, v0 + 0.5]
    # keep contour points that lie on the convex hull (within 2 px): drops notches where the line / wire cut in
    d = np.array([abs(cv2.pointPolygonTest(hull, (float(p[0] - u0 - 0.5), float(p[1] - v0 - 0.5)), True)) for p in pts])
    pts = pts[d < 1.5]
    return f, pts


if __name__ == "__main__":
    for n in map(int, sys.argv[1:]):
        f, pts = silhouette(n)
        sel = pts[::3]
        (HERE / f"disc_sil_f{n:03d}.json").write_text(json.dumps({"frame": n, "points_px": sel.round(2).tolist(),
            "method": "Oklab chroma>0.025, hue 45-115 deg in ROI; largest component; contour points on the convex hull"}))
        dbg = f.copy()
        for p in sel:
            cv2.circle(dbg, (int(p[0]), int(p[1])), 1, (0, 0, 255), -1)
        cv2.imwrite(str(HERE / f"disc_sil_f{n:03d}.jpg"), dbg[480:900, 560:1400])
        print(n, len(sel), sel[:, 0].min(), sel[:, 0].max(), sel[:, 1].min(), sel[:, 1].max())
