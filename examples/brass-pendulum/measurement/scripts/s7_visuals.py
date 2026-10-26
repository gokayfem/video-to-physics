"""Stage 6b: annotated reference frames, crops and visual evidence -> measurement/visuals, measurement/crops."""
import cv2, numpy as np, os, sys, csv, json, pickle
sys.path.insert(0, os.path.dirname(__file__))
from common import *
VER = "v001"; STATUS = "reference measurement - draft"
F = load_frames(); N = 241
A = np.load(os.path.join(CACHE, "analysis.npz")); Bz = np.load(os.path.join(CACHE, "bob_raw.npz"))
H = np.load(os.path.join(CACHE, "H_ecc.npy")); WD = pickle.load(open(os.path.join(CACHE, "wire_diag.pkl"), "rb"))
WR = pickle.load(open(os.path.join(CACHE, "wire_rows.pkl"), "rb")); E = pickle.load(open(os.path.join(CACHE, "bob_edges.pkl"), "rb"))
cA, rad, cS, pu, pv, th = Bz["cA"], Bz["radA"], A["cS"], A["pu"], A["pv"], A["theta"]
VIS = os.path.join(MEAS, "visuals"); CR = os.path.join(MEAS, "crops"); os.makedirs(VIS, exist_ok=True); os.makedirs(CR, exist_ok=True)
def P(u, v, z=1, o=(0, 0)): return (int(round((u - o[0]) * z * 16)), int(round((v - o[1]) * z * 16)))  # shift=4 subpixel drawing
SH = 4
def banner(img, lines, h=None):
    h = h or 26 * len(lines) + 10
    out = np.vstack([np.full((h, img.shape[1], 3), 25, np.uint8), img])
    for i, l in enumerate(lines):
        cv2.putText(out, l, (10, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return out
def tcol(n):
    c = cv2.applyColorMap(np.array([[int(255 * n / 240)]], np.uint8), cv2.COLORMAP_TURBO)[0, 0]; return tuple(int(x) for x in c)
def track_in_frame(n):
    """all 241 bob centres expressed in frame n's raw geometry (stabilized track mapped with H_n)."""
    Q = (H[n] @ np.c_[cS, np.ones(N)].T).T; return Q[:, :2] / Q[:, 2:]
def draw_frame(n, z=1.0, crop=None, show_track=True, show_wire=True):
    img = np.ascontiguousarray(F[n]).copy()
    o = (0, 0)
    if crop:
        u0, v0, u1, v1 = crop; img = img[v0:v1, u0:u1]; o = (u0, v0)
    if z != 1: img = cv2.resize(img, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    if show_track:
        T = track_in_frame(n)
        for k in range(N - 1):
            cv2.line(img, P(*T[k], z, o), P(*T[k + 1], z, o), tcol(k), 1, cv2.LINE_AA, SH)
    if show_wire:
        vv, uu, _ = WD[n]
        for a, b in zip(uu[::3], vv[::3]): cv2.circle(img, P(a, b, z, o), int(16 * 1.2), (0, 255, 255), -1, cv2.LINE_AA, SH)
    cv2.circle(img, P(*cA[n], z, o), int(16 * rad[n] * z), (255, 255, 0), 1, cv2.LINE_AA, SH)
    cv2.drawMarker(img, (int((cA[n, 0] - o[0]) * z), int((cA[n, 1] - o[1]) * z)), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
    cv2.line(img, P(pu[n], pv[n], z, o), P(*cA[n], z, o), (0, 0, 255), 1, cv2.LINE_AA, SH)
    cv2.drawMarker(img, (int((pu[n] - o[0]) * z), int((pv[n] - o[1]) * z)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
    return img
legend = "legend: cyan circle/+ = fitted bob silhouette & centre (A); yellow dots = detected wire centreline; red x = pivot (tracked wire departure); red line = pivot->bob; track coloured by time (blue t=0 -> red t=10 s)"
# V1: frame 0 full with full trajectory + landmarks
img = draw_frame(0)
for r in json.load(open(os.path.join(MEAS, "static_landmarks_def.json"))):
    cv2.circle(img, (int(r["u0"]), int(r["v0"])), 6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, r["name"], (int(r["u0"]) + 8, int(r["v0"]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
img = banner(img, [f"{VER} | frame 0, t = 0.000 s | {STATUS} | isolated component (measurement)",
                   "OBSERVED: bob centre track of all 241 frames (mapped into frame-0 geometry with the measured camera-drift homography), wire centreline, pivot, static landmarks (green)",
                   legend])
cv2.imwrite(os.path.join(VIS, f"{VER}_annotated_f000_full_track.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
# V2: turning points and zero crossings, crop pivot..bob
pk = A["peaks"]; zc = A["zc"]
sel = [(int(round(zc[0, 0] * 24)), "zero crossing"), (int(pk[0, 0]), "turning pt R"), (int(pk[1, 0]), "turning pt L"),
       (int(pk[9, 0]), "turning pt L (largest)"), (int(pk[10, 0]), "turning pt R"), (240, "last frame")]
tiles = []
for n, lbl in sel:
    im = draw_frame(n, z=0.75, crop=(880, 40, 1320, 960))
    cv2.rectangle(im, (0, 0), (im.shape[1], 50), (25, 25, 25), -1)
    cv2.putText(im, f"f{n} t={n/24:.3f}s {lbl}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(im, f"angle {th[n]:+.2f} deg  sag {WR[n]['sagitta_quad_px']:+.1f}px", (6, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(im)
grid = np.hstack(tiles)
grid = banner(grid, [f"{VER} | frames {', '.join(str(s[0]) for s in sel)} | {STATUS}",
                     "OBSERVED per frame: pivot (red x), wire centreline (yellow), bob circle fit (cyan); full 10 s bob track drawn in each frame's own geometry (zoom drift included)",
                     legend])
cv2.imwrite(os.path.join(VIS, f"{VER}_turning_points_zero_crossings.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
# V3: wire straightness - rectified, horizontally exaggerated strips
def wire_strip(n, ex=6, half=14):
    vv, uu, _ = WD[n]
    top = np.array([uu[0], vv[0]]); bot = np.array([uu[-1], vv[-1]])
    L = np.linalg.norm(bot - top); d = (bot - top) / L; nrm = np.array([d[1], -d[0]])  # nrm points toward +u roughly
    s = np.arange(0, L, 1.0); o = np.arange(-half, half + 0.001, 1.0 / ex)
    S_, O_ = np.meshgrid(s, o, indexing="ij")
    U = top[0] + S_ * d[0] - O_ * nrm[0]; V = top[1] + S_ * d[1] - O_ * nrm[1]
    g = cv2.remap(np.ascontiguousarray(F[n]), (U - 0.5).astype(np.float32), (V - 0.5).astype(np.float32), cv2.INTER_LINEAR)
    g = g[::2]  # vertical compression x0.5 to keep size
    # chord line (straight reference) at offset 0
    c = int(round(half * ex)); g[:, c] = (0, 0, 255)
    # measured centreline offsets
    for a, b in zip(uu[::4], vv[::4]):
        sp = (np.array([a, b]) - top) @ d; op = -(np.array([a, b]) - top) @ nrm
        x = int(round((op + half) * ex)); y = int(sp / 2)
        if 0 <= x < g.shape[1] and 0 <= y < g.shape[0]: g[y, max(0, x - 1):x + 2] = (0, 255, 255)
    g = np.vstack([np.full((50, g.shape[1], 3), 25, np.uint8), g])
    cv2.putText(g, f"f{n}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(g, f"{th[n]:+.1f}deg", (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(g, f"s={WR[n]['sagitta_quad_px']:+.1f}", (4, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return g
wsel = [int(pk[1, 0]), int(round(zc[1, 0] * 24)), int(pk[0, 0]), int(pk[9, 0]), int(round(zc[10, 0] * 24)), int(pk[10, 0])]
strips = [wire_strip(n) for n in wsel]
hh = max(s.shape[0] for s in strips)
strips = [np.vstack([s, np.zeros((hh - s.shape[0], s.shape[1], 3), np.uint8)]) for s in strips]
ws = np.hstack([np.hstack([s, np.full((hh, 8, 3), 80, np.uint8)]) for s in strips])
ws = banner(ws, [f"{VER} | wire straightness | {STATUS}",
                 "Image resampled along the chord top->bottom of the detected wire; across-wire axis", "stretched 6x, along-wire x0.5. Red = straight chord;",
                 "yellow = measured centreline. OBSERVED: wire bows", "~3-4 px at turning points, straight near zero crossings."], h=140)
cv2.imwrite(os.path.join(VIS, f"{VER}_wire_straightness_rectified.jpg"), ws, [cv2.IMWRITE_JPEG_QUALITY, 92])
cv2.imwrite(os.path.join(CR, f"{VER}_wire_rectified_strips.png"), ws)
# raw wire crops (no exaggeration) at the same frames, mid-wire region
for n in wsel:
    im = np.ascontiguousarray(F[n][300:700, 1000:1200]).copy()
    vv, uu, _ = WD[n]
    a0, vm, a1 = WR[n]["slope"], WR[n]["vm"], WR[n]["icpt"]
    cv2.line(im, P(a0 * (300 - vm) + a1, 300, 1, (1000, 300)), P(a0 * (700 - vm) + a1, 700, 1, (1000, 300)), (0, 0, 255), 1, cv2.LINE_AA, SH)
    im = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    im = banner(im, [f"{VER} f{n} t={n/24:.3f}s", "u1000-1200 v300-700 x2", "red=best straight line"], h=86)
    cv2.imwrite(os.path.join(CR, f"{VER}_wire_mid_f{n:03d}.png"), im)
# V4: camera drift anaglyph raw and stabilized
g0 = cv2.cvtColor(np.ascontiguousarray(F[0]), cv2.COLOR_BGR2GRAY); g1 = cv2.cvtColor(np.ascontiguousarray(F[240]), cv2.COLOR_BGR2GRAY)
Tc = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1.0]]); Hcv = np.linalg.inv(Tc) @ H[240] @ Tc
g1s = cv2.warpPerspective(g1, Hcv, (1920, 1088), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
an1 = cv2.merge([g1, g1, g0]); an2 = cv2.merge([g1s, g1s, g0])
an = np.hstack([cv2.resize(an1, (960, 544)), cv2.resize(an2, (960, 544))])
cv2.putText(an, "RAW: R=frame 0, G/B=frame 240", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(an, "after drift homography H_240^-1", (970, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
an = banner(an, [f"{VER} | frames 0 vs 240 | {STATUS}",
                 "Anaglyph: red channel = frame 0, green/blue = frame 240 (a dark object only in frame 0 shows cyan, only in frame 240 shows red). OBSERVED: static stand/base",
                 "do not overlap in raw frames (colour fringes = camera zoom/dolly, scale x1.033, base corners move ~25-30 px);",
                 "after the per-frame homography the static scene overlaps (landmark residual RMS 0.77 px). Bob differs because it swings."])
cv2.imwrite(os.path.join(VIS, f"{VER}_camera_drift_f000_vs_f240.jpg"), an, [cv2.IMWRITE_JPEG_QUALITY, 92])
# V5: cadence strip frames 0..12 bob crops
tiles = []
for n in range(0, 13):
    c = cA[n]; u0, v0 = int(c[0]) - 75, int(c[1]) - 75
    im = np.ascontiguousarray(F[n][v0:v0 + 150, u0:u0 + 150]).copy()
    cv2.circle(im, P(*c, 1, (u0, v0)), int(16 * rad[n]), (255, 255, 0), 1, cv2.LINE_AA, SH)
    cv2.drawMarker(im, (int(c[0] - u0), int(c[1] - v0)), (255, 255, 0), cv2.MARKER_CROSS, 12, 1)
    du = cA[n, 0] - cA[n - 1, 0] if n else 0
    col = (0, 0, 255) if n % 4 == 0 and n else (255, 255, 255)
    cv2.putText(im, f"f{n} du={du:+.1f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    tiles.append(im)
st = np.hstack(tiles)
st = banner(st, [f"{VER} | frames 0-12 (crop follows the bob) | {STATUS}",
                 "OBSERVED: du = frame-to-frame step of bob centre [px]; red = steps into frames n=4k, systematically ~1.6x larger (4-frame generator cadence)"])
cv2.imwrite(os.path.join(VIS, f"{VER}_bob_cadence_f000-012.jpg"), st, [cv2.IMWRITE_JPEG_QUALITY, 92])
# bob overlay samples (masks / edge points) into crops
sys.argv = ["x"]
import viz_bob
for grp in [[0, 13, 21], [29, 112, 163], [179, 231, 240]]:
    im = viz_bob.overlay(grp[0]); ims = [viz_bob.overlay(n) for n in grp]
    out = banner(np.hstack(ims), [f"{VER} | bob edge points (green = used, red = rejected), fitted circle (cyan) & centre A (+), method B (magenta x) | {STATUS}"])
    cv2.imwrite(os.path.join(CR, f"{VER}_bob_fit_f{grp[0]:03d}_{grp[1]:03d}_{grp[2]:03d}.png"), out)
# pivot crops
im = np.hstack([cv2.resize(np.ascontiguousarray(F[n][70:190, 1030:1150]), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST) for n in [int(pk[1, 0]), int(round(zc[1, 0] * 24)), int(pk[0, 0])]])
for i, n in enumerate([int(pk[1, 0]), int(round(zc[1, 0] * 24)), int(pk[0, 0])]):
    cv2.drawMarker(im, (int((pu[n] - 1030) * 4) + 480 * i, int((pv[n] - 70) * 4)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 24, 2)
    cv2.putText(im, f"f{n} {th[n]:+.1f}deg", (480 * i + 6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
im = banner(im, [f"{VER} | pin/loop at left TP, zero crossing, right TP (x4) | {STATUS}", "red x = tracked pivot (wire departure). OBSERVED: loop does not rock; pivot fixed to <0.5 px after drift"])
cv2.imwrite(os.path.join(VIS, f"{VER}_pivot_closeups.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(sorted(os.listdir(VIS)), sorted(os.listdir(CR)))
