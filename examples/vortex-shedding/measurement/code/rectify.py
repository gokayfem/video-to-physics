import sys, os, numpy as np, cv2, json
sys.path.insert(0, os.path.dirname(__file__)); from geom import *
W = os.path.join(ROOT, 'measurement/work'); OUT = os.path.join(ROOT, 'measurement')
fr = np.load(os.path.join(W, 'frames_u8.npy'), mmap_mode='r'); N = len(fr)
Hs = np.load(os.path.join(OUT, 'stabilisation_H_frame0_to_n.npy'))
xs, ys, mu, mv = rect_maps()
Hh, Ww = fr[0].shape[:2]
# validity: inside image, and not occluded by cylinder
valid = (mu > 2) & (mu < Ww - 3) & (mv > 2) & (mv < Hh - 3)
sil = []
for z in np.linspace(0.003, 0.036, 34):
    for a in np.linspace(0, 2*np.pi, 72, endpoint=False):
        sil.append(project([-0.04 + 0.005*np.cos(a), 0.005*np.sin(a), z]))
hull = cv2.convexHull(np.array(sil, np.float32) - 0.5)
silmask = np.zeros((Hh, Ww), np.uint8); cv2.fillConvexPoly(silmask, np.round(hull).astype(np.int32).reshape(-1, 2), 255)
Xg, Yg = np.meshgrid(xs, ys)
dC = np.hypot(Xg - C[0], Yg - C[1]); dcyl = np.hypot(-0.04 - C[0], -C[1])
inside = cv2.remap(silmask, mu, mv, cv2.INTER_NEAREST) > 0
occl = inside & (dC > dcyl - 0.005)
valid &= ~occl
valid &= Xg > -0.100   # rake/needles upstream
# trim columns with <50% valid rows (far right is off-frame)
colfrac = valid.mean(0); xmax = xs[colfrac > 0.5].max()
print('visible x range', xs[valid.any(0)].min(), xmax, 'rows valid frac', valid.mean())
os.makedirs(os.path.join(OUT, 'rectified'), exist_ok=True)
np.save(os.path.join(W, 'rect_valid.npy'), valid)
json.dump({'res_m_per_px': RES, 'x_range': XR, 'y_range': YR, 'z_plane': ZD, 'row0': '+y edge', 'col0': '-x edge',
           'x_visible_max_50pct_rows': float(xmax), 'map': 'pixel = refracted(n=1.333, flat surface z=0.023) pinhole of frame0 camera, then stabilisation homography H_n',
           'note': 'pixels outside valid mask (rect_valid.png) are black'}, open(os.path.join(OUT, 'rectified/meta.json'), 'w'), indent=1)
cv2.imwrite(os.path.join(OUT, 'rectified/rect_valid.png'), valid.astype(np.uint8)*255)
stack = np.zeros((N,) + valid.shape + (3,), np.uint8)
for i in range(N):
    H = Hs[i]; q = np.stack([mu + 0.0, mv, np.ones_like(mu)], -1) @ H.T   # mu,mv in index coords; H in pixel coords ~ same up to 0.5 shift (negligible for near-identity H)
    u = (q[..., 0] / q[..., 2]).astype(np.float32); v = (q[..., 1] / q[..., 2]).astype(np.float32)
    im = cv2.remap(np.asarray(fr[i]), u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    im[~valid] = 0; stack[i] = im
    cv2.imwrite(os.path.join(OUT, f'rectified/r{i:04d}.png'), im, [cv2.IMWRITE_PNG_COMPRESSION, 3])
np.save(os.path.join(W, 'rect_stack.npy'), stack)
