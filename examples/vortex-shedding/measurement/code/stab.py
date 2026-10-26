import sys, os, numpy as np, cv2, json
sys.path.insert(0, os.path.dirname(__file__)); from geom import *
W = os.path.join(ROOT, 'measurement/work'); OUT = os.path.join(ROOT, 'measurement')
fr = np.load(os.path.join(W, 'frames_u8.npy'), mmap_mode='r')
N = len(fr); s = 0.5
# static mask: exclude channel interior dye region (dye plane |y|<0.062, x>-0.10) and its surroundings by projected polygon
poly = []
for x in np.linspace(-0.105, 0.30, 40): poly.append(project([x, -0.064, 0.013]))
for x in np.linspace(0.30, -0.105, 40): poly.append(project([x, 0.064, 0.013]))
poly = np.array(poly)
mask = np.full(fr[0].shape[:2], 255, np.uint8)
cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 0)
cv2.circle(mask, (770, 540), 70, 0, -1)  # cylinder body
mask[:60] = 0  # dark top, featureless
ms = cv2.resize(mask, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
g = lambda i: cv2.GaussianBlur(cv2.cvtColor(cv2.resize(np.asarray(fr[i]), None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
T = g(0); Hs = [np.eye(3)]; ccs = [1.0]; Wp = np.eye(3, dtype=np.float32)
crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
for i in range(1, N):
    try:
        cc, Wp = cv2.findTransformECC(T, g(i), Wp.copy(), cv2.MOTION_HOMOGRAPHY, crit, ms, 5)
    except cv2.error:
        cc = np.nan
    S = np.diag([s, s, 1.0]); H = np.linalg.inv(S) @ Wp.astype(float) @ S   # full-res frame0 -> frame i
    Hs.append(H / H[2, 2]); ccs.append(cc)
Hs = np.array(Hs)
# displacement at probe points (full res)
pts = np.array([[200, 900], [1700, 900], [200, 150], [1700, 150], [960, 544], [776.9, 604.0]], float)
def app(H, p):
    q = np.c_[p, np.ones(len(p))] @ H.T; return q[:, :2] / q[:, 2:]
disp = np.array([np.linalg.norm(app(H, pts) - pts, axis=1) for H in Hs])
np.save(os.path.join(OUT, 'stabilisation_H_frame0_to_n.npy'), Hs)
with open(os.path.join(OUT, 'camera_stability.csv'), 'w') as f:
    f.write('frame,t_s,ecc_cc,disp_px_corner_max,disp_px_centre,disp_px_cyl_base,' + ','.join(f'h{i}{j}' for i in range(3) for j in range(3)) + '\n')
    for i in range(N):
        f.write(f'{i},{i/24:.4f},{ccs[i]:.5f},{disp[i,:4].max():.3f},{disp[i,4]:.3f},{disp[i,5]:.3f},' + ','.join(f'{v:.8g}' for v in Hs[i].ravel()) + '\n')
cv2.imwrite(os.path.join(W, 'static_mask.png'), mask)
print('max corner disp', disp[:, :4].max(1)[[60, 120, 180, 240]], 'centre', disp[[60, 120, 180, 240], 4], 'cyl', disp[[60,120,180,240],5], 'cc min', np.nanmin(ccs))
