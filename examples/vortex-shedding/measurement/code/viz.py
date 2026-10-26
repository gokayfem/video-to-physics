import sys, os, json, numpy as np, cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__)); from geom import *
W = os.path.join(ROOT, 'measurement/work'); OUT = os.path.join(ROOT, 'measurement'); V = os.path.join(OUT, 'visuals')
st = np.load(os.path.join(W, 'rect_stack.npy'), mmap_mode='r'); d = np.load(os.path.join(W, 'analysis.npz')); J = json.load(open(os.path.join(W, 'analysis.json')))
t = d['t']; xs = d['xs']; ys = d['ys']; U = 0.015; D = 0.01; XC = -0.04
def lab(img, s, scale=1.0):
    cv2.rectangle(img, (0, 0), (img.shape[1], int(34 * scale)), (0, 0, 0), -1)
    cv2.putText(img, s, (8, int(25 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * scale, (255, 255, 255), 2, cv2.LINE_AA); return img
# 1. rectified strip t=0..10 s
c0 = int((-0.10 + 0.3) / RES); c1 = int((0.262 + 0.3) / RES)
rows = []
for s_ in range(11):
    i = min(s_ * 24, 240); im = np.ascontiguousarray(st[i][40:440, c0:c1]).copy()
    for xm in range(-90, 261, 10):   # tick marks every 10 mm, D units labelled at cylinder
        c = int((xm / 1000 + 0.3) / RES) - c0; cv2.line(im, (c, 392), (c, 400 if xm % 50 else 380), (255, 255, 255), 1)
    cv2.putText(im, f't = {s_} s (f{i})', (10, 30), 0, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    xf = d['front'][i]
    if not np.isnan(xf): c = int((xf + 0.3) / RES) - c0; cv2.line(im, (c, 0), (c, 399), (0, 255, 0), 2)
    xp = XC + 0.6 * D + 1.5 * U * t[i]; c = int((xp + 0.3) / RES) - c0
    if c < im.shape[1]: cv2.line(im, (c, 0), (c, 399), (0, 0, 255), 2)
    rows.append(cv2.resize(im, None, fx=0.5, fy=0.5))
strip = np.vstack(rows); hdr = np.zeros((40, strip.shape[1], 3), np.uint8)
strip = np.vstack([lab(hdr, 'v001 | rectified dye plane z=13mm, x -100..+262mm, |y|<50mm, 0.5mm/px | measured   green=dye front  red=x0+1.5U t (phys. max)', 1.0), strip])
cv2.imwrite(os.path.join(V, 'v001_rectified_strip_t0-10s_measured.jpg'), strip, [cv2.IMWRITE_JPEG_QUALITY, 90])
# 2. front plot
fr = d['front']; x0 = fr[0]; tsat = J['t_sat']
fig, ax = plt.subplots(figsize=(9, 5))
m = t < tsat
ax.plot(t[m], fr[m] * 1000, 'k.', ms=3, label='AI dye front (source-connected, measured)')
ax.plot(t[~m], fr[~m] * 1000, '.', c='0.7', ms=3, label='after front left view / far wake fades (not a front)')
for k, c, l in ((1.5, 'r', 'x0 + 1.5U t  (max local speed, phys. upper bound)'), (1.0, 'orange', 'x0 + U t'), (0.85, 'g', 'x0 + 0.85U t (vortex carriage)')):
    ax.plot(t, (x0 + k * U * t) * 1000, c, label=l)
ax.axhline(J['xmax_vis'] * 1000, ls=':', c='b', label='edge of visible dye plane')
pe = np.polyfit(t[m & (t > 0.2)], fr[m & (t > 0.2)], 1)
ax.plot(t[m], np.polyval(pe, t[m]) * 1000, 'k--', lw=0.8, label=f'fit: {pe[0]*100:.2f} cm/s = {pe[0]/U:.1f} U')
ax.set_xlabel('t [s]'); ax.set_ylabel('x_front [mm] (cylinder at -40)'); ax.grid(alpha=.3); ax.legend(fontsize=8, loc='lower right')
ax.set_title('v001 | dye front x_front(t) vs physical bounds (U=1.5 cm/s) | measured', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(V, 'v001_dye_front_vs_bounds_measured.png'), dpi=130); plt.close(fig)
# 3. probe + spectrum
fig, ax = plt.subplots(2, 1, figsize=(9, 6.5))
for k, c in ((3, 'C0'), (4, 'C1'), (5, 'C2')):
    ax[0].plot(t, d[f'probe{k}_asym'], c, lw=1, label=f'x = cyl + {k}D')
ax[0].axvspan(0, 2.5, color='0.9'); ax[0].set_xlabel('t [s]'); ax[0].set_ylabel('dye cover upper - lower half'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[0].set_title('v001 | wake probe (dye asymmetry, window ±0.5D) | measured; grey = transient excluded from spectrum', fontsize=9)
F = d['F4'] / d['F4'].max(); f = d['f4']
ax[1].plot(f, F, 'C1', label=f'probe 4D, t>=2.5 s: peak {J["fs4"]:.2f} Hz')
fphys = 0.18 * U / D; ax[1].axvline(fphys, c='r', ls='--', label=f'physical St=0.18 at U=1.5cm/s: {fphys:.2f} Hz')
ax[1].set_xlim(0, 3); ax[1].set_xlabel('f [Hz]'); ax[1].set_ylabel('|FFT| (norm.)'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(V, 'v001_wake_probe_signal_spectrum_measured.png'), dpi=130); plt.close(fig)
# 4. first roll-up annotated (frame 18 original + rectified crop sheet)
fr_u8 = np.load(os.path.join(W, 'frames_u8.npy'), mmap_mode='r')
im = np.asarray(fr_u8[18]).copy()
def P(X): u, v = project(X); return int(u - .5), int(v - .5)
cv2.ellipse(im, P([-0.003, -0.016, 0.013]), (55, 35), -20, 0, 360, (0, 255, 0), 3)
cv2.putText(im, 'FIRST ROLL-UP: lower side (y<0, near wall), t=0.75 s (f18)', (P([-0.003, -0.016, 0.013])[0] - 200, P([-0.003, -0.016, 0.013])[1] + 90), 0, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
cv2.ellipse(im, P([0.0, 0.016, 0.013]), (55, 35), -20, 0, 360, (0, 200, 255), 2)
cv2.putText(im, 'upper side (y>0): hook only, rolls up at t=0.92 s (f22)', (P([0.0, 0.016, 0.013])[0] - 150, P([0.0, 0.016, 0.013])[1] - 60), 0, 0.9, (0, 200, 255), 2, cv2.LINE_AA)
for X in ([-0.04, 0, 0.003],): cv2.drawMarker(im, P(X), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
im = lab(im, 'v001 | f18 t=0.75 s original frame (+y = far wall = image up-left) | measured', 1.4)
rs = cv2.imread(os.path.join(W, 'rollup_sheet.jpg')); rs = cv2.resize(rs, (1920, int(rs.shape[0] * 1920 / rs.shape[1])))
cv2.imwrite(os.path.join(V, 'v001_first_rollup_annotated_measured.jpg'), np.vstack([im, lab(rs, 'v001 | rectified near wake f8..f26 (top = +y) | measured', 1.4)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
# 5. x-t diagram + vortex tracks
vt = np.loadtxt(os.path.join(OUT, 'vortex_detections.csv'), delimiter=',', skiprows=1)
fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
for a_, Z, nm in ((ax[0], d['A'], 'upper half (y>0)'), (ax[1], d['B'], 'lower half (y<0)')):
    a_.imshow(Z, aspect='auto', origin='lower', extent=[xs[0] * 1000, xs[-1] * 1000, 0, 10], cmap='magma', vmax=0.5)
    sel = vt[:, 3] > 0 if 'upper' in nm else vt[:, 3] < 0
    a_.plot(vt[sel, 2] * 1000, vt[sel, 1], 'c.', ms=1.5, label='blob maxima')
    for k, c in ((1.0, 'w'), (0.85, 'lime')):
        for x0_ in np.arange(-40, 260, 55): a_.plot([x0_, x0_ + k * U * 1000 * 10], [0, 10], c, lw=0.6, alpha=.6)
    a_.set_xlim(-100, 262); a_.set_xlabel('x [mm]'); a_.set_title(nm + ' — lines: 0.85U (lime) / U (white) slopes', fontsize=9)
ax[0].set_ylabel('t [s]'); fig.suptitle(f'v001 | x-t dye coverage + vortex blobs | measured (vortex speed {J["trk"]["med"]*100:.1f} cm/s vs 0.85U = 1.3 cm/s)', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(V, 'v001_xt_diagram_vortex_tracks_measured.png'), dpi=120); plt.close(fig)
# 6. camera stability
cs = np.loadtxt(os.path.join(OUT, 'camera_stability.csv'), delimiter=',', skiprows=1)
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(cs[:, 1], cs[:, 3], label='max over 4 corner probes'); ax.plot(cs[:, 1], cs[:, 4], label='image centre'); ax.plot(cs[:, 1], cs[:, 5], label='cylinder base')
ax.set_xlabel('t [s]'); ax.set_ylabel('displacement vs frame 0 [px]'); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.set_title('v001 | camera drift (ECC homography on static regions) | measured', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(V, 'v001_camera_drift_measured.png'), dpi=130); plt.close(fig)
# 7. defects: colour budget + cylinder crops
cf = d['cf']
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for j, nm in enumerate(('upstream x<-46mm', 'near wake -30..+30mm', 'far wake >+30mm')):
    for k, (c, l) in enumerate((('m', 'magenta'), ('c', 'cyan'), ('y', 'yellow'))):
        ax[j].plot(t, cf[:, 4 * j + k], c, label=l)
    ax[j].set_title(nm, fontsize=9); ax[j].set_xlabel('t [s]'); ax[j].grid(alpha=.3)
ax[0].axhline(3 / 7, c='m', ls=':'); ax[0].axhline(2 / 7, c='k', ls=':'); ax[0].set_ylabel('area fraction of dye colour'); ax[0].legend(fontsize=8)
fig.suptitle('v001 | dye colour budget per region (needles feed 3/7 magenta, 2/7 cyan, 2/7 yellow) | measured', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(V, 'v001_colour_budget_measured.png'), dpi=120); plt.close(fig)
Hs = np.load(os.path.join(OUT, 'stabilisation_H_frame0_to_n.npy')); crops = []
for i in (0, 60, 120, 180, 240):
    im = cv2.warpPerspective(np.asarray(fr_u8[i]), np.linalg.inv(Hs[i]), (1920, 1088))[380:720, 380:900].copy()
    cv2.putText(im, f'f{i}', (8, 30), 0, 1, (255, 255, 255), 2); crops.append(im)
cv2.imwrite(os.path.join(V, 'v001_cylinder_needles_constancy_measured.jpg'), np.vstack([lab(np.zeros((40, 2600, 3), np.uint8), 'v001 | stabilised crops of cylinder + needles f0,60,120,180,240 | measured'), np.hstack(crops)]))
print('ok')
