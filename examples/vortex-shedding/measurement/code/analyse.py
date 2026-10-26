import sys, os, json, numpy as np, cv2
from scipy import ndimage as ndi, signal
sys.path.insert(0, os.path.dirname(__file__)); from geom import *
W = os.path.join(ROOT, 'measurement/work'); OUT = os.path.join(ROOT, 'measurement')
st = np.load(os.path.join(W, 'rect_stack.npy'), mmap_mode='r'); val = np.load(os.path.join(W, 'rect_valid.npy'))
N = len(st); fps = 24.; t = np.arange(N) / fps
xs = -0.30 + RES * (np.arange(st.shape[2]) + 0.5); ys = 0.06 - RES * (np.arange(st.shape[1]) + 0.5)
band = (np.abs(ys) < 0.046)[:, None] & val
X_CYL, D, U = -0.04, 0.01, 0.015
TH = 30
_a0 = np.asarray(st[0]).astype(np.int16); _c0 = (_a0.max(2) - _a0.min(2)).astype(np.uint8)
BGC = cv2.GaussianBlur(cv2.medianBlur(_c0, 31), (0, 0), 6).astype(np.int16)   # dye-free background chroma (streaks removed by median)
def dyemask(im):
    a = im.astype(np.int16); ch = a.max(2) - a.min(2)
    return (ch - BGC > TH) & band
def colours(im):
    b, g, r = [im[..., k].astype(np.int16) for k in range(3)]
    mag = (r > g + 40) & (b > g + 10) & (r >= b - 10)
    cya = (b > r + 40) & (g > r + 20)
    yel = (r > b + 50) & (g > b + 40)
    return mag, cya, yel
cols_ok = val[np.abs(ys) < 0.05].mean(0) > 0.6     # columns mostly visible
xmax_vis = xs[cols_ok].max()
M = np.zeros((N,) + val.shape, bool)
for i in range(N): M[i] = dyemask(st[i])
cov = M.sum(1) / np.maximum(band.sum(0), 1)          # (N, nx) column coverage
covs = ndi.uniform_filter1d(cov, 8, axis=1)
# dye front: furthest x (visible) with smoothed coverage > 1.5 %, among columns reachable continuously from the cylinder
front = np.full(N, np.nan); gaps = []
for i in range(N):
    on = (covs[i] > 0.015) & cols_ok & (xs > -0.09)
    idx = np.where(on)[0]
    if not len(idx): continue
    # continuous run starting upstream
    front[i] = xs[idx.max()]
saturated = front >= xmax_vis - 0.003
x0 = front[0]
# physical bound lines
bound_max = x0 + 1.5 * U * t; bound_carry = x0 + 0.85 * U * t; bound_U = x0 + U * t
# speed of front (linear fit over unsaturated part after 0.25 s)
sel = (~saturated) & (t > 0.2) & ~np.isnan(front)
sel_early = sel & (t <= 1.5)
p_front = np.polyfit(t[sel], front[sel], 1); p_early = np.polyfit(t[sel_early], front[sel_early], 1); res = front[sel] - np.polyval(p_front, t[sel])
# bootstrap-ish uncertainty from fit residual
se = np.sqrt(np.sum(res ** 2) / (sel.sum() - 2) / np.sum((t[sel] - t[sel].mean()) ** 2))
t_sat = t[saturated][0] if saturated.any() else None
np.savetxt(os.path.join(OUT, 'dye_front.csv'), np.c_[np.arange(N), t, front, saturated, bound_U, bound_max],
           delimiter=',', header='frame,t_s,x_front_m,at_visible_edge,x0+U*t,x0+1.5U*t', fmt='%.5f', comments='')
# ---------- x-t diagrams (upper / lower half coverage)
up = (ys > 0.003) & (ys < 0.05); lo = (ys < -0.003) & (ys > -0.05)
A = np.array([M[i][up].sum(0) for i in range(N)], float) / np.maximum(band[up].sum(0), 1)
B = np.array([M[i][lo].sum(0) for i in range(N)], float) / np.maximum(band[lo].sum(0), 1)
# wake probe: lateral dye centroid at x = X_CYL + 4D (window +-0.5D); also signed asymmetry A-B at 3,4,5 D
def probe(xc, half=0.005):
    c = (xs > xc - half) & (xs < xc + half)
    yc = np.array([(M[i][:, c].sum(1) * ys).sum() / max(M[i][:, c].sum(), 1) for i in range(N)])
    asym = A[:, c].mean(1) - B[:, c].mean(1)
    return yc, asym
pr = {k: probe(X_CYL + k * D) for k in (2, 3, 4, 5, 6, 8)}
# spectrum: steady part t >= 2.5 s, probe 4D asymmetry
def spec(sig, tmin=2.5):
    s = sig[t >= tmin] - sig[t >= tmin].mean(); n = len(s)
    w = np.hanning(n); nf = 8 * n
    F = np.abs(np.fft.rfft(s * w, nf)); f = np.fft.rfftfreq(nf, 1 / fps)
    k = np.argmax(F[f > 0.08]) + np.sum(f <= 0.08)
    return f, F, f[k]
fpk = {k: spec(pr[k][1])[2] for k in pr}
f4, F4, fs4 = spec(pr[4][1])
# alternative: per-cycle periods from zero crossings of the 4D asymmetry after 2.5 s
s4 = pr[4][1] - pr[4][1][t >= 2.5].mean()
s4s = ndi.gaussian_filter1d(s4, 1.5)
zc = np.where((s4s[:-1] < 0) & (s4s[1:] >= 0) & (t[:-1] >= 2.5))[0]
tz = t[zc] + (-s4s[zc]) / (s4s[zc + 1] - s4s[zc]) / fps
periods = np.diff(tz)
# ---------- convection speed: x-shift maximising correlation between frames i and i+k in the developed wake region
def shift_speed(Z, k=6, x_lo=-0.0, x_hi=None):
    x_hi = x_hi or xmax_vis - 0.01
    c = (xs > x_lo) & (xs < x_hi); sp = []
    for i in range(int(2.5 * fps), N - k):
        a = Z[i, c] - Z[i, c].mean(); b = Z[i + k, c] - Z[i + k, c].mean()
        lags = np.arange(-40, 161)
        cc = [np.sum(a[max(0, -l):len(a) - max(0, l)] * b[max(0, l):len(b) - max(0, -l)]) / (len(a) - abs(l)) for l in lags]
        cc = np.array(cc); j = np.argmax(cc)
        if 0 < j < len(cc) - 1:
            d = 0.5 * (cc[j - 1] - cc[j + 1]) / (cc[j - 1] - 2 * cc[j] + cc[j + 1]); sp.append((lags[j] + d) * RES * fps / k)
    return np.array(sp)
spA = shift_speed(A); spB = shift_speed(B)
vconv = np.r_[spA, spB]
# near wake vs far wake convection
spA_near = shift_speed(A, x_lo=-0.03, x_hi=0.04); spA_far = shift_speed(A, x_lo=0.06)
# ---------- spatial wavelength: FFT of A(x) and B(x) at late frames over visible developed wake
def wavelen(Z, x_lo=-0.01):
    c = (xs > x_lo) & (xs < xmax_vis - 0.005); L = []
    for i in range(int(4 * fps), N, 6):
        s = Z[i, c] - Z[i, c].mean(); n = len(s); nf = 16 * n
        F = np.abs(np.fft.rfft(s * np.hanning(n), nf)); k = np.fft.rfftfreq(nf, RES)
        m = (k > 1 / 0.15) & (k < 1 / 0.015); L.append(1 / k[m][np.argmax(F[m])])
    return np.array(L)
lamA = wavelen(A); lamB = wavelen(B)
# ---------- vortex detection (blurred dye coverage maxima) in frames t>=3 s
vort = []
for i in range(0, N, 1):
    m = ndi.gaussian_filter(M[i].astype(float), 10)        # sigma 2.5 mm
    mx = ndi.maximum_filter(m, size=60)
    pk = np.argwhere((m == mx) & (m > 0.25) & band)
    for r, c in pk:
        if xs[c] < X_CYL + 1.5 * D or xs[c] > xmax_vis - 0.008: continue
        lab, _ = ndi.label(m > 0.6 * m[r, c]); comp = lab == lab[r, c]
        dia = 2 * np.sqrt(comp.sum() / np.pi) * RES
        vort.append((i, t[i], xs[c], ys[r], m[r, c], dia))
vort = np.array(vort)
np.savetxt(os.path.join(OUT, 'vortex_detections.csv'), vort, delimiter=',', header='frame,t_s,x_m,y_m,peak_cov,eq_diam_m(60%peak)', fmt='%.5f', comments='')
late = vort[vort[:, 1] >= 3.0]
yU = late[late[:, 3] > 0, 3]; yL = late[late[:, 3] < 0, 3]
# linking for tracks -> convection speed of cores
tracks = []; active = []
for i in range(N):
    cur = vort[vort[:, 0] == i]; used = set(); new_active = []
    for tr in active:
        last = tr[-1]; best = None; bd = 1e9
        for j, v in enumerate(cur):
            if j in used: continue
            dx = v[2] - last[2]; dy = v[3] - last[3]
            if -0.002 < dx < 0.010 and abs(dy) < 0.004 and np.hypot(dx, dy) < bd: best, bd = j, np.hypot(dx, dy)
        if best is not None: used.add(best); tr.append(cur[best]); new_active.append(tr)
        else: tracks.append(tr)
    for j, v in enumerate(cur):
        if j not in used: new_active.append([v])
    active = new_active
tracks += active
trk_speeds = []; trk_back = 0
for tr in tracks:
    tr = np.array(tr)
    if len(tr) >= 12:
        pf = np.polyfit(tr[:, 1], tr[:, 2], 1); trk_speeds.append((pf[0], tr[:, 2].mean(), tr[:, 1].min()))
        if pf[0] < -0.002: trk_back += 1
trk_speeds = np.array(trk_speeds)
# ---------- symmetry index in near wake (x in [xcyl+1D, xcyl+5D]): corr of dye mask with its y-mirror
c_nw = (xs > X_CYL + 1 * D) & (xs < X_CYL + 5 * D); ny = len(ys); mid = np.argmin(np.abs(ys))
sym = []
for i in range(N):
    m = M[i][:, c_nw].astype(float); mm = m[::-1]
    a = m - m.mean(); b = mm - mm.mean(); sym.append((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))
sym = np.array(sym)
# ---------- colour fractions per region over time
cf = []
for i in range(N):
    mg, cy, ye = colours(st[i]); rows = []
    for (xa, xb) in ((-0.09, X_CYL - 0.006), (X_CYL + 0.01, 0.03), (0.03, xmax_vis)):
        c = (xs > xa) & (xs < xb); sel_ = band[:, c]
        n_m, n_c, n_y = mg[:, c][sel_].sum(), cy[:, c][sel_].sum(), ye[:, c][sel_].sum(); tot = max(n_m + n_c + n_y, 1)
        rows += [n_m / tot, n_c / tot, n_y / tot, tot]
    cf.append(rows)
cf = np.array(cf)
# ---------- upstream streak steadiness (y positions at x=-0.07 per frame)
cst = int((-0.07 + 0.3) / RES)
def streak_ys(im):
    a = im.astype(int); ch = (a.max(2) - a.min(2))[:, cst - 4:cst + 4].mean(1)
    p, _ = signal.find_peaks(ch, height=60, distance=15); return 0.06 - RES * (p + 0.5)
sy = [streak_ys(st[i]) for i in range(N)]
n7 = np.array([len(s) == 7 for s in sy]); sy7 = np.array([s for s in sy if len(s) == 7])
streak_std_mm = sy7.std(0) * 1000 if len(sy7) else None
streak_drift_mm = (sy7[-1] - sy7[0]) * 1000 if len(sy7) else None
# ---------- static object constancy and flicker (stabilised image space)
fr = np.load(os.path.join(W, 'frames_u8.npy'), mmap_mode='r'); Hs = np.load(os.path.join(OUT, 'stabilisation_H_frame0_to_n.npy'))
cyl_top = (slice(445, 540), slice(735, 805)); rake = (slice(380, 900), slice(240, 600)); bgr = (slice(900, 1080), slice(1200, 1900))
obj = []
f0 = np.asarray(fr[0]).astype(float)
for i in range(N):
    im = cv2.warpPerspective(np.asarray(fr[i]), np.linalg.inv(Hs[i]), (1920, 1088), flags=cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS).astype(float)
    obj.append([np.abs(im[cyl_top] - f0[cyl_top]).mean(), np.abs(im[rake] - f0[rake]).mean(), im[bgr].mean(), np.asarray(fr[i])[60:1000, 100:1800].mean()])
obj = np.array(obj)
dyediff = np.array([0] + [np.mean(st[i][band].astype(np.int16) != st[i - 1][band].astype(np.int16)) for i in range(1, N)])
np.savez(os.path.join(W, 'analysis.npz'), t=t, xs=xs, ys=ys, front=front, A=A, B=B, cov=cov, sym=sym, cf=cf, obj=obj, dyediff=dyediff,
         f4=f4, F4=F4, **{f'probe{k}_yc': pr[k][0] for k in pr}, **{f'probe{k}_asym': pr[k][1] for k in pr}, vconv=vconv, tz=tz)
res_d = dict(p_early=p_early.tolist(), xmax_vis=xmax_vis, x0=x0, p_front=p_front.tolist(), se=se, t_sat=t_sat, gaps=gaps[:20], n_gaps=len(gaps),
             fpk=fpk, fs4=fs4, periods=periods.tolist(), vA=[np.median(spA), np.percentile(spA, 16), np.percentile(spA, 84)],
             vB=[np.median(spB), np.percentile(spB, 16), np.percentile(spB, 84)], vA_near=float(np.median(spA_near)), vA_far=float(np.median(spA_far)),
             lamA=[np.median(lamA), lamA.std()], lamB=[np.median(lamB), lamB.std()],
             yU=[np.median(yU), yU.std(), len(yU)], yL=[np.median(yL), yL.std(), len(yL)],
             dia=[np.median(late[:, 5]), np.percentile(late[:, 5], 16), np.percentile(late[:, 5], 84)],
             trk=dict(n=len(trk_speeds), med=float(np.median(trk_speeds[:, 0])) if len(trk_speeds) else None,
                      p16=float(np.percentile(trk_speeds[:, 0], 16)) if len(trk_speeds) else None, p84=float(np.percentile(trk_speeds[:, 0], 84)) if len(trk_speeds) else None, back=trk_back),
             sym=dict(t=t[::6].tolist(), v=sym[::6].round(3).tolist()),
             cf_up=cf[:, 0:3].mean(0).tolist(), cf_near=cf[24:, 4:7].mean(0).tolist(), cf_far=cf[72:, 8:11].mean(0).tolist(),
             streak_std_mm=None if streak_std_mm is None else streak_std_mm.round(3).tolist(), streak_drift_mm=None if streak_drift_mm is None else streak_drift_mm.round(3).tolist(), n7_frac=n7.mean(),
             cyl_diff=[obj[:, 0].mean(), obj[:, 0].max()], rake_diff=[obj[:, 1].mean(), obj[:, 1].max()],
             bg_mean=[obj[:, 2].min(), obj[:, 2].max(), np.abs(np.diff(obj[:, 2])).max()], glob=[obj[:, 3].min(), obj[:, 3].max(), np.abs(np.diff(obj[:, 3])).max()],
             dyediff_min=float(dyediff[1:].min()), dup_frames=int((dyediff[1:] < 0.02).sum()))
json.dump(res_d, open(os.path.join(W, 'analysis.json'), 'w'), indent=1, default=float)
print(json.dumps(res_d, indent=1, default=float)[:6000])
