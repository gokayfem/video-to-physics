"""Role E production-output checks (independent of ROOT/solver code). Reads solver/out per CONTRACT interface:
  dye/fNNNN.png  16-bit RGB, R,G,B = tracer 0,1,2 in [0,1] (x in [-0.30,0.30], y in [-0.06,0.06], row 0 = +y, col 0 = -x)
  fields.npz     per-frame u, v, omega on 1 mm grid + probe series Cd(t), Cl(t), wake v(t)
  meta.json      params, St, Cd, Cl, ...
Key names are matched tolerantly (see ALIASES); everything is in SI.
"""
import json, os, glob
import numpy as np
import cv2
from scipy.ndimage import map_coordinates

ALIASES = {
    'u': ['u', 'ux', 'U'], 'v': ['v', 'uy', 'V'], 'omega': ['omega', 'w', 'vort', 'vorticity'],
    'x': ['x', 'xg', 'x_grid', 'xs'], 'y': ['y', 'yg', 'y_grid', 'ys'], 't': ['t', 't_frames', 'time', 'frame_t'],
    'Cd': ['Cd', 'cd', 'Cd_t', 'ts_Cd'], 'Cl': ['Cl', 'cl', 'Cl_t', 'ts_Cl'], 'tp': ['ts_t', 't_probe', 'tp', 't_series', 't_forces', 'probe_t'],
    'pv': ['ts_probe_v', 'probe_v', 'wake_v', 'pv', 'v_probe', 'wake_probe_v'],
}
GEOM = dict(U=0.015, D=0.01, xc=-0.04, yc=0.0, W=0.12, x0=-0.30, x1=0.30, x_tip=-0.093,
            needles_y=[-0.024, -0.016, -0.008, 0.0, 0.008, 0.016, 0.024], fps=24.0)


def _get(d, k):
    for a in ALIASES[k]:
        if a in d: return np.asarray(d[a])
    return None


def load(outdir):
    meta = json.load(open(os.path.join(outdir, 'meta.json'))) if os.path.exists(os.path.join(outdir, 'meta.json')) else {}
    z = np.load(os.path.join(outdir, 'fields.npz'), allow_pickle=True)
    d = {k: z[k] for k in z.files}
    F = {k: _get(d, k) for k in ALIASES}
    nf = F['u'].shape[0]
    if F['x'] is None: F['x'] = np.arange(-0.30, 0.30 + 1e-9, 0.001)
    if F['y'] is None: F['y'] = np.arange(-0.06, 0.06 + 1e-9, 0.001)
    if F['u'].shape[1:] == (len(F['x']), len(F['y'])) and len(F['x']) != len(F['y']):
        for k in ('u', 'v', 'omega'):
            if F[k] is not None: F[k] = np.transpose(F[k], (0, 2, 1))
    if F['y'][0] > F['y'][-1]:   # make y ascending
        F['y'] = F['y'][::-1]
        for k in ('u', 'v', 'omega'):
            if F[k] is not None: F[k] = F[k][:, ::-1, :]
    if F['t'] is None: F['t'] = np.arange(nf) / GEOM['fps']
    if F['tp'] is None and F['Cl'] is not None: F['tp'] = np.linspace(F['t'][0], F['t'][-1], len(F['Cl']))
    P = meta.get('params', {})
    if F['tp'] is not None and F['tp'][0] < F['t'][0] - 1e-6:   # probe series includes precursor: keep the frame window only
        w = F['tp'] >= F['t'][0] - 1e-9
        for k in ('tp', 'Cd', 'Cl', 'pv'):
            if F[k] is not None: F[k] = F[k][w]
    U = float(meta.get('U') or P.get('U_ref') or P.get('U') or GEOM['U'])
    F['U'] = U; F['meta'] = meta; F['keys'] = list(d.keys())
    F['dye_files'] = sorted(glob.glob(os.path.join(outdir, 'dye', 'f*.png')))
    return F


def dye_frame(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    im = im[..., ::-1].astype(np.float32)   # BGR on disk -> RGB
    return im / (255.0 if im.max() <= 255 and cv2.imread(path, cv2.IMREAD_UNCHANGED).dtype == np.uint8 else 65535.0)


def dye_grid(shape):
    """pixel-centre coordinates for a dye image of shape (H,W)"""
    H, W = shape[:2]
    px = 0.6 / (W - 1) if (W - 1) % 10 == 0 else 0.6 / W
    if (W - 1) % 10 == 0:
        xs = -0.30 + np.arange(W) * px; ys = 0.06 - np.arange(H) * px
    else:
        xs = -0.30 + (np.arange(W) + 0.5) * px; ys = 0.06 - (np.arange(H) + 0.5) * px
    return xs, ys, px


def ndye(F):
    return len(F['dye']) if F.get('dye') is not None else len(F['dye_files'])


def getdye(F, n):
    return F['dye'][n] if F.get('dye') is not None else dye_frame(F['dye_files'][n])


# ---------------------------------------------------------------- helpers
def zero_cross_freq(t, s):
    s = s - np.mean(s)
    i = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    if len(i) < 2: return np.nan
    tc = t[i] - s[i] * (t[i + 1] - t[i]) / (s[i + 1] - s[i])
    return (len(tc) - 1) / (tc[-1] - tc[0])


def fft_freq(t, s):
    s = s - np.mean(s); n = len(s); dt = np.median(np.diff(t))
    N = 16 * n; S = np.abs(np.fft.rfft(s * np.hanning(n), N)); f = np.fft.rfftfreq(N, dt)
    k = np.argmax(S[1:]) + 1
    return f[k]


def lowpass(t, s, fc):
    from scipy.signal import butter, filtfilt
    dt = np.median(np.diff(t)); nyq = 0.5 / dt
    if fc >= nyq * 0.95: return np.asarray(s, float)
    b, a = butter(4, fc / nyq); return filtfilt(b, a, s)


def ls_freq(t, s, f0):
    """least-squares fit of a*sin+b*cos+c over a frequency grid (unbiased for short, non-integer-cycle windows)"""
    fs = np.linspace(0.7 * f0, 1.3 * f0, 1201); best = (np.inf, f0)
    for f in fs:
        A = np.c_[np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t), np.ones_like(t)]
        r = s - A @ np.linalg.lstsq(A, s, rcond=None)[0]; e = r @ r
        if e < best[0]: best = (e, f)
    return best[1]


def robust_St(t, s, D, U):
    """FFT peak -> low-pass at 2.5 f (removes remeshing spikes) -> LS sinusoid frequency fit"""
    f0 = fft_freq(t, s); sf = lowpass(t, s, 2.5 * f0)
    return ls_freq(t, sf, f0) * D / U, sf


def scan_freq(t, s, f0):
    s = s - np.mean(s); fs = np.linspace(0.5 * f0, 1.5 * f0, 2001)
    P = [abs(np.sum(s * np.exp(-2j * np.pi * f * t))) for f in fs]
    return fs[int(np.argmax(P))]


def interp_vel(F, i_float, X, Y):
    """bilinear u,v at points (X,Y) at fractional frame index"""
    x, y = F['x'], F['y']; dx = x[1] - x[0]; dy = y[1] - y[0]
    ci = (X - x[0]) / dx; ri = (Y - y[0]) / dy
    i0 = int(np.clip(np.floor(i_float), 0, len(F['t']) - 2)); w = i_float - i0
    out = []
    for k in ('u', 'v'):
        a0 = map_coordinates(F[k][i0], [ri, ci], order=1, mode='nearest')
        a1 = map_coordinates(F[k][i0 + 1], [ri, ci], order=1, mode='nearest')
        out.append((1 - w) * a0 + w * a1)
    return out


def res(name, ok, value, tol, note=''):
    return dict(check=name, passed=bool(ok), value=value, tol=tol, note=note)


# ---------------------------------------------------------------- checks
def c_divergence(F, tol_rms=0.05):
    x, y = F['x'], F['y']; dx = x[1] - x[0]
    XX, YY = np.meshgrid(x, y)
    r = np.hypot(XX - GEOM['xc'], YY - GEOM['yc'])
    m = (r > GEOM['D'] / 2 + 3 * dx) & (np.abs(YY) < GEOM['W'] / 2 - 3 * dx) & (XX > x[0] + 2 * dx) & (XX < x[-1] - 2 * dx)
    rd, ro, mx = [], [], []
    for i in range(0, len(F['t']), max(1, len(F['t']) // 40)):
        du = np.gradient(F['u'][i], dx, axis=1); dv = np.gradient(F['v'][i], y[1] - y[0], axis=0)
        om = np.gradient(F['v'][i], dx, axis=1) - np.gradient(F['u'][i], y[1] - y[0], axis=0)
        dvg = (du + dv)[m]; rd.append(np.mean(dvg ** 2)); ro.append(np.mean(om[m] ** 2)); mx.append(np.max(np.abs(dvg)))
    ratio = float(np.sqrt(np.mean(rd) / np.mean(ro)))
    return res('divergence_rms_over_vorticity_rms', ratio <= tol_rms, ratio, tol_rms,
               f'max|div| = {max(mx):.3g} 1/s, rms|omega| = {np.sqrt(np.mean(ro)):.3g} 1/s (central diff, 1 mm grid, ring of 3 dx excluded)')


def c_noslip(F, tol=0.05):
    x, y = F['x'], F['y']; dx = x[1] - x[0]
    XX, YY = np.meshgrid(x, y); r = np.hypot(XX - GEOM['xc'], YY - GEOM['yc'])
    inside = r < GEOM['D'] / 2 - 2 * dx      # solid core (cut cells of a ~1 mm solver grid excluded)
    sp = np.hypot(F['u'], F['v'])
    cyl = float(np.max(sp[:, inside])) / F['U'] if inside.any() else np.nan
    wall_rows = [j for j in range(len(y)) if abs(abs(y[j]) - GEOM['W'] / 2) < 0.26 * dx]
    wall = float(np.max(sp[:, wall_rows, :])) / F['U'] if wall_rows else np.nan
    ok = (np.isnan(cyl) or cyl <= tol) and (np.isnan(wall) or wall <= tol)
    return res('noslip_max_speed_over_U', ok, dict(cylinder_interior=cyl, walls=wall), tol,
               'solid-core points r < R - 2 mm and grid rows on y = +-W/2 (if present)')


def c_strouhal(F, St_ref, tol_agree=0.02, tol_ref=0.02, twin=None):
    D, U = GEOM['D'], F['U']
    tp, cl, pv = F['tp'], F['Cl'], F['pv']
    sel = slice(None) if twin is None else (tp >= twin[0]) & (tp <= twin[1])
    St_cl, _ = robust_St(tp[sel], cl[sel], D, U)
    St_cl_fft = fft_freq(tp[sel], cl[sel]) * D / U
    St_pv = robust_St(tp[sel], pv[sel], D, U)[0] if pv is not None else np.nan
    # St from the field itself (v on the wake centreline 3 D behind the cylinder)
    j = np.argmin(np.abs(F['y'] - GEOM['yc'])); i = np.argmin(np.abs(F['x'] - (GEOM['xc'] + 3 * D)))
    St_fld = ls_freq(F['t'], F['v'][:, j, i] - F['v'][:, j, i].mean(), St_cl * U / D) * D / U
    agree = abs(St_cl - St_pv) / St_cl if np.isfinite(St_pv) else abs(St_cl - St_fld) / St_cl
    agree_f = abs(St_cl - St_fld) / St_cl
    dref = abs(St_cl - St_ref) / St_ref
    return [res('St_Cl_vs_probe', agree <= tol_agree, dict(St_Cl=St_cl, St_Cl_fft=St_cl_fft, St_probe=St_pv, rel=agree), tol_agree),
            res('St_Cl_vs_field_v(5D wake)', agree_f <= 0.05, dict(St_field=St_fld, rel=agree_f), 0.05,
                'periodogram of centreline v at xc+5D (window ~3 cycles: 5 % tolerance)'),
            res('St_vs_reference', dref <= tol_ref, dict(St=St_cl, St_ref=St_ref, rel=dref), tol_ref)]


def c_forces(F, Cd_ref, Clamp_ref, tol_cd=0.05, tol_cl=0.10):
    f0 = fft_freq(F['tp'], F['Cl'])
    cd = float(np.mean(lowpass(F['tp'], F['Cd'], 5 * f0))); clf = lowpass(F['tp'], F['Cl'], 2.5 * f0)
    cla = float((np.max(clf) - np.min(clf)) / 2)
    spikes = float(np.max(np.abs(F['Cl'] - clf)))
    return [res('Cd_mean_vs_reference', abs(cd - Cd_ref) / Cd_ref <= tol_cd, dict(Cd=cd, Cd_ref=Cd_ref, rel=abs(cd - Cd_ref) / Cd_ref), tol_cd),
            res('Cl_amp_vs_reference', abs(cla - Clamp_ref) / Clamp_ref <= tol_cl, dict(Cl_amp=cla, ref=Clamp_ref, rel=abs(cla - Clamp_ref) / Clamp_ref,
                Cl_max_raw_spike=spikes), tol_cl, 'Cl low-passed at 2.5 f_shed (raw spikes reported separately)')]


def wake_ke(F):
    """fluctuation KE 0.5<(u-ubar)^2+(v-vbar)^2> in the wake box (ubar = time mean over the window), per frame"""
    x = F['x']; D = GEOM['D']
    mx = (x > GEOM['xc'] + D) & (x < GEOM['xc'] + 16 * D)
    ub = F['u'][:, :, mx].mean(axis=0); vb = F['v'][:, :, mx].mean(axis=0)
    return np.array([0.5 * np.mean((F['u'][i][:, mx] - ub) ** 2 + (F['v'][i][:, mx] - vb) ** 2) for i in range(len(F['t']))])


def c_energy(F, St, tol=0.03):
    """wake perturbation KE (box 1..16 D behind cylinder) must be periodic: no drift between whole periods"""
    ke = wake_ke(F); t = F['t']; T = GEOM['D'] / (St * F['U'])
    nper = int((t[-1] - t[0]) // T)
    if nper < 2:
        p = np.polyfit(t, ke, 1); drift = p[0] * (t[-1] - t[0]) / np.mean(ke); per = []
    else:
        per = [np.mean(ke[(t >= t[0] + k * T) & (t < t[0] + (k + 1) * T)]) for k in range(nper)]
        drift = (per[-1] - per[0]) / np.mean(per)
    return res('wake_KE_periodic_no_drift', abs(drift) <= tol, dict(rel_drift=float(drift), period_means=[float(v) for v in per]), tol,
               'fluctuation KE (about the window mean) over x in [xc+D, xc+16D]; first vs last whole shedding period')


def c_convection(F, St, lo=0.82, hi=0.93):
    """phase speed of v on the wake centreline from space-time cross-correlation (|lag| < T/2)"""
    x = F['x']; t = F['t']; D = GEOM['D']; U = F['U']
    j = np.argmin(np.abs(F['y'] - GEOM['yc']))
    T = D / (St * U); dt = t[1] - t[0]
    base = GEOM['xc'] + 4 * D
    dxs, lags = [], []
    for dxn in (0.5, 1.0, 1.5, 2.0):
        for b0 in (base, base + 3 * D, base + 6 * D):
            i1 = np.argmin(np.abs(x - b0)); i2 = np.argmin(np.abs(x - (b0 + dxn * D)))
            s1 = F['v'][:, j, i1] - F['v'][:, j, i1].mean(); s2 = F['v'][:, j, i2] - F['v'][:, j, i2].mean()
            maxlag = int(0.5 * T / dt)
            ks = np.arange(-maxlag, maxlag + 1)
            cc = np.array([np.sum(s1[max(0, -k):len(s1) - max(0, k)] * s2[max(0, k):len(s2) - max(0, -k)]) for k in ks])
            m = np.argmax(cc)
            if 0 < m < len(ks) - 1:
                d = 0.5 * (cc[m - 1] - cc[m + 1]) / (cc[m - 1] - 2 * cc[m] + cc[m + 1])
            else:
                d = 0
            lags.append((ks[m] + d) * dt); dxs.append(x[i2] - x[i1])
    dxs = np.array(dxs); lags = np.array(lags)
    c = float(np.sum(dxs * lags) / np.sum(lags ** 2)) if np.sum(lags ** 2) > 0 else np.nan
    ratio = c / U
    return res('vortex_convection_speed_over_U', lo <= ratio <= hi, ratio, [lo, hi], 'centreline v phase speed, 12 pairs, LS fit dx = c*lag')


def c_tracer(F, every=1, up_tol=0.003, c_tol=0.02):
    files = range(ndye(F))
    if not files: return [res('tracer_present', False, 0, '>0 frames')]
    c0 = getdye(F, 0); xs, ys, px = dye_grid(c0.shape)
    upstream = xs < GEOM['x_tip'] - up_tol
    cmin, cmax, upmax, upext, mass, front, outflux, fspeed = 1, 0, 0, -1, [], [], [], []
    # velocity at the dye outlet column for the outflow flux
    tt = np.arange(len(files)) / GEOM['fps']
    for n in range(0, len(files), every):
        c = getdye(F, n); s = c.sum(axis=2)
        cmin = min(cmin, float(c.min())); cmax = max(cmax, float(c.max()))
        upmax = max(upmax, float(c[:, upstream].max()))
        colsu = np.where((s > c_tol).any(axis=0))[0]
        if len(colsu): upext = max(upext, GEOM['x_tip'] - xs[colsu.min()])
        mass.append(c.sum(axis=(0, 1)) * px * px)
        cols = np.where((s > 0.1).any(axis=0))[0]
        front.append(xs[cols.max()] if len(cols) else np.nan)
        # outflow flux through last column: sum c*u dy
        i_f = min(n * len(F['t']) / len(files), len(F['t']) - 1.001)
        uo, _ = interp_vel(F, i_f, np.full(len(ys), xs[-1]), ys)
        outflux.append((c[:, -1, :] * np.clip(uo, 0, None)[:, None]).sum(axis=0) * px)
    mass = np.array(mass); outflux = np.array(outflux); front = np.array(front); tt = tt[::every]
    dtf = tt[1] - tt[0]
    # injected estimate I(t) = M(t) + int outflow; must grow linearly (constant sources)
    I = mass + np.concatenate([np.zeros((1, 3)), np.cumsum(0.5 * (outflux[1:] + outflux[:-1]) * dtf, axis=0)])
    lin = []
    for k in range(3):
        if I[-1, k] - I[0, k] <= 0: lin.append(1.0); continue
        p = np.polyfit(tt, I[:, k], 1); r = I[:, k] - np.polyval(p, tt)
        lin.append(float(np.max(np.abs(r)) / (I[-1, k] - I[0, k])))
    # front speed vs the max velocity magnitude in the field
    ok_f = np.isfinite(front) & (front < xs[-1] - 0.005)
    umax = float(np.max(np.hypot(F['u'], F['v'])))
    if ok_f.sum() > 12:
        k = 6; fs = (front[k:] - front[:-k]) / (k * dtf); fs = fs[ok_f[k:] & ok_f[:-k]]
        fsmax = float(np.max(fs)) if len(fs) else np.nan
    else:
        fsmax = np.nan
    return [res('tracer_bounds', cmin >= -c_tol and cmax <= 1 + c_tol, dict(min=cmin, max=cmax), [-c_tol, 1 + c_tol]),
            res('no_dye_upstream_of_needles', upext <= up_tol, dict(upstream_extent_m=upext, max_c_beyond_tol=upmax), up_tol,
                f'extent of c > {c_tol} upstream of the needle tips must be <= {up_tol*1e3:.1f} mm (source radius + ~2 solver cells)'),
            res('dye_front_speed_le_max_velocity', (not np.isfinite(fsmax)) or fsmax <= 1.05 * umax,
                dict(front_speed_max=fsmax, umax=umax, ratio=fsmax / umax if np.isfinite(fsmax) else None), '<= 1.05 umax', '6-frame differences'),
            res('tracer_mass_budget_linear', max(lin) <= 0.05, dict(max_dev_frac=lin), 0.05,
                'I(t) = mass(t) + cumulative outflow must be linear (constant needle sources); max deviation / total injected')]


def c_dye_advection(F, n_pairs=8, lag_frames=8, scales=(0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)):
    """advect dye-edge points of frame n with the frames' (u,v) (RK2) to frame n+lag; best-fit velocity scale must be ~1"""
    nF = ndye(F)
    rng = np.random.default_rng(0)
    errs = np.zeros(len(scales)); errs0 = 0; npts = 0; npairs = 0
    starts = np.linspace(0, nF - lag_frames - 1, n_pairs).astype(int)
    for n in starts:
        c = getdye(F, n).sum(axis=2); c2 = getdye(F, n + lag_frames).sum(axis=2)
        xs, ys, px = dye_grid(c.shape)
        gy, gx = np.gradient(c)
        g = np.hypot(gx, gy)
        cand = np.argwhere((g > 0.02) & (c > 0.1) & (np.abs(c2 - c) > 0.1))   # dye that moved (unsteady edges, fronts)
        cand = cand[(xs[cand[:, 1]] > GEOM['x_tip'] + 0.01) & (xs[cand[:, 1]] < 0.2) & (np.hypot(xs[cand[:, 1]] - GEOM['xc'], ys[cand[:, 0]]) > 0.008)]
        if len(cand) < 20: continue
        npts += len(cand); npairs += 1
        cand = cand[rng.choice(len(cand), min(1500, len(cand)), replace=False)]
        X0 = xs[cand[:, 1]]; Y0 = ys[cand[:, 0]]; cv = c[cand[:, 0], cand[:, 1]]
        nfld = len(F['t'])
        for si, s in enumerate(scales):
            X, Y = X0.copy(), Y0.copy(); sub = 4; h = 1.0 / GEOM['fps'] / sub
            for k in range(lag_frames * sub):
                fi = (n + k / sub) * (nfld - 1) / (nF - 1)
                u1, v1 = interp_vel(F, fi, X, Y)
                fi2 = (n + (k + 0.5) / sub) * (nfld - 1) / (nF - 1)
                u2, v2 = interp_vel(F, fi2, X + 0.5 * h * s * u1, Y + 0.5 * h * s * v1)
                X += h * s * u2; Y += h * s * v2
            cj = (X - xs[0]) / (xs[1] - xs[0]); ri = (Y - ys[0]) / (ys[1] - ys[0])
            ca = map_coordinates(c2, [ri, cj], order=1, mode='nearest')
            errs[si] += np.mean(np.abs(ca - cv))
        errs0 += np.mean(np.abs(c2[cand[:, 0], cand[:, 1]] - cv))
    if npairs == 0:
        return res('dye_advected_by_velocity_scale', False, dict(best_scale=None, n_points=0), [0.85, 1.15], 'FAIL-SAFE: no moving dye points found')
    k = int(np.argmin(errs))
    if 0 < k < len(errs) - 1:   # parabolic refinement in scale
        s0, s1, s2 = scales[k - 1:k + 2]; e0, e1, e2 = errs[k - 1:k + 2]
        A = np.polyfit([s0, s1, s2], [e0, e1, e2], 2); best = float(-A[1] / (2 * A[0])) if A[0] > 0 else scales[k]
    else:
        best = float(scales[k])
    return res('dye_advected_by_velocity_scale', 0.85 <= best <= 1.15, dict(best_scale=best, err_by_scale=dict(zip(map(str, scales), (errs / max(1, n_pairs)).round(4).tolist())),
               err_no_motion=errs0 / max(1, n_pairs), n_points=int(npts), n_pairs=npairs), [0.85, 1.15],
               f'moving dye points (|dc|>0.2 over the lag) advected RK2 over {lag_frames} frames with s*(u,v); best s should be 1')


def run_all(outdir, ref, fast=False, F=None):
    F = load(outdir) if F is None else F
    St_ref, Cd_ref, Cl_ref = ref['St'], ref['Cd_mean'], ref['Cl_amp']
    R = [c_divergence(F), c_noslip(F)]
    R += c_strouhal(F, St_ref)
    R += c_forces(F, Cd_ref, Cl_ref)
    St_here = R[2]['value']['St_Cl']
    R.append(c_energy(F, St_here))
    R.append(c_convection(F, St_here))
    R += c_tracer(F, every=2 if fast else 1)
    R.append(c_dye_advection(F))
    return dict(outdir=outdir, keys=F['keys'], n_frames=len(F['t']), n_dye=ndye(F),
                all_passed=all(r['passed'] for r in R), results=R)
