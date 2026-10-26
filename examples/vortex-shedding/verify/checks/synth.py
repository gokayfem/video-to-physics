"""Synthetic production-like output (kinematic Karman street, exactly divergence-free via a streamfunction) for testing the
checks and for negative controls. Not a physics model; only used to exercise vlib.
defects: None | 'st10' | 'dye15' | 'diverge' | 'upstream' | 'energy'"""
import numpy as np
from scipy.ndimage import map_coordinates
from vlib import GEOM

U, D, xc, h = GEOM['U'], GEOM['D'], GEOM['xc'], GEOM['W'] / 2


def make(St=0.19, Cd=1.41, Clamp=0.55, defect=None, nframes=241, dye_px=0.0005, t_spin=4.0, Uc_ratio=0.87):
    fps = GEOM['fps']; f = St * U / D
    x = np.round(np.arange(-0.30, 0.30 + 1e-9, 0.001), 6); y = np.round(np.arange(-0.06, 0.06 + 1e-9, 0.001), 6)
    XX, YY = np.meshgrid(x, y); dx = 0.001
    a = D / 2; sig = 0.3 * D; b = 0.0065
    taper = (1 - (YY / h) ** 2) ** 2
    r2 = np.maximum((XX - xc) ** 2 + YY ** 2, a * a)
    psi_base = 1.5 * U * (YY - YY ** 3 / (3 * h * h)) - U * a * a * YY / r2 * taper
    A0 = 0.45 * U * sig / 0.6
    Uc = Uc_ratio * U * (-1 if defect == 'upstream' else 1)
    xb = xc + 1.5 * D if Uc > 0 else 0.30

    def field(t):
        psi = psi_base.copy()
        amp = A0 * ((1 + 0.6 * t / 10.0) if (defect == 'energy' and t > 0) else 1.0)
        for k in range(int((t + 60) * 2 * f) + 1):
            tk = -60 + k / (2 * f)
            if tk > t: break
            xk = xb + Uc * (t - tk)
            if xk < xc + 1.5 * D - 1e-9 or xk > 0.32: continue
            ramp = min(1.0, (t - tk) * f * 2)
            sgn = -1 if k % 2 == 0 else 1
            yk = b if sgn < 0 else -b
            psi += sgn * amp * ramp * np.exp(-((XX - xk) ** 2 + (YY - yk) ** 2) / (2 * sig ** 2)) * taper
        u = np.gradient(psi, dx, axis=0); v = -np.gradient(psi, dx, axis=1)
        if defect == 'diverge':
            u = u * (1 + 0.3 * np.sin(2 * np.pi * XX / 0.03))
        inside = (XX - xc) ** 2 + YY ** 2 < a * a
        u[inside] = 0; v[inside] = 0
        return u.astype(np.float32), v.astype(np.float32)

    t = np.arange(nframes) / fps
    us, vs = zip(*[field(tt) for tt in t])
    u = np.array(us); v = np.array(vs)
    om = np.gradient(v, dx, axis=2) - np.gradient(u, dx, axis=1)
    tp = np.arange(0, t[-1] + 1e-9, 0.01)
    fp = f * (1.1 if defect == 'st10' else 1.0)
    Cl = Clamp * np.sin(2 * np.pi * fp * tp); Cdt = Cd + 0.03 * np.sin(4 * np.pi * fp * tp)
    pv = 0.6 * U * np.sin(2 * np.pi * fp * tp + 0.7)
    F = dict(x=x, y=y, t=t, u=u, v=v, omega=om.astype(np.float32), tp=tp, Cl=Cl, Cd=Cdt, pv=pv, U=U, meta={}, keys=['synthetic'])

    # dye: semi-Lagrangian RK2 on a dye_px grid with velocity bilinearly interpolated from the 1 mm grid
    s_dye = 1.5 if defect == 'dye15' else 1.0
    Wd = int(round(0.6 / dye_px)) + 1; Hd = int(round(0.12 / dye_px)) + 1
    xs = -0.30 + np.arange(Wd) * dye_px; ys = 0.06 - np.arange(Hd) * dye_px
    DX, DY = np.meshgrid(xs, ys)
    c = np.zeros((3, Hd, Wd), np.float32)
    blobs = []
    for i, yn in enumerate(GEOM['needles_y']):
        blobs.append(([0, 1, 2, 0, 1, 2, 0][i], np.exp(-((DX - GEOM['x_tip']) ** 2 + (DY - yn) ** 2) / (2 * 0.0004 ** 2))))
    sub = 2; hstep = 1 / fps / sub
    nsp = int(round(t_spin / s_dye * fps))   # same dye-front position at t = 0 for all variants
    cache = {}

    def vel(tt, X, Y):
        key = round(tt, 6)
        if key not in cache:
            if len(cache) > 4: cache.clear()
            cache[key] = field(tt)
        uu, vv = cache[key]
        ri = (Y - y[0]) / dx; ci = (X - x[0]) / dx
        return (map_coordinates(uu, [ri, ci], order=1, mode='nearest') * s_dye,
                map_coordinates(vv, [ri, ci], order=1, mode='nearest') * s_dye)

    frames = []
    for n in range(-nsp, nframes):
        if n >= 0:
            frames.append(np.clip(np.transpose(c, (1, 2, 0)), 0, 1).copy())
        for k in range(sub):
            tt = n / fps + k * hstep
            u1, v1 = vel(round(tt * 48) / 48, DX, DY)
            u2, v2 = vel(round(tt * 48) / 48, DX - 0.5 * hstep * u1, DY - 0.5 * hstep * v1)
            Xb = DX - hstep * u2; Yb = DY - hstep * v2
            cj = (Xb - xs[0]) / dye_px; ri = (ys[0] - Yb) / dye_px
            for ch in range(3):
                c[ch] = map_coordinates(c[ch], [ri, cj], order=1, mode='constant', cval=0.0)
            for ch, bl in blobs:
                c[ch] = np.maximum(c[ch], bl)
    F['dye'] = frames
    return F


def write(F, outdir):
    import os, json, cv2
    os.makedirs(os.path.join(outdir, 'dye'), exist_ok=True)
    for n, fr in enumerate(F['dye']):
        cv2.imwrite(os.path.join(outdir, 'dye', f'f{n:04d}.png'), (fr[..., ::-1] * 65535 + 0.5).astype(np.uint16))
    np.savez(os.path.join(outdir, 'fields.npz'), x=F['x'], y=F['y'], t=F['t'], u=F['u'], v=F['v'], omega=F['omega'],
             t_probe=F['tp'], Cd=F['Cd'], Cl=F['Cl'], probe_v=F['pv'])
    json.dump(dict(U=F['U'], synthetic=True), open(os.path.join(outdir, 'meta.json'), 'w'))
