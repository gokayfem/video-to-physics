"""(base of v004 renderer, see lagr4.py) v002 Lagrangian dye (streaklines) from the solver's velocity history.
Particles are released continuously from each needle's source disk (7 cross-stream material lines spanning
+-r_s = 0.35 mm, one particle per line per substep dt = 1/48 s) and advected with RK4 (dt = 1/48 s) in the
0.5 mm / (1/24 s) velocity history (bilinear in space, linear in time). This is the forward equivalent of the
backward criterion "a pixel is dye k if its back-trajectory passes the source disk of needle k after t_rel".
Each frame: material lines rasterised (anti-aliased, 1/16 px precision) as c = 1 inside the streak, union per
colour, Gaussian blur sigma = sqrt(0.5^2 + 2 D_m age_max / px^2) px (D_m = 1e-9 m^2/s).
t_rel chosen so the outer streaks' (y = +-16, +-24 mm) particle front is at x_cyl + 0.6 D at t = 0.
Usage: lagr.py RUN_DIR T0_ND  -> RUN_DIR/lag/dye-NNNN.u16 + RUN_DIR/lag/lag_meta.json"""
import sys, os, glob, json, numpy as np, cv2

RUN, T0 = sys.argv[1], float(sys.argv[2])
NX, NY, PX = 2400, 480, 0.025            # D units (0.25 mm)
TS = 0.01 / 0.015
DT = 1.0 / 32.0                           # nondim substep = 1/48 s
files = sorted(glob.glob(RUN + "/vel/v-*.f32"))
tv = np.array([float(os.path.basename(f)[2:-4]) for f in files])
print("velocity history", tv[0], tv[-1], len(tv))
cache = {}
NVX = os.path.getsize(files[0]) // (240 * 2 * 4)          # 800 (x in [-10,30] D) or 1200 (full channel)
VX0 = -10.0 if NVX == 800 else -30.0


def vel(k):
    if k not in cache:
        if len(cache) > 6:
            cache.pop(min(cache))
        cache[k] = np.fromfile(files[k], np.float32).reshape(240, NVX, 2)
    return cache[k]


def bilin(V, x, y):
    fi = (x - VX0) / 0.05 - 0.5
    fj = (6.0 - y) / 0.05 - 0.5
    fi = np.clip(fi, 0, NVX - 1.001); fj = np.clip(fj, 0, 238.999)
    i0 = fi.astype(int); j0 = fj.astype(int); a = fi - i0; b = fj - j0
    out = (V[j0, i0] * ((1 - a) * (1 - b))[:, None] + V[j0, i0 + 1] * (a * (1 - b))[:, None]
           + V[j0 + 1, i0] * ((1 - a) * b)[:, None] + V[j0 + 1, i0 + 1] * (a * b)[:, None])
    out[(x < VX0) | (x > 30)] = 0.0
    return out


def u_at(t, P):
    k = int(np.clip(np.searchsorted(tv, t, side="right") - 1, 0, len(tv) - 2))
    w = (t - tv[k]) / (tv[k + 1] - tv[k])
    return (1 - w) * bilin(vel(k), P[:, 0], P[:, 1]) + w * bilin(vel(k + 1), P[:, 0], P[:, 1])


def rk4(t, P):
    k1 = u_at(t, P); k2 = u_at(t + DT / 2, P + DT / 2 * k1)
    k3 = u_at(t + DT / 2, P + DT / 2 * k2); k4 = u_at(t + DT, P + DT * k3)
    return P + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


XN, YN = -9.3, np.array([-2.4, -1.6, -0.8, 0.0, 0.8, 1.6, 2.4])
COL = [0, 1, 2, 0, 1, 2, 0]
RS = 0.035
OFF = np.linspace(-RS, RS, 7)
GAPPX = 6.0   # 1.5 mm

# ---- 1. choose t_rel: probe particles from the outer needles released on the substep grid
tr_c = np.arange(np.ceil(tv[0] / DT) * DT, T0 - 2.0, DT)
outer = [0, 1, 5, 6]
P = np.array([[XN, YN[k]] for tr in tr_c for k in outer]); rel = np.repeat(tr_c, len(outer))
t = tr_c[0]
while t < T0 - 1e-9:
    act = rel <= t + 1e-9
    P[act] = rk4(t, P[act]); t += DT
fx = P[:, 0].reshape(len(tr_c), len(outer)).mean(axis=1)      # mean front of the 4 outer streaks per release time
print("probe fronts", tr_c[::8], fx[::8]); ok = np.where(np.diff(np.sign(fx + 3.4)))[0]
k = ok[-1]
tr_exact = tr_c[k] + (-3.4 - fx[k]) * (tr_c[k + 1] - tr_c[k]) / (fx[k + 1] - fx[k])
TREL = DT * np.floor(tr_exact / DT + 0.5)
print("t_rel (nd)", TREL, "exact", tr_exact, "-> t_rel - t0 =", (TREL - T0) * TS, "s")

# ---- 2. v004 streaklines: mass-conserving Gaussian streaks
# One material line per needle (its centreline), NSUB particles released per substep.  Segment stretch lam = ds/ds0
# (ds0 = separation at release).  Cross-section Gaussian: sig^2 = sig0^2/lam^2 + 2 D_m age + sig_aa^2,
# peak c = sig_e0/(lam*sig) with sig_e0 = sqrt(sig0^2 + sig_aa^2)  ->  c*sig = sig_e0/lam (mass per unit material length conserved).
# Rendering: each segment is sampled every <= 0.25 px; each sample deposits w = c*h*sqrt(2 pi)*sig (px units) bilinearly into
# a sigma-bin grid; each bin is blurred with its sigma (log bins, 0.3-12 px); overlapping streaks add per tracer; clip at 1.
NSUB = int(os.environ.get("NSUB", 12))
SIG_AA = 0.5                                     # px
SIG_E0 = float(os.environ.get("SIG_E0", 1.955))  # px: matches v003 upstream streak (peak 1, cross-section integral 4.9 px)
SIG0 = np.sqrt(SIG_E0 ** 2 - SIG_AA ** 2)        # px (physical unstretched streak, 0.47 mm)
DM = 1e-9
GAP = float(os.environ.get("GAPMM", 3.0)) * 1e-3 / 2.5e-4   # px: segments longer than this are not drawn (mass reported)
BINS = np.exp(np.linspace(np.log(0.3), np.log(12.0), 60))
nsteps = int(round((T0 + float(os.environ.get('TWIN', 15.0)) - TREL) / DT))
os.makedirs(RUN + "/lag4", exist_ok=True)
xs = -30 + (np.arange(NX) + 0.5) * PX; ys = 6 - (np.arange(NY) + 0.5) * PX
SOLID = (xs[None, :] + 4.0) ** 2 + ys[:, None] ** 2 <= 0.25
seeds = np.array([[XN, YN[k]] for k in range(7)]); sid = np.arange(7)
P = np.zeros((0, 2)); lid = np.zeros(0, int); trel_p = np.zeros(0); ds0 = np.zeros(0)
t = TREL; fronts = {}; budget = {}
for s in range(nsteps + 1):
    un = u_at(t, seeds); sp = np.hypot(un[:, 0], un[:, 1]) * DT / NSUB    # release spacing (D units)
    for j in range(NSUB - 1, -1, -1):
        P = np.vstack([P, seeds + un * (j * DT / NSUB)]); lid = np.concatenate([lid, sid])
        trel_p = np.concatenate([trel_p, np.full(7, t - j * DT / NSUB)]); ds0 = np.concatenate([ds0, sp])
    n = (t - T0) * 16.0
    if n > -1e-6 and abs(n - round(n)) < 1e-6 and round(n) <= 240:
        n = int(round(n))
        grids = np.zeros((3, len(BINS), NY + 2, NX + 2), np.float32)
        m_inj = m_drawn = m_drop = m_out = 0.0
        for L in range(7):
            mk = lid == L
            q = P[mk]; a0 = trel_p[mk]; s0 = ds0[mk]
            qx = (q[:, 0] + 30.0) / PX - 0.5; qy = (6.0 - q[:, 1]) / PX - 0.5      # px coords
            dsx, dsy = np.diff(qx), np.diff(qy); ds = np.hypot(dsx, dsy)
            s0p = 0.5 * (s0[1:] + s0[:-1]) / PX                                        # px
            lam = np.maximum(ds / s0p, 1e-3)
            age = (t - 0.5 * (a0[1:] + a0[:-1])) * TS
            sig = np.sqrt(SIG0 ** 2 / lam ** 2 + 2 * DM * age / (2.5e-4) ** 2 + SIG_AA ** 2)
            c = SIG_E0 / (lam * sig)
            mseg = np.sqrt(2 * np.pi) * SIG_E0 * s0p                                   # conserved mass per segment (px^2)
            m_inj += mseg.sum()
            xm = 0.5 * (qx[1:] + qx[:-1])
            out = xm > NX - 1; m_out += mseg[out].sum()
            ok = (ds <= GAP) & ~out
            m_drop += mseg[~ok & ~out].sum()
            idx = np.where(ok)[0]
            if len(idx) == 0: continue
            ns = np.maximum(1, np.ceil(ds[idx] / 0.25)).astype(int)
            rep = np.repeat(idx, ns)
            fr = (np.concatenate([np.arange(k_) for k_ in ns]) + 0.5) / np.repeat(ns, ns)
            X = qx[rep] + fr * dsx[rep]; Y = qy[rep] + fr * dsy[rep]
            h = ds[rep] / np.repeat(ns, ns)
            w = c[rep] * h * np.sqrt(2 * np.pi) * sig[rep]
            m_drawn += w.sum()
            b = np.clip(np.round((np.log(sig[rep]) - np.log(BINS[0])) / np.log(BINS[1] / BINS[0])).astype(int), 0, len(BINS) - 1)
            x0 = np.floor(X).astype(int); y0 = np.floor(Y).astype(int); fx = X - x0; fy = Y - y0
            g = grids[COL[L]]
            for dx_, dy_, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                xi = np.clip(x0 + dx_ + 1, 0, NX + 1); yi = np.clip(y0 + dy_ + 1, 0, NY + 1)
                np.add.at(g, (b, yi, xi), (w * ww).astype(np.float32))
        img = np.zeros((3, NY, NX), np.float32)
        for cch in range(3):
            for bi, sg in enumerate(BINS):
                G = grids[cch, bi]
                if G.any():
                    img[cch] += cv2.GaussianBlur(G, (0, 0), float(sg))[1:-1, 1:-1]
        mass_img_unclipped = float(img.sum())
        img = np.clip(img, 0, 1)
        out16 = (np.transpose(img, (1, 2, 0)) * 65535 + 0.5).astype(np.uint16)
        out16[SOLID] = 0
        out16.tofile(RUN + f"/lag4/dye-{n:04d}.u16")
        mass_png = float(out16.astype(np.float64).sum() / 65535)
        oo = np.isin(lid, [0, 1, 5, 6])
        fronts[n] = float(P[oo, 0].max() * 0.01)
        budget[n] = dict(injected=m_inj, outflow=m_out, dropped_gap=m_drop, drawn=m_drawn, image_unclipped=mass_img_unclipped,
                         png=mass_png, rel_err_png_vs_injected_minus_out=mass_png / (m_inj - m_out) - 1)
        if n % 24 == 0:
            print("frame", n, "particles", len(P), {k_: round(v_, 4) if isinstance(v_, float) else v_ for k_, v_ in budget[n].items()}, flush=True)
    if s < nsteps:
        P = rk4(t, P); t += DT
json.dump(dict(t_rel_nd=float(TREL), t_rel_s=float((TREL - T0) * TS), t0_nd=T0,
               front_definition="max x of particles on the outer streaks (needles y=+-16, +-24 mm), mean over the 4 for t_rel",
               front_outer_x_m_by_frame=fronts, renderer="v004 mass-conserving Gaussian streaks",
               formula="sig^2 = sig0^2/lam^2 + 2 D_m age + sig_aa^2 ; c = sig_e0/(lam sig), sig_e0 = sqrt(sig0^2+sig_aa^2); lam = ds/ds0 per segment",
               sig0_m=SIG0 * 2.5e-4, sig_aa_px=SIG_AA, sig_e0_m=SIG_E0 * 2.5e-4, D_m=DM, particles_per_substep=NSUB, dt_substep_s=DT * TS,
               gap_m=GAP * 2.5e-4, mass_budget_px2=budget), open(RUN + "/lag4/lag_meta.json", "w"), indent=1)
print("done")
