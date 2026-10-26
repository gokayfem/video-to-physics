"""v002 Lagrangian dye (streaklines) from the solver's velocity history.
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

# ---- 2. streaklines
nsteps = int(round((T0 + float(os.environ.get('TWIN', 15.0)) - TREL) / DT))
lines = [[] for _ in range(7 * len(OFF))]          # per material line: list of particle arrays (release order)
P = np.zeros((0, 2)); lid = np.zeros(0, int); age0 = np.zeros(0)
os.makedirs(RUN + "/lag", exist_ok=True)
SOLID = None
xs = -30 + (np.arange(NX) + 0.5) * PX; ys = 6 - (np.arange(NY) + 0.5) * PX
SOLID = (xs[None, :] + 4.0) ** 2 + ys[:, None] ** 2 <= 0.25
fronts = {}
t = TREL
seeds = np.array([[XN + 0.0, YN[k] + o] for k in range(7) for o in OFF])
sid = np.arange(len(seeds))
for s in range(nsteps + 1):
    # release a new particle on every material line
    # 4 particles per material line per substep: sub-step release positions advanced by the local needle velocity
    un = u_at(t, seeds)
    for j in (3, 2, 1, 0):
        P = np.vstack([P, seeds + un * (j * DT / 4)]); lid = np.concatenate([lid, sid])
        age0 = np.concatenate([age0, np.full(len(sid), t - j * DT / 4)])
    n = (t - T0) * 16.0
    if n > -1e-6 and abs(n - round(n)) < 1e-6 and round(n) <= 240:
        n = int(round(n))
        img = np.zeros((3, NY, NX), np.float32)
        for L in range(len(seeds)):
            m = lid == L
            q = P[m]                                   # oldest first
            px = ((q[:, 0] + 30.0) / PX - 0.5) * 16; py = ((6.0 - q[:, 1]) / PX - 0.5) * 16
            pts = np.stack([px, py], 1).round().astype(np.int32)
            # break the material line where neighbours separated by > 1.5 mm (e.g. split at the stagnation point)
            seg = np.hypot(np.diff(px), np.diff(py)) / 16.0 > GAPPX
            cut = np.where(seg)[0] + 1
            parts = [q_ for q_ in np.split(pts, cut) if len(q_) > 1]
            canv = np.zeros((NY, NX), np.uint8)
            cv2.polylines(canv, [q_.reshape(-1, 1, 2) for q_ in parts], False, 255, thickness=1, lineType=cv2.LINE_AA, shift=4)
            c = COL[L // len(OFF)]
            img[c] = np.maximum(img[c], canv / 255.0)
        agemax = (t - TREL) * TS
        sig = np.sqrt(0.25 + 2e-9 * agemax / (2.5e-4) ** 2)
        for c in range(3):
            img[c] = np.clip(cv2.GaussianBlur(img[c], (0, 0), sig) * 1.0, 0, 1)
            # concentration 1 inside the streak: renormalise the blurred union so a streak core reads 1
            img[c] = np.clip(img[c] / 0.6, 0, 1)
        out = (np.transpose(img, (1, 2, 0)) * 65535 + 0.5).astype(np.uint16)
        out[SOLID] = 0
        out.tofile(RUN + f"/lag/dye-{n:04d}.u16")
        oo = np.isin(lid // len(OFF), outer)
        fronts[n] = float(P[oo, 0].max() * 0.01)
        if n % 24 == 0:
            print("frame", n, "particles", len(P), "outer front x(m)", fronts[n], flush=True)
    if s < nsteps:
        P = rk4(t, P); t += DT
json.dump(dict(t_rel_nd=float(TREL), t_rel_s=float((TREL - T0) * TS), t_rel_exact_nd=float(tr_exact), t0_nd=T0,
               front_definition="max x of particles on the outer streaks (needles y=+-16, +-24 mm), mean over the 4 for t_rel",
               front_outer_x_m_by_frame=fronts, dt_substep_s=DT * TS, r_s_m=RS * 0.01, lines_per_needle=len(OFF), particles_per_line_per_substep=4, line_break_gap_m=GAPPX*2.5e-4,
               blur="sigma = sqrt(0.5^2 + 2 D_m age_max/px^2) px, D_m=1e-9; core normalised (/0.6, clip 1)",
               vel_history=[float(tv[0]), float(tv[-1])]), open(RUN + "/lag/lag_meta.json", "w"), indent=1)
print("done")
