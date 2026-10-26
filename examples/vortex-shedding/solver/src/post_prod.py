"""Production post-processing: PNG dye frames, fields.npz, meta.json, params/physics.json, visuals.
Usage: python post_prod.py FINEST_LEVEL [other levels...]"""
import json, sys, os, hashlib, glob, time, subprocess
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/..") + "/"
ROOT = os.path.abspath(R + "..") + "/"
D, U, NU = 0.01, 0.015, 1e-6
TS = D / U                       # 0.6667 s per nondim time unit
levels = [int(a) for a in sys.argv[1:]] or [11, 10, 9]
F = levels[0]
NX, NY = 2400, 480


def load(L):
    d = np.loadtxt(R + f"runs/L{L}/forces.dat")
    _, k = np.unique(d[:, 0], return_index=True)   # restarts append: keep unique t
    d = d[k]
    ev = open(R + f"runs/L{L}/events.txt").read()
    plan = [l for l in ev.splitlines() if l.startswith("plan")][-1]
    kv = dict(p.split("=") for p in plan.replace("plan: ", "").split() if "=" in p)
    t0, trel = float(kv["target_t0"]), float(kv["t_rel"])
    fr = [l for l in ev.splitlines() if l.startswith("front reached")]
    tfront = float(fr[-1].split("at t=")[1].split()[0]) if fr else None
    rst = [l for l in ev.splitlines() if l.startswith("restart")]
    return d, t0, trel, tfront, rst


def shedding(t, cl, cd, ta, tb):
    m = (t >= ta) & (t <= tb)
    tt, cc = t[m], cl[m]
    k = np.where((cc[:-1] < 0) & (cc[1:] >= 0))[0]
    tc = tt[k] - cc[k] * (tt[k + 1] - tt[k]) / (cc[k + 1] - cc[k])
    if len(tc) < 2:
        return None
    T = np.diff(tc).mean()
    mm = (t >= tc[0]) & (t <= tc[-1])
    amps = []
    for a, b in zip(tc[:-1], tc[1:]):
        q = (t >= a) & (t <= b)
        amps.append(0.5 * np.ptp(cl[q]))
    tm, cm_ = t[mm], cl[mm]; w0 = 2 * np.pi / T
    A = np.c_[np.sin(w0 * tm), np.cos(w0 * tm), np.ones_like(tm), np.sin(3 * w0 * tm), np.cos(3 * w0 * tm)]
    co = np.linalg.lstsq(A, cm_, rcond=None)[0]
    fit_amp = float(np.hypot(co[0], co[1]))
    env = [a_ for a_ in amps]
    return dict(Cl_amp_fit_fundamental=fit_amp, Cl_envelope_change_per_period=float(np.max(np.abs(np.diff(env)) / np.mean(env))) if len(env) > 1 else None,
                St=1.0 / T, T_s=T * TS, f_Hz=1 / (T * TS), Cd_mean=float(cd[mm].mean()),
                Cl_amp=float(np.mean(amps)), Cl_amp_spread=float(np.ptp(amps) / np.mean(amps)) if len(amps) > 1 else 0.,
                Cd_amp=float(0.5 * np.ptp(cd[mm])), crossings=tc.tolist(), n_periods=len(tc) - 1)


_px = -0.30 + (np.arange(NX) + 0.5) * 2.5e-4
_py = 0.06 - (np.arange(NY) + 0.5) * 2.5e-4
SOLID = (_px[None, :] + 0.04) ** 2 + _py[:, None] ** 2 <= 0.005 ** 2


VER = os.environ.get("VER", "v001")
DYE_SRC = os.environ.get("DYE_SRC", "raw")


def read_eul(L, n):
    return read_dye(L, n, "raw")


def read_dye(L, n, src=None):
    if src is None:
        src = DYE_SRC if L == F else "raw"
    f = R + f"runs/L{L}/{src}/dye-{n:04d}.u16"
    if not os.path.exists(f):
        return None
    c = np.fromfile(f, dtype=np.uint16).reshape(NY, NX, 3).copy()
    c[SOLID] = 0          # inside the cylinder = 0
    return c


def read_uvw(L, n):
    return np.fromfile(R + f"runs/L{L}/raw/uvw-{n:04d}.f32", dtype=np.float32).reshape(120, 600, 3)


def composite(c):
    """subtractive dye on white: magenta, cyan, yellow"""
    c = c.astype(np.float32) / 65535.0
    cols = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], np.float32)  # RGB
    img = np.ones(c.shape[:2] + (3,), np.float32)
    for k in range(3):
        img *= 1 - c[..., k:k + 1] * (1 - cols[k])
    return img


res = {}
for L in levels:
    try:
        d, t0, trel, tfront, rst = load(L)
    except Exception as e:
        print("skip", L, e); continue
    t, cd, cl, vp = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    win = shedding(t, cl, cd, t0, t0 + 15)
    pre = shedding(t, cl, cd, trel - 30, trel)
    # phase: Cl ~ sin(2 pi f (t - t_cross_up)); phi0 at t0
    allc = shedding(t, cl, cd, trel - 30, t0 + 15)
    tc = np.array(allc["crossings"]); tcb = tc[tc <= t0].max(); T = 1 / allc["St"]
    phi0 = (2 * np.pi * (t0 - tcb) / T) % (2 * np.pi)
    res[L] = dict(level=L, dx_mm=600 / 2**L, cells_per_D=2**L / 60, t0_nd=t0, t_rel_nd=trel,
                  t_rel_s=(trel - t0) * TS, t_front_nd=tfront,
                  front_travel_s=(tfront - trel) * TS if tfront else None,
                  window=win, precursor_last=pre, phase_phi0_rad=phi0,
                  Cl_at_t0=float(np.interp(t0, t, cl)), restarts=rst,
                  t_total_nd=float(t[-1]), cells_end=int(d[-1, 5]))
    print(L, json.dumps({k: v for k, v in res[L].items() if k not in ("window", "precursor_last")}, default=str))
    print("  window", {k: v for k, v in win.items() if k != "crossings"})

# dye convergence metric vs finest
for L in levels[1:]:
    if L not in res: continue
    out = {}
    for n, ts in ((120, 5), (240, 10)):
        a, b = read_eul(L, n), read_eul(F, n)   # Eulerian tracer vs Eulerian tracer (same anchored phase)
        if a is None or b is None: continue
        a = a.astype(np.float64) / 65535; b = b.astype(np.float64) / 65535
        out[f"t{ts}s_L1rel"] = float(np.abs(a - b).sum() / b.sum())
        out[f"t{ts}s_mass_ratio"] = float(a.sum() / b.sum())
    res[L]["dye_diff_vs_finest"] = out

# ---- outputs for finest level
os.makedirs(R + "out/dye", exist_ok=True)
nfr = 0
for n in range(241):
    c = read_dye(F, n)
    if c is None: continue
    cv2.imwrite(R + f"out/dye/f{n:04d}.png", np.ascontiguousarray(c[..., ::-1]))
    nfr += 1
uvw = np.stack([read_uvw(F, n) for n in range(241)])
_fx = -0.30 + (np.arange(600) + 0.5) * 1e-3; _fy = 0.06 - (np.arange(120) + 0.5) * 1e-3
FSOLID = (_fx[None, :] + 0.04) ** 2 + _fy[:, None] ** 2 <= 0.005 ** 2   # 1 mm cell centres inside the cylinder
uvw[:, FSOLID, :] = 0.0
uvw = np.nan_to_num(uvw)
d, t0, trel, tfront, _ = load(F)
x = -0.30 + (np.arange(600) + 0.5) * 1e-3
y = 0.06 - (np.arange(120) + 0.5) * 1e-3
np.savez_compressed(R + "out/fields.npz",
                    u=(uvw[..., 0] * U).astype(np.float32), v=(uvw[..., 1] * U).astype(np.float32),
                    omega=(uvw[..., 2] * U / D).astype(np.float32), x=x, y=y, t=np.arange(241) / 24.0,
                    ts_t=(d[:, 0] - t0) * TS, ts_Cd=d[:, 1], ts_Cl=d[:, 2], ts_probe_v=d[:, 3] * U,
                    probe_xy=np.array([-0.01, 0.0]), solid_mask=FSOLID,
                    note="u,v [m/s], omega [1/s], arrays [frame, row(+y first), col(-x first)]; u=v=omega=0 inside the cylinder (solid_mask, r<=D/2 at cell centre); "
                         "ts_* = full time series, t relative to window start (s); Cd,Cl based on U=0.015, D=0.01")

# concentration statistics for the scene's opacity gain (no per-frame rescaling)
cstats = {}
for n in (0, 120, 240):
    c = read_dye(F, n).astype(np.float64) / 65535
    cs_ = c.max(axis=2)
    col = int(round((-0.04 + 0.05 + 0.30) / 2.5e-4 - 0.5))          # x = x_cyl + 5D
    v = cs_[:, col]
    ent = {"x_cyl+5D": {"max": float(v.max()), "p50_of_dyed(c>0.02)": float(np.median(v[v > 0.02])) if (v > 0.02).any() else 0.,
                        "p90_of_dyed": float(np.percentile(v[v > 0.02], 90)) if (v > 0.02).any() else 0.}}
    dyed = np.where((cs_ > 0.02).any(axis=0))[0]
    if len(dyed):
        xf = dyed.max(); band = cs_[:, max(0, xf - 20):xf + 1]
        ent["front"] = {"x_front_m(c>0.02)": float(_px[xf]), "max_in_last_5mm": float(band.max())}
    # near-source reference: 5 mm downstream of needles
    v2 = cs_[:, int(round((-0.093 + 0.005 + 0.30) / 2.5e-4 - 0.5))]
    ent["needle+5mm_max"] = float(v2.max())
    cstats[f"t={n/24:.0f}s"] = ent

# check: Gamma+ peak times in the final window (s)
G = []
for n in range(241):
    om = uvw[n, :, :, 2]
    o = np.nan_to_num(om[np.ix_(np.abs(y) <= 0.01, (x >= -0.03) & (x <= -0.02))]); G.append(o[o > 0].sum())
G = np.array(G)
gp = [n / 24 for n in range(1, 240) if G[n] >= G[n - 1] and G[n] >= G[n + 1] and G[n] > 0.5 * G.max()]

EXTRA = {}
# wake kinetic energy per period (x in [x_cyl+1D, +0.13 m]) and Cl envelope, window of the finest level
_wk = (x >= -0.03) & (x <= 0.13)
_um = uvw[:, :, _wk, :2].mean(axis=0)
ke = np.array([np.nansum((uvw[n][:, _wk, :2] - _um) ** 2) for n in range(241)])   # fluctuation KE in the wake
Tf = 1 / res[F]["window"]["St"] * TS * 24   # frames per period
npf = int(240 // Tf)
kp = [ke[int(round(i * Tf)):int(round((i + 1) * Tf))].mean() for i in range(npf)]
EXTRA["wake_KE_per_period_rel_change"] = [float(b_ / a_ - 1) for a_, b_ in zip(kp[:-1], kp[1:])]
EXTRA["Cl_envelope_change_per_period"] = res[F]["window"]["Cl_envelope_change_per_period"]
if DYE_SRC == "lag":
    lm = json.load(open(R + f"runs/L{F}/lag/lag_meta.json"))
    EXTRA["lagrangian"] = lm
    xc = {}
    for n in range(0, 241, 24):
        e = read_eul(F, n)
        if e is None: continue
        l_ = read_dye(F, n).astype(np.float64) / 65535; e = e.astype(np.float64) / 65535
        em = e.max(axis=2)
        outer = (np.abs(_py) >= 0.012)
        fr = {}
        for thr in (0.02, 0.25):
            cols = np.where((em[outer] > thr).any(axis=0))[0]
            fr[f"eul_front_c>{thr}"] = float(_px[cols.max()]) if len(cols) else None
        xc[f"frame{n}"] = dict(L1_lag_vs_eul=float(np.abs(l_ - e).sum() / l_.sum()), lag_mass_over_eul=float(l_.sum() / e.sum()),
                               lag_front_outer_particles_m=lm["front_outer_x_m_by_frame"].get(str(n)), **fr)
    EXTRA["lag_vs_eulerian_crosscheck"] = xc

sha = {f: hashlib.sha256(open(R + "src/" + f, "rb").read()).hexdigest()
       for f in ("prod.c", "cylforce.h", "bench.c", "post_prod.py", "post_bench.py", "phase.py", "lagr.py", "ccomp.sh")}
bench = json.load(open(R + "runs/bench_run/bench_results.json")) if os.path.exists(R + "runs/bench_run/bench_results.json") else {}
w = res[F]["window"]
meta = dict(version=VER, extra=EXTRA, status="solver output; 2-D NS in the dye plane (model-conditioned)",
            params=dict(D=D, U_ref=U, nu=NU, Re=150, U_in_plug=0.013874, U_in_plug_B=0.01352, lambda_s=6.17e-3, channel_x=[-0.3, 0.3],
                        walls_y=[-0.06, 0.06], cyl=[-0.04, 0.0], needles_x=-0.093,
                        needles_y=[-0.024, -0.016, -0.008, 0.0, 0.008, 0.016, 0.024],
                        needle_colour=[0, 1, 2, 0, 1, 2, 0], source="c=1 disk r=0.35 mm (max(r0, finest cell))",
                        outflow="u Neumann, p=0", inflow="plug"),
            code_sha256=sha, solver="Basilisk centered NS + embed + tracer (BCG), adaptive quadtree",
            levels={L: dict(dx_mm=600 / 2**L) for L in res}, finest_level=F,
            St=w["St"], f_Hz=w["f_Hz"], Cd_mean=w["Cd_mean"], Cl_amp=w["Cl_amp"],
            phase=dict(definition="Cl ~ sin(2 pi f t + phi0); t=0 = window start",
                       phi0_rad=res[F]["phase_phi0_rad"], Cl_at_t0=res[F]["Cl_at_t0"],
                       anchor="measurement (A via lead): lower-side (omega>0) vortex rolls up first at t = 0.75 +- 0.08 s",
                       criterion="max of Gamma+ = int(omega>0) dA over x in [x_cyl+1D, x_cyl+2D], |y|<1D, placed at t = 0.75 s; "
                                 "Gamma+ max <-> Cl phase relation measured on L9 (psi_E = 0.987 period after Cl up-crossing); "
                                 "t0 = Cl up-crossing + phi*T, phi = 0.7618 (solver/runs/L9_phi0/phase_anchor.json)",
                       phi_period_fraction=0.7617858,
                       Gamma_plus_peaks_s=gp,
                       rerun="restart from runs/L%d/dump-* with PHI argument; only the window shifts" % F),
            U_in_calibration=dict(definition="U = 0.015 m/s on the centreline at x_cyl in the EMPTY channel (with floor drag)",
                run_B_value="runs/empty_L9_Uin0.01352: L9 (1.17 mm), U_in=0.01352 -> u_cl(x_cyl)=0.9748 U at t=131 D/U (steady)",
                run_calibrated="runs/empty_L10: L10 (0.59 mm, = delivered production level), U_in=0.013874 -> u_cl(x_cyl)=0.99948 U at t=56 D/U (rising <1e-5 per D/U)",
                U_in_m_s=0.013874, method="U_in = 0.01352/0.9748 (linear rescale), verified at L10 (-0.05 %)"),
            Cl_spikes="raw Cl spikes only at restart instants (t=179.5, 195.0 D/U: first steps after restoring a dump onto a finer adaptive grid); none in the window",
            t0_solver_nd=t0, release_start_s=res[F]["t_rel_s"], front_travel_s=res[F]["front_travel_s"],
            dye_png=dict(size=[NX, NY], m_per_px=2.5e-4, x0_edge=-0.30, y0_edge=0.06,
                         map="pixel centres: column j -> x = -0.30 + (j+0.5)*0.25 mm ; row i -> y = +0.06 - (i+0.5)*0.25 mm",
                         normalisation="c = 1 at the needle source disk, then transport only; no per-frame rescaling; 0 inside cylinder",
                         c_stats=cstats,
                         channels="R,G,B = tracer 0,1,2 concentration*65535 (16-bit)", frames=nfr, fps=24),
            fields_grid="1 mm, 600x120, cell centres", convergence=res, benchmark=bench,
            timing_note="see solver/PROGRESS.log", generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
json.dump(meta, open(R + "out/meta.json", "w"), indent=1, default=float)

os.makedirs(ROOT + "params", exist_ok=True)
phys = {
    "D_m": {"value": D, "tag": "declared"}, "U_ref_m_s": {"value": U, "tag": "declared (B: dye-plane centreline at x_cyl)"},
    "nu_m2_s": {"value": NU, "tag": "assumed"}, "rho_kg_m3": {"value": 998.0, "tag": "assumed"},
    "Re": {"value": 150, "tag": "derived"}, "U_in_plug_m_s": {"value": 0.013874, "tag": "derived (solver empty-channel calibration: B value 0.01352 gave 0.9745 U at x_cyl; rescaled)"},
    "lambda_floor_1_s": {"value": 6.17e-3, "tag": "derived (B: (pi^2/4) nu/h^2)"},
    "channel_half_width_m": {"value": 0.06, "tag": "declared"}, "cylinder_center_m": {"value": [-0.04, 0.0], "tag": "declared"},
    "needle_tips": {"value": {"x": -0.093, "y": [-0.024, -0.016, -0.008, 0, 0.008, 0.016, 0.024]}, "tag": "declared"},
    "source_radius_m": {"value": 3.5e-4, "tag": "declared (B: needle OD/2)"},
    "t_release_s": {"value": res[F]["t_rel_s"], "tag": "derived (B: potential-flow travel 3.87 s)"},
    "phase_phi0_rad": {"value": res[F]["phase_phi0_rad"], "tag": "fitted to measurement: lower-side vortex at 0.75 s (the only free choice)"},
    "St": {"value": w["St"], "tag": "derived (solver)"}, "Cd_mean": {"value": w["Cd_mean"], "tag": "derived (solver)"},
    "Cl_amp": {"value": w["Cl_amp"], "tag": "derived (solver)"}, "f_shed_Hz": {"value": w["f_Hz"], "tag": "derived (solver)"},
    "finest_dx_m": {"value": 0.6 / 2**F, "tag": "numerical"},
}
json.dump(phys, open(ROOT + "params/physics.json", "w"), indent=1, default=float)

# ---- visuals
V = R + "visuals/"
fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
for L in levels:
    if L not in res: continue
    dd, t0L, trL, _, _ = load(L)
    ts = (dd[:, 0] - t0L) * TS
    ax[0].plot(ts, dd[:, 1], lw=0.7, label=f"L{L} ({600/2**L:.2f} mm)")
    ax[1].plot(ts, dd[:, 2], lw=0.7)
for a in ax:
    a.axvspan(0, 10, color="orange", alpha=0.15)
    a.axvline(res[F]["t_rel_s"], color="k", ls=":", lw=0.8)
ax[0].set_ylabel("Cd"); ax[1].set_ylabel("Cl"); ax[1].set_xlabel("t (s), 0 = window start; dotted = dye release")
ax[0].set_ylim(0.9, 1.8); ax[1].set_ylim(-0.9, 0.9); ax[1].set_xlim(-40, 11)
ax[0].legend(fontsize=8)
ax[0].set_title(f"{VER} | Cd, Cl (t) precursor + 10 s window (shaded) | solver   St={w['St']:.4f} Cd={w['Cd_mean']:.3f} Cl'={w['Cl_amp']:.3f}")
fig.tight_layout(); fig.savefig(V + f"{VER}_ClCd_time_solver.png", dpi=110); plt.close(fig)

Ls = [L for L in sorted(res)]
fig, ax = plt.subplots(1, 4, figsize=(14, 3.3))
h = [600 / 2**L for L in Ls]
for a, key in zip(ax[:3], ("St", "Cd_mean", "Cl_amp")):
    a.plot(h, [res[L]["window"][key] for L in Ls], "o-"); a.set_xscale("log"); a.set_xlabel("max-level cell (mm)"); a.set_title(key)
tg = {"St": 0.1911, "Cd_mean": 1.409, "Cl_amp": 0.555}
for a, key in zip(ax[:3], ("St", "Cd_mean", "Cl_amp")):
    a.axhline(tg[key], color="gray", ls="--", lw=0.8)
for L in Ls[1:]:
    dd = res[L].get("dye_diff_vs_finest", {})
    ax[3].plot([600 / 2**L], [dd.get("t5s_L1rel", np.nan)], "bs"); ax[3].plot([600 / 2**L], [dd.get("t10s_L1rel", np.nan)], "r^")
ax[3].set_title("dye L1 diff vs finest (blue 5 s, red 10 s)"); ax[3].set_xscale("log")
fig.suptitle(f"{VER} | production convergence (dashed = B blockage-corrected targets) | solver", fontsize=10)
fig.tight_layout(); fig.savefig(V + f"{VER}_convergence_solver.png", dpi=110); plt.close(fig)

# contact sheet: vorticity + dye at t = 0,2,...,10 s (crop x in [-0.12, 0.30])
fig, ax = plt.subplots(6, 2, figsize=(16, 13))
for r, n in enumerate(range(0, 241, 48)):
    om = uvw[n, :, :, 2] * U / D
    ax[r, 0].imshow(om, cmap="RdBu_r", vmin=-3, vmax=3, extent=[-0.3, 0.3, -0.06, 0.06])
    ax[r, 0].set_xlim(-0.12, 0.3); ax[r, 0].set_ylabel(f"t={n/24:.0f} s")
    c = read_dye(F, n)
    ax[r, 1].imshow(composite(c), extent=[-0.3, 0.3, -0.06, 0.06]); ax[r, 1].set_xlim(-0.12, 0.3)
    for a in ax[r]:
        a.add_patch(plt.Circle((-0.04, 0), 0.005, color="k"))
ax[0, 0].set_title("vorticity (1/s), ±3"); ax[0, 1].set_title("dye (magenta/cyan/yellow)")
fig.suptitle(f"{VER} | t = 0..10 s | solver L{F} ({600/2**F:.2f} mm)", fontsize=12)
fig.tight_layout(); fig.savefig(V + f"{VER}_contact_sheet_solver.jpg", dpi=80); plt.close(fig)

# preview mp4 1200x240
tmp = R + "runs/preview_frames"; os.makedirs(tmp, exist_ok=True)
for n in range(241):
    c = read_dye(F, n)
    img = (composite(c) * 255).astype(np.uint8)
    img = cv2.resize(img, (1200, 240), interpolation=cv2.INTER_AREA)
    cv2.circle(img, (int((-0.04 + 0.3) / 0.0005), 120), 10, (0, 0, 0), -1)
    cv2.putText(img, f"{VER} | t={n/24:5.2f} s | solver L{F}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(f"{tmp}/p{n:04d}.png", img[..., ::-1])
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", f"{tmp}/p%04d.png", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "18", V + f"{VER}_dye_preview_solver.mp4"], check=False)
print("frames", nfr, "done")

# close-up of streaks near the cylinder at t = 0 and 5 s
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for a, n in zip(ax, (0, 120)):
    a.imshow(composite(read_dye(F, n)), extent=[-0.3, 0.3, -0.06, 0.06], interpolation="nearest")
    a.set_xlim(-0.10, 0.03); a.set_ylim(-0.035, 0.035); a.add_patch(plt.Circle((-0.04, 0), 0.005, color="k"))
    a.axvline(-0.034, color="k", lw=0.5, ls=":"); a.set_title(f"t = {n/24:.0f} s (dotted: x_cyl+0.6D)")
fig.suptitle(f"{VER} | dye close-up near cylinder, t = 0 and 5 s | solver L{F} ({DYE_SRC})", fontsize=11)
fig.tight_layout(); fig.savefig(V + f"{VER}_closeup_solver.png", dpi=110); plt.close(fig)
# convergence table
rows = [["level", "cell (mm)", "St", "Cd mean", "Cl' (fit)", "Cl env/period"]]
for L in sorted(res):
    ww = res[L]["window"]
    rows.append([f"L{L}", f"{600/2**L:.2f}", f"{ww['St']:.4f}", f"{ww['Cd_mean']:.3f}", f"{ww['Cl_amp_fit_fundamental']:.3f}", f"{100*ww['Cl_envelope_change_per_period']:.2f} %"])
rows.append(["B target", "", "0.1911", "1.409", "0.555", "< 1 %"])
rows.append(["E ref", "", "", "1.368", "0.499", ""])
fig, a = plt.subplots(figsize=(8, 0.45 * len(rows) + 0.8)); a.axis("off")
a.table(cellText=rows, loc="center", cellLoc="center").scale(1, 1.5)
a.set_title(f"{VER} | production convergence table | solver (delivered: L{F})", fontsize=10)
fig.tight_layout(); fig.savefig(V + f"{VER}_convergence_table_solver.png", dpi=110); plt.close(fig)
print("extra", json.dumps(EXTRA.get("wake_KE_per_period_rel_change")), json.dumps(EXTRA.get("lag_vs_eulerian_crosscheck", {}).get("frame0")))
