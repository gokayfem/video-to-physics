"""v003 outputs from a velocity-history run (no solver frame output needed):
out/dye/*.png from RUN/lag, out/fields.npz from RUN/vel (0.5 mm -> 1 mm), window St/Cd/Cl + Gamma+ check into meta,
visuals v003 contact sheet + preview mp4.  Usage: post_v003.py RUN_DIR T0_ND"""
import os, sys, json, glob, hashlib, subprocess, numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.dirname(os.path.abspath(__file__)) + "/../"
RUN, T0 = sys.argv[1], float(sys.argv[2])
LAGDIR = os.environ.get("LAGDIR", "lag"); VER = os.environ.get("VER", "v003")
TS, U, D = 0.01 / 0.015, 0.015, 0.01
NX, NY = 2400, 480
px = -0.30 + (np.arange(NX) + 0.5) * 2.5e-4; py = 0.06 - (np.arange(NY) + 0.5) * 2.5e-4
SOLID = (px[None, :] + 0.04) ** 2 + py[:, None] ** 2 <= 0.005 ** 2
# ---- fields from the velocity history
x5 = -0.30 + (np.arange(1200) + 0.5) * 5e-4; y5 = 0.06 - (np.arange(240) + 0.5) * 5e-4
x1 = -0.30 + (np.arange(600) + 0.5) * 1e-3; y1 = 0.06 - (np.arange(120) + 0.5) * 1e-3
S1 = (x1[None, :] + 0.04) ** 2 + y1[:, None] ** 2 <= 0.005 ** 2
uu, vv, ww = [], [], []
for n in range(241):
    f = R + RUN + f"/vel/v-{T0 + n / 16.0:09.4f}.f32"
    V = np.fromfile(f, np.float32).reshape(240, 1200, 2).astype(np.float64) * U
    om = np.gradient(V[..., 1], 5e-4, axis=1) - np.gradient(V[..., 0], -5e-4, axis=0)   # rows go -y
    blk = lambda a: a.reshape(120, 2, 600, 2).mean(axis=(1, 3))
    u1, v1, w1 = blk(V[..., 0]), blk(V[..., 1]), blk(om)
    for a in (u1, v1, w1):
        a[S1] = 0.0
    uu.append(u1); vv.append(v1); ww.append(w1)
uu, vv, ww = [np.array(a, np.float32) for a in (uu, vv, ww)]
d = np.vstack([np.loadtxt(R + "runs/L11/forces.dat", usecols=(0, 1, 2, 3)), np.loadtxt(R + RUN + "/forces.dat", usecols=(0, 1, 2, 3))])
_, k = np.unique(d[:, 0], return_index=True); d = d[k]
np.savez_compressed(R + "out/fields.npz", u=uu, v=vv, omega=ww, x=x1, y=y1, t=np.arange(241) / 24.0,
                    ts_t=(d[:, 0] - T0) * TS, ts_Cd=d[:, 1], ts_Cl=d[:, 2], ts_probe_v=d[:, 3] * U, probe_xy=np.array([-0.01, 0.0]),
                    solid_mask=S1, note="u,v [m/s], omega [1/s] (central differences on the 0.5 mm history, 2x2 block-averaged to 1 mm); "
                    "arrays [frame, row(+y first), col(-x first)]; 0 inside the cylinder; ts_* relative to window start (s)")
# ---- window shedding numbers
t, cd, cl = d[:, 0], d[:, 1], d[:, 2]
m_ = (t >= T0) & (t <= T0 + 15)
kk = np.where((cl[:-1] < 0) & (cl[1:] >= 0))[0]
tc = t[kk] - cl[kk] * (t[kk + 1] - t[kk]) / (cl[kk + 1] - cl[kk]); tc = tc[(tc >= T0) & (tc <= T0 + 15)]
T = np.diff(tc).mean(); mm = (t >= tc[0]) & (t <= tc[-1]); w0 = 2 * np.pi / T
A = np.c_[np.sin(w0 * t[mm]), np.cos(w0 * t[mm]), np.ones(mm.sum())]
co = np.linalg.lstsq(A, cl[mm], rcond=None)[0]
win = dict(St=float(1 / T), f_Hz=float(1 / (T * TS)), Cd_mean=float(cd[mm].mean()), Cl_amp_fit=float(np.hypot(co[0], co[1])))
G = []
for n in range(241):
    o = ww[n][np.ix_(np.abs(y1) <= 0.01, (x1 >= -0.03) & (x1 <= -0.02))]; G.append(o[o > 0].sum())
G = np.array(G)
gp = []
for n in range(1, 240):
    if G[n] >= G[n - 1] and G[n] >= G[n + 1] and G[n] > 0.5 * G.max():
        a, b, c = G[n - 1], G[n], G[n + 1]; gp.append(float(n / 24 + 0.5 * (a - c) / (a - 2 * b + c) / 24))
print("window", win, "Gamma+ peaks", gp)
# ---- dye PNGs
def dye(n):
    c = np.fromfile(R + RUN + f"/{LAGDIR}/dye-{n:04d}.u16", np.uint16).reshape(NY, NX, 3).copy(); c[SOLID] = 0; return c
for n in range(241):
    cv2.imwrite(R + f"out/dye/f{n:04d}.png", np.ascontiguousarray(dye(n)[..., ::-1]))
cst = {}
for n in (0, 120, 240):
    c = dye(n).astype(np.float64).max(axis=2) / 65535; col = int(round((0.01 + 0.30) / 2.5e-4 - 0.5)); v = c[:, col]
    cst[f"t={n/24:.0f}s"] = {"x_cyl+5D_max": float(v.max()), "x_cyl+5D_p50_dyed": float(np.median(v[v > 0.02])) if (v > 0.02).any() else 0.}
# ---- meta
mt = json.load(open(R + "out/meta.json"))
mt.update(St=win["St"], f_Hz=win["f_Hz"], Cd_mean=win["Cd_mean"], Cl_amp=win["Cl_amp_fit"], finest_level=11, t0_solver_nd=T0)
mt["phase"]["Gamma_plus_peaks_s"] = gp
mt["dye_png"]["c_stats"] = cst
mt["fields_grid"] = "1 mm, 600x120 (block-averaged from the 0.5 mm velocity history)"
mt["code_sha256"] = {f: hashlib.sha256(open(R + "src/" + f, "rb").read()).hexdigest() for f in
                     ("prod.c", "cylforce.h", "bench.c", "lagr.py", "lagr4.py", "ke_v003.py", "post_v003.py", "meta_patch.py", "post_prod.py", "phase.py", "ccomp.sh")}
mt["window_numbers_v003"] = win
json.dump(mt, open(R + "out/meta.json", "w"), indent=1, default=float)
# ---- visuals
cols = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], np.float32)
def comp(c):
    c = c.astype(np.float32) / 65535; img = np.ones(c.shape[:2] + (3,), np.float32)
    for k_ in range(3): img *= 1 - c[..., k_:k_ + 1] * (1 - cols[k_])
    return img
fig, ax = plt.subplots(6, 2, figsize=(16, 13))
for r, n in enumerate(range(0, 241, 48)):
    ax[r, 0].imshow(ww[n], cmap="RdBu_r", vmin=-3, vmax=3, extent=[-0.3, 0.3, -0.06, 0.06]); ax[r, 0].set_ylabel(f"t={n/24:.0f} s")
    ax[r, 1].imshow(comp(dye(n)), extent=[-0.3, 0.3, -0.06, 0.06])
    for a in ax[r]:
        a.set_xlim(-0.12, 0.3); a.add_patch(plt.Circle((-0.04, 0), 0.005, color="k"))
ax[0, 0].set_title("vorticity (1/s), +-3"); ax[0, 1].set_title("Lagrangian dye (magenta/cyan/yellow)")
fig.suptitle(f"{VER} | t = 0..10 s | solver L11 (0.29 mm), window t0 = {T0} D/U | St {win['St']:.4f} Cd {win['Cd_mean']:.3f} Cl' {win['Cl_amp_fit']:.3f}", fontsize=12)
fig.tight_layout(); fig.savefig(R + f"visuals/{VER}_contact_sheet_solver.jpg", dpi=80); plt.close(fig)
tmp = R + "runs/preview_frames"; os.makedirs(tmp, exist_ok=True)
for n in range(241):
    img = cv2.resize((comp(dye(n)) * 255).astype(np.uint8), (1200, 240), interpolation=cv2.INTER_AREA)
    cv2.circle(img, (int((-0.04 + 0.3) / 0.0005), 120), 10, (0, 0, 0), -1)
    cv2.putText(img, f"{VER} | t={n/24:5.2f} s | solver L11 Lagrangian", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(f"{tmp}/p{n:04d}.png", img[..., ::-1])
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", f"{tmp}/p%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "18", R + f"visuals/{VER}_dye_preview_solver.mp4"], check=False)
print("done")
