"""v003 periodicity: wake fluctuation KE per shedding period, 4 bands, from the 0.5 mm velocity history
(runs/L11/vel: t = 198-224.5 D/U, x in [-0.10,0.30] m; runs/L11e/vel: t >= 224.5, full channel), and Cl envelope.
Per period [tc_i, tc_{i+1}] (Cl up-crossings): KE_i = mean over the period of sum (u - <u>_i)^2 + (v - <v>_i)^2 dA,
<u>_i = period mean.  Writes runs/L11e/ke_periods.json and visuals/v003_KE_per_period_solver.png.  Usage: ke_v003.py"""
import os, glob, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.dirname(os.path.abspath(__file__)) + "/../"
BANDS = {"near [-0.03,0.03]": (-3, 3), "mid [0.03,0.08]": (3, 8), "far [0.08,0.13]": (8, 13), "outlet [0.13,0.25]": (13, 25)}
snaps = []
for d in ("runs/L11/vel", "runs/L11e/vel"):
    for f in sorted(glob.glob(R + d + "/v-*.f32")):
        snaps.append((float(os.path.basename(f)[2:-4]), f))
snaps = sorted(dict(snaps).items())
tv = np.array([s[0] for s in snaps])
xc = lambda nvx: (-10.0 if nvx == 800 else -30.0) + (np.arange(nvx) + 0.5) * 0.05
d = np.vstack([np.loadtxt(R + "runs/L11/forces.dat", usecols=(0, 1, 2)), np.loadtxt(R + "runs/L11e/forces.dat", usecols=(0, 1, 2))])
_, k = np.unique(d[:, 0], return_index=True); d = d[k]
t, cl = d[:, 0], d[:, 2]
k = np.where((cl[:-1] < 0) & (cl[1:] >= 0))[0]
tc = t[k] - cl[k] * (t[k + 1] - t[k]) / (cl[k + 1] - cl[k])
tc = tc[(tc >= tv[0]) & (tc <= tv[-1])]
res = []
for a, b in zip(tc[:-1], tc[1:]):
    sel = [i for i in range(len(tv)) if a <= tv[i] < b]
    U = []
    for i in sel:
        V = np.fromfile(snaps[i][1], np.float32); nvx = V.size // 480
        V = V.reshape(240, nvx, 2); x = xc(nvx); m = (x >= -10) & (x <= 30)
        U.append(V[:, m])
    U = np.array(U); x = xc(800)
    um = U.mean(axis=0); fl = ((U - um) ** 2).sum(axis=3).mean(axis=0)   # (240, 800)
    row = {"t_start": float(a), "t_end": float(b), "T": float(b - a)}
    for name, (lo, hi) in BANDS.items():
        row[name] = float(fl[:, (x >= lo) & (x <= hi)].sum() * 0.05 ** 2)
    q = (t >= a) & (t <= b); row["Cl_amp"] = float(0.5 * np.ptp(cl[q]))
    res.append(row)
for i in range(1, len(res)):
    for name in list(BANDS) + ["Cl_amp"]:
        res[i]["rel_change_" + name] = res[i][name] / res[i - 1][name] - 1
json.dump(res, open(R + "runs/L11e/ke_periods.json", "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
tm = [0.5 * (r["t_start"] + r["t_end"]) for r in res]
for name in BANDS:
    ax[0].plot(tm, [r[name] / res[-1][name] for r in res], "o-", label=name)
    ax[1].plot(tm[1:], [100 * r["rel_change_" + name] for r in res[1:]], "o-", label=name)
ax[1].plot(tm[1:], [100 * r["rel_change_Cl_amp"] for r in res[1:]], "k^--", label="Cl envelope")
ax[1].axhspan(-1, 1, color="green", alpha=0.12); ax[1].set_ylabel("change per period (%)")
ax[0].set_ylabel("fluct. KE per period / last"); ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
for a_ in ax:
    a_.axvline(179.5, color="gray", ls=":"); a_.axvline(224.5, color="gray", ls="--"); a_.set_xlabel("t (D/U); dotted: L9->L11 restart (179.5)")
fig.suptitle("v003 | wake fluctuation KE per shedding period, 4 bands (green: +-1 %) | solver L11", fontsize=10)
fig.tight_layout(); fig.savefig(R + "visuals/v003_KE_per_period_solver.png", dpi=110)
for r in res:
    print(f"{r['t_start']:.2f}-{r['t_end']:.2f}", " ".join(f"{100*r.get('rel_change_'+n, 0):+.2f}" for n in list(BANDS) + ['Cl_amp']))
