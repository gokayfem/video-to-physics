"""Schäfer–Turek 2D-2 benchmark post-processing: St, Cd_max, Cl_max per level + plot."""
import json, sys, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__)) + "/../"
levels = [8, 9, 10]
res = {}
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for L in levels:
    f = R + f"runs/bench_run/forces-{L}.dat"
    if not os.path.exists(f):
        continue
    d = np.loadtxt(f)
    t, cd, cl = d[:, 0], d[:, 1], d[:, 2]
    ax[0].plot(t, cd, lw=0.8, label=f"L{L} ({0.1/(2.2/2**L):.0f} cells/D)")
    ax[1].plot(t, cl, lw=0.8)
    # upward zero crossings of Cl
    k = np.where((cl[:-1] < 0) & (cl[1:] >= 0))[0]
    tc = t[k] - cl[k] * (t[k + 1] - t[k]) / (cl[k + 1] - cl[k])
    tc = tc[tc > 1.0]
    if len(tc) < 4 or np.ptp(cl[t > t[-1] - 1]) < 0.5:
        res[L] = {"status": "not yet periodic", "t_end": float(t[-1])}
        continue
    # use the last 3 complete periods
    tc = tc[-4:]
    T = np.diff(tc).mean()
    m = (t >= tc[0]) & (t <= tc[-1])
    res[L] = {"t_end": float(t[-1]), "St": float(0.1 / T), "Cd_max": float(cd[m].max()),
              "Cl_max": float(cl[m].max()), "Cd_mean": float(cd[m].mean()),
              "periods_used": 3, "cells_per_D": 0.1 / (2.2 / 2**L)}
ax[0].set_ylabel("Cd"); ax[1].set_ylabel("Cl"); ax[1].set_xlabel("t (benchmark units)")
ax[0].legend(fontsize=8); ax[0].set_ylim(2.9, 3.5)
ax[0].set_title("v001 | Schäfer–Turek 2D-2 (Re 100) Cd, Cl vs t | solver")
txt = "published: St 0.295–0.305, Cd_max 3.22–3.24, Cl_max 0.99–1.01\n" + "\n".join(
    f"L{L}: " + (f"St {r['St']:.4f}  Cd_max {r['Cd_max']:.4f}  Cl_max {r['Cl_max']:.4f}" if "St" in r else r["status"])
    for L, r in res.items())
ax[1].text(0.01, 0.02, txt, transform=ax[1].transAxes, fontsize=8, family="monospace",
           bbox=dict(fc="white", alpha=0.8))
fig.tight_layout()
fig.savefig(R + "visuals/v001_benchmark_turek_solver.png", dpi=110)
json.dump(res, open(R + "runs/bench_run/bench_results.json", "w"), indent=1)
print(json.dumps(res, indent=1))
