"""Phase anchor: lower-side (omega>0) near-wake vortex circulation maximum at t = 0.75 s.
Criterion: Gamma+(t) = integral of omega>0 over x in [x_cyl+1D, x_cyl+2D] (|y| < 1 D), from the 1 mm fields of a
production run; psi_E = phase (fraction of period after a Cl upward zero crossing) of its maxima.
Output phi such that t0 = cross + phi*T places the Gamma+ maximum at t = 0.75 s.  Usage: phase.py LEVEL"""
import sys, os, json, numpy as np
R = os.path.dirname(os.path.abspath(__file__)) + "/../runs/"
L = int(sys.argv[1])
TS = 0.01 / 0.015
d = np.loadtxt(R + f"L{L}/forces.dat"); t, cl = d[:, 0], d[:, 2]
ev = open(R + f"L{L}/events.txt").read().splitlines()
plan = [l for l in ev if l.startswith("plan")][-1]
kv = dict(p.split("=") for p in plan.replace("plan: ", "").split() if "=" in p)
t0 = float(kv["target_t0"])
x = -0.30 + (np.arange(600) + 0.5) * 1e-3; y = 0.06 - (np.arange(120) + 0.5) * 1e-3
mx = (x >= -0.03) & (x <= -0.02); my = np.abs(y) <= 0.01
G, tf = [], []
for n in range(241):
    f = R + f"L{L}/raw/uvw-{n:04d}.f32"
    if not os.path.exists(f): break
    om = np.fromfile(f, np.float32).reshape(120, 600, 3)[..., 2]
    o = np.nan_to_num(om[np.ix_(my, mx)]); G.append(o[o > 0].sum() * 1e-2); tf.append(t0 + n / 16.)
G, tf = np.array(G), np.array(tf)
k = np.where((cl[:-1] < 0) & (cl[1:] >= 0))[0]
tc = t[k] - cl[k] * (t[k + 1] - t[k]) / (cl[k + 1] - cl[k]); tc = tc[tc > t0 - 12]
T = np.diff(tc).mean()
pk = [i for i in range(1, len(G) - 1) if G[i] >= G[i - 1] and G[i] >= G[i + 1] and G[i] > 0.5 * G.max()]
# parabolic refinement of peak times
tp = []
for i in pk:
    a, b, c = G[i - 1], G[i], G[i + 1]; den = a - 2 * b + c
    tp.append(tf[i] + (0.5 * (a - c) / den if den != 0 else 0) / 16.)
psi = [((tpk - tc[tc <= tpk].max()) / T) % 1 for tpk in tp]
psi_E = float(np.angle(np.mean(np.exp(2j * np.pi * np.array(psi)))) / (2 * np.pi) % 1)
phi = (psi_E - 0.75 / TS / T) % 1
# Cl value at the Gamma+ maximum (sign convention check)
cl_at = [float(np.interp(tpk, t, cl)) for tpk in tp]
out = dict(level=L, criterion="max of Gamma+ = int(omega>0) over x in [x_cyl+1D, x_cyl+2D], |y|<1D (1 mm field)",
           T_nd=T, peaks_t_nd=tp, psi_peaks=psi, psi_E=psi_E, Cl_at_peaks=cl_at, phi=phi,
           t_event_target_s=0.75)
print(json.dumps(out, indent=1))
json.dump(out, open(R + f"L{L}/phase_anchor.json", "w"), indent=1)
