"""Dynamic run for the actual loading schedule.  usage: dynamic.py N dt_inv [rho_inf] [tag]
Outputs solver/out/dyn_<tag>.npz: samples at t = k/120 s (24 fps x 5 = frame + 4 subframes), k = 0..1200."""
import sys, time, json
import numpy as np
from setup_model import load_params, make_rod, schedule, ROOT
from rod import integrate, newton_static

N = int(sys.argv[1]); dtinv = float(sys.argv[2])
rho = float(sys.argv[3]) if len(sys.argv) > 3 else 0.85
tag = sys.argv[4] if len(sys.argv) > 4 else f"N{N}_dt{int(dtinv)}"
P, raw = load_params()
T_END = P["loading"]["duration_s"]
rod = make_rod(P, N, "dyn")
if "nodamp" in tag:
    rod.eta = 0.0; rod.alphaM = 0.0; rod.drag = False; rod.cair = 0.0
# initial state: static equilibrium with the empty platform (t=0)
th0, ok, _ = newton_static(rod, np.zeros(N), 0.0)
out_t = np.arange(0, int(round(T_END * 120)) + 1) / 120.0
t0 = time.time()
outs, hist = integrate(rod, T_END, 1.0 / dtinv, out_t, rho_inf=rho, th_init=th0, lamfun=lambda t: t)
wall = time.time() - t0
rt = P["tip_body"]["rigid_top_strip_m"]
Y = []; Z = []; tipb = []
for t, th in zip(outs["t"], outs["th"]):
    y, z = rod.positions(th, t)
    Y.append(y[: N + 1]); Z.append(z[: N + 1]); tipb.append([y[N + 1], z[N + 1]])
Y = np.array(Y); Z = np.array(Z)
ytop = Y[:, N] + rt * np.sin(outs["th"][:, -1]); ztop = Z[:, N] + rt * np.cos(outs["th"][:, -1])
# energy balance:  E(t) - E0 - int P_ext + int P_diss
E = hist["T"] + hist["U"] + hist["V"]
tt = hist["t"]
cum = lambda p: np.concatenate([[0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(tt))])
Wext = cum(hist["Pext"]); Wd = cum(hist["Pdiss"])
resid = E - E[0] - Wext + Wd
np.savez_compressed(ROOT / "solver" / "out" / f"dyn_{tag}.npz", t=outs["t"], th=outs["th"], w=outs["w"], Y=Y, Z=Z,
                    ytop=ytop, ztop=ztop, tipcom=np.array(tipb), h=rod.h, N=N, dt=1 / dtinv, rho_inf=rho,
                    e_t=tt, e_T=hist["T"], e_U=hist["U"], e_V=hist["V"], e_Wext=Wext, e_Wdiss=Wd, e_resid=resid,
                    iters=hist["iters"], theta0=rod.theta0, wall_s=wall)
print(json.dumps(dict(tag=tag, wall_s=wall, ytop_10s_mm=float(ytop[-1] * 1e3), max_abs_resid_J=float(np.max(np.abs(resid))),
                      Wext_J=float(Wext[-1]), Wdiss_J=float(Wd[-1]), mean_iters=float(np.mean(hist["iters"])))))
