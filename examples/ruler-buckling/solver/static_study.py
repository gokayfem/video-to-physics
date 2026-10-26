"""Static verification + convergence: linear buckling, elastica post-buckling, actual load path, imperfection sensitivity.
Writes solver/out/static.json and solver/out/static_paths.npz."""
import json, time, sys
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.special import ellipk, ellipe
from setup_model import load_params, make_rod, make_point_load_rod, body_props, fracs_static, ROOT
from rod import newton_static, arclength_path, angle_control_path

OUT = ROOT / "solver" / "out"
P, raw = load_params()
EI = P["strip"]["E_Pa"] * P["strip"]["width_m"] * P["strip"]["thickness_m"] ** 3 / 12
L = P["strip"]["free_length_m"]
Lf = L - P["tip_body"]["rigid_top_strip_m"]
g = P["g"]
res = {}
t0 = time.time()


def richardson(vals, hs):  # 2nd order, last two
    return vals[-1] + (vals[-1] - vals[-2]) / ((hs[-2] / hs[-1]) ** 2 - 1)


def obs_order(v):
    return float(np.log2(abs(v[-3] - v[-2]) / abs(v[-2] - v[-1])))


# ---- A: ideal Euler clamped-free, L = 0.280, tip point load
Pcr_an = np.pi ** 2 * EI / (4 * L ** 2)
Ns = [35, 70, 140, 280, 560]
pc = []
for N in Ns:
    r = make_point_load_rod(P, N, e=0.0, Lfree=L)
    G = -r.hess_V(np.zeros(N), 1.0)
    pc.append(float(eigh(r.Kb, G, eigvals_only=True)[0]))
res["A_euler_tip_load_L280"] = dict(analytic_N=Pcr_an, N=Ns, solver_N=pc, rel_err=[p / Pcr_an - 1 for p in pc],
                                    richardson=richardson(pc, Ns and [1 / n for n in Ns]),
                                    observed_order=obs_order(pc))
res["A_euler_tip_load_L280"]["richardson_rel_err"] = res["A_euler_tip_load_L280"]["richardson"] / Pcr_an - 1

# ---- A2: rigid offset e = 30 mm on L = 0.268: kL tan kL = L/e
e = 0.03
kL = brentq(lambda x: x * np.tan(x) - Lf / e, 1e-3, np.pi / 2 - 1e-9)
Pe_an = EI * (kL / Lf) ** 2
pe = []
for N in Ns:
    r = make_point_load_rod(P, N, e=e, Lfree=Lf)
    th = np.zeros(N)
    # K_b + lam*H1 singular; H linear in lam
    G = -r.hess_V(th, 1.0)
    pe.append(float(eigh(r.Kb, G, eigvals_only=True)[0]))
res["A2_rigid_offset_e30mm_L268"] = dict(analytic_N=Pe_an, solver_N=pe, rel_err=[p / Pe_an - 1 for p in pe],
                                         richardson_rel_err=richardson(pe, [1 / n for n in Ns]) / Pe_an - 1)

# ---- D: full model critical weight count (self-weight + tip body growing with the stack)
def mineig(r, lam):
    return eigh(r.Kb + r.hess_V(np.zeros(r.N), lam), eigvals_only=True)[0]

ncr = []
NsD = [67, 134, 268, 536]
for N in NsD:
    r = make_rod(P, N, "actual", imperfection=0.0)
    ncr.append(float(brentq(lambda lam: mineig(r, lam), 3.0, 10.0, xtol=1e-12)))
# independent reference: shooting on the linearised ODE  EI th'' + (W + q (Lf - s)) th = 0, th(0)=0, EI th'(Lf) = W e th(Lf)
q = P["strip"]["rho_kg_m3"] * P["strip"]["width_m"] * P["strip"]["thickness_m"] * g

def shoot(lam):
    m, _, eb, _, _ = body_props(P, fracs_static(lam))
    W = m * g
    sol = solve_ivp(lambda s, y: [y[1], -(W + q * (Lf - s)) * y[0] / EI], [0, Lf], [0.0, 1.0], rtol=1e-12, atol=1e-14, method="DOP853")
    th, dth = sol.y[0, -1], sol.y[1, -1]
    return EI * dth - W * eb * th

n_shoot = brentq(shoot, 3.0, 10.0, xtol=1e-12)
mD, _, eD, _, _ = body_props(P, fracs_static(n_shoot))
res["D_full_model"] = dict(N=NsD, n_cr_solver=ncr, n_cr_shooting_ODE=n_shoot,
                           rel_err_n=[x / n_shoot - 1 for x in ncr],
                           W_cr_shooting_N=mD * g, e_at_cr_m=eD,
                           W_cr_solver_N=[body_props(P, fracs_static(x))[0] * g for x in ncr],
                           observed_order=obs_order(ncr), richardson=richardson(ncr, [1 / n for n in NsD]),
                           buckles_on_weight_no=int(np.ceil(n_shoot)))
res["D_full_model"]["W_rel_err"] = [w / (mD * g) - 1 for w in res["D_full_model"]["W_cr_solver_N"]]
print("linear done %.1fs" % (time.time() - t0), json.dumps(res, indent=0)[:1500], flush=True)

# ---- B: elastica post-buckling (ideal), path following through the bifurcation (imperfection L*1e-6)
el = {}
paths = {}
for N in [35, 70, 140, 280]:
    r = make_point_load_rod(P, N, e=0.0, Lfree=L, imperfection=0.0)
    wv, vv = eigh(r.Kb, -r.hess_V(np.zeros(N), 1.0))
    path = [(0.0, np.zeros(N))] + angle_control_path(r, np.radians(np.linspace(0.5, 150, 120)), wv[0], vv[:, 0] * 1e-3, scale_first=True)
    lam = np.array([p[0] for p in path]); TH = np.array([p[1] for p in path])
    y = np.array([r.positions(th, l)[0][N] for l, th in zip(lam, TH)])
    z = np.array([r.positions(th, l)[1][N] for l, th in zip(lam, TH)])
    alpha = TH[:, -1]
    p_ = np.sin(alpha / 2)
    K = ellipk(p_ ** 2); Ee = ellipe(p_ ** 2)
    P_an = EI * K ** 2 / L ** 2
    d_an = 2 * p_ / K * L
    m = alpha > np.radians(10)
    el[N] = dict(max_rel_err_P=float(np.max(np.abs(lam[m] / P_an[m] - 1))),
                 max_err_delta_over_L=float(np.max(np.abs(y[m] - d_an[m]) / L)),
                 steps=len(path))
    # value at fixed alpha = 90 deg by interpolation
    el[N]["P_over_Pcr_at_alpha90"] = float(np.interp(np.pi / 2, alpha, lam) / Pcr_an)
    paths[f"elastica_N{N}_P"] = lam; paths[f"elastica_N{N}_y"] = y; paths[f"elastica_N{N}_z"] = z; paths[f"elastica_N{N}_alpha"] = alpha
pa = np.sin(np.pi / 4)
el["analytic_P_over_Pcr_at_alpha90"] = float((2 * ellipk(pa ** 2) / np.pi) ** 2)
res["B_elastica"] = el
print("elastica done %.1fs" % (time.time() - t0), json.dumps(el), flush=True)

# ---- C: actual load path (n = 0 .. 12 weights), imperfection sensitivity, mesh convergence at n = 10
def actual_path(r):
    # load control (Newton) to n=3, then tip-angle control to beyond n = 12
    path = []
    th = np.zeros(r.N)
    for lam in np.linspace(0, 5.5, 23):
        th, ok, _ = newton_static(r, th, lam); path.append((lam, th.copy()))
    a0 = th[-1]
    sgn = np.sign(r.theta0[-1]) if r.theta0[-1] != 0 else -1.0
    al = sgn * np.concatenate([abs(a0) * np.geomspace(1.02, np.radians(5) / abs(a0), 60), np.radians(np.linspace(5.5, 150, 160))])
    p2 = angle_control_path(r, al, 5.5, th if abs(a0) > 1e-9 else sgn * 1e-4 * np.linspace(0, 1, r.N), lam_stop=12.2)
    for lam, t_ in p2:
        path.append((lam, t_))
        if lam > 12.2:
            break
    return path


act = {}
for N in [67, 134, 268, 536]:
    r = make_rod(P, N, "actual")
    path = actual_path(r)
    lam = np.array([p[0] for p in path]); TH = np.array([p[1] for p in path])
    Y = np.array([r.positions(th, l)[0] for l, th in zip(lam, TH)])
    Z = np.array([r.positions(th, l)[1] for l, th in zip(lam, TH)])
    # tip (strip top end = top of flexible part + 12 mm rigid along tangent)
    rt = P["tip_body"]["rigid_top_strip_m"]
    ytop = Y[:, N] + rt * np.sin(TH[:, -1]); ztop = Z[:, N] + rt * np.cos(TH[:, -1])
    paths[f"actual_N{N}_lam"] = lam; paths[f"actual_N{N}_ytop"] = ytop; paths[f"actual_N{N}_ztop"] = ztop
    paths[f"actual_N{N}_thtip"] = TH[:, -1]
    # shape at n = 10: solve directly
    th10, ok, _ = newton_static(r, TH[np.argmin(abs(lam - 10))], 10.0)
    yy, zz = r.positions(th10, 10.0)
    s_nodes = np.arange(N + 1) * r.h
    paths[f"actual_N{N}_shape10_s"] = s_nodes; paths[f"actual_N{N}_shape10_y"] = yy[: N + 1]; paths[f"actual_N{N}_shape10_z"] = zz[: N + 1]
    act[N] = dict(ytop_n10_mm=float((yy[N] + rt * np.sin(th10[-1])) * 1e3), ztop_drop_n10_mm=float((Lf + rt - zz[N] - rt * np.cos(th10[-1])) * 1e3),
                  theta_tip_n10_deg=float(np.degrees(th10[-1])), steps=len(path))
    for n in range(0, 13):
        act[N][f"ytop_mm_n{n}"] = float(np.interp(n, lam, ytop) * 1e3)
    print("actual N", N, act[N]["ytop_n10_mm"], "%.1fs" % (time.time() - t0), flush=True)
# shape convergence: max |r_N(s) - r_536(s)| at n=10 on common s
sref = paths["actual_N536_shape10_s"]
for N in [67, 134, 268]:
    s = paths[f"actual_N{N}_shape10_s"]
    dy = np.interp(sref, s, paths[f"actual_N{N}_shape10_y"]) - paths["actual_N536_shape10_y"]
    dz = np.interp(sref, s, paths[f"actual_N{N}_shape10_z"]) - paths["actual_N536_shape10_z"]
    act[N]["shape10_max_dev_vs_N536_mm"] = float(np.max(np.hypot(dy, dz)) * 1e3)
res["C_actual_static"] = act

imp = {}
for d0 in [-Lf / 5000, -Lf / 1000, -Lf / 500, -Lf * 1e-6]:
    r = make_rod(P, 268, "actual", imperfection=d0)
    path = actual_path(r)
    lam = np.array([p[0] for p in path]); TH = np.array([p[1] for p in path])
    Y = np.array([r.positions(th, l)[0][r.N] for l, th in zip(lam, TH)])
    key = "L/%d" % round(Lf / abs(d0))
    paths["imp_" + key + "_lam"] = lam; paths["imp_" + key + "_y"] = Y
    imp[key] = {f"y_mm_n{n}": float(np.interp(n, lam, Y) * 1e3) for n in range(4, 13)}
res["C_imperfection_sensitivity_N268"] = imp
res["runtime_s"] = time.time() - t0
(OUT / "static.json").write_text(json.dumps(res, indent=1))
np.savez_compressed(OUT / "static_paths.npz", **paths)
print("static done %.1fs" % (time.time() - t0))
