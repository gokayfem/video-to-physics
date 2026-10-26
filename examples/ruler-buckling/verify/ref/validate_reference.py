"""Validate the independent elastica reference and produce reference numbers + visual v001.
Run:  <venv-python> verify/ref/validate_reference.py      (writes verify/out/reference.json, verify/visuals/v001_*)"""
import json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import elastica as E
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V = Path(__file__).resolve().parents[1]
OUT, VIS = V / "out", V / "visuals"
OUT.mkdir(exist_ok=True); VIS.mkdir(exist_ok=True)
t0 = time.time()
res = {"_doc": "role-E independent reference (verify/ref/elastica.py). SI. g=9.80665. E=193 GPa ASSUMED (contract).",
       "validation": {}}
val = res["validation"]

rod = E.Rod()
ideal = E.Rod(rho=0.0)  # q = 0

# 1) Euler closed form
Wc = E.critical_load(ideal)
val["euler_closed"] = dict(shoot=Wc, closed=E.euler_closed(ideal), rel=Wc / E.euler_closed(ideal) - 1)
# 2) eccentric tip load closed form tan(kL)=1/(ke)
for e in (0.005, 0.02, 0.05):
    a, b = E.critical_load(ideal, e), E.eccentric_closed(ideal, e)
    val[f"eccentric_e{e}"] = dict(shoot=a, closed=b, rel=a / b - 1)
# 3) Greenhill heavy column (W -> 0): find q_cr by scaling rho until Wcr = 0
qcr = E.greenhill_qcr(rod)
from scipy.optimize import brentq
def lin_q_only(rh):
    r = E.Rod(rho=rh)
    from scipy.integrate import solve_ivp
    f = lambda s, u: [u[1], -(r.q * (r.L - s)) * u[0] / r.EI]
    sol = solve_ivp(f, (0, r.L), [0.0, 1.0], method="DOP853", rtol=1e-13, atol=1e-18)
    return sol.y[1, -1]
rho_c = brentq(lin_q_only, 1e4, 1e6, xtol=1e-12, rtol=1e-15)
q_shoot = E.Rod(rho=rho_c).q
val["greenhill"] = dict(shoot_qcr=q_shoot, closed_qcr=qcr, rel=q_shoot / qcr - 1)
# 4) elliptic-integral elastica, post-buckling branch (ideal)
errs = []
branch_ideal = []
for alpha_deg in (5, 10, 20, 40, 60, 90, 120, 150):
    al = math.radians(alpha_deg)
    W, yt, zt = E.elliptic_tip(ideal, al)
    sol = E.equilibrium(ideal, W, 0.0, 0.0, n_out=2)
    errs.append(dict(alpha_deg=alpha_deg, W=W, rel_y=sol["tip_y"] / yt - 1, rel_z=sol["tip_z"] / zt - 1,
                     rel_thetaL=sol["theta_L"] / al - 1))
    branch_ideal.append((W, yt, zt))
val["elliptic"] = errs
# 5) clamped-free + tip mass vibration (FE, no gravity load) vs frequency equation
vib = []
for Mt in (0.0, 0.0087, 0.1, 0.2):
    fc = E.tipmass_freq_closed(rod, Mt)
    ff = E.fe_frequency(rod, ne=30, W=0.0, M_tip=Mt, with_q=False)
    vib.append(dict(M_tip=Mt, closed_Hz=fc, fe_Hz=ff, rel=ff / fc - 1))
val["tipmass_vibration"] = vib
# 6) FE buckling eigenvalue with self-weight vs shooting (independent discretisation)
fe = E.fe_critical(rod, ne=30, e=0.01)
sh = E.critical_load(rod, 0.01)
val["fe_vs_shoot_q_e0.01"] = dict(fe=fe, shoot=sh, rel=fe / sh - 1)

# ---------------- scene numbers
rows = E.critical_tip_mass_curve(rod, n_max=12)
n_buckle = next(r["n"] for r in rows if r["ratio"] > 1)
res["scene"] = dict(config=E.DECLARED, m0_kg=E.scene_tipbody(0).M, e0_m=E.scene_tipbody(0).e,
    
    EI_Nm2=rod.EI, q_N_per_m=rod.q, strip_free_mass_kg=rod.rho * rod.A * rod.L,
    euler_ideal_N=rod.euler, euler_ideal_kg=rod.euler / rod.g,
    Wcr_selfweight_e0_N=E.critical_load(rod, 0.0),
    per_weight_count=rows, n_buckle=n_buckle,
    note="ratio = W(n)/Wcr(e(n)); first n with ratio>1 is the weight that buckles the strip (perfect, d=0)")
# continuous critical added mass (weights as a continuum at the CoM height of the stack)
def g_of_madd(madd):
    nf = madd / 0.02
    tb0 = E.scene_tipbody(0); t = 0.0045; pz = 0.012 + 0.003
    M = tb0.M + madd
    e = (tb0.M * tb0.e + madd * (pz + nf * t / 2)) / M
    return M * rod.g - E.critical_load(rod, e)
madd_cr = brentq(g_of_madd, 0.0, 0.24, xtol=1e-12)
res["scene"]["critical_added_mass_kg_continuous"] = madd_cr
# plate-modulus bracket
rodp = E.Rod(plate_modulus=True)
rows_p = E.critical_tip_mass_curve(rodp, n_max=12)
res["scene"]["n_buckle_plate_modulus"] = next(r["n"] for r in rows_p if r["ratio"] > 1)
res["scene"]["Wcr_ratio_plate_over_beam"] = rows_p[n_buckle]["Wcr_N"] / rows[n_buckle]["Wcr_N"]

# post-buckling branch for the scene at the buckling weight count geometry (perfect + tab imperfection)
tb = E.scene_tipbody(n_buckle)
Wcr_n = rows[n_buckle]["Wcr_N"]
post = []
for f in np.linspace(1.0005, 1.6, 30):
    W = f * Wcr_n
    s = E.equilibrium(rod, W, tb.e, 0.0, n_out=2)
    post.append(dict(W_over_Wcr=f, W=W, tip_y=s["tip_y"], tip_drop=rod.L - s["tip_z"], theta_L=s["theta_L"]))
res["scene"]["post_buckling_branch_n%d" % n_buckle] = post
# per-weight-count static equilibrium with the tab imperfection (natural path)
per_n = []
for n in range(0, 13):
    tbn = E.scene_tipbody(n)
    s = E.equilibrium(rod, tbn.M * rod.g, tbn.e, -1e-4, n_out=2)  # illustrative 0.1 mm lateral CoM offset
    s0 = E.equilibrium(rod, tbn.M * rod.g, tbn.e, 0.0, n_out=2)
    per_n.append(dict(n=n, W=tbn.M * rod.g, d_m=tbn.d, e_m=tbn.e, tip_y_tab=s["tip_y"], tip_y_perfect=s0["tip_y"],
                      tip_drop_tab=rod.L - s["tip_z"]))
res["scene"]["static_per_weight_count"] = per_n
# small-amplitude frequency vs n (straight state, tip body), for the dynamics reviewer
res["scene"]["freq_Hz_per_n_straight"] = [
    dict(n=n, f_Hz=E.fe_frequency(rod, ne=30, W=E.scene_tipbody(n).M * rod.g, M_tip=E.scene_tipbody(n).M,
                                    e=E.scene_tipbody(n).e, J_c=E.scene_tipbody(n).J_c)) for n in range(0, n_buckle)]
def cont(cfg):
    tb0 = E.scene_tipbody(0, cfg=cfg)
    def gg(madd):
        M = tb0.M + madd; e = (tb0.M * tb0.e + madd * (0.015 + madd / 0.02 * 0.0045 / 2)) / M
        return M * rod.g - E.critical_load(rod, e)
    m = brentq(gg, 0, 0.24, xtol=1e-13)
    return dict(m0_kg=tb0.M, Wcr_N=(tb0.M + m) * rod.g, n_cont=m / 0.02,
                n_buckle=next(r["n"] for r in E.critical_tip_mass_curve(rod, n_max=12, cfg=cfg) if r["ratio"] > 1))
res["scene"]["variants"] = {"peg40_from_geometry(m0=13.94g)": cont({}), "m0=11.96g_declared": cont({"m0_override": 0.01196}),
                            "peg20(m0=11.95g)": cont({"peg_h": 0.02})}
res["scene"]["production_claim"] = dict(Pcr_N=1.358, n=6.32,
    rel_vs_peg40=1.358 / res["scene"]["variants"]["peg40_from_geometry(m0=13.94g)"]["Wcr_N"] - 1,
    rel_vs_m0_11p96=1.358 / res["scene"]["variants"]["m0=11.96g_declared"]["Wcr_N"] - 1)
res["runtime_s"] = time.time() - t0
(OUT / "reference.json").write_text(json.dumps(res, indent=1))

# ---------------- visual v001
fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
a = ax[0]
Wg = np.linspace(1.0005, 3.0, 60) * ideal.euler
# shooting branch vs elliptic closed form
al = np.radians(np.linspace(1, 150, 80))
ell = np.array([E.elliptic_tip(ideal, x) for x in al])
a.plot(ell[:, 1] / ideal.L, ell[:, 0] / ideal.euler, "-", color="0.6", lw=4, label="elliptic integrals (closed form)")
sh = [E.equilibrium(ideal, W, 0, 0, n_out=2) for W in ell[::6, 0]]
a.plot([s["tip_y"] / ideal.L for s in sh], ell[::6, 0] / ideal.euler, "o", color="C3", ms=5, label="shooting (this ref.)")
a.axhline(1, color="k", lw=0.8, ls=":")
a.set_xlabel("tip lateral deflection / L"); a.set_ylabel("P / (pi^2 EI / 4L^2)")
maxe = max(max(abs(x["rel_y"]), abs(x["rel_z"])) for x in errs)
a.set_title(f"v001 | ideal elastica | max rel err {maxe:.1e}\nEuler check rel {val['euler_closed']['rel']:.1e}")
a.legend(fontsize=8)
a = ax[1]
ns = [r["n"] for r in rows]
a.bar(ns, [r["W_N"] for r in rows], color=["C3" if r["ratio"] > 1 else "C0" for r in rows], label="tip weight W(n)")
a.plot(ns, [r["Wcr_N"] for r in rows], "k.-", label="W_cr(e(n)) (self-weight + eccentric CoM)")
a.axhline(rod.euler, color="0.5", ls="--", lw=1, label="pi^2EI/4L^2 (no self-weight, e=0)")
a.set_xlabel("number of 20 g weights on peg"); a.set_ylabel("N")
a.set_title(f"v001 | scene | buckles when weight #{n_buckle} is added\n(M_add,cr = {madd_cr*1e3:.1f} g continuous)")
a.legend(fontsize=8)
a = ax[2]
a.plot([p["n"] for p in per_n], [1e3 * p["tip_y_perfect"] for p in per_n], "o-", label="perfect (d=0)")
a.plot([p["n"] for p in per_n], [1e3 * p["tip_y_tab"] for p in per_n], "s--", label="0.1 mm lateral CoM offset (imperfection)")
s0 = E.equilibrium(rod, rows[n_buckle]["W_N"], tb.e, 0, n_out=301)
ins = a.inset_axes([0.08, 0.45, 0.3, 0.5])
for n in range(n_buckle, 13):
    tbn = E.scene_tipbody(n); s = E.equilibrium(rod, tbn.M * rod.g, tbn.e, 0, n_out=301)
    ins.plot(1e3 * s["y"], 1e3 * s["z"], lw=1)
ins.set_aspect("equal"); ins.set_title("shapes n>=%d (mm)" % n_buckle, fontsize=7); ins.tick_params(labelsize=6)
a.set_xlabel("weights on peg"); a.set_ylabel("tip lateral deflection [mm]")
a.set_title("v001 | static tip deflection per weight count\n(static equilibria; dynamics not included)")
a.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(VIS / "v001_elastica_branch_euler_check.png", dpi=110)
print(json.dumps(dict(validation=val, n_buckle=n_buckle, madd_cr=madd_cr, runtime=res["runtime_s"]), indent=1, default=float)[:4000])
