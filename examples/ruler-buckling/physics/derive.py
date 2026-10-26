"""Role B: formulation numbers for the ruler-buckling scene. Run: <venv python> physics/derive.py
Writes physics/benchmarks.json and physics/visuals/v002_load_deflection.png. SI units throughout."""
import json, math, hashlib, pathlib
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq, fsolve
from scipy.integrate import solve_ivp
from scipy.special import ellipk, ellipe

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = json.loads((ROOT / "config/scene_params.json").read_text())
g = 9.81
R, PL, W = P["ruler"], P["platform"], P["slotted_weights"]
E, nu, rho = R["material_fea"]["E_Pa"], R["material_fea"]["nu"], R["material_fea"]["rho_kg_m3"]
b, t, L = R["width_m"], R["thickness_m"], R["free_length_m"]
I = b * t**3 / 12; EI = E * I; A = b * t; q = rho * A * g
D_ratio = 1 / (1 - nu**2)
px, py, pz = PL["plate_size_m"]; tx, ty, tz = PL["tab_size_m"]
rho_al, rho_ss, rho_br = 2700.0, 7900.0, 8500.0
Lf = L - tz                       # flexible length: clamp (jaw top) -> bottom of the platform tabs
mw = W["mass_each_kg"]; hw = W["thickness_m"]

# ---------- rigid tip body (reference point O = strip mid-surface at s = Lf, tab bottom) ----------
parts = []  # (name, mass, z_above_O, Jx_about_own_cm)  Jx: rotation about X (bending in Y-Z plane)
m = rho_al * px * py * pz; parts.append(("plate", m, tz + pz / 2, m * (py**2 + pz**2) / 12))
for side in ("front", "back"):
    m = rho_al * tx * ty * tz; parts.append((f"tab_{side}", m, tz / 2, m * (ty**2 + tz**2) / 12 + m * (t / 2 + ty / 2)**2))
m = rho_ss * b * t * tz; parts.append(("strip_in_tabs", m, tz / 2, m * tz**2 / 12))
S = PL["screws"]; ms = rho_ss * math.pi * (S["head_diameter_m"] / 2)**2 * S["head_height_m"] * 1.6  # head + short shank (est.)
parts.append(("screws_x2", 2 * ms, tz - S["z_from_top_m"], 0.0))
pg = dict(PL["peg"]); pg["height_m"] = max(pg["height_m"], 0.040)  # lead decision 2026-10-26T09:22:56Z: peg 40 mm (config not yet updated)
m = rho_ss * math.pi * (pg["diameter_m"] / 2)**2 * pg["height_m"]
parts.append(("peg", m, tz + pz + pg["height_m"] / 2, m * (3 * (pg["diameter_m"] / 2)**2 + pg["height_m"]**2) / 12))
m0 = sum(p[1] for p in parts)
z_stack0 = tz + pz
ro, ri = W["outer_diameter_m"] / 2, W["hole_diameter_m"] / 2

def body(n):
    """total mass, CoM height e above O, rotary inertia about CoM (x-axis) for n (continuous) weights."""
    mt = m0 + n * mw
    zc_w = z_stack0 + n * hw / 2
    e = (sum(p[1] * p[2] for p in parts) + n * mw * zc_w) / mt
    J = sum(p[3] + p[1] * (p[2] - e)**2 for p in parts)
    Jw_own = n * mw * (3 * (ro**2 + ri**2) + hw**2) / 12 + mw * n * ((n * hw)**2 - hw**2) / 12  # stack of n discs
    J += Jw_own + n * mw * (zc_w - e)**2
    return mt, e, J

# ---------- linear stability + vibration: Hermite beam FE on [0, Lf] ----------
def fe(nel, Lb, W_tip, e, qq, Mtip=0.0, Jtip=0.0, EIb=EI, mass=False):
    h = Lb / nel; nd = 2 * (nel + 1)
    K = np.zeros((nd, nd)); KG = np.zeros((nd, nd)); M = np.zeros((nd, nd))
    k0 = EIb / h**3 * np.array([[12, 6*h, -12, 6*h], [6*h, 4*h*h, -6*h, 2*h*h], [-12, -6*h, 12, -6*h], [6*h, 2*h*h, -6*h, 4*h*h]])
    g0 = 1 / (30 * h) * np.array([[36, 3*h, -36, 3*h], [3*h, 4*h*h, -3*h, -h*h], [-36, -3*h, 36, -3*h], [3*h, -h*h, -3*h, 4*h*h]])
    m0e = rho * A * h / 420 * np.array([[156, 22*h, 54, -13*h], [22*h, 4*h*h, 13*h, -3*h*h], [54, 13*h, 156, -22*h], [-13*h, -3*h*h, -22*h, 4*h*h]])
    for i in range(nel):
        sm = (i + 0.5) * h; N = W_tip + qq * (Lb - sm)
        ix = slice(2 * i, 2 * i + 4)
        K[ix, ix] += k0; KG[ix, ix] += N * g0; M[ix, ix] += m0e
    KG[-1, -1] += W_tip * e                      # dead load on rigid offset e: -W e theta^2/2 potential
    M[-2:, -2:] += np.array([[Mtip, Mtip * e], [Mtip * e, Mtip * e * e + Jtip]])
    keep = slice(2, nd)                          # clamp w=theta=0 at s=0
    return K[keep, keep], KG[keep, keep], M[keep, keep]

def lam_min(nel, Lb, W_tip, e, qq, EIb=EI):
    K, KG, _ = fe(nel, Lb, W_tip, e, qq, EIb=EIb)
    return eigh(K - KG, K, eigvals_only=True, subset_by_index=[0, 0])[0]

def crit_scale(nel, Lb, e, qq, EIb=EI):  # W_cr for fixed e, qq
    return brentq(lambda Wt: lam_min(nel, Lb, Wt, e, qq, EIb), 0.0, 20.0 * EIb / Lb**2)

NEL = 160
P_E = math.pi**2 * EI / (4 * L**2)
checks = {}
checks["FE_vs_euler_L"] = crit_scale(NEL, L, 0.0, 0.0) / P_E - 1
kL = brentq(lambda x: x * math.tan(x) - Lf / 0.03, 1e-6, math.pi / 2 - 1e-9)
checks["FE_vs_rigid_offset_analytic(e=30mm)"] = crit_scale(NEL, Lf, 0.03, 0.0) / (EI * (kL / Lf)**2) - 1
qcr = brentq(lambda qq: lam_min(NEL, Lf, 0.0, 0.0, qq), 1e-3, 100)
checks["FE_vs_greenhill_7.837"] = qcr * Lf**3 / EI / 7.8373 - 1

def n_crit(Lb=Lf, use_e=True, use_q=True, EIb=EI, nel=NEL):
    def f(n):
        mt, e, _ = body(n)
        return lam_min(nel, Lb, mt * g, e if use_e else 0.0, q if use_q else 0.0, EIb)
    n = brentq(f, -float(m0 / mw) + 1e-6, 40.0)
    mt, e, _ = body(n)
    return n, mt * g, e

cases = {}
cases["A_ideal_euler_L280"] = {"W_cr_N": P_E, "formula": "pi^2 EI/(4 L^2), L=0.280, point load at strip tip, no self-weight"}
cases["A_ideal_euler_L280"]["n_cr_cont"] = (P_E / g - m0) / mw
for key, kw in {"B_Lflex268_no_e_no_q": dict(use_e=False, use_q=False), "C_Lflex268_with_e_no_q": dict(use_q=False),
                "D_corrected_full": {}, "E_corrected_full_plateD_bound": dict(EIb=EI * D_ratio)}.items():
    n, Wc, e = n_crit(**kw); cases[key] = {"W_cr_N": Wc, "n_cr_cont": n, "e_at_cr_m": e}
conv = {nel: n_crit(nel=nel)[1] for nel in (20, 40, 80, 160, 320)}
for c in cases.values():
    c["m_cr_kg"] = c["W_cr_N"] / g; c["buckles_at_weight_no"] = int(math.floor(c["n_cr_cont"]) + 1)

n_cr, W_cr, e_cr = n_crit()

# frequencies vs n (pre-buckling, about straight state)
def freqs(n, nel=NEL, k=3):
    mt, e, J = body(n)
    K, KG, M = fe(nel, Lf, mt * g, e, q, Mtip=mt, Jtip=J)
    w2 = eigh(K - KG, M, eigvals_only=True, subset_by_index=[0, k - 1])
    return [math.sqrt(max(x, 0)) / (2 * math.pi) for x in w2]
freq_table = {str(n): freqs(n) for n in range(0, 13) if n < n_cr}
mt0, e0, J0 = body(0)
k_tip = 3 * EI / Lf**3
f_rayleigh0 = math.sqrt(k_tip / (mt0 + 33 / 140 * rho * A * Lf)) / (2 * math.pi)

# ---------- nonlinear planar elastica with self-weight + rigid tip body (shooting) ----------
def shoot(M0, Wt, e, a0, Lb=Lf, qq=q, H=0.0):
    k0c = a0 * (math.pi / (2 * Lb))**2
    def rhs(s, y):
        th, Mm, x, z = y
        N = Wt + qq * (Lb - s)
        return [Mm / EI + k0c * math.cos(math.pi * s / (2 * Lb)), -N * math.sin(th) - H * math.cos(th), math.sin(th), math.cos(th)]
    sol = solve_ivp(rhs, (0, Lb), [0.0, M0, 0.0, 0.0], rtol=1e-10, atol=1e-13, dense_output=True)
    th, Mm, x, z = sol.y[:, -1]
    return Mm - Wt * e * math.sin(th) - 0.0, sol   # tip: moment from CoM offset (H applied at O)

def equilibrium(n, a0, guess, H=0.0, Lb=Lf, qq=q, e_over=None, W_over=None):
    mt, e, _ = body(n)
    Wt = mt * g if W_over is None else W_over
    e = e if e_over is None else e_over
    M0 = fsolve(lambda v: shoot(v[0], Wt, e, a0, Lb, qq, H)[0], [guess], xtol=1e-13)[0]
    r, sol = shoot(M0, Wt, e, a0, Lb, qq, H)
    th, Mm, x, z = sol.y[:, -1]
    return dict(M0=M0, res=r, theta_tip=th, x_O=x, z_O=z, x_top=x + tz * math.sin(th), x_cm=x + e * math.sin(th),
                drop_top=(Lb + tz) - (z + tz * math.cos(th)), kappa_root=M0 / EI, sol=sol, e=e, W=Wt)

a0 = L / 1000
ns = np.round(np.arange(0, 12.0001, 0.05), 3)
imp = {}
for a in (L / 2000, L / 1000, L / 500):
    gs = 1e-6; rows = []
    for n in ns:
        r = equilibrium(n, a, gs); gs = r["M0"]; rows.append(r)
    imp[a] = rows
# perfect branch: start from the imperfect solution at n=12, remove imperfection, continue downward
perf = {}
gs = imp[L / 1000][-1]["M0"]
for n in ns[::-1]:
    if n <= n_cr + 1e-3: break
    r = equilibrium(n, 0.0, gs)
    if abs(r["M0"]) < 1e-9: break
    gs = r["M0"]; perf[float(n)] = r

# validation of the shooter against the closed-form ideal elastica (e=0, q=0, L=0.28)
el_rows, el_err = [], 0.0
gs = 1e-3
for p in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9):
    K, Ee = ellipk(p * p), ellipe(p * p)
    ratio = (2 * K / math.pi)**2; Wt = ratio * P_E
    row = dict(p=p, alpha_tip_deg=2 * math.degrees(math.asin(p)), P_over_Pcr=ratio, delta_over_L=2 * p / K, drop_over_L=2 - 2 * Ee / K)
    M0 = fsolve(lambda v: shoot(v[0], Wt, 0.0, 0.0, L, 0.0)[0], [Wt * row["delta_over_L"] * L], xtol=1e-14)[0]
    _, sol = shoot(M0, Wt, 0.0, 0.0, L, 0.0)
    row["shooter_delta_over_L"] = sol.y[2, -1] / L
    el_err = max(el_err, abs(row["shooter_delta_over_L"] - row["delta_over_L"]))
    el_rows.append(row)

# post-buckled stiffness -> f1 estimate (lateral force H at O, finite difference)
post_f = {}
for n in range(int(math.floor(n_cr)) + 1, 13):
    r0 = perf.get(float(n)) or equilibrium(n, 0.0, imp[L/1000][int(n / 0.05)]["M0"])
    dH = 1e-3
    r1 = equilibrium(n, 0.0, r0["M0"], H=dH)
    kH = dH / (r1["x_O"] - r0["x_O"])
    mt, e, J = body(n)
    Meff = mt + 33 / 140 * rho * A * Lf
    post_f[str(n)] = dict(k_lat_N_per_m=kH, f1_est_Hz=math.sqrt(max(kH, 0) / Meff) / (2 * math.pi))

def stat(r, n):
    sig = E * t / 2 * abs(r["kappa_root"]) + r["W"] / A
    return dict(n=n, W_N=r["W"], e_m=r["e"], x_top_mm=r["x_top"] * 1e3, x_cm_mm=r["x_cm"] * 1e3, theta_tip_deg=math.degrees(r["theta_tip"]),
                drop_top_mm=r["drop_top"] * 1e3, kappa_root_1pm=r["kappa_root"], sigma_max_MPa=sig / 1e6, searle_beta_root=b * b * abs(r["kappa_root"]) / t,
                residual_Nm=r["res"])
nl_imp = [stat(imp[L / 1000][int(round(n / 0.05))], n) for n in range(13)]
nl_perf = [stat(perf[float(n)], n) if float(n) in perf else dict(n=n, x_top_mm=0.0, theta_tip_deg=0.0) for n in range(13)]

mass_props = {str(n): dict(zip(("m_kg", "e_m", "J_kgm2"), body(n))) for n in range(13)}
out = dict(
    _doc="Role B benchmarks (physics/FORMULATION.md). SI. n = number of 20 g weights on the peg. Generated by physics/derive.py.",
    inputs=dict(E=E, nu=nu, rho=rho, b=b, t=t, L_free_clamp_to_top=L, L_flex=Lf, I=I, EI=EI, q_selfweight_N_per_m=q,
                strip_free_mass_kg=rho * A * L, tip_body_parts=[dict(name=p[0], m_kg=p[1], z_above_O_m=p[2]) for p in parts], m0_kg=m0, g=g),
    fe_validation=dict(checks, mesh_convergence_W_cr_N={str(k): v for k, v in conv.items()}),
    critical_loads=cases,
    mass_properties=mass_props,
    elastica_ideal_closed_form=dict(_doc="clamped-free, tip point load, no self-weight, L=0.280: P/Pcr=(2K(p)/pi)^2, delta/L=2p/K, drop/L=2-2E/K, p=sin(alpha/2)", rows=el_rows, shooter_max_abs_err_delta_over_L=el_err),
    nonlinear_corrected=dict(_doc="full model (Lf, rigid tip body, self-weight); x_top = strip top end lateral (Y) displacement; perfect branch and a0=L/1000 imperfection",
                             perfect=nl_perf, imperfect_L1000=nl_imp),
    frequencies_prebuckled_Hz=dict(_doc="FE, about straight state, incl. axial preload; modes 1-3", table=freq_table, rayleigh_n0_no_preload_Hz=f_rayleigh0),
    frequencies_postbuckled_f1_est_Hz=post_f,
)
(ROOT / "physics/benchmarks.json").write_text(json.dumps(out, indent=1, default=float))

# ---------- figure ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
a = ax[0]
cols = {L / 2000: "#9bb7d4", L / 1000: "#1f5fa8", L / 500: "#0b2f5a"}
for aa, rows in imp.items():
    a.plot([abs(r["x_top"]) * 1e3 for r in rows], ns, color=cols[aa], lw=2 if aa == L / 1000 else 1.2, label=f"imperfect, a0 = L/{int(round(L / aa))} = {aa * 1e3:.2f} mm")
pn = sorted(perf); a.plot([0, 0] + [abs(perf[n]["x_top"]) * 1e3 for n in pn], [0, n_cr] + pn, color="#c0392b", lw=2.2, label="perfect (bifurcation + elastica branch)")
a.plot([0, 250], [n_cr] * 2, "--", color="#c0392b", lw=1); a.text(60, n_cr - 0.45, f"corrected P_cr: n = {n_cr:.2f} (W = {W_cr:.3f} N)", color="#c0392b")
nA = cases["A_ideal_euler_L280"]["n_cr_cont"]
a.plot([0, 250], [nA] * 2, ":", color="k", lw=1.2); a.text(4, nA + 0.12, f"naive Euler pi^2EI/4L^2: n = {nA:.2f}")
a.set_xlim(0, 250); a.set_ylim(0, 12.3); a.set_yticks(range(13))
a.set_xlabel("strip top-end lateral deflection |Y| [mm]"); a.set_ylabel("number of 20 g weights on the peg")
a.set_title("v002 | static equilibria (quasi-static) | model-conditioned, E assumed 193 GPa, peg 40 mm", fontsize=10)
a.grid(alpha=0.3); a.legend(loc="lower right", fontsize=9)
b2 = ax[1]
for n in range(int(n_cr) + 1, 13, 1):
    if float(n) in perf:
        s = perf[float(n)]["sol"]; ss = np.linspace(0, Lf, 200); y = s.sol(ss)
        th = perf[float(n)]["theta_tip"]
        X = np.r_[y[2], y[2][-1] + tz * math.sin(th)]; Z = np.r_[y[3], y[3][-1] + tz * math.cos(th)]
        b2.plot(X * 1e3, Z * 1e3 + 95, lw=1.5, label=f"n={n}")
b2.plot([0, 0], [95, 95 + L * 1e3], "k--", lw=1, label="straight")
b2.axhline(95, color="0.4", lw=3); b2.text(5, 85, "vise jaw top z = 95 mm", fontsize=8)
b2.set_aspect("equal"); b2.set_xlabel("Y [mm]"); b2.set_ylabel("Z [mm]"); b2.set_title("perfect post-buckled shapes (rigid tip body not drawn)", fontsize=10)
b2.grid(alpha=0.3); b2.legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(ROOT / "physics/visuals/v002_load_deflection.png", dpi=110)

print(json.dumps(dict(checks=checks, conv=conv, cases=cases, m0=m0, body0=body(0), n_cr=n_cr, freq=freq_table, post_f=post_f, el_err=el_err,
                      f_ray0=f_rayleigh0), indent=1, default=float))
for r in nl_imp: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
for r in nl_perf: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
