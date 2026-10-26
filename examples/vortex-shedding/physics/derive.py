#!/usr/bin/env python
"""Role B: reproducible derivation of the vortex-shedding physics targets.

Run:  <venv>/python physics/derive.py   (from the clip root or anywhere)
Writes physics/benchmarks.json and physics/visuals/v001_expectations.png.
All inputs come from config/scene_params.json + the literature constants below (cited in FORMULATION.md).
"""
import json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P = json.load(open(os.path.join(ROOT, "config", "scene_params.json")))

# ---------------- inputs (assumed/declared in scene_params) ----------------
nu = P["water"]["nu_m2_s"]; rho = P["water"]["rho_kg_m3"]
h = P["water"]["depth"]; D = P["cylinder"]["D"]; xc = P["cylinder"]["center"][0]
W = P["channel"]["inner_width"]; L = P["channel"]["length"]
U = P["flow"]["U_m_s"]                       # DECISION: approach velocity at the dye plane (see FORMULATION §2)
x_in, x_out = -L / 2, L / 2
x_needle = P["injectors"]["x_tip"]; y_needles = P["injectors"]["y"]
z_floor = P["water"]["surface_z"] - h; z_dye = P["dye"]["z_sheet"]
zeta_dye = (z_dye - z_floor) / h
Re = U * D / nu
T_conv = D / U

# ---------------- 1. shallow-layer friction ----------------
# laminar half-Poiseuille (no-slip floor, stress-free surface): u(zeta) = 1.5*ubar*(2 zeta - zeta^2)
shape_mid = 1.5 * (2 * zeta_dye - zeta_dye ** 2)       # u_mid / u_bar
lam_bulk = 3 * nu / h ** 2                              # depth-averaged bottom friction (developed profile)
lam_mode = (math.pi ** 2 / 4) * nu / h ** 2             # decay rate of slowest vertical mode (floor no-slip, free surface)
lam = lam_mode                                          # DECISION: linear drag used in the 2-D mid-plane model
c_f = 2 * lam_bulk * h / (U / shape_mid)                # equivalent friction coefficient on u_bar
S_shallow = c_f * D / (2 * h)                           # Chen & Jirka (1995) wake stability number
S_c = 0.2                                               # lower bound of critical S (vortex street disappears above ~0.2-0.5)
decay_10s = math.exp(-lam * 10.0)
drag_vs_visc = (lam * D / U) / (1.0 / Re)               # friction per convective time vs viscous 1/Re
brinkman_len = math.sqrt(nu / lam)

# ---------------- 2. developing 2-D lateral profile (plug inflow + drag), parabolized marching ----------------
def march(U_in, lam_, ny=241, dx=2e-4, x_stop=xc):
    y = np.linspace(-W / 2, W / 2, ny); dy = y[1] - y[0]
    u = np.full(ny, U_in); u[0] = u[-1] = 0.0
    v = np.zeros(ny)
    Q = np.trapezoid(u, y) if hasattr(np, "trapezoid") else np.trapz(u, y)
    Q = U_in * W  # plug flux
    x = x_in; n = ny - 2
    trap = (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)
    while x < x_stop - 1e-12:
        ui = u[1:-1]; vi = v[1:-1]
        a_lo = -vi / (2 * dy) - nu / dy ** 2
        a_up = vi / (2 * dy) - nu / dy ** 2
        a_d = ui / dx + 2 * nu / dy ** 2 + lam_
        def solve(rhs):
            A = np.zeros((n, n)); idx = np.arange(n)
            A[idx, idx] = a_d; A[idx[1:], idx[:-1]] = a_lo[1:]; A[idx[:-1], idx[1:]] = a_up[:-1]
            return np.linalg.solve(A, rhs)
        u1 = solve(ui * ui / dx); u2 = solve(-np.ones(n))  # u = u1 + G*u2, G = dp/dx/rho
        full1 = np.r_[0, u1, 0]; full2 = np.r_[0, u2, 0]
        G = (Q - trap(full1, y)) / trap(full2, y)
        unew = full1 + G * full2
        dudx = (unew - u) / dx
        v = -np.concatenate([[0], np.cumsum(0.5 * (dudx[1:] + dudx[:-1]) * dy)])
        u = unew; x += dx
    return y, u

def core_speedup(lam_):
    # march at the physical inflow speed (profile shape depends on U_in through nu/(U x)); one fixed-point update
    Uin = U
    for _ in range(2):
        y, u = march(Uin, lam_)
        s = u[len(u) // 2] / Uin
        Uin = U / s
    return s, y, u / Uin

su_drag, y_prof, u_prof_drag = core_speedup(lam)
su_nodrag, _, u_prof_nodrag = core_speedup(0.0)
U_in = U / su_drag                         # plug inflow that gives U at the cylinder station (no cylinder)
U_in_nodrag = U / su_nodrag

# ---------------- blockage (Allen & Vincenti 1944 / Maskell-type effective velocity) ----------------
beta = D / W
def williamson_St(re):  # Williamson 1996 (laminar parallel shedding, 49<Re<178)
    return -3.3265 / re + 0.1816 + 1.6e-4 * re
# unbounded mean Cd / Cl amplitude (2-D DNS literature, see FORMULATION §6)
lit = {
    100: {"St": 0.164, "Cd": 1.33, "Cl_amp": 0.33, "src": "Park, Kwon & Choi 1998 (Re100: St 0.165, Cd 1.33, Cl' 0.332); Henderson 1995 Cd 1.35"},
    150: {"St": 0.183, "Cd": 1.32, "Cl_amp": 0.52, "src": "Inoue & Hatakeyama 2002 (Re150: St 0.183, Cd 1.32, Cl' 0.52); Henderson 1995 Cd fit 1.33"},
}
Cd_inf = lit[150]["Cd"]
eps_s = (math.pi ** 2 / 12) * beta ** 2     # potential-flow image (solid) blockage, circular cylinder
eps_w = Cd_inf * beta / 4                   # wake blockage (Maskell / Allen & Vincenti form)
eps = eps_s + eps_w
Re_eff = Re * (1 + eps)
St_inf = williamson_St(Re)
St_bounded = williamson_St(Re_eff) * (1 + eps)
Cd_bounded = Cd_inf * (1 + eps) ** 2
Cl_bounded = lit[150]["Cl_amp"] * (1 + eps) ** 2
f_shed = St_bounded * U / D
T_shed = 1 / f_shed
n_periods = 10.0 * f_shed
n_vortices = 2 * n_periods
Uc_ratio = (0.85, 0.90)
Uc = 0.875 * U
a_spacing = Uc / f_shed                     # streamwise spacing of same-sign vortices
b_ratio = (0.28, 0.40)                      # lateral/streamwise ratio, near wake laminar (Kármán 0.281; measured 0.3-0.4)
b_spacing = [r * a_spacing for r in b_ratio]

# ---------------- 3. precursor length ----------------
sigma_lin = 0.12        # linear growth rate of the global mode at Re~150 in D/U units (Stuart-Landau fits; order 0.1)
A_seed = 1e-3           # seeded asymmetric perturbation amplitude relative to saturated Cl
t_grow = math.log(1 / A_seed) / sigma_lin
t_settle = 40.0         # nonlinear saturation + amplitude settling to <1% (D/U)
t_flow_through = (x_out - x_in) / U / T_conv
precursor_DU = max(t_grow + t_settle, 2 * t_flow_through) * 1.5
precursor_s = precursor_DU * T_conv

# ---------------- 4. dye: travel from needle to front (potential flow around the cylinder, uniform U) ----------------
a = D / 2
def vel(x, y):
    X = x - xc; r2 = X * X + y * y
    ur = 1 - a * a * (X * X - y * y) / r2 ** 2
    vr = -a * a * 2 * X * y / r2 ** 2
    return U * ur, U * vr

def travel(y0, x_target, dt=2e-3):
    x, y, t = x_needle, y0, 0.0
    while x < x_target:
        k1 = vel(x, y); k2 = vel(x + dt / 2 * k1[0], y + dt / 2 * k1[1])
        k3 = vel(x + dt / 2 * k2[0], y + dt / 2 * k2[1]); k4 = vel(x + dt * k3[0], y + dt * k3[1])
        x += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); y += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        t += dt
        if t > 200: return float("inf"), None
    return t, (x, y)

x_front0 = xc + P["dye"]["front_x_rel_cyl_D"] * D
t_travel = {f"{yn*1000:+.0f}mm": travel(yn, x_front0)[0] for yn in y_needles if abs(yn) > 1e-9}
t_release = -t_travel["+8mm"]               # DECISION: release anchored on the +/-8 mm streaks (closest to the wake)
front_at_t0 = {}
for yn in y_needles:
    if abs(yn) < 1e-9:
        front_at_t0["+0mm"] = "stagnation: streak spreads around cylinder front (x <= xc - D/2 + ~1 mm)"
        continue
    tt = travel(yn, 0.2)[0]  # not used; positions via direct integration below
# positions at t=0 of each streak front given common release time
def pos_after(y0, T, dt=2e-3):
    x, y, t = x_needle, y0, 0.0
    while t < T:
        k1 = vel(x, y); k2 = vel(x + dt / 2 * k1[0], y + dt / 2 * k1[1])
        k3 = vel(x + dt / 2 * k2[0], y + dt / 2 * k2[1]); k4 = vel(x + dt * k3[0], y + dt * k3[1])
        x += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); y += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]); t += dt
    return x
for yn in y_needles:
    if abs(yn) > 1e-9:
        front_at_t0[f"{yn*1000:+.0f}mm"] = round(pos_after(yn, -t_release), 4)

# dye front x_front(t) for t in [0,10]: outer streaks (free stream, potential flow) and wake-entrained dye (U_c)
ts = np.linspace(0, 10, 41)
x_front_outer = [pos_after(0.024, -t_release + t) for t in ts]
x_front_wake_lo = x_front0 + 0.85 * U * ts - 0.0  # entrained dye convects at U_c
x_front_wake_lo = np.where(ts * U < 1.5 * D, x_front0 + 0.5 * U * ts, x_front0 + 0.75 * D + 0.85 * U * (ts - 1.5 * D / U))
x_front_wake_hi = x_front0 + 0.90 * U * ts
x_ai_fill = xc + 0.17       # downstream end of the visible wake region (NOTES: 170 mm visible)

# ---------------- 5. dye resolution ----------------
D_m = 1e-9; Sc = nu / D_m
t_res = (x_out - x_needle) / U
sig_mol = math.sqrt(2 * D_m * t_res)
streak_w = (0.5e-3, 1.0e-3)
dx_req = streak_w[0] / 4

out = {
    "_doc": "Role B targets for 02-vortex-shedding. Generated by physics/derive.py; rationale in physics/FORMULATION.md. "
            "Status tags: assumed (declared inputs), literature, model-conditioned (2-D model + corrections), derived.",
    "version": "v001",
    "inputs_assumed": {"nu": nu, "rho": rho, "U_approach_dye_plane": U, "D": D, "W": W, "h": h, "L": L, "x_cyl": xc,
                       "Re": Re, "D_over_U_s": T_conv},
    "decisions": {
        "U_definition": "U = 0.015 m/s is the approach velocity at the dye plane (mid-depth) on the centreline at the cylinder station "
                        "(empty channel). Re = U D/nu = 150 kept. The 2-D velocity field IS the mid-depth field (dye moves with it). "
                        f"Depth-averaged bulk velocity at the cylinder = U/{shape_mid:.3f} = {U/shape_mid:.4f} m/s (derived, not used in the model).",
        "friction": f"ADD linear drag -lambda*u to the 2-D momentum eq. lambda = (pi^2/4) nu/h^2 = {lam:.5f} 1/s "
                    f"(range 2.47-3.0 nu/h^2 = {lam_mode:.5f}-{lam_bulk:.5f}). Benchmarks/controls run with lambda = 0.",
        "inflow": f"plug (uniform) at x=-0.30 with U_in = {U_in:.5f} m/s (gives {U} m/s centreline at x_cyl in the empty channel "
                  f"with drag; without drag U_in would be {U_in_nodrag:.5f}). Solver must verify with an empty-channel run (<=1 %).",
        "outflow": "x=+0.30: u_n zero-gradient (Neumann), p = 0 (Dirichlet); tracers zero-gradient. Outflow is 34 D downstream.",
    },
    "shallow_friction": {"zeta_dye": zeta_dye, "u_mid_over_u_bar": shape_mid, "lambda_bulk_3nu_h2": lam_bulk,
                         "lambda_mode_pi2_4_nu_h2": lam_mode, "lambda_used": lam, "c_f": c_f, "S_shallow": S_shallow,
                         "S_critical_low": S_c, "S_over_Sc": S_shallow / S_c, "vorticity_decay_factor_10s": decay_10s,
                         "friction_vs_viscous_at_D": drag_vs_visc, "brinkman_length_m": brinkman_len},
    "inflow_profile": {"core_speedup_at_xcyl_with_drag": su_drag, "core_speedup_no_drag": su_nodrag, "U_in_plug": U_in},
    "blockage": {"beta": beta, "eps_solid": eps_s, "eps_wake": eps_w, "eps_total": eps, "Re_eff": Re_eff,
                 "method": "U_eff = U(1+eps); St_b = St_W(Re_eff)(1+eps); Cd_b = Cd_inf (1+eps)^2 (Allen & Vincenti 1944; Maskell 1963)"},
    "targets_production": {
        "status": "model-conditioned (2-D, blockage-corrected; drag effect expected < 1 % on St)",
        "St_unbounded_Williamson": St_inf, "St": St_bounded, "St_tol_rel": 0.03,
        "f_shed_Hz": f_shed, "T_shed_s": T_shed, "periods_in_10s": n_periods, "vortices_shed_in_10s": n_vortices,
        "Cd_mean": Cd_bounded, "Cd_tol_rel": 0.05, "Cl_amp": Cl_bounded, "Cl_amp_tol_rel": 0.10,
        "Uc_over_U": list(Uc_ratio), "vortex_spacing_a_m": a_spacing, "vortex_spacing_a_over_D": a_spacing / D,
        "row_spacing_b_m": b_spacing, "row_spacing_b_over_a": list(b_ratio),
        "formation_length_over_D": [1.0, 1.5],
    },
    "dye": {"Sc": Sc, "molecular_sigma_over_transit_m": sig_mol, "streak_width_m": list(streak_w),
            "grid_spacing_required_m": dx_req, "source": "Dirichlet c = 1 in a disk r = 0.35 mm (needle OD/2) at each tip; one tracer per colour group",
            "x_front_t0": x_front0, "travel_time_needle_to_front_s": t_travel, "t_release_s": t_release,
            "front_x_at_t0_per_needle": front_at_t0,
            "x_front_t": {"t_s": ts.tolist(), "outer_streak_y24mm": [float(v) for v in x_front_outer],
                          "wake_dye_low": x_front_wake_lo.tolist(), "wake_dye_high": x_front_wake_hi.tolist()},
            "visible_wake_end_x": x_ai_fill,
            "front_at_2p5s_outer": float(np.interp(2.5, ts, x_front_outer)),
            "front_at_10s_outer": float(x_front_outer[-1]),
            "front_at_10s_wake": [float(x_front_wake_lo[-1]), float(x_front_wake_hi[-1])],
            "AI_defect_check": "dye cannot reach x > front_at_t(t) + 5 mm (outer) at time t; AI fills to x_cyl+0.17 by 2.5 s -> defect"},
    "precursor": {"growth_rate_DU": sigma_lin, "seed_amp": A_seed, "t_grow_DU": t_grow, "t_settle_DU": t_settle,
                  "flow_through_DU": t_flow_through, "required_DU": precursor_DU, "required_s": precursor_s,
                  "unseeded_impulsive_start_s_estimate": (math.log(1e10) / sigma_lin + t_settle) * T_conv,
                  "periodicity_criterion": "over the last 5 periods: |dCl_amp|/Cl_amp < 1 %, |dSt|/St < 0.5 %, |dCd_mean| < 0.5 %"},
    "benchmarks": {
        "schaefer_turek_2D2": {"src": "Schäfer & Turek 1996, Notes Num. Fluid Mech. 52, 547-566 (DFG benchmark 2D-2)",
                               "geometry": {"channel": [2.2, 0.41], "D": 0.1, "center": [0.2, 0.2]},
                               "nu": 1e-3, "rho": 1.0, "inflow": "parabolic U(0,y)=4 Um y(H-y)/H^2, Um = 1.5, Ubar = 1.0",
                               "walls": "no-slip y=0, y=0.41 and cylinder", "outflow": "do-nothing / p=0 at x=2.2",
                               "Re": 100, "coeff_def": "Cd=2Fd/(rho Ubar^2 D), Cl=2Fl/(rho Ubar^2 D), St=D f/Ubar",
                               "St": [0.2950, 0.3050], "Cd_max": [3.22, 3.24], "Cl_max": [0.99, 1.01],
                               "dp_front_back_at_t0_plus_half_period": [2.46, 2.50]},
        "unbounded": {"Re100": {"St_Williamson": williamson_St(100), **lit[100]},
                      "Re150": {"St_Williamson": williamson_St(150), **lit[150]},
                      "tolerance": {"St_rel": 0.02, "Cd_rel": 0.03},
                      "domain_requirement": "lateral width >= 40 D (blockage <= 2.5 %), inflow >= 15 D upstream, outflow >= 30 D, lambda = 0, uniform inflow"},
        "hopf_onset": {"Re_c": [46.0, 47.5], "src": "Provansal, Mathis & Boyer 1987 (Re_c~47); Jackson 1987 (46.2); Williamson 1989 (~49 exp.)",
                       "control": "Re = 40 must decay to a steady symmetric wake (Cl_amp -> 0), Re = 55 must shed; unbounded settings, lambda = 0"},
        "convergence": {"levels": "3 levels, refinement ratio 2 (e.g. Basilisk max level 10/11/12 on 0.6 m -> D/dx = 17/34/68)",
                        "finest_D_over_dx_min": 64, "observed_order_min": 1.0, "observed_order_expected": 2.0,
                        "GCI_fine_St_max": 0.01, "GCI_fine_Cd_max": 0.02, "GCI_fine_Cl_max": 0.03,
                        "dye_field": "L1 difference of tracer fields (frames 0,120,240) level 12 vs 13 < 0.5 x (level 11 vs 12); tracer regions dx <= 0.125 mm"},
        "divergence": {"max_abs_div_times_D_over_U": 1e-3, "rms_div_times_D_over_U": 1e-5},
        "tracer": {"mass_balance_rel": 0.01, "bounds": [-1e-3, 1.001], "def": "|sum(c dV) - (injected - outflow)| / injected"},
        "energy": "no energy injection: with lambda>0 and fixed inflow, domain KE must be statistically stationary after precursor (drift < 1 %/10 s)",
    },
}
json.dump(out, open(os.path.join(HERE, "benchmarks.json"), "w"), indent=1, default=float)

# ---------------- visual ----------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
fig = plt.figure(figsize=(15, 9.5), dpi=110)
fig.suptitle("v001 | expectations | physics   (02-vortex-shedding, Re = 150, D = 10 mm, U = 15 mm/s; model-conditioned targets)",
             fontsize=13, weight="bold")
gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1])
# (a) channel map
ax = fig.add_subplot(gs[0, :])
ax.add_patch(Rectangle((x_in, -W / 2), L, W, fc="#eef3f8", ec="k"))
ax.add_patch(Circle((xc, 0), a, fc="k"))
for yn in y_needles: ax.plot([x_needle - 0.006, x_needle], [yn, yn], color="0.4", lw=2)
cols = ["#d4145a", "#008bd8", "#f29e00"]
for i, (t, ls) in enumerate([(0, "-"), (2.5, "--"), (5, "-."), (10, ":")]):
    xo = np.interp(t, ts, x_front_outer); xw = np.interp(t, ts, x_front_wake_lo)
    ax.axvline(xo, color="tab:green", ls=ls, lw=1.6, label=f"dye front outer t={t:g}s: x={xo*1000:.0f} mm")
    ax.plot([xw, xw], [-0.012, 0.012], color="tab:purple", ls=ls, lw=3)
# schematic vortex street at t = 10 s (positions only, phase arbitrary)
b = 0.34 * a_spacing; xs = xc + 1.5 * D
k = 0
while xs < x_front_wake_lo[-1]:
    ax.add_patch(Circle((xs, (b / 2) * (1 if k % 2 == 0 else -1)), 0.22 * a_spacing / 2, fc="none", ec=cols[k % 3], lw=1.5))
    xs += a_spacing / 2; k += 1
ax.axvspan(xc, x_ai_fill, ymin=0.02, ymax=0.06, color="red", alpha=0.5)
ax.text(xc + 0.002, -W / 2 + 0.008, "AI: wake filled to here by ~2.5 s (defect)", color="red", fontsize=8)
ax.annotate("", xy=(xc + 1.5 * D + a_spacing, 0.04), xytext=(xc + 1.5 * D, 0.04), arrowprops=dict(arrowstyle="<->"))
ax.text(xc + 1.5 * D + 0.002, 0.043, f"a = Uc/f = {a_spacing*1000:.0f} mm, b = {b*1000:.0f} mm", fontsize=8)
ax.set_xlim(x_in, x_out); ax.set_ylim(-W / 2, W / 2); ax.set_aspect("equal")
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title(f"(a) top view: needles x={x_needle}, cylinder x={xc}; release t={t_release:.2f}s; purple = wake-dye front (U_c), circles = expected vortices at t=10 s",
             fontsize=9)
ax.legend(fontsize=7, loc="upper right", ncol=2)
# (b) St(Re)
ax = fig.add_subplot(gs[1, 0])
re = np.linspace(50, 180, 200)
ax.plot(re, williamson_St(re), "k", label="Williamson 1996 (unbounded)")
st_b = [williamson_St(r * (1 + eps)) * (1 + eps) for r in re]
ax.plot(re, st_b, "tab:blue", label=f"blockage D/W=1/12 corrected (eps={eps:.3f})")
ax.fill_between(re, np.array(st_b) * 0.97, np.array(st_b) * 1.03, color="tab:blue", alpha=0.15, label="±3 % production tol")
for r in (100, 150): ax.plot(r, lit[r]["St"], "o", color="tab:orange")
ax.plot([], [], "o", color="tab:orange", label="2-D DNS (Park 98 / Inoue 02)")
ax.plot(Re, St_bounded, "*", ms=14, color="tab:red", label=f"target St={St_bounded:.4f}, f={f_shed:.3f} Hz")
ax.axvspan(46, 47.5, color="0.8"); ax.text(52, 0.125, "Re_c≈46-47 (left)", fontsize=8)
ax.set_xlabel("Re"); ax.set_ylabel("St"); ax.set_title("(b) Strouhal target", fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=.3)
# (c) inflow profile at cylinder station
ax = fig.add_subplot(gs[1, 1])
ax.plot(y_prof * 1000, u_prof_drag * U_in * 1000, "tab:blue", label=f"plug U_in={U_in*1000:.2f} mm/s + drag λ={lam:.4f}/s")
ax.plot(y_prof * 1000, u_prof_nodrag * U_in_nodrag * 1000, "tab:gray", ls="--", label=f"plug U_in={U_in_nodrag*1000:.2f} mm/s, λ=0")
ax.axhline(U * 1000, color="k", lw=.8); ax.text(-58, U * 1000 + 0.3, "U = 15 mm/s at cylinder", fontsize=8)
for yn in y_needles: ax.axvline(yn * 1000, color="0.85", lw=.8)
ax.set_xlabel("y [mm]"); ax.set_ylabel("u at x_cyl, no cylinder [mm/s]")
ax.set_title(f"(c) lateral profile at x_cyl (2-D mid-depth); u_mid/u_bar = {shape_mid:.3f}", fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=.3)
# (d) dye front vs time
ax = fig.add_subplot(gs[1, 2])
ax.plot(ts, (np.array(x_front_outer) - xc) / D, "tab:green", label="outer streak front (y=±24 mm)")
ax.fill_between(ts, (x_front_wake_lo - xc) / D, (x_front_wake_hi - xc) / D, color="tab:purple", alpha=.3, label="wake-entrained dye front")
ax.axhline((x_ai_fill - xc) / D, color="red", ls="--", label="AI: filled by ~2.5 s")
ax.plot([2.5], [(x_ai_fill - xc) / D], "rx", ms=10)
ax.set_xlabel("t [s]"); ax.set_ylabel("(x_front − x_cyl)/D"); ax.grid(alpha=.3)
ax.set_title(f"(d) dye front: {np.interp(2.5, ts, x_front_outer)*1000:.0f} mm at 2.5 s, {x_front_outer[-1]*1000:.0f} mm at 10 s", fontsize=10)
ax.legend(fontsize=7)
txt = (f"St={St_bounded:.4f} (unbounded {St_inf:.4f})  f={f_shed:.3f} Hz  T={T_shed:.2f} s  periods/10s={n_periods:.2f}  vortices/10s≈{n_vortices:.1f}\n"
       f"Cd={Cd_bounded:.3f}  Cl'={Cl_bounded:.3f}  Uc/U=0.85-0.90  a={a_spacing*1000:.1f} mm  λ={lam:.5f}/s  S={S_shallow:.4f}<<0.2  precursor≥{precursor_s:.0f} s")
fig.text(0.5, 0.005, txt, ha="center", fontsize=9, family="monospace")
fig.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig(os.path.join(HERE, "visuals", "v001_expectations.png"))
print(json.dumps({k: out[k] for k in ("decisions", "shallow_friction", "inflow_profile", "blockage", "targets_production", "precursor")}, indent=1, default=float))
print(json.dumps(out["dye"]["travel_time_needle_to_front_s"]), out["dye"]["t_release_s"], out["dye"]["front_x_at_t0_per_needle"],
      out["dye"]["front_at_2p5s_outer"], out["dye"]["front_at_10s_outer"], out["dye"]["front_at_10s_wake"])
