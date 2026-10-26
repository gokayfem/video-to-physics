"""Post-processing: convergence tables, deformation.npz on the scene grid, meta.json, visuals v001_*."""
import json, subprocess, hashlib, datetime, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe
from setup_model import load_params, schedule, body_props, fracs_time, ROOT, PJ, sha

OUT = ROOT / "solver" / "out"; VIS = ROOT / "solver" / "visuals"; VIS.mkdir(exist_ok=True)
P, raw = load_params()
SP = json.loads((ROOT / "config" / "scene_params.json").read_text())
ST = json.loads((OUT / "static.json").read_text())
SPATH = np.load(OUT / "static_paths.npz")
BM = json.loads((ROOT / "physics" / "benchmarks.json").read_text())
g = P["g"]; rt = P["tip_body"]["rigid_top_strip_m"]
L = P["strip"]["free_length_m"]; Lf = L - rt
EI = P["strip"]["E_Pa"] * P["strip"]["width_m"] * P["strip"]["thickness_m"] ** 3 / 12
times, sched_src = schedule(P); tau = P["loading"]["ramp_s"]

D = {k: np.load(OUT / f"dyn_{k}.npz") for k in ["N67_dt960", "N134_dt960", "N268_dt960", "N268_dt480", "N268_dt1920", "N268_dt960_nodamp_rho1"]}
FIN = D["N268_dt1920"]  # finest = production
t = FIN["t"]

# ---------------- dynamic convergence
def traj_err(a, b):
    return float(np.max(np.hypot(a["ytop"] - b["ytop"], a["ztop"] - b["ztop"])) * 1e3)
conv_dyn = {
    "mesh_at_dt960_maxdev_tip_traj_vs_N268_mm": {k: traj_err(D[k], D["N268_dt960"]) for k in ["N67_dt960", "N134_dt960"]},
    "dt_at_N268_maxdev_tip_traj_vs_dt1920_mm": {k: traj_err(D[k], FIN) for k in ["N268_dt480", "N268_dt960"]},
    "ytop_10s_mm": {k: float(D[k]["ytop"][-1] * 1e3) for k in D},
}
# buckling time: first time |y_top| > 10 mm ; supercritical time: tip load crosses W_cr
ncr = ST["D_full_model"]["n_cr_shooting_ODE"]
Wcr = ST["D_full_model"]["W_cr_shooting_N"]
W_t = np.array([body_props(P, fracs_time(tt, times, tau))[1] * g for tt in t])
ytop = FIN["ytop"]; ztop = FIN["ztop"]
def first(mask):
    i = np.argmax(mask); return float(t[i]) if mask.any() else None
t_super = first(W_t >= Wcr)
t_b10 = first(np.abs(ytop) > 0.010)
t_b50 = first(np.abs(ytop) > 0.050)
for k in D:
    conv_dyn.setdefault("t_buckle_10mm_s", {})[k] = float(D[k]["t"][np.argmax(np.abs(D[k]["ytop"]) > 0.01)])
E = FIN["e_T"] + FIN["e_U"] + FIN["e_V"]
nd = D["N268_dt960_nodamp_rho1"]
En = nd["e_T"] + nd["e_U"] + nd["e_V"]
energy = {
    "production_max_abs_balance_residual_J": float(np.max(np.abs(FIN["e_resid"]))),
    "production_Wext_J": float(FIN["e_Wext"][-1]), "production_Wdiss_J": float(FIN["e_Wdiss"][-1]),
    "production_resid_rel_to_max_strain_energy": float(np.max(np.abs(FIN["e_resid"])) / np.max(FIN["e_U"])),
    "undamped_rho1_max_abs_balance_residual_J": float(np.max(np.abs(nd["e_resid"]))),
    "undamped_rho1_resid_rel_to_max_strain_energy": float(np.max(np.abs(nd["e_resid"])) / np.max(nd["e_U"])),
    "_doc": "residual = E(t) - E(0) - int(dV/dt|_q + 1/2 w^T dM/dt|_q w) dt + int(P_diss) dt ; E = T + U_bend + V_grav. Includes the generalized-alpha numerical dissipation and the mass-step at weight release.",
}
# post-buckled sway frequency (after last weight released, t > 5 s): zero crossings of ytop - mean
m = t > 5.2
yy = ytop[m] - ytop[m].mean()
zc = np.where(np.diff(np.sign(yy)) != 0)[0]
f_sway = float((len(zc) - 1) / 2 / (t[m][zc[-1]] - t[m][zc[0]])) if len(zc) > 2 else None

# ---------------- deformation.npz on the scene grid (11 x 301)
R = SP["ruler"]; nx, nz = R["mesh_segments_width"], R["mesh_segments_length"]
xs = np.linspace(-R["width_m"] / 2, R["width_m"] / 2, nx + 1)
s_loc = np.linspace(0, R["length_m"], nz + 1)
z_jaw = SP["vise"]["jaw_top_z_m"]; z_bot = R["bottom_z_m"]; s_cl = z_jaw - z_bot
N = int(FIN["N"]); h = float(FIN["h"])
th = FIN["th"]; Y = FIN["Y"]; Z = FIN["Z"]
T = len(t)
cy = np.zeros((T, nz + 1)); cz = np.zeros((T, nz + 1)); cth = np.zeros((T, nz + 1))
node_th = np.concatenate([np.zeros((T, 1)), 0.5 * (th[:, :-1] + th[:, 1:]), th[:, -1:]], 1)
for j, s in enumerate(s_loc):
    if s <= s_cl + 1e-12:
        cy[:, j] = 0.0; cz[:, j] = z_bot + s; cth[:, j] = 0.0
    else:
        sf = s - s_cl
        if sf <= Lf + 1e-12:
            u = sf / h; k0 = min(int(np.floor(u + 1e-9)), N - 1); a = u - k0
            cy[:, j] = (1 - a) * Y[:, k0] + a * Y[:, k0 + 1]
            cz[:, j] = z_jaw + (1 - a) * Z[:, k0] + a * Z[:, k0 + 1]
            cth[:, j] = (1 - a) * node_th[:, k0] + a * node_th[:, k0 + 1]
        else:
            d = sf - Lf
            cy[:, j] = Y[:, N] + d * np.sin(th[:, -1]); cz[:, j] = z_jaw + Z[:, N] + d * np.cos(th[:, -1]); cth[:, j] = th[:, -1]
cy += R["center_xy_m"][1]
pos = np.zeros((T, nz + 1, nx + 1, 3), np.float32)
pos[..., 0] = R["center_xy_m"][0] + xs[None, None, :]
pos[..., 1] = cy[:, :, None]; pos[..., 2] = cz[:, :, None]
# tip body rigid transform: origin O (tab bottom = top of flexible part) ; rotation about +X by -theta_N
O0 = np.array([R["center_xy_m"][0], R["center_xy_m"][1], z_jaw + Lf])
Ot = np.stack([np.full(T, O0[0]), cy[:, 0] * 0 + R["center_xy_m"][1] + Y[:, N], z_jaw + Z[:, N]], 1)
rx = -th[:, -1]
Mtb = np.zeros((T, 4, 4)); c, s_ = np.cos(rx), np.sin(rx)
Mtb[:, 0, 0] = 1; Mtb[:, 1, 1] = c; Mtb[:, 1, 2] = -s_; Mtb[:, 2, 1] = s_; Mtb[:, 2, 2] = c; Mtb[:, 3, 3] = 1
Mtb[:, :3, 3] = Ot - np.einsum("tij,j->ti", Mtb[:, :3, :3], O0)
nw_on = np.array([np.sum(fracs_time(tt, times, tau)) for tt in t])
state = np.where(W_t < Wcr, 0, np.where(np.abs(ytop) < 0.005, 1, 2)).astype(np.int8)
np.savez_compressed(OUT / "deformation.npz",
                    t=t, frame=t * 24.0, positions=pos, centreline_y=cy, centreline_z=cz, theta_rad=cth,
                    s_local=s_loc, x_across=xs, tip_load_N=W_t, n_weights_seated=nw_on, buckling_state=state,
                    tipbody_matrix_world=Mtb, tipbody_origin_world=Ot, tipbody_rot_x_rad=rx,
                    strip_top_world=np.stack([np.full(T, O0[0]), cy[:, -1], cz[:, -1]], 1))

# ---------------- visuals
plt.rcParams.update({"font.size": 9})
# 1 load-deflection
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
p_ = np.sin(np.linspace(1e-4, np.radians(150), 400) / 2); K = ellipk(p_ ** 2); Ee = ellipe(p_ ** 2)
ax[0].plot(2 * p_ / K, (2 * K / np.pi) ** 2, "k-", lw=2, label="analytic elastica (elliptic integrals)")
ax[0].plot(2 - 2 * Ee / K, (2 * K / np.pi) ** 2, "k--", lw=1.2, label="analytic: vertical drop / L")
for Nn, mk in [(35, "o"), (280, "x")]:
    Pp = SPATH[f"elastica_N{Nn}_P"] / ST["A_euler_tip_load_L280"]["analytic_N"]
    ax[0].plot(SPATH[f"elastica_N{Nn}_y"][::6] / L, Pp[::6], mk, ms=4, mfc="none", label=f"solver N={Nn}: lateral")
    ax[0].plot((L - SPATH[f"elastica_N{Nn}_z"][::6]) / L, Pp[::6], mk, ms=3, color="C3" if Nn == 280 else "C2", mfc="none")
ax[0].axhline(1, color="0.6", lw=0.8)
ax[0].set_xlabel("tip deflection / L"); ax[0].set_ylabel("P / P_cr (P_cr = pi^2 EI / 4L^2)")
ax[0].set_title("v001 | static | ideal clamped-free elastica: solver vs analytic", fontsize=9); ax[0].legend(fontsize=7)
for Nn in [67, 268]:
    ax[1].plot(np.abs(SPATH[f"actual_N{Nn}_ytop"]) * 1e3, SPATH[f"actual_N{Nn}_lam"], lw=2 if Nn == 268 else 1, label=f"solver N={Nn}, a0=L/1000")
for kk, cc in [("L/500", "C4"), ("L/5000", "C5"), ("L/1000000", "C7")]:
    ax[1].plot(np.abs(SPATH["imp_" + kk + "_y"]) * 1e3, SPATH["imp_" + kk + "_lam"], ":", color=cc, label=f"a0={kk} (flex top)")
bi = BM["nonlinear_corrected"]["imperfect_L1000"]
ax[1].plot([r["x_top_mm"] for r in bi], [r["n"] for r in bi], "ks", ms=5, mfc="none", label="physics/benchmarks v002 (B, shooting)")
ax[1].axhline(ncr, color="r", lw=0.8); ax[1].text(5, ncr + 0.15, f"n_cr = {ncr:.3f}  (W_cr = {Wcr:.4f} N)", color="r")
ax[1].set_xlabel("strip top lateral deflection |Y| [mm]"); ax[1].set_ylabel("number of 20 g weights (continuous)")
ax[1].set_title("v001 | static | actual model (L_f=0.268, tip body, self-weight)", fontsize=9); ax[1].legend(fontsize=7); ax[1].set_ylim(0, 12.5)
fig.tight_layout(); fig.savefig(VIS / "v001_load_deflection.png", dpi=110); plt.close(fig)

# 2 convergence
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
A = ST["A_euler_tip_load_L280"]; A2 = ST["A2_rigid_offset_e30mm_L268"]; Dd = ST["D_full_model"]
ax[0].loglog(A["N"], np.abs(A["rel_err"]), "o-", label="Euler tip load (vs pi^2EI/4L^2)")
ax[0].loglog(A["N"], np.abs(A2["rel_err"]), "s-", label="rigid offset e=30 mm (vs kL tan kL = L/e)")
ax[0].loglog(Dd["N"], np.abs(Dd["rel_err_n"]), "^-", label="full model n_cr (vs ODE shooting)")
nn = np.array([35, 560]); ax[0].loglog(nn, 0.2 / nn ** 2, "k:", lw=0.8, label="O(h^2)"); ax[0].loglog(nn, 0.02 / nn, "k--", lw=0.8, label="O(h)")
ax[0].axhline(5e-3, color="r", lw=0.8); ax[0].text(40, 6e-3, "0.5 % target", color="r")
ax[0].set_xlabel("segments N"); ax[0].set_ylabel("|relative error| of critical load"); ax[0].legend(fontsize=7)
ax[0].set_title("v001 | P_cr mesh convergence", fontsize=9)
for k, st in [("N67_dt960", ":"), ("N134_dt960", "--"), ("N268_dt480", "-."), ("N268_dt1920", "-")]:
    ax[1].plot(D[k]["t"], D[k]["ytop"] * 1e3, st, label=k)
ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("strip top Y [mm]"); ax[1].legend(fontsize=7)
ax[1].set_title("v001 | dynamic tip trajectory, mesh & dt levels", fontsize=9)
for k in ["N67_dt960", "N134_dt960"]:
    ax[2].semilogy(t, np.abs(D[k]["ytop"] - D["N268_dt960"]["ytop"]) * 1e3 + 1e-9, label=f"{k} vs N268 (mesh)")
for k in ["N268_dt480", "N268_dt960"]:
    ax[2].semilogy(t, np.abs(D[k]["ytop"] - FIN["ytop"]) * 1e3 + 1e-9, label=f"{k} vs dt1920 (time)")
ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("|dY_top| [mm]"); ax[2].legend(fontsize=7); ax[2].set_ylim(1e-6, 100)
ax[2].set_title("v001 | dynamic convergence error", fontsize=9)
fig.tight_layout(); fig.savefig(VIS / "v001_convergence.png", dpi=110); plt.close(fig)

# 3 dynamic time series + energy
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
ax[0].plot(t, ytop * 1e3, label="strip top Y"); ax[0].plot(t, (ztop - (Lf + rt)) * 1e3, label="strip top dZ")
for tt in times: ax[0].axvline(tt, color="0.8", lw=0.6)
ax[0].axvline(t_super, color="r", lw=0.8); ax[0].set_ylabel("mm"); ax[0].legend(fontsize=7)
ax[0].set_title(f"v001 | dynamic N=268 dt=0.52 ms | weights seated {', '.join(f'{x:.1f}' for x in times)} s (grey); W>W_cr at red", fontsize=9)
ax[1].plot(t, W_t, label="tip load W(t) [N]"); ax[1].axhline(Wcr, color="r", lw=0.8, label="W_cr"); ax[1].legend(fontsize=7)
et = FIN["e_t"]
ax[2].plot(et, FIN["e_U"] * 1e3, label="bending U"); ax[2].plot(et, FIN["e_T"] * 1e3, label="kinetic T")
ax[2].plot(et, FIN["e_Wdiss"] * 1e3, label="dissipated"); ax[2].plot(et, FIN["e_resid"] * 1e3, "k", label="balance residual")
ax[2].set_ylabel("mJ"); ax[2].set_xlabel("t [s]"); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(VIS / "v001_dynamic_timeseries.png", dpi=110); plt.close(fig)

# 4 preview MP4: side view (camera along +X looking at the Y-Z plane), true 24 fps, 241 frames
frames_dir = OUT / "_frames"; shutil.rmtree(frames_dir, ignore_errors=True); frames_dir.mkdir()
TB = P["tip_body"]; hw = TB["weight_thickness_m"]; z0w = TB["first_weight_base_offset_m"]
fig = plt.figure(figsize=(9.6, 5.44), dpi=100)
for n in range(241):
    i = 5 * n
    fig.clf(); a = fig.add_axes([0.06, 0.08, 0.9, 0.86])
    a.add_patch(plt.Rectangle((-0.03, 0.063), 0.06, 0.032, color="#1f5f5f"))
    a.plot(cy[i] * 1e0, cz[i], color="0.35", lw=2.2)
    Rm = Mtb[i, 1:3, 1:3]; o = Mtb[i, 1:3, 3]
    def tr(yv, zv):
        p = np.stack([yv, zv]); return (Rm @ p).T + o
    zt = z_jaw + Lf
    for (y0, y1, z_0, z_1, col) in [(-0.0125, 0.0125, zt + 0.012, zt + 0.015, "#999"), (-0.00125 - 0.002, -0.00025, zt, zt + 0.012, "#aaa"), (0.00025, 0.00225, zt, zt + 0.012, "#aaa"), (-0.002, 0.002, zt + 0.015, zt + 0.055, "#777")]:
        q = tr(np.array([y0, y1, y1, y0]), np.array([z_0, z_0, z_1, z_1])); a.fill(q[:, 0], q[:, 1], color=col)
    fr = fracs_time(t[i], times, tau)
    for j, f in enumerate(fr):
        if f > 0:
            zb = zt + z0w + j * hw
            q = tr(np.array([-0.0125, 0.0125, 0.0125, -0.0125]), np.array([zb, zb, zb + hw * 0.95, zb + hw * 0.95]))
            a.fill(q[:, 0], q[:, 1], color="#c9a050", alpha=0.35 + 0.65 * f)
    a.set_xlim(-0.33, 0.12); a.set_ylim(0.05, 0.43); a.set_aspect("equal")
    a.set_xlabel("world Y [m]  (-Y = camera side)"); a.set_ylabel("world Z [m]")
    a.text(0.01, 0.97, f"v001 | frame {n:03d}  t = {t[i]:.3f} s | solver preview (side view, not render) | weights seated {int(round(nw_on[i] - 1e-9 + 0.0)) if nw_on[i] % 1 < 1e-6 else nw_on[i]:.2f} | W = {W_t[i]:.3f} N (W_cr {Wcr:.3f})",
           transform=a.transAxes, fontsize=7.5, va="top")
    a.text(0.01, 0.92, f"strip top Y = {ytop[i]*1e3:7.1f} mm   state = {['subcritical', 'supercritical', 'buckled'][state[i]]}", transform=a.transAxes, fontsize=7.5, va="top")
    fig.savefig(frames_dir / f"f{n:04d}.png")
plt.close(fig)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(frames_dir / "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=960:544", str(VIS / "v001_preview_side.mp4")], check=True)
for n in [0, 96, 120, 144, 240]:
    shutil.copy(frames_dir / f"f{n:04d}.png", VIS / f"v001_still_f{n:03d}.png")
shutil.rmtree(frames_dir)

# ---------------- meta.json
code_sha = {f: sha(ROOT / "solver" / f) for f in ["rod.py", "setup_model.py", "static_study.py", "dynamic.py", "post.py"]}
meta = dict(
    version="v001", created_utc=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    params_sha256=sha(PJ), solver_sha256=code_sha, benchmarks_sha256=sha(ROOT / "physics" / "benchmarks.json"),
    model="geometrically exact planar Kirchhoff rod (inextensible), discrete-elastica variational FD (link angles), rigid eccentric tip body, self-weight; bending about the strip's weak axis in the world Y-Z plane",
    production_run=dict(N=N, h_m=h, dt_s=float(FIN["dt"]), integrator="generalized-alpha", rho_inf=float(FIN["rho_inf"]), newton_tol_rad=1e-9),
    sampling=dict(rate_hz=120, doc="sample k at t = k/120 s; frame n (24 fps) = sample 5n; samples 5n+1..5n+4 are the 4 subframes", n_samples=T, frames=241),
    schedule=dict(source=sched_src, weight_contact_start_s=[float(x) for x in times], ramp_s=tau, ramp_shape="smootherstep 6x^5-15x^4+10x^3 of (t-t0)/tau (contact force)",
                  weight_release_s=[float(x + tau) for x in times], mass_joins_at="release (t0+tau)",
                  slot_offset_above_tab_bottom_m=[float(z0w + j * hw) for j in range(len(times))],
                  doc="weight j sits in slot j of the stack (bottom at the given offset along the strip tangent above the tab bottom O); the scene should seat weight j on the peg at t0_j and release it at t0_j+tau; before t0_j the hand carries it (no load)."),
    grid=dict(rows=nz + 1, cols=nx + 1, doc="positions[t, row, col, xyz] world metres; row = 1 mm along strip from its bottom end (z=0.075), col = across width x in [-12.5, +12.5] mm; mid-surface (solidify +-t/2 along normal n = (0, cos th, -sin th))",
              clamped_rows="0..20 (fixed)", flexible_rows="20..288", rigid_rows="288..300 (inside platform tabs)"),
    tipbody=dict(origin_undeformed_world=O0.tolist(), doc="tipbody_matrix_world[t] maps undeformed world coords of the platform/peg/seated weights to deformed ones (rotation about +X by tipbody_rot_x_rad about O, then translation)"),
    buckling_state_codes={"0": "W < W_cr (subcritical)", "1": "W >= W_cr, |Y_top| < 5 mm", "2": "buckled |Y_top| >= 5 mm"},
    results=dict(n_cr=ncr, W_cr_N=Wcr, t_supercritical_s=t_super, t_buckle_Ytop_10mm_s=t_b10, t_Ytop_50mm_s=t_b50,
                 ytop_10s_mm=float(ytop[-1] * 1e3), ztop_drop_10s_mm=float((Lf + rt - ztop[-1]) * 1e3), theta_top_10s_deg=float(np.degrees(th[-1, -1])),
                 max_abs_ytop_mm=float(np.max(np.abs(ytop)) * 1e3), postbuckled_sway_freq_Hz=f_sway,
                 static_n7_ytop_mm=ST["C_actual_static"]["268"]["ytop_mm_n7"]),
    convergence=dict(static={k: ST[k] for k in ["A_euler_tip_load_L280", "A2_rigid_offset_e30mm_L268", "D_full_model"]},
                     elastica=ST["B_elastica"], actual_static={k: {kk: v for kk, v in ST["C_actual_static"][k].items() if not kk.startswith("ytop_mm_n") or kk in ("ytop_mm_n6", "ytop_mm_n7")} for k in ST["C_actual_static"]},
                     imperfection_sensitivity=ST["C_imperfection_sensitivity_N268"], dynamic=conv_dyn),
    energy=energy,
    assumptions="see params/physics.json provenance; damping Rayleigh alpha=0.063/s on M, beta=3.2e-4 s on bending K + quadratic air drag (Cd 1.9); imperfection L/1000 bow toward -Y; mass joins at release; screws' small lateral eccentricity neglected; peg/weights rigidly attached (no sliding); elastic material (spring temper per FORMULATION).",
)
(OUT / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
print(json.dumps(dict(results=meta["results"], dyn=conv_dyn, energy=energy), indent=1, default=float))
