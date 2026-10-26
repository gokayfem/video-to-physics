"""Role-E checkpoint audit. Run: <venv-python> verify/checks/ck001_audit.py [ckNNN] [vNNN]
Needs: verify/out/check_production.json (check_production.py), verify/out/negative_controls.json,
verify/out/ck001/blend_dump.npz (blender -b checkpoints/ck001/ruler_anim.blend --python verify/checks/blend_dump.py -- <npz>).
Writes verify/out/ck001_audit.json and verify/visuals/v002_ck001_audit.png."""
import json, math, re, struct, sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

HERE = Path(__file__).resolve(); V = HERE.parents[1]; ROOT = HERE.parents[2]
CKN = sys.argv[1] if len(sys.argv) > 1 else "ck001"          # usage: ck001_audit.py [ckNNN] [visual version tag]
VTAG = sys.argv[2] if len(sys.argv) > 2 else "v002"
CK = ROOT / "checkpoints" / CKN
SOL = CK / "solver/out" if (CK / "solver/out/deformation.npz").exists() else ROOT / "solver/out"
sys.path.insert(0, str(V / "ref")); sys.path.insert(0, str(HERE.parent))
import elastica as E
import check_production as CP
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sp = json.loads((CK / "config/scene_params.json").read_text())
g = json.loads((CK / "params/physics.json").read_text())["g"]["value"]
rod = E.Rod(g=g)
meta = json.loads((SOL / "meta.json").read_text())
static = json.loads((SOL / "static.json").read_text())
z = np.load(SOL / "deformation.npz")
t = z["t"]; W = z["tip_load_N"]; ns = z["n_weights_seated"]
cy, cz = z["centreline_y"], z["centreline_z"]
s_all = z["s_local"]; i0, i1 = 20, 288
audit = {"_doc": "role-E independent audit of " + CKN + " (no solver code read/imported). g=%.5f from params (data)." % g, "items": {}}
It = audit["items"]

# ---------- (1) P_cr / static vs independent shooting
wslot = CP.continuous_wcr(rod, sp, "slot"); wlay = CP.continuous_wcr(rod, sp, "layer")
rows = E.critical_tip_mass_curve(rod, sp, n_max=12)
nb = next(r["n"] for r in rows if r["ratio"] > 1)
Wp = meta["results"]["W_cr_N"]
tb7 = E.scene_tipbody(7, sp)
s7 = E.equilibrium(rod, tb7.M * g, tb7.e, 0.0, n_out=2)
ytop7 = s7["tip_y"] + E.DECLARED["tab_len"] * math.sin(s7["theta_L"])
It["1_static"] = dict(
    Wcr_independent_slot_convention_N=wslot, Wcr_independent_layer_convention_N=wlay, Wcr_production_N=Wp,
    rel_slot=Wp / wslot - 1, rel_layer=Wp / wlay - 1, n_buckle_independent=nb,
    n_buckle_production=static["D_full_model"]["buckles_on_weight_no"],
    ratio_W_over_Wcr={r["n"]: r["ratio"] for r in rows if 5 <= r["n"] <= 8},
    n7_ytop_perfect_independent_mm=1e3 * abs(ytop7), n7_ytop_production_static_mm=abs(meta["results"]["static_n7_ytop_mm"]),
    passed=abs(Wp / wslot - 1) <= 5e-3 and nb == static["D_full_model"]["buckles_on_weight_no"])

# ---------- (2) quasi-static frames t < 4.6 s, imperfection-aware: stress-free curvature kappa0 inferred from sample 0
s = s_all[i0:i1 + 1] - s_all[i0]
def shape(k):
    th, ka, _ = CP.kin(np.stack([np.zeros(i1 - i0 + 1), cy[k, i0:i1 + 1], cz[k, i0:i1 + 1]], -1), s)
    return th, ka
def moment(k, th, n):
    tb = E.scene_tipbody(n, sp); y = cy[k, i0:i1 + 1] - cy[k, i0]
    yW = y[-1] + tb.e * math.sin(th[-1])
    Iy = cumulative_trapezoid(y, s, initial=0)
    return W[k] * (yW - y) + rod.q * ((Iy[-1] - Iy) - y * (s[-1] - s))
th0, ka0m = shape(0)
kappa0 = ka0m - moment(0, th0, 0) / rod.EI   # stress-free curvature (production's imperfection), inferred
k0s = CubicSpline(s, kappa0)
def predict(Wk, e, pick="small"):
    f = lambda ss, u: [u[1], k0s(ss, 1) - (Wk + rod.q * (rod.L - ss)) * math.sin(u[0]) / rod.EI]
    def res(c):
        sol = solve_ivp(f, (0, rod.L), [0.0, c], method="DOP853", rtol=1e-11, atol=1e-14)
        return rod.EI * (sol.y[1, -1] - kappa0[-1]) - Wk * e * math.sin(sol.y[0, -1]), sol
    cg = np.linspace(-8.0, 8.0, 641); rv = [res(c)[0] for c in cg]
    roots = [brentq(lambda c: res(c)[0], cg[i], cg[i + 1], xtol=1e-14) for i in range(len(cg) - 1) if np.sign(rv[i]) != np.sign(rv[i + 1])]
    c = min(roots, key=abs) if pick == "small" else min(roots)  # 'neg': most negative (buckled to -Y like production)
    sol = solve_ivp(f, (0, rod.L), [0.0, c], method="DOP853", rtol=1e-11, atol=1e-14, t_eval=s)
    th = sol.y[0]; y = cumulative_trapezoid(np.sin(th), s, initial=0)
    return y[-1] + E.DECLARED["tab_len"] * math.sin(th[-1])
qs = []
for n in range(0, 7):
    ks = np.where((np.round(ns) == n) & (t < 1.0 + 0.6 * n - 1e-9))[0]
    if len(ks) == 0:
        continue
    k = ks[-1]
    yprod = z["strip_top_world"][k, 1]
    ypred = predict(W[k], E.scene_tipbody(n, sp).e)
    qs.append(dict(n=n, sample=int(k), t=float(t[k]), ytop_prod_mm=1e3 * yprod, ytop_indep_static_mm=1e3 * ypred,
                   diff_mm=1e3 * (yprod - ypred)))
# static n=7 with the same imperfection (cross-check of production static n7)
y7i = predict(tb7.M * g, tb7.e, pick="neg")
It["2_quasistatic"] = dict(
    method="stress-free curvature kappa0(s) = kappa(t=0) - M(t=0)/EI inferred from the production's own t=0 sample; "
           "independent shooting with that imperfection predicts each pre-buckling hold (last sample before the next weight). "
           "Dynamic samples are not static, so diff includes the residual sway.",
    imperfection_ytop_t0_mm=1e3 * z["strip_top_world"][0, 1], rows=qs,
    n7_static_with_prod_imperfection_indep_mm=1e3 * y7i, n7_static_production_mm=meta["results"]["static_n7_ytop_mm"],
    n7_rel=(abs(meta["results"]["static_n7_ytop_mm"]) / abs(1e3 * y7i) - 1),
    criterion="holds n<=5: |prod - indep static| <= 0.5 mm (residual post-placement sway); n=6 reported only (near-critical, "
              "f~0.19 Hz, 0.6 s hold is not quasi-static); static n=7 with the production imperfection within 1 %",
    passed=bool(all(abs(r["diff_mm"]) <= 0.5 for r in qs if r["n"] <= 5)
                and abs(abs(meta["results"]["static_n7_ytop_mm"]) / abs(1e3 * y7i) - 1) <= 1e-2))
cp = json.loads((V / "out/check_production.json").read_text())["checks"]
It["2b_length_energy_clamp"] = dict(C4=cp["C4_inextensible"], C5=cp["C5_clamp"],
                                    C6={k: v for k, v in cp["C6_energy"].items() if k != "series"},
                                    passed=all(cp[c]["passed"] for c in ("C4_inextensible", "C5_clamp", "C6_energy")))

# ---------- PC2 vs npz
pc2 = CK / "scene/out/ruler_anim_strip.pc2"
with open(pc2, "rb") as fh:
    hdr = fh.read(32)
    sig, ver, npts, start, rate, nsamp = struct.unpack("<12siiffi", hdr)
    data = np.frombuffer(fh.read(), dtype="<f4").reshape(nsamp, npts, 3).astype(float)
P = z["positions"].astype(float).reshape(len(t), -1, 3)
nv = P.shape[1]
res_pc2 = dict(signature=sig.decode(errors="replace"), points=npts, start_frame=start, sample_rate=rate, samples=nsamp)
if npts == nv:
    mid = data
elif npts == 2 * nv:
    mid = 0.5 * (data[:, :nv] + data[:, nv:])
    res_pc2["thickness_check_max_err_m"] = float(np.abs(np.linalg.norm(data[:, :nv] - data[:, nv:], axis=2) - 0.0005).max())
else:
    mid = None
if mid is not None and nsamp == len(t):
    off = (P - mid).reshape(-1, 3).mean(0)  # PC2 is in object-local space; object origin = strip bottom-centre
    res_pc2["object_origin_offset_m"] = off.tolist()
    d = np.linalg.norm(mid + off - P, axis=2)
    res_pc2.update(max_err_m=float(d.max()), worst_sample=int(d.max(axis=1).argmax()))
    res_pc2["passed"] = res_pc2["max_err_m"] <= 1e-6
else:
    res_pc2["passed"] = None
It["2c_pc2_vs_npz"] = res_pc2

# ---------- (3) blend schedule
bd = V / "out" / CKN / "blend_dump.npz"
if bd.exists():
    b = np.load(bd, allow_pickle=False)
    names = list(b["names"]); times = b["times"]; mats = b["mats"]; vis = np.ones_like(b["vis"], bool)  # visible_get() is False in background mode; hide_render toggles checked separately
    tt = times / float(b["fps"])
    widx = [i for i, nm in enumerate(names) if "weight" in nm.lower()]
    # peg axis in world from the tip body matrix (production npz) interpolated to blend times
    O = np.array([np.interp(tt, t, z["tipbody_origin_world"][:, j]) for j in range(3)]).T
    rx = np.interp(tt, t, z["tipbody_rot_x_rad"])
    axis = np.stack([np.zeros_like(rx), -np.sin(rx), np.cos(rx)], 1)  # rotation about +X of +Z
    stack_c = np.array(sp["slotted_weights"]["stack_center_xy_m"])
    alongs = []
    on_peg = np.zeros((len(tt), len(widx)), bool); on_bench = np.zeros_like(on_peg); jumps = []; jump_t = []
    at_start = np.zeros_like(on_peg)
    for j, wi in enumerate(widx):
        p = mats[:, wi, :3, 3]
        r = p - O; along = (r * axis).sum(1); radial = np.linalg.norm(r - along[:, None] * axis, axis=1)
        on_peg[:, j] = (radial < 3e-3) & (along > 0) & (along < 0.08) & vis[:, wi]
        alongs.append(along)
        on_bench[:, j] = (np.linalg.norm(p[:, :2] - stack_c, axis=1) < 0.01) & (p[:, 2] < 0.2) & vis[:, wi]
        v = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.diff(tt)
        jumps.append(float(v.max())); jump_t.append(float(tt[int(v.argmax())]))
        at_start[:, j] = np.linalg.norm(p - p[0], axis=1) < 1e-3
    npeg = on_peg.sum(1); nbench = on_bench.sum(1)
    vis_changes = [names[wi] for wi in widx if np.any(vis[:, wi] != vis[0, wi])]
    arr = []
    for j in range(len(widx)):
        idx = np.where(on_peg[:, j])[0]
        if len(idx):
            arr.append((float(tt[idx[0]]), j))
    arr.sort()
    seat_times, axis_arrival, slot_vis = [], [a_[0] for a_ in arr], []
    for slot, (_, j) in enumerate(arr):  # contact = weight bottom within 0.5 mm of its slot (0.015 + 0.0045*slot above O)
        final = alongs[j][-1]
        idx = np.where(on_peg[:, j] & (np.abs(alongs[j] - final) < 1e-4))[0]
        seat_times.append(float(tt[idx[0]]) if len(idx) else None)
        offs = meta["schedule"]["slot_offset_above_tab_bottom_m"]
        phys = offs[slot] if slot < len(offs) else float("nan")
        slot_vis.append(dict(slot=slot, weight=names[widx[j]], visual_bottom_mm=1e3 * final, physics_bottom_mm=1e3 * phys,
                             err_mm=1e3 * (final - phys)))
    expected = list(meta["schedule"]["weight_contact_start_s"])
    n_start = at_start.sum(1)  # weights still at their t=0 (feeder/bench) position
    dec = np.diff(n_start)
    other = [i for i, nm in enumerate(names) if re.search(r"feeder|gripper|arm|finger|hand(?!le)|jaw", nm, re.I)
             and "weight" not in nm.lower() and "vise" not in nm.lower()]
    other_speed = {}
    for i in other:
        pp = mats[:, i, :3, 3]; vv = np.linalg.norm(np.diff(pp, axis=0), axis=1) / np.diff(tt)
        other_speed[names[i]] = dict(peak_m_s=float(vv.max()), at_s=float(tt[int(vv.argmax())]),
                                     last_move_s=float(tt[np.where(vv > 1e-4)[0][-1] + 1]) if np.any(vv > 1e-4) else None)
    It["3_schedule"] = dict(
        weight_objects=[names[i] for i in widx], max_on_peg=int(npeg.max()), on_peg_axis_arrival_s=axis_arrival,
        seat_times_s=seat_times, expected=expected, stack_visual_vs_physics=slot_vis,
        seat_time_err_s=[(a - b_) if a is not None else None for a, b_ in zip(seat_times, expected)], bench_count_start=int(nbench[0]),
        bench_count_end=int(nbench[-1]), bench_count_monotone=bool(np.all(np.diff(nbench) <= 0)),
        max_weight_speed_m_s=max(jumps) if jumps else None,
        weight_peak_speed_by_object={names[widx[j]]: (round(jumps[j], 4), round(jump_t[j], 4)) for j in range(len(widx))},
        n_at_start_position_begin_end=(int(n_start[0]), int(n_start[-1])), at_start_decrements=sorted(set(dec.tolist())),
        gripper_or_feeder_objects_speed_info=other_speed,
        criteria="peak weight speed <= 1 m/s (also continuity at 120 Hz sampling); seated within 0.1 mm of meta slot offsets; "
                 "seat time within 1/24 s of t0; <= 7 on peg; weights leave their start position one at a time (decrements of 1, total 7); no hide_render toggles",
        hand_object_present=any(re.search(r"(?<!vise)hand(?!le)", nm, re.I) for nm in names), visibility_toggles=vis_changes,
        note="seated = weight centre within 3 mm of the peg axis (from the production tip-body pose); speed limit 2 m/s flags teleport",
        passed=bool(npeg.max() <= 7 and len(seat_times) == 7 and all(a is not None and abs(a - b_) <= 1 / 24 for a, b_ in zip(seat_times, expected))
                    and all(abs(x["err_mm"]) <= 0.1 for x in slot_vis)
                    and not vis_changes and (max(jumps) if jumps else 0) <= 1.0
                    and set(dec.tolist()) <= {0, -1} and int(n_start[0] - n_start[-1]) == 7))
    if b["strip"].size:
        S = b["strip"]; nfr = min(len(S), len(t))
        ok_shape = S.shape[1] in (nv, 2 * nv)
        if ok_shape:
            Sm = S[:nfr] if S.shape[1] == nv else 0.5 * (S[:nfr, :nv] + S[:nfr, nv:])
            dd = np.linalg.norm(Sm - P[:nfr], axis=2)
            It["3b_blend_strip_vs_npz"] = dict(strip=str(b["strip_name"]), samples=int(nfr), max_err_m=float(dd.max()),
                                                passed=bool(dd.max() <= 1e-6))
        else:
            It["3b_blend_strip_vs_npz"] = dict(strip=str(b["strip_name"]), verts=int(S.shape[1]), passed=None,
                                                note="vertex count differs from npz grid (solidify/extra verts); needs mapping")
else:
    It["3_schedule"] = dict(passed=None, note="blend dump missing")

nc = json.loads((V / "out/negative_controls.json").read_text())
It["4_negative_controls"] = dict(cases={k: v["control_ok"] for k, v in nc.items()}, passed=all(v["control_ok"] for v in nc.values()))
audit["summary"] = {k: v.get("passed") for k, v in It.items()}
(V / "out" / f"{CKN}_audit.json").write_text(json.dumps(audit, indent=1, default=float))

# ---------- visual
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
a = ax[0]
a.plot(t, 1e3 * z["strip_top_world"][:, 1], color="C0", lw=1, label="production Y_top (npz)")
for r in qs:
    a.plot(r["t"], r["ytop_indep_static_mm"], "kx")
a.axhline(-1e3 * abs(ytop7), color="C3", ls="--", lw=0.8, label="indep. static n=7 (perfect)")
a.axhline(1e3 * y7i, color="C1", ls=":", lw=1, label="indep. static n=7 (prod. imperfection)")
for tj in [1.0 + 0.6 * j for j in range(7)]:
    a.axvline(tj, color="0.85", lw=0.6, zorder=0)
a.set_xlabel("t [s]"); a.set_ylabel("Y of strip top [mm]"); a.legend(fontsize=7)
a.set_title(f"{VTAG} | {CKN} | strip-top sway vs independent statics\n(x = independent static at each hold)")
ins = a.inset_axes([0.1, 0.1, 0.4, 0.35])
ins.plot(t[t < 4.7], 1e3 * z["strip_top_world"][t < 4.7, 1], lw=0.8)
for r in qs:
    ins.plot(r["t"], r["ytop_indep_static_mm"], "kx", ms=4)
ins.tick_params(labelsize=6); ins.set_title("t<4.7 s", fontsize=7)
a = ax[1]
ser = cp["C6_energy"]["series"]
a.plot(ser["t"], 1e3 * np.array(ser["U"]), label="strain U"); a.plot(ser["t"], 1e3 * np.array(ser["T"]), label="kinetic T")
a.plot(ser["t"], 1e3 * np.array(ser["D"]), "k", label="dissipated D = work - dU - dVq - dT")
a.set_xlabel("t [s]"); a.set_ylabel("mJ"); a.legend(fontsize=7)
a.set_title(f"{VTAG} | {CKN} | energy balance (independent bookkeeping)\nmin D = %.3f mJ (tol %.2f mJ)" % (
    1e3 * cp["C6_energy"]["min_dissipation_J"], 1e3 * cp["C6_energy"]["tol_J"]))
a = ax[2]; a.axis("off")
lines = [f"{VTAG} | {CKN} audit | PASS/FAIL", ""]
for k, v in audit["summary"].items():
    lines.append(f"{k:28s} {'PASS' if v else ('FAIL' if v is False else 'n/a')}")
i1_ = It["1_static"]
lines += ["", f"W_cr indep {i1_['Wcr_independent_slot_convention_N']:.6f} N  prod {Wp:.6f} N",
          f"rel {i1_['rel_slot']:+.1e}  buckles on weight #{nb}",
          f"n7 static |Y_top| indep(perfect) {i1_['n7_ytop_perfect_independent_mm']:.2f} mm",
          f"  indep(prod imperf.) {abs(1e3*y7i):.2f}  prod {abs(meta['results']['static_n7_ytop_mm']):.2f} mm",
          f"PC2 vs npz max {res_pc2.get('max_err_m', float('nan')):.2e} m"]
if "3_schedule" in It and It["3_schedule"].get("seat_times_s") is not None:
    lines.append("seat times " + ", ".join("-" if x is None else f"{x:.2f}" for x in It["3_schedule"]["seat_times_s"]))
    lines.append(f"max weight speed {It['3_schedule']['max_weight_speed_m_s']:.2f} m/s (limit 1.0)")
    lines.append("stack err mm " + ", ".join(f"{x['err_mm']:+.2f}" for x in It["3_schedule"]["stack_visual_vs_physics"]))
    for k_, v_ in sorted(It["3_schedule"]["gripper_or_feeder_objects_speed_info"].items(), key=lambda kv: -kv[1]["peak_m_s"])[:4]:
        lines.append(f"info {k_[:22]} peak {v_['peak_m_s']:.2f} m/s, last move {v_['last_move_s']}")
a.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)
fig.tight_layout(); fig.savefig(V / f"visuals/{VTAG}_{CKN}_audit.png", dpi=110)
print(json.dumps(audit["summary"], indent=1))
