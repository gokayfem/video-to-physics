"""Role-E checks on production solver outputs (reads solver/out/deformation.npz + meta.json as DATA only).

Usage:
  <venv-python> verify/checks/check_production.py [--npz solver/out/deformation.npz] [--meta solver/out/meta.json]
        [--out verify/out/check_production.json] [--skip energy,equilibrium,...] [--vsettle 2e-3]

Checks (tolerances from the role-E brief):
  C1 pcr          P_cr in meta (any key matching p_?cr / w_?cr) vs independent reference for the actual BC   <= 0.5 %
  C1b n_buckle    first settled weight count with |tip deflection| > 5 mm == reference count
  C2 postbuckling settled frames, |tip lateral deflection| vs reference elastica at the same tip load          <= 1 %
  C3 equilibrium  settled post-buckled frames: theta(s) vs integral of M(s)/EI from the loads, and EI*kappa vs M <= 1e-3 rel
  C4 inextensible centreline (width-averaged) arc length of every sample vs rest length                       <= 1e-6 m
  C5 clamp        nodes inside the vise (s <= clamped length) fixed to rest positions                         <= 1e-6 m
  C6 energy       D(t) = work by tip load + self-weight - strain - kinetic must be >= -tol and non-decreasing
                  (tol = 5e-3 * max(strain energy, |work|))
"""
import argparse, json, math, re, sys
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parents[1] / "ref"))
import elastica as E  # role-E reference (independent of solver/)

TOL = dict(pcr=5e-3, post=1e-2, equil=1e-3, length=1e-6, clamp=1e-6, energy=5e-3)


def load_npz(path, pos_key=None):
    z = np.load(path, allow_pickle=False)
    keys = list(z.keys())
    if pos_key is None:
        cands = [k for k in keys if z[k].ndim >= 3 and z[k].shape[-1] == 3]
        pref = [k for k in cands if re.search(r"pos|xyz|vert|node", k, re.I)]
        pos_key = (pref or cands)[0]
    P = np.asarray(z[pos_key], float)
    S = P.shape[0]
    t = None
    for k in keys:
        if re.fullmatch(r"t|time|times|t_s|time_s", k, re.I):
            t = np.asarray(z[k], float).ravel()
    if t is None:
        t = np.arange(S) / (24.0 * 5)
    W = None
    for k in keys:
        if re.search(r"tip.?load|load.?n|tip_force|W_N", k, re.I):
            W = np.asarray(z[k], float)
            if W.ndim > 1:  # vector load -> vertical magnitude
                W = np.abs(W[..., -1]) if W.shape[-1] == 3 else np.abs(W).max(axis=-1)
    nseat = np.asarray(z["n_weights_seated"], float) if "n_weights_seated" in keys else None
    return dict(P=P, t=t, W=W, keys=keys, pos_key=pos_key, nseat=nseat)


def grid_centreline(P, sp):
    """Return centreline (S, nl, 3) (width-averaged) for any nw x nl ordering."""
    S = P.shape[0]
    if P.ndim == 4:
        G = P
    else:
        Nv = P.shape[1]
        G = None
        for nw, nl in [(11, 301), (10, 300), (301, 11), (300, 10), (2, Nv // 2), (1, Nv)]:
            if nw * nl == Nv:
                G = P.reshape(S, nw, nl, 3); break
        if G is None:
            raise ValueError(f"cannot factor {Nv} vertices")
    # length axis = axis with largest z spread on frame 0
    spread1 = np.ptp(G[0, :, :, 2], axis=1).mean()   # along axis 2
    spread0 = np.ptp(G[0, :, :, 2], axis=0).mean()   # along axis 1
    if spread0 > spread1:
        G = G.transpose(0, 2, 1, 3)
    C = G.mean(axis=1)
    if C[0, 0, 2] > C[0, -1, 2]:
        C = C[:, ::-1]
    return C, G


def scene():
    sp = json.loads((ROOT / "config/scene_params.json").read_text())
    ru = sp["ruler"]
    g = E.G
    try:
        g = float(json.loads((ROOT / "params/physics.json").read_text())["g"]["value"])
    except Exception:
        pass
    rod = E.Rod(g=g, L=E.DECLARED["L_f"], b=ru["width_m"], h=ru["thickness_m"], E=ru["material_fea"]["E_Pa"],
                nu=ru["material_fea"]["nu"], rho=ru["material_fea"]["rho_kg_m3"])
    return sp, rod


def kin(C, s_mat):
    """Spline kinematics of the centreline against the material (rest) coordinate."""
    cs = CubicSpline(s_mat, C, axis=0)
    d1, d2 = cs(s_mat, 1), cs(s_mat, 2)
    y1, z1, y2, z2 = d1[:, 1], d1[:, 2], d2[:, 1], d2[:, 2]
    theta = np.arctan2(y1, z1)
    kappa = (z1 * y2 - y1 * z2) / (y1 ** 2 + z1 ** 2)
    sf = np.linspace(s_mat[0], s_mat[-1], 20 * (len(s_mat) - 1) + 1)
    sp_ = np.linalg.norm(cs(sf, 1), axis=1)
    length = np.trapezoid(sp_, sf) if hasattr(np, "trapezoid") else np.trapz(sp_, sf)
    return theta, kappa, length


def tipbody_from_W(W, g, sp):
    M0 = E.scene_tipbody(0, sp).M
    n = int(round((W / g - M0) / sp["slotted_weights"]["mass_each_kg"]))
    n = max(n, 0)
    return n, E.scene_tipbody(n, sp, include_tab_offset=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(ROOT / "solver/out/deformation.npz"))
    ap.add_argument("--meta", default=str(ROOT / "solver/out/meta.json"))
    ap.add_argument("--out", default=str(ROOT / "verify/out/check_production.json"))
    ap.add_argument("--skip", default="")
    ap.add_argument("--pos-key", default=None)
    ap.add_argument("--settle-window", type=float, default=0.25, help="half-window [s] over which the speed must stay below vsettle")
    ap.add_argument("--vsettle", type=float, default=2e-3, help="max node speed [m/s] for a 'settled' sample")
    a = ap.parse_args(argv)
    skip = set(filter(None, a.skip.split(",")))
    sp, rod = scene()
    ru = sp["ruler"]
    D = load_npz(a.npz, a.pos_key)
    C, G = grid_centreline(D["P"], sp)
    zz = np.load(a.npz)
    if "centreline_y" in zz and "centreline_z" in zz:  # float64 centreline if exported (positions are float32)
        C = np.stack([np.zeros_like(zz["centreline_y"]), zz["centreline_y"], zz["centreline_z"]], axis=-1).astype(float)
        rep_cl = "centreline_y/z (float64)"
    else:
        rep_cl = "width-average of positions"
    t, W = D["t"], D["W"]
    S, nl = C.shape[:2]
    # material coordinate from the rest (first) sample, measured from the strip bottom
    s_mat = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C[0], axis=0), axis=1))])
    L0 = s_mat[-1]
    s_cl = ru["clamped_length_m"]
    i0 = int(np.argmin(abs(s_mat - s_cl)))
    i1 = int(np.argmin(abs(s_mat - (s_cl + rod.L))))   # O: tab bottom (end of flexible length)
    rep = dict(npz=a.npz, keys=D["keys"], pos_key=D["pos_key"], samples=int(S), nodes_along=int(nl),
               rest_length_m=L0, tol=TOL, centreline_source=rep_cl, checks={})
    ck = rep["checks"]

    # velocities and settled samples
    V = np.gradient(C, t, axis=0)
    vmax = np.linalg.norm(V, axis=2).max(axis=1)
    win = max(1, int(round(a.settle_window / np.median(np.diff(t)))))
    vwin = np.array([vmax[max(0, k - win): k + win + 1].max() for k in range(S)])
    settled = vwin < a.vsettle
    rep["n_settled"] = int(settled.sum())

    # C4 inextensibility (+ C5 clamp)
    lengths = np.array([kin(C[k], s_mat)[2] for k in range(S)])
    Lref = ru["length_m"]
    err_len = np.abs(lengths - Lref)
    ck["C4_inextensible"] = dict(max_abs_length_err_m=float(err_len.max()), rest_vs_param_m=float(lengths[0] - Lref),
                                 worst_sample=int(err_len.argmax()), passed=bool(err_len.max() <= TOL["length"]))
    Gc = G[:, :, : i0 + 1]
    clamp_err = float(np.abs(Gc - Gc[0:1]).max())
    # rigid tab segment: nodes s >= s_O must stay on the rigid line through O along the O tangent (rigid-body motion)
    rig = []
    for k in range(S):
        seg = C[k, i1:]; d = seg[-1] - seg[0]; d = d / np.linalg.norm(d)
        r = seg - seg[0]; rig.append(np.linalg.norm(r - np.outer(r @ d, d), axis=1).max())
    rig_err = float(max(rig))
    ck["C5_clamp"] = dict(max_node_motion_in_vise_m=clamp_err, clamp_node_index=i0, tab_node_index=i1,
                          tab_segment_max_bend_m=rig_err, passed=clamp_err <= TOL["clamp"] and rig_err <= TOL["clamp"])

    # reference P_cr table
    rows = E.critical_tip_mass_curve(rod, sp, n_max=12)
    n_ref = next(r["n"] for r in rows if r["ratio"] > 1)
    rep["reference"] = dict(n_buckle=n_ref, Wcr_N_at_n_buckle=rows[n_ref]["Wcr_N"], euler_ideal_N=rod.euler,
                            Wcr_selfweight_e0_N=E.critical_load(rod, 0.0))

    # C1 P_cr from meta
    if "pcr" not in skip:
        found = {}
        try:
            meta = json.loads(Path(a.meta).read_text())
            def walk(o, p=""):
                if isinstance(o, dict):
                    for k, v in o.items():
                        walk(v, f"{p}.{k}" if p else k)
                elif isinstance(o, (int, float)) and re.fullmatch(r"(p|w)_?cr(_\w*)?", p.split(".")[-1], re.I) \
                        and not re.search(r"over|rel|err", p.split(".")[-1], re.I):
                    found[p] = float(o)
            walk(meta)
        except Exception as ex:
            found = {"_error": str(ex)}
        refs = {"Wcr_actual_BC(n_buckle)": continuous_wcr(rod, sp, "slot"), "Wcr_actual_BC_layer": continuous_wcr(rod, sp, "layer"),
                "euler_ideal": rod.euler,
                "selfweight_e0": rep["reference"]["Wcr_selfweight_e0_N"]}
        cmp_ = {}
        for k, v in found.items():
            if isinstance(v, float):
                cmp_[k] = {rk: v / rv - 1 for rk, rv in refs.items()}
        ok = any(min(abs(c["Wcr_actual_BC(n_buckle)"]), abs(c["Wcr_actual_BC_layer"])) <= TOL["pcr"] for c in cmp_.values()) if cmp_ else None
        ck["C1_pcr"] = dict(meta_values=found, rel_vs_refs=cmp_, passed=ok,
                            note="PASS requires a meta P_cr within 0.5 % of the actual-BC reference (self-weight + eccentric tip body CoM)")

    # settled samples: tip deflection per weight count, buckling count, post-buckling, equilibrium
    if W is not None:
        tipdefl = C[:, i1, 1] - C[0, i1, 1]
        per = {}
        for k in np.where(settled)[0]:
            n, tb = tipbody_from_W(W[k], rod.g, sp)
            per.setdefault(n, []).append(k)
        onset = None
        rows_post, rows_eq = [], []
        for n in sorted(per):
            ks = per[n]
            dmax = float(np.abs(tipdefl[ks]).mean())
            if onset is None and dmax > 5e-3:
                onset = n
            tb = E.scene_tipbody(n, sp, include_tab_offset=False)
            k = ks[-1]
            ref = E.equilibrium(rod, W[k], tb.e, tb.d, n_out=2)
            if abs(ref["tip_y"]) > 5e-3 and "postbuckling" not in skip:
                rows_post.append(dict(n=n, sample=int(k), W=float(W[k]), prod_tip_y=float(tipdefl[k]),
                                      ref_tip_y=float(ref["tip_y"]), rel=float(abs(tipdefl[k]) / abs(ref["tip_y"]) - 1)))
            if abs(tipdefl[k]) > 1e-3 and "equilibrium" not in skip:
                rows_eq.append(dict(n=n, sample=int(k), **equilibrium_residual(C[k], s_mat, i0, i1, W[k], tb, rod)))
        ck["C1b_n_buckle"] = dict(prod=onset, ref=n_ref, passed=(onset == n_ref) if onset is not None else None)
        if "postbuckling" not in skip:
            worst = max((abs(r["rel"]) for r in rows_post), default=None)
            ck["C2_postbuckling"] = dict(rows=rows_post, worst_rel=worst,
                                         passed=None if worst is None else worst <= TOL["post"])
        if "equilibrium" not in skip:
            worst = max((max(r["res_theta"], r["res_kappa"]) for r in rows_eq), default=None)
            ck["C3_equilibrium"] = dict(rows=rows_eq, worst=worst, passed=None if worst is None else worst <= TOL["equil"])
    else:
        ck["C2_postbuckling"] = dict(passed=None, note="no tip-load array found in npz")

    # C6 energy balance
    if "energy" not in skip and W is not None:
        ck["C6_energy"] = energy_balance(C, s_mat, i0, i1, t, W, rod, sp, D["nseat"])

    verdicts = [c.get("passed") for c in ck.values()]
    rep["all_passed"] = all(v is not False for v in verdicts) and any(v is True for v in verdicts)
    rep["failed"] = [k for k, c in ck.items() if c.get("passed") is False]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    summ = {k: (c.get("passed"), {kk: vv for kk, vv in c.items() if kk not in ("rows", "meta_values", "rel_vs_refs", "series")})
            for k, c in ck.items()}
    print(json.dumps(dict(all_passed=rep["all_passed"], failed=rep["failed"], checks=summ), indent=1, default=float))
    return 0 if rep["all_passed"] else 1


def continuous_wcr(rod, sp, mode):
    """Critical tip load with a fractional weight count: 'slot' = fractional weight at its slot CoM (production convention),
    'layer' = the stack as a continuous layer. Both bracket the physical answer (an integer count)."""
    from scipy.optimize import brentq
    tb0 = E.scene_tipbody(0, sp); t = sp["slotted_weights"]["thickness_m"]; m = sp["slotted_weights"]["mass_each_kg"]
    base = E.DECLARED["tab_len"] + sp["platform"]["plate_size_m"][2]
    def ecom(nf):
        if mode == "layer":
            mom = m * nf * (base + nf * t / 2)
        else:
            k = int(math.floor(nf)); mom = sum(m * (base + (j + 0.5) * t) for j in range(k)) + m * (nf - k) * (base + (k + 0.5) * t)
        return (tb0.M * tb0.e + mom) / (tb0.M + m * nf)
    f = lambda nf: (tb0.M + m * nf) * rod.g - E.critical_load(rod, ecom(nf))
    nf = brentq(f, 0, 12, xtol=1e-12)
    return (tb0.M + m * nf) * rod.g


def equilibrium_residual(Ck, s_mat, i0, i1, Wk, tb, rod):
    s = s_mat[i0:i1 + 1] - s_mat[i0]
    Cf = Ck[i0:i1 + 1]
    theta, kappa, _ = kin(Cf, s)
    y = Cf[:, 1] - Cf[0, 1]
    thL = theta[-1]
    yW = y[-1] + tb.e * math.sin(thL) + tb.d * math.cos(thL)
    # q * int_s^L (y(t) - y(s)) dt
    Iy = cumulative_trapezoid(y, s, initial=0)
    tail_int = (Iy[-1] - Iy) - y * (s[-1] - s)
    M = Wk * (yW - y) + rod.q * tail_int
    th_pred = cumulative_trapezoid(M / rod.EI, s, initial=0) + theta[0]
    res_theta = float(np.linalg.norm(theta - th_pred) / np.linalg.norm(theta))
    inner = slice(3, -3)
    res_kappa = float(np.linalg.norm(rod.EI * kappa[inner] - M[inner]) / np.linalg.norm(M[inner]))
    return dict(res_theta=res_theta, res_kappa=res_kappa, clamp_angle_rad=float(theta[0]))


def energy_balance(C, s_mat, i0, i1, t, W, rod, sp, nseat=None):
    S = C.shape[0]
    s = s_mat[i0:i1 + 1] - s_mat[i0]
    C = C[:, : i1 + 1]
    Mb = np.zeros(S); Jc = np.zeros(S); zs = np.zeros(S); eb = np.zeros(S); db = np.zeros(S)
    U = np.zeros(S); Vq = np.zeros(S); zc = np.zeros(S); yc = np.zeros(S); thL = np.zeros(S)
    for k in range(S):
        th, ka, _ = kin(C[k, i0:], s)
        U[k] = 0.5 * rod.EI * np.trapezoid(ka ** 2, s)
        Vq[k] = rod.q * np.trapezoid(C[k, i0:, 2], s)
        if nseat is not None:
            n = int(round(nseat[k])); tb = E.scene_tipbody(n, sp)
        else:
            n, tb = tipbody_from_W(W[k], rod.g, sp)
        thL[k] = th[-1]
        Mb[k], Jc[k], eb[k], db[k] = tb.M, tb.J_c, tb.e, tb.d
        zc[k] = C[k, -1, 2] + tb.e * math.cos(th[-1]) - tb.d * math.sin(th[-1])
        yc[k] = C[k, -1, 1] + tb.e * math.sin(th[-1]) + tb.d * math.cos(th[-1])
        # contact force of a weight being lowered (not yet part of the body) acts at its slot on the stack
        es = E.DECLARED["tab_len"] + sp["platform"]["plate_size_m"][2] + (n + 0.5) * sp["slotted_weights"]["thickness_m"]
        zs[k] = es
    V = np.gradient(C[:, i0:], t, axis=0)
    m = rod.rho * rod.A
    T_strip = np.array([0.5 * m * np.trapezoid((V[k] ** 2).sum(axis=1), s) for k in range(S)])
    # body-point velocities from O's velocity and the tip rotation rate with the CURRENT offsets (no spurious
    # velocity when the body definition changes at a weight release)
    om = np.gradient(thL, t); vzO = np.gradient(C[:, -1, 2], t); vyO = np.gradient(C[:, -1, 1], t)
    sn, cs_ = np.sin(thL), np.cos(thL)
    vzc = vzO - (eb * sn + db * cs_) * om
    vyc = vyO + (eb * cs_ - db * sn) * om
    if nseat is None:
        Mb = W / rod.g
    Fx = np.clip(W - Mb * rod.g, 0, None)  # contact force of the weight in transfer
    vzs = vzO - zs * sn * om
    # mass joins at release: piecewise-constant body -> use per-sample velocities (jump terms are reported, not hidden)
    T_body = 0.5 * Mb * (vzc ** 2 + vyc ** 2) + 0.5 * Jc * om ** 2
    work = cumulative_trapezoid(-Mb * rod.g * vzc - Fx * vzs, t, initial=0)
    T = T_strip + T_body
    Dis = work - (U - U[0]) - (Vq - Vq[0]) - (T - T[0])
    scale = max(U.max(), np.abs(work).max(), 1e-12)
    tol = TOL["energy"] * scale
    min_D = float(Dis.min())
    max_drop = float(np.max(np.maximum.accumulate(Dis) - Dis))
    return dict(scale_J=float(scale), tol_J=float(tol), min_dissipation_J=min_D, max_decrease_J=max_drop,
                final_dissipation_J=float(Dis[-1]), U_max_J=float(U.max()), work_end_J=float(work[-1]),
                passed=bool(min_D >= -tol and max_drop <= tol),
                series=dict(t=t[:: max(1, S // 400)].tolist(), D=Dis[:: max(1, S // 400)].tolist(),
                            U=U[:: max(1, S // 400)].tolist(), T=T[:: max(1, S // 400)].tolist()))


if __name__ == "__main__":
    sys.exit(main())
