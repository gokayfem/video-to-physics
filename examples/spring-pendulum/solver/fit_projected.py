"""Projected fit of the elastic-pendulum + hinged-weight model (physics/FORMULATION.md) to image tracks through the
camera of record, on the opening ANCHOR window [0, T_a] (CHANGELOG anchor-window rule); after T_a the physics runs
unchanged and the remainder is a DIVERGENCE MAP (AI motion vs corrected physics), never a fit target.

Model -> image:
  solver state in the gravity frame (metres, A = origin)  ->  static-scene frame  X_s = R_y(beta)^T X  (optional fitted
  level tilt, clip-01 precedent: static scene rotated by +beta about world +Y through A)  ->  camera units  lam * X_s,
  lam = Lcam / ell  (Lcam = equilibrium |AH| in camera units, ell = the same in metres)  ->  per-frame camera projection.
Tracked points: hook = H (lower hook / hinge), wctr = H + b n (cylinder centre = CoM), optional wbot = H + (b + h/2) n.

Free parameters (bounded, all physical):
  ell [m]          equilibrium |AH| in metres (set by the swing frequency; g assumed)  -> model-conditioned scale
  Lcam             equilibrium |AH| in camera units
  bcam             hinge -> cylinder centre in camera units (b = bcam / lam)
  rax              omega_axial / sqrt(g/ell)  (>1.05 keeps L0 > 0);  omega_axial = rax sqrt(g/ell)
  phi0, psi0       initial spring / body angle in XZ [rad];  de0 initial stretch offset [m]
  phid0, psid0, ded0  their rates
  gamma_sw, gamma_ax >= 0   (M1; M0 = declared assumption Q_ASSUMED), chosen by BIC + bootstrap + plausibility
  beta [rad]       optional static tilt (--fit-tilt)
  phiy0, psiy0, phidy0, psidy0   optional out-of-plane initial state (--fit-3d)

Usage:
  .venv/bin/python solver/fit_projected.py --tracks measurement/tracks.csv --t-anchor 4.0 [--camera project|fake:<json>]
      [--r-over-ell 0.085 --h-over-ell 0.2] [--fit-tilt] [--fit-3d] [--damping auto|fitted|assumed] [--nboot 30] --out DIR
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import spring_pendulum as sp  # noqa: E402
import model_build as mb  # noqa: E402

G = sp.G_STD
Q_ASSUMED = 500.0
Q_RANGE = (100.0, 5000.0)
H_FIT = 1.0 / 768.0

NAMES_ALL = ["ell", "Lcam", "bcam", "rax", "phi0", "psi0", "de0", "phid0", "psid0", "ded0", "gamma_sw", "gamma_ax", "beta",
             "phiy0", "psiy0", "phidy0", "psidy0", "pz", "dw", "alpha"]
LO = dict(dw=-0.1, alpha=-0.1, pz=-0.2, ell=0.03, Lcam=1e-3, bcam=1e-4, rax=1.05, phi0=-0.8, psi0=-0.8, de0=-0.2, phid0=-10, psid0=-30, ded0=-3,
          gamma_sw=0.0, gamma_ax=0.0, beta=-0.1, phiy0=-0.5, psiy0=-0.5, phidy0=-10, psidy0=-30)
HI = dict(dw=0.1, alpha=0.1, pz=0.6, ell=3.0, Lcam=1e4, bcam=1e4, rax=20.0, phi0=0.8, psi0=0.8, de0=0.2, phid0=10, psid0=30, ded0=3,
          gamma_sw=3.0, gamma_ax=10.0, beta=0.1, phiy0=0.5, psiy0=0.5, phidy0=10, psidy0=30)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------- cameras
class Cams:
    """Per-frame cameras: R (N,3,3) world->cv, C (N,3) [camera units], f, cx, cy."""

    @classmethod
    def from_project(cls, frames, kind="reference"):
        spec = importlib.util.spec_from_file_location("cam_project", ROOT / "camera" / "project.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        cams = [m.load_camera(frame=int(n), kind=kind) for n in frames]
        o = cls(); o.R = np.array([c["R"] for c in cams], float); o.C = np.array([c["C"] for c in cams], float)
        o.f = np.array([c["f_px"] for c in cams], float); o.cx = np.array([c["cx"] for c in cams], float)
        o.cy = np.array([c["cy"] for c in cams], float)
        o.source = f"camera/project.py load_camera(frame, kind={kind!r}) on params/camera.json"
        return o

    @classmethod
    def from_fake(cls, path, frames):
        J = json.loads(Path(path).read_text()); n = len(frames)
        o = cls(); o.R = np.repeat(np.array(J["R"], float)[None], n, 0); o.C = np.repeat(np.array(J["C"], float)[None], n, 0)
        o.f = np.full(n, float(J["f_px"])); o.cx = np.full(n, float(J["cx"])); o.cy = np.full(n, float(J["cy"]))
        o.source = f"fake camera {path}"
        return o

    def subset(self, idx):
        o = Cams(); o.R, o.C, o.f, o.cx, o.cy = self.R[idx], self.C[idx], self.f[idx], self.cx[idx], self.cy[idx]
        o.source = self.source; return o

    def project(self, X):  # X (N,3) or (N,3,B) in camera units
        if X.ndim == 2:
            X = X[..., None]
        Xc = np.einsum("nij,njb->nib", self.R, X - self.C[..., None])
        u = self.cx[:, None] + self.f[:, None] * Xc[:, 0] / Xc[:, 2]
        v = self.cy[:, None] + self.f[:, None] * Xc[:, 1] / Xc[:, 2]
        return np.stack([u, v], 1)  # (N,2,B)

    def ray_plane_y0(self, uv):
        d = np.column_stack([(uv[:, 0] - self.cx) / self.f, (uv[:, 1] - self.cy) / self.f, np.ones(len(uv))])
        w = np.einsum("nji,nj->ni", self.R, d)
        s = -self.C[:, 1] / w[:, 1]
        return self.C + s[:, None] * w


def Ry(b):
    c, s = math.cos(b), math.sin(b)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# ------------------------------------------------------------------------------------------- tracks
def load_tracks(path, points):
    import csv
    with open(path) as f:
        rd = csv.DictReader(f); rows = list(rd)
    fr = np.array([int(float(r["frame"])) for r in rows])
    Y = {}; S = {}
    for p in points:
        Y[p] = np.array([[float(r[f"{p}_u_px"]), float(r[f"{p}_v_px"])] for r in rows])
        S[p] = np.array([[float(r[f"{p}_sigma_u_px"]), float(r[f"{p}_sigma_v_px"])] for r in rows])
    ok = np.all([np.all(np.isfinite(Y[p]), 1) for p in points], 0)
    return dict(frames=fr[ok], t=fr[ok] / 24.0, y={p: Y[p][ok] for p in points}, s={p: S[p][ok] for p in points})


# ------------------------------------------------------------------------------------------- model
class Geom:
    def __init__(self, geom: dict, points):
        self.g = dict(mb.GEOM_DEFAULT); self.g.update(geom or {}); self.points = points


def build_model(P, geom: Geom):
    return mb.build_model(P, geom.g)


def points_from_Y(Y, M, geom, h, e=None):
    """Y (N,17,B) -> dict point -> (N,3,B) metres, gravity frame, origin = pivot.
    hook = H (hinge, eye inner top); ring = eye-ring centre = H + e n; wctr = CoM = cylinder centre; wbot = bottom face."""
    X, n = Y[:, 0:3], Y[:, 6:9]
    b = np.array([m.b for m in M])[None, None, :]
    H = X - b * n
    out = {"hook": H, "wctr": X}
    if e is not None:
        out["ring"] = H + np.array(e)[None, None, :] * n
    if "wbot" in geom.points:
        hh = np.array(h)[None, None, :]
        out["wbot"] = X + 0.5 * hh * n
    return out


class Fitter:
    def __init__(self, data, cams, geom, t_end):
        self.d, self.cams, self.geom = data, cams, geom
        self.frames = data["frames"]
        self.t_end = t_end
        self.tgrid = np.arange(0, int(round(t_end * 24)) + 1) / 24.0     # frame times 0..t_end (integration nodes)
        self.idx = self.frames                                          # rows of tgrid that are measured

    def predict(self, Ps, t_end=None, h=H_FIT):
        """Ps list of param dicts -> dict point -> uv (Nmeas,2,B) for measured frames up to t_end."""
        t_end = self.t_end if t_end is None else t_end
        tg = np.arange(0, int(round(t_end * 24)) + 1) / 24.0
        built = [build_model(P, self.geom) for P in Ps]
        Ms = [b[0] for b in built]
        st = {}
        try:
            Y = sp.integrate(Ms, tg, h, on_compression="record", stats=st)
        except sp.SolverGuardError:
            Y = np.full((len(tg), sp.NY, len(Ms)), np.nan); st["min_tension_N"] = [-1.0] * len(Ms)
        self.last_min_tension = np.array(st.get("min_tension_N", [1.0] * len(Ms)))
        pts = points_from_Y(Y, Ms, self.geom, [b[1]["h"] for b in built], [b[1]["e"] for b in built])
        sel = self.frames[self.frames <= int(round(t_end * 24))]
        cams = self.cams.subset(np.searchsorted(self.frames, sel))
        out = {}
        for j, P in enumerate(Ps):                                     # observed weight-centre definition nuisances
            dw, al = P.get("dw", 0.0), P.get("alpha", 0.0)
            if (dw != 0.0 or al != 0.0) and "wctr" in pts:
                lam = built[j][1]["lam"]; Mj = Ms[j]
                n = Y[:, 6:9, j]; H = Y[:, 0:3, j] - Mj.b * n
                nv = n @ Ry(al).T                                       # visible axis = n rotated by alpha about +Y
                pts["wctr"][:, :, j] = H + (Mj.b + dw / lam) * nv
        for p in self.geom.points:
            Xw = pts[p][sel]                                            # (n,3,B)
            uv = np.empty((len(sel), 2, len(Ps)))
            for j, P in enumerate(Ps):
                lam = built[j][1]["lam"]
                Xs = np.einsum("ji,nj->ni", Ry(P.get("beta", 0.0)), Xw[:, :, j]) * lam   # R_y(beta)^T X
                Xs[:, 2] -= P.get("pz", 0.0)                                            # pivot pz below A
                Xs[:, 0] += P.get("px", 0.0)
                uv[:, :, j] = cams.project(Xs)[:, :, 0]
            out[p] = uv
        return out, sel, built, Y

    def residual_matrix(self, Ps, mask_frames, y=None):
        pred, sel, _, _ = self.predict(Ps)
        y = self.d["y"] if y is None else y
        rows = np.searchsorted(self.frames, sel)
        m = mask_frames[rows]
        R = []
        for p in self.geom.points:
            r = (pred[p] - y[p][rows][:, :, None]) / self.d["s"][p][rows][:, :, None]
            R.append(r[m].reshape(-1, len(Ps)))
        Rm = np.concatenate(R, 0)
        bad = (self.last_min_tension < 0) | ~np.all(np.isfinite(Rm), 0)      # compressive spring / guard -> excluded region
        Rm[:, bad] = 1e4
        return Rm

    def fit(self, init, free, mask, fixed=None, y=None, max_nfev=60):
        names = [n for n in NAMES_ALL if n in free]
        base = dict(beta=0.0, phiy0=0.0, psiy0=0.0, phidy0=0.0, psidy0=0.0); base.update(init)
        base.pop("Q_fixed", None)
        fixed = fixed or {}
        lo = np.array([LO[n] for n in names]); hi = np.array([HI[n] for n in names])
        x0 = np.clip(np.array([base[n] for n in names], float), lo + 1e-12, hi - 1e-12)
        scale = np.array([max(abs(x), 1e-3) for x in x0])

        def P_of(x):
            P = dict(base); P.update(zip(names, x)); P.update(fixed)
            if "Q_fixed" in P:
                w = math.sqrt(G / P["ell"]); P["gamma_sw"] = w / (2 * P["Q_fixed"]); P["gamma_ax"] = P["rax"] * w / (2 * P["Q_fixed"])
            return P
        cache = {}

        def both(x):
            key = x.tobytes()
            if key in cache:
                return cache[key]
            steps = 1e-6 * scale
            X = [x] + [np.where(np.arange(len(x)) == i, np.minimum(x[i] + steps[i], hi[i]) if x[i] + steps[i] <= hi[i] else x[i] - steps[i], x) for i in range(len(x))]
            Rm = self.residual_matrix([P_of(xx) for xx in X], mask, y)
            r0 = Rm[:, 0]
            J = np.column_stack([(Rm[:, i + 1] - r0) / (X[i + 1][i] - x[i]) for i in range(len(x))])
            cache.clear(); cache[key] = (r0, J)
            return r0, J
        r = least_squares(lambda x: both(x)[0], x0, jac=lambda x: both(x)[1], bounds=(lo, hi), x_scale=scale,
                          xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=max_nfev)
        P = P_of(r.x)
        r0, J = both(r.x)
        n_obs = len(r0); chi2 = float(r0 @ r0); dof = n_obs - len(names); s2 = chi2 / max(dof, 1)
        Cv = np.linalg.pinv(J.T @ J, rcond=1e-14); sdf = np.sqrt(np.abs(np.diag(Cv)))
        corr = Cv / np.outer(sdf, sdf)
        at_b = {n: bool(abs(P[n] - LO[n]) < 1e-9 * max(1, abs(LO[n])) or abs(P[n] - HI[n]) < 1e-9 * max(1, abs(HI[n])) or
                        (n.startswith("gamma") and P[n] < 1e-9)) for n in names}
        return dict(P=P, names=names, chi2=chi2, dof=dof, reduced_chi2=s2, bic=chi2 + len(names) * math.log(n_obs),
                    sd_formal=dict(zip(names, sdf.tolist())), sd_scaled=dict(zip(names, (sdf * math.sqrt(max(s2, 1))).tolist())),
                    corr={a: {b: float(corr[i, j]) for j, b in enumerate(names)} for i, a in enumerate(names)},
                    at_bound=at_b, success=bool(r.success), nfev=int(r.nfev), cond_J=float(np.linalg.cond(J)), n_obs=n_obs)


# ------------------------------------------------------------------------------------------- helpers
def rms(r):
    return float(np.sqrt(np.mean(np.sum(r ** 2, axis=1)))) if len(r) else None


def sinefit(t, y, w_grid):
    best = None
    for w in w_grid:
        A = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones_like(t)])
        c, *_ = np.linalg.lstsq(A, y, rcond=None); rr = np.sum((A @ c - y) ** 2)
        if best is None or rr < best[0]:
            best = (rr, w, c)
    return best[1], best[2]


def turning_points(t, y):
    """parabola-refined extrema of a sampled signal: list of (t, value)."""
    out = []
    for i in range(1, len(y) - 1):
        if (y[i] - y[i - 1]) * (y[i + 1] - y[i]) < 0:
            c2, c1, c0 = np.polyfit(t[i - 1:i + 2] - t[i], y[i - 1:i + 2], 2)
            if c2 != 0:
                tt = -c1 / (2 * c2); out.append((t[i] + tt, c0 - c1 ** 2 / (4 * c2)))
    return out


def initial_guess(F, fitm, args):
    cams = F.cams; d = F.d
    A_px = None
    hp = "ring" if "ring" in d["y"] else "hook"
    Hw = cams.ray_plane_y0(d["y"][hp]); Ww = cams.ray_plane_y0(d["y"]["wctr"])
    Hw = Hw + np.array([-args.pivot_dx, 0, args.pivot_dz]); Ww = Ww + np.array([-args.pivot_dx, 0, args.pivot_dz])
    t = d["t"]; m = fitm
    phi = np.arctan2(Hw[:, 0], -Hw[:, 2]); psi = np.arctan2(Ww[:, 0] - Hw[:, 0], -(Ww[:, 2] - Hw[:, 2]))
    Lc = np.hypot(Hw[:, 0], Hw[:, 2]); bc = np.hypot(Ww[:, 0] - Hw[:, 0], Ww[:, 2] - Hw[:, 2])
    w_sw, c = sinefit(t[m], phi[m], np.linspace(2.0, 15.0, 500))
    Lcam = float(np.median(Lc[m])); bcam = float(np.median(bc[m]))
    # ell from the swing frequency with the two-body correction (iterate on the linear-mode formula)
    ell = G / w_sw ** 2
    for _ in range(20):
        P = dict(ell=ell, Lcam=Lcam, bcam=bcam, rax=2.0, pz=args.pivot_dz, phi0=0, psi0=0, de0=0, phid0=0, psid0=0, ded0=0, gamma_sw=0, gamma_ax=0)
        M, _ = build_model(P, F.geom)
        w_model = sp.lateral_modes(M)[0]
        ell *= (w_model / w_sw) ** 2
    phi0 = c[0] + c[2]; phid0 = c[1] * w_sw
    wp, cp = sinefit(t[m], psi[m], [w_sw]); psi0 = cp[0] + cp[2]; psid0 = cp[1] * w_sw
    lam = Lcam / ell
    de = (Lc - Lcam) / lam
    wa, ca = sinefit(t[m], de[m], np.linspace(1.5 * w_sw, 6 * w_sw, 400))
    amp_ax = float(np.hypot(ca[0], ca[1]))
    rax = wa / math.sqrt(G / ell)
    return dict(ell=ell, Lcam=Lcam, bcam=bcam, rax=float(np.clip(rax, 1.1, 19)), phi0=float(phi0), psi0=float(psi0),
                de0=float(ca[0] + ca[2]), phid0=float(phid0), psid0=float(psid0), ded0=float(ca[1] * wa),
                gamma_sw=0.0, gamma_ax=0.0, beta=0.0), dict(w_sw=float(w_sw), w_ax_guess=float(wa), amp_ax_m_guess=amp_ax,
                                                             phi_meas=phi, psi_meas=psi, de_meas=de, Lc=Lc)


def bootstrap(F, R, mask, nboot, seed, block, fixed):
    if nboot <= 0:
        return None
    pred, sel, _, _ = F.predict([R["P"]])
    rows = np.searchsorted(F.frames, sel)
    idx_fit = rows[mask[rows]]; n = len(idx_fit)
    rng = np.random.default_rng(seed); D = []
    for bb in range(nboot):
        ii = []
        while len(ii) < n:
            s0 = rng.integers(0, max(1, n - block + 1)); ii.extend(range(s0, min(s0 + block, n)))
        src = idx_fit[np.array(ii[:n])]
        yb = {p: F.d["y"][p].copy() for p in F.geom.points}
        for p in F.geom.points:
            model_uv = pred[p][:, :, 0]
            full = np.full_like(F.d["y"][p], np.nan); full[rows] = model_uv
            res = F.d["y"][p] - full
            yb[p][idx_fit] = full[idx_fit] + res[src]
        Rb = F.fit(R["P"], R["names"], mask, fixed=fixed, y=yb, max_nfev=30)
        D.append([Rb["P"][k] for k in R["names"]] + [Rb["P"]["gamma_sw"], Rb["P"]["gamma_ax"]])
        print(f"  boot {bb + 1}/{nboot}", flush=True)
    D = np.array(D); names = R["names"] + ["gamma_sw_derived", "gamma_ax_derived"]
    out = {"n": nboot, "block_len_frames": block, "method": "moving-block residual bootstrap on the anchor window, refit from best"}
    for i, k in enumerate(names):
        out[k] = {"sd": float(D[:, i].std(ddof=1)), "p2.5": float(np.percentile(D[:, i], 2.5)), "p97.5": float(np.percentile(D[:, i], 97.5))}
    Cc = np.corrcoef(D[:, :len(R["names"])].T)
    out["corr"] = {a: {b: float(Cc[i, j]) for j, b in enumerate(R["names"])} for i, a in enumerate(R["names"])}
    return out


def flags_for(R, boot):
    fl = {}
    for n in R["names"]:
        f = []
        sd = max(R["sd_scaled"][n], ((boot or {}).get(n) or {}).get("sd", 0.0))
        if R["at_bound"][n]:
            f.append(f"AT_BOUND (approx one-sided 95% upper limit {R['P'][n] + 1.645 * sd:.3g})")
        elif abs(R["P"][n]) > 0 and sd / abs(R["P"][n]) > 0.5:
            f.append(f"UNIDENTIFIED (relative sd {sd / abs(R['P'][n]):.2g})")
        for m, c in R["corr"][n].items():
            if m != n and abs(c) > 0.95:
                f.append(f"STRONGLY_CORRELATED with {m} ({c:+.3f})")
        fl[n] = f
    return fl


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default=str(ROOT / "measurement" / "tracks.csv"))
    ap.add_argument("--points", nargs="+", default=["hook", "wctr"])
    ap.add_argument("--camera", default="project", help="'project' (camera/project.py) or 'fake:<json>'")
    ap.add_argument("--t-anchor", type=float, required=True)
    ap.add_argument("--rax-fixed", type=float, default=None,
                    help="declared assumption omega_axial/sqrt(g/ell) when the axial bob is not identifiable (FORMULATION s6)")
    ap.add_argument("--pivot-dz", type=float, default=0.0, help="pivot depth below A in camera units (lead: P_eff 0.225071 W_in)")
    ap.add_argument("--pivot-dx", type=float, default=0.0)
    ap.add_argument("--fit-pivot", action="store_true", help="DIAGNOSTIC: fit pz")
    ap.add_argument("--k-coil", action="store_true", help="k from the coil formula G d^4/(8 D^3 N) (declared assumption)")
    ap.add_argument("--k-scale", type=float, default=1.0, help="sensitivity: multiply the coil-formula k")
    ap.add_argument("--rock-rest", action="store_true", help="rocking mode starts at rest: psi0, psid0 on the swing-mode shape (not fitted)")
    ap.add_argument("--axial-rest", action="store_true", help="axial mode starts at rest (de0 quasi-static, ded0 = 0; not fitted)")
    ap.add_argument("--fit-wctr-offset", action="store_true",
                    help="fit dw: offset of the MEASURED weight-centre marker along the body axis (camera units; point-definition nuisance)")
    ap.add_argument("--fit-body-offset", action="store_true",
                    help="fit alpha: static angle between the visible cylinder axis and the hinge->CoM line (static asymmetry)")
    ap.add_argument("--free-bcam", action="store_true", help="DIAGNOSTIC: fit the hinge->cylinder-centre distance instead of the measured ratio")
    ap.add_argument("--geom-json", default=None, help="JSON overriding model_build.GEOM_DEFAULT (measured ratios to ell)")
    ap.add_argument("--fit-tilt", action="store_true"); ap.add_argument("--fit-3d", action="store_true")
    ap.add_argument("--damping", choices=["auto", "fitted", "assumed"], default="auto")
    ap.add_argument("--nboot", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--truth", default=None, help="synthetic test: JSON with the true parameters for comparison")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = load_tracks(a.tracks, a.points)
    frames = data["frames"]
    cams = Cams.from_project(frames) if a.camera == "project" else Cams.from_fake(a.camera.split(":", 1)[1], frames)
    gj = json.loads(Path(a.geom_json).read_text()) if a.geom_json else {}
    if a.free_bcam:
        gj["b_mode"] = "fitted"
    if a.k_coil:
        gj["k_mode"] = "coil_formula"; gj["G_shear"] = 79.3e9 * a.k_scale
    if a.axial_rest:
        gj["axial_rest"] = True
    if a.rock_rest:
        gj["rock_rest"] = True
    geom = Geom(gj, a.points)
    F = Fitter(data, cams, geom, a.t_anchor)
    fitm = data["t"] <= a.t_anchor + 1e-9

    init, ig = initial_guess(F, fitm, a)
    init["pz"] = a.pivot_dz; init["px"] = a.pivot_dx; init["dw"] = 0.0; init["alpha"] = 0.0
    print("initial guess", json.dumps(init), json.dumps({k: v for k, v in ig.items() if not isinstance(v, np.ndarray)}), flush=True)
    base = (["ell", "Lcam"] + (["bcam"] if a.free_bcam else []) + ([] if (a.rax_fixed or a.k_coil) else ["rax"]) +
            ["phi0"] + ([] if a.rock_rest else ["psi0"]) + ([] if a.axial_rest else ["de0"]) + ["phid0"] + ([] if a.rock_rest else ["psid0"]) +
            ([] if a.axial_rest else ["ded0"]) +
            (["pz"] if a.fit_pivot else []) + (["dw"] if a.fit_wctr_offset else []) + (["alpha"] if a.fit_body_offset else []))
    if a.fit_tilt:
        base.append("beta")
    if a.fit_3d:
        base += ["phiy0", "psiy0", "phidy0", "psidy0"]
    # multi-start over the axial frequency (the axial bob may be weak): keep the best chi2
    if a.rax_fixed:
        init["rax"] = a.rax_fixed
    if a.axial_rest:
        init["de0"] = 0.0; init["ded0"] = 0.0
    starts = [a.rax_fixed] if a.rax_fixed else [init["rax"]] if a.k_coil else sorted(set([round(init["rax"], 3), 1.8, 2.0, 2.2, 3.0]))
    best = None; starts_log = []
    for rs in starts:
        R = F.fit(dict(init, rax=rs), base, fitm, fixed={"Q_fixed": Q_ASSUMED}, max_nfev=40)
        starts_log.append({"rax_start": rs, "rax": R["P"]["rax"], "chi2": R["chi2"]})
        print("start", starts_log[-1], flush=True)
        if best is None or R["chi2"] < best["chi2"]:
            best = R
    R0 = best  # MQ (Q fixed at the declared assumption)
    R1 = F.fit(dict(R0["P"], gamma_sw=max(R0["P"]["gamma_sw"], 1e-3), gamma_ax=max(R0["P"]["gamma_ax"], 1e-3)),
               base + ["gamma_sw", "gamma_ax"], fitm)
    sel = {"MQ": {"bic": R0["bic"], "chi2": R0["chi2"], "Q_fixed": Q_ASSUMED},
           "M1": {"bic": R1["bic"], "chi2": R1["chi2"], "gamma_sw": R1["P"]["gamma_sw"], "gamma_ax": R1["P"]["gamma_ax"],
                  "sd_scaled": {k: R1["sd_scaled"][k] for k in ("gamma_sw", "gamma_ax")}, "at_bound": {k: R1["at_bound"][k] for k in ("gamma_sw", "gamma_ax")}},
           "multistart": starts_log}
    boot1 = bootstrap(F, R1, fitm, min(a.nboot, 20), 11, block=24, fixed=None) if a.damping == "auto" and a.nboot > 0 else None
    w_sw = sp.lateral_modes(build_model(R1["P"], geom)[0])[0]
    Qsw = w_sw / (2 * R1["P"]["gamma_sw"]) if R1["P"]["gamma_sw"] > 0 else math.inf
    ident = (R1["bic"] < R0["bic"] - 2.0 and not R1["at_bound"]["gamma_sw"] and R1["P"]["gamma_sw"] > 3 * R1["sd_scaled"]["gamma_sw"]
             and boot1 is not None and boot1["gamma_sw_derived"]["p2.5"] > 0 and Q_RANGE[0] <= Qsw <= Q_RANGE[1])
    sel["damping_evidence"] = {"bootstrap_M1": {k: boot1[k] for k in ("gamma_sw_derived", "gamma_ax_derived")} if boot1 else None,
                               "Q_swing_M1": Qsw, "Q_plausible_range": Q_RANGE, "identified_rule": bool(ident), "mode": a.damping}
    use_M1 = (a.damping == "fitted") or (a.damping == "auto" and ident)
    R = R1 if use_M1 else R0
    fixed = None if use_M1 else {"Q_fixed": Q_ASSUMED}
    prod_label = "M1 (gamma_sw, gamma_ax fitted >= 0)" if use_M1 else f"MQ (Q = {Q_ASSUMED:g} declared assumption for swing and axial)"
    print("production:", prod_label, flush=True)
    boot = bootstrap(F, R, fitm, a.nboot, 7, block=24, fixed=fixed)
    P = R["P"]

    # ---- full-clip model, metrics, divergence map
    M, info = build_model(P, geom)
    tg = np.arange(0, 241) / 24.0
    Yfull = sp.integrate(M, tg, H_FIT)
    Fall = Fitter(data, cams, geom, 10.0)
    pred, selfr, _, _ = Fall.predict([P], t_end=10.0)
    rows = np.searchsorted(frames, selfr)
    resid = {p: pred[p][:, :, 0] - data["y"][p][rows] for p in a.points}
    anc = data["t"][rows] <= a.t_anchor + 1e-9
    n4 = (selfr % 4) == 0
    rall = np.concatenate([resid[p] for p in a.points], 0)
    anc2 = np.concatenate([anc for _ in a.points]); n42 = np.concatenate([n4 for _ in a.points])
    sig = float(np.median(np.concatenate([np.sqrt(np.mean(data["s"][p] ** 2, 1)) for p in a.points])))
    metrics = {"rms_anchor_px": rms(rall[anc2]), "rms_after_anchor_px": rms(rall[~anc2]),
               "per_point": {p: {"rms_anchor_px": rms(resid[p][anc]), "rms_anchor_u_v_px": np.sqrt(np.mean(resid[p][anc] ** 2, 0)).tolist(),
                                 "rms_after_px": rms(resid[p][~anc])} for p in a.points},
               "median_sigma_px": sig, "R2_anchor_target_px": max(2.0, 3 * sig),
               "supplementary_n_mod4_eq0": {"rms_anchor_px": rms(rall[anc2 & n42]), "rms_after_px": rms(rall[(~anc2) & n42])}}
    metrics["R2_anchor_pass"] = metrics["rms_anchor_px"] <= metrics["R2_anchor_target_px"]
    # angles in the static (camera) frame about the pivot P: swing = P -> ring (eye centre), body = ring -> weight centre,
    # length = |P - ring| / lam (model-conditioned m).  Measured: pixel ray ∩ plane Y = 0; model: same points, same mapping.
    piv = np.array([P.get("px", 0.0), 0.0, -P.get("pz", 0.0)])
    hp = "ring" if "ring" in data["y"] else "hook"
    Hm = cams.ray_plane_y0(data["y"][hp]) - piv; Wm = cams.ray_plane_y0(data["y"]["wctr"]) - piv
    phi_m = np.arctan2(Hm[:, 0], -Hm[:, 2]); psi_m = np.arctan2(Wm[:, 0] - Hm[:, 0], -(Wm[:, 2] - Hm[:, 2]))
    Lm = np.hypot(Hm[:, 0], Hm[:, 2]) / info["lam"]
    Rb = Ry(P.get("beta", 0.0))
    Hmod = Yfull[:, 0:3] - M.b * Yfull[:, 6:9]
    ringmod = (Hmod + (info["e"] if hp == "ring" else 0.0) * Yfull[:, 6:9]) @ Rb
    nvis = Yfull[:, 6:9] @ Ry(P.get("alpha", 0.0)).T
    Xs = (Hmod + (M.b + P.get("dw", 0.0) / info["lam"]) * nvis) @ Rb       # visible weight-centre marker
    phi_mod = np.arctan2(ringmod[:, 0], -ringmod[:, 2]); psi_mod = np.arctan2(Xs[:, 0] - ringmod[:, 0], -(Xs[:, 2] - ringmod[:, 2]))
    L_mod = np.hypot(ringmod[:, 0], ringmod[:, 2])
    div = {}
    for name, ym, ymod in (("swing_phi", phi_m, phi_mod[frames]), ("body_psi", psi_m, psi_mod[frames]), ("axial_len", Lm, L_mod[frames])):
        tpd = turning_points(data["t"], ym); tpm = turning_points(tg, L_mod if name == "axial_len" else (phi_mod if name == "swing_phi" else psi_mod))
        rowsd = []
        for (td, ad) in tpd:
            cand = [(tm, am) for tm, am in tpm if (am - np.mean(ymod)) * (ad - np.mean(ym)) > 0] or tpm
            tm, am = min(cand, key=lambda z: abs(z[0] - td))
            rowsd.append({"t_data_s": float(td), "A_data": float(ad), "t_model_s": float(tm), "A_model": float(am),
                          "dt_model_minus_data_s": float(tm - td), "in_anchor": bool(td <= a.t_anchor)})
        div[name] = rowsd
    divergence = {"turning_points": div, "per_frame_px_dist": {p: np.hypot(*resid[p].T).tolist() for p in a.points},
                  "units": {"swing_phi": "rad, pivot -> ring (static frame)", "body_psi": "rad, ring -> weight centre", "axial_len": "m, |pivot - ring| (model-conditioned)"},
                  "note": "after T_a this is a DIVERGENCE MAP (AI motion vs corrected physics), not a fit target"}
    der = M.derived()
    result = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {"tracks": str(a.tracks), "tracks_sha256": sha(a.tracks), "points": a.points, "camera": cams.source,
                   "camera_json_sha256": sha(ROOT / "params/camera.json") if a.camera == "project" else sha(a.camera.split(":", 1)[1]),
                   "geometry": geom.g, "h_fit_s": H_FIT},
        "model": sp.MODEL_ID, "fit_window": f"ANCHOR t <= {a.t_anchor} s", "t_anchor_s": a.t_anchor,
        "initial_guess": init, "initial_guess_info": {k: v for k, v in ig.items() if not isinstance(v, np.ndarray)},
        "model_selection": sel, "production_fit": prod_label, "params": P, "sd_scaled": R["sd_scaled"], "sd_formal": R["sd_formal"],
        "reduced_chi2": R["reduced_chi2"], "cond_J": R["cond_J"], "correlation": R["corr"], "bootstrap": boot,
        "flags": flags_for(R, boot), "fit_3d": a.fit_3d, "fit_tilt": a.fit_tilt, "rax_fixed": a.rax_fixed, "free_bcam": a.free_bcam, "pivot_dz": a.pivot_dz, "fit_pivot": a.fit_pivot,
        "k_coil": a.k_coil, "k_scale": a.k_scale, "axial_rest": a.axial_rest, "rock_rest": a.rock_rest,
        "derived": {"model": {k: v for k, v in info.items()}, "modes": der, "camera_units_per_m": info["lam"],
                    "Q_swing": (w_sw / (2 * P["gamma_sw"]) if P["gamma_sw"] > 0 else math.inf),
                    "provenance": "ell, b, r, h, k, L0: model-conditioned (frequencies + assumed g + assumed density); camera_units_per_m fitted"},
        "metrics": metrics, "divergence_map": divergence,
        "initial_state_gravity_frame": {"X0": list(M.X0), "V0": list(M.V0), "n0": list(M.n0), "u0": list(M.u0)},
    }
    (out / "fit_projected.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    (out / "ic.json").write_text(json.dumps({"X0": list(M.X0), "V0": list(M.V0), "n0": list(M.n0), "u0": list(M.u0),
                                             "source": f"fit_projected {out.name}, T_a {a.t_anchor} s"}, indent=2) + "\n")
    uvfull = {p: np.full((241, 2), np.nan) for p in a.points}
    for p in a.points:
        uvfull[p][selfr] = pred[p][:, :, 0]
    np.savez(out / "fit_series.npz", frames=frames, t=data["t"], **{f"y_{p}": data["y"][p] for p in a.points},
             **{f"s_{p}": data["s"][p] for p in a.points}, **{f"uv_{p}": uvfull[p] for p in a.points},
             tg=tg, phi_mod=phi_mod, psi_mod=psi_mod, L_mod=L_mod, phi_meas=phi_m, psi_meas=psi_m, L_meas=Lm, t_anchor=a.t_anchor)
    summary = {"production": prod_label, "params": P, "sd_scaled": R["sd_scaled"],
               "boot_sd": {k: v["sd"] for k, v in (boot or {}).items() if isinstance(v, dict) and "sd" in v},
               "flags": result["flags"], "metrics": metrics, "modes": {k: der[k] for k in ("omega_axial_rad_s", "omega_lateral_rad_s", "ratio_axial_over_swing", "static_stretch_m")},
               "cam_units_per_m": info["lam"], "selection": {k: sel[k] for k in ("MQ", "M1")}}
    if a.truth:
        T = json.loads(Path(a.truth).read_text())
        summary["truth_comparison"] = {k: {"true": T[k], "fit": P[k], "sd": (boot or {}).get(k, {}).get("sd", R["sd_scaled"].get(k)),
                                           "z": (P[k] - T[k]) / max((boot or {}).get(k, {}).get("sd", R["sd_scaled"].get(k, 1e-30)) or 1e-30, 1e-30)}
                                       for k in R["names"] if k in T}
    (out / "fit_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
