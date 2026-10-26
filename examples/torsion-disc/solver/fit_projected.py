"""Projected fit of the disc-on-wire model (physics/FORMULATION.md) to the reference measurements through the camera of
record, on the opening ANCHOR window [0, T_a] (CHANGELOG anchor-window rule); after T_a the physics runs unchanged and the
remainder is a DIVERGENCE MAP (AI motion vs corrected physics), never a fit target.

Observation channels (solver/NOTES.md "Interface"; one CSV, rows = frames, any subset of channels):
  pt:<name>   image point  <name>_u_px, <name>_v_px, <name>_sigma_u_px, <name>_sigma_v_px      [px]
              names: B (top-face centre = wire attachment), C (disc centre = CoM), mk1 / mk2 (rim ends of the mark line,
              top face; mk1 at body azimuth mu, mk2 at mu + pi), A (clamp, static check)
  psi         world azimuth of the mark direction (mk2 -> mk1 ... i.e. body azimuth mu), horizontal projection, measured on the
              rectified disc: psi_deg, psi_sigma_deg   [deg]   (unwrapped; an arbitrary multiple of 180 deg is absorbed in mu)
  Bw          world position of B (top-face centre) in camera units from measurement's per-frame pose fit (camera-projected
              silhouette fit): Bw_x, Bw_y, Bw_z, Bw_sigma_x, Bw_sigma_y, Bw_sigma_z   [camera units]
  nrm         disc UP-normal, world x / y components: nrm_x, nrm_y, nrm_sigma   [-]  (tilt; small-angle radians)
Model -> image: solver state (metres, gravity frame, A = origin) x lam (camera units per metre, lam = Lcam / ell) -> per-frame
camera projection (camera/project.py load_camera(frame)); angles are compared directly.

Free parameters (bounded, physical; FORMULATION s6): Lcam, om_t, psi0, psid0, mu, phx0, phy0, phxd0, phyd0
  [+ ell (scale_mode swing)] [+ thx0, thy0, thxd0, thyd0 (--free-tilt)] [+ gamma_t >= 0 (M1)]
Damping: MQ = Q_t declared (1000) vs M1 = gamma_t fitted >= 0; M1 adopted only if identifiable (BIC, bootstrap, Q range).

Usage:
  .venv/bin/python solver/fit_projected.py --obs measurement/tracks_for_fit.csv --channels pt:B psi nrm --t-anchor 5.0 \
      [--camera project|fake:<json>] [--geom-json G.json] [--scale-mode prompt|swing] [--free-tilt] [--nboot 20] --out DIR
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import torsion_disc as td  # noqa: E402
import model_build as mb  # noqa: E402

G = td.G_STD
Q_T_ASSUMED = 1000.0
Q_T_RANGE = (100.0, 2e4)
Q_SW, Q_TILT = 500.0, 50.0
H_FIT = 1.0 / 384.0

NAMES_ALL = ["Ax", "Ay", "ell", "Lcam", "om_t", "psi0", "psid0", "mu", "phx0", "phy0", "phxd0", "phyd0", "thx0", "thy0", "thxd0", "thyd0", "gamma_t"]
LO = dict(Ax=-0.2, Ay=-0.2, ell=0.01, Lcam=1e-4, om_t=0.02, psi0=-6.0, psid0=-10, mu=-20, phx0=-0.3, phy0=-0.3, phxd0=-5, phyd0=-5,
          thx0=-0.3, thy0=-0.3, thxd0=-10, thyd0=-10, gamma_t=0.0)
HI = dict(Ax=0.2, Ay=0.2, ell=3.0, Lcam=1e5, om_t=30.0, psi0=6.0, psid0=10, mu=20, phx0=0.3, phy0=0.3, phxd0=5, phyd0=5,
          thx0=0.3, thy0=0.3, thxd0=10, thyd0=10, gamma_t=2.0)


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
        return np.stack([self.cx[:, None] + self.f[:, None] * Xc[:, 0] / Xc[:, 2],
                         self.cy[:, None] + self.f[:, None] * Xc[:, 1] / Xc[:, 2]], 1)  # (N,2,B)

    def ray_plane(self, uv, normal, d):
        """intersection of pixel rays with the plane normal . X + d = 0 (camera units)."""
        dc = np.column_stack([(uv[:, 0] - self.cx) / self.f, (uv[:, 1] - self.cy) / self.f, np.ones(len(uv))])
        w = np.einsum("nji,nj->ni", self.R, dc)
        s = -(self.C @ normal + d) / (w @ normal)
        return self.C + s[:, None] * w


# ------------------------------------------------------------------------------------------- observations
def load_obs(path, channels):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    fr = np.array([int(float(r["frame"])) for r in rows])
    y, s, kind = {}, {}, {}
    for ch in channels:
        if ch.startswith("pt:"):
            p = ch[3:]
            y[ch] = np.array([[float(r[f"{p}_u_px"]), float(r[f"{p}_v_px"])] for r in rows])
            s[ch] = np.array([[float(r[f"{p}_sigma_u_px"]), float(r[f"{p}_sigma_v_px"])] for r in rows]); kind[ch] = "px"
        elif ch == "Bw":
            y[ch] = np.array([[float(r["Bw_x"]), float(r["Bw_y"]), float(r["Bw_z"])] for r in rows])
            s[ch] = np.array([[float(r["Bw_sigma_x"]), float(r["Bw_sigma_y"]), float(r["Bw_sigma_z"])] for r in rows]); kind[ch] = "cu"
        elif ch == "psi":
            y[ch] = np.radians(np.array([[float(r["psi_deg"])] for r in rows]))
            s[ch] = np.radians(np.array([[float(r["psi_sigma_deg"])] for r in rows])); kind[ch] = "rad"
        elif ch == "nrm":
            y[ch] = np.array([[float(r["nrm_x"]), float(r["nrm_y"])] for r in rows])
            s[ch] = np.array([[float(r["nrm_sigma"]), float(r["nrm_sigma"])] for r in rows]); kind[ch] = "rad"
        else:
            raise ValueError(ch)
    ok = np.all([np.all(np.isfinite(y[c]), 1) & np.all(np.isfinite(s[c]), 1) for c in channels], 0)
    if "psi" in y:
        y["psi"][ok, 0] = np.unwrap(y["psi"][ok, 0])
    return dict(frames=fr[ok], t=fr[ok] / 24.0, y={c: y[c][ok] for c in channels}, s={c: s[c][ok] for c in channels},
                kind=kind, channels=list(channels))


# ------------------------------------------------------------------------------------------- model features
def body_axes(q):
    """q (N,4,B) -> e1, e2, e3 (N,3,B) body axes in world."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    e1 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], 1)
    e2 = np.stack([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], 1)
    e3 = np.stack([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)], 1)
    return e1, e2, e3


def features(Y, Ms, mus):
    """Y (N,14,B) -> dict: B, C, mk1, mk2 (N,3,B) metres; mark azimuth (N,B) (unwrapped); up-normal (N,3,B)."""
    X = Y[:, 0:3]
    e1, e2, e3 = body_axes(Y[:, 6:10])
    hc = np.array([M.h_c for M in Ms])[None, None, :]; R = np.array([M.R for M in Ms])[None, None, :]
    mu = np.array(mus)[None, None, :]
    B = X + hc * e3
    md = np.cos(mu) * e1 + np.sin(mu) * e2
    d = np.arctan2(md[:, 1], md[:, 0]) - mu[0]
    az = mu[0] + np.unwrap((d + np.pi) % (2 * np.pi) - np.pi, axis=0)          # branch continuous with mu (|twist| < pi)
    return {"B": B, "C": X, "mk1": B + R * md, "mk2": B - R * md, "A": np.zeros_like(X), "az": az, "nup": e3}


class Fitter:
    def __init__(self, data, cams, geom, t_end):
        self.d, self.cams, self.geom = data, cams, geom
        self.frames = data["frames"]
        self.t_end = t_end

    def predict(self, Ps, t_end=None, h=H_FIT):
        t_end = self.t_end if t_end is None else t_end
        tg = np.arange(0, int(round(t_end * 24)) + 1) / 24.0
        built = [mb.build_model(P, self.geom) for P in Ps]
        Ms = [b[0] for b in built]
        try:
            Y = td.integrate(Ms, tg, h, on_slack="record")
            self.bad = np.zeros(len(Ms), bool)
        except td.SolverGuardError:
            Y = np.full((len(tg), td.NY, len(Ms)), np.nan); self.bad = np.ones(len(Ms), bool)
        F = features(Y, Ms, [P.get("mu", 0.0) for P in Ps])
        sel = self.frames[self.frames <= int(round(t_end * 24))]
        rows = np.searchsorted(self.frames, sel)
        cams = self.cams.subset(rows)
        lam = np.array([b[1]["lam"] for b in built])
        piv = np.array([[P.get("Ax", 0.0), P.get("Ay", 0.0), 0.0] for P in Ps]).T[None]     # (1,3,B) pivot offset, camera units
        out = {}
        for ch in self.d["channels"]:
            if ch.startswith("pt:"):
                Xw = F[ch[3:]][sel] * lam[None, None, :] + piv
                out[ch] = cams.project(Xw)
            elif ch == "Bw":
                out[ch] = F["B"][sel] * lam[None, None, :] + piv
            elif ch == "psi":
                out[ch] = F["az"][sel][:, None, :]
            elif ch == "nrm":
                out[ch] = F["nup"][sel][:, 0:2, :]
        return out, sel, built, Y

    def residual_matrix(self, Ps, mask_frames, y=None):
        pred, sel, _, _ = self.predict(Ps)
        y = self.d["y"] if y is None else y
        rows = np.searchsorted(self.frames, sel)
        R = []
        for ch in self.d["channels"]:
            m = (mask_frames[ch] if isinstance(mask_frames, dict) else mask_frames)[rows]
            dlt = pred[ch] - y[ch][rows][:, :, None]
            if ch == "psi":                               # line orientation: the mark is a full diameter -> compare mod pi
                dlt = wrap_half(dlt)
            r = dlt / self.d["s"][ch][rows][:, :, None]
            R.append(r[m].reshape(-1, len(Ps)))
        Rm = np.concatenate(R, 0)
        Rm[:, self.bad | ~np.all(np.isfinite(Rm), 0)] = 1e4
        return Rm

    def fit(self, init, free, mask, fixed=None, y=None, max_nfev=60):
        names = [n for n in NAMES_ALL if n in free]
        base = dict(init); fixed = fixed or {}
        lo = np.array([LO[n] for n in names]); hi = np.array([HI[n] for n in names])
        x0 = np.clip(np.array([base[n] for n in names], float), lo + 1e-12, hi - 1e-12)
        scale = np.array([max(abs(x), 1e-2) for x in x0])

        def P_of(x):
            P = dict(base); P.update(zip(names, x)); P.update(fixed)
            if "Q_t_fixed" in P:
                P["gamma_t"] = P["om_t"] / (2 * P["Q_t_fixed"])
            return P
        cache = {}

        def both(x):
            key = x.tobytes()
            if key in cache:
                return cache[key]
            steps = 1e-6 * scale
            X = [x] + [np.where(np.arange(len(x)) == i, x[i] + steps[i] if x[i] + steps[i] <= hi[i] else x[i] - steps[i], x) for i in range(len(x))]
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
                        (n == "gamma_t" and P[n] < 1e-9)) for n in names}
        return dict(P=P, names=names, chi2=chi2, dof=dof, reduced_chi2=s2, bic=chi2 + len(names) * math.log(n_obs),
                    sd_formal=dict(zip(names, sdf.tolist())), sd_scaled=dict(zip(names, (sdf * math.sqrt(max(s2, 1))).tolist())),
                    corr={a: {b: float(corr[i, j]) for j, b in enumerate(names)} for i, a in enumerate(names)},
                    at_bound=at_b, success=bool(r.success), nfev=int(r.nfev), cond_J=float(np.linalg.cond(J)), n_obs=n_obs)


# ------------------------------------------------------------------------------------------- helpers
def sinefit(t, y, w_grid, decay=0.0):
    best = None
    for w in w_grid:
        A = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones_like(t)])
        c, *_ = np.linalg.lstsq(A, y, rcond=None); rr = np.sum((A @ c - y) ** 2)
        if best is None or rr < best[0]:
            best = (rr, w, c)
    return best[1], best[2]


def turning_points(t, y, k=2):
    """local-quadratic refined extrema (fit over +-k samples): list of (t, value)."""
    out = []
    for i in range(k, len(y) - k):
        if (y[i] - y[i - 1]) * (y[i + 1] - y[i]) < 0 and (y[i] >= y[i - k:i + k + 1].max() or y[i] <= y[i - k:i + k + 1].min()):
            c2, c1, c0 = np.polyfit(t[i - k:i + k + 1] - t[i], y[i - k:i + k + 1], 2)
            if c2 != 0:
                tt = -c1 / (2 * c2); out.append((t[i] + tt, c0 - c1 ** 2 / (4 * c2)))
    return out


def wrap_half(x):
    """wrap an angle difference to [-pi/2, pi/2) (orientation of a full-diameter line)."""
    return (x + np.pi / 2) % np.pi - np.pi / 2


def twist_series(az, mu):
    """continuous twist (rad) of a line orientation series relative to mu (mod pi branch chosen near 0 at the start)."""
    return np.unwrap(wrap_half(np.asarray(az) - mu), period=np.pi)


def rms(r):
    return float(np.sqrt(np.mean(np.sum(r ** 2, axis=1)))) if len(r) else None


def initial_guess(F, fitm, a, geom):
    d = F.d; cams = F.cams; t = d["t"]; m = fitm
    g = dict(mb.GEOM_DEFAULT); g.update(geom)
    # Lcam: B (or C) ray with the vertical plane through A facing the camera (Y = 0)
    pt = "pt:B" if "pt:B" in d["y"] else ("pt:C" if "pt:C" in d["y"] else None)
    if "Bw" in d["y"]:
        Bw = d["y"]["Bw"]; Lcam = float(np.median(np.linalg.norm(Bw[m], axis=1)))
        phx, phy = np.arctan2(Bw[:, 0], -Bw[:, 2]), np.arctan2(Bw[:, 1], -Bw[:, 2]); pt = "Bw"
    elif pt is not None:
        Bw = cams.ray_plane(d["y"][pt], np.array([0.0, 1.0, 0.0]), 0.0)
        Lcam = float(np.median(np.hypot(Bw[m, 0], Bw[m, 2])))
        Bh = cams.ray_plane(d["y"][pt], np.array([0.0, 0.0, 1.0]), Lcam)          # horizontal plane z = -Lcam
        if a.fit_pivot:
            a.piv0 = (float(np.mean(Bh[m, 0])), float(np.mean(Bh[m, 1]))); Bh = Bh - np.array([a.piv0[0], a.piv0[1], 0.0])
        phx, phy = Bh[:, 0] / Lcam, Bh[:, 1] / Lcam
    else:
        Lcam = a.lcam_init; phx = phy = np.zeros(len(t))
    if "psi" in d["y"]:
        az = d["y"]["psi"][:, 0]
    else:
        e = cams.ray_plane(d["y"]["pt:mk1"], np.array([0, 0, 1.0]), Lcam) - cams.ray_plane(d["y"]["pt:mk2"], np.array([0, 0, 1.0]), Lcam)
        az = np.unwrap(np.arctan2(e[:, 1], e[:, 0]))
    if a.om_t_start:
        # slow twist (no turning point in the window): local quadratic az ~ a0 + a1 t + a2 t^2, psi'' = -om^2 (az - mu)
        a2, a1, a0 = np.polyfit(t[m], az[m], 2)
        w_t = a.om_t_start; mu = float(a0 + 2 * a2 / w_t ** 2); psi0 = float(a0 - mu); psid0 = float(a1)
    else:
        w_t, c = sinefit(t[m], az[m], np.linspace(0.3, 12.0, 1200))
        psi0 = float(c[0]); psid0 = float(c[1] * w_t); mu = float(c[2])
    # swing (if any): frequency guess from the point pendulum at the prompt / swing scale
    lam_p = g["R_cam"] / g["R_prompt"]
    ell = Lcam / lam_p if a.scale_mode == "prompt" else (a.ell_init or Lcam / lam_p)
    if g.get("bend_ends") == "rigid_disc":
        global Q_TILT
        Q_TILT = Q_SW          # rigid joint: the disc rocks only with the swing (declared swing-air-drag Q for both terms)
    P0 = dict(ell=ell, Lcam=Lcam, om_t=w_t, psi0=psi0, psid0=psid0, mu=mu, phx0=0.0, phy0=0.0, phxd0=0.0, phyd0=0.0,
              thx0=0.0, thy0=0.0, thxd0=0.0, thyd0=0.0, gamma_t=w_t / (2 * Q_T_ASSUMED), Q_sw=Q_SW, Q_tilt=Q_TILT, Ax=0.0, Ay=0.0)
    if getattr(a, "piv0", None):
        P0["Ax"], P0["Ay"] = a.piv0
    M, info = mb.build_model(P0, geom)
    w1 = info["omega_lateral"][0]
    if pt is not None:
        wx, cx = sinefit(t[m], phx[m], np.linspace(0.4 * w1, 2.5 * w1, 500))
        wy, cy = sinefit(t[m], phy[m], [wx])
        P0.update(phx0=float(cx[0]), phxd0=float(cx[1] * wx), phy0=float(cy[0]), phyd0=float(cy[1] * wx))
        if a.scale_mode == "swing":
            for _ in range(20):
                M, info = mb.build_model(P0, geom); P0["ell"] *= (info["omega_lateral"][0] / wx) ** 2
    else:
        wx = None
    return P0, dict(w_twist=float(w_t), w_swing_guess=(float(wx) if wx else None), w1_model=float(w1), Lcam_meas=Lcam)


def bootstrap(F, R, mask, nboot, seed, block, fixed):
    if nboot <= 0:
        return None
    pred, sel, _, _ = F.predict([R["P"]])
    rows = np.searchsorted(F.frames, sel)
    rng = np.random.default_rng(seed); D = []; YB = []
    for bb in range(nboot):
        yb = {c: F.d["y"][c].copy() for c in F.d["channels"]}
        for c in F.d["channels"]:
            mc = mask[c] if isinstance(mask, dict) else mask
            idx_fit = rows[mc[rows]]; n = len(idx_fit)
            ii = []
            while len(ii) < n:
                s0 = rng.integers(0, max(1, n - block + 1)); ii.extend(range(s0, min(s0 + block, n)))
            src = idx_fit[np.array(ii[:n])]
            full = np.full_like(F.d["y"][c], np.nan); full[rows] = pred[c][:, :, 0]
            res = F.d["y"][c] - full
            if c == "psi":
                res = -wrap_half(full - F.d["y"][c])
            yb[c][idx_fit] = full[idx_fit] + res[src]
        YB.append(yb)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(NWORK, nboot)) as ex:
        for bb, Rb in enumerate(ex.map(_boot_one, [(F, R["P"], R["names"], mask, fixed, yb) for yb in YB])):
            D.append([Rb["P"][k] for k in R["names"]] + [Rb["P"]["gamma_t"]])
            print(f"  boot {bb + 1}/{nboot}", flush=True)
    D = np.array(D); names = R["names"] + ["gamma_t_derived"]
    out = {"n": nboot, "block_len_frames": block, "method": "moving-block residual bootstrap on the anchor window, refit from best"}
    for i, k in enumerate(names):
        out[k] = {"sd": float(D[:, i].std(ddof=1)), "p2.5": float(np.percentile(D[:, i], 2.5)), "p97.5": float(np.percentile(D[:, i], 97.5))}
    return out


NWORK = 8


def _boot_one(args):
    F, P, names, mask, fixed, yb = args
    return F.fit(P, names, mask, fixed=fixed, y=yb, max_nfev=30)


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
    ap.add_argument("--obs", required=True)
    ap.add_argument("--channels", nargs="+", default=["pt:B", "psi", "nrm"], help="pt:<name> | Bw | psi | nrm")
    ap.add_argument("--camera", default="project", help="'project' (camera/project.py) or 'fake:<json>'")
    ap.add_argument("--t-anchor", type=float, required=True, help="anchor window end [s] (all channels unless overridden)")
    ap.add_argument("--t-anchor-channel", nargs="*", default=[],
                    help="per-mode windows, e.g. psi=7.5 pt:B=2.5 nrm=2.5 (lead decision 2026-10-26T03:42Z)")
    ap.add_argument("--bend-ends", choices=["clamped", "rigid_disc", "free"], default="rigid_disc")
    ap.add_argument("--geom-json", default=None, help="JSON overriding model_build.GEOM_DEFAULT (measured camera-unit ratios)")
    ap.add_argument("--scale-mode", choices=["prompt", "swing"], default="prompt")
    ap.add_argument("--ell-init", type=float, default=None)
    ap.add_argument("--lcam-init", type=float, default=1.0)
    ap.add_argument("--om-t-start", type=float, nargs="*", default=None,
                    help="multi-start torsion frequencies [rad/s] (slow twist without turning points); best chi2 kept")
    ap.add_argument("--fit-pivot", action="store_true",
                    help="fit the horizontal suspension point offset (Ax, Ay) [camera units] relative to A (camera role placed A by "
                         "assumption vertically above the frame-0 disc centre; static-asymmetry option, clip-01/02 precedent)")
    ap.add_argument("--free-tilt", action="store_true", help="tilt (wobble) mode ICs free instead of on the swing-mode shape")
    ap.add_argument("--no-swing", action="store_true", help="swing ICs fixed at 0 (diagnostic)")
    ap.add_argument("--damping", choices=["auto", "fitted", "assumed"], default="auto")
    ap.add_argument("--nboot", type=int, default=20)
    ap.add_argument("--max-nfev", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--truth", default=None, help="synthetic test: JSON with the true parameters for comparison")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = load_obs(a.obs, a.channels)
    frames = data["frames"]
    cams = Cams.from_project(frames) if a.camera == "project" else Cams.from_fake(a.camera.split(":", 1)[1], frames)
    gj = json.loads(Path(a.geom_json).read_text()) if a.geom_json else {}
    gj["scale_mode"] = a.scale_mode
    gj["tilt_mode"] = "free" if a.free_tilt else "mode1"
    gj["bend_ends"] = a.bend_ends
    Ta = {c: a.t_anchor for c in a.channels}
    for item in a.t_anchor_channel:
        k, v = item.split("="); Ta[k] = float(v)
    t_int = max(Ta.values())
    F = Fitter(data, cams, gj, t_int)
    fitmask = {c: data["t"] <= Ta[c] + 1e-9 for c in a.channels}
    fitm = data["t"] <= t_int + 1e-9          # union (used by the initial guess)
    starts = a.om_t_start or [None]
    a.om_t_start = starts[0]
    init, ig = initial_guess(F, fitm, a, gj)
    print("initial guess", json.dumps(init), json.dumps(ig), flush=True)
    base = (["Ax", "Ay"] if a.fit_pivot else []) + (["ell"] if a.scale_mode == "swing" else []) + ["Lcam", "om_t", "psi0", "psid0", "mu"]
    if not a.no_swing:
        base += ["phx0", "phy0", "phxd0", "phyd0"]
    if a.free_tilt:
        base += ["thx0", "thy0", "thxd0", "thyd0"]
    if not any(c.startswith("pt:") or c == "Bw" for c in a.channels):
        base.remove("Lcam")
    R0 = None; starts_log = []
    for w0 in starts:
        a.om_t_start = w0
        init_w, _ = initial_guess(F, fitm, a, gj) if w0 is not None else (init, None)
        Rw = F.fit(init_w, base, fitmask, fixed={"Q_t_fixed": Q_T_ASSUMED}, max_nfev=a.max_nfev)
        starts_log.append({"om_t_start": w0, "om_t": Rw["P"]["om_t"], "mu": Rw["P"]["mu"], "chi2": Rw["chi2"]})
        print("start", starts_log[-1], flush=True)
        if R0 is None or Rw["chi2"] < R0["chi2"]:
            R0 = Rw
    print("MQ chi2", R0["chi2"], "nfev", R0["nfev"], flush=True)
    P1 = {k: v for k, v in R0["P"].items() if k != "Q_t_fixed"}
    R1 = F.fit(dict(P1, gamma_t=max(P1["gamma_t"], 1e-4)), base + ["gamma_t"], fitmask, max_nfev=a.max_nfev)
    print("M1 chi2", R1["chi2"], "gamma_t", R1["P"]["gamma_t"], flush=True)
    sel = {"MQ": {"bic": R0["bic"], "chi2": R0["chi2"], "Q_t_fixed": Q_T_ASSUMED, "multistart": starts_log},
           "M1": {"bic": R1["bic"], "chi2": R1["chi2"], "gamma_t": R1["P"]["gamma_t"], "sd_scaled": R1["sd_scaled"]["gamma_t"],
                  "at_bound": R1["at_bound"]["gamma_t"]}}
    pre = (R1["bic"] < R0["bic"] - 2.0 and not R1["at_bound"]["gamma_t"] and R1["P"]["gamma_t"] > 3 * R1["sd_scaled"]["gamma_t"])
    boot1 = bootstrap(F, R1, fitmask, min(a.nboot, 10), 11, block=24, fixed=None) if a.damping == "auto" and a.nboot > 0 and pre else None
    Qt = R1["P"]["om_t"] / (2 * R1["P"]["gamma_t"]) if R1["P"]["gamma_t"] > 0 else math.inf
    ident = (R1["bic"] < R0["bic"] - 2.0 and not R1["at_bound"]["gamma_t"] and R1["P"]["gamma_t"] > 3 * R1["sd_scaled"]["gamma_t"]
             and boot1 is not None and boot1["gamma_t_derived"]["p2.5"] > 0 and Q_T_RANGE[0] <= Qt <= Q_T_RANGE[1])
    sel["damping_evidence"] = {"bootstrap_M1_gamma_t": boot1["gamma_t_derived"] if boot1 else "skipped: M1 fails the BIC / bound / 3-sigma pre-test", "Q_t_M1": Qt,
                               "Q_plausible_range": Q_T_RANGE, "identified_rule": bool(ident), "mode": a.damping}
    use_M1 = (a.damping == "fitted") or (a.damping == "auto" and ident)
    R = R1 if use_M1 else R0
    fixed = None if use_M1 else {"Q_t_fixed": Q_T_ASSUMED}
    prod_label = "M1 (gamma_t fitted >= 0)" if use_M1 else f"MQ (Q_t = {Q_T_ASSUMED:g} declared assumption)"
    print("production:", prod_label, flush=True)
    boot = bootstrap(F, R, fitmask, a.nboot, 7, block=24, fixed=fixed)
    P = dict(R["P"]); P.pop("Q_t_fixed", None)
    P["Q_t"] = P["om_t"] / (2 * P["gamma_t"]) if P["gamma_t"] > 0 else None

    # ---- full-clip model, metrics, divergence map
    M, info = mb.build_model(P, gj)
    tg = np.arange(0, 241) / 24.0
    Fall = Fitter(data, cams, gj, 10.0)
    pred, selfr, _, Yfull = Fall.predict([P], t_end=10.0)
    Yfull = Yfull[:, :, 0]
    rows = np.searchsorted(frames, selfr)
    ancc = {c: data["t"][rows] <= Ta[c] + 1e-9 for c in a.channels}
    anc = ancc[a.channels[0]]
    n4 = (selfr % 4) == 0
    resid = {c: pred[c][:, :, 0] - data["y"][c][rows] for c in a.channels}
    if "psi" in resid:
        resid["psi"] = wrap_half(resid["psi"])
    metrics = {"per_channel": {}}
    for c in a.channels:
        k = data["kind"][c]; f = {"px": 1.0, "rad": 180 / math.pi, "cu": 1.0}[k]
        u = {"px": "px", "rad": "deg", "cu": "camera_units"}[k]
        anc = ancc[c]
        metrics["per_channel"][c] = {"t_anchor_s": Ta[c], f"rms_anchor_{u}": rms(f * resid[c][anc]), f"rms_after_{u}": rms(f * resid[c][~anc]),
                                     f"median_sigma_{u}": float(np.median(f * np.sqrt(np.mean(data['s'][c] ** 2, 1)))),
                                     f"rms_anchor_n_mod4_eq0_{u}": rms(f * resid[c][anc & n4])}
    px = [c for c in a.channels if data["kind"][c] == "px"]
    if px:
        rall = np.concatenate([resid[c] for c in px], 0); anc2 = np.concatenate([ancc[c] for c in px])
        sig = float(np.median(np.concatenate([np.sqrt(np.mean(data["s"][c] ** 2, 1)) for c in px])))
        metrics.update(rms_anchor_px=rms(rall[anc2]), rms_after_anchor_px=rms(rall[~anc2]), median_sigma_px=sig,
                       R2_anchor_target_px=max(2.0, 3 * sig))
        metrics["R2_anchor_pass"] = metrics["rms_anchor_px"] <= metrics["R2_anchor_target_px"]
    if "psi" in a.channels:
        metrics["rms_anchor_twist_deg"] = metrics["per_channel"]["psi"]["rms_anchor_deg"]
        metrics["rms_after_twist_deg"] = metrics["per_channel"]["psi"]["rms_after_deg"]
    # divergence map: turning points of twist (mark azimuth) and sway (B horizontal, world, via the model's own projection)
    Fm = features(Yfull[:, :, None], [M], [P.get("mu", 0.0)])
    div = {}
    series = {}
    if "psi" in a.channels:
        series["twist_deg"] = (np.degrees(twist_series(data["y"]["psi"][:, 0], P["mu"])), np.degrees(Fm["az"][:, 0] - P["mu"]))
    if "nrm" in a.channels:
        series["tilt_x_rad"] = (data["y"]["nrm"][:, 0], Fm["nup"][:, 0, 0]); series["tilt_y_rad"] = (data["y"]["nrm"][:, 1], Fm["nup"][:, 1, 0])
    if "Bw" in a.channels:
        BwM = Fm["B"][:, :, 0] * info["lam"]
        series["B_x_cu"] = (data["y"]["Bw"][:, 0], BwM[:, 0]); series["B_y_cu"] = (data["y"]["Bw"][:, 1], BwM[:, 1])
    if "pt:B" in a.channels:
        series["B_u_px"] = (data["y"]["pt:B"][:, 0], np.interp(tg, data["t"][rows], pred["pt:B"][:, 0, 0]))
    for name, (ym, ymod) in series.items():
        tpd = turning_points(data["t"], ym); tpm = turning_points(tg, ymod)
        rowsd = []
        for (tdd, ad) in tpd:
            cand = [(tm, am) for tm, am in tpm if (am - np.mean(ymod)) * (ad - np.mean(ym)) > 0] or tpm
            if not cand:
                continue
            tm, am = min(cand, key=lambda z: abs(z[0] - tdd))
            rowsd.append({"t_data_s": float(tdd), "A_data": float(ad), "t_model_s": float(tm), "A_model": float(am),
                          "dt_model_minus_data_s": float(tm - tdd), "in_anchor": bool(tdd <= (Ta.get("psi", a.t_anchor) if name.startswith("twist") else min(Ta.values())))})
        div[name] = rowsd
    divergence = {"turning_points": div, "note": "after T_a this is a DIVERGENCE MAP (AI motion vs corrected physics), not a fit target"}
    # twist <-> sway coupling (justifies per-mode windows): rerun with the other mode's ICs removed
    cpl = {}
    try:
        Pns = dict(P, phx0=0.0, phy0=0.0, phxd0=0.0, phyd0=0.0, thx0=0.0, thy0=0.0, thxd0=0.0, thyd0=0.0)
        Pnt = dict(P, psi0=0.0, psid0=0.0)
        Ms_, _ = mb.build_model(Pns, gj); Mt_, _ = mb.build_model(Pnt, gj)
        Yc = td.integrate([M, Ms_, Mt_], tg, H_FIT)
        Fc = features(Yc, [M, Ms_, Mt_], [P["mu"]] * 3)
        cpl = {"max_abs_twist_change_from_sway_deg": float(np.degrees(np.max(np.abs(Fc["az"][:, 0] - Fc["az"][:, 1])))),
               "max_abs_B_change_from_twist_m": float(np.max(np.abs(Fc["B"][:, :, 0] - Fc["B"][:, :, 2]))),
               "max_abs_B_change_from_twist_px_approx": float(np.max(np.abs(Fc["B"][:, :, 0] - Fc["B"][:, :, 2])) * info["lam"] * 2602 / 4.7),
               "note": "full model vs the same model with the sway (resp. twist) initial state removed, 0-10 s"}
    except Exception as e:  # noqa: BLE001
        cpl = {"error": str(e)[:200]}
    print("coupling", cpl, flush=True)
    der = M.derived()
    result = {
        "twist_sway_coupling": cpl,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {"obs": str(a.obs), "obs_sha256": sha(a.obs), "channels": a.channels, "camera": cams.source,
                   "camera_json_sha256": sha(ROOT / "params/camera.json") if a.camera == "project" else sha(a.camera.split(":", 1)[1]),
                   "geometry": dict(mb.GEOM_DEFAULT, **gj), "h_fit_s": H_FIT},
        "model": td.MODEL_ID, "fit_window": {c: f"ANCHOR t <= {Ta[c]} s" for c in a.channels}, "t_anchor_s": a.t_anchor, "t_anchor_per_channel_s": Ta, "free": R["names"],
        "initial_guess": init, "initial_guess_info": ig, "model_selection": sel, "production_fit": prod_label, "params": P,
        "sd_scaled": R["sd_scaled"], "sd_formal": R["sd_formal"], "reduced_chi2": R["reduced_chi2"], "cond_J": R["cond_J"],
        "correlation": R["corr"], "bootstrap": boot, "flags": flags_for(R, boot), "scale_mode": a.scale_mode, "free_tilt": a.free_tilt,
        "derived": {"model": {k: v for k, v in info.items() if k != "g"}, "modes": der, "camera_units_per_m": info["lam"],
                    "provenance": "ell/R/t/kappa: model-conditioned (scale: %s; brass density assumed); om_t, mu, ICs fitted" % a.scale_mode},
        "metrics": metrics, "divergence_map": divergence,
        "initial_state_gravity_frame": {"X0": list(M.X0), "V0": list(M.V0), "q0": list(M.q0), "w0": list(M.w0)},
    }
    (out / "fit_projected.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    predfull = {c: np.full((241,) + pred[c].shape[1:2], np.nan) for c in a.channels}
    for c in a.channels:
        predfull[c][selfr] = pred[c][:, :, 0]
    np.savez(out / "fit_series.npz", frames=frames, t=data["t"], tg=tg, t_anchor=a.t_anchor, Ta_json=json.dumps(Ta), mu=P["mu"],
             **{f"y_{c}": data["y"][c] for c in a.channels}, **{f"s_{c}": data["s"][c] for c in a.channels},
             **{f"pred_{c}": predfull[c] for c in a.channels}, Y=Yfull, lam=info["lam"], R_m=M.R, hc_m=M.h_c)
    summary = {"production": prod_label, "params": P, "sd_scaled": R["sd_scaled"],
               "boot_sd": {k: v["sd"] for k, v in (boot or {}).items() if isinstance(v, dict) and "sd" in v},
               "flags": result["flags"], "metrics": metrics,
               "modes": {k: der[k] for k in ("T_torsion_s", "T_lateral_s", "Q_torsion")},
               "cam_units_per_m": info["lam"], "selection": {k: sel[k] for k in ("MQ", "M1")}, "damping_identified": bool(ident)}
    if a.truth:
        T = json.loads(Path(a.truth).read_text())
        sdb = lambda k: max((boot or {}).get(k, {}).get("sd") or 0.0, R["sd_scaled"].get(k) or 0.0, 1e-30)
        summary["truth_comparison"] = {k: {"true": T[k], "fit": P[k], "sd": sdb(k), "z": (P[k] - T[k]) / sdb(k)}
                                       for k in R["names"] if k in T}
    (out / "fit_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
