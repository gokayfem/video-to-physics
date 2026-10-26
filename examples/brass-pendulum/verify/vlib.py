"""Verifier helpers: trajectory I/O, Hermite interpolation, crossings, peaks, event-aware quadrature."""
import csv, json, math, os, sys
import numpy as np
from scipy.optimize import brentq

REQUIRED = ["t_s", "theta_rad", "omega_rad_s", "alpha_rad_s2", "x_m", "z_m", "KE_J", "PE_J", "E_J",
            "tension_N", "P_diss_W"]


def read_traj(path, required=REQUIRED):
    with open(path) as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [[float(x) for x in row] for row in r if row]
    A = np.array(rows, dtype=float)
    d = {h: A[:, i] for i, h in enumerate(header)}
    missing = [c for c in required if c not in d]
    return d, header, missing


def write_traj(path, cols, header):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        n = len(cols[header[0]])
        for i in range(n):
            f.write(",".join(repr(float(cols[h][i])) for h in header) + "\n")


# ---- quintic Hermite on [0,1] in s; y(s) with y, y', y'' at ends (derivatives wrt t, scaled by h)
_M5 = None


def _m5():
    global _M5
    if _M5 is None:
        rows = []
        def row(s, d):
            r = []
            for k in range(6):
                if d == 0:
                    r.append(s ** k)
                elif d == 1:
                    r.append(k * s ** (k - 1) if k >= 1 else 0.0)
                else:
                    r.append(k * (k - 1) * s ** (k - 2) if k >= 2 else 0.0)
            return r
        M = np.array([row(0, 0), row(0, 1), row(0, 2), row(1, 0), row(1, 1), row(1, 2)], float)
        _M5 = np.linalg.inv(M)
    return _M5


def herm5(y0, d0, dd0, y1, d1, dd1, h):
    """coefficients c_k (in s=(t-t0)/h) of the quintic Hermite."""
    return _m5() @ np.array([y0, d0 * h, dd0 * h * h, y1, d1 * h, dd1 * h * h])


def herm3(y0, d0, y1, d1, h):
    return np.array([y0, d0 * h, -3 * y0 - 2 * d0 * h + 3 * y1 - d1 * h, 2 * y0 + d0 * h - 2 * y1 + d1 * h])


def pval(c, s):
    r = 0.0
    for x in c[::-1]:
        r = r * s + x
    return r


def up_crossings(t, th, om, al):
    """Times of upward zero crossings of theta located with quintic Hermite (theta, omega, alpha)."""
    out = []
    for i in range(len(t) - 1):
        if th[i] < 0.0 <= th[i + 1] or (th[i] == 0.0 and om[i] > 0 and i > 0):
            if th[i] == 0.0:
                out.append(t[i]); continue
            h = t[i + 1] - t[i]
            c = herm5(th[i], om[i], al[i], th[i + 1], om[i + 1], al[i + 1], h)
            s = brentq(lambda s: pval(c, s), 0.0, 1.0, xtol=1e-16, rtol=1e-15)
            out.append(t[i] + s * h)
    return np.array(out)


def extrema(t, th, om, al, jerk=None):
    """Turning points (omega = 0) -> (time, theta). omega interpolated by cubic Hermite (omega, alpha)
    or quintic if jerk is given; theta by quintic Hermite."""
    out = []
    for i in range(len(t) - 1):
        if om[i] == 0.0 and i == 0:
            out.append((t[0], th[0])); continue
        if (om[i] > 0) != (om[i + 1] > 0) and om[i + 1] != 0.0 or (om[i + 1] == 0.0 and om[i] != 0.0):
            h = t[i + 1] - t[i]
            if jerk is not None:
                co = herm5(om[i], al[i], jerk[i], om[i + 1], al[i + 1], jerk[i + 1], h)
            else:
                co = herm3(om[i], al[i], om[i + 1], al[i + 1], h)
            fa, fb = pval(co, 0.0), pval(co, 1.0)
            if fa * fb > 0:
                continue
            s = brentq(lambda s: pval(co, s), 0.0, 1.0, xtol=1e-16, rtol=1e-15) if fb != 0 else 1.0
            ct = herm5(th[i], om[i], al[i], th[i + 1], om[i + 1], al[i + 1], h)
            out.append((t[i] + s * h, pval(ct, s)))
    return out


_GLX, _GLW = np.polynomial.legendre.leggauss(10)


def cumulative_dissipation(t, om, al, jerk, P_of_omega):
    """W(t_n) = int_0^{t_n} P(omega(t)) dt with omega from quintic Hermite (omega, alpha, jerk) on each
    interval, split at omega roots (so |omega| kinks are integrated exactly by Gauss-Legendre)."""
    W = np.zeros(len(t))
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        c = herm5(om[i], al[i], jerk[i], om[i + 1], al[i + 1], jerk[i + 1], h)
        # roots of omega in (0,1): sample and bracket
        ss = np.linspace(0, 1, 9)
        vals = [pval(c, s) for s in ss]
        cuts = [0.0]
        for j in range(8):
            if vals[j] * vals[j + 1] < 0:
                cuts.append(brentq(lambda s: pval(c, s), ss[j], ss[j + 1], xtol=1e-16, rtol=1e-15))
        cuts.append(1.0)
        acc = 0.0
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b <= a:
                continue
            xs = 0.5 * (b - a) * _GLX + 0.5 * (a + b)
            acc += 0.5 * (b - a) * sum(w * P_of_omega(pval(c, x)) for w, x in zip(_GLW, xs))
        W[i + 1] = W[i] + acc * h
    return W


def simpson_cumulative(t, y):
    """Composite Simpson at even indices (uniform grid). Returns (idx, W)."""
    h = t[1] - t[0]
    n = (len(t) - 1) // 2 * 2
    idx = np.arange(0, n + 1, 2)
    W = np.zeros(len(idx))
    for k in range(1, len(idx)):
        i = idx[k - 1]
        W[k] = W[k - 1] + h / 3 * (y[i] + 4 * y[i + 1] + y[i + 2])
    return idx, W


def boole_cumulative(t, y):
    h = t[1] - t[0]
    n = (len(t) - 1) // 4 * 4
    idx = np.arange(0, n + 1, 4)
    W = np.zeros(len(idx))
    for k in range(1, len(idx)):
        i = idx[k - 1]
        W[k] = W[k - 1] + 2 * h / 45 * (7 * y[i] + 32 * y[i + 1] + 12 * y[i + 2] + 32 * y[i + 3] + 7 * y[i + 4])
    return idx, W


def status(ok):
    return "PASS" if ok else "FAIL"


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    return o
