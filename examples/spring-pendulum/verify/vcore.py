"""Shared verifier analysis helpers (venv; numpy + scipy). Never imports solver/."""
import math
import numpy as np
from vmodel import quat_to_R
from vio import pos, vel, quat

OUT_DT = 1.0 / 96.0


def state_from_cols(model, d, rows=slice(None)):
    E = pos(d)[rows]; vE = vel(d)[rows]
    if model.body != "rigid":
        return np.concatenate([E, vE], -1)
    q = quat(d)
    if q is None:
        raise ValueError("rigid model but no quaternion columns")
    q = q[rows]; q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w = np.stack([d["wx"], d["wy"], d["wz"]], -1)[rows]
    R = quat_to_R(q)
    r = np.einsum("...ij,j->...i", R, model.e_b - model.gb)
    G = E - r
    vG = vE - np.cross(w, r)
    wb = np.einsum("...ji,...j->...i", R, w)
    return np.concatenate([G, vG, q, wb], -1)


def _tableau():
    from ref_solver import gauss_tableau
    return gauss_tableau(4)


def local_propagate(model, Y, dt=OUT_DT, sub=8, tol=1e-16):
    """Batch GL4 propagation of every state in Y (N,n) over dt; also integrates P_diss.
    Returns (Y_next (N,n), dW (N,))."""
    A, b, c = _tableau()
    h = dt / sub
    y = np.concatenate([Y, np.zeros((len(Y), 1))], -1)

    def f(z):
        out = np.empty_like(z)
        out[..., :-1] = model.rhs(z[..., :-1])
        out[..., -1] = model.observables(z[..., :-1])["P_diss_W"]
        return out
    for _ in range(sub):
        K = np.broadcast_to(f(y), (4,) + y.shape).copy()
        for _it in range(80):
            Z = y[None] + h * np.einsum("ij,jkl->ikl", A, K)
            Kn = f(Z)
            dl = np.max(np.abs(Kn - K)) * h
            K = Kn
            if dl < tol * (1 + np.max(np.abs(y))):
                break
        y = y + h * np.einsum("j,jkl->kl", b, K)
    return y[:, :-1], y[:, -1]


def grid_index(t, dt=OUT_DT, tol=1e-9):
    """indices k with t == k*dt (to tol) -> (mask, k)."""
    k = t / dt
    kr = np.round(k)
    return np.abs(k - kr) < tol, kr.astype(int)


def herm3(t0, t1, x0, v0, x1, v1, tt):
    h = t1 - t0
    s = (tt - t0) / h
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1; h10 = s ** 3 - 2 * s ** 2 + s; h01 = -2 * s ** 3 + 3 * s ** 2; h11 = s ** 3 - s ** 2
    return h00 * x0 + h10 * h * v0 + h01 * x1 + h11 * h * v1


def interp_hermite(t, X, V, tq):
    """cubic Hermite interpolation of X (N,3) with V at query times tq (clipped to range)."""
    tq = np.clip(tq, t[0], t[-1])
    i = np.clip(np.searchsorted(t, tq) - 1, 0, len(t) - 2)
    return herm3(t[i][:, None], t[i + 1][:, None], X[i], V[i], X[i + 1], V[i + 1], tq[:, None])


def time_scale_estimate(t, Xp, tr, Xr, Vr, span=0.02):
    """s minimising RMS |Xp(t) - Xr(s t)|; golden search on [1-span, 1+span] after a coarse scan."""
    def cost(s):
        return float(np.sqrt(np.mean(np.sum((Xp - interp_hermite(tr, Xr, Vr, s * t)) ** 2, -1))))
    ss = np.linspace(1 - span, 1 + span, 801)
    cs = [cost(s) for s in ss]
    j = int(np.argmin(cs))
    a, b = ss[max(j - 1, 0)], ss[min(j + 1, len(ss) - 1)]
    g = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        c1, c2 = b - g * (b - a), a + g * (b - a)
        if cost(c1) < cost(c2):
            b = c2
        else:
            a = c1
    s = 0.5 * (a + b)
    return s, cost(s), cost(1.0)
