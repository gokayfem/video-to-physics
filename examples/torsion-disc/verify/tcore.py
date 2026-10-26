"""Shared verifier analysis helpers (venv; numpy + scipy). Never imports solver/."""
import math
import numpy as np
from tmodel import quat_to_R, rmin_R, R_to_quat, EZ
from tio import pos, vel, quat

OUT_DT = 1.0 / 96.0


def unit_quats(d):
    q = quat(d)
    return None if q is None else q / np.linalg.norm(q, axis=-1, keepdims=True)


def omega_world(d):
    return np.stack([d["wx"], d["wy"], d["wz"]], -1)


def attach_pos(model, d):
    """P from columns if present, else C + R p_b."""
    if all(k in d for k in ("px_m", "py_m", "pz_m")):
        return np.stack([d["px_m"], d["py_m"], d["pz_m"]], -1)
    R = quat_to_R(unit_quats(d))
    return pos(d) + np.einsum("nij,j->ni", R, model.p_b)


def state_from_cols(model, d, rows=slice(None)):
    """verifier state vectors from (canonical-named) production columns: C, v_C, q (disc), w (world)."""
    q = unit_quats(d)[rows]
    w = omega_world(d)[rows]
    R = quat_to_R(q)
    wb = np.einsum("nji,nj->ni", R, w)
    if model.mode == "rigid_pivot":
        return np.concatenate([q, wb], -1)
    C = pos(d)[rows]; vC = vel(d)[rows]
    r = np.einsum("nij,j->ni", R, model.p_b)
    P = C + r
    vP = vC + np.cross(w, r)
    u = (model.A - P); u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    udot = -vP / model.L
    udot = udot - np.sum(udot * u, -1, keepdims=True) * u
    out = []
    for i in range(len(q)):
        Rw = rmin_R(EZ, u[i])
        ww = np.cross(u[i], udot[i])
        wwb = Rw.T @ ww
        out.append(np.r_[R_to_quat(Rw), wwb[:2], q[i], wb[i]])
    return np.array(out)


def _tableau():
    from ref_solver import gauss_tableau
    return gauss_tableau(4)


def local_propagate(model, Y, dt=OUT_DT, sub=4, tol=1e-15):
    """Batch GL4 propagation of every state in Y (N,n) over dt, with W (dissipation) and Z (int tau_z) augmented.
    Returns (Y_next (N,n), dW (N,), dZ (N,))."""
    A, b, c = _tableau()
    h = dt / sub
    y = np.concatenate([Y, np.zeros((len(Y), 2))], -1)

    def f(z):
        out = np.empty_like(z)
        out[..., :-2], out[..., -2], out[..., -1] = model.rhs(z[..., :-2], aug=True)
        return out
    for _ in range(sub):
        K = np.broadcast_to(f(y), (4,) + y.shape).copy()
        for _it in range(80):
            Kn = f(y[None] + h * np.einsum("ij,jkl->ikl", A, K))
            dl = np.max(np.abs(Kn - K)) * h
            K = Kn
            if dl < tol * (1 + np.max(np.abs(y))):
                break
        y = y + h * np.einsum("j,jkl->kl", b, K)
    return y[:, :-2], y[:, -2], y[:, -1]


def grid_index(t, dt=OUT_DT, tol=1e-9):
    k = t / dt
    kr = np.round(k)
    return np.abs(k - kr) < tol, kr.astype(int)


def herm3(t0, t1, x0, v0, x1, v1, tt):
    h = t1 - t0
    s = (tt - t0) / h
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1; h10 = s ** 3 - 2 * s ** 2 + s; h01 = -2 * s ** 3 + 3 * s ** 2; h11 = s ** 3 - s ** 2
    return h00 * x0 + h10 * h * v0 + h01 * x1 + h11 * h * v1


def interp_hermite(t, X, V, tq):
    tq = np.clip(tq, t[0], t[-1])
    i = np.clip(np.searchsorted(t, tq) - 1, 0, len(t) - 2)
    return herm3(t[i][:, None], t[i + 1][:, None], X[i], V[i], X[i + 1], V[i + 1], tq[:, None])


def time_scale_estimate(t, Xp, tr, Xr, Vr, span=0.02):
    """s minimising RMS |Xp(t) - Xr(s t)|; coarse scan then golden search."""
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
