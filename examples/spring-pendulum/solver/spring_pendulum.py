"""Production solver for simulation 02 (vertical spring).  Owner: physics+dynamics (role B).

Model id (physics/FORMULATION.md): elastic_pendulum_hinged_body_v1
  * fixed anchor A = world origin (top hook bearing on the crossbar; ideal ball joint, moment free);
  * massless-or-light linear spring A -> H (H = lower-hook / ring contact point), axial only
    (two-force member: both ends are moment-free hooks, see FORMULATION s2);
    spring mass m_s included by the Rayleigh (linear velocity profile) approximation:
        T_s = 1/2 (m_s/3) |dH/dt|^2,   V_s,grav = m_s g H_z / 2;
  * rigid axisymmetric weight hinged at H (ball joint, ring on hook), centre of mass
    X = H + b n, n = unit body axis pointing from the hinge to the CoM; transverse inertia I_perp
    about the CoM; spin about the own axis omega_3 = 0 (conserved exactly for an axisymmetric body
    hinged on its axis with zero initial spin; orientation integrated as an auxiliary quaternion).
  * body = "point": b = 0, I_perp = 0 (3-DOF elastic pendulum; orientation = spring axis, display only).

Coordinates q = (X, n) with the constraint |n| = 1 (Lagrange multiplier, index-reduced):
    T = 1/2 m |X'|^2 + 1/2 I_perp |n'|^2 + 1/2 mu |X' - b n'|^2,      mu = m_s / 3
    V = 1/2 k (|s| - L0)^2 + m g X_z + m_s g s_z / 2,                  s = H = X - b n
    D = 1/2 c_s (s_hat . s')^2 + 1/2 c_x |X'|^2 + 1/2 c_r |n'|^2       (Rayleigh, all c >= 0)
  =>  [[a, c],[c, d]] (x) I3 . [X''; n''] = [F_X; F_n + lambda n]
      a = m + mu,  c = -mu b,  d = I_perp + mu b^2
      F_X = -T s_hat - (m + m_s/2) g e_z - c_s s_r s_hat - c_x X'
      F_n = +b T s_hat + b (m_s/2) g e_z + b c_s s_r s_hat - c_r n'
      T = k (|s| - L0) (spring tension, >0 stretched), s_r = s_hat . s'
      lambda chosen so that n . n'' = -|n'|^2 (keeps |n| = 1, n . n' = 0 analytically).
  P_diss = c_s s_r^2 + c_x |X'|^2 + c_r |n'|^2 >= 0, and dE/dt = -P_diss exactly.

Integrator: classical RK4, FIXED step dividing each 1/96 s output interval exactly (default
h = 1/3072 s), output samples are integration nodes, deterministic and bitwise repeatable.
v2 (verifier P9 finding): after every step (n, n', q) are projected exactly onto |n| = 1, n.n' = 0, |q| = 1
(O(h^5) per step, order 4 kept); guards raise on non-finite state, |n|-1 > 1e-12, per-step drift > 1e-8, and on
compressive spring force (tension < 0; a hooked spring cannot push) unless on_compression="record".
The RHS is smooth (no non-smooth damping), so 4th order holds everywhere.  The dissipated work W is
integrated as an extra state with the same RK4.  The integrator is vectorised over a batch axis
(used by the fit to evaluate many parameter vectors at once; production uses batch = 1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, replace

import numpy as np

SOLVER_ID = "rk4-fixed-step-elastic-pendulum-v2-projected"
MODEL_ID = "elastic_pendulum_hinged_body_v1"
G_STD = 9.80665
DEFAULT_H = 1.0 / 3072.0


@dataclass(frozen=True)
class Model:
    k: float            # spring rate [N/m]
    L0: float           # effective natural length A->H incl. rigid hooks and coil preload [m]
    m: float            # weight mass [kg]
    m_s: float          # spring mass [kg] (Rayleigh)
    b: float            # hinge H -> weight CoM [m]
    I_perp: float       # weight transverse moment of inertia about its CoM [kg m^2]
    c_s: float          # axial dashpot along the spring [N s/m]
    c_x: float          # linear drag on the CoM velocity [N s/m]
    c_r: float          # rotational drag on n' [N m s]
    g: float
    X0: tuple           # CoM position [m]
    V0: tuple           # CoM velocity [m/s]
    n0: tuple           # body axis (unit, hinge -> CoM)
    u0: tuple           # n' [1/s]
    body: str = "rigid"  # "rigid" | "point"

    @property
    def mu(self):
        return self.m_s / 3.0

    def as_dict(self):
        d = asdict(self)
        d.update(self.derived())
        return d

    def derived(self):
        """Equilibrium and linear-mode quantities (FORMULATION s3)."""
        g = self.g
        W = (self.m + 0.5 * self.m_s) * g                  # static tension at H
        delta = W / self.k                                  # static stretch
        ell = self.L0 + delta                               # equilibrium |A H|
        om_ax = math.sqrt(self.k / (self.m + self.mu))
        om_sw = lateral_modes(self)
        return {"static_tension_N": W, "static_stretch_m": delta, "ell_eq_m": ell,
                "H_eq": [0.0, 0.0, -ell], "X_eq": [0.0, 0.0, -ell - self.b],
                "omega_axial_rad_s": om_ax, "f_axial_Hz": om_ax / (2 * math.pi),
                "omega_lateral_rad_s": om_sw, "f_lateral_Hz": [w / (2 * math.pi) for w in om_sw],
                "ratio_axial_over_swing": om_ax / om_sw[0]}


def lateral_modes(M: Model):
    """Small-oscillation lateral (in-plane) angular frequencies about equilibrium.
    rigid: DOFs (phi = spring angle, psi = body angle),
       Mass = [[(m+mu) l^2, m l b], [m l b, m b^2 + I_perp]],  K = diag((m + m_s/2) g l, m g b)
    point: omega^2 = (m + m_s/2) g / ((m + mu) l)."""
    W = (M.m + 0.5 * M.m_s) * M.g
    ell = M.L0 + W / M.k
    if M.body == "point" or (M.b == 0.0 and M.I_perp == 0.0):
        return [math.sqrt(W * ell / ((M.m + M.mu) * ell * ell))]
    Mm = np.array([[(M.m + M.mu) * ell * ell, M.m * ell * M.b], [M.m * ell * M.b, M.m * M.b ** 2 + M.I_perp]])
    K = np.diag([W * ell, M.m * M.g * M.b])
    from scipy.linalg import eigh
    w2 = eigh(K, Mm, eigvals_only=True)
    return [float(math.sqrt(x)) for x in sorted(w2)]


def lateral_mode_shapes(M: Model):
    W = (M.m + 0.5 * M.m_s) * M.g
    ell = M.L0 + W / M.k
    Mm = np.array([[(M.m + M.mu) * ell * ell, M.m * ell * M.b], [M.m * ell * M.b, M.m * M.b ** 2 + M.I_perp]])
    K = np.diag([W * ell, M.m * M.g * M.b])
    from scipy.linalg import eigh
    w2, V = eigh(K, Mm)
    return np.sqrt(w2), V, ell


# ------------------------------------------------------------------------------------------------
# batched state: dict of arrays with shape (B,) per component
#   X (3,B) V (3,B) n (3,B) u (3,B) q (4,B) W (B,)
# packed as one array Y of shape (17, B): 0:3 X, 3:6 V, 6:9 n, 9:12 u, 12:16 q (w,x,y,z), 16 W
# ------------------------------------------------------------------------------------------------
NY = 17


def _params_arrays(Ms):
    f = lambda name: np.array([getattr(M, name) for M in Ms], float)
    P = {k: f(k) for k in ("k", "L0", "m", "m_s", "b", "I_perp", "c_s", "c_x", "c_r", "g")}
    P["mu"] = P["m_s"] / 3.0
    P["point"] = np.array([M.body == "point" for M in Ms])
    if np.any(P["point"]) and not np.all(P["point"]):
        raise ValueError("batch must not mix point and rigid bodies")
    P["is_point"] = bool(P["point"][0])
    for nm in ("c_s", "c_x", "c_r"):
        if np.any(P[nm] < 0):
            raise ValueError(f"{nm} < 0: negative damping (energy injection) is not allowed")
    if np.any(P["k"] <= 0) or np.any(P["m"] <= 0) or np.any(P["L0"] <= 0):
        raise ValueError("non-positive k, m or L0")
    a = P["m"] + P["mu"]; c = -P["mu"] * P["b"]; d = P["I_perp"] + P["mu"] * P["b"] ** 2
    P["a"], P["c"], P["d"], P["det"] = a, c, d, a * d - c * c
    return P


def initial_state(Ms):
    B = len(Ms)
    Y = np.zeros((NY, B))
    for j, M in enumerate(Ms):
        n = np.array(M.n0, float)
        u = np.array(M.u0, float)
        if M.body == "point":
            s = np.array(M.X0, float); n = s / np.linalg.norm(s); u = np.zeros(3)
        if abs(np.linalg.norm(n) - 1) > 1e-12:
            raise ValueError("n0 must be a unit vector")
        if abs(n @ u) > 1e-12:
            raise ValueError("u0 must be orthogonal to n0")
        Y[0:3, j] = M.X0; Y[3:6, j] = M.V0; Y[6:9, j] = n; Y[9:12, j] = u
        Y[12:16, j] = quat_min_rotation(np.array([0.0, 0.0, -1.0]), n)
    return Y


def quat_min_rotation(a, b):
    """unit quaternion (w,x,y,z) of the minimal rotation taking unit a to unit b (zero twist)."""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    w = 1.0 + a @ b
    v = np.cross(a, b)
    q = np.array([w, *v])
    return q / np.linalg.norm(q)


def rhs(Y, P):
    X, V, n, u, q = Y[0:3], Y[3:6], Y[6:9], Y[9:12], Y[12:16]
    b, k, L0, g, m, m_s = P["b"], P["k"], P["L0"], P["g"], P["m"], P["m_s"]
    s = X - b * n
    L = np.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2)
    sh = s / L
    sd = V - b * u
    sr = sh[0] * sd[0] + sh[1] * sd[1] + sh[2] * sd[2]
    T = k * (L - L0)
    ax = T + P["c_s"] * sr
    FX = -ax * sh - P["c_x"] * V
    FX[2] -= (m + 0.5 * m_s) * g
    dY = np.empty_like(Y)
    dY[0:3] = V
    Pd = P["c_s"] * sr * sr + P["c_x"] * (V[0] ** 2 + V[1] ** 2 + V[2] ** 2)
    if P["is_point"]:
        dY[3:6] = FX / P["a"]
        dY[6:16] = 0.0
    else:
        Fn = b * ax * sh - P["c_r"] * u
        Fn[2] += b * 0.5 * m_s * g
        a_, c_, det = P["a"], P["c"], P["det"]
        uu = u[0] ** 2 + u[1] ** 2 + u[2] ** 2
        nFX = n[0] * FX[0] + n[1] * FX[1] + n[2] * FX[2]
        nFn = n[0] * Fn[0] + n[1] * Fn[1] + n[2] * Fn[2]
        nn = n[0] ** 2 + n[1] ** 2 + n[2] ** 2
        lam = (-uu * det + c_ * nFX - a_ * nFn) / (a_ * nn)       # exact n.n'' = -|n'|^2 for any |n| (no drift source)
        G = Fn + lam * n
        dY[3:6] = (P["d"] * FX - c_ * G) / det
        dY[6:9] = u
        dY[9:12] = (-c_ * FX + a_ * G) / det
        # auxiliary orientation: world angular velocity w = n x n' (omega_3 = 0); q' = 1/2 (0,w) * q
        w = np.cross(n, u, axis=0)
        qw, qx, qy, qz = q
        dY[12] = 0.5 * (-w[0] * qx - w[1] * qy - w[2] * qz)
        dY[13] = 0.5 * (w[0] * qw + w[1] * qz - w[2] * qy)
        dY[14] = 0.5 * (-w[0] * qz + w[1] * qw + w[2] * qx)
        dY[15] = 0.5 * (w[0] * qy - w[1] * qx + w[2] * qw)
        Pd = Pd + P["c_r"] * uu
    dY[16] = Pd
    return dY


class SolverGuardError(RuntimeError):
    """Raised when the state is non-finite or the |n| = 1 invariant is violated beyond 1e-12 after projection."""


class SpringCompressionError(RuntimeError):
    """Raised when the spring would push (tension < 0): a hooked extension spring cannot carry compression
    (the model would need slack/contact, which is out of scope).  P7."""


NORM_TOL = 1e-12          # post-projection guard on | |n| - 1 | and |n . n'|
STEP_DRIFT_TOL = 1e-8     # pre-projection per-step drift guard (the step is too large if exceeded)


def _project(Y):
    """Exact projection onto the constraint manifold after each RK4 step (v2, verifier finding P9 / CHANGELOG
    2026-10-26T01:53:54Z): n <- n/|n|, n' <- n' - (n.n') n, q <- q/|q|.  The RK4 drift removed per step is O(h^5),
    so the 4th order is retained; returns the largest pre-projection | |n| - 1 |."""
    n = Y[6:9]
    nr = np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
    drift = np.max(np.abs(nr - 1.0))
    n = n / nr
    u = Y[9:12]
    u = u - (n[0] * u[0] + n[1] * u[1] + n[2] * u[2]) * n
    q = Y[12:16]
    q = q / np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    Y[6:9] = n; Y[9:12] = u; Y[12:16] = q
    return float(drift)


def integrate(Ms, t_out, h=DEFAULT_H, on_compression="raise", stats=None):
    """Ms: Model or list of Models (batch).  Returns Y of shape (len(t_out), 17, B).
    v2: exact projection of (n, n', q) after every step; guards: non-finite state, post-projection | |n|-1 | > 1e-12,
    per-step pre-projection drift > 1e-8 (step too large) -> SolverGuardError; tension < 0 at any step node ->
    SpringCompressionError (on_compression="raise", default) or recorded only (on_compression="record").
    stats (dict, optional) receives max_step_drift, min_tension_N (per batch), first_compression_t."""
    single = isinstance(Ms, Model)
    Ms = [Ms] if single else list(Ms)
    t_out = np.asarray(t_out, float)
    if t_out[0] != 0.0 or np.any(np.diff(t_out) <= 0):
        raise ValueError("t_out must start at 0 and increase strictly")
    P = _params_arrays(Ms)
    Y = initial_state(Ms)
    out = np.empty((len(t_out), NY, len(Ms)))
    out[0] = Y
    max_drift = 0.0
    tmin = np.full(len(Ms), np.inf); first_c = None
    rigid = not P["is_point"]

    def tension(Yc):
        s = Yc[0:3] - P["b"] * Yc[6:9]
        return P["k"] * (np.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2) - P["L0"])

    T0 = tension(Y); tmin = np.minimum(tmin, T0)
    if np.any(T0 < 0) and on_compression == "raise":
        raise SpringCompressionError(f"spring compressed at t = 0 (tension {T0.min():.4g} N)")
    t_now = 0.0
    for i in range(1, len(t_out)):
        dt = t_out[i] - t_out[i - 1]
        nsub = max(1, int(math.ceil(dt / h - 1e-9)))
        hs = dt / nsub
        for j in range(nsub):
            k1 = rhs(Y, P)
            k2 = rhs(Y + 0.5 * hs * k1, P)
            k3 = rhs(Y + 0.5 * hs * k2, P)
            k4 = rhs(Y + hs * k3, P)
            Y = Y + (hs / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t_now = t_out[i - 1] + (j + 1) * hs
            if not np.all(np.isfinite(Y)):
                raise SolverGuardError(f"non-finite state at t = {t_now:.6f} s")
            if rigid:
                d = _project(Y)
                max_drift = max(max_drift, d)
                if d > STEP_DRIFT_TOL:
                    raise SolverGuardError(f"per-step |n| drift {d:.3g} > {STEP_DRIFT_TOL} at t = {t_now:.6f} s (step too large)")
                n = Y[6:9]; u = Y[9:12]
                e1 = np.max(np.abs(np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2) - 1.0))
                e2 = np.max(np.abs(n[0] * u[0] + n[1] * u[1] + n[2] * u[2]))
                if e1 > NORM_TOL or e2 > NORM_TOL * max(1.0, float(np.max(np.abs(u)))):
                    raise SolverGuardError(f"| |n|-1 | = {e1:.3g}, n.n' = {e2:.3g} after projection at t = {t_now:.6f} s")
            Tn = tension(Y)
            tmin = np.minimum(tmin, Tn)
            if np.any(Tn < 0):
                if first_c is None:
                    first_c = t_now
                if on_compression == "raise":
                    raise SpringCompressionError(f"spring compressed (tension {Tn.min():.4g} N < 0) at t = {t_now:.6f} s: "
                                                 "a hooked spring cannot push; re-equilibrate the initial state (model_build.reequilibrate)")
        out[i] = Y
    if stats is not None:
        stats.update(max_step_drift=max_drift, min_tension_N=tmin.tolist(), first_compression_t=first_c)
    return out[..., 0] if single else out


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def euler_xyz_from_matrix(R):
    """Blender 'XYZ' Euler (R = Rz(c) Ry(b) Rx(a))."""
    b = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    a = math.atan2(R[2, 1], R[2, 2])
    c = math.atan2(R[1, 0], R[0, 0])
    return a, b, c


def euler_rates(e, w):
    """dE/dt for Blender XYZ Euler e=(a,b,c), world angular velocity w.
    w = a' Rz Ry ex + b' Rz ey + c' ez."""
    a, b, c = e
    cz, sz, cy, sy = math.cos(c), math.sin(c), math.cos(b), math.sin(b)
    col_a = np.array([cz * cy, sz * cy, -sy])
    col_b = np.array([-sz, cz, 0.0])
    col_c = np.array([0.0, 0.0, 1.0])
    A = np.column_stack([col_a, col_b, col_c])
    return np.linalg.solve(A, w)


def derived(M: Model, Y):
    """Y: (N, 17) single trajectory.  Returns dict of per-sample arrays."""
    P = _params_arrays([M])
    Pn = {k: (v[0] if isinstance(v, np.ndarray) else v) for k, v in P.items()}
    X, V, n, u, q, W = Y[:, 0:3], Y[:, 3:6], Y[:, 6:9], Y[:, 9:12], Y[:, 12:16], Y[:, 16]
    b = M.b
    s = X - b * n
    L = np.linalg.norm(s, axis=1)
    sh = s / L[:, None]
    sd = V - b * u
    sr = np.sum(sh * sd, axis=1)
    T = M.k * (L - M.L0)
    if M.body == "point":
        n = sh
        u = np.zeros_like(n)   # display orientation only (no rotational dynamics in the point model)
    acc = np.array([rhs(Y[i][:, None], P)[:, 0] for i in range(len(Y))])
    KE = 0.5 * M.m * np.sum(V * V, 1) + 0.5 * M.I_perp * np.sum(u * u, 1) + 0.5 * M.mu * np.sum(sd * sd, 1)
    d0 = M.derived()
    ell = d0["ell_eq_m"]
    # PE relative to static equilibrium (s = (0,0,-ell), X = (0,0,-ell-b))
    PE = 0.5 * M.k * (L - M.L0) ** 2 + M.m * M.g * X[:, 2] + 0.5 * M.m_s * M.g * s[:, 2]
    PE0 = 0.5 * M.k * (ell - M.L0) ** 2 + M.m * M.g * (-ell - b) + 0.5 * M.m_s * M.g * (-ell)
    PE = PE - PE0
    P_diss = M.c_s * sr ** 2 + M.c_x * np.sum(V * V, 1) + (0.0 if M.body == "point" else M.c_r * np.sum(u * u, 1))
    if M.body == "point":
        Rm = [quat_to_matrix(quat_min_rotation(np.array([0, 0, -1.0]), nn)) for nn in n]
        qq = np.array([quat_min_rotation(np.array([0, 0, -1.0]), nn) for nn in n])
        wv = np.zeros_like(n)
    else:
        qq = q / np.linalg.norm(q, axis=1)[:, None]
        Rm = [quat_to_matrix(x) for x in qq]
        wv = np.cross(n, u)
    eul = np.array([euler_xyz_from_matrix(R) for R in Rm])
    eul = np.unwrap(eul, axis=0)
    eul_rate = np.array([euler_rates(e, w) for e, w in zip(eul, wv)])
    n_from_q = np.array([R @ np.array([0, 0, -1.0]) for R in Rm])
    return dict(X=X, V=V, A=acc[:, 3:6], n=n, u=u, H=s, Hdot=sd, L=L, sr=sr, tension=T, q=qq, euler=eul,
                euler_rate=eul_rate, omega=wv, KE=KE, PE=PE, E=KE + PE, P_diss=P_diss, W=W,
                n_q_consistency=np.max(np.abs(n_from_q - n)), norm_n_err=np.max(np.abs(np.linalg.norm(n, axis=1) - 1)))


def simulate(M: Model, t_out, h=DEFAULT_H, on_compression="raise"):
    st = {}
    Y = integrate(M, t_out, h, on_compression=on_compression, stats=st)
    d = derived(M, Y)
    d["integrator_stats"] = st
    d["t"] = np.asarray(t_out, float)
    d["Y"] = Y
    d["model"] = M
    return d


# ---- params/physics.json --------------------------------------------------------------------------
def _val(e, default=None):
    if isinstance(e, dict):
        v = e.get("value", default)
        return default if v is None else v
    return default if e is None else e


def model_from_physics_json(p: dict) -> Model:
    if p.get("model_id") != MODEL_ID:
        raise ValueError(f"model_id {p.get('model_id')!r} != {MODEL_ID}")
    P = p["parameters"]
    ic = p["initial_state"]
    v = lambda k: float(_val(P[k]))
    damp = P["damping"]
    M = Model(k=v("k"), L0=v("L0"), m=v("m"), m_s=v("m_s"), b=v("b"), I_perp=v("I_perp"),
              c_s=float(_val(damp["c_s"])), c_x=float(_val(damp["c_x"])), c_r=float(_val(damp["c_r"])),
              g=v("g"), X0=tuple(_val(ic["X0"])), V0=tuple(_val(ic["V0"])), n0=tuple(_val(ic["n0"])),
              u0=tuple(_val(ic["u0"])), body=p.get("body", "rigid"))
    return M


def with_(M: Model, **kw) -> Model:
    return replace(M, **kw)
