"""Production solver for simulation 03 (torsion disc).  Owner: physics+dynamics (role B).

Model id (physics/FORMULATION.md): disc_on_wire_rigid6_v1
  * clamp A = world origin (wire leaves the black clamp here; fixed);
  * thin steel wire A -> B, inextensible string of length ell for translation (|B - A| = ell, tension T >= 0),
    with torsional stiffness kappa (linear in the twist psi) and end bending boundary-layer springs
    k_A (clamp) and k_B (disc attachment), k = sqrt(E I_wire T0) (FORMULATION s3);
  * rigid uniform brass disc (radius R, thickness t), wire attached at the centre of its TOP face B;
    centre of mass X = B - (t/2) n_up, i.e. r = B - X = R(q) p, p = (0, 0, t/2) in body coordinates
    (body z = disc normal, up at rest; body x = zero-twist reference direction).

State (per batch column): X (3) CoM, V (3), q (4: w,x,y,z, body -> world), w (3: WORLD angular velocity), W (1:
dissipated work)  -> NY = 14.  Constraint |B| = ell enforced by the tension multiplier (index-1) and an exact projection
after every RK4 step (positions: X shifted along B so |B| = ell; velocities: radial component of dB/dt removed from V;
q normalised).  Projection removes O(h^5) per step, 4th order kept.

Potential  V = m g X_z + 1/2 kappa psi^2 + k_A (1 + s_z) + k_B (1 - s . n),
   s = B/|B| (wire direction A -> B), n = R(q) (0,0,-1) (disc DOWNWARD normal),
   psi = 2 atan2(q_z, q_w)  (swing-twist decomposition q = q_swing (x) q_twist, twist about BODY z; q_swing is the minimal
   rotation of body z; FORMULATION s3.2).  d psi / d theta_world = G = (w x + ... ) see grad_psi().
Rayleigh  D = 1/2 c_t psidot^2 + 1/2 c_x |V|^2 + 1/2 c_r |w_perp|^2,  psidot = G . w,  w_perp = w - (w . n) n.
dE/dt = -P_diss = -(c_t psidot^2 + c_x |V|^2 + c_r |w_perp|^2)  exactly (tension does no work: s . dB/dt = 0).

Integrator: classical RK4, FIXED step dividing each 1/96 s output interval exactly (default h = 1/3072 s); output samples
are integration nodes; deterministic, bitwise repeatable; vectorised over a batch axis (the fit evaluates many parameter
vectors at once; production uses batch = 1).  Guards (clip-02 lesson): non-finite state, | |q|-1 | and | |B|-ell | after
projection > 1e-12 (relative), per-step pre-projection drift > 1e-8, slack wire (T < 0), |psi| > 1.9 pi (branch limit of 2 atan2(q_z, q_w) on the continuously integrated q;
the decomposition itself is singular only for a 180 deg disc tilt, w = z = 0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, replace

import numpy as np

SOLVER_ID = "rk4-fixed-step-disc-on-wire-v1-projected"
MODEL_ID = "disc_on_wire_rigid6_v1"
G_STD = 9.80665
DEFAULT_H = 1.0 / 3072.0
NY = 14


@dataclass(frozen=True)
class Model:
    m: float            # disc mass [kg]
    R: float            # disc radius [m]
    t: float            # disc thickness [m]
    ell: float          # wire length A -> B (clamp exit -> top-face centre) [m]
    kappa: float        # torsion constant [N m / rad]
    k_A: float          # clamp-end bending stiffness [N m]
    k_B: float          # disc-end bending stiffness [N m]
    c_t: float          # torsional dashpot [N m s]
    c_x: float          # linear drag on the CoM velocity [N s / m]
    c_r: float          # tilt (rocking) rotational drag on w_perp [N m s]
    g: float
    X0: tuple           # CoM [m]
    V0: tuple           # CoM velocity [m/s]
    q0: tuple           # orientation (w, x, y, z), body -> world
    w0: tuple           # world angular velocity [rad/s]

    @property
    def I_z(self):
        return 0.5 * self.m * self.R ** 2

    @property
    def I_perp(self):
        return self.m * (3 * self.R ** 2 + self.t ** 2) / 12.0

    @property
    def h_c(self):
        return 0.5 * self.t

    def as_dict(self):
        d = asdict(self)
        d.update(I_z=self.I_z, I_perp=self.I_perp, h_c=self.h_c)
        d.update(self.derived())
        return d

    def derived(self):
        om_t = math.sqrt(self.kappa / self.I_z)
        wl, V = lateral_modes(self)
        gam_t = self.c_t / (2 * self.I_z)
        return {"omega_torsion_rad_s": om_t, "T_torsion_s": 2 * math.pi / om_t,
                "gamma_torsion_1_s": gam_t, "Q_torsion": (om_t / (2 * gam_t) if gam_t > 0 else None),
                "omega_lateral_rad_s": wl, "T_lateral_s": [2 * math.pi / w for w in wl],
                "lateral_mode_shapes_phi_theta": V.T.tolist(),
                "omega_point_pendulum_rad_s": math.sqrt(self.g / (self.ell + self.h_c)),
                "static_tension_N": self.m * self.g,
                "X_eq": [0.0, 0.0, -self.ell - self.h_c], "B_eq": [0.0, 0.0, -self.ell]}


def lateral_matrices(M: Model):
    """Linearised in-plane swing (phi = wire angle) + tilt (theta = disc normal angle) about equilibrium:
    Mass = [[m l^2, m l h], [m l h, m h^2 + I_perp]],  K = [[m g l + k_A + k_B, -k_B], [-k_B, m g h + k_B]]."""
    l, h = M.ell, M.h_c
    Mm = np.array([[M.m * l * l, M.m * l * h], [M.m * l * h, M.m * h * h + M.I_perp]])
    K = np.array([[M.m * M.g * l + M.k_A + M.k_B, -M.k_B], [-M.k_B, M.m * M.g * h + M.k_B]])
    return Mm, K


def lateral_modes(M: Model):
    from scipy.linalg import eigh
    Mm, K = lateral_matrices(M)
    w2, V = eigh(K, Mm)
    o = np.argsort(w2)
    V = V[:, o] / V[0, o]          # normalised to phi = 1  (theta / phi = mode ratio)
    return [float(math.sqrt(x)) for x in w2[o]], V


# ------------------------------------------------------------------------------------------------ quaternion helpers
def quat_mul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def quat_min_rotation(a, b):
    """unit quaternion (w,x,y,z) of the minimal rotation taking unit a to unit b."""
    a = np.asarray(a, float) / np.linalg.norm(a); b = np.asarray(b, float) / np.linalg.norm(b)
    w = 1.0 + a @ b
    v = np.cross(a, b)
    q = np.array([w, *v])
    return q / np.linalg.norm(q)


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot_batch(q):
    """q (4,B) -> R (B,3,3)."""
    w, x, y, z = q
    R = np.empty((q.shape[1], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def twist_angle(q):
    """psi = 2 atan2(q_z, q_w): twist about BODY z in q = q_swing (x) q_twist (works on (4,) or (4,B) or (N,4).T)."""
    return 2.0 * np.arctan2(q[3], q[0])


def grad_psi(q):
    """d psi / d theta for a WORLD-frame (left) rotation increment dq = 1/2 (0, dtheta) (x) q:
    G = (w x + z y... ) = [w y + z x, z y - w x, w^2 + z^2] / (w^2 + z^2)  (FORMULATION s3.2)."""
    w, x, y, z = q
    d = w * w + z * z
    return np.array([(w * y + z * x) / d, (z * y - w * x) / d, np.ones_like(w)])


def quat_from_tilt_twist(n_down, psi):
    """q = q_swing (x) q_twist: twist psi about body z, then the minimal rotation taking body z to -n_down (disc up-normal)."""
    qs = quat_min_rotation([0.0, 0.0, 1.0], -np.asarray(n_down, float))
    qt = np.array([math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2)])
    q = quat_mul(qs, qt)
    return q / np.linalg.norm(q)


# ------------------------------------------------------------------------------------------------ batched dynamics
def _params_arrays(Ms):
    f = lambda name: np.array([getattr(M, name) for M in Ms], float)
    P = {k: f(k) for k in ("m", "R", "t", "ell", "kappa", "k_A", "k_B", "c_t", "c_x", "c_r", "g")}
    for nm in ("c_t", "c_x", "c_r"):
        if np.any(P[nm] < 0):
            raise ValueError(f"{nm} < 0: negative damping (energy injection) is not allowed")
    for nm in ("m", "R", "ell", "kappa"):
        if np.any(P[nm] <= 0):
            raise ValueError(f"non-positive {nm}")
    if np.any(P["t"] < 0) or np.any(P["k_A"] < 0) or np.any(P["k_B"] < 0):
        raise ValueError("negative t, k_A or k_B")
    P["Iz"] = 0.5 * P["m"] * P["R"] ** 2
    P["Ip"] = P["m"] * (3 * P["R"] ** 2 + P["t"] ** 2) / 12.0
    P["hc"] = 0.5 * P["t"]
    return P


def initial_state(Ms):
    B = len(Ms)
    Y = np.zeros((NY, B))
    for j, M in enumerate(Ms):
        q = np.array(M.q0, float)
        if abs(np.linalg.norm(q) - 1) > 1e-12:
            raise ValueError("q0 must be a unit quaternion")
        Rm = quat_to_matrix(q)
        r = Rm @ np.array([0.0, 0.0, M.h_c])
        Bp = np.array(M.X0) + r
        if abs(np.linalg.norm(Bp) - M.ell) > 1e-12 * max(1.0, M.ell):
            raise ValueError(f"initial state violates |B| = ell ({np.linalg.norm(Bp)} vs {M.ell})")
        Bd = np.array(M.V0) + np.cross(M.w0, r)
        if abs(Bp @ Bd) > 1e-12 * max(1.0, M.ell * np.linalg.norm(Bd)):
            raise ValueError("initial velocity violates B . dB/dt = 0")
        Y[0:3, j] = M.X0; Y[3:6, j] = M.V0; Y[6:10, j] = q; Y[10:13, j] = M.w0
    return Y


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _kinematics(Y, P):
    X, V, q, w = Y[0:3], Y[3:6], Y[6:10], Y[10:13]
    qw, qx, qy, qz = q
    e3 = np.array([2 * (qx * qz + qy * qw), 2 * (qy * qz - qx * qw), 1 - 2 * (qx * qx + qy * qy)])   # body z in world
    r = e3 * P["hc"]                                    # (3,B) = R (0,0,hc)
    n = -e3                                             # disc downward normal
    Rm = None
    Bp = X + r
    L = np.sqrt(_dot(Bp, Bp))
    s = Bp / L
    Bd = V + _cross(w, r)
    return X, V, q, w, Rm, r, n, Bp, L, s, Bd


def forces(Y, P):
    """Returns accelerations (a, alpha), tension T and P_diss for the batch (index-1 constrained Newton-Euler)."""
    X, V, q, w, Rm, r, n, Bp, L, s, Bd = _kinematics(Y, P)
    m, g = P["m"], P["g"]
    # bending potentials: dV/dB (perpendicular to s) and dV/dn
    ez_perp = np.array([np.zeros_like(L), np.zeros_like(L), np.ones_like(L)]) - s[2] * s     # e_z - s_z s
    sn = _dot(s, n)
    gB = (P["k_A"] * ez_perp - P["k_B"] * (n - sn * s)) / L
    tau_n = P["k_B"] * _cross(n, s)                                                         # -n x dV/dn, dV/dn = -k_B s
    # torsion
    psi = twist_angle(q)
    G = grad_psi(q)
    psid = _dot(G, w)
    wn = _dot(w, n)
    wperp = w - wn * n
    F0 = -gB - P["c_x"] * V
    F0[2] -= m * g
    # world inertia: I = Rm diag(Ip, Ip, Iz) Rm^T = Ip 1 + (Iz - Ip) e3 e3^T, e3 = body z in world = -n
    Ip, Iz = P["Ip"], P["Iz"]
    Iw = Ip * w + (Iz - Ip) * wn * n
    tau0 = _cross(r, -gB) + tau_n - (P["kappa"] * psi + P["c_t"] * psid) * G - P["c_r"] * wperp - _cross(w, Iw)
    Iinv = lambda v: v / Ip + (1.0 / Iz - 1.0 / Ip) * _dot(v, n) * n
    a0 = F0 / m
    al0 = Iinv(tau0)
    u = _cross(r, s)
    Iu = Iinv(u)
    wxr = _cross(w, r)
    num = _dot(s, a0) + _dot(u, al0) + _dot(s, _cross(w, wxr)) + _dot(Bd, Bd) / L
    T = num / (1.0 / m + _dot(u, Iu))
    a = a0 - (T / m) * s
    al = al0 - T * Iu
    Pd = P["c_t"] * psid * psid + P["c_x"] * _dot(V, V) + P["c_r"] * _dot(wperp, wperp)
    return a, al, T, Pd, psi


def rhs(Y, P, aux=None):
    a, al, T, Pd, psi = forces(Y, P)
    if aux is not None:
        aux["T"] = T; aux["psi"] = psi
    q, w = Y[6:10], Y[10:13]
    dY = np.empty_like(Y)
    dY[0:3] = Y[3:6]
    dY[3:6] = a
    qw, qx, qy, qz = q
    dY[6] = 0.5 * (-w[0] * qx - w[1] * qy - w[2] * qz)
    dY[7] = 0.5 * (w[0] * qw + w[1] * qz - w[2] * qy)
    dY[8] = 0.5 * (-w[0] * qz + w[1] * qw + w[2] * qx)
    dY[9] = 0.5 * (w[0] * qy - w[1] * qx + w[2] * qw)
    dY[10:13] = al
    dY[13] = Pd
    return dY


class SolverGuardError(RuntimeError):
    """Non-finite state, constraint violated beyond tolerance after projection, step too large, or |psi| too large."""


class SlackWireError(RuntimeError):
    """Tension < 0: a wire cannot push (P7).  The model would need slack dynamics, which is out of scope."""


NORM_TOL = 1e-12
STEP_DRIFT_TOL = 1e-8
PSI_MAX = 1.9 * math.pi          # psi = 2 atan2(q_z, q_w) of the CONTINUOUS q is single-valued on (-2 pi, 2 pi)


def _project(Y, P):
    """Exact projection onto |q| = 1, |B| = ell, B . dB/dt = 0.  Returns the largest pre-projection relative drift."""
    q = Y[6:10]
    qn = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    dq = np.max(np.abs(qn - 1.0))
    Y[6:10] = q / qn
    X, V, q, w, Rm, r, n, Bp, L, s, Bd = _kinematics(Y, P)
    dL = np.max(np.abs(L - P["ell"]) / P["ell"])
    X = X + (P["ell"] - L) * s                     # moves B radially onto the sphere (orientation unchanged)
    V = V - _dot(s, Bd) * s                        # removes the radial velocity of B
    Y[0:3] = X; Y[3:6] = V
    return float(max(dq, dL))


def _check(Y, P, t_now):
    X, V, q, w, Rm, r, n, Bp, L, s, Bd = _kinematics(Y, P)
    e1 = np.max(np.abs(np.sqrt(_dot(q, q) + q[3] ** 2) - 1.0))
    e2 = np.max(np.abs(L - P["ell"]) / P["ell"])
    e3 = np.max(np.abs(_dot(s, Bd)) / np.maximum(1.0, np.sqrt(_dot(Bd, Bd))))
    if e1 > NORM_TOL or e2 > NORM_TOL or e3 > NORM_TOL:
        raise SolverGuardError(f"constraint after projection: |q|-1 {e1:.3g}, |B|/ell-1 {e2:.3g}, s.dB {e3:.3g} at t = {t_now:.6f} s")


def integrate(Ms, t_out, h=DEFAULT_H, on_slack="raise", stats=None):
    """Ms: Model or list of Models (batch).  Returns Y of shape (len(t_out), 14, B) (or (N,14) for a single Model).
    stats (dict) receives max_step_drift, min_tension_N, max_tension_N, max_abs_psi, first_slack_t."""
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
    _, _, T0, _, psi0 = forces(Y, P)
    tmin = T0.copy(); tmax = T0.copy(); pmax = np.abs(psi0); first_s = None
    if np.any(T0 < 0) and on_slack == "raise":
        raise SlackWireError(f"slack wire at t = 0 (tension {T0.min():.4g} N)")
    t_now = 0.0
    for i in range(1, len(t_out)):
        dt = t_out[i] - t_out[i - 1]
        nsub = max(1, int(math.ceil(dt / h - 1e-9)))
        hs = dt / nsub
        for j in range(nsub):
            aux = {}
            k1 = rhs(Y, P, aux)                      # tension / twist guards evaluated at every step node
            Tn, psin = aux["T"], aux["psi"]
            tmin = np.minimum(tmin, Tn); tmax = np.maximum(tmax, Tn); pmax = np.maximum(pmax, np.abs(psin))
            if np.any(np.abs(psin) > PSI_MAX):
                raise SolverGuardError(f"|psi| > 1.9 pi at t = {t_now:.6f} s (twist branch limit 2 pi)")
            if np.any(Tn < 0):
                if first_s is None:
                    first_s = t_now
                if on_slack == "raise":
                    raise SlackWireError(f"slack wire (tension {Tn.min():.4g} N < 0) at t = {t_now:.6f} s")
            k2 = rhs(Y + 0.5 * hs * k1, P)
            k3 = rhs(Y + 0.5 * hs * k2, P)
            k4 = rhs(Y + hs * k3, P)
            Y = Y + (hs / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t_now = t_out[i - 1] + (j + 1) * hs
            if not np.all(np.isfinite(Y)):
                raise SolverGuardError(f"non-finite state at t = {t_now:.6f} s")
            d = _project(Y, P)
            max_drift = max(max_drift, d)
            if d > STEP_DRIFT_TOL:
                raise SolverGuardError(f"per-step constraint drift {d:.3g} > {STEP_DRIFT_TOL} at t = {t_now:.6f} s (step too large)")
        _check(Y, P, t_now)                          # post-projection constraint guard at every output node
        out[i] = Y
    if stats is not None:
        stats.update(max_step_drift=max_drift, min_tension_N=tmin.tolist(), max_tension_N=tmax.tolist(),
                     max_abs_psi_rad=pmax.tolist(), first_slack_t=first_s)
    return out[..., 0] if single else out


# ------------------------------------------------------------------------------------------------ derived output
def euler_xyz_from_matrix(R):
    """Blender 'XYZ' Euler (R = Rz(c) Ry(b) Rx(a))."""
    b = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    a = math.atan2(R[2, 1], R[2, 2])
    c = math.atan2(R[1, 0], R[0, 0])
    return a, b, c


def euler_rates(e, w):
    """dE/dt for Blender XYZ Euler e = (a,b,c), world angular velocity w = a' Rz Ry ex + b' Rz ey + c' ez."""
    a, b, c = e
    cz, sz, cy, sy = math.cos(c), math.sin(c), math.cos(b), math.sin(b)
    A = np.column_stack([[cz * cy, sz * cy, -sy], [-sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return np.linalg.solve(A, w)


def derived(M: Model, Y):
    """Y: (N, 14) single trajectory -> dict of per-sample arrays (world frame, A = origin)."""
    P = _params_arrays([M])
    Yb = np.moveaxis(Y, 0, 1)                                       # (14, N): samples treated as a batch
    Pn = {k: np.full(Y.shape[0], v[0]) if isinstance(v, np.ndarray) else v for k, v in P.items()}
    a, al, T, Pd, psi = forces(Yb, Pn)
    X, V, q, w, Rm, r, n, Bp, L, s, Bd = _kinematics(Yb, Pn)
    G = grad_psi(q)
    psid = _dot(G, w)
    Iw = M.I_perp * w + (M.I_z - M.I_perp) * _dot(w, n) * n
    KE = 0.5 * M.m * _dot(V, V) + 0.5 * _dot(w, Iw)
    PE_g = M.m * M.g * (X[2] - (-M.ell - M.h_c))
    PE_t = 0.5 * M.kappa * psi ** 2
    PE_b = M.k_A * (1 + s[2]) + M.k_B * (1 - _dot(s, n))
    E = KE + PE_g + PE_t + PE_b
    qq = q.T
    Rm = rot_batch(q)
    eul = np.unwrap(np.array([euler_xyz_from_matrix(R) for R in Rm]), axis=0)
    eul_rate = np.array([euler_rates(e, ww) for e, ww in zip(eul, w.T)])
    phi_x = np.arctan2(s[0], -s[2]); phi_y = np.arctan2(s[1], -s[2])
    th_x = np.arctan2(n[0], -n[2]); th_y = np.arctan2(n[1], -n[2])
    tilt = np.arccos(np.clip(-n[2], -1, 1)); bend_B = np.arccos(np.clip(_dot(s, n), -1, 1)); swing = np.arccos(np.clip(-s[2], -1, 1))
    ex_body_world = Rm[:, :, 0]                                     # body x (zero-twist reference direction) in world
    return dict(X=X.T, V=V.T, A=a.T, q=qq, w=w.T, alpha=al.T, B=Bp.T, Bdot=Bd.T, s=s.T, n=n.T, L=L, psi=psi, psidot=psid,
                tension=T, tension_dir=s.T, torque_wire_z=-M.kappa * psi, euler=eul, euler_rate=eul_rate,
                phi_x=phi_x, phi_y=phi_y, theta_x=th_x, theta_y=th_y, tilt=tilt, bend_B=bend_B, swing=swing,
                ex_world=ex_body_world, KE=KE, PE_grav=PE_g, PE_twist=PE_t, PE_bend=PE_b, PE=PE_g + PE_t + PE_b, E=E,
                P_diss=Pd, W=Y[:, 13],
                norm_q_err=float(np.max(np.abs(np.linalg.norm(qq, axis=1) - 1))),
                len_err=float(np.max(np.abs(L - M.ell))))


def simulate(M: Model, t_out, h=DEFAULT_H, on_slack="raise"):
    st = {}
    Y = integrate(M, t_out, h, on_slack=on_slack, stats=st)
    d = derived(M, Y)
    d["integrator_stats"] = st
    d["t"] = np.asarray(t_out, float)
    d["Y"] = Y
    d["model"] = M
    return d


# ------------------------------------------------------------------------------------------------ params/physics.json
def _val(e, default=None):
    if isinstance(e, dict):
        v = e.get("value", default)
        return default if v is None else v
    return default if e is None else e


def model_from_physics_json(p: dict) -> Model:
    if p.get("model_id") != MODEL_ID:
        raise ValueError(f"model_id {p.get('model_id')!r} != {MODEL_ID}")
    P = p["parameters"]; ic = p["initial_state"]; damp = P["damping"]
    v = lambda k: float(_val(P[k]))
    return Model(m=v("m"), R=v("R"), t=v("t"), ell=v("ell"), kappa=v("kappa"), k_A=v("k_A"), k_B=v("k_B"),
                 c_t=float(_val(damp["c_t"])), c_x=float(_val(damp["c_x"])), c_r=float(_val(damp["c_r"])), g=v("g"),
                 X0=tuple(_val(ic["X0"])), V0=tuple(_val(ic["V0"])), q0=tuple(_val(ic["q0"])), w0=tuple(_val(ic["w0"])))


def with_(M: Model, **kw) -> Model:
    return replace(M, **kw)
