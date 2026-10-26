"""Verifier-owned model of the 3-D elastic pendulum (clip 02). numpy only (also runs inside Blender).

Derivation in verify/PLAN.md.  Never imports solver/.
State (rigid):  y = [x_G(3), v_G(3), q(4: w,x,y,z), w_b(3)]   (13)
State (point):  y = [x_E(3), v_E(3)]                           (6)
Composite body = weight (M, I_c at C, eye at e_b in body frame) + lumped spring mass mu = m_s/3 at E
(Rayleigh KE); spring gravity total m_s/2 g applied at E.
"""
import math
import numpy as np

G_STD = 9.80665
ZHAT = np.array([0.0, 0.0, 1.0])


def _arr(v, n=3):
    a = np.array(v, dtype=float).reshape(-1)
    if a.size != n:
        raise ValueError(f"expected {n} values, got {v}")
    return a


class Model:
    FIELDS = ("g", "A", "k", "L0", "F_i", "M", "m_s", "body", "e_b", "I_c", "c_v", "c_a", "c_r",
              "E0", "vE0", "q0", "w0")

    def __init__(self, k, L0, M, g=G_STD, A=(0, 0, 0), F_i=0.0, m_s=0.0, body="point", e_b=(0, 0, 0),
                 I_c=(0, 0, 0), c_v=0.0, c_a=0.0, c_r=0.0, E0=None, vE0=(0, 0, 0), q0=(1, 0, 0, 0), w0=(0, 0, 0),
                 provenance=None):
        self.k, self.L0, self.M, self.g = float(k), float(L0), float(M), float(g)
        self.A = _arr(A)
        self.F_i, self.m_s = float(F_i), float(m_s)
        self.body = body
        self.e_b = _arr(e_b)
        Ic = np.array(I_c, dtype=float)
        self.I_c = np.diag(Ic) if Ic.shape == (3,) else Ic.reshape(3, 3)
        self.c_v, self.c_a, self.c_r = float(c_v), float(c_a), float(c_r)
        self.E0 = _arr(E0) if E0 is not None else self.A - np.array([0, 0, self.ell_eq])
        self.vE0 = _arr(vE0)
        self.q0 = _arr(q0, 4) / np.linalg.norm(q0)
        self.w0 = _arr(w0)
        self.provenance = provenance or {}
        if body == "point":
            self.e_b = np.zeros(3)
        self._derive()

    def _derive(self):
        self.mu = self.m_s / 3.0
        self.Mt = self.M + self.mu
        self.W = (self.M + 0.5 * self.m_s) * self.g          # static load on the spring
        self.gb = self.mu * self.e_b / self.Mt               # composite CoM in body frame (origin = C)
        if self.body == "rigid":
            I = self.I_c.copy()
            for m, r in ((self.M, -self.gb), (self.mu, self.e_b - self.gb)):
                I += m * ((r @ r) * np.eye(3) - np.outer(r, r))
            self.I_G = I
            self.I_G_inv = np.linalg.inv(I)

    # ---------------------------------------------------------------- analytic quantities
    @property
    def ell_eq(self):
        return self.L0 + ((self.M + 0.5 * self.m_s) * self.g - self.F_i) / self.k

    @property
    def omega_z(self):
        return math.sqrt(self.k / (self.M + self.m_s / 3.0))

    @property
    def omega_s(self):
        return math.sqrt((self.M + 0.5 * self.m_s) * self.g / ((self.M + self.m_s / 3.0) * self.ell_eq))

    @property
    def damped(self):
        return self.c_v != 0 or self.c_a != 0 or self.c_r != 0

    def as_dict(self):
        return dict(g=self.g, A=self.A.tolist(), k=self.k, L0=self.L0, F_i=self.F_i, M=self.M, m_s=self.m_s,
                    body=self.body, e_b=self.e_b.tolist(), I_c=self.I_c.tolist(), c_v=self.c_v, c_a=self.c_a,
                    c_r=self.c_r, E0=self.E0.tolist(), vE0=self.vE0.tolist(), q0=self.q0.tolist(), w0=self.w0.tolist())

    def replace(self, **kw):
        d = self.as_dict(); d.update(kw)
        prov = dict(self.provenance); prov["replaced"] = sorted(kw)
        return Model(provenance=prov, **d)

    @property
    def ndim(self):
        return 13 if self.body == "rigid" else 6

    # ---------------------------------------------------------------- state conversions
    def initial_state(self):
        if self.body != "rigid":
            return np.r_[self.E0, self.vE0]
        R = quat_to_R(self.q0)
        C = self.E0 - R @ self.e_b
        G = C + R @ self.gb
        vG = self.vE0 + np.cross(self.w0, R @ (self.gb - self.e_b))
        wb = R.T @ self.w0
        return np.r_[G, vG, self.q0, wb]

    def kinematics(self, y):
        """y (...,n) -> dict of world quantities (E, vE, C, vC, R, w_world)."""
        y = np.asarray(y, float)
        if self.body != "rigid":
            E, vE = y[..., 0:3], y[..., 3:6]
            R = np.broadcast_to(np.eye(3), y.shape[:-1] + (3, 3))
            w = np.zeros_like(E)
            return dict(E=E, vE=vE, C=E, vC=vE, G=E, vG=vE, R=R, w=w, wb=w, q=np.broadcast_to(np.array([1.0, 0, 0, 0]), y.shape[:-1] + (4,)))
        G, vG, q, wb = y[..., 0:3], y[..., 3:6], y[..., 6:10], y[..., 10:13]
        R = quat_to_R(q)
        w = np.einsum("...ij,...j->...i", R, wb)
        rE = np.einsum("...ij,j->...i", R, self.e_b - self.gb)
        rC = np.einsum("...ij,j->...i", R, -self.gb)
        E, C = G + rE, G + rC
        vE, vC = vG + np.cross(w, rE), vG + np.cross(w, rC)
        return dict(E=E, vE=vE, C=C, vC=vC, G=G, vG=vG, R=R, w=w, wb=wb, q=q / np.linalg.norm(q, axis=-1, keepdims=True), rE=rE, rC=rC)

    # ---------------------------------------------------------------- dynamics
    def rhs(self, y):
        y = np.asarray(y, float)
        K = self.kinematics(y)
        d = K["E"] - self.A
        ell = np.linalg.norm(d, axis=-1, keepdims=True)
        u = d / ell
        elld = np.sum(u * K["vE"], axis=-1, keepdims=True)
        T = self.F_i + self.k * (ell - self.L0)
        F_E = -(T + self.c_a * elld) * u - 0.5 * self.m_s * self.g * ZHAT   # spring tension + all spring gravity (mu g + (m_s/2-mu) g)
        F_C = -self.M * self.g * ZHAT - self.c_v * K["vC"]
        out = np.empty_like(y)
        if self.body != "rigid":
            out[..., 0:3] = K["vE"]
            out[..., 3:6] = (F_E + F_C) / self.Mt   # point: E == C, mu at E
            return out
        F = F_E + F_C
        tau = np.cross(K["rE"], F_E) + np.cross(K["rC"], F_C) - self.c_r * K["w"]
        tau_b = np.einsum("...ji,...j->...i", K["R"], tau)
        wb = K["wb"]
        Iw = wb @ self.I_G.T
        wdot = (tau_b - np.cross(wb, Iw)) @ self.I_G_inv.T
        q = y[..., 6:10]
        out[..., 0:3] = K["vG"]
        out[..., 3:6] = F / self.Mt
        out[..., 6:10] = 0.5 * quat_mul(q, np.concatenate([np.zeros(wb.shape[:-1] + (1,)), wb], axis=-1))
        out[..., 10:13] = wdot
        return out

    def observables(self, y):
        """Canonical output columns (dict of arrays) from states y (N,n)."""
        K = self.kinematics(y)
        d = K["E"] - self.A
        ell = np.linalg.norm(d, axis=-1)
        u = d / ell[..., None]
        elld = np.sum(u * K["vE"], axis=-1)
        T = self.F_i + self.k * (ell - self.L0)
        KE = 0.5 * self.Mt * np.sum(K["vG"] ** 2, axis=-1)
        if self.body == "rigid":
            KE = KE + 0.5 * np.einsum("...i,ij,...j->...", K["wb"], self.I_G, K["wb"])
        PE = 0.5 * self.k * (ell - self.L0) ** 2 + self.F_i * (ell - self.L0) + self.g * (self.M * K["C"][..., 2] + 0.5 * self.m_s * K["E"][..., 2])
        Pd = self.c_v * np.sum(K["vC"] ** 2, axis=-1) + self.c_a * elld ** 2 + self.c_r * np.sum(K["w"] ** 2, axis=-1)
        L = self.Mt * np.cross(K["G"] - self.A, K["vG"])
        if self.body == "rigid":
            L = L + np.einsum("...ij,...j->...i", K["R"], K["wb"] @ self.I_G.T)
        q = K["q"]
        out = dict(x_m=K["E"][..., 0], y_m=K["E"][..., 1], z_m=K["E"][..., 2],
                   vx_m_s=K["vE"][..., 0], vy_m_s=K["vE"][..., 1], vz_m_s=K["vE"][..., 2],
                   xc_m=K["C"][..., 0], yc_m=K["C"][..., 1], zc_m=K["C"][..., 2],
                   qw=q[..., 0], qx=q[..., 1], qy=q[..., 2], qz=q[..., 3],
                   wx=K["w"][..., 0], wy=K["w"][..., 1], wz=K["w"][..., 2],
                   ell_m=ell, elldot_m_s=elld, tension_N=T, KE_J=KE, PE_J=PE, E_J=KE + PE, P_diss_W=Pd,
                   Lx=L[..., 0], Ly=L[..., 1], Lz=L[..., 2])
        return out

    def E_static(self):
        """Energy of the static equilibrium (minimum of E for the hanging rest state)."""
        ys = self.replace(E0=(self.A - np.array([0, 0, self.ell_eq])).tolist(), vE0=[0, 0, 0], q0=[1, 0, 0, 0], w0=[0, 0, 0]).initial_state()
        return float(self.observables(ys[None])["E_J"][0])

    def eye_accel(self, E, vE):
        """Point model only: acceleration of E from (E, vE) arrays (N,3) -- used for quintic Hermite."""
        y = np.concatenate([E, vE], axis=-1)
        return self.rhs(y)[..., 3:6]


# ---------------------------------------------------------------- quaternion helpers (w,x,y,z)
def quat_mul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def quat_to_R(q):
    q = np.asarray(q, float)
    n2 = np.sum(q * q, axis=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    s = 2.0 / n2
    R = np.stack([np.stack([1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)], -1),
                  np.stack([s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)], -1),
                  np.stack([s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)], -1)], -2)
    return R


def rot_angle(Ra, Rb):
    """angle of Ra^T Rb (rad), robust for small angles."""
    D = np.einsum("...ji,...jk->...ik", Ra, Rb)
    s = 0.5 * np.linalg.norm(np.stack([D[..., 2, 1] - D[..., 1, 2], D[..., 0, 2] - D[..., 2, 0], D[..., 1, 0] - D[..., 0, 1]], -1), axis=-1)
    c = 0.5 * (np.trace(D, axis1=-2, axis2=-1) - 1)
    return np.arctan2(s, c)


def axis_angle_quat(axis, ang):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    return np.r_[math.cos(ang / 2), math.sin(ang / 2) * axis]


# ---------------------------------------------------------------- closed forms (PLAN.md B1-B8)
def axial_exact(m, t, a):
    """B2: eye z for release from rest at z = -(ell_eq + a) on the vertical (point model or rigid with C below E)."""
    gam = (m.c_a + m.c_v) / (2 * m.Mt)
    wd = math.sqrt(m.omega_z ** 2 - gam ** 2)
    zeq = m.A[2] - m.ell_eq
    return zeq - a * np.exp(-gam * t) * (np.cos(wd * t) + gam / wd * np.sin(wd * t)), wd, gam


def conical(m, alpha):
    """B3: exact circular orbit at cone half-angle alpha (point, undamped)."""
    Tn = m.W / math.cos(alpha)
    ell = m.L0 + (Tn - m.F_i) / m.k
    Om = math.sqrt(Tn / (m.Mt * ell))
    rho = ell * math.sin(alpha)
    return dict(ell=ell, Omega=Om, rho=rho, z=m.A[2] - ell * math.cos(alpha), Lz=m.Mt * rho * rho * Om)


def rigid_planar_modes(m, axis="y"):
    """B5: linear swing-mode angular frequencies (rad/s) and mode shapes for (phi_spring, phi_body)
    about the given horizontal axis; eye straight above the CoM at distance d."""
    d = float(np.linalg.norm(m.e_b))
    I = m.I_c[1, 1] if axis == "y" else m.I_c[0, 0]
    l = m.ell_eq
    Mm = np.array([[m.Mt * l * l, m.M * l * d], [m.M * l * d, m.M * d * d + I]])
    Kk = np.diag([m.W * l, m.M * m.g * d])
    ev, V = np.linalg.eig(np.linalg.solve(Mm, Kk))
    o = np.argsort(ev.real)
    return np.sqrt(ev.real[o]), V[:, o].real


def autoparametric_rate(m, eps, detune=0.0):
    """B8: linear-phase swing growth rate for axial amplitude eps*ell_eq."""
    s0 = 0.75 * eps * m.omega_s
    v = s0 * s0 - (detune / 2) ** 2
    return math.sqrt(v) if v > 0 else 0.0
