"""Verifier-owned model of the clip-03 torsion disc (rigid disc on a massless inextensible wire). numpy only.

Derivation in verify/PLAN.md. Never imports solver/.
Modes
  rigid_pivot : disc locked perpendicular to the wire; wire+disc = one rigid body about the fixed clamp point A.
                state y = [q_d(4), w_db(3)]                                     (7)   speeds v = w_db (3)
  string      : wire pivots freely at A (massless rod, inextensible), disc tilts at P (optional bending k_b).
                state y = [q_w(4), w_wb_xy(2), q_d(4), w_db(3)]                 (13)  speeds v = (w_wb_x, w_wb_y, w_db) (5)
                The wire frame never spins about its own axis (w_wb_z = 0); only u = R_w e_z is physical.
Twist psi (selectable):  vertical : R_d = Rmin(z->a) Rz(psi)
                          transport: R_d = Rmin(u->a) Rmin(z->u) Rz(psi)
                          euler_zyx: psi = atan2(R10, R00)
Potential gradients wrt virtual rotations are complex-step derivatives (exact to round-off).
"""
import math
import numpy as np

G_STD = 9.80665
EZ = np.array([0.0, 0.0, 1.0])
EX = np.array([1.0, 0.0, 0.0])
CS = 1e-30  # complex-step size


def _arr(v, n=3):
    a = np.array(v, dtype=float).reshape(-1)
    if a.size != n:
        raise ValueError(f"expected {n} values, got {v}")
    return a


# ---------------------------------------------------------------- quaternion / rotation helpers (w,x,y,z)
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
    return np.stack([np.stack([1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)], -1),
                     np.stack([s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)], -1),
                     np.stack([s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)], -1)], -2)


def R_to_quat(R):
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        q = [0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S]
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        S = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / S
        q[1 + i] = 0.25 * S
        q[1 + j] = (R[j, i] + R[i, j]) / S
        q[1 + k] = (R[k, i] + R[i, k]) / S
    q = np.array(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def rot_angle(Ra, Rb):
    """angle of Ra^T Rb (rad), robust for small angles."""
    D = np.einsum("...ji,...jk->...ik", Ra, Rb)
    s = 0.5 * np.linalg.norm(np.stack([D[..., 2, 1] - D[..., 1, 2], D[..., 0, 2] - D[..., 2, 0], D[..., 1, 0] - D[..., 0, 1]], -1), axis=-1)
    c = 0.5 * (np.trace(D, axis1=-2, axis2=-1) - 1)
    return np.arctan2(s, c)


def axis_angle_quat(axis, ang):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    return np.r_[math.cos(ang / 2), math.sin(ang / 2) * axis]


def axis_angle_R(axis, ang):
    return quat_to_R(axis_angle_quat(axis, ang))


def skew(v):
    v = np.asarray(v)
    z = np.zeros_like(v[..., 0])
    return np.stack([np.stack([z, -v[..., 2], v[..., 1]], -1), np.stack([v[..., 2], z, -v[..., 0]], -1),
                     np.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def rmin_apply(a, b, v):
    """apply the minimal rotation taking unit a to unit b to v (holomorphic; a.b != -1)."""
    k = np.cross(a, b)
    c = np.sum(a * b, -1)[..., None]
    return c * v + np.cross(k, v) + k * (np.sum(k * v, -1)[..., None]) / (1.0 + c)


def rmin_R(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return np.stack([rmin_apply(a, b, e) for e in np.eye(3)], -1)


def catan2(y, x):
    """atan2 valid for complex-step arguments (first order in the imaginary part)."""
    if np.iscomplexobj(y) or np.iscomplexobj(x):
        yr, xr = np.real(y), np.real(x)
        return np.arctan2(yr, xr) + 1j * (xr * np.imag(y) - yr * np.imag(x)) / (xr * xr + yr * yr)
    return np.arctan2(y, x)


def twist_of(conv, u, a, x, q=None):
    """psi from wire direction u, disc normal a = R e_z, disc x axis x = R e_x (all (...,3), complex-step safe).
    'vertical' with the (continuous) quaternion q given: psi = 2 atan2(q_z, q_w) -- the same swing-twist angle, lifted to the
    quaternion double cover, i.e. valid on (-2 pi, 2 pi) instead of wrapping at +-pi (clip-03 ck001 twists by > 180 deg)."""
    if conv == "vertical" and q is not None:
        return 2.0 * catan2(q[..., 3], q[..., 0])
    if conv == "vertical":
        xp = rmin_apply(a, np.broadcast_to(EZ, a.shape), x)
    elif conv == "transport":
        xp = rmin_apply(u, np.broadcast_to(EZ, u.shape), rmin_apply(a, u, x))
    elif conv == "euler_zyx":
        xp = x
    else:
        raise ValueError(conv)
    return catan2(xp[..., 1], xp[..., 0])


def _h(y):
    """h(y) = arctan(sqrt y)/sqrt y, holomorphic near y = 0."""
    y = np.asarray(y)
    small = np.abs(y) < 1e-6
    ys = np.where(small, y, 0.0)
    ser = 1 - ys / 3 + ys ** 2 / 5 - ys ** 3 / 7 + ys ** 4 / 9
    yl = np.where(small, 1.0, y)
    sq = np.sqrt(yl.astype(complex)) if np.iscomplexobj(yl) else np.sqrt(yl)
    lg = np.arctan(sq) / sq
    return np.where(small, ser, lg)


def bend_angle2(u, a):
    """beta^2, beta = angle(u, a) (holomorphic; u.a > 0)."""
    c = np.sum(u * a, -1)
    w = np.cross(u, a)
    S = np.sum(w * w, -1)
    y = S / (c * c)
    return y * _h(y) ** 2


class Model:
    FIELDS = ("mode", "g", "A", "L", "M", "I_c", "p_b", "kappa", "k_a", "k_b", "bend_law", "twist", "c_t", "c_s", "c_v", "c_r", "c_rp",
              "q0", "w0", "u0", "udot0")

    def __init__(self, L, M, I_c, kappa, mode="rigid_pivot", g=G_STD, A=(0, 0, 0), p_b=(0, 0, 0), k_a=0.0, k_b=0.0, bend_law="angle2",
                 twist="transport", c_t=0.0, c_s=0.0, c_v=0.0, c_r=0.0, c_rp=0.0, q0=(1, 0, 0, 0), w0=(0, 0, 0), u0=(0, 0, 1),
                 udot0=(0, 0, 0), provenance=None):
        if mode not in ("rigid_pivot", "string"):
            raise ValueError(mode)
        self.mode = mode
        self.L, self.M, self.g = float(L), float(M), float(g)
        self.A = _arr(A)
        Ic = np.array(I_c, dtype=float)
        self.I_c = np.diag(Ic) if Ic.shape == (3,) else Ic.reshape(3, 3)
        self.p_b = _arr(p_b)
        self.kappa, self.k_a, self.k_b = float(kappa), float(k_a), float(k_b)
        self.bend_law, self.twist = bend_law, twist
        self.c_t, self.c_s, self.c_v, self.c_r, self.c_rp = float(c_t), float(c_s), float(c_v), float(c_r), float(c_rp)
        self.q0 = _arr(q0, 4) / np.linalg.norm(q0)
        self.w0 = _arr(w0)
        self.u0 = _arr(u0) / np.linalg.norm(u0)
        self.udot0 = _arr(udot0)
        self.provenance = provenance or {}

    # ---------------------------------------------------------------- bookkeeping
    def as_dict(self):
        return dict(mode=self.mode, g=self.g, A=self.A.tolist(), L=self.L, M=self.M, I_c=self.I_c.tolist(), p_b=self.p_b.tolist(),
                    kappa=self.kappa, k_a=self.k_a, k_b=self.k_b, bend_law=self.bend_law, twist=self.twist, c_t=self.c_t,
                    c_s=self.c_s, c_v=self.c_v, c_r=self.c_r, c_rp=self.c_rp, q0=self.q0.tolist(), w0=self.w0.tolist(), u0=self.u0.tolist(), udot0=self.udot0.tolist())

    def replace(self, **kw):
        d = self.as_dict(); d.update(kw)
        prov = dict(self.provenance); prov["replaced"] = sorted(kw)
        return Model(provenance=prov, **d)

    @property
    def damped(self):
        return any(c != 0 for c in (self.c_t, self.c_s, self.c_v, self.c_r, self.c_rp))

    @property
    def nspeed(self):
        return 3 if self.mode == "rigid_pivot" else 5

    @property
    def ndim(self):
        return 7 if self.mode == "rigid_pivot" else 13

    @property
    def omega_t(self):
        return math.sqrt(self.kappa / self.I_c[2, 2])

    @property
    def gamma_t(self):
        return (self.c_t + self.c_r) / (2 * self.I_c[2, 2])

    # ---------------------------------------------------------------- initial state
    def initial_state(self):
        if self.mode == "rigid_pivot":
            return np.r_[self.q0, quat_to_R(self.q0).T @ self.w0]
        u = self.u0
        Rw = rmin_R(EZ, u) if u[2] > -0.999 else axis_angle_R(EX, math.pi)
        qw = R_to_quat(Rw)
        ww = np.cross(u, self.udot0)            # wire angular velocity, perpendicular to u
        wwb = Rw.T @ ww
        return np.r_[qw, wwb[:2], self.q0, quat_to_R(self.q0).T @ self.w0]

    # ---------------------------------------------------------------- kinematics
    def kin(self, y):
        y = np.asarray(y, float)
        sh = y.shape[:-1]
        K = {}
        if self.mode == "rigid_pivot":
            qd, wdb = y[..., 0:4], y[..., 4:7]
            Rd = quat_to_R(qd)
            Rw = Rd
            v = wdb
            wwb = wdb
        else:
            qw, wxy, qd, wdb = y[..., 0:4], y[..., 4:6], y[..., 6:10], y[..., 10:13]
            Rw, Rd = quat_to_R(qw), quat_to_R(qd)
            wwb = np.concatenate([wxy, np.zeros(sh + (1,))], -1)
            v = np.concatenate([wxy, wdb], -1)
        u = Rw[..., :, 2]
        a = Rd[..., :, 2]
        xd = Rd[..., :, 0]
        ww = np.einsum("...ij,...j->...i", Rw, wwb)
        wd = np.einsum("...ij,...j->...i", Rd, wdb)
        P = self.A - self.L * u
        r = np.einsum("...ij,j->...i", Rd, self.p_b)      # C -> P
        C = P - r
        n = self.nspeed
        # Jacobians (world): v_C = J_C v, w_d = J_w_d v, w_wire = J_ww v
        if self.mode == "rigid_pivot":
            rho = C - self.A
            J_C = -np.einsum("...ij,...jk->...ik", skew(rho), Rd)
            J_d = Rd
            J_ww = Rd
            aC0 = np.cross(wd, np.cross(wd, rho))
        else:
            J_C = np.zeros(sh + (3, n)); J_d = np.zeros(sh + (3, n)); J_ww = np.zeros(sh + (3, n))
            J_C[..., :, 0] = self.L * Rw[..., :, 1]
            J_C[..., :, 1] = -self.L * Rw[..., :, 0]
            J_C[..., :, 2:5] = np.einsum("...ij,...jk->...ik", skew(r), Rd)
            J_d[..., :, 2:5] = Rd
            J_ww[..., :, 0:2] = Rw[..., :, 0:2]
            aC0 = -self.L * np.cross(ww, np.cross(ww, u)) - np.cross(wd, np.cross(wd, r))
        vC = np.einsum("...ij,...j->...i", J_C, v)
        udot = np.cross(ww, u)
        K.update(qd=qd, Rw=Rw, Rd=Rd, u=u, a=a, xd=xd, ww=ww, wd=wd, wdb=wdb, wwb=wwb, P=P, C=C, r=r, v=v, vC=vC, udot=udot,
                 J_C=J_C, J_d=J_d, J_ww=J_ww, aC0=aC0)
        return K

    # ---------------------------------------------------------------- potentials (complex-step gradients)
    def _vint(self, u, a, x, q=None):
        V = 0.5 * self.kappa * twist_of(self.twist, u, a, x, q) ** 2
        if self.mode == "string" and self.k_b != 0.0:
            if self.bend_law == "angle2":
                V = V + 0.5 * self.k_b * bend_angle2(u, a)
            else:
                V = V + self.k_b * (1.0 - np.sum(u * a, -1))
        if self.mode == "string" and self.k_a != 0.0:
            V = V + self.k_a * (1.0 - u[..., 2])          # clamp boundary-layer bending, cos law (wire vs vertical)
        return V

    def grads(self, u, a, x, q):
        """values psi, V_int and gradients wrt world virtual rotations of wire (3) and disc (3): (...,6) each."""
        E = np.eye(3)
        ones = np.zeros(u.shape[:-1] + (6, 3), complex)
        uc = ones + u[..., None, :]; ac = ones + a[..., None, :]; xc = ones + x[..., None, :]
        qc = np.zeros(u.shape[:-1] + (6, 4), complex) + q[..., None, :]
        for k in range(3):
            uc[..., k, :] += 1j * CS * np.cross(E[k], u)
            ac[..., 3 + k, :] += 1j * CS * np.cross(E[k], a)
            xc[..., 3 + k, :] += 1j * CS * np.cross(E[k], x)
            qc[..., 3 + k, :] += 1j * CS * 0.5 * quat_mul(np.r_[0.0, E[k]], q)
        psi_c = twist_of(self.twist, uc, ac, xc, qc)
        V_c = self._vint(uc, ac, xc, qc)
        psi = np.real(psi_c[..., 0]); V = np.real(V_c[..., 0])
        return psi, V, np.imag(psi_c) / CS, np.imag(V_c) / CS

    # ---------------------------------------------------------------- dynamics
    def _forces(self, K):
        u, a, x = K["u"], K["a"], K["xd"]
        psi, V, gpsi, gV = self.grads(u, a, x, K["qd"])
        J_C, J_d, J_ww, v = K["J_C"], K["J_d"], K["J_ww"], K["v"]
        Jt = lambda J, f: np.einsum("...ij,...i->...j", J, f)
        psidot = np.sum(gpsi[..., :3] * K["ww"], -1) + np.sum(gpsi[..., 3:] * K["wd"], -1)
        FC = -self.M * self.g * EZ - self.c_v * K["vC"]
        Q = Jt(J_C, FC)
        Q = Q - Jt(J_ww, gV[..., :3]) - Jt(J_d, gV[..., 3:])
        if self.c_t:
            Q = Q - self.c_t * psidot[..., None] * (Jt(J_ww, gpsi[..., :3]) + Jt(J_d, gpsi[..., 3:]))
        if self.c_s:
            B = -np.einsum("...ij,...jk->...ik", skew(u), J_ww)         # udot = B v
            Q = Q - self.c_s * Jt(B, K["udot"])
        if self.c_r:
            Q = Q - self.c_r * Jt(J_d, K["wd"])
        if self.c_rp:
            Q = Q - self.c_rp * Jt(J_d, self.w_perp(K))
        return Q, dict(psi=psi, V=V, gpsi=gpsi, gV=gV, psidot=psidot)

    @staticmethod
    def w_perp(K):
        """disc angular velocity without its component along the disc normal (tilt rate)."""
        a, wd = K["a"], K["wd"]
        return wd - np.sum(wd * a, -1)[..., None] * a

    def mass_matrix(self, K):
        Iw = np.einsum("...ij,jk,...lk->...il", K["Rd"], self.I_c, K["Rd"])
        Mm = self.M * np.einsum("...ji,...jk->...ik", K["J_C"], K["J_C"]) + np.einsum("...ji,...jk,...kl->...il", K["J_d"], Iw, K["J_d"])
        return Mm, Iw

    def accel(self, y):
        K = self.kin(y)
        Q, aux = self._forces(K)
        Mm, Iw = self.mass_matrix(K)
        wd = K["wd"]
        gyro = np.cross(wd, np.einsum("...ij,...j->...i", Iw, wd))
        rhs = Q - self.M * np.einsum("...ji,...j->...i", K["J_C"], K["aC0"]) - np.einsum("...ji,...j->...i", K["J_d"], gyro)
        vdot = np.linalg.solve(Mm, rhs[..., None])[..., 0]
        return vdot, K, Q, aux, Mm

    def rhs(self, y, aug=False):
        """dy/dt; with aug=True also returns (P_diss, tau_z) from the same evaluation."""
        y = np.asarray(y, float)
        vdot, K, Q, aux, _ = self.accel(y)
        out = np.empty_like(y)
        if self.mode == "rigid_pivot":
            q = y[..., 0:4]
            out[..., 0:4] = 0.5 * quat_mul(q, np.concatenate([np.zeros(y.shape[:-1] + (1,)), y[..., 4:7]], -1))
            out[..., 4:7] = vdot
        else:
            qw, qd = y[..., 0:4], y[..., 6:10]
            z = np.zeros(y.shape[:-1] + (1,))
            out[..., 0:4] = 0.5 * quat_mul(qw, np.concatenate([z, y[..., 4:6], z], -1))
            out[..., 4:6] = vdot[..., 0:2]
            out[..., 6:10] = 0.5 * quat_mul(qd, np.concatenate([z, y[..., 10:13]], -1))
            out[..., 10:13] = vdot[..., 2:5]
        if aug:
            Pd = self._pdiss(K, aux)
            return out, Pd, np.sum(Q * self.v_alpha(K), -1)
        return out

    def _pdiss(self, K, aux):
        return (self.c_t * aux["psidot"] ** 2 + self.c_s * np.sum(K["udot"] ** 2, -1) + self.c_v * np.sum(K["vC"] ** 2, -1)
                + self.c_r * np.sum(K["wd"] ** 2, -1) + self.c_rp * np.sum(self.w_perp(K) ** 2, -1))

    def v_alpha(self, K):
        """generalized-speed direction of a global rotation about +Z."""
        zd = np.einsum("...ji,j->...i", K["Rd"], EZ)
        if self.mode == "rigid_pivot":
            return zd
        zw = np.einsum("...ji,j->...i", K["Rw"], EZ)
        return np.concatenate([zw[..., :2], zd], -1)

    def observables(self, y):
        y = np.asarray(y, float)
        vdot, K, Q, aux, Mm = self.accel(y)
        v = K["v"]
        KE = 0.5 * np.einsum("...i,...ij,...j->...", v, Mm, v)
        PE = self.M * self.g * K["C"][..., 2] + aux["V"]
        aC = np.einsum("...ij,...j->...i", K["J_C"], vdot) + K["aC0"]
        Fw = self.M * aC + self.M * self.g * EZ + self.c_v * K["vC"]     # wire force on disc
        T = np.sum(Fw * K["u"], -1)
        Pd = self._pdiss(K, aux)
        Iw = np.einsum("...ij,jk,...lk->...il", K["Rd"], self.I_c, K["Rd"])
        Lvec = self.M * np.cross(K["C"] - self.A, K["vC"]) + np.einsum("...ij,...j->...i", Iw, K["wd"])
        tau_z = np.sum(Q * self.v_alpha(K), -1)
        q = y[..., 0:4] if self.mode == "rigid_pivot" else y[..., 6:10]
        u, a = K["u"], K["a"]
        beta = np.arctan2(np.linalg.norm(np.cross(u, a), axis=-1), np.sum(u * a, -1))
        out = dict(x_m=K["C"][..., 0], y_m=K["C"][..., 1], z_m=K["C"][..., 2],
                   vx_m_s=K["vC"][..., 0], vy_m_s=K["vC"][..., 1], vz_m_s=K["vC"][..., 2],
                   px_m=K["P"][..., 0], py_m=K["P"][..., 1], pz_m=K["P"][..., 2],
                   qw=q[..., 0], qx=q[..., 1], qy=q[..., 2], qz=q[..., 3],
                   wx=K["wd"][..., 0], wy=K["wd"][..., 1], wz=K["wd"][..., 2],
                   ux=u[..., 0], uy=u[..., 1], uz=u[..., 2],
                   swing_x_rad=np.arctan2(-u[..., 1], u[..., 2]), swing_y_rad=np.arctan2(u[..., 0], u[..., 2]),
                   tilt_rad=beta, psi_rad=aux["psi"], psidot_rad_s=aux["psidot"],
                   psi_vertical_rad=twist_of("vertical", u, a, K["xd"], K["qd"]), psi_transport_rad=twist_of("transport", u, a, K["xd"]),
                   psi_euler_rad=twist_of("euler_zyx", u, a, K["xd"]),
                   tension_N=T, Fwx_N=Fw[..., 0], Fwy_N=Fw[..., 1], Fwz_N=Fw[..., 2],
                   KE_J=KE, PE_J=PE, E_J=KE + PE, P_diss_W=Pd, Lx=Lvec[..., 0], Ly=Lvec[..., 1], Lz=Lvec[..., 2], tau_z_N_m=tau_z)
        return out

    def E_static(self):
        ys = self.replace(q0=[1, 0, 0, 0], w0=[0, 0, 0], u0=[0, 0, 1], udot0=[0, 0, 0]).initial_state()
        return float(self.observables(ys[None])["E_J"][0])

    # ---------------------------------------------------------------- closed forms (PLAN.md)
    def torsion_exact(self, t, psi0, psidot0=0.0):
        """B2/B3: pure twist about the vertical (C on axis below A, principal z axis)."""
        I = self.I_c[2, 2]; w0 = self.omega_t; gam = self.gamma_t
        wd = math.sqrt(w0 * w0 - gam * gam)
        psi = np.exp(-gam * t) * (psi0 * np.cos(wd * t) + (psidot0 + gam * psi0) / wd * np.sin(wd * t))
        return psi, wd, gam

    def swing_modes(self, axis="y"):
        """B4: linear swing angular frequencies (rad/s). axis = rotation axis of the swing plane."""
        d = self.p_b[2]   # p_b is the body offset C -> P, so the CoM hangs d below the attachment
        I = self.I_c[1, 1] if axis == "y" else self.I_c[0, 0]
        if self.mode == "rigid_pivot":
            l = self.L + d
            return np.array([math.sqrt(self.M * self.g * l / (I + self.M * l * l))]), None
        L, M, g, kb, ka = self.L, self.M, self.g, self.k_b, self.k_a
        if self.bend_law != "cos" and kb != 0:
            pass  # angle2 and cos laws agree to second order: identical linear modes
        Mm = np.array([[M * L * L, M * L * d], [M * L * d, M * d * d + I]])
        Kk = np.array([[M * g * L + ka + kb, -kb], [-kb, M * g * d + kb]])
        ev, V = np.linalg.eig(np.linalg.solve(Mm, Kk))
        o = np.argsort(ev.real)
        return np.sqrt(ev.real[o]), V[:, o].real


def conical_rigid_pivot(m, alpha):
    """B5 rigid_pivot (kappa = 0): uniform rotation w = Omega z with the body axis tilted by alpha (C displaced to +x).
    Returns Omega and the IC (q0, w0). Plain Newton-Euler about the fixed point A."""
    Rd = axis_angle_R([0, 1, 0], -alpha)   # body z -> (-sin a, 0, cos a): C = A - R(L z + p_b) is displaced to +x
    rho = -Rd @ (m.L * EZ + m.p_b)
    IA_b = m.I_c + m.M * ((rho @ rho) * np.eye(3) - np.outer(Rd.T @ rho, Rd.T @ rho))
    IA = Rd @ IA_b @ Rd.T
    lhs = np.cross(EZ, IA @ EZ)[1]
    rhs = -m.M * m.g * np.cross(rho, EZ)[1]
    Om2 = rhs / lhs
    if Om2 <= 0:
        raise ValueError("no steady conical orbit for this geometry")
    Om = math.sqrt(Om2)
    return dict(Omega=Om, q0=R_to_quat(Rd).tolist(), w0=[0.0, 0.0, Om], rho=rho, Lz=float((IA @ (Om * EZ))[2]))   # I_A about A already contains M rho x (w x rho)


def conical_string(m, alpha):
    """B5 string (kappa = 0, k_b = 0, c = 0): wire at angle alpha from the vertical (P displaced to +x), disc tilt theta,
    uniform rotation Omega about z. Newton-Euler on the disc: force T u at P, gravity; solved for (Omega, theta)."""
    from scipy.optimize import brentq
    u = np.array([-math.sin(alpha), 0.0, math.cos(alpha)])
    P = m.A - m.L * u
    T = m.M * m.g / u[2]

    def resid(theta):
        Rd = axis_angle_R([0, 1, 0], theta)
        r = Rd @ m.p_b
        C = P - r
        Om2 = -T * u[0] / (m.M * C[0])                    # M(-Om^2 C_x) = T u_x
        Iw = Rd @ m.I_c @ Rd.T
        lhs = Om2 * np.cross(EZ, Iw @ EZ)[1]                 # w x I w
        rhs = np.cross(r, T * u)[1]
        return lhs - rhs, Om2, Rd, C
    # bracket theta between -alpha..alpha*? scan
    th = np.linspace(-1.2, 1.2, 2401)
    f = np.array([resid(x)[0] for x in th])
    idx = np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
    cands = []
    for i in idx:
        r0 = brentq(lambda x: resid(x)[0], th[i], th[i + 1], xtol=1e-17, rtol=1e-15, maxiter=200)
        _, Om2, Rd, C = resid(r0)
        if Om2 > 0:
            cands.append((abs(r0 + alpha), r0, Om2, Rd, C))
    if not cands:
        raise ValueError("no steady conical solution")
    _, th0, Om2, Rd, C = min(cands)       # the branch with the disc normal close to the wire
    Om = math.sqrt(Om2)
    Iw = Rd @ m.I_c @ Rd.T
    Lz = float((Iw @ (Om * EZ))[2] + m.M * np.cross(C - m.A, np.cross(Om * EZ, C - m.A))[2])
    return dict(Omega=Om, theta=th0, q0=R_to_quat(Rd).tolist(), w0=[0.0, 0.0, Om], u0=u.tolist(),
                udot0=np.cross(Om * EZ, u).tolist(), C=C, T=T, Lz=Lz)
