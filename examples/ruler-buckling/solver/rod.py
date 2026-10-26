"""Geometrically exact planar Kirchhoff rod (inextensible, unshearable) for the ruler-buckling clip.

Discretisation (discrete elastica / variational finite differences):
  flexible length Lf split into N straight links of length h; generalized coordinates = link angles theta_i
  (theta measured from vertical, positive towards the lateral axis y).  Link i represents the rod at s=(i-1/2)h.
  Bending energy  U = sum_j k_j/2 (dtheta_j - dtheta0_j)^2 ,  k_1 = 2EI/h (clamp: half distance), k_j = EI/h.
  -> second-order accurate in h for statics and dynamics (verified: see run_all.py convergence study).
Node masses lumped (rho*A*h, h/2 at the tip node). A rigid tip body (top strip piece held by the platform tabs,
platform plate, tabs, screws, peg, stacked slotted weights) is attached at the top of the flexible part; its pieces
lie on the rod tangent at offsets e_k above the attachment point.  Its dead load acts at its centre of mass.
Kinematics: x_k = x_0 + sum_{i<=k} l_i t(phi_i),  t(phi) = (sin phi, cos phi)  in (y, z).
Dynamics: exact point-mass Lagrangian (mass matrix M = l_i l_j cos(phi_i-phi_j) S_max(i,j), Coriolis term
l_i l_j sin(phi_i-phi_j) S_max w_j^2) + rotational inertia of the tip body; generalized-alpha time integration.
Damping: stiffness-proportional joint dashpots (eta*K_b) + quadratic air drag on the strip (flat plate).
"""
import numpy as np

G0 = 9.81


class Rod:
    def __init__(self, P, N, tip_pieces=None, self_weight=True, imperfection_m=None, drag=True, eta=None):
        self.P = P
        s = P["strip"]
        self.E, self.rho, self.b, self.t = s["E_Pa"], s["rho_kg_m3"], s["width_m"], s["thickness_m"]
        self.EI = self.E * self.b * self.t ** 3 / 12.0
        self.A = self.b * self.t
        self.Lf = s["free_length_m"] - P["tip_body"]["rigid_top_strip_m"]
        self.N = N
        self.h = self.Lf / N
        h = self.h
        self.k = np.full(N, self.EI / h)
        self.k[0] = 2.0 * self.EI / h
        # difference matrix D: kappa = D theta  (kappa_1 = theta_1 - 0)
        D = np.eye(N) - np.eye(N, k=-1)
        self.D = D
        self.Kb = D.T @ (self.k[:, None] * D)
        mnode = np.full(N, self.rho * self.A * h if self_weight else 0.0)
        mnode[-1] *= 0.5
        self.mnode_inertial = np.full(N, self.rho * self.A * h)
        self.mnode_inertial[-1] *= 0.5
        self.mnode_grav = mnode
        self.g = G0
        # imperfection: stress-free shape y0 = d0 (1 - cos(pi s / 2Lf))  -> theta0 = y0'
        d0 = P["imperfection"]["tip_offset_m"] if imperfection_m is None else imperfection_m
        sm = (np.arange(N) + 0.5) * h
        self.theta0 = d0 * np.pi / (2 * self.Lf) * np.sin(np.pi * sm / (2 * self.Lf))
        self.kappa0 = D @ self.theta0
        self.tip_pieces = tip_pieces  # callable t-> (m_b_inertial, m_b_grav, e, J) or fixed tuple
        self.drag = drag
        self.cair = 0.5 * P["damping"]["rho_air"] * P["damping"]["Cd_plate"] * self.b if drag else 0.0
        self.eta = P["damping"]["eta_s"] if eta is None else eta
        self.alphaM = P["damping"].get("alpha_M_per_s", 0.0) if eta is None else 0.0

    # ---------- tip body ----------
    def tip(self, lam):
        """lam: either time-like state passed to tip_pieces. returns m_inertial, m_grav, e, J."""
        return self.tip_pieces(lam)

    # ---------- kinematic helpers ----------
    def _ext(self, th, lam, inertial=False):
        tp = self.tip(lam)
        mi, mg, e, J = tp[:4]
        if inertial and len(tp) > 4:
            e = tp[4]
        phi = np.append(th, th[-1])
        l = np.append(np.full(self.N, self.h), e)
        Mi = np.append(self.mnode_inertial, mi)
        Mg = np.append(self.mnode_grav, mg)
        Si = np.cumsum(Mi[::-1])[::-1]
        Sg = np.cumsum(Mg[::-1])[::-1]
        return phi, l, Si, Sg, J, mi, mg, e

    def _PT(self, vphi):  # P^T for vectors (N+1) -> N
        out = vphi[:-1].copy()
        out[-1] += vphi[-1]
        return out

    def _PTMP(self, Mphi):
        M = Mphi[:-1, :-1].copy()
        M[-1, :] += Mphi[-1, :-1]
        M[:, -1] += Mphi[:-1, -1]
        M[-1, -1] += Mphi[-1, -1]
        return M

    def mass(self, th, lam):
        phi, l, Si, Sg, J, *_ = self._ext(th, lam, inertial=True)
        n = len(phi)
        idx = np.arange(n)
        Smax = Si[np.maximum.outer(idx, idx)]
        dphi = phi[:, None] - phi[None, :]
        Mphi = np.outer(l, l) * np.cos(dphi) * Smax
        M = self._PTMP(Mphi)
        M[-1, -1] += J
        return M, (phi, l, Si, dphi, Smax)

    def coriolis(self, th, w, lam, cache=None):
        phi, l, Si, dphi, Smax = cache
        wphi = np.append(w, w[-1])
        cphi = (np.outer(l, l) * np.sin(dphi) * Smax) @ (wphi ** 2)
        return self._PT(cphi)

    def positions(self, th, lam):
        """node positions (N+1 nodes incl. clamp node 0) in (y,z) relative to clamp; + tip CoM"""
        phi, l, *_rest = self._ext(th, lam)
        y = np.concatenate([[0.0], np.cumsum(l * np.sin(phi))])
        z = np.concatenate([[0.0], np.cumsum(l * np.cos(phi))])
        return y, z  # y[N] = top of flexible part, y[N+1] = tip-body CoM

    def grad_U(self, th):
        return self.D.T @ (self.k * (self.D @ th - self.kappa0))

    def U(self, th):
        r = self.D @ th - self.kappa0
        return 0.5 * np.sum(self.k * r * r)

    def V(self, th, lam):
        phi, l, Si, Sg, *_ = self._ext(th, lam)
        _, mg_all = None, None
        Mg = np.append(self.mnode_grav, self.tip(lam)[1])
        z = np.cumsum(l * np.cos(phi))
        return self.g * np.sum(Mg * z)

    def grad_V(self, th, lam):
        phi, l, Si, Sg, *_ = self._ext(th, lam)
        return self._PT(-self.g * l * np.sin(phi) * Sg)

    def hess_V(self, th, lam):
        phi, l, Si, Sg, *_ = self._ext(th, lam)
        d = -self.g * l * np.cos(phi) * Sg
        H = np.diag(d[:-1])
        H[-1, -1] += d[-1]
        return H

    def drag_force(self, th, w, lam):
        if not self.drag:
            return np.zeros(self.N)
        h = self.h
        tp = np.stack([np.cos(th), -np.sin(th)], 1)  # t'(theta)
        v = np.cumsum(h * tp * w[:, None], 0)  # node velocities (nodes 1..N)
        vn = np.sum(v * tp, 1)  # normal component (normal ~ t' of the link below the node)
        F = -(self.cair * h * np.abs(vn) * vn)[:, None] * tp
        F[-1] *= 0.5
        # generalized: Q_i = sum_{k>=i} F_k . h t'_i
        Fc = np.cumsum(F[::-1], 0)[::-1]
        return -h * np.sum(Fc * tp, 1)  # returned as resisting "force" on LHS (positive = opposing)

    def static_residual(self, th, lam):
        return self.grad_U(th) + self.grad_V(th, lam)

    def static_jac(self, th, lam):
        return self.Kb + self.hess_V(th, lam)

    def energy(self, th, w, lam):
        M, _ = self.mass(th, lam)
        return 0.5 * w @ M @ w, self.U(th), self.V(th, lam)


# ------------------------------------------------------------------ static solvers
def newton_static(rod, th, lam, tol=1e-11, maxit=50):
    for it in range(maxit):
        r = rod.static_residual(th, lam)
        if np.linalg.norm(r) < tol * max(1.0, rod.EI):
            return th, True, it
        th = th - np.linalg.solve(rod.static_jac(th, lam), r)
    return th, np.linalg.norm(rod.static_residual(th, lam)) < 1e-8, maxit


def arclength_path(rod, lam0, lam_max, ds=0.05, max_steps=4000, scale_th=None, th_init=None):
    """Pseudo-arclength continuation in (theta, lam). Returns list of (lam, theta)."""
    N = rod.N
    sc = scale_th if scale_th is not None else 1.0
    th = np.zeros(N) if th_init is None else th_init.copy()
    th, ok, _ = newton_static(rod, th, lam0)
    path = [(lam0, th.copy())]
    # initial tangent: d theta/d lam from J dth = -dR/dlam
    def dRdl(th, lam, eps=1e-6):
        return (rod.static_residual(th, lam + eps) - rod.static_residual(th, lam - eps)) / (2 * eps)
    lam = lam0
    tvec = np.append(-np.linalg.solve(rod.static_jac(th, lam), dRdl(th, lam)) * sc, 1.0)
    tvec /= np.linalg.norm(tvec)
    step = ds
    for k in range(max_steps):
        if lam >= lam_max:
            break
        x0 = np.append(th * sc, lam)
        for attempt in range(8):
            x = x0 + step * tvec
            conv = False
            for it in range(30):
                thx, lx = x[:-1] / sc, x[-1]
                R = rod.static_residual(thx, lx)
                g = tvec @ (x - x0) - step
                Jm = np.zeros((N + 1, N + 1))
                Jm[:N, :N] = rod.static_jac(thx, lx) / sc
                Jm[:N, N] = dRdl(thx, lx)
                Jm[N, :] = tvec
                F = np.append(R, g)
                if np.linalg.norm(R) < 1e-10 * max(1, rod.EI) and abs(g) < 1e-12:
                    conv = True
                    break
                x = x - np.linalg.solve(Jm, F)
            if conv:
                break
            step *= 0.5
        if not conv:
            raise RuntimeError("arclength failed at lam=%g" % lam)
        newt = x - x0
        newt /= np.linalg.norm(newt)
        tvec = newt
        th, lam = x[:-1] / sc, x[-1]
        path.append((lam, th.copy()))
        if it < 4:
            step = min(step * 1.3, ds * 4)
    return path


# ------------------------------------------------------------------ dynamics (generalized-alpha)
def gen_alpha_params(rho_inf):
    am = (2 * rho_inf - 1) / (rho_inf + 1)
    af = rho_inf / (rho_inf + 1)
    g = 0.5 - am + af
    b = 0.25 * (1 - am + af) ** 2
    return am, af, g, b


def dyn_rhs_terms(rod, th, w, lam):
    M, cache = rod.mass(th, lam)
    f = rod.coriolis(th, w, lam, cache) + rod.grad_U(th) + rod.grad_V(th, lam) + rod.eta * (rod.Kb @ w) + rod.alphaM * (M @ w) \
        + rod.drag_force(th, w, lam)
    return M, f


def integrate(rod, t_end, dt, out_times, rho_inf=0.8, th_init=None, lamfun=None, tol=1e-9, record_energy=True):
    """lamfun(t) -> argument passed to rod.tip(). Returns dict with states at out_times and energy history."""
    am, af, gm, bt = gen_alpha_params(rho_inf)
    N = rod.N
    th = np.zeros(N) if th_init is None else th_init.copy()
    w = np.zeros(N)
    t = 0.0
    M, f = dyn_rhs_terms(rod, th, w, lamfun(t))
    a = np.linalg.solve(M, -f)
    nsteps = int(round(t_end / dt))
    outs = {"t": [], "th": [], "w": []}
    oi = 0
    out_times = np.asarray(out_times)
    hist = {"t": [0.0], "T": [], "U": [], "V": [], "Pext": [], "Pdiss": [], "iters": []}

    def ener(th, w, t):
        T, U, V = rod.energy(th, w, lamfun(t))
        return T, U, V

    def powers(th, w, t, eps=1e-5):
        # explicit-time power: dV/dt|_q + 1/2 w^T dM/dt|_q w ; dissipation: w.(eta Kb w + drag)
        Vp = rod.V(th, lamfun(t + eps)); Vm = rod.V(th, lamfun(max(t - eps, 0)))
        Mp, _ = rod.mass(th, lamfun(t + eps)); Mm, _ = rod.mass(th, lamfun(max(t - eps, 0)))
        de = (t + eps) - max(t - eps, 0)
        pe = (Vp - Vm) / de + 0.5 * w @ ((Mp - Mm) / de) @ w
        Mc, _ = rod.mass(th, lamfun(t))
        pd = w @ (rod.eta * (rod.Kb @ w) + rod.alphaM * (Mc @ w) + rod.drag_force(th, w, lamfun(t)))
        return pe, pd

    if record_energy:
        T, U, V = ener(th, w, t); hist["T"].append(T); hist["U"].append(U); hist["V"].append(V)
        pe, pd = powers(th, w, t); hist["Pext"].append(pe); hist["Pdiss"].append(pd)

    def emit(t_target, thA, wA, aA, thB, wB, aB, tA):
        # cubic Hermite interpolation of theta between steps
        s = (t_target - tA) / dt
        h00 = 2 * s**3 - 3 * s**2 + 1; h10 = s**3 - 2 * s**2 + s; h01 = -2 * s**3 + 3 * s**2; h11 = s**3 - s**2
        thi = h00 * thA + h10 * dt * wA + h01 * thB + h11 * dt * wB
        wi = (1 - s) * wA + s * wB
        outs["t"].append(t_target); outs["th"].append(thi); outs["w"].append(wi)

    while oi < len(out_times) and out_times[oi] <= 1e-12:
        outs["t"].append(0.0); outs["th"].append(th.copy()); outs["w"].append(w.copy()); oi += 1

    for n in range(nsteps):
        t1 = t + dt
        tm = t + (1 - af) * dt
        lam_m = lamfun(tm)
        th1 = th + dt * w + 0.5 * dt * dt * a  # predictor
        it_used = 0
        for it in range(30):
            a1 = (th1 - th - dt * w) / (bt * dt * dt) - (0.5 / bt - 1) * a
            w1 = w + dt * ((1 - gm) * a + gm * a1)
            thm = (1 - af) * th1 + af * th
            wm = (1 - af) * w1 + af * w
            amv = (1 - am) * a1 + am * a
            M, f = dyn_rhs_terms(rod, thm, wm, lam_m)
            R = M @ amv + f
            Kt = rod.Kb + rod.hess_V(thm, lam_m)
            Jt = (1 - am) / (bt * dt * dt) * M + (1 - af) * (gm / (bt * dt) * (rod.eta * rod.Kb + rod.alphaM * M) + Kt)
            d = np.linalg.solve(Jt, -R)
            th1 = th1 + d
            it_used = it + 1
            if np.linalg.norm(d) < tol:
                break
        a1 = (th1 - th - dt * w) / (bt * dt * dt) - (0.5 / bt - 1) * a
        w1 = w + dt * ((1 - gm) * a + gm * a1)
        while oi < len(out_times) and out_times[oi] <= t1 + 1e-12:
            emit(out_times[oi], th, w, a, th1, w1, a1, t)
            oi += 1
        th, w, a, t = th1, w1, a1, t1
        if record_energy:
            T, U, V = ener(th, w, t)
            hist["t"].append(t); hist["T"].append(T); hist["U"].append(U); hist["V"].append(V)
            pe, pd = powers(th, w, t); hist["Pext"].append(pe); hist["Pdiss"].append(pd)
            hist["iters"].append(it_used)
    for k in outs:
        outs[k] = np.array(outs[k])
    for k in hist:
        hist[k] = np.array(hist[k])
    return outs, hist


def angle_control_path(rod, alphas, lam_guess, th_guess, dRdl_eps=1e-6, tol=1e-10, lam_stop=np.inf, scale_first=False):
    """Path following with the tip-link angle as control parameter (works through the bifurcation of the
    perfect column and along the imperfect path): unknowns (theta, lam), constraint theta_N = alpha."""
    N = rod.N
    th, lam = th_guess.copy(), lam_guess
    path = []
    for al in alphas:
        th = th * (al / th[-1]) if scale_first and len(path) == 0 else th
        for it in range(60):
            R = rod.static_residual(th, lam)
            dR = (rod.static_residual(th, lam + dRdl_eps) - rod.static_residual(th, lam - dRdl_eps)) / (2 * dRdl_eps)
            Jm = np.zeros((N + 1, N + 1))
            Jm[:N, :N] = rod.static_jac(th, lam); Jm[:N, N] = dR; Jm[N, N - 1] = 1.0
            F = np.append(R, th[-1] - al)
            if np.linalg.norm(R) < tol * max(1, rod.EI) and abs(F[-1]) < 1e-13:
                break
            try:
                d = np.linalg.solve(Jm, -F)
            except np.linalg.LinAlgError:
                raise RuntimeError("singular at alpha=%g lam=%g it=%d |dR|=%g" % (al, lam, it, np.linalg.norm(dR)))
            th = th + d[:N]; lam = lam + d[N]
        else:
            raise RuntimeError("angle control failed at alpha=%g" % al)
        path.append((lam, th.copy()))
        if lam > lam_stop:
            break
    return path
