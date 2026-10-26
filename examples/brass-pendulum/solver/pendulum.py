"""Production solver for simulation 01 (brass pendulum).  Owner: dynamics implementation.

Model (planar rigid pendulum about a fixed horizontal pivot axis through P, world +Y):

    I_P * theta'' = - m g d sin(theta)                      (gravity)
                    - c_v * theta'                          (linear viscous torque)
                    - k_q * |theta'| * theta'               (quadratic aerodynamic torque, opposes motion)
                    - tau_c * tanh(theta' / omega_s)        (optional smoothed Coulomb torque)

theta is defined by position (CONTRACT): bob centre = P + L_c (sin th, 0, -cos th).

Production model id (physics/FORMULATION.md): planar_compound_sphere_rigid_massless_rod_v1:
    I_P th'' = -m g L_c sin th - c th',  I_P = m L_c^2 + 2/5 m r^2,  c = I_P omega0 / Q.
The k_q and Coulomb terms are kept available but default OFF (physics: air damping is sub-pixel).

General mass model (the production model is the case solid_sphere + massless wire):
    bob: mass m_b (given, or density * 4/3 pi r^3), centre at L_c from P,
         inertia about own centre: 0 ("point_mass") or 2/5 m_b r^2 ("solid_sphere")
    wire: uniform rod of linear density mu from P to length l_w (default L_c - r), straight and rigid
    m   = m_b + mu l_w
    d   = (m_b L_c + mu l_w^2 / 2) / m
    I_P = m_b L_c^2 + I_bob_c + mu l_w^3 / 3
  or inertia_model == "explicit": m, d, I_P given directly.

Integrator: classical explicit RK4 with a FIXED step that divides every output interval exactly
(default h = 1/3072 s = 1/32 of the 1/96 s output spacing).  Chosen because
  * the error is controlled and estimable a priori and by step halving (theoretical order 4);
  * output samples are integration nodes (no dense-output interpolation error);
  * it is deterministic and bitwise repeatable (no adaptive step selection);
  * at h = 1/3072 s, (omega_max h) ~ 1.7e-3, global error ~1e-13 rad, i.e. at round-off level.
Non-smooth terms (k_q |omega| omega, Coulomb): a step in which omega changes sign is split at the
located turning point (omega = 0) so each piece has a smooth right-hand side (order 4 retained).
The dissipated work W(t) = int_0^t P_diss dt is integrated as a third state with the same RK4 so
the energy balance E(t) - E(0) + W(t) checks the integrator, not a separate quadrature.

Outputs (simulate): theta, omega, alpha, x, z (bob centre), KE, PE (zero at theta = 0), E,
tension (force the suspension exerts on the BOB along the wire, toward P:
T = m_b (g cos th + L_c omega^2); drag acts tangentially and gravity is resolved radially),
P_diss (power removed by the dissipative torques, >= 0), W_diss.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

SOLVER_ID = "rk4-fixed-step-v1"
G_STD = 9.80665
DEFAULT_H = 1.0 / 3072.0


def _v(params: dict, key: str, default=None):
    """Read a value from a physics.json-style dict: entries are {"value": ...} or raw numbers."""
    for section in (params, params.get("parameters", {})):
        if key in section:
            e = section[key]
            if isinstance(e, dict):
                return e.get("value", default)
            return e
    return default


@dataclass(frozen=True)
class Model:
    m: float          # total moving mass [kg]
    d: float          # pivot to centre of mass [m]
    I_P: float        # moment of inertia about pivot axis [kg m^2]
    L_c: float        # pivot to bob centre [m]
    m_bob: float      # bob mass [kg]
    c_v: float        # linear viscous torque coefficient [N m s / rad]
    k_q: float        # quadratic torque coefficient [N m s^2 / rad^2]
    tau_c: float      # Coulomb torque magnitude [N m]
    omega_s: float    # Coulomb smoothing rate [rad/s]
    g: float          # [m/s^2]
    theta0: float     # [rad]
    omega0: float     # [rad/s]

    @property
    def L_eq(self) -> float:
        """Equivalent simple-pendulum length I_P / (m d)."""
        return self.I_P / (self.m * self.d)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["L_eq"] = self.L_eq
        return d


MODEL_IDS = ("planar_compound_sphere_rigid_massless_rod_v1",)


def _val(e, default=None):
    if isinstance(e, dict):
        v = e.get("value", default)
        return default if v is None else v
    return default if e is None else e


def model_from_physics_json(p: dict) -> Model:
    """Parse the physics/FORMULATION.md section 7 schema (params/physics.json, model_id
    planar_compound_sphere_rigid_massless_rod_v1).  Numeric L_c, bob_radius are required; bob_mass is
    taken if given, else 4/3 pi r^3 rho_bob; damping.c is taken if given, else I_P omega0 / Q
    (Q null / 'inf' -> conservative, validation only).  Optional (default off) extra dissipative terms:
    damping.k_q (quadratic, N m s^2), damping.tau_c (smoothed Coulomb, N m), damping.coulomb_omega_s."""
    mid = p.get("model_id")
    if mid not in MODEL_IDS:
        raise ValueError(f"model_id {mid!r} not supported (expected one of {MODEL_IDS})")
    g = float(_val(p.get("g"), G_STD))
    L_c = float(_val(p.get("L_c")))
    r = float(_val(p.get("bob_radius")))
    rho = float(_val(p.get("rho_bob"), 8500.0))
    m_b = _val(p.get("bob_mass"))
    m_b = float(m_b) if m_b is not None else rho * 4.0 / 3.0 * math.pi * r ** 3
    wire = p.get("wire", {}) or {}
    mu = 0.0
    if wire.get("mass_included"):
        d_w = float(_val(wire.get("diameter"))); rho_w = float(_val(wire.get("rho")))
        mu = rho_w * math.pi * d_w * d_w / 4.0
    l_w = _val(p.get("wire_length"))
    l_w = float(l_w) if l_w is not None else L_c - r
    if abs(l_w - (L_c - r)) > 1e-9:
        raise ValueError(f"wire_length {l_w} != L_c - r = {L_c - r}")
    m = m_b + mu * l_w
    d = (m_b * L_c + mu * l_w * l_w / 2.0) / m
    I_P = m_b * L_c * L_c + 0.4 * m_b * r * r + mu * l_w ** 3 / 3.0
    L_eq = I_P / (m * d)
    L_eq_file = _val(p.get("L_eq"))
    if L_eq_file is not None and abs(L_eq_file - L_eq) > 1e-9 * L_eq:
        raise ValueError(f"L_eq in file {L_eq_file} inconsistent with I_P/(m d) = {L_eq}")
    damp = p.get("damping", {}) or {}
    law = damp.get("law", "linear_viscous_pivot")
    if law != "linear_viscous_pivot":
        raise ValueError(f"damping.law {law!r} not supported")
    c = _val(damp.get("c"))
    if c is None:
        Q = _val(damp.get("Q"))
        if Q is None or (isinstance(Q, str) and Q.lower() in ("inf", "infinity")) or float(Q) == math.inf:
            c = 0.0
        else:
            c = I_P * math.sqrt(g / L_eq) / float(Q)
    c = float(c)
    k_q = float(_val(damp.get("k_q"), 0.0))
    tau_c = float(_val(damp.get("tau_c"), 0.0))
    om_s = float(_val(damp.get("coulomb_omega_s"), 1e-3))
    th0 = float(_val(p.get("theta_init")))
    om0 = float(_val(p.get("omega_init"), 0.0))
    for name, val in (("c", c), ("k_q", k_q), ("tau_c", tau_c)):
        if val < 0:
            raise ValueError(f"{name} = {val} < 0: negative damping (energy injection) is not allowed")
    return Model(m, d, I_P, L_c, m_b, c, k_q, tau_c, om_s, g, th0, om0)


def model_from_params(params: dict) -> Model:
    """Dispatch: schema with model_id (production, FORMULATION.md s7) or the flat legacy/test dict."""
    if "model_id" in params and "parameters" not in params:
        return model_from_physics_json(params)
    g = float(_v(params, "g_m_s2", G_STD))
    L_c = float(_v(params, "L_c_m"))
    inertia_model = _v(params, "inertia_model", "point_mass")
    r = float(_v(params, "bob_radius_m", 0.0) or 0.0)
    m_b = _v(params, "bob_mass_kg")
    if m_b is None:
        rho = float(_v(params, "bob_density_kg_m3"))
        m_b = rho * 4.0 / 3.0 * math.pi * r ** 3
    m_b = float(m_b)
    if inertia_model == "explicit":
        m = float(_v(params, "m_total_kg"))
        d = float(_v(params, "d_com_m"))
        I_P = float(_v(params, "I_P_kg_m2"))
    else:
        if inertia_model == "point_mass":
            I_bc = 0.0
        elif inertia_model == "solid_sphere":
            I_bc = 0.4 * m_b * r * r
        else:
            raise ValueError(f"unknown inertia_model {inertia_model!r}")
        mu = float(_v(params, "wire_linear_density_kg_m", 0.0) or 0.0)
        l_w = _v(params, "wire_length_m")
        l_w = float(l_w) if l_w is not None else max(L_c - r, 0.0)
        m = m_b + mu * l_w
        d = (m_b * L_c + mu * l_w * l_w / 2.0) / m
        I_P = m_b * L_c * L_c + I_bc + mu * l_w ** 3 / 3.0
    c_v = float(_v(params, "c_v_N_m_s", 0.0) or 0.0)
    k_q = float(_v(params, "k_q_N_m_s2", 0.0) or 0.0)
    tau_c = float(_v(params, "tau_c_N_m", 0.0) or 0.0)
    omega_s = float(_v(params, "coulomb_omega_s_rad_s", 1e-3) or 1e-3)
    th0 = float(_v(params, "theta0_rad"))
    om0 = float(_v(params, "omega0_rad_s", 0.0) or 0.0)
    for name, val in (("c_v", c_v), ("k_q", k_q), ("tau_c", tau_c)):
        if val < 0:
            raise ValueError(f"{name} = {val} < 0: negative damping (energy injection) is not allowed")
    if not (m > 0 and d > 0 and I_P > 0 and L_c > 0 and g > 0):
        raise ValueError("non-positive mass/length/inertia/g")
    return Model(m, d, I_P, L_c, m_b, c_v, k_q, tau_c, omega_s, g, th0, om0)


def model_from_values(*, L_c, theta0, omega0=0.0, c_v=0.0, k_q=0.0, tau_c=0.0, omega_s=1e-3,
                      g=G_STD, m_bob=1.0, r=0.0, inertia_model="point_mass", mu=0.0, l_w=None):
    p = {"L_c_m": L_c, "theta0_rad": theta0, "omega0_rad_s": omega0, "c_v_N_m_s": c_v,
         "k_q_N_m_s2": k_q, "tau_c_N_m": tau_c, "coulomb_omega_s_rad_s": omega_s, "g_m_s2": g,
         "bob_mass_kg": m_bob, "bob_radius_m": r, "inertia_model": inertia_model,
         "wire_linear_density_kg_m": mu}
    if l_w is not None:
        p["wire_length_m"] = l_w
    return model_from_params(p)


def _coeffs(M: Model):
    inv = 1.0 / M.I_P
    return M.m * M.g * M.d * inv, M.c_v * inv, M.k_q * inv, M.tau_c * inv, 1.0 / M.omega_s


def _rhs_factory(M: Model):
    a, cv, kq, tc, ws_inv = _coeffs(M)
    sin, tanh, fabs = math.sin, math.tanh, math.fabs
    cvI, kqI, tcI = M.c_v, M.k_q, M.tau_c

    if tc == 0.0:
        def acc(th, om):
            return -a * sin(th) - cv * om - kq * fabs(om) * om

        def pdiss(om):
            return cvI * om * om + kqI * fabs(om) * om * om
    else:
        def acc(th, om):
            return -a * sin(th) - cv * om - kq * fabs(om) * om - tc * tanh(om * ws_inv)

        def pdiss(om):
            return cvI * om * om + kqI * fabs(om) * om * om + tcI * om * tanh(om * ws_inv)
    return acc, pdiss


def rk4_step(acc, pdiss, th, om, w, h):
    k1t = om;              k1o = acc(th, om);                          k1w = pdiss(om)
    o2 = om + 0.5 * h * k1o
    k2t = o2;              k2o = acc(th + 0.5 * h * k1t, o2);          k2w = pdiss(o2)
    o3 = om + 0.5 * h * k2o
    k3t = o3;              k3o = acc(th + 0.5 * h * k2t, o3);          k3w = pdiss(o3)
    o4 = om + h * k3o
    k4t = o4;              k4o = acc(th + h * k3t, o4);                k4w = pdiss(o4)
    h6 = h / 6.0
    return (th + h6 * (k1t + 2 * k2t + 2 * k3t + k4t),
            om + h6 * (k1o + 2 * k2o + 2 * k3o + k4o),
            w + h6 * (k1w + 2 * k2w + 2 * k3w + k4w))


def integrate(M: Model, t_out, h: float = DEFAULT_H):
    """Return theta, omega, W_diss at t_out (t_out[0] must be 0, strictly increasing)."""
    t_out = np.asarray(t_out, dtype=float)
    if t_out[0] != 0.0 or np.any(np.diff(t_out) <= 0):
        raise ValueError("t_out must start at 0 and increase strictly")
    acc, pdiss = _rhs_factory(M)
    nonsmooth = (M.k_q != 0.0) or (M.tau_c != 0.0)
    n = len(t_out)
    TH = np.empty(n); OM = np.empty(n); W = np.empty(n)
    th, om, w = M.theta0, M.omega0, 0.0
    TH[0], OM[0], W[0] = th, om, w
    for i in range(1, n):
        dt = t_out[i] - t_out[i - 1]
        k = max(1, int(math.ceil(dt / h - 1e-9)))
        hs = dt / k
        for _ in range(k):
            th1, om1, w1 = rk4_step(acc, pdiss, th, om, w, hs)
            if nonsmooth and om * om1 < 0.0:
                th1, om1, w1 = _step_through_turning_point(acc, pdiss, th, om, w, hs)
            th, om, w = th1, om1, w1
        TH[i], OM[i], W[i] = th, om, w
    return TH, OM, W


def _step_through_turning_point(acc, pdiss, th, om, w, h):
    """omega changes sign inside [0, h]: the |omega| omega and Coulomb terms are not smooth there,
    which would reduce RK4 to ~3rd order.  Locate s with omega(s) = 0 (RK4 sub-step from the node,
    Illinois/regula-falsi on s), step to it, then step the remainder h - s.  Both pieces have a
    smooth right-hand side, so 4th order is retained."""
    a, fa = 0.0, om
    b, fb = h, rk4_step(acc, pdiss, th, om, w, h)[1]
    side = 0
    s = h
    for _ in range(100):
        s = (a * fb - b * fa) / (fb - fa)
        fs = rk4_step(acc, pdiss, th, om, w, s)[1]
        if fs == 0.0 or (b - a) < 1e-17:
            break
        if fs * fb < 0:
            a, fa = b, fb
            b, fb = s, fs
            side = 0
        else:
            b, fb = s, fs
            if side == -1:
                fa *= 0.5
            side = -1
        if abs(fs) < 1e-16:
            break
    t1, o1, w1 = rk4_step(acc, pdiss, th, om, w, s)
    return rk4_step(acc, pdiss, t1, o1, w1, h - s)


def derived(M: Model, theta, omega):
    theta = np.asarray(theta, float); omega = np.asarray(omega, float)
    a, cv, kq, tc, ws_inv = _coeffs(M)
    alpha = -a * np.sin(theta) - cv * omega - kq * np.abs(omega) * omega
    P = M.c_v * omega ** 2 + M.k_q * np.abs(omega) ** 3
    if tc != 0.0:
        alpha = alpha - tc * np.tanh(omega * ws_inv)
        P = P + M.tau_c * omega * np.tanh(omega * ws_inv)
    KE = 0.5 * M.I_P * omega ** 2
    PE = M.m * M.g * M.d * (1.0 - np.cos(theta))
    x = M.L_c * np.sin(theta)
    z = -M.L_c * np.cos(theta)
    T = M.m_bob * (M.g * np.cos(theta) + M.L_c * omega ** 2)
    return dict(alpha=alpha, x=x, z=z, KE=KE, PE=PE, E=KE + PE, tension=T, P_diss=P)


def simulate(params, t_out, h: float = DEFAULT_H) -> dict:
    """params: physics.json dict or Model.  Returns dict of numpy arrays (SI)."""
    M = params if isinstance(params, Model) else model_from_params(params)
    t_out = np.asarray(t_out, dtype=float)
    th, om, W = integrate(M, t_out, h)
    out = dict(t=t_out, theta=th, omega=om, W_diss=W)
    out.update(derived(M, th, om))
    out["model"] = M
    return out


# ---- analytic references used by self-checks -------------------------------------------------
def exact_period(M: Model, theta_amp: float) -> float:
    """Exact period of the undamped rigid pendulum: 4 sqrt(I_P/(m g d)) K(k), k = sin(amp/2)."""
    from scipy.special import ellipk
    k = math.sin(theta_amp / 2.0)
    return 4.0 * math.sqrt(M.I_P / (M.m * M.g * M.d)) * float(ellipk(k * k))


def crossing_times(M: Model, t_grid, theta, omega, h: float = DEFAULT_H, direction=+1):
    """Times where theta crosses 0 with sign(omega) == direction, refined by Newton iteration
    on an RK4 sub-step from the preceding grid node (local error O(s^5))."""
    acc, pdiss = _rhs_factory(M)
    out = []
    for i in range(len(t_grid) - 1):
        if theta[i] * theta[i + 1] < 0 and np.sign(omega[i]) == direction:
            s = -theta[i] / omega[i]
            for _ in range(30):
                k = max(1, int(math.ceil(abs(s) / h)))
                th, om, _w = theta[i], omega[i], 0.0
                for _ in range(k):
                    th, om, _w = rk4_step(acc, pdiss, th, om, _w, s / k)
                ds = -th / om
                s += ds
                if abs(ds) < 1e-15:
                    break
            out.append(t_grid[i] + s)
    return np.array(out)
