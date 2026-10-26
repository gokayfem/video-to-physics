"""Representative-case sanity numbers for the brass pendulum formulation (physics/ role).

Not production code. Pure closed forms + one ODE cross-check. Run:
  ROOT/.venv/bin/python physics/sanity_calc.py  > physics/sanity_calc_output.json
All SI. Assumed inputs are labelled; nothing here is a fit to the clip.
"""
import json, math, sys
import numpy as np
from scipy.special import ellipk
from scipy.integrate import solve_ivp, quad

g = 9.80665                      # contract, assumed
# --- air: U.S. Standard Atmosphere 1976 formulas at sea-level pressure, 20 C ---
T_air = 293.15; p_air = 101325.0; R_air = 287.05287
rho_air = p_air / (R_air * T_air)                       # ideal gas
mu_air = 1.458e-6 * T_air**1.5 / (T_air + 110.4)        # Sutherland (USSA-1976)
nu_air = mu_air / rho_air
rho_brass = 8500.0               # assumed (C26000 ~8530, C36000 ~8500; range 8400-8730)
rho_steel = 7850.0; E_steel = 200e9   # assumed (190-210 GPa)

def morrison_cd(Re):
    a = 24/Re
    b = 2.6*(Re/5.0)/(1+(Re/5.0)**1.52)
    c = 0.411*(Re/2.63e5)**-7.94/(1+(Re/2.63e5)**-8.0)
    d = 0.25*(Re/1e6)/(1+Re/1e6)
    return a+b+c+d

def cyl_cd(Re):  # White (2006) eq. 7.? style fit for infinite cylinder, 1<Re<2e5
    return 1 + 10.0/Re**(2/3)

def case(T_meas, th0_deg, r, d_wire, label):
    th0 = math.radians(th0_deg)
    k = math.sin(th0/2)
    fac = 2*ellipk(k*k)/math.pi          # T/T0 ; scipy ellipk takes m=k^2
    T0 = T_meas / fac
    W0 = 2*math.pi/T0
    L_eq = g / W0**2
    # compound sphere, massless rigid wire: L_eq = Lc + (2/5) r^2 / Lc
    Lc = (L_eq + math.sqrt(L_eq**2 - 1.6*r*r))/2
    l_wire = Lc - r
    L_simple_err = (L_eq - Lc)/Lc          # error if one treated Lc as simple length
    m = 4/3*math.pi*r**3*rho_brass
    I = m*Lc**2 + 0.4*m*r*r
    # wire-mass correction (uniform wire from pivot to bob top)
    A_w = math.pi*d_wire**2/4
    m_w = rho_steel*A_w*l_wire
    I_w = I + m_w*l_wire**2/3
    M_w = m*Lc + m_w*l_wire/2
    L_eq_w_sameLc = I_w/((m+m_w)*(M_w/(m+m_w)))   # = I_w/M_w
    dLeq_wire_rel = L_eq_w_sameLc/L_eq - 1
    # kinematics
    wmax = math.sqrt(2*m*g*Lc*(1-math.cos(th0))/I)
    vmax = Lc*wmax
    Re = vmax*2*r/nu_air
    Cd = morrison_cd(Re)
    KC = vmax*T_meas/(2*r)
    delta_stokes = math.sqrt(2*nu_air/W0)
    # tension (axial wire force on bob), massless wire
    Tmax = m*(g*1 + Lc*wmax**2)
    Tmin = m*g*math.cos(th0)
    Tstat = m*g
    shear_max = m*g*math.sin(th0)*(0.4*m*r*r/I)
    # wire stretch
    strain_stat = Tstat/(E_steel*A_w)
    stretch_stat = strain_stat*l_wire
    stretch_dyn = (Tmax-Tmin)/(E_steel*A_w)*l_wire
    k_ax = E_steel*A_w/l_wire
    f_axial = math.sqrt(k_ax/m)/(2*math.pi)
    stress = Tmax/A_w
    # buoyancy & added mass (potential-flow added mass 1/2 rho V)
    V = 4/3*math.pi*r**3
    buoy_rel = rho_air/rho_brass                     # reduces m g d by this fraction
    addm_rel = 0.5*rho_air*V*Lc**2/I                 # increases I by this fraction
    dT_rel_air = 0.5*(buoy_rel + addm_rel)           # period lengthening, relative
    # damping estimates (per cycle fractional amplitude loss, small-damping)
    A_proj = math.pi*r*r
    # (i) quadratic sphere drag, SHM approx: dA/A per cycle = (4/3) rho Cd A Lc^3 th0 / I
    dq_sphere = 4/3*rho_air*Cd*A_proj*Lc**3*th0/I
    # (ii) quadratic wire drag (cylinder, Cd at mid-span Re)
    Re_w = 0.5*vmax*d_wire/nu_air
    Cd_w = cyl_cd(Re_w)
    # torque = 1/2 rho Cd d w|w| * int_0^l s^3 ds = 1/2 rho Cd d l^4/4 ; equivalent coefficient
    Cw = 0.5*rho_air*Cd_w*d_wire*l_wire**4/4
    Cs = 0.5*rho_air*Cd*A_proj*Lc**3
    dq_wire = dq_sphere*Cw/Cs
    # (iii) Stokes oscillating-sphere linear drag (Stokes 1851; Landau-Lifshitz §24): F = 6 pi mu r (1 + r/delta) U
    c_stokes = 6*math.pi*mu_air*r*(1+r/delta_stokes)*Lc**2   # torque coeff, N m s
    gamma_st = c_stokes/(2*I)
    dl_stokes = gamma_st*T_meas                               # log decrement per cycle
    # combined air-only estimate (linear+quadratic, both are rough)
    dA_air = dq_sphere + dq_wire + dl_stokes
    Q_air = math.pi/dA_air
    # Coulomb pivot: amplitude loss per cycle = 4 tau_f/(m g Lc) ; give tau_f for a 1% loss/cycle
    tau_1pct = 0.01*th0*m*g*Lc/4
    # 8-cycle amplitude ratios for range of Q
    Qs = [100, 300, 1000, 3000, round(Q_air)]
    ratios = {str(q): math.exp(-math.pi*8/q) for q in Qs}
    # --- ODE cross-check: conservative period and quadratic drag decay ---
    def rhs(t, y, cq=0.0):
        th, w = y
        return [w, -(m*g*Lc*math.sin(th) + cq*w*abs(w))/I]
    def zero_cross(t, y, *a): return y[1]
    zero_cross.direction = 1
    sol = solve_ivp(rhs, [0, 10*T_meas], [th0, 0.0], method='DOP853', rtol=1e-12, atol=1e-14,
                    events=zero_cross)
    ev = sol.t_events[0]
    T_ode = np.diff(ev).mean() if len(ev) > 1 else float('nan')
    sol2 = solve_ivp(rhs, [0, 8*T_meas+0.01], [th0, 0.0], args=(Cs+Cw,), method='DOP853', rtol=1e-11, atol=1e-14,
                     dense_output=True, events=zero_cross)
    # amplitude after 8 cycles (max of theta near t=8T)
    tt = np.linspace(7.5*T_meas, 8*T_meas+0.01, 20001)
    A8 = sol2.sol(tt)[0].max()
    return dict(
        label=label,
        inputs=dict(T_meas_s=T_meas, theta0_deg=th0_deg, r_m=r, d_wire_m=d_wire, rho_brass=rho_brass,
                    rho_air=rho_air, mu_air=mu_air, nu_air=nu_air, g=g, E_steel=E_steel),
        finite_amp_factor_T_over_T0=fac, T0_small_angle_s=T0, L_eq_m=L_eq,
        L_eq_if_amplitude_ignored_m=g*(T_meas/2/math.pi)**2,
        Lc_m=Lc, l_wire_m=l_wire, Lc_over_r=Lc/r, rel_err_if_Lc_treated_as_simple=L_simple_err,
        bob_mass_kg=m, I_P_kgm2=I, wire_mass_kg=m_w, rel_change_Leq_from_wire_mass_at_fixed_Lc=dLeq_wire_rel,
        omega_max_rad_s=wmax, v_max_m_s=vmax, Re_sphere_max=Re, Cd_morrison_at_Re_max=Cd,
        KC_number=KC, stokes_layer_m=delta_stokes,
        tension_static_N=Tstat, tension_min_N=Tmin, tension_max_N=Tmax, wire_shear_max_N=shear_max,
        wire_stress_max_Pa=stress, wire_strain_static=strain_stat, wire_stretch_static_m=stretch_stat,
        wire_stretch_dynamic_variation_m=stretch_dyn, axial_mode_freq_Hz=f_axial,
        period_shift_rel_buoyancy_addedmass=dT_rel_air,
        dA_over_A_per_cycle_quadratic_sphere=dq_sphere, dA_over_A_per_cycle_quadratic_wire=dq_wire,
        Re_wire_mid=Re_w, Cd_wire=Cd_w,
        log_decrement_stokes_linear=dl_stokes, dA_over_A_per_cycle_air_total=dA_air, Q_air_only=Q_air,
        amp_ratio_after_8_cycles_air_only=math.exp(-8*dA_air),
        amp_after_8_cycles_ODE_quadratic_only_deg=math.degrees(A8),
        coulomb_torque_for_1pct_per_cycle_Nm=tau_1pct,
        amp_ratio_after_8_cycles_by_Q=ratios,
        ode_conservative_period_s=T_ode, ode_vs_elliptic_rel=(T_ode/T_meas - 1),
    )

out = [case(1.25, 6.0, 0.015, 0.3e-3, "representative: T=1.25 s, 6 deg, r=1.5 cm, wire 0.3 mm"),
       case(1.25, 4.0, 0.015, 0.3e-3, "prompted amplitude 4 deg"),
       case(1.25, 6.0, 0.030, 0.3e-3, "image-ratio case Lc/r~13 -> r~3 cm (rough sheet estimate)")]
json.dump(out, sys.stdout, indent=1)
