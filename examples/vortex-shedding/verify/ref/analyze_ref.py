"""Summarise role-E reference runs -> verify/out/reference.json + visuals (v001)."""
import json, os, glob, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, '..', 'out'); VIS = os.path.join(HERE, '..', 'visuals')
BENCH = dict(St=(0.295, 0.305), Cd_max=(3.22, 3.24), Cl_max=(0.99, 1.01))


def zc(t, s):
    s = s - s.mean(); i = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    tc = t[i] - s[i] * (t[i + 1] - t[i]) / (s[i + 1] - s[i]); return tc


def stats(f, nper=3):
    z = np.load(f); m = json.loads(str(z['meta'])); t, cd, cl, pv = z['t'], z['Cd'], z['Cl'], z['pv']
    tc = zc(t, cl)
    if len(tc) < nper + 2: return dict(file=os.path.basename(f), meta=m, ok=False, t_end=float(t[-1]))
    t0, t1 = tc[-nper - 1], tc[-1]; sel = (t >= t0) & (t <= t1)
    f_cl = nper / (t1 - t0)
    tcp = zc(t[sel], pv[sel]); f_pv = (len(tcp) - 1) / (tcp[-1] - tcp[0]) if len(tcp) > 2 else np.nan
    # saturation: Cl amplitude growth between last two periods
    per = np.diff(tc[-nper - 1:])
    a_last = [np.ptp(cl[(t >= tc[k]) & (t < tc[k + 1])]) / 2 for k in range(len(tc) - 3, len(tc) - 1)]
    St = f_cl * m['D'] / m['Umean']
    extra = {}
    if 'u_up5D' in z.files:
        ua = float(np.mean(z['u_up5D'][sel])) / (1 - 0.25 / 25)   # remove potential-flow deficit a^2/r^2 at 5 D
        extra = dict(U_approach_est=ua, St_Uapproach=float(St / ua), Cd_mean_Uapproach=float(cd[sel].mean() / ua ** 2),
                     Cl_amp_Uapproach=float(np.ptp(cl[sel]) / 2 / ua ** 2))
    return dict(**extra, file=os.path.basename(f), ok=True, done=m['done'], cells=m['cells'], vel_dofs=m['vel_dofs'], dt=m['dt'],
                hcyl=m['hcyl'], window=[float(t0), float(t1)], St=float(St), St_probe=float(f_pv * m['D'] / m['Umean']),
                Cd_max=float(cd[sel].max()), Cd_mean=float(cd[sel].mean()), Cl_max=float(cl[sel].max()),
                Cl_amp=float(np.ptp(cl[sel]) / 2), Cl_amp_growth_last_period=float(a_last[-1] / a_last[0] - 1),
                period_spread=float(np.ptp(per) / per.mean()), t_end=float(t[-1]))


R = {'_doc': 'Role E independent reference (FEniCSx 0.11 IPCS, Taylor-Hood P2/P1, gmsh, semi-implicit AB/CN). Not the production code.',
     'benchmark_ST2D2_published': BENCH}
st = sorted(glob.glob(os.path.join(OUT, 'ref_st2d2_L*.npz'))); pr = sorted(glob.glob(os.path.join(OUT, 'ref_prod_M*.npz')))  # M1: B v001 U_in, M1b: B v002 U_in, M1c: E-calibrated U_in (used)
R['st2d2'] = [stats(f) for f in st]
for s in R['st2d2']:
    if s['ok']:
        s['in_band'] = {k: BENCH[k][0] <= s[k] <= BENCH[k][1] for k in BENCH}
        s['rel_err_to_band_mid'] = {k: (s[k] - np.mean(BENCH[k])) / np.mean(BENCH[k]) for k in BENCH}
ok3 = [s for s in R['st2d2'] if s['ok']]
if len(ok3) >= 3:
    ext = {}
    for k in ('Cd_max', 'Cl_max', 'St'):
        a1, a2, a3 = [s[k] for s in ok3[-3:]]
        d1, d2 = a2 - a1, a3 - a2
        ext[k] = a3 + d2 * d2 / (d1 - d2) if abs(d1 - d2) > 1e-12 and d1 * d2 > 0 and abs(d2) < abs(d1) else None
    R['st2d2_aitken_extrapolated'] = ext
R['prod_Re150'] = [stats(f, nper=4) for f in pr]
wf = os.path.join(OUT, 'ref_unbounded_W40.npz')
if os.path.exists(wf):
    R['control_wide_W40_Re150'] = stats(wf, nper=4)
    R['control_wide_W40_Re150']['_doc'] = 'W = 40 D (blockage 2.5 %), plug inflow 15 D upstream, lambda = 0: solver vs Williamson 1996 St = 0.1834, 2-D Cd ~ 1.33'
B = json.load(open(os.path.join(HERE, '..', '..', 'physics', 'benchmarks.json')))['targets_production']
R['B_targets'] = dict(St=B['St'], Cd_mean=B['Cd_mean'], Cl_amp=B['Cl_amp'])
best = [s for s in R['prod_Re150'] if s['ok'] and s.get('done')]
if best:
    b = best[-1]
    na = False   # lead decision: normalise by U_ref = 0.015 m/s with U_in calibrated in the empty channel
    R['production_reference'] = dict(St=b['St_Uapproach'] if na else b['St'], Cd_mean=b['Cd_mean_Uapproach'] if na else b['Cd_mean'],
                                     Cl_amp=b['Cl_amp_Uapproach'] if na else b['Cl_amp'],
                                     normalisation='measured approach centreline speed (u at 5 D upstream / 0.99)' if na else 'U_ref',
                                     approach_normalised=dict(St=b.get('St_Uapproach'), Cd_mean=b.get('Cd_mean_Uapproach'), Cl_amp=b.get('Cl_amp_Uapproach')),
                                     U_in_calibration=dict(method='empty-channel run (same mesh/solver, lambda D/U = 0.0041123): u(x_cyl, 0) = 0.97171 U at U_in 0.90139 U -> U_in = 0.92763 U; check run gives u(x_cyl,0) = 0.99898 U (-0.10 %)', files=['ref_empty_cal1.npz', 'ref_empty_cal2.npz']),
                                     U_approach_over_U_ref=b.get('U_approach_est'), source=b['file'],
                                     setup=f"production channel scaled by D: W = 12 D, inlet 26 D upstream (plug U_in = {json.loads(str(np.load(os.path.join(OUT, b['file']))['meta']))['Uin']} U)" + ', outlet 34 D, '
                                           'lambda D/U = 0.0041123 (B decision), Re = 150, U = centreline approach speed (B def.)',
                                     vs_B={})
    pr_ = R['production_reference']; pr_['vs_B'] = {k: (pr_[k] - R['B_targets'][k]) / R['B_targets'][k] for k in ('St', 'Cd_mean', 'Cl_amp')}
json.dump(R, open(os.path.join(OUT, 'reference.json'), 'w'), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))

# visuals
fig, ax = plt.subplots(2, 2, figsize=(13, 7))
for f in st:
    z = np.load(f); m = json.loads(str(z['meta']))
    lab = f"h_cyl={m['hcyl']} ({m['cells']} cells)"
    ax[0, 0].plot(z['t'], z['Cd'], lw=.8, label=lab); ax[1, 0].plot(z['t'], z['Cl'], lw=.8, label=lab)
ax[0, 0].axhspan(*BENCH['Cd_max'], color='g', alpha=.3, label='published Cd_max band'); ax[0, 0].set_ylim(2.9, 3.35)
ax[1, 0].axhspan(*BENCH['Cl_max'], color='g', alpha=.3, label='published Cl_max band')
ax[0, 0].set_title('v001 | Schafer-Turek 2D-2 Cd(t) | verify (FEniCSx IPCS)'); ax[1, 0].set_title('v001 | 2D-2 Cl(t) | verify')
ax[0, 0].legend(fontsize=7); ax[1, 0].legend(fontsize=7); ax[1, 0].set_xlabel('t')
for f in pr + ([wf] if os.path.exists(wf) else []):
    z = np.load(f); m = json.loads(str(z['meta']))
    tt = z['t'] * 0.01 / 0.015
    ax[0, 1].plot(tt, z['Cd'], lw=.8, label=f"W={m['W']:.0f} D, U_in={m['Uin']:.3f} U, lambda={m['drag']:.4f} (Cd by U_ref)"); ax[1, 1].plot(tt, z['Cl'], lw=.8)
ax[0, 1].axhline(B['Cd_mean'], color='k', ls='--', label=f"B target Cd_mean {B['Cd_mean']:.3f}"); ax[0, 1].set_ylim(1.2, 1.7)
ax[0, 1].set_title('v001 | production channel Re 150 Cd(t) | verify'); ax[1, 1].set_title('v001 | production channel Re 150 Cl(t) | verify')
ax[1, 1].set_xlabel('t [s] (SI: D = 10 mm, U = 15 mm/s)'); ax[0, 1].legend(fontsize=7)
for a in ax.flat: a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(VIS, 'v001_reference_forces.png'), dpi=110)
