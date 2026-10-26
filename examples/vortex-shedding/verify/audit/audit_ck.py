"""Role E checkpoint audit extras: phase anchor, dye front at t=0, release consistency (periodic back-mapping), L10 vs L11.
usage: python audit_ck.py <solver_out_dir> <label>"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checks'))
import vlib
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['axes.titlesize'] = 9
od, label = sys.argv[1], sys.argv[2]
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
F = vlib.load(od); G = vlib.GEOM; D, xc = G['D'], G['xc']; U = F['U']
x, y, t = F['x'], F['y'], F['t']; dx = x[1] - x[0]
meta = F['meta']; A = {}
# ---- phase anchor: Gamma+/- in x in [xc+1D, xc+2D], |y|<1D (own vorticity from u,v)
mx = (x > xc + D) & (x < xc + 2 * D); my = np.abs(y) < D
om = np.gradient(F['v'], dx, axis=2) - np.gradient(F['u'], y[1] - y[0], axis=1)
sub = om[:, my][:, :, mx]
Gp = np.clip(sub, 0, None).sum(axis=(1, 2)) * dx * dx; Gm = np.clip(sub, None, 0).sum(axis=(1, 2)) * dx * dx
def peaks(s):
    i = [k for k in range(1, len(s) - 1) if s[k] >= s[k - 1] and s[k] > s[k + 1]]
    out = []
    for k in i:   # parabolic refinement
        d = 0.5 * (s[k - 1] - s[k + 1]) / (s[k - 1] - 2 * s[k] + s[k + 1]); out.append(float(t[k] + d * (t[1] - t[0])))
    return out
pp, pm = peaks(Gp), peaks(-Gm)
T = 1 / (vlib.fft_freq(F['tp'], F['Cl']))
pp = [p for p in pp if Gp[int(round(p * 24))] > 0.5 * Gp.max()]; pm = [p for p in pm if -Gm[int(round(p * 24))] > 0.5 * (-Gm).max()]
A['phase_anchor'] = dict(Gamma_plus_peaks_s=pp, Gamma_minus_peaks_s=pm, first_plus=pp[0] if pp else None, first_minus=pm[0] if pm else None,
                         target='Gamma+ peak at 0.75 +- 0.08 s, lower side (omega>0) first',
                         passed=bool(pp and abs(pp[0] - 0.75) <= 0.08 and (not pm or pp[0] < pm[0])),
                         meta_claim=meta.get('phase', {}).get('Gamma_plus_peaks_s'))
# ---- dye front at t = 0
c0 = vlib.getdye(F, 0); xs, ys, px = vlib.dye_grid(c0.shape); s0 = c0.max(axis=2)
fr = {}
for thr in (0.02, 0.1, 0.25, 0.5):
    cols = np.where((s0 > thr).any(axis=0))[0]; fr[str(thr)] = float((xs[cols.max()] - xc) / D) if len(cols) else None
# per needle streak front (c>0.25*core) along each needle's row band
per = {}
for yn in G['needles_y']:
    rows = np.abs(ys - yn) < 0.004
    cols = np.where((s0[rows] > 0.25).any(axis=0))[0]; per[f'{yn*1e3:+.0f}mm'] = float((xs[cols.max()] - xc) / D) if len(cols) else None
lead = max(v for v in per.values() if v is not None)
A['dye_front_t0'] = dict(front_over_D_by_threshold=fr, per_needle_front_c025_over_D=per, target_over_D=0.6,
                         passed=bool(abs(fr['0.25'] - 0.6) <= 0.3), note='(x_front - x_cyl)/D; tolerance +-0.3 D (3 mm) on the c > 0.25 front')
# ---- release consistency: advect particles from the needle tips from t_rel to 0 using the periodic flow (u(t<0) = u(t + kT))
trel = meta.get('release_start_s'); res = {}
if trel is not None:
    Tn = T
    def fidx(tt):
        while tt < 0: tt += Tn
        return tt * 24.0
    Ys = np.array(G['needles_y']); X = np.full_like(Ys, G['x_tip']); Y = Ys.copy(); h = 1 / 96.0; tt = trel
    while tt < -1e-9:
        u1, v1 = vlib.interp_vel(F, fidx(tt), X, Y); u2, v2 = vlib.interp_vel(F, fidx(tt + h / 2), X + h / 2 * u1, Y + h / 2 * v1)
        X = X + h * u2; Y = Y + h * v2; tt += h
    res = dict(release_start_s=trel, period_used_s=Tn, particle_x_at_t0_over_D={f'{yn*1e3:+.0f}mm': float((xx - xc) / D) for yn, xx in zip(Ys, X)},
               lead_particle_over_D=float((X.max() - xc) / D))
    outer = [res['particle_x_at_t0_over_D'][k] for k in ('-24mm', '-16mm', '+16mm', '+24mm')]
    res['outer_particles_over_D'] = outer
    res['outer_png_front_c025_over_D'] = [per[k] for k in ('-24mm', '-16mm', '+16mm', '+24mm')]
    res['png_minus_particle_outer_D'] = [a - b for a, b in zip(res['outer_png_front_c025_over_D'], outer)]
    res['passed'] = bool(all(abs(o - 0.6) <= 0.3 for o in outer) and all(abs(d) <= 0.3 for d in res['png_minus_particle_outer_D']))
    res['criterion'] = 'outer streaks (y=+-16, +-24 mm): independently advected particles at x_cyl+0.6D +- 0.3D at t=0, and PNG c>0.25 front within 0.3D of them'
    res['note'] = 'pure-advection particles released at the needle tips at t_rel with the periodic (back-mapped) frame velocities; compare with the c>0.25 dye front'
A['release_consistency'] = res
# ---- L10 vs L11 (from meta convergence)
cv = meta.get('convergence', {})
if '10' in cv and '11' in cv:
    w10, w11 = cv['10']['window'], cv['11']['window']
    ref = json.load(open(os.path.join(ROOT, 'verify', 'out', 'reference.json')))['production_reference']
    A['L10_vs_L11'] = {k: dict(L10=w10[k2], L11=w11[k2], rel=(w10[k2] - w11[k2]) / w11[k2], E_ref=ref[k3], L11_vs_E=(w11[k2] - ref[k3]) / ref[k3])
                       for k, k2, k3 in [('St', 'St', 'St'), ('Cd_mean', 'Cd_mean', 'Cd_mean'), ('Cl_amp', 'Cl_amp_fit_fundamental', 'Cl_amp')]}
json.dump(A, open(os.path.join(ROOT, 'verify', 'out', f'{label}_audit_extras.json'), 'w'), indent=1, default=float)
print(json.dumps(A, indent=1, default=float))
# ---- visual
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
ax[0].plot(t, Gp, label='Gamma+ (omega>0, lower)'); ax[0].plot(t, -Gm, label='-Gamma- (upper)'); ax[0].axvspan(0.67, 0.83, color='g', alpha=.25, label='anchor 0.75+-0.08 s')
ax[0].set_xlim(0, 5); ax[0].set_xlabel('t [s]'); ax[0].legend(fontsize=7); ax[0].set_title(f'{label} | near-wake circulation, phase anchor | verify')
ax[1].imshow(np.clip(c0, 0, 1), extent=[(xs[0] - xc) / D, (xs[-1] - xc) / D, ys[-1] / D, ys[0] / D], aspect='equal')
ax[1].axvline(0.6, color='w', ls='--', lw=1); ax[1].add_patch(plt.Circle((0, 0), 0.5, fill=False, color='w'))
if res: ax[1].plot([v for v in res['particle_x_at_t0_over_D'].values()], np.array(G['needles_y']) / D, 'wx')
ax[1].set_xlim(-6, 4); ax[1].set_ylim(-3.5, 3.5); ax[1].set_xlabel('(x-x_cyl)/D'); ax[1].set_title(f'{label} | dye f0000 (t=0), dashed = target +0.6 D, x = advected particles | verify')
if 'L10_vs_L11' in A:
    ks = list(A['L10_vs_L11']); w = 0.25
    for i, (nm, key) in enumerate([('L10', 'L10'), ('L11', 'L11'), ('E ref', 'E_ref')]):
        ax[2].bar(np.arange(3) + (i - 1) * w, [A['L10_vs_L11'][k][key] / A['L10_vs_L11'][k]['E_ref'] for k in ks], w, label=nm)
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(ks); ax[2].axhline(1, color='k', lw=.5); ax[2].set_ylim(0.8, 1.25); ax[2].legend(fontsize=7)
    ax[2].set_title(f'{label} | L10 vs L11 vs E reference (ratio to E) | verify')
plt.tight_layout(); plt.savefig(os.path.join(ROOT, 'verify', 'visuals', f'{label}_audit_phase_front_convergence.png'), dpi=100)
