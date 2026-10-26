"""Run all role-E production-output checks on a solver output dir (CONTRACT interface).
Usage: python run_checks.py [--out ROOT/solver/out] [--label ck001] [--fast]
Writes verify/out/checks_<label>.json and verify/visuals/<label>_production_checks.png"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['axes.titlesize'] = 8.5

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(ROOT, 'solver', 'out'))
ap.add_argument('--label', default='ck')
ap.add_argument('--fast', action='store_true')
ap.add_argument('--ref', default=None, help='override reference: St,Cd_mean,Cl_amp')
a = ap.parse_args()
ref = (dict(zip(['St', 'Cd_mean', 'Cl_amp'], map(float, a.ref.split(',')))) if a.ref else
       json.load(open(os.path.join(ROOT, 'verify', 'out', 'reference.json')))['production_reference'])
F = vlib.load(a.out)
r = vlib.run_all(a.out, ref, fast=a.fast, F=F)
r['reference'] = ref
json.dump(r, open(os.path.join(ROOT, 'verify', 'out', f'checks_{a.label}.json'), 'w'), indent=1, default=float)
for x in r['results']:
    print(('PASS' if x['passed'] else 'FAIL'), x['check'], json.dumps(x['value'], default=float)[:160])

# visual
D, U, xc = vlib.GEOM['D'], F['U'], vlib.GEOM['xc']
fig, ax = plt.subplots(2, 3, figsize=(16, 7.5))
n = vlib.ndye(F) // 2; c = vlib.getdye(F, n); xs, ys, px = vlib.dye_grid(c.shape)
ax[0, 0].imshow(np.clip(c, 0, 1), extent=[xs[0], xs[-1], ys[-1], ys[0]], aspect='equal')
ax[0, 0].add_patch(plt.Circle((xc, 0), D / 2, color='w', fill=False)); ax[0, 0].set_xlim(-0.12, 0.2)
ax[0, 0].set_title(f'{a.label} | dye frame {n} (t={n/24:.2f}s) | verify')
ax[0, 1].plot(F['tp'], F['Cl'], label='Cl'); ax[0, 1].plot(F['tp'], F['Cd'], label='Cd')
if F['pv'] is not None: ax[0, 1].plot(F['tp'], F['pv'] / U, label='wake probe v/U', alpha=.7)
st = r['results'][2]['value']; ax[0, 1].set_title(f"{a.label} | forces: St_Cl={st['St_Cl']:.4f} ref {ref['St']:.4f} | verify"); ax[0, 1].legend(fontsize=7)
ke = vlib.wake_ke(F); ax[0, 2].plot(F['t'], ke / ke.mean()); ax[0, 2].set_title(f'{a.label} | wake perturbation KE / mean | verify')
j = np.argmin(np.abs(F['y'])); m = (F['x'] > xc) & (F['x'] < xc + 16 * D)
ax[1, 0].imshow(F['v'][:, j, m] / U, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-1, vmax=1,
                extent=[(F['x'][m][0] - xc) / D, (F['x'][m][-1] - xc) / D, F['t'][0], F['t'][-1]])
cr = [x for x in r['results'] if x['check'] == 'vortex_convection_speed_over_U'][0]['value']
x0 = 4; ax[1, 0].plot([x0, x0 + 8], [2, 2 + 8 * D / (cr * U)], 'k--', label=f'fit Uc/U={cr:.3f}'); ax[1, 0].legend(fontsize=7)
ax[1, 0].set_xlabel('(x - xc)/D'); ax[1, 0].set_ylabel('t [s]'); ax[1, 0].set_title(f'{a.label} | centreline v/U x-t | verify')
adv = [x for x in r['results'] if x['check'] == 'dye_advected_by_velocity_scale'][0]['value']
ks = list(adv['err_by_scale'].keys()); ax[1, 1].plot([float(k) for k in ks], [adv['err_by_scale'][k] for k in ks], 'o-')
ax[1, 1].axhline(adv['err_no_motion'], color='gray', ls=':', label='no motion'); ax[1, 1].axvline(1, color='k', lw=.5)
ax[1, 1].set_xlabel('velocity scale s used to advect dye points'); ax[1, 1].set_ylabel('mean |c_adv - c0|'); ax[1, 1].legend(fontsize=7)
ax[1, 1].set_title(f"{a.label} | dye-vs-velocity consistency, best s={adv['best_scale']:.3f} | verify")
ax[1, 2].axis('off')
txt = '\n'.join(f"{'PASS' if x['passed'] else 'FAIL'}  {x['check']}" for x in r['results'])
ax[1, 2].text(0, 1, txt, va='top', family='monospace', fontsize=8.5, color='k')
for aa in ax.flat[:5]: aa.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(ROOT, 'verify', 'visuals', f'{a.label}_production_checks.png'), dpi=100)
print('all_passed', r['all_passed'])
