"""Run the production-output checks on synthetic data: clean baseline (written to disk and re-read through the CONTRACT
loader) + 5 negative controls, each of which must be caught by its target check.
Usage: python negative_controls.py <scratch_dir_for_baseline_io>"""
import json, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth, vlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'verify', 'out')
scratch = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, 'synthetic_baseline')
REF = dict(St=0.19, Cd_mean=1.41, Cl_amp=0.55)   # synthetic truth (tests discrimination of the checks, not physics)
TARGET = {'st10': ['St_vs_reference', 'St_Cl_vs_field_v(5D wake)'],
          'dye15': ['dye_advected_by_velocity_scale'],
          'diverge': ['divergence_rms_over_vorticity_rms'],
          'upstream': ['vortex_convection_speed_over_U'],
          'energy': ['wake_KE_periodic_no_drift']}
summary = {}
t0 = time.time()
for defect in [None, 'st10', 'dye15', 'diverge', 'upstream', 'energy']:
    F = synth.make(defect=defect)
    if defect is None:
        synth.write(F, scratch)
        r = vlib.run_all(scratch, REF)            # through the file interface (PNG 16-bit + npz + meta)
        import cv2
        fr = F['dye'][120]
        cv2.imwrite(os.path.join(ROOT, 'verify', 'visuals', 'v001_synthetic_dye_f120.png'), (fr[..., ::-1] * 255).astype(np.uint8))
    else:
        r = vlib.run_all(None, REF, F=F)
    failed = [x['check'] for x in r['results'] if not x['passed']]
    name = defect or 'baseline'
    caught = None if defect is None else all(c in failed for c in TARGET[defect][:1])
    summary[name] = dict(failed_checks=failed, target=TARGET.get(defect), caught=caught,
                         results={x['check']: dict(passed=x['passed'], value=x['value']) for x in r['results']})
    print(f'{name:10s} failed={failed} caught={caught}  [{time.time()-t0:.0f}s]', flush=True)
summary['_verdict'] = dict(baseline_clean=len(summary['baseline']['failed_checks']) == 0,
                           all_controls_caught=all(summary[k]['caught'] for k in TARGET))
json.dump(summary, open(os.path.join(OUT, 'negative_controls.json'), 'w'), indent=1, default=float)
print(json.dumps(summary['_verdict']))
