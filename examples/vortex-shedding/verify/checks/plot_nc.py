"""v001 negative-control matrix visual from verify/out/negative_controls.json"""
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); S = json.load(open(os.path.join(H, '..', 'out', 'negative_controls.json')))
cases = [k for k in S if not k.startswith('_')]; checks = list(S['baseline']['results'].keys())
M = np.array([[0 if S[c]['results'][k]['passed'] else 1 for c in cases] for k in checks])
fig, ax = plt.subplots(figsize=(9, 6.5)); ax.imshow(M, cmap='RdYlGn_r', vmin=-0.3, vmax=1.3, aspect='auto')
ax.set_xticks(range(len(cases))); ax.set_xticklabels([c + ('\n(target caught)' if S[c].get('caught') else '') for c in cases], fontsize=8)
ax.set_yticks(range(len(checks))); ax.set_yticklabels(checks, fontsize=8)
for i in range(len(checks)):
    for j, c in enumerate(cases):
        tgt = S[c].get('target') or []
        ax.text(j, i, ('FAIL' if M[i, j] else 'pass') + (' *' if checks[i] in tgt else ''), ha='center', va='center', fontsize=7)
ax.set_title('v001 | negative controls (synthetic; * = target check) | verify\n'
             'st10: St +10 %  dye15: dye advected 1.5x  diverge: div(u)!=0  upstream: vortices move upstream\nenergy: wake KE grows 60 %/10 s', fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(H, '..', 'visuals', 'v001_negative_controls.png'), dpi=110)
