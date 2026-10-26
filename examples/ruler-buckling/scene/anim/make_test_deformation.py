"""ANALYTIC TEST deformation for developing the scene importer (NOT solver output, NOT physics of record).

  python scene/anim/make_test_deformation.py [out.npz]      (needs numpy + scipy)

Quasi-static imperfect clamped-free elastica under a dead tip load P(t):
  perfect elastica: sin(theta/2) = k sn(lambda*s, k), lambda*L = K(k), k = sin(alpha/2), P/Pcr = (2K(k)/pi)^2
  imperfection (test only): P/Pcr = (1 - alpha0/alpha) (2K/pi)^2, alpha0 = 2e-3 rad
Load: platform+peg 7.5 g + 20 g per weight, ramped over 0.15 s from each seat time (7 weights, 1.0 s + 0.6 s k).\nThen a kinematic test topple (35 deg, 1.5 s) + damped sway after the 7th weight - shapes from the elastica family.
Writes the deformation.npz interface (see scene/NOTES.md, "Animation / deformation interface").
"""
import json
import sys

import numpy as np
from scipy.special import ellipj, ellipk

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
P = json.loads((ROOT / "config/scene_params.json").read_text())
R = P["ruler"]
E, b, h = R["material_fea"]["E_Pa"], R["width_m"], R["thickness_m"]
L, Lc, Ltot = R["free_length_m"], R["clamped_length_m"], R["length_m"]
EI = E * b * h ** 3 / 12.0
PCR = np.pi ** 2 * EI / (4 * L ** 2)
g = 9.80665
NX, NS = R["mesh_segments_width"], R["mesh_segments_length"]
FPS, SUB, NFR = 24, 5, 241
t = np.arange((NFR - 1) * SUB + 1) / (FPS * SUB)
wt = 1.0 + 0.6 * np.arange(7.0)                      # lead schedule (CHANGELOG 2026-10-26)
m0 = 0.0075


def smooth(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


mass = m0 + 0.02 * smooth((t[:, None] - wt[None, :]) / 0.15).sum(1)
Pt = mass * g


def ratio(a):
    k = np.sin(a / 2); return (1 - 2e-3 / a) * (2 * ellipk(k * k) / np.pi) ** 2


def alpha_of(p):
    lo, hi = 2e-3 * (1 + 1e-9), np.pi * 0.95
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ratio(mid) < p else (lo, mid)
    return 0.5 * (lo + hi)


# TEST ONLY: 7 weights (147.5 g) are below the textbook Pcr; to exercise the importer we scale the load so the 7th weight
# buckles, then impose a slow topple (~1.5 s) toward -Y with a damped sway (the solver owns the real dynamics).
alpha_qs = np.array([alpha_of(p / PCR) for p in Pt])
t_on, a_eq = wt[-1] + 0.15, np.radians(35.0)
u = np.clip((t - t_on) / 1.5, 0, 1); S = u * u * (3 - 2 * u)
tt = np.clip(t - (t_on + 1.5), 0, None)
sway = np.where(t > t_on + 1.5, 0.10 * np.exp(-tt / 0.9) * np.sin(2 * np.pi * tt / 0.85), 0.0)
alpha = np.maximum(alpha_qs, a_eq * (S + sway))
Pt = np.where(t >= t_on, np.maximum(Pt, PCR * 1.0001), Pt)
s = np.arange(NS + 1) * (Ltot / NS)            # node arc length from the bottom end
xs = np.linspace(-b / 2, b / 2, NX + 1)
x0, y0, z0 = R["center_xy_m"][0], R["center_xy_m"][1], R["bottom_z_m"]
mid = np.empty((len(t), NS + 1, NX + 1, 3), np.float64)
sm = 0.5 * (s[:-1] + s[1:])                       # segment midpoints
for n, a in enumerate(alpha):
    k = np.sin(a / 2); K = ellipk(k * k); sig = np.clip(sm - Lc, 0, None)
    sn = ellipj(K * sig / L, k * k)[0]
    th = 2 * np.arcsin(np.clip(k * sn, -1, 1))
    ds = np.diff(s)
    y = np.concatenate([[0], np.cumsum(-np.sin(th) * ds)])   # bends toward -Y (camera side); solver decides the real sign
    z = np.concatenate([[0], np.cumsum(np.cos(th) * ds)])
    mid[n, :, :, 0] = x0 + xs[None, :]
    mid[n, :, :, 1] = y0 + y[:, None]
    mid[n, :, :, 2] = z0 + z[:, None]
out = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "scene/anim/test_deformation.npz")
np.savez_compressed(out, t_s=t, mid_surface_m=mid.astype(np.float64), tip_load_N=Pt, buckled=(Pt > PCR),
                    weight_times_s=wt, fps=FPS, subframes=SUB - 1,
                    source="ANALYTIC_TEST elastica (scene/anim/make_test_deformation.py) - not solver output")
on = int(np.argmax(Pt > PCR))
print(f"Pcr={PCR:.4f} N ({PCR / g * 1000:.1f} g)  onset t={t[on]:.4f} s frame={t[on] * FPS:.2f}  alpha_end={np.degrees(alpha[-1]):.1f} deg"
      f"  tip dy_end={mid[-1, -1, NX // 2, 1] * 1000:.1f} mm  -> {out}")
