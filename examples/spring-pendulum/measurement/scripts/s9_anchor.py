"""Stage 9: apply the anchor-window rule declared in config/CHANGELOG.md (2026-10-25T23:41:09Z) BEFORE any fit:
T_a = first turning point (any measured oscillation mode) after which that mode's half-cycle amplitude increases by
> 3 sigma vs the previous half-cycle, or a visible discontinuity (attachment jump, coil-count change, shape change).
Minimum 2.5 s; if none, T_a = 10 s with a held-out last 2.5 s. Also reports trends (amplitude, half-period) that the
rule does not test. Writes measurement/anchor_window.json."""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
AN = json.load(open(os.path.join(MEAS, "analysis.json")))
def scan(hc, name, note=""):
    ev = []
    for a, b in zip(hc[:-1], hc[1:]):
        dz = (b["amp"] - a["amp"]) / np.hypot(a["s_amp"], b["s_amp"])
        ev.append(dict(turning_point_t=b["t_start"], prev_amp=a["amp"], amp=b["amp"], z=dz))
    trig = [e for e in ev if e["z"] > 3 and e["turning_point_t"] >= 0]
    return dict(mode=name, note=note, n_half_cycles=len(hc), max_increase_z=max((e["z"] for e in ev), default=None),
                max_increase_at_t=max(ev, key=lambda e: e["z"])["turning_point_t"] if ev else None, first_trigger_t=trig[0]["turning_point_t"] if trig else None, steps=ev)
# start-of-clip turning point (one-sided quadratic over frames 0-5; window variants 4-6 frames give the systematic)
A_ = np.load(os.path.join(CACHE, "analysis.npz")); y_ = A_["phid"]; s_ = A_["sphid"]; t_ = np.arange(N) / FPS; ests = []
for k in (4, 5, 6):
    X_ = np.c_[np.ones(k), t_[:k], t_[:k] ** 2]; w_ = 1 / np.maximum(s_[:k], 0.05); co, *_ = np.linalg.lstsq(X_ * w_[:, None], y_[:k] * w_, rcond=None)
    te = -co[1] / (2 * co[2]); ests.append((te, co[0] + co[1] * te + co[2] * te ** 2))
ests = np.array(ests); tp0 = dict(t=float(ests[1, 0]), y=float(ests[1, 1]), sy=float(np.hypot(0.086, ests[:, 1].std() + 0.05)))
tp1 = AN["turning_points_swing"][0]
hc0 = dict(t_start=tp0["t"], t_end=tp1["t"], amp=abs(tp1["y"] - tp0["y"]) / 2, s_amp=0.5 * np.hypot(tp0["sy"], tp1["sy"]), half_period=tp1["t"] - tp0["t"], s_half_period=0.01)
modes = [scan([hc0] + AN["half_cycles_swing"], "lateral swing (image angle of ring about P_eff)", "primary oscillation mode"),
         scan(AN["half_cycles_weight_tilt_at_swing_tps"], "weight rocking (absolute weight tilt at swing turning points)", "locked to the swing frequency"),
         scan(AN["half_cycles_weight_rel_tilt_at_swing_tps"], "weight rocking relative to pivot-ring line", "locked to the swing frequency")]
raw_r = scan(AN["half_cycles_radius"], "pivot-ring distance r(t), raw turning points (NOT an independent mode)",
             "r(t) is a mixture of swing-locked w and 2w terms; no independent axial (vertical bob) frequency is detected (analysis.json vertical_bob_search); half-cycles of a mixture alternate large/small by construction")
disc = AN["discontinuity_screen"]; coils = AN["coils"]
visible_disc = dict(attachment_jump="none: S-hook static to <=1.2 px after drift removal; ring and weight tracks have no acceleration outliers > 6 robust sigma (ring/weight centre screens 0 flagged)",
                    coil_count_change="none: open-coil gap count is 11 or 12; the 12th gap is the boundary turn between the open and close-wound sections and appears only when the spring is swung to image-right (mean swing +%.1f deg vs %.1f deg), reversibly, %d transitions; the close-wound section keeps ~%d scallops (right edge median). Interpreted as a near-closed coil opening on the outside of the bend, not a coil-count change" % (
                        coils["mean_swing_when_12_deg"], coils["mean_swing_when_11_deg"], coils["n_transitions"], coils["dense_scallops_R_median"]),
                    shape_change="none visible: single-frame tracker outliers at frame 116 (ring radius -3 px, method A vs B differ by 5 px; ring looks unchanged in a 2x zoom) and weight width/tilt outliers at frames 75-77, 135-136, 143-144 coincide with motion blur/cadence frames; weight silhouette length is constant (sd 0.8 px)")
trig_ts = [m["first_trigger_t"] for m in modes if m["first_trigger_t"] is not None]
if trig_ts:
    Ta = max(min(trig_ts), 2.5); held = None; reason = "rule triggered"
else:
    Ta = 10.0; held = [7.5, 10.0]; reason = "no qualifying amplitude increase (> 3 sigma) in any measured mode and no visible discontinuity"
hc = AN["half_cycles_swing"]; tm = np.array([h["t_mid"] for h in hc]); am = np.array([h["amp"] for h in hc]); sa = np.array([h["s_amp"] for h in hc]); hp = np.array([h["half_period"] for h in hc]); shp = np.array([h["s_half_period"] for h in hc])
k = np.argmax(am); co1 = np.polyfit(tm[:k + 1], am[:k + 1], 1, w=1 / sa[:k + 1], cov="unscaled")
cop = np.polyfit(tm, hp, 1, cov=True)
trends = dict(swing_amp_first_deg=float(am[0]), swing_amp_peak_deg=float(am[k]), swing_amp_peak_t=float(tm[k]), swing_amp_last_deg=float(am[-1]),
              growth_pct_first_to_peak=float(100 * (am[k] / am[0] - 1)), growth_rate_deg_per_s_until_peak=[float(co1[0][0]), float(np.sqrt(co1[1][0, 0]))],
              half_period_first_s=float(hp[0]), half_period_last_s=float(hp[-1]), half_period_trend_s_per_s=[float(cop[0][0]), float(np.sqrt(cop[1][0, 0]))],
              half_period_sd_s=float(hp.std()), half_period_median_sd_s=float(np.median(shp)),
              note="the declared rule tests half-cycle-to-half-cycle jumps only; the swing amplitude instead grows gradually (+%.1f %% from %.2f s to %.2f s, a non-physical energy injection for a passive system) and then decays faster; the half-period lengthens by %.1f %% over the clip. Both lie inside a 10 s anchor window." % (
                  100 * (am[k] / am[0] - 1), tm[0], tm[k], 100 * (hp[-1] / hp[0] - 1)))
out_tp0 = dict(start_turning_point=tp0, note="phase at t=0: weight essentially at its left turning point (swing %.2f deg at t=0, turning point at t=%.3f s, %.2f deg); not released exactly at t=0" % (y_[0], tp0["t"], tp0["y"]))
out = dict(start_of_clip=out_tp0, rule="config/CHANGELOG.md 2026-10-25T23:41:09Z", T_a_s=Ta, held_out_s=held, reason=reason, modes=modes, raw_radius_literal=raw_r,
           literal_note="if the rule were applied literally to the raw r(t) turning points (not an oscillation mode), the first > 3 sigma increase would be at t = %s s" % (raw_r["first_trigger_t"]),
           visible_discontinuities=visible_disc, trends_not_tested_by_rule=trends,
           near_misses=[dict(mode=m["mode"], max_z=m["max_increase_z"], at_t=m["max_increase_at_t"]) for m in modes])
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return round(float(o), 5)
    return o
json.dump(clean(out), open(os.path.join(MEAS, "anchor_window.json"), "w"), indent=1)
print("T_a =", Ta, reason); print(json.dumps(clean(out["near_misses"]))); print(out["literal_note"]); print(json.dumps(clean(trends), indent=1))
