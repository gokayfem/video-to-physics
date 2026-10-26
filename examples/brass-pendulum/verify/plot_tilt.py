"""Tilt audit figure from verify/check_tilt.py JSON(s).  .venv/bin/python verify/plot_tilt.py R.json [R2.json ...] --out PNG --caption TXT"""
import argparse, json, math, textwrap
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("reports", nargs="+"); ap.add_argument("--out", required=True); ap.add_argument("--caption", default=""); a = ap.parse_args()
R = [json.load(open(p)) for p in a.reports]; r = R[0]; tl = r["level_tilt_rad"]; ex = 25  # exaggeration for drawing only
fig, ax = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw=dict(width_ratios=[1, 1.25]))
A = ax[0]; e = math.radians(ex) * tl / abs(tl)
A.plot([0, 0], [0, -1], c="k", lw=2, label="true gravity -Z_world (PND_Swing rest, unparented)")
A.plot([0, math.sin(e) * 1.1], [0, math.cos(e) * 1.1 - 1.1 - 0.0 + 1.1 - 1.1], alpha=0)
A.plot([-math.sin(e) * 0.0, math.sin(e)], [0, math.cos(e)], c="C3", lw=2, label="stand up-axis (PND_StaticTilt @ +Y)")
A.plot([0, -math.sin(e)], [0, -math.cos(e)], c="C3", lw=1, ls="--", label="stand 'down' axis")
A.add_patch(plt.Circle((0, -1), 0.08, color="goldenrod")); A.plot(0, 0, "ko")
A.annotate(f"stand up-axis at +{tl:.6f} rad\ntoward +X_world (drawn x{ex/math.degrees(abs(tl)):.0f})", (math.sin(e) * .6, math.cos(e) * .6), fontsize=8)
A.annotate("bob rest in stand frame = +tilt\n(world angle bob - stand down = -tilt)", (0.05, -0.75), fontsize=8)
A.set_xlim(-0.8, 0.8); A.set_ylim(-1.25, 1.1); A.set_aspect("equal"); A.set_xlabel("X_world"); A.set_ylabel("Z_world"); A.legend(fontsize=7, loc="upper left")
A.set_title("geometry (angle exaggerated for drawing)")
B = ax[1]; B.axis("off")
L = []
for rr, p in zip(R, a.reports):
    L += [f"{p.split('/')[-2] if 'p10' in p else 'checkpoint .blend'}  blend {rr['blend_sha256'][:12]}  -> {rr['status']}",
          f"  PND_Swing parent: {rr['swing']['parent']}  under tilt: {rr['swing_under_tilt']}  constraints: {rr['swing']['constraints']}",
          f"  swing world rot vs RotY(-theta): {rr['swing_world_rotation_vs_RotY(-theta)_max_abs']:.1e};  translation {rr['swing_world_translation_max_abs_m']:.1e} m",
          f"  bob angle from -Z_world vs solver theta (241 frames): {rr['bob_angle_from_minusZ_world_vs_theta_max_rad']:.1e} rad",
          f"  tilt empty rot_euler {[round(x, 9) for x in rr['tilt_empty']['rotation_euler']]}, origin {rr['tilt_empty_world_origin_m']}, animated {rr['tilt_empty']['animated']}",
          f"  stand up-axis angle {rr['stand_up_axis_angle_in_world_rad']['max']:.9f} vs params {tl:.9f} (err {rr['stand_up_axis_angle_in_world_rad']['max_err']:.1e})",
          f"  bob rest dir in stand frame {rr['bob_rest_direction_in_stand_frame_rad']:+.9f} rad;  world (bob - stand down) {rr['bob_rest_minus_stand_down_in_world_angle_rad']:+.9f}",
          f"  bob in stand frame = theta + tilt, max err {rr['bob_in_stand_frame_equals_theta_plus_tilt_max_err_rad']:.1e} rad",
          f"  cameras under tilt: {sum(rr['cameras_under_tilt'].values())}/{len(rr['cameras_under_tilt'])} ({', '.join(k.replace('PND_Cam_','') for k in rr['cameras_under_tilt'])})",
          f"  NOT under tilt: {rr['objects_not_under_tilt']}", ""]
B.text(0, 1, "\n".join(L), va="top", family="monospace", fontsize=7.2)
fig.suptitle("\n".join(textwrap.wrap(a.caption, 150)), fontsize=9.5); fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(a.out, dpi=110)
