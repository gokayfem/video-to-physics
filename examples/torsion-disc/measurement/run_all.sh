#!/bin/zsh
# Clip 03 measurement + camera pipeline (from ROOT). ~20 min (s1_drift ECC is the slow part, diagnostic only).
set -e; cd "${0:A:h}/.."; P=.venv/bin/python
$P measurement/scripts/s1_drift.py                 # KLT-chain + ECC homographies (diagnostic; chain used for seed propagation)
$P camera/edges.py camera/frames/f_000.png f000    # frame-0 sub-pixel frame edges
$P camera/calibrate.py && $P camera/calib_variants.py
$P measurement/scripts/s1d_edges_track.py          # edges in every frame
$P camera/calib_per_frame.py && $P camera/calib_per_frame.py --fix-f=2602
$P measurement/scripts/s2_disc_edges.py            # disc silhouette
$P measurement/scripts/s5_wire.py                  # wire line / top / bottom
rm -f camera/out/camera_poses.json; $P camera/locate.py   # needs measurement/cache/apex_offset.npy from a previous s4 run (bootstrap: run s3+s4 camera-centred first)
$P camera/pose_bundle.py --f=2602 --fix-f && cp camera/out/camera_poses.json camera/out/camera_poses_raw.json   # per-frame camera position (orbit), raw
$P measurement/scripts/s3_disc_pose.py             # disc pose per frame (world)
$P measurement/scripts/s4_twist.py                 # psi, methods A and B
$P measurement/scripts/s6_analysis.py
$P measurement/scripts/s9_anchor.py
$P camera/drift_fit.py                             # similarity-vs-homography diagnostic
$P camera/write_params.py                          # params/camera.json + camera/drift.csv
$P camera/reproject_check.py                       # R4 + overlay
tools/with_render_lock.sh ${BLENDER:-blender} --background --factory-startup --python camera/test/validate_projection_blender.py
$P measurement/scripts/s10_visuals.py
$P measurement/scripts/s11_summary.py
# ---- v002 (VR03-04/05): camera path smoothing, then the downstream stages again with the smoothed camera
$P camera/smooth_path.py && $P camera/plot_smoothing.py
$P measurement/scripts/s3_disc_pose.py && $P measurement/scripts/s4_twist.py && $P measurement/scripts/s6_analysis.py && $P measurement/scripts/s9_anchor.py
$P camera/write_params.py && $P camera/reproject_check.py
tools/with_render_lock.sh ${BLENDER:-blender} --background --factory-startup --python camera/test/validate_projection_blender.py
$P measurement/scripts/s12_P_track.py && $P measurement/scripts/s10_visuals.py && $P measurement/scripts/s11_summary.py
