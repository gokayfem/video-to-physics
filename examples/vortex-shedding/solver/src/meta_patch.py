"""Patch out/meta.json + params/physics.json with the Lagrangian-dye / window / periodicity records.
Usage: meta_patch.py VERSION RUN_DIR T0_ND RESTART_T_ND [KE_JSON]"""
import json, os, sys
R = os.path.dirname(os.path.abspath(__file__)) + "/../"
VER, RUN, T0, TRS = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
KEJ = sys.argv[5] if len(sys.argv) > 5 else None
TS = 0.01 / 0.015
m = json.load(open(R + "out/meta.json"))
LAGDIR = os.environ.get("LAGDIR", "lag")
lm = json.load(open(R + RUN + f"/{LAGDIR}/lag_meta.json"))
m["version"] = VER
m["dye_method"] = ("Lagrangian streaklines: forward particle tracking from the needle source disks (r_s = 0.35 mm, 7 material lines per needle, "
                   "4 particles per line per 1/48 s substep), RK4 dt = 1/48 s, bilinear on the 0.5 mm / (1/24 s) velocity history of the delivered run "
                   "(equivalent to the backward criterion 'trajectory passes the source disk after t_rel'); c = 1 inside the streak; lines broken at "
                   "> 1.5 mm particle gaps; blur sigma = sqrt(0.5^2 + 2 D_m age_max/px^2) px, D_m = 1e-9 m^2/s. src/lagr.py")
m["lagrangian"] = lm
if LAGDIR == "lag4":
    m["dye_method"] = ("v004 mass-conserving Lagrangian streaks: one material line per needle (12 particles per 1/48 s substep), RK4 dt = 1/48 s on the "
                       "0.5 mm / (1/24 s) velocity history; per segment stretch lam = ds/ds0; Gaussian cross-section sig^2 = sig0^2/lam^2 + 2 D_m age + sig_aa^2, "
                       "peak c = sig_e0/(lam sig) with sig_e0 = sqrt(sig0^2 + sig_aa^2) (mass per material length conserved); sig0 = 0.47 mm, sig_aa = 0.5 px, "
                       "D_m = 1e-9 m^2/s; sig0 chosen so the unstretched streak matches v003 frame 0 (cross-section integral 4.9 px = 1.22 mm, FWHM 1.15 mm); "
                       "streaks add per tracer, clip at 1; segments > 3 mm not drawn (mass reported). src/lagr4.py")
    m["dye_mass_budget_px2"] = lm["mass_budget_px2"]
m["release_start_s"] = lm["t_rel_s"]
m["front_definition"] = (lm["front_definition"] + "; outer-streak particle front at t=0: %.4f m (target x_cyl+0.6D = -0.0340 m)"
                         % lm["front_outer_x_m_by_frame"]["0"])
m["window"] = dict(t0_nd=T0, last_restart_from_L9_state_nd=179.5, delay_after_L9_to_L11_restart_D_over_U=T0 - 179.5,
                   continuation_restart_nd=TRS, run_dir=RUN)
if KEJ:
    ke = json.load(open(R + KEJ))
    m.setdefault("periodicity", {}).update(per_period=ke, definition="fluctuation KE about the period mean, 4 wake bands (x in m), per Cl up-crossing period",
        status="PASS from t=263 D/U: all 4 bands < 1 %/period over 263.0-268.3 and 268.3-273.5 (outlet band decays -2.8,-2.2,-1.7,-0.9,-0.5 %/period); "
               "Cl envelope < 0.5 %/period. Earlier drift: restart transients advecting through the bands.")
json.dump(m, open(R + "out/meta.json", "w"), indent=1, default=float)
p = json.load(open(R + "../params/physics.json"))
p["t_release_s"] = {"value": lm["t_rel_s"], "tag": "derived (Lagrangian particle front of outer streaks at x_cyl+0.6D at t=0)"}
p["dye_model"] = {"value": "v004 mass-conserving Gaussian streaks: sig^2 = sig0^2/lam^2 + 2 D_m age + sig_aa^2, c = sig_e0/(lam sig), sig0 = 0.47 mm, D_m = 1e-9 m^2/s" if LAGDIR == "lag4" else "Lagrangian streaklines, r_s = 0.35 mm, D_m = 1e-9 m^2/s blur", "tag": "declared (lead v002/v004); sig0 derived (matches v003 frame 0)"}
p["window_t0_solver_nd"] = {"value": T0, "tag": "derived (whole periods after the anchored window; Gamma+ at 0.75 s)"}
json.dump(p, open(R + "../params/physics.json", "w"), indent=1, default=float)
print("patched", VER, T0, lm["t_rel_s"])
