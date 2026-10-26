"""Calibration uncertainty study (camera role): re-solves camera/calibrate.py under
(a) jackknife over silhouette edges, (b) perturbed edge picks (endpoints shifted +-3 px, resampled),
(c) pivot weights, (d) bob-track halves, (e) swing azimuth free vs fixed 0.
Writes camera/out/calib_variants.json and prints the spread of key quantities."""
import numpy as np, json, pathlib, sys, copy, cv2
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import calibrate as C
import edges as E
from scipy.spatial.transform import Rotation as Rot
ROOT = C.ROOT
base_obs = C.load_obs()
def summ(s, obs):
    f, cx, cy, rv, Cc, g, phi, e, Rb, th = C.unpack(s.x)
    R = Rot.from_rotvec(rv).as_matrix()
    _, det = C.residuals(s.x, obs, detail=True)
    st = np.concatenate([det[k] for k in C.USE_EDGES if k in det])
    fwd = R.T @ [0, 0, 1]
    return dict(f=float(f), Cx=float(Cc[0]), Cy=float(Cc[1]), Cz=float(Cc[2]), dist=float(np.linalg.norm(Cc)),
                pitch_deg=float(np.degrees(np.arcsin(fwd[2]))), yaw_deg=float(np.degrees(np.arctan2(fwd[0], fwd[1]))),
                roll_deg=float(np.degrees(Rot.from_matrix(R).as_euler("zyx")[2])),
                phi_deg=float(np.degrees(phi)), e=float(e), Rb=float(Rb), static_rms=float(np.sqrt(np.mean(st**2))),
                bz1=float(g["bz1"]), bx0=float(g["bx0"]), bx1=float(g["bx1"]), by0=float(g["by0"]), by1=float(g["by1"]),
                theta_max_deg=float(np.degrees(np.abs(th).max())))
V = {}
ALL = list(C.USE_EDGES)
def run(tag, obs=None, **mods):
    saved = {k: getattr(C, k) for k in mods}
    for k, v in mods.items(): setattr(C, k, v)
    try:
        s = C.solve(obs or base_obs, verbose=0); print(tag, flush=True); V[tag] = summ(s, obs or base_obs)
    finally:
        for k, v in saved.items(): setattr(C, k, v)
run("nominal_free_phi")
run("phi0", PHI0=True)
for k in ALL:
    run("drop_" + k, USE_EDGES=[e for e in ALL if e != k])
run("sigP_1.5", SIG_P=1.5, SIG_PEFF=1.5)
run("no_Peff", USE_PEFF=False)
run("Peff_sig5", SIG_PEFF=5.0)
run("phi0_Peff_sig5", PHI0=True, SIG_PEFF=5.0)
B = base_obs["bob"]
for name, sl in (("bob_first_half", slice(0, 120)), ("bob_second_half", slice(120, None))):
    o = dict(base_obs); o["bob"] = B[sl]; run(name, obs=o)
img = cv2.imread(str(ROOT / "camera/frames/f_120.png")); Fi = E.feature_images(img)
rng = np.random.default_rng(7)
for i in range(6):
    o = dict(base_obs); o["edges"] = {}
    for k in ALL:
        u0, v0, u1, v1, sr, ch = E.EDGES[k]
        d = rng.uniform(-3, 3, 4)
        pts = E.sample_edge(Fi, u0+d[0], v0+d[1], u1+d[2], v1+d[3], sr, ch)
        n, c, keep, rms = E.fit_line(pts); o["edges"][k] = pts[keep]
    run(f"repick_{i}", obs=o)
json.dump(V, open(ROOT / "camera/out/calib_variants.json", "w"), indent=1)
keys = ["f", "dist", "Cx", "Cy", "Cz", "pitch_deg", "yaw_deg", "roll_deg", "phi_deg", "e", "Rb", "bz1", "static_rms", "theta_max_deg"]
print(f"{'variant':24s}" + "".join(f"{k:>10s}" for k in keys))
for t, v in V.items(): print(f"{t:24s}" + "".join(f"{v[k]:10.4g}" for k in keys))
A = np.array([[v[k] for k in keys] for t, v in V.items() if not t.startswith("phi0")])
print(f"{'SD (excl phi0)':24s}" + "".join(f"{x:10.3g}" for x in A.std(0)))
print(f"{'range':24s}" + "".join(f"{x:10.3g}" for x in np.ptp(A, 0)))
