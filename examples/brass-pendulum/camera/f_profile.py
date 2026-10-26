"""Profile of the calibration cost vs a fixed focal length (camera role): tests how strongly the
static edges + bob data constrain f (and checks the geometry role's f=2749 px at frame 120).
Writes camera/out/f_profile.json."""
import numpy as np, json, pathlib, sys
sys.argv = [sys.argv[0], "--phi0"]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import calibrate as C
C.VERB = False
obs = C.load_obs()
base = C.residuals
out = []
for F in [2250, 2350, 2431, 2500, 2600, 2749, 2850]:
    def res(x, o, detail=False, F=F):
        r = base(x, o, detail)
        if detail: return r
        return np.r_[r, (x[0] - F)/0.01]
    C_res = C.residuals; C.residuals = res
    x0 = C.initial(obs); x0[0] = F
    s = C.least_squares(res, x0, args=(obs,), method="trf", x_scale="jac", max_nfev=300)
    C.residuals = C_res
    _, det = base(s.x, obs, detail=True)
    st = np.concatenate([det[k] for k in C.USE_EDGES])
    f, cx, cy, rv, Cc, g, phi, e, Rb, th = C.unpack(s.x)
    row = {"f_px": F, "static_rms_px": float(np.sqrt(np.mean(st**2))), "bob_uv_rms_px": float(np.sqrt(np.mean(det["bob_uv"]**2))),
           "bob_r_rms_px": float(np.sqrt(np.mean(det["bob_r"]**2))), "P_ring_res": np.round(det["P_ring"], 2).tolist(),
           "dist_L": float(np.linalg.norm(Cc)), "cost": float(s.cost)}
    print(row); out.append(row)
json.dump(out, open(C.ROOT / "camera/out/f_profile.json", "w"), indent=1)
