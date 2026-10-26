"""Static camera calibration of reference frame 120 + swing-plane fit (camera role).

Model (stand frame S, Z up, origin = pivot P_ring where the wire leaves the ring on the pin,
X along the arm toward its free end, Y horizontal away from the camera side):
  base  box x[bx0,bx1] y[by0,by1] z[bz0,bz1]   rounded edges radius rb
  post  box x[px0,px1] y[py0,py1] z[bz1,pz1]   rounded radius rp
  arm   box x[..,ax1]  y[-aw/2,aw/2] z[az0,az1] rounded radius ra (P on the arm's mid-plane y=0)
Camera: pinhole, square pixels, principal point fixed at image centre (960,544) unless --free-pp,
        f_px, world->camera rotation (rotvec), centre C.  OpenCV camera axes (x right, y down, z fwd).
Observations (all in frame-120 pixel coordinates):
  * sub-pixel silhouette edge points (camera/edges.py on frame 120); residual = distance to the
    image line of the plane through C tangent to the (rounded) model edge.
  * pivot image point P_ring (ring bottom, picked on 6x zoom, sigma 1.5 px).
  * auxiliary bob track (camera/bob_quick_f120.csv, mapped to frame-120 coords with the drift
    similarity): centre (u,v) and image radius r, for a bob on a circle about P_eff = P + (0,0,-e)
    of radius 1 (scale gauge: L_eff = 1) in a vertical plane at azimuth phi from the arm axis.
Soft priors (stated assumptions): post centred on arm in Y, post and arm cross-sections square
(sigma 15 %), fillet radii 0..0.04 units.
Writes camera/out/calib_result.json and prints per-edge residuals.
"""
import numpy as np, json, pathlib, sys, cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
ROOT = pathlib.Path(__file__).resolve().parents[1]
W, H = 1920, 1088
FREE_PP = "--free-pp" in sys.argv
VERB = True
PHI0 = "--phi0" in sys.argv          # adopt swing plane = arm vertical plane (phi = 0)
EDGEFILE = ROOT / "camera/out/edges_f120.json"
P_RING = np.array([1089.5, 127.0]); SIG_P = 0.5
USE_PEFF = "--no-peff" not in sys.argv; SIG_PEFF = 0.5
SIG_EDGE = 0.5; SIG_BOB = 1.0; SIG_R = 0.6

# edge -> (object, sharp-edge definition, faces outward normals)  ; definitions use param dict g
# each edge: point A(g), direction d, outward normals n1,n2 (for rounding), object radius key
def edge_defs(g):
    X, Y, Z = np.eye(3)
    aw2 = g["aw"]/2
    return {
        "b_left_top":     (np.r_[g["bx0"], 0, g["bz1"]], Y, -X, Z, "rb"),
        "b_back_top":     (np.r_[0, g["by1"], g["bz1"]], X, Y, Z, "rb"),
        "b_back_top2":    (np.r_[0, g["by1"], g["bz1"]], X, Y, Z, "rb"),
        "b_front_left_v": (np.r_[g["bx0"], g["by0"], 0], Z, -X, -Y, "rb"),
        "b_front_bottom": (np.r_[0, g["by0"], g["bz0"]], X, -Y, -Z, None),
        "b_back_right_v": (np.r_[g["bx1"], g["by1"], 0], Z, X, Y, "rb"),
        "p_left":         (np.r_[g["px0"], g["py0"], 0], Z, -X, -Y, "rp"),
        "p_right":        (np.r_[g["px1"], g["py1"], 0], Z, X, Y, "rp"),
        "p_top_front":    (np.r_[0, g["py0"], g["pz1"]], X, -Y, Z, "rp"),
        "p_top_right":    (np.r_[g["px1"], 0, g["pz1"]], Y, X, Z, "rp"),
        "p_bot_front":    (np.r_[0, g["py0"], g["bz1"]], X, -Y, -Z, None),
        "p_bot_right":    (np.r_[g["px1"], 0, g["bz1"]], Y, X, -Z, None),
        "a_top":          (np.r_[0, -aw2, g["az1"]], X, -Y, Z, "ra"),
        "a_bottom":       (np.r_[0, aw2, g["az0"]], X, Y, -Z, "ra"),
        "a_end_right":    (np.r_[g["ax1"], aw2, 0], Z, X, Y, "ra"),
        "a_front_bottom2":(np.r_[0, -aw2, g["az0"]], X, -Y, -Z, "ra"),
    }
USE_EDGES = ["b_left_top", "b_back_top", "b_front_left_v", "b_front_bottom", "b_back_right_v",
             "p_left", "p_right", "p_top_front", "p_top_right", "p_bot_front", "p_bot_right", "a_top", "a_bottom", "a_end_right", "a_front_bottom2"]
GEO = ["bx0", "bx1", "by0", "by1", "bz0", "bz1", "px0", "px1", "py0", "py1", "pz1", "aw", "az0", "az1", "ax1", "rb", "rp", "ra"]

def tangent_plane_normal(A, d, n1, n2, r, C):
    """unit normal m of the plane through C containing direction d and tangent (outer side) to the
    cylinder of radius r around the axis A - r n1 - r n2 (r=0: plane through the sharp edge)."""
    d = d/np.linalg.norm(d)
    Ax = A - r*(n1 + n2) if r > 0 else A
    w = Ax - C; wp = w - (w @ d)*d; D = np.linalg.norm(wp); wh = wp/D; e = np.cross(d, wh)
    if r <= 0: return e      # plane containing C, d and A: normal = d x wh
    a = r/D; b = np.sqrt(max(1 - a*a, 0))
    best = None
    for sgn in (1, -1):
        m = a*wh + sgn*b*e      # m.(Ax-C) = a*D = r
        T = Ax - r*m            # tangent point
        score = (T - Ax) @ (n1 + n2)
        if best is None or score > best[0]: best = (score, m)
    return best[1]

def unpack(x):
    i = 0
    f = x[i]; i += 1
    if FREE_PP: cx, cy = x[i:i+2]; i += 2
    else: cx, cy = W/2, H/2
    rv = x[i:i+3]; i += 3; C = x[i:i+3]; i += 3
    g = dict(zip(GEO, x[i:i+len(GEO)])); i += len(GEO)
    phi, e, Rb = x[i:i+3]; i += 3
    th = x[i:]
    return f, cx, cy, rv, C, g, phi, e, Rb, th

def project(Xw, f, cx, cy, R, C):
    Xc = (np.atleast_2d(Xw) - C) @ R.T
    return np.c_[f*Xc[:, 0]/Xc[:, 2] + cx, f*Xc[:, 1]/Xc[:, 2] + cy], Xc

def residuals(x, obs, detail=False):
    f, cx, cy, rv, C, g, phi, e, Rb, th = unpack(x)
    R = Rot.from_rotvec(rv).as_matrix()
    Kinv_T = np.linalg.inv(np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])).T
    res = []; det = {}
    defs = edge_defs(g)
    for name in USE_EDGES:
        A, d, n1, n2, rk = defs[name]
        r = abs(g[rk]) if rk else 0.0
        m = tangent_plane_normal(A, d, n1, n2, r, C)
        l = Kinv_T @ (R @ m)
        P = obs["edges"][name]
        dist = (P @ l[:2] + l[2]) / np.hypot(l[0], l[1])
        res.append(dist/SIG_EDGE); det[name] = dist
    uvP, _ = project(np.zeros(3), f, cx, cy, R, C)
    res.append((uvP[0] - P_RING)/SIG_P); det["P_ring"] = uvP[0] - P_RING
    if USE_PEFF:
        uvE, _ = project(np.array([0, 0, -e]), f, cx, cy, R, C)
        res.append((uvE[0] - obs["P_eff"])/SIG_PEFF); det["P_eff"] = uvE[0] - obs["P_eff"]
    # bob
    Pe = np.array([0, 0, -e])
    B = Pe + np.c_[np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi), -np.cos(th)]
    uvB, Xc = project(B, f, cx, cy, R, C)
    Dn = np.linalg.norm(Xc, axis=1); cosb = Xc[:, 2]/Dn
    rB = f*Rb/np.sqrt(Dn**2 - Rb**2)/np.sqrt(cosb)
    bo = obs["bob"]
    res.append(((uvB - bo[:, :2])/SIG_BOB).ravel()); det["bob_uv"] = uvB - bo[:, :2]
    res.append((rB - bo[:, 2])/SIG_R); det["bob_r"] = rB - bo[:, 2]
    # priors
    pri = [((g["py0"] + g["py1"])/2)/0.01,
           ((g["px1"] - g["px0"]) - (g["py1"] - g["py0"]))/(0.15*(g["px1"] - g["px0"])),
           ((g["az1"] - g["az0"]) - g["aw"])/(0.15*g["aw"])]
    if PHI0: pri.append(phi/1e-5)
    for k in ("rb", "rp", "ra"):
        pri.append(max(0, abs(g[k]) - 0.04)/0.005)
    res.append(np.array(pri))
    out = np.concatenate([np.atleast_1d(r).ravel() for r in res])
    return (out, det) if detail else out

def load_obs():
    E = json.load(open(EDGEFILE))
    edges = {k: np.array(E[k]["points"]) for k in USE_EDGES}
    bf = ROOT / "camera/out/bob_meas_f120.csv"      # measurement-role track (preferred) mapped to f120
    if not bf.exists() or "--quick-bob" in sys.argv: bf = ROOT / "camera/out/bob_quick_f120.csv"
    print("bob source:", bf.name)
    B = np.loadtxt(bf, delimiter=",", skiprows=1)
    pe = json.load(open(ROOT / "camera/out/pivot_f120.json"))
    return {"edges": edges, "bob": B[:, 2:5], "P_eff": np.array([pe["u_px"], pe["v_px"]])}

def initial(obs):
    f = 2480.0
    Xc = np.array([1.347, -0.012, -1.0]); Yc = np.array([0.742, 0.0137, 1.0]); Zc = np.array([0, -1.0, 0])
    M = np.c_[Xc/np.linalg.norm(Xc), Yc/np.linalg.norm(Yc), Zc]
    U, _, Vt = np.linalg.svd(M); R = U @ Vt
    dP = np.array([(P_RING[0] - 960)/f, (P_RING[1] - 544)/f, 1.0]) * 3.5
    C = -R.T @ dP
    def ray_plane(u, v, axis, val):
        dc = np.array([(u - 960)/f, (v - 544)/f, 1.0]); dw = R.T @ dc
        s = (val - C[axis])/dw[axis]; return C + s*dw
    bz1 = -1.12
    q = ray_plane(483, 912, 2, bz1); bx0, by0 = q[0], q[1]
    q = ray_plane(1483, 935, 2, bz1); bx1, by1 = q[0], q[1]
    q = ray_plane(483, 1030, 0, bx0); bz0 = q[2]
    q = ray_plane(691, 880, 2, bz1); px0, py0 = q[0], q[1]
    q = ray_plane(840, 897, 2, bz1); px1, py1 = q[0], q[1]
    q = ray_plane(700, 70, 1, py0); pz1 = q[2]
    g = dict(bx0=bx0, bx1=bx1, by0=by0, by1=by1, bz0=bz0, bz1=bz1, px0=px0, px1=px1, py0=py0, py1=py1, pz1=pz1,
             aw=0.10, az0=-0.02, az1=0.09, ax1=-0.05, rb=0.02, rp=0.01, ra=0.01)
    if VERB: print("init geo", {k: round(v, 3) for k, v in g.items()})
    bo = obs["bob"]
    th0 = (bo[:, 0] - 1090)/600.0
    x = [f] + ([960, 544] if FREE_PP else []) + list(Rot.from_matrix(R).as_rotvec()) + list(C) + [g[k] for k in GEO] + [0.0, 0.08, 0.08] + list(th0)
    return np.array(x, float)

def sparsity(x0, obs):
    from scipy.sparse import lil_matrix
    r0 = residuals(x0, obs); N = len(obs["bob"]); G = len(x0) - N
    S = lil_matrix((len(r0), len(x0)), dtype=int); S[:, :G] = 1
    # locate bob rows: after edges and pivot rows
    ne = sum(len(obs["edges"][k]) for k in USE_EDGES) + 2 + (2 if USE_PEFF else 0)
    for n in range(N):
        S[ne + 2*n, G + n] = 1; S[ne + 2*n + 1, G + n] = 1; S[ne + 2*N + n, G + n] = 1
    return S

def solve(obs, x0=None, verbose=1):
    x0 = initial(obs) if x0 is None else x0
    s = least_squares(residuals, x0, args=(obs,), method="trf", loss="linear", x_scale="jac", max_nfev=300, tr_solver="exact",
                      verbose=verbose)
    return s

def report(s, obs):
    r, det = residuals(s.x, obs, detail=True)
    f, cx, cy, rv, C, g, phi, e, Rb, th = unpack(s.x)
    out = {}
    allstatic = []
    for k in USE_EDGES:
        d = det[k]; allstatic.append(d); out[k] = float(np.sqrt(np.mean(d**2)))
        print(f"  {k:15s} rms {out[k]:.3f} px  mean {d.mean():+.3f}  n={len(d)}")
    allstatic = np.concatenate(allstatic)
    print("  P_ring resid", det["P_ring"], " P_eff resid", det.get("P_eff"))
    print("  static edge-point RMS %.3f px (n=%d)" % (np.sqrt(np.mean(allstatic**2)), len(allstatic)))
    print("  bob uv rms %.3f px, r rms %.3f px" % (np.sqrt(np.mean(det["bob_uv"]**2)), np.sqrt(np.mean(det["bob_r"]**2))))
    print("  f %.1f px  pp (%.1f,%.1f)  C %s  phi %.2f deg  e %.4f  Rb %.4f  theta range %.2f..%.2f deg" % (
        f, cx, cy, np.round(C, 4), np.degrees(phi), e, Rb, np.degrees(th.min()), np.degrees(th.max())))
    print("  geo", {k: round(float(v), 4) for k, v in g.items()})
    return out, allstatic, det

if __name__ == "__main__":
    obs = load_obs()
    s = solve(obs)
    print("cost", s.cost, "status", s.status)
    out, allstatic, det = report(s, obs)
    # covariance (Gauss-Newton) for key params
    J = s.jac; dof = max(1, len(s.fun) - len(s.x)); s2 = 2*s.cost/dof
    try:
        cov = np.linalg.pinv(J.T @ J) * s2
    except Exception:
        cov = None
    np.save(ROOT / "camera/out/calib_x.npy", s.x)
    if cov is not None: np.save(ROOT / "camera/out/calib_cov.npy", cov)
    print("chi2/dof", s2)
