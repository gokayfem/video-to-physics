"""TEST ONLY - synthetic dye sequence in the solver's interface format (NOT a flow solution).
Streaklines from the 7 needle tips advected (RK2) by an analytic field: potential flow round the cylinder + a Karman street of
Lamb-Oseen vortices (St 0.18, convected at 0.85 U). Gaussian line spreading sigma = sqrt(s0^2 + g^2 * s).
Also bakes, for the scene checks: 3 marker discs (tracer 0) at known pixels and an 11-cell frame-number barcode (tracer 2).
usage: python make_test_dye.py OUT_DIR
Format (CONTRACT): f0000..f0240.png, 16-bit RGB, R,G,B = tracer 0,1,2 in 0..1, x [-0.30,0.30], y [-0.06,0.06], 0.25 mm/px,
row 0 = +y, col 0 = -x."""
import json, os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
P = geom.P; inj = P["injectors"]; cy = P["cylinder"]; dy = P["dye"]
U, D = 0.015, cy["D"]; a = D / 2; xc, yc = cy["center"]
St = 0.18; T = D / (St * U); Uc = 0.85 * U; Gam = 2.0 * U * D; rc0 = 0.3 * D; h = 0.28 * Uc * T
FPS, NF = 24, 241; dt_rel = 1 / 96; T0 = (xc + 0.6 * D - inj["x_tip"]) / U

def vortices(t):
    n = np.arange(int(np.floor((t - 60) / (T / 2))), int(np.floor(t / (T / 2))) + 1)
    tb = n * T / 2; age = t - tb; ok = age >= 0
    n, age = n[ok], age[ok]
    sgn = np.where(n % 2 == 0, 1, -1)                     # +1: upper row (clockwise, Gamma<0)
    xv = xc + 1.0 * D + Uc * age; yv = sgn * h / 2 * np.minimum(1, 0.4 + age / T)
    G = -sgn * Gam * np.minimum(1, age / (0.5 * T)); rc = np.sqrt(rc0 ** 2 + 4 * 2e-6 * age)
    return xv, yv, G, rc

def vel(x, y, t):
    X, Y = x - xc, y - yc; r2 = np.maximum(X * X + Y * Y, a * a)
    u = U * (1 - a * a * (X * X - Y * Y) / r2 ** 2); v = -U * 2 * a * a * X * Y / r2 ** 2
    xv, yv, G, rc = vortices(t)
    dx = x[:, None] - xv[None]; dyy = y[:, None] - yv[None]; q = dx * dx + dyy * dyy + 1e-12
    k = G[None] / (2 * np.pi * q) * (1 - np.exp(-q / rc[None] ** 2))
    k *= np.clip((x[:, None] - xc - 0.5 * D) / D, 0, 1)    # street acts only behind the cylinder
    u += (-k * dyy).sum(1); v += (k * dx).sum(1)
    wall = np.clip((0.06 - np.abs(y)) / 0.006, 0, 1)        # crude no-penetration at the walls
    return u * wall, v * wall

ys = np.array(inj["y"]); tr = np.array(P["v3"]["dye_streak_colors"])
px_, py_, pid, pt0 = [np.zeros(0)] * 4
t = -np.ceil(T0 * 48) / 48; dt = 1 / 48; nrel = 0
cols = np.arange(geom.NXI); rows = np.arange(geom.NYI)
XX, YY = geom.px_to_world(cols[None, :], rows[:, None])
inside_cyl = (XX - xc) ** 2 + (YY - yc) ** 2 < a * a
s0 = dy["sigma0"]; g = dy["sigma_growth"]; W0 = np.sqrt(2 * np.pi) * (s0 / geom.DX) * (U * dt_rel / geom.DX)  # peak c ~ 1 at the tip (sigma0)
# markers (world positions are the pixel centres of these pixels) + barcode cells
MK = [(720, 80), (1280, 60), (1600, 400)]
BC_X = [-0.085 + 0.0055 * i for i in range(11)]; BC_Y = 0.050; BC_H = 0.0015
def frame_img(fi):
    c = np.zeros((3, geom.NYI, geom.NXI), np.float32)
    age = fi / FPS - pt0; sig = np.sqrt(s0 ** 2 + g * g * U * np.maximum(age, 0)) / geom.DX
    bins = np.clip(np.round(np.log(sig / 2.0) / np.log(1.15)), 0, 30).astype(int)
    for b in np.unique(bins):
        sb = 2.0 * 1.15 ** b
        for k in range(3):
            sel = (bins == b) & (tr[pid.astype(int)] == k)
            if not sel.any(): continue
            cx_, ry_ = geom.world_to_px(px_[sel], py_[sel]); acc = np.zeros((geom.NYI, geom.NXI), np.float32)
            i0 = np.floor(cx_).astype(int); j0 = np.floor(ry_).astype(int); fx = cx_ - i0; fy = ry_ - j0
            for di, dj, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                ii, jj = i0 + di, j0 + dj; ok = (ii >= 0) & (ii < geom.NXI) & (jj >= 0) & (jj < geom.NYI)
                np.add.at(acc, (jj[ok], ii[ok]), wt[ok])
            c[k] += cv2.GaussianBlur(acc, (0, 0), sb) * W0
    c[:, inside_cyl] = 0
    for (cc, rr) in MK:   # marker discs r = 1.5 mm
        m = (cols[None, :] - cc) ** 2 + (rows[:, None] - rr) ** 2 <= 6 ** 2; c[0][m] = 1
    bits = [1] + [(fi >> (8 - i)) & 1 for i in range(9)] + [1]
    for xb, bt in zip(BC_X, bits):
        if bt:
            m = (np.abs(XX - xb) <= BC_H) & (np.abs(YY - BC_Y) <= BC_H); c[2][m] = 1
    return np.clip(c, 0, 1)

fi = 0
while fi < NF:
    tf = fi / FPS
    if abs(t - tf) < 1e-9:
        c = frame_img(fi)
        cv2.imwrite(os.path.join(OUT, f"f{fi:04d}.png"), np.round(c[::-1].transpose(1, 2, 0) * 65535).astype(np.uint16))
        fi += 1; continue
    # release new particles over [t, t+dt)
    nr = int(round(dt / dt_rel)); tr_ = t + np.arange(nr) * dt_rel
    px_ = np.concatenate([px_, np.full(7 * nr, inj["x_tip"])]); py_ = np.concatenate([py_, np.tile(ys, nr)])
    pid = np.concatenate([pid, np.tile(np.arange(7), nr)]); pt0 = np.concatenate([pt0, np.repeat(tr_, 7)])
    u1, v1 = vel(px_, py_, t); xm, ym = px_ + 0.5 * dt * u1, py_ + 0.5 * dt * v1
    u2, v2 = vel(xm, ym, t + 0.5 * dt); px_ = px_ + dt * u2; py_ = py_ + dt * v2
    keep = (px_ < 0.31) & ((px_ - xc) ** 2 + (py_ - yc) ** 2 > (0.98 * a) ** 2)
    px_, py_, pid, pt0 = px_[keep], py_[keep], pid[keep], pt0[keep]
    t = round((t + dt) * 48) / 48
json.dump({"TEST": "SYNTHETIC TEST SEQUENCE - analytic Karman-like kinematics, NOT a Navier-Stokes solution",
           "generator": "scene/anim/make_test_dye.py", "frames": NF, "fps": FPS, "map": "CONTRACT: x[-0.30,0.30] y[-0.06,0.06] 0.25mm/px row0=+y col0=-x",
           "markers_px_col_row": MK, "markers_world_xy": [list(map(float, geom.px_to_world(cc, rr))) for cc, rr in MK],
           "barcode": {"tracer": 2, "cells_x": BC_X, "y": BC_Y, "half_size": BC_H, "bits": "guard, 9 bits MSB->LSB of frame index, guard"},
           "params": {"U": U, "St": St, "T_s": T, "Gamma": Gam, "h": h, "T0_release_before_t0_s": T0}},
          open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print("TEST_DYE_OK", NF, len(px_))
