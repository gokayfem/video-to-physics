"""Weight-steel profile metric (VR-05), re-implemented from visual_review/scripts/weight_steel.py (read-only reference;
same sampling: rotated rectangle 0.90 x width, axis -0.30h..+0.38h about the weight centre/tilt; 61 x 41 samples; Oklab).
Render geometry from visual_review/out/ck002/predicted.csv (same baked trajectory), reference from the weight track.
.venv/bin/python look/weight_steel_local.py --tags ck002,new_v016 --frames 0,120 [--plot out.png] [--json out.json]
Render images: look/out/review/<tag>_fNNN_full.png (50 % renders, cubic upscale like the reviewer's eevee50)."""
import argparse, json, pathlib, sys
import numpy as np, cv2
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import metrics as MX
WT = np.genfromtxt(ROOT / "measurement/weight_track.csv", delimiter=",", names=True)
PR = np.genfromtxt(ROOT / "visual_review/out/ck002/predicted.csv", delimiter=",", names=True)


def geom(kind, n):
    if kind == "ref":
        return WT["u_px"][n], WT["v_px"][n], WT["tilt_deg"][n], WT["width_px"][n], WT["length_px"][n]
    return PR["wc_u"][n], PR["wc_v"][n], PR["weight_tilt_deg"][n], 143.0, 176.0


def sample(lab, u, v, tilt, wpx, hpx):
    t = np.radians(tilt); ax = np.array([np.sin(t), np.cos(t)]); pe = np.array([np.cos(t), -np.sin(t)])
    xs = np.linspace(-0.90, 0.90, 61); ys = np.linspace(-0.30, 0.38, 41) * hpx
    pts = np.array([[u, v]])[None] + xs[None, :, None] * (wpx / 2) * pe[None, None] + ys[:, None, None] * ax[None, None]
    m = [cv2.remap(np.ascontiguousarray(lab[..., c]).astype(np.float32), (pts[..., 0] - 0.5).astype(np.float32),
                   (pts[..., 1] - 0.5).astype(np.float32), cv2.INTER_LINEAR) for c in range(3)]
    return np.stack(m, -1)


def load(kind, n):
    if kind == "ref":
        return MX.ref_frame(n)
    im = cv2.imread(str(HERE / "out" / "review" / f"{kind}_f{n:03d}_full.png"))
    return cv2.resize(im, (1920, 1088), interpolation=cv2.INTER_CUBIC)


def stats(kind, frames):
    Ls, Ps, Cs = [], [], []
    for n in frames:
        s = sample(MX.oklab(load(kind, n)), *geom(kind, n))
        Ls.append(s[..., 0].ravel()); Ps.append(s[..., 0].mean(0)); Cs.append(np.hypot(s[..., 1], s[..., 2]).ravel())
    L = np.concatenate(Ls); C = np.concatenate(Cs); P = np.mean(Ps, 0)
    x = np.linspace(-0.9, 0.9, 61)
    return {"L_mean": float(L.mean()), "L_p05": float(np.percentile(L, 5)), "L_p95": float(np.percentile(L, 95)),
            "frac_below_0.35": float((L < 0.35).mean()), "frac_above_0.80": float((L > 0.80).mean()), "chroma": float(C.mean()),
            "centre_L": float(P[25:36].mean()), "band_L_at_-0.35": float(P[np.argmin(abs(x + 0.35))]), "band_L_at_+0.35": float(P[np.argmin(abs(x - 0.35))]),
            "flank_L_0.6-0.75": float(P[(abs(x) >= 0.6) & (abs(x) <= 0.75)].mean()), "profile": P.round(4).tolist()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", default="ck002"); ap.add_argument("--frames", default="0,120")
    ap.add_argument("--plot"); ap.add_argument("--json")
    a = ap.parse_args(); fr = [int(f) for f in a.frames.split(",")]
    res = {k: stats(k, fr) for k in ["ref"] + a.tags.split(",")}
    for k, v in res.items():
        print(f"{k:10s} meanL {v['L_mean']:.3f} p05-p95 {v['L_p05']:.2f}-{v['L_p95']:.2f} <0.35 {v['frac_below_0.35']*100:4.1f}% "
              f"C {v['chroma']:.3f} centre {v['centre_L']:.2f} bands {v['band_L_at_-0.35']:.2f}/{v['band_L_at_+0.35']:.2f} flanks {v['flank_L_0.6-0.75']:.2f}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"frames": fr, "stats": res}, indent=1))
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.4), facecolor="#fcfcfb")
        x = np.linspace(-0.9, 0.9, 61)
        cols = {"ref": "#2a78d6"}; pal = ["#b0b0a8", "#eb6834", "#2fa36b"]
        for i, (k, v) in enumerate(res.items()):
            ax.plot(x, v["profile"], lw=2.2 if k != "ref" else 2.6, color=cols.get(k, pal[(i - 1) % 3]),
                    label=f"{'reference' if k == 'ref' else k}: mean L {v['L_mean']:.3f}, <0.35 {v['frac_below_0.35']*100:.0f} %, C {v['chroma']:.3f}")
        ax.set_xlabel("position across the cylinder (-0.9 left ... +0.9 right)"); ax.set_ylabel("Oklab L (mean along axis)")
        ax.set_title(f"Weight side profile, frames {a.frames} (metric of visual_review/scripts/weight_steel.py)", loc="left", fontsize=10)
        ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(a.plot, dpi=110)
