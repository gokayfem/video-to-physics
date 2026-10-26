"""Lagrangian dye from a velocity history: material lines / streaklines advected with RK4 and rendered as
mass-conserving Gaussian filaments into 16-bit tracer PNGs.

Velocity: any callable vel(t, P (n, 2)) -> (n, 2) in m/s, or `GridVelocity` built from arrays (t, x, y, u, v)
(bilinear in space, linear in time). Dye: `Source` (point needle releasing a streakline, n_sub particles per substep)
and `MaterialLine` (a polyline marked at time t0). Any number of sources/lines; each writes to a tracer index.

Filament rendering (per segment between consecutive particles of one filament):
    lam = ds / ds0  (stretch since release),   sigma^2 = sigma0^2 / lam^2 + 2 D age + sigma_aa^2,
    c   = strength * sigma_e0 / (lam * sigma),  sigma_e0 = sqrt(sigma0^2 + sigma_aa^2)
so c * sigma * ds = strength * sigma_e0 * ds0: tracer mass per unit material length is conserved (stretching thins
and dilutes, diffusion widens). Segments are sampled every <= sample_px, splatted bilinearly into log-spaced sigma
bins, each bin is Gaussian-blurred once, tracers sum and are clipped to [0, 1].

Output (SPEC tracer contract): PNG 16-bit RGB, channel k = tracer 3g + k of group g; group 0 in OUT/fNNNN.png,
group g > 0 in OUT/g<g>/fNNNN.png; meta.json holds the plane extent, px size, pixel-centre rule, frame times and a
mass budget per frame and tracer.
"""
from __future__ import annotations

import json
import math
import struct
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

PIXEL_RULE = "x_i = x0 + (i + 0.5) px (col 0 at the -x edge), y_j = y1 - (j + 0.5) px (row 0 at the +y edge)"


# ============================================================================ geometry / velocity
@dataclass
class Plane:
    """Rectified world plane rendered to an image: extent [x0, x1] x [y0, y1] (m), pixel size px (m)."""
    x0: float
    x1: float
    y0: float
    y1: float
    px: float

    @property
    def nx(self) -> int:
        return int(round((self.x1 - self.x0) / self.px))

    @property
    def ny(self) -> int:
        return int(round((self.y1 - self.y0) / self.px))

    def to_pixel(self, P):
        """World (n, 2) -> fractional pixel coordinates (col, row), pixel centres at integers."""
        P = np.asarray(P, float)
        return np.stack([(P[:, 0] - self.x0) / self.px - 0.5, (self.y1 - P[:, 1]) / self.px - 0.5], 1)

    def centres(self):
        """Pixel-centre coordinates xs (nx,), ys (ny,) following PIXEL_RULE."""
        return self.x0 + (np.arange(self.nx) + 0.5) * self.px, self.y1 - (np.arange(self.ny) + 0.5) * self.px


class GridVelocity:
    """Velocity history on a tensor grid: t (nt,), x (nx,), y (ny,) ascending, u, v (nt, ny, nx).
    Bilinear in space, linear in time (held constant outside [t0, t_end]); outside the grid: 'zero' or 'clamp'."""

    def __init__(self, t, x, y, u, v, outside: str = "zero"):
        self.t, self.x, self.y = (np.atleast_1d(np.asarray(a, float)) for a in (t, x, y))
        self.u = np.asarray(u, float).reshape(len(self.t), len(self.y), len(self.x))
        self.v = np.asarray(v, float).reshape(self.u.shape)
        self.outside = outside

    def _bilinear(self, k, P):
        fi = (P[:, 0] - self.x[0]) / (self.x[1] - self.x[0])
        fj = (P[:, 1] - self.y[0]) / (self.y[1] - self.y[0])
        inside = (fi >= 0) & (fi <= len(self.x) - 1) & (fj >= 0) & (fj <= len(self.y) - 1)
        fi = np.clip(fi, 0, len(self.x) - 1.000001)
        fj = np.clip(fj, 0, len(self.y) - 1.000001)
        i, j = fi.astype(int), fj.astype(int)
        a, b = fi - i, fj - j
        out = np.empty((len(P), 2))
        for c, F in enumerate((self.u[k], self.v[k])):
            out[:, c] = (F[j, i] * (1 - a) * (1 - b) + F[j, i + 1] * a * (1 - b) + F[j + 1, i] * (1 - a) * b
                         + F[j + 1, i + 1] * a * b)
        if self.outside == "zero":
            out[~inside] = 0.0
        return out

    def __call__(self, t, P):
        P = np.atleast_2d(P)
        if len(self.t) == 1:
            return self._bilinear(0, P)
        k = int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 2))
        w = float(np.clip((t - self.t[k]) / (self.t[k + 1] - self.t[k]), 0.0, 1.0))
        return (1 - w) * self._bilinear(k, P) + w * self._bilinear(k + 1, P)


def rk4_advect(vel: Callable, t: float, P, h: float):
    k1 = vel(t, P)
    k2 = vel(t + 0.5 * h, P + 0.5 * h * k1)
    k3 = vel(t + 0.5 * h, P + 0.5 * h * k2)
    k4 = vel(t + h, P + h * k3)
    return P + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


# ============================================================================ dye objects
@dataclass
class Source:
    """Point source (needle) releasing a streakline into tracer `tracer` while t_start <= t < t_end."""
    xy: Sequence[float]
    tracer: int = 0
    t_start: float | None = None          # default: first frame time
    t_end: float = math.inf
    strength: float = 1.0                 # peak concentration of the unstretched streak
    sigma0: float | None = None           # physical streak half-width (m); default RenderConfig.sigma0


@dataclass
class MaterialLine:
    """Polyline of material points marked at time t0 (e.g. a dye line laid across the flow)."""
    points: np.ndarray
    tracer: int = 0
    t0: float | None = None
    strength: float = 1.0
    sigma0: float | None = None


@dataclass
class RenderConfig:
    sigma0: float = 1e-3                  # m, unstretched filament Gaussian sigma
    sigma_aa_px: float = 0.5              # px, anti-aliasing floor
    diffusivity: float = 0.0              # m^2/s, molecular diffusion of the dye
    gap_max: float = math.inf             # m, segments longer than this are not drawn (mass reported as dropped)
    sample_px: float = 0.25               # px, max sample spacing along a segment
    bin_ratio: float = 1.05               # ratio between neighbouring sigma bins
    clip: bool = True                     # clip concentrations to [0, 1]


@dataclass
class _Filament:
    tracer: int
    strength: float
    sigma0: float
    P: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    t_rel: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ds0: np.ndarray = field(default_factory=lambda: np.zeros(0))       # per segment (len(P) - 1)

    def append(self, P, t_rel, ds0_new):
        if len(self.P):
            self.ds0 = np.concatenate([self.ds0, ds0_new])
        else:
            self.ds0 = np.asarray(ds0_new[1:], float)
        self.P = np.vstack([self.P, P])
        self.t_rel = np.concatenate([self.t_rel, t_rel])


# ============================================================================ rendering
def _blur(img, sigma):
    try:
        import cv2
        return cv2.GaussianBlur(img, (0, 0), float(sigma), borderType=cv2.BORDER_CONSTANT)
    except ImportError:
        pass
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(img, sigma, mode="constant", truncate=4.0)
    except ImportError:
        r = int(math.ceil(4 * sigma))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        k /= k.sum()
        img = np.apply_along_axis(lambda a: np.convolve(a, k, "same"), 0, img)
        return np.apply_along_axis(lambda a: np.convolve(a, k, "same"), 1, img)


def render_filaments(filaments, t: float, plane: Plane, cfg: RenderConfig, n_tracers: int):
    """Render filaments at time t -> (img (n_tracers, ny, nx) float, unclipped; budget {tracer: {...}} in px^2)."""
    nx, ny, px = plane.nx, plane.ny, plane.px
    budget = {k: dict(injected=0.0, dropped_gap=0.0, drawn=0.0) for k in range(n_tracers)}
    samples = {k: [] for k in range(n_tracers)}
    s_aa = cfg.sigma_aa_px
    for f in filaments:
        if len(f.P) < 2:
            continue
        q = plane.to_pixel(f.P)
        d = np.diff(q, axis=0)
        ds = np.hypot(d[:, 0], d[:, 1])
        ds0 = f.ds0 / px
        lam = np.maximum(ds / np.maximum(ds0, 1e-12), 1e-3)
        age = np.maximum(t - 0.5 * (f.t_rel[1:] + f.t_rel[:-1]), 0.0)
        s0 = f.sigma0 / px
        sig = np.sqrt(s0**2 / lam**2 + 2.0 * cfg.diffusivity * age / px**2 + s_aa**2)
        se0 = math.sqrt(s0**2 + s_aa**2)
        c = f.strength * se0 / (lam * sig)
        m = f.strength * math.sqrt(2 * math.pi) * se0 * ds0
        b = budget[f.tracer]
        b["injected"] += float(m.sum())
        ok = ds * px <= cfg.gap_max
        b["dropped_gap"] += float(m[~ok].sum())
        idx = np.where(ok & (ds0 > 0))[0]
        if len(idx) == 0:
            continue
        ns = np.maximum(1, np.ceil(ds[idx] / cfg.sample_px)).astype(int)
        rep = np.repeat(idx, ns)
        nrep = np.repeat(ns, ns)
        fr = (np.arange(len(rep)) - np.repeat(np.cumsum(ns) - ns, ns) + 0.5) / nrep
        X = q[rep, 0] + fr * d[rep, 0]
        Y = q[rep, 1] + fr * d[rep, 1]
        w = c[rep] * (ds[rep] / nrep) * math.sqrt(2 * math.pi) * sig[rep]
        b["drawn"] += float(w.sum())
        samples[f.tracer].append((X, Y, w, sig[rep]))
    img = np.zeros((n_tracers, ny, nx), np.float32)
    for k, parts in samples.items():
        if not parts:
            continue
        X, Y, w, sg = (np.concatenate([p[i] for p in parts]) for i in range(4))
        lo = float(sg.min())
        lr = math.log(cfg.bin_ratio)
        bins = np.round(np.log(sg / lo) / lr).astype(int)
        pad = int(math.ceil(4 * lo * cfg.bin_ratio ** bins.max())) + 2
        H, W = ny + 2 * pad, nx + 2 * pad
        x0 = np.floor(X).astype(int)
        y0 = np.floor(Y).astype(int)
        fx, fy = X - x0, Y - y0
        for bi in np.unique(bins):
            sel = bins == bi
            G = np.zeros(H * W)
            for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                xi, yi = x0[sel] + dx + pad, y0[sel] + dy + pad
                inb = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
                G += np.bincount((yi * W + xi)[inb], (w[sel] * ww[sel])[inb], minlength=H * W)
            G = _blur(G.reshape(H, W).astype(np.float32), lo * cfg.bin_ratio**bi)
            img[k] += G[pad:pad + ny, pad:pad + nx]
    for k in budget:
        budget[k]["image_unclipped"] = float(img[k].sum(dtype=np.float64))
    return img, budget


# ============================================================================ PNG I/O
def _png_bytes_rgb16(a):
    h, w, _ = a.shape
    raw = b"".join(b"\x00" + a[r].astype(">u2").tobytes() for r in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def write_tracer_png(path, conc):
    """conc (3, ny, nx) or (ny, nx, 3) floats in [0, 1] -> 16-bit RGB PNG (R, G, B = tracer 0, 1, 2)."""
    a = np.asarray(conc, float)
    if a.shape[0] == 3 and a.ndim == 3 and a.shape[-1] != 3:
        a = np.transpose(a, (1, 2, 0))
    a16 = (np.clip(a, 0, 1) * 65535 + 0.5).astype(np.uint16)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        if not cv2.imwrite(str(path), np.ascontiguousarray(a16[..., ::-1])):
            raise OSError(f"cv2 could not write {path}")
    except ImportError:
        Path(path).write_bytes(_png_bytes_rgb16(a16))
    return a16


def read_tracer_png(path):
    """16-bit (or 8-bit) RGB PNG -> float (ny, nx, 3) RGB in [0, 1]."""
    try:
        import cv2
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        im = im[..., :3][..., ::-1]
    except ImportError:
        from PIL import Image
        im = np.asarray(Image.open(path))
    scale = 65535.0 if im.dtype == np.uint16 else 255.0
    return im.astype(np.float64) / scale


# ============================================================================ driver
def advect_dye(vel: Callable, plane: Plane, frame_times, sources: Sequence[Source] = (),
               lines: Sequence[MaterialLine] = (), dt: float | None = None, n_sub: int = 8,
               cfg: RenderConfig | None = None, n_tracers: int | None = None):
    """Generator over frames: yields (frame_index, t, img (n_tracers, ny, nx) unclipped, budget, filaments).
    Particles are advected with RK4 substeps <= dt that land exactly on every frame time; each active source
    releases n_sub particles per substep (pre-advected with the source velocity across the substep)."""
    cfg = cfg or RenderConfig()
    frame_times = np.asarray(frame_times, float)
    dt = dt or (float(np.min(np.diff(frame_times))) if len(frame_times) > 1 else 1e-2)
    n_tr = n_tracers or 1 + max([s.tracer for s in sources] + [l_.tracer for l_ in lines] + [0])
    src_f = [_Filament(s.tracer, s.strength, s.sigma0 or cfg.sigma0) for s in sources]
    line_f = [None] * len(lines)
    t_starts = [frame_times[0] if s.t_start is None else s.t_start for s in sources]
    t_lines = [frame_times[0] if l_.t0 is None else l_.t0 for l_ in lines]
    t = min([frame_times[0]] + t_starts + t_lines)
    events = np.unique(np.concatenate([frame_times, t_starts, t_lines]))
    last_rel = [None] * len(sources)
    fi = 0
    while fi < len(frame_times):
        for i, l_ in enumerate(lines):
            if line_f[i] is None and t >= t_lines[i] - 1e-12:
                P = np.asarray(l_.points, float)
                f = _Filament(l_.tracer, l_.strength, l_.sigma0 or cfg.sigma0)
                f.append(P, np.full(len(P), t), np.concatenate([[0.0], np.hypot(*np.diff(P, axis=0).T)]))
                line_f[i] = f
        for i, s in enumerate(sources):
            if t_starts[i] - 1e-12 <= t < s.t_end:
                seed = np.asarray(s.xy, float)[None]
                if last_rel[i] is None:
                    src_f[i].append(seed, np.array([t]), np.zeros(1))
                else:
                    h = t - last_rel[i]
                    u = vel(t, seed)[0]
                    j = np.arange(n_sub - 1, -1, -1)
                    src_f[i].append(seed + u[None] * (j * h / n_sub)[:, None], t - j * h / n_sub,
                                    np.full(n_sub, np.hypot(*u) * h / n_sub))
                last_rel[i] = t
        fils = [f for f in src_f + line_f if f is not None]
        if abs(t - frame_times[fi]) < 1e-9 * max(1.0, abs(t)):
            img, budget = render_filaments(fils, t, plane, cfg, n_tr)
            yield fi, t, img, budget, fils
            fi += 1
            if fi == len(frame_times):
                break
        nxt = events[events > t + 1e-12]
        h = min(dt, (nxt[0] if len(nxt) else t + dt) - t)
        for f in fils:
            f.P = rk4_advect(vel, t, f.P, h)
        t = t + h


def write_dye_sequence(out_dir, vel: Callable, plane: Plane, frame_times, sources: Sequence[Source] = (),
                       lines: Sequence[MaterialLine] = (), frame_numbers=None, dt: float | None = None,
                       n_sub: int = 8, cfg: RenderConfig | None = None, n_tracers: int | None = None,
                       keep_images: bool = False) -> dict:
    """Run `advect_dye` and write 16-bit tracer PNGs (frame number n -> fNNNN.png) plus meta.json.
    Returns the meta dict (and 'images' (n_frames, n_tracers, ny, nx) clipped float when keep_images)."""
    cfg = cfg or RenderConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame_numbers = list(range(len(frame_times))) if frame_numbers is None else list(frame_numbers)
    n_tr = n_tracers or 1 + max([s.tracer for s in sources] + [l_.tracer for l_ in lines] + [0])
    groups = (n_tr + 2) // 3
    frames, images = [], []
    for fi, t, img, budget, _ in advect_dye(vel, plane, frame_times, sources, lines, dt, n_sub, cfg, n_tr):
        n = frame_numbers[fi]
        im = np.clip(img, 0, 1) if cfg.clip else img
        for g in range(groups):
            c = np.zeros((3, plane.ny, plane.nx))
            c[: min(3, n_tr - 3 * g)] = im[3 * g: 3 * g + 3]
            a16 = write_tracer_png((out if g == 0 else out / f"g{g}") / f"f{n:04d}.png", c)
            for ch in range(min(3, n_tr - 3 * g)):
                budget[3 * g + ch]["png"] = float(a16[..., ch].sum(dtype=np.float64) / 65535.0)
        for b in budget.values():
            b["outside_frame"] = b["drawn"] - b["image_unclipped"]
            b["clip_loss"] = b["image_unclipped"] - b.get("png", b["image_unclipped"])
        frames.append(dict(n=n, t=float(t), file=f"f{n:04d}.png", budget_px2={str(k): v for k, v in budget.items()}))
        if keep_images:
            images.append(im)
    meta = dict(
        plane=dict(x0=plane.x0, x1=plane.x1, y0=plane.y0, y1=plane.y1, px=plane.px, nx=plane.nx, ny=plane.ny),
        pixel_centre_rule=PIXEL_RULE,
        encoding="16-bit RGB PNG, channel k = tracer 3g + k, concentration 0..1; group 0 in ./, group g in ./g<g>/",
        tracers=[dict(index=k, group=k // 3, channel="RGB"[k % 3], dir="." if k < 3 else f"g{k // 3}") for k in range(n_tr)],
        frames=frames,
        formula="sigma^2 = sigma0^2/lam^2 + 2 D age + sigma_aa^2; c = strength*sigma_e0/(lam*sigma); "
                "sigma_e0^2 = sigma0^2 + sigma_aa^2; lam = ds/ds0 per segment",
        mass_units="budget in px^2 * concentration; multiply by px^2 (m^2) for world units",
        config=asdict(cfg) | dict(gap_max=None if math.isinf(cfg.gap_max) else cfg.gap_max, n_sub=n_sub, dt=dt),
        sources=[dict(xy=list(map(float, s.xy)), tracer=s.tracer, t_start=s.t_start,
                      t_end=None if math.isinf(s.t_end) else s.t_end, strength=s.strength) for s in sources],
        lines=[dict(n_points=len(l_.points), tracer=l_.tracer, t0=l_.t0, strength=l_.strength) for l_ in lines],
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    if keep_images:
        meta["images"] = np.array(images)
    return meta
