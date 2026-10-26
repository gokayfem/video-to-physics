"""Verification checks: convergence (observed order, Richardson, GCI), energy balance, periodicity per spatial band,
frequency / Strouhal, divergence, tracer mass budget, dye-advection scale fit and a negative-control harness.

Every check returns a plain dict  {check, passed, value, tol, note}  (see `result`) so results can be dumped to
json directly. All inputs are arrays; nothing here knows about a particular case. This module deliberately does not
import the solver-side modules (v2p.dye, v2p.elastica, v2p.ode) so it can serve as an independent verifier.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np


def result(check: str, passed: bool, value, tol, note: str = "") -> dict:
    return dict(check=check, passed=bool(passed), value=value, tol=tol, note=note)


# ============================================================================ convergence
def observed_order(f_fine: float, f_medium: float, f_coarse: float, r: float = 2.0, r32: float | None = None) -> float:
    """Observed order p from three solutions with refinement ratios r = h_medium/h_fine and r32 = h_coarse/h_medium
    (r32 defaults to r; non-constant ratios solved by fixed-point iteration, Celik et al. 2008)."""
    e21, e32 = f_medium - f_fine, f_coarse - f_medium
    if e21 == 0 or e32 == 0:
        return float("nan")
    r32 = r if r32 is None else r32
    p = abs(math.log(abs(e32 / e21))) / math.log(r)
    if r32 == r:
        return math.log(abs(e32 / e21)) / math.log(r)
    s = math.copysign(1.0, e32 / e21)
    for _ in range(100):
        q = math.log((r**p - s) / (r32**p - s))
        p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r)
        if abs(p_new - p) < 1e-12:
            break
        p = p_new
    return p


def richardson(f_fine: float, f_medium: float, r: float = 2.0, p: float = 2.0) -> float:
    """Richardson-extrapolated value f_fine + (f_fine - f_medium)/(r^p - 1)."""
    return f_fine + (f_fine - f_medium) / (r**p - 1.0)


def gci(f_fine: float, f_medium: float, f_coarse: float, r: float = 2.0, Fs: float = 1.25, p: float | None = None) -> dict:
    """Roache grid convergence index (safety factor Fs = 1.25 for three-grid studies).
    Returns p (observed unless given), f_ext, relative errors, GCI_fine, GCI_coarse and the asymptotic-range ratio
    GCI_coarse / (r^p GCI_fine) (~1 inside the asymptotic range)."""
    p = observed_order(f_fine, f_medium, f_coarse, r) if p is None else p
    e21 = abs((f_medium - f_fine) / f_fine)
    e32 = abs((f_coarse - f_medium) / f_medium)
    g21 = Fs * e21 / (r**p - 1.0)
    g32 = Fs * e32 / (r**p - 1.0)
    return dict(p=p, f_ext=richardson(f_fine, f_medium, r, p), e21=e21, e32=e32, gci_fine=g21, gci_coarse=g32,
                asymptotic_ratio=g32 / (r**p * g21) if g21 > 0 else float("nan"))


def check_convergence(values, r: float = 2.0, p_expected: float = 2.0, p_tol: float = 0.3, gci_tol: float | None = None,
                      name: str = "convergence") -> dict:
    """values = (coarse, medium, fine). Passes if |p - p_expected| <= p_tol (and GCI_fine <= gci_tol if given)."""
    f3, f2, f1 = values[-3:]
    g = gci(f1, f2, f3, r)
    ok = abs(g["p"] - p_expected) <= p_tol and (gci_tol is None or g["gci_fine"] <= gci_tol)
    return result(name, ok, g, dict(p_expected=p_expected, p_tol=p_tol, gci_tol=gci_tol))


# ============================================================================ energy
def cumulative_trapezoid(t, y):
    t, y = np.asarray(t, float), np.asarray(y, float)
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))])


def energy_balance(t, E, P_in=None, P_diss=None, W_in=None, W_diss=None, tol: float = 1e-3,
                   name: str = "energy_balance") -> dict:
    """Residual R(t) = E - E0 - W_in + W_diss, with works given directly or integrated (trapezoid) from powers.
    Passes if max|R| <= tol * scale, scale = max(|E - E0|, |W_in|, |W_diss|, tiny)."""
    E = np.asarray(E, float)
    zero = np.zeros_like(E)
    Win = np.asarray(W_in, float) if W_in is not None else (cumulative_trapezoid(t, P_in) if P_in is not None else zero)
    Wd = np.asarray(W_diss, float) if W_diss is not None else (cumulative_trapezoid(t, P_diss) if P_diss is not None else zero)
    R = E - E[0] - Win + Wd
    scale = max(np.max(np.abs(E - E[0])), np.max(np.abs(Win)), np.max(np.abs(Wd)), 1e-300)
    rel = float(np.max(np.abs(R)) / scale)
    return result(name, rel <= tol, dict(max_abs_residual=float(np.max(np.abs(R))), rel_residual=rel, scale=float(scale)), tol,
                  "R = E - E0 - W_in + W_diss relative to the largest energy/work magnitude")


# ============================================================================ periodicity
def period_edges(t, s=None, period: float | None = None):
    """Period boundaries from mean-removed up-crossings of a reference signal s, or from a known period."""
    t = np.asarray(t, float)
    if s is not None:
        x = np.asarray(s, float) - np.mean(s)
        i = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
        return t[i] - x[i] * (t[i + 1] - t[i]) / (x[i + 1] - x[i])
    n = int((t[-1] - t[0]) / period + 1e-9)
    return t[0] + period * np.arange(n + 1)


def change_per_period(t, series, edges):
    """Per-period means of series and their relative change from one period to the next."""
    t, series = np.asarray(t, float), np.asarray(series, float)
    means = np.array([series[(t >= a) & (t < b)].mean() for a, b in zip(edges[:-1], edges[1:])])
    return means, means[1:] / means[:-1] - 1.0


def band_fluctuation_energy(t, x, u, v, bands: dict, edges):
    """Per period and band: area-mean of |u - <u>_period|^2 over the period (u, v: (nt, ny, nx); bands: name ->
    (x_lo, x_hi)). Returns {band: (n_periods,) array}."""
    t, x = np.asarray(t, float), np.asarray(x, float)
    out = {k: [] for k in bands}
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (t >= a) & (t < b)
        U, V = u[sel], v[sel]
        fl = ((U - U.mean(0)) ** 2 + (V - V.mean(0)) ** 2).mean(0)
        for k, (lo, hi) in bands.items():
            out[k].append(float(fl[:, (x >= lo) & (x <= hi)].mean()))
    return {k: np.array(v_) for k, v_ in out.items()}


def periodicity_per_band(per_period: dict, tol: float = 0.02, name: str = "periodicity_per_band") -> dict:
    """per_period: {band: per-period values}. Passes if every band changes by <= tol per period (last periods:
    all consecutive pairs are checked)."""
    val = {}
    for k, m in per_period.items():
        m = np.asarray(m, float)
        ch = m[1:] / m[:-1] - 1.0
        val[k] = dict(max_abs_change=float(np.max(np.abs(ch))) if len(ch) else float("nan"), changes=ch.tolist())
    ok = all(np.isfinite(d["max_abs_change"]) and d["max_abs_change"] <= tol for d in val.values())
    return result(name, ok, val, tol, "relative change of each band's per-period value between consecutive periods")


# ============================================================================ frequency / Strouhal
def zero_crossing_frequency(t, s, hysteresis: float = 0.5) -> float:
    """Mean up-crossing rate of the mean-removed signal. An up-crossing counts only after the signal was below
    -hysteresis*std and then rises above +hysteresis*std (robust to noise); its time is the last zero up-crossing
    in that rise, linearly interpolated."""
    t = np.asarray(t, float)
    x = np.asarray(s, float) - np.mean(s)
    hb = hysteresis * np.std(x)
    tc, armed, last = [], False, None
    for i in range(len(x) - 1):
        if x[i] < -hb:
            armed = True
        if x[i] < 0 <= x[i + 1]:
            last = t[i] - x[i] * (t[i + 1] - t[i]) / (x[i + 1] - x[i])
        if armed and x[i + 1] > hb and last is not None:
            tc.append(last)
            armed = False
    if len(tc) < 2:
        return float("nan")
    return (len(tc) - 1) / (tc[-1] - tc[0])


def spectrum_peak_frequency(t, s, pad: int = 16) -> float:
    """Hann-windowed, zero-padded FFT peak with parabolic interpolation (uniform sampling assumed)."""
    x = np.asarray(s, float) - np.mean(s)
    dt = float(np.median(np.diff(t)))
    N = pad * len(x)
    S = np.abs(np.fft.rfft(x * np.hanning(len(x)), N))
    f = np.fft.rfftfreq(N, dt)
    k = int(np.argmax(S[1:]) + 1)
    if 0 < k < len(S) - 1:
        a, b, c = np.log(S[k - 1:k + 2] + 1e-300)
        k = k + 0.5 * (a - c) / (a - 2 * b + c)
    return float(k * (f[1] - f[0]))


def sine_fit_frequency(t, s, f0: float, span: float = 0.3, n: int = 401) -> float:
    """Least-squares fit of a sin + b cos + c over a frequency grid around f0, refined by golden-section search
    (unbiased for short, non-integer-cycle windows)."""
    t = np.asarray(t, float)
    s = np.asarray(s, float)
    tc = t - t.mean()

    def sse(f):
        A = np.c_[np.sin(2 * np.pi * f * tc), np.cos(2 * np.pi * f * tc), np.ones_like(tc)]
        r = s - A @ np.linalg.lstsq(A, s, rcond=None)[0]
        return float(r @ r)

    fs = np.linspace((1 - span) * f0, (1 + span) * f0, n)
    k = int(np.argmin([sse(f) for f in fs]))
    a, b = fs[max(k - 1, 0)], fs[min(k + 1, n - 1)]
    g = 0.5 * (math.sqrt(5) - 1)
    for _ in range(60):
        c, d = b - g * (b - a), a + g * (b - a)
        if sse(c) < sse(d):
            b = d
        else:
            a = c
    return 0.5 * (a + b)


def dominant_frequency(t, s) -> dict:
    """Three independent estimates: spectrum peak, sine fit (seeded by the spectrum), mean zero-crossing rate."""
    f_fft = spectrum_peak_frequency(t, s)
    return dict(f_fit=sine_fit_frequency(t, s, f_fft), f_fft=f_fft, f_zero=zero_crossing_frequency(t, s))


def strouhal(t, s, length: float, speed: float) -> dict:
    """St = f L / U for the three frequency estimates of `dominant_frequency`."""
    f = dominant_frequency(t, s)
    return {k.replace("f_", "St_"): v * length / speed for k, v in f.items()} | f


def check_frequency(t, s, f_ref: float, tol: float = 0.02, agree_tol: float = 0.02, name: str = "frequency") -> dict:
    """Passes if the sine-fit frequency is within tol of f_ref and the three estimates agree within agree_tol."""
    f = dominant_frequency(t, s)
    est = [f["f_fit"], f["f_fft"], f["f_zero"]]
    spread = (max(est) - min(est)) / f["f_fit"]
    rel = abs(f["f_fit"] / f_ref - 1)
    return result(name, rel <= tol and spread <= agree_tol, f | dict(rel_err=rel, spread=spread, f_ref=f_ref),
                  dict(tol=tol, agree_tol=agree_tol))


# ============================================================================ fields
def divergence_ratio(x, y, u, v, mask=None) -> float:
    """rms(div u) / rms(vorticity) with second-order central differences; u, v: (ny, nx) or (nt, ny, nx) on the
    tensor grid (x (nx,), y (ny,)); mask (ny, nx) selects the evaluated points (e.g. away from walls)."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    if u.ndim == 2:
        u, v = u[None], v[None]
    du_dx = np.gradient(u, x, axis=2)
    dv_dy = np.gradient(v, y, axis=1)
    om = np.gradient(v, x, axis=2) - np.gradient(u, y, axis=1)
    div = du_dx + dv_dy
    m = np.ones(u.shape[1:], bool) if mask is None else np.asarray(mask, bool)
    m = m.copy()
    m[[0, -1], :] = False
    m[:, [0, -1]] = False
    return float(np.sqrt(np.mean(div[:, m] ** 2) / max(np.mean(om[:, m] ** 2), 1e-300)))


def check_divergence(x, y, u, v, mask=None, tol: float = 0.05, name: str = "divergence_ratio") -> dict:
    r = divergence_ratio(x, y, u, v, mask)
    return result(name, r <= tol, r, tol, "rms(div u)/rms(curl u), central differences, boundary ring excluded")


def tracer_mass_budget(t, mass, outflux=None, injected=None, tol: float = 0.05, name: str = "tracer_mass_budget") -> dict:
    """Tracer bookkeeping. I(t) = mass(t) + cumulative outflow. With `injected` (array, expected cumulative
    injection) the check is max|I - injected| / max(injected); otherwise I(t) must be linear in t (constant-rate
    sources): max deviation from the LS line / total range."""
    t = np.asarray(t, float)
    mass = np.asarray(mass, float)
    I = mass + (cumulative_trapezoid(t, outflux) if outflux is not None else 0.0)
    if injected is not None:
        inj = np.asarray(injected, float)
        dev = float(np.max(np.abs(I - inj)) / max(np.max(np.abs(inj)), 1e-300))
        note = "max |mass + outflow - injected| / max injected"
    else:
        p = np.polyfit(t, I, 1)
        dev = float(np.max(np.abs(I - np.polyval(p, t))) / max(np.ptp(I), 1e-300))
        note = "max deviation of mass + outflow from a straight line / its range"
    return result(name, dev <= tol, dev, tol, note)


# ============================================================================ dye advection
def grid_velocity(t, x, y, u, v) -> Callable:
    """vel(tq, X, Y) -> (u, v) from a history u, v (nt, ny, nx) on ascending x, y: bilinear in space (clamped at the
    border), linear in time."""
    t, x, y = (np.asarray(a, float) for a in (t, x, y))

    def bil(F, X, Y):
        fi = np.clip((X - x[0]) / (x[1] - x[0]), 0, len(x) - 1.000001)
        fj = np.clip((Y - y[0]) / (y[1] - y[0]), 0, len(y) - 1.000001)
        i, j = fi.astype(int), fj.astype(int)
        a, b = fi - i, fj - j
        return (F[j, i] * (1 - a) * (1 - b) + F[j, i + 1] * a * (1 - b) + F[j + 1, i] * (1 - a) * b + F[j + 1, i + 1] * a * b)

    def vel(tq, X, Y):
        if len(t) == 1:
            return bil(u[0], X, Y), bil(v[0], X, Y)
        k = int(np.clip(np.searchsorted(t, tq, side="right") - 1, 0, len(t) - 2))
        w = float(np.clip((tq - t[k]) / (t[k + 1] - t[k]), 0, 1))
        return ((1 - w) * bil(u[k], X, Y) + w * bil(u[k + 1], X, Y), (1 - w) * bil(v[k], X, Y) + w * bil(v[k + 1], X, Y))
    return vel


def _sample(img, xs, ys, X, Y):
    fi = np.clip((X - xs[0]) / (xs[1] - xs[0]), 0, len(xs) - 1.000001)
    fj = np.clip((Y - ys[0]) / (ys[1] - ys[0]), 0, len(ys) - 1.000001)
    i, j = fi.astype(int), fj.astype(int)
    a, b = fi - i, fj - j
    return (img[j, i] * (1 - a) * (1 - b) + img[j, i + 1] * a * (1 - b) + img[j + 1, i] * (1 - a) * b
            + img[j + 1, i + 1] * a * b)


def dye_advection_scale(c0, c1, xs, ys, vel: Callable, t0: float, t1: float, scales=(0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5),
                        n_sub: int = 8, grad_min: float = 0.02, c_min: float = 0.1, max_points: int = 3000, seed: int = 0) -> dict:
    """Advect dye-edge pixel centres of image c0 (time t0) with s*(u, v) (RK2, n_sub steps) to t1, sample c1 there
    and compare with c0. The best scale s (parabolic refinement) should be ~1 if the dye moves with the velocity.
    c0, c1: (ny, nx) scalar images on pixel-centre coordinates xs (nx,), ys (ny,) (either sign of spacing)."""
    c0, c1 = np.asarray(c0, float), np.asarray(c1, float)
    gy, gx = np.gradient(c0)
    cand = np.argwhere((np.hypot(gx, gy) > grad_min) & (c0 > c_min))
    if len(cand) == 0:
        return dict(best_scale=float("nan"), n_points=0, errors={})
    rng = np.random.default_rng(seed)
    cand = cand[rng.choice(len(cand), min(max_points, len(cand)), replace=False)]
    X0, Y0, cv = xs[cand[:, 1]], ys[cand[:, 0]], c0[cand[:, 0], cand[:, 1]]
    errs = []
    h = (t1 - t0) / n_sub
    for s in scales:
        X, Y = X0.copy(), Y0.copy()
        for k in range(n_sub):
            tk = t0 + k * h
            u1, v1 = vel(tk, X, Y)
            u2, v2 = vel(tk + 0.5 * h, X + 0.5 * h * s * u1, Y + 0.5 * h * s * v1)
            X, Y = X + h * s * u2, Y + h * s * v2
        errs.append(float(np.mean(np.abs(_sample(c1, xs, ys, X, Y) - cv))))
    errs = np.array(errs)
    k = int(np.argmin(errs))
    best = float(scales[k])
    if 0 < k < len(scales) - 1:
        A = np.polyfit(scales[k - 1:k + 2], errs[k - 1:k + 2], 2)
        if A[0] > 0:
            best = float(-A[1] / (2 * A[0]))
    return dict(best_scale=best, n_points=int(len(cand)), errors=dict(zip(map(str, scales), errs.round(6).tolist())))


def check_dye_advection(c0, c1, xs, ys, vel, t0, t1, lo: float = 0.85, hi: float = 1.15, name="dye_advection_scale", **kw):
    r = dye_advection_scale(c0, c1, xs, ys, vel, t0, t1, **kw)
    return result(name, lo <= r["best_scale"] <= hi, r, [lo, hi], "best-fit velocity scale advecting dye edges between frames")


# ============================================================================ negative controls
def _scale_key(inputs, key="t", factor=1.1):
    out = dict(inputs)
    out[key] = np.asarray(inputs[key], float) * factor
    return out


def _add_noise(inputs, key, sigma=0.1, seed=0):
    out = dict(inputs)
    a = np.asarray(inputs[key], float)
    out[key] = a + sigma * np.std(a) * np.random.default_rng(seed).standard_normal(a.shape)
    return out


def _shift_key(inputs, key, offset=0.0):
    out = dict(inputs)
    out[key] = np.asarray(inputs[key], float) + offset
    return out


def _reverse_key(inputs, key):
    out = dict(inputs)
    out[key] = np.asarray(inputs[key])[::-1].copy()
    return out


def _drift(inputs, key, rate=0.01, t_key="t"):
    out = dict(inputs)
    t = np.asarray(inputs[t_key], float)
    a = np.asarray(inputs[key], float)
    out[key] = a * (1 + rate * (t - t[0]) / max(t[-1] - t[0], 1e-300)).reshape((-1,) + (1,) * (a.ndim - 1))
    return out


PERTURBATIONS: dict[str, Callable] = {
    "scale": _scale_key,       # key *= factor (e.g. t *= 1.1 -> 10 % frequency error, u *= 1.2 -> wrong speed)
    "noise": _add_noise,       # key += sigma * std * N(0, 1)
    "shift": _shift_key,       # key += offset
    "reverse": _reverse_key,   # key reversed along axis 0
    "drift": _drift,           # key grows linearly by `rate` over the record
}


def negative_control(check: Callable, inputs: dict, perturbation: str | Callable, expect_fail: bool = True,
                     **perturb_kw) -> dict:
    """Run `check(**inputs)` on clean and perturbed inputs. The control passes when the clean run passes and the
    perturbed run fails (or passes, with expect_fail=False). perturbation: a PERTURBATIONS name or a callable
    inputs -> inputs."""
    fn = PERTURBATIONS[perturbation] if isinstance(perturbation, str) else perturbation
    clean = check(**inputs)
    bad = check(**fn(inputs, **perturb_kw))
    ok = clean["passed"] and (bad["passed"] != expect_fail)
    label = perturbation if isinstance(perturbation, str) else getattr(perturbation, "__name__", "custom")
    return result(f"negative_control[{clean['check']}|{label}{perturb_kw or ''}]", ok,
                  dict(clean_passed=clean["passed"], perturbed_passed=bad["passed"], perturbed_value=bad["value"]),
                  "perturbed must fail" if expect_fail else "perturbed must pass")


def run_negative_controls(check: Callable, inputs: dict, controls) -> list:
    """controls: iterable of (perturbation, kwargs) or (perturbation, kwargs, expect_fail)."""
    out = []
    for c in controls:
        name, kw = c[0], (c[1] if len(c) > 1 else {})
        out.append(negative_control(check, inputs, name, expect_fail=c[2] if len(c) > 2 else True, **kw))
    return out
