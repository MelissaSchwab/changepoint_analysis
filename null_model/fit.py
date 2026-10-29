"""Fit the steady-state null model to one core's SOC-F14C profile.

Usage: python fit.py CORE VARIANT [N_STARTS]
Writes fits/CORE_VARIANT.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import forward as fw
from cores import CORES

HERE = Path(__file__).resolve().parent

warnings.filterwarnings("ignore")

# name, transform, lower, upper (in fitted space)
BASE = [
    ("D0", "log", -3.0, 1.5),     # surface mixing, cm2/yr
    ("zD", "log", 0.0, 3.0),      # mixing e-folding depth, cm
    ("k0", "log", -3.5, 0.0),     # surface decomposition rate, 1/yr
    ("zk", "log", 0.0, 2.5),      # decomposition e-folding depth, cm
    ("kmin", "log", -7.0, -3.0),  # deep decomposition floor, 1/yr
    ("fs", "lin", 0.0, 1.0),      # fraction of input at surface
    ("zr", "log", 0.0, 2.0),      # root input e-folding depth, cm
    ("v", "log", -4.0, -1.0),     # burial (advection) velocity, cm/yr
]
FLOOR = [
    ("fD", "log", -4.0, 0.0),     # mixing multiplier below floor
    ("fk", "log", -3.0, 0.0),     # decomposition multiplier below floor
]


def spec(variant, z_al):
    """Return the parameter specification for a null-model variant.

    Each entry is ``(name, transform, lower, upper)``. N1 and N2 add the
    floor multipliers. N2 also adds the free floor depth ``zf``.
    """
    s = list(BASE)
    if variant in ("N1", "N2"):
        s += FLOOR
    if variant == "N2":
        s += [("zf", "lin", max(5.0, z_al - 15.0), 180.0)]
    return s


def unpack(x, s):
    """Map a fitted-space vector to a parameter dict in physical units."""
    return {n: (10 ** v if t == "log" else v)
            for (n, t, _, _), v in zip(s, x)}


def _to_fitted(d, s):
    """Map a parameter dict to a fitted-space vector (inverse of unpack)."""
    return np.array([np.log10(d[n]) if t == "log" else d[n]
                     for (n, t, _, _) in s])


def residuals(x, s, core, variant):
    """Return thickness-weighted F14C residuals for one parameter vector.

    Residuals are scaled by the square root of increment thickness, so
    their sum of squares is the thickness-weighted RSS.
    """
    p = unpack(x, s)
    C, Cs = fw.run(p, core["z_al"], variant, core["t_sample"])
    fm = fw.increment_f(C, Cs, core["tops"], core["bottoms"])
    w = np.sqrt(core["bottoms"] - core["tops"])
    return (fm - core["fm"]) * w


def fit(core_name, variant, n_starts=12, seed=1):
    """Fit one null variant to one core by multistart least squares.

    Starts include random draws inside the bounds, warm starts from the
    next simpler variant and any earlier fit of the same variant. The
    best solution is written to ``fits/CORE_VARIANT.json`` and returned.
    """
    core = CORES[core_name]
    s = spec(variant, core["z_al"])
    lo = np.array([a for (_, _, a, _) in s])
    hi = np.array([b for (_, _, _, b) in s])
    rng = np.random.default_rng(seed)
    best = None
    starts = [lo + rng.uniform(0.05, 0.95, lo.size) * (hi - lo)
              for _ in range(n_starts)]
    # Warm starts from the nested simpler model (N0 -> N1 -> N2), so that
    # the richer null can never fit worse than the simpler one.
    prev = {"N1": "N0", "N2": "N1"}.get(variant)
    prev_path = HERE / f"fits/{core_name}_{prev}.json"
    if prev and prev_path.exists():
        pp = json.loads(prev_path.read_text())["params"]
        pp.setdefault("fD", 1.0)
        pp.setdefault("fk", 1.0)

        def tox(d):
            return np.clip(_to_fitted(d, s), lo + 1e-9, hi - 1e-9)

        if variant == "N1":
            starts = [tox(pp)] + starts
        else:
            warm = []
            floor_grid = np.arange(max(5.0, core["z_al"] - 15.0), 180.0, 10.0)
            for zf in [core["z_al"]] + list(floor_grid):
                d = dict(pp)
                d["zf"] = float(zf)
                warm.append(tox(d))
            starts = warm + starts
    # Also restart from any earlier fit of this same variant, so a better
    # optimum found in an earlier run is never lost.
    own = HERE / f"fits/{core_name}_{variant}.json"
    extra = HERE / f"fits/{core_name}_{variant}_seed.json"
    for src in (own, extra):
        if src.exists():
            d = json.loads(src.read_text())["params"]
            if all(n in d for (n, _, _, _) in s):
                x = _to_fitted(d, s)
                starts = [np.clip(x, lo + 1e-9, hi - 1e-9)] + starts
    for i, x0 in enumerate(starts):
        try:
            r = least_squares(residuals, x0, bounds=(lo, hi),
                              args=(s, core, variant), x_scale="jac",
                              max_nfev=400, diff_step=1e-3)
        except Exception as e:  # noqa: BLE001
            print("start", i, "failed", e, flush=True)
            continue
        print(f"{core_name} {variant} start {i}: wRSS={2 * r.cost:.5f}",
              flush=True)
        if best is None or r.cost < best.cost:
            best = r
    p = unpack(best.x, s)
    C, Cs = fw.run(p, core["z_al"], variant, core["t_sample"])
    fm_mod = fw.increment_f(C, Cs, core["tops"], core["bottoms"])
    w = core["bottoms"] - core["tops"]
    out = {
        "core": core_name, "variant": variant, "params": p,
        "n_params": len(s),
        "wrss": float(np.sum(w * (fm_mod - core["fm"]) ** 2)),
        "fm_model": fm_mod.tolist(),
        "grid_F": (Cs / C).tolist(), "grid_C": (C / C.max()).tolist(),
    }
    (HERE / "fits").mkdir(exist_ok=True)
    (HERE / f"fits/{core_name}_{variant}.json").write_text(
        json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    n_starts_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    fit(sys.argv[1], sys.argv[2], n_starts_arg)
