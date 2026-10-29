"""Steady-state soil organic carbon / radiocarbon forward model (null model).

1-D advection-diffusion-decay model for bulk soil organic carbon (SOC) and
its 14C content, in the tradition of Elzein & Balesdent (1995), Baisden et
al. (2002) and the vertically resolved soil C schemes of Koven et al.
(2013). Carbon stocks are at steady state; 14C is integrated through time
under the real atmospheric 14C history (IntCal20 before 1950, Hua et al.
2022 NH zone 1 bomb curve after 1950), so the bomb spike and past
atmospheric 14C variability are both included.

    dC/dt  = d/dz(D dC/dz) - v dC/dz - k(z) C + I(z)
    dC*/dt = d/dz(D dC*/dz) - v dC*/dz - (k(z) + lam) C* + I(z) Fatm(t)

C*/C is the modelled F14C. None of the parameters change through time:
by construction the model contains NO thermal history. Three null
families are provided:

  N0  smooth:      D, k decline smoothly (exponentially) with depth.
  N1  modern AL:   as N0, but mixing and decomposition are reduced below the
                   field-observed active-layer depth z_AL (the present-day
                   permafrost table, held fixed through time).
  N2  free floor:  as N1, but the depth of the mixing/decomposition floor
                   is a free parameter (diagnostic, not a null).

Units: depth cm, time yr, D cm2 yr-1, v cm yr-1, k yr-1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

LAM = 1.0 / 8267.0          # true 14C mean life (yr)
LIBBY = 8033.0              # Libby mean life used for conventional ages
DZ = 1.0
ZMAX = 300.0
Z = np.arange(DZ / 2, ZMAX, DZ)   # cell centres
N = Z.size

_EXT = Path(__file__).resolve().parent / "curves"


def _load_atm():
    """Return atmospheric F14C at time of growth versus calendar year AD.

    Values are decay-corrected to the time of growth. IntCal20 supplies
    the record before 1950 and the NH zone 1 post-bomb curve after it.
    """
    # Columns: cal BP, 14C age, error. Post-bomb cal BP values are negative.
    ic = np.loadtxt(_EXT / "3Col_intcal20.14C")
    pb = np.loadtxt(_EXT / "postbomb_NH1.14C")
    # IntCal20: F at growth = exp(-age/8033) * exp(calBP / 8267)
    yr_ic = 1950.0 - ic[:, 0]
    f_ic = np.exp(-ic[:, 1] / LIBBY) * np.exp(ic[:, 0] * LAM)
    yr_pb = 1950.0 - pb[:, 0]
    f_pb = np.exp(-pb[:, 1] / LIBBY)
    keep = yr_ic < 1950.0
    yr = np.concatenate([yr_ic[keep], yr_pb])
    f = np.concatenate([f_ic[keep], f_pb])
    o = np.argsort(yr)
    yr, f = yr[o], f[o]
    # extend past 2019.5 with the recent mean decline (~0.4 % yr-1, NH1)
    slope = (f[-1] - f[-6]) / (yr[-1] - yr[-6])
    ext = np.arange(yr[-1] + 1, 2026.0)
    f_ext = f[-1] + slope * (ext - yr[-1])
    return np.concatenate([yr, ext]), np.concatenate([f, f_ext])


ATM_YR, ATM_F = _load_atm()


def fatm(year: float) -> float:
    """Return atmospheric F14C at ``year`` by linear interpolation."""
    return float(np.interp(year, ATM_YR, ATM_F))


def _time_steps(t_end: float):
    """Variable time stepping: coarse in the deep past, annual after 1850."""
    edges = np.concatenate([
        np.arange(-48000.0, -8000.0, 100.0),
        np.arange(-8000.0, 1850.0, 20.0),
        np.arange(1850.0, t_end + 1e-9, 1.0),
    ])
    if edges[-1] < t_end:
        edges = np.append(edges, t_end)
    return edges


def profiles(p: dict, z_al: float | None, variant: str):
    """Return D(z), k(z) and the normalised input profile I(z).

    The input profile is returned as ``inputs`` to avoid the ambiguous
    single-letter name ``I`` (PEP 8, E741).
    """
    D = p["D0"] * np.exp(-Z / p["zD"])
    k = p["kmin"] + p["k0"] * np.exp(-Z / p["zk"])
    if variant in ("N1", "N2"):
        zf = z_al if variant == "N1" else p["zf"]
        below = Z > zf
        D = np.where(below, D * p["fD"], D)
        k = np.where(below, k * p["fk"], k)
    # inputs: fraction fs into the surface cell, rest distributed as roots
    inputs = (1 - p["fs"]) * np.exp(-Z / p["zr"])
    inputs = inputs / inputs.sum()
    inputs[0] += p["fs"]
    return D, k, inputs


def operator(D, k, v):
    """Finite-volume operator A (N x N) for diffusion + upwind advection - k.

    Zero-flux top boundary. At the bottom boundary the diffusive flux has
    zero gradient and advection flows out.
    """
    Dface = np.zeros(N + 1)
    Dface[1:-1] = 0.5 * (D[:-1] + D[1:])
    main = np.zeros(N)
    lower = np.zeros(N - 1)   # coefficient of C[i-1] in row i
    upper = np.zeros(N - 1)   # coefficient of C[i+1] in row i
    # diffusion
    main -= (Dface[:-1] + Dface[1:]) / DZ**2
    lower += Dface[1:-1] / DZ**2
    upper += Dface[1:-1] / DZ**2
    # advection (downward, v > 0), upwind
    main -= v / DZ
    lower += v / DZ
    main -= k
    return sp.diags([lower, main, upper], [-1, 0, 1], format="csc")


def run(p: dict, z_al: float | None, variant: str, t_sample: float):
    """Integrate to t_sample; return C (steady), Cstar (14C) on the grid."""
    D, k, inputs = profiles(p, z_al, variant)
    A = operator(D, k, p["v"])
    Iden = sp.identity(N, format="csc")
    # steady-state carbon
    C = splu(-A).solve(inputs)
    # 14C: start from steady state at the first edge
    Al = A - LAM * Iden
    edges = _time_steps(t_sample)
    Cs = splu(-Al).solve(inputs * fatm(edges[0]))
    lus = {}
    for t0, t1 in zip(edges[:-1], edges[1:]):
        dt = round(t1 - t0, 6)
        if dt not in lus:
            lus[dt] = splu((Iden - dt * Al).tocsc())
        Cs = lus[dt].solve(Cs + dt * inputs * fatm(0.5 * (t0 + t1)))
    return C, Cs


def increment_f(C, Cs, tops, bottoms):
    """Carbon-weighted bulk F14C over each sampled increment."""
    out = np.empty(len(tops))
    for i, (a, b) in enumerate(zip(tops, bottoms)):
        overlap = np.minimum(Z + DZ / 2, b) - np.maximum(Z - DZ / 2, a)
        w = np.clip(overlap, 0, None)
        out[i] = (w * Cs).sum() / (w * C).sum()
    return out


def grid_f(C, Cs):
    """Return modelled F14C on the 1-cm grid."""
    return Cs / C
