"""Kneedle knee detection on CROPS cost curves.

Faithful to Satopaa et al. (2011) Section III. For convex decreasing
curves (positive concavity) the paper's inversion replaces y_i with
y_max - y_i before applying the difference-curve logic.

Reference
---------
Satopaa V et al. (2011) ICDCS Workshops, IEEE.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator, UnivariateSpline

from changepoint_analysis.config import (
    ELBOW_TOLERANCE,
    KNEEDLE_SMOOTHING,
    S_VALUES,
)


@dataclass
class KneedleResult:
    """Result of one Kneedle run on a (k, cost) curve.

    Attributes
    ----------
    ks:
        Number of changepoints, sorted ascending.
    costs:
        Segmentation cost at each k.
    k_norm:
        k normalised to [0, 1].
    cost_norm:
        Cost normalised to [0, 1] after Kneedle inversion. Runs from 0
        (at lowest k, highest raw cost) to 1 (at highest k, lowest raw
        cost). The normalised curve therefore goes from (0, 0) to (1, 1).
    diff:
        Difference curve: ``cost_norm - k_norm``. Peaks at the knee.
    knee_ks_per_S:
        Mapping from sensitivity S to the list of detected knee k values.
    knee_ks:
        Union of detected knees across the S sweep.
    knee_region_ks:
        Tolerance-defined near-knee region, capped at the supplied
        ``max_k``. This is an operationally defined heuristic band, not
        a derived confidence region.
    tolerance:
        Tolerance value used to define the knee region.
    peak_diff:
        Diff value at the anchor (knee or global maximum). NaN when the
        curve is too short or flat for knee detection.
    """

    ks: np.ndarray
    costs: np.ndarray
    k_norm: np.ndarray
    cost_norm: np.ndarray
    diff: np.ndarray
    knee_ks_per_S: dict[float, list[int]]
    knee_ks: list[int]
    knee_region_ks: list[int]
    tolerance: float
    peak_diff: float

    def region_at_tolerance(
        self,
        tolerance: float,
        max_k: int | None = None,
    ) -> list[int]:
        """Return the knee region at a different tolerance without recomputing.

        The normalisation, smoothing, difference curve, and S sweep are
        independent of tolerance. This method reuses those cached results
        to derive the knee region at any tolerance.

        Parameters
        ----------
        tolerance:
            Knee region width as a fraction of the peak diff.
        max_k:
            Stratigraphic cap on knee region membership.

        Returns
        -------
        list of int
            Sorted unique k values whose diff value is at least
            ``peak_diff * (1 - tolerance)``.
        """
        if not np.isfinite(self.peak_diff) or self.peak_diff <= 0:
            return []
        region_mask = self.diff >= self.peak_diff * (1.0 - tolerance)
        if max_k is not None:
            region_mask = region_mask & (self.ks.astype(int) <= int(max_k))
        return sorted(int(k) for k, m in zip(self.ks, region_mask) if m)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _smoothing_spline(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Apply a smoothing spline to y(x).

    Used only when ``KNEEDLE_SMOOTHING`` is ``True``. Falls back to PCHIP
    when fewer than four points are available or when y is flat.
    """
    if len(x) < 4:
        return PchipInterpolator(x, y)(x)
    variance = float(np.var(y))
    if variance <= 0:
        return PchipInterpolator(x, y)(x)
    spline = UnivariateSpline(x, y, s=len(x) * variance * 0.01)
    return spline(x)


def _interior_local_maxima(diff: np.ndarray) -> np.ndarray:
    """Return a boolean mask of interior local maxima of diff.

    True at index i when ``diff[i] > diff[i-1]`` and ``diff[i] > diff[i+1]``.
    Endpoint indices are always False.
    """
    is_max = np.zeros_like(diff, dtype=bool)
    if len(diff) >= 3:
        is_max[1:-1] = (diff[1:-1] > diff[:-2]) & (diff[1:-1] > diff[2:])
    return is_max


def _kneedle_traverse(
    diff: np.ndarray,
    is_local_max: np.ndarray,
    k_norm: np.ndarray,
    S: float,
) -> list[int]:
    """Forward-confirm traversal (Satopaa et al. 2011, step 6).

    For each interior local maximum at index i, set threshold
    ``T_i = diff[i] - S * mean_dx``. Walk forward. A knee is confirmed
    at i when diff drops below T_i before the next local maximum. A local
    minimum that subsequently rises again invalidates the candidate.
    """
    n = len(diff)
    if n < 3:
        return []
    mean_dx = float(np.mean(np.diff(k_norm))) if n > 1 else 1.0

    knees: list[int] = []
    i = 1
    while i < n - 1:
        if not is_local_max[i]:
            i += 1
            continue

        threshold = diff[i] - S * mean_dx
        j = i + 1
        confirmed = False
        prev = diff[i]

        while j < n:
            if is_local_max[j]:
                break
            if diff[j] < threshold:
                knees.append(i)
                confirmed = True
                break
            if j + 1 < n and diff[j] < prev and diff[j] < diff[j + 1]:
                break
            prev = diff[j]
            j += 1

        # Boundary case: curve falls below threshold at the final index.
        # The inner loop can exit by exhaustion, local-max break, or
        # local-min break. The last two paths can leave a confirmed knee
        # undetected if diff[-1] < threshold at termination.
        if not confirmed and j >= n - 1 and diff[-1] < threshold:
            knees.append(i)

        i = max(j, i + 1)
    return knees


def _empty_result(
    ks: np.ndarray,
    costs: np.ndarray,
    tolerance: float,
) -> KneedleResult:
    """Construct a degenerate KneedleResult for too-short or flat inputs."""
    return KneedleResult(
        ks=ks,
        costs=costs,
        k_norm=ks,
        cost_norm=costs,
        diff=np.zeros_like(ks),
        knee_ks_per_S={},
        knee_ks=[],
        knee_region_ks=[],
        tolerance=tolerance,
        peak_diff=float("nan"),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def kneedle_knee(
    ks: np.ndarray,
    costs: np.ndarray,
    S_values: tuple = S_VALUES,
    tolerance: float = ELBOW_TOLERANCE,
    max_k: int | None = None,
    smooth: bool = KNEEDLE_SMOOTHING,
) -> KneedleResult:
    """Detect a knee point and knee region on a convex decreasing curve.

    Faithful to Satopaa et al. 2011 Section III, with the knee inversion
    and the forward-confirm traversal of step 6.

    For repeated queries at different tolerances on the same curve, call
    once and then use ``result.region_at_tolerance(tol, max_k)`` to
    derive new regions without redoing the normalisation work.

    Parameters
    ----------
    ks:
        Number of changepoints. Must be sorted ascending and unique.
    costs:
        Total segmentation cost at each k. Must be non-increasing.
    S_values:
        Sensitivity grid for knee detection.
    tolerance:
        Knee region width. ``tolerance=0.20`` includes all k whose diff
        value is at least 80 % of the peak diff.
    max_k:
        Stratigraphic cap on knee region membership. Set to
        ``N_SEGMENTS - 1`` to exclude physically impossible candidates.
        The knee detection itself is not capped.
    smooth:
        If ``True``, apply a smoothing spline before knee detection.

    Returns
    -------
    KneedleResult
        Empty result (with zero-filled diff) when fewer than three
        distinct k values are available or when the curve is flat.
    """
    ks = np.asarray(ks, dtype=float)
    costs = np.asarray(costs, dtype=float)

    if len(ks) < 3:
        return _empty_result(ks, costs, tolerance)

    y_raw = _smoothing_spline(ks, costs) if smooth else costs.copy()
    k_span = ks.max() - ks.min()
    c_span = y_raw.max() - y_raw.min()
    if k_span <= 0 or c_span <= 0:
        return _empty_result(ks, costs, tolerance)

    k_norm = (ks - ks.min()) / k_span
    # Kneedle inversion (Satopaa et al. 2011, Section III). The raw curve
    # is convex decreasing. The inversion replaces cost with (cost_max -
    # cost) so the normalised curve goes from (0, 0) to (1, 1) and the
    # difference curve diff = cost_norm - k_norm peaks where the
    # normalised curve bulges furthest above the diagonal — the knee.
    cost_norm = (y_raw.max() - y_raw) / c_span
    diff = cost_norm - k_norm

    is_local_max = _interior_local_maxima(diff)

    knee_ks_per_S: dict[float, list[int]] = {}
    for S in S_values:
        indices = _kneedle_traverse(diff, is_local_max, k_norm, S)
        knee_ks_per_S[S] = sorted(set(int(ks[i]) for i in indices))

    knee_ks = (
        sorted(set().union(*knee_ks_per_S.values()))
        if knee_ks_per_S else []
    )

    # Anchor at the detected knee with the highest diff value, falling
    # back to the global diff maximum when no knee was detected.
    if knee_ks:
        anchor_idx = int(np.where(ks.astype(int) == knee_ks[0])[0][0])
        for k_val in knee_ks:
            matches = np.where(ks.astype(int) == k_val)[0]
            if len(matches) == 0:
                continue
            idx = int(matches[0])
            if diff[idx] > diff[anchor_idx]:
                anchor_idx = idx
    else:
        anchor_idx = int(np.argmax(diff))

    peak_diff = float(diff[anchor_idx])
    if peak_diff <= 0:
        return _empty_result(ks, costs, tolerance)

    region_mask = diff >= peak_diff * (1.0 - tolerance)
    if max_k is not None:
        region_mask = region_mask & (ks.astype(int) <= int(max_k))
    knee_region_ks = sorted(int(k) for k, m in zip(ks, region_mask) if m)

    return KneedleResult(
        ks=ks,
        costs=costs,
        k_norm=k_norm,
        cost_norm=cost_norm,
        diff=diff,
        knee_ks_per_S=knee_ks_per_S,
        knee_ks=knee_ks,
        knee_region_ks=knee_region_ks,
        tolerance=tolerance,
        peak_diff=peak_diff,
    )
