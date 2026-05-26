"""PELT changepoint detection and the CROPS penalty-range algorithm.

CROPS (Haynes, Eckley & Fearnhead 2017) runs PELT over a continuous range
of penalties rather than a single fixed value. It returns the set of
distinct optimal segmentations, one per penalty interval, without
repeating equivalent solutions.

All PELT cost computation uses the interpolated 1-cm grid. Breakpoints are
snapped to measured segment boundaries before any inference, and all
statistical tests operate on the original measured segment values.

References
----------
Haynes K, Eckley IA, Fearnhead P (2017) J Comput Graph Stat 26(1).
Killick R, Fearnhead P, Eckley IA (2012) J Am Stat Assoc 107.
"""
from __future__ import annotations

import bisect
import warnings

import numpy as np
import ruptures as rpt

from changepoint_analysis.config import (
    BETA_MAX_HEADROOM,
    BETA_MAX_RETRIES,
    COST_MODELS,
    PELT_JUMP,
    PELT_MIN_SIZE,
)

# Scale-aware tolerance for penalty comparisons. Two penalty values closer
# than this fraction of their magnitude are treated as identical. Machine
# epsilon (~2e-16) is too tight for penalty scales that may reach 1e6+.
_BETA_RTOL: float = 1e-9


def _beta_close(a: float, b: float) -> bool:
    """Return True when two penalty values are effectively identical.

    Uses a scale-aware absolute tolerance rather than ``np.isclose``
    defaults, which are inappropriate for large penalty ranges.
    """
    tol = _BETA_RTOL * max(abs(a), abs(b), 1.0)
    return abs(a - b) <= tol


# ── PELT helpers ──────────────────────────────────────────────────────────────

def build_pelt(model: str, signal: np.ndarray) -> rpt.Pelt:
    """Construct and fit a PELT instance for a given cost model.

    Parameters
    ----------
    model:
        ruptures cost model name. Must be one of the models in
        ``config.COST_MODELS``; this is enforced at runtime.
    signal:
        1-D array of F14C values on the 1-cm grid.

    Returns
    -------
    rpt.Pelt
        Fitted PELT instance. The fit is performed once here; callers
        call ``algo.predict(pen=beta)`` for each penalty without
        re-fitting. ``predict`` does not mutate the fitted state, so the
        same instance is safe to reuse across the CROPS penalty sweep.

    Raises
    ------
    ValueError
        If ``model`` is not in ``config.COST_MODELS``.

    Notes
    -----
    RBF results are treated as exploratory rather than primary inference:
    the RBF kernel assumes smooth similarity structure in feature space,
    which is difficult to justify for sparse, piecewise-constant,
    irregularly sampled stratigraphic data.
    """
    if model not in COST_MODELS:
        raise ValueError(
            f"model={model!r} is not in config.COST_MODELS {COST_MODELS}. "
            f"If 'rbf' is missing, ruptures CostRbf is unavailable on this "
            f"installation."
        )
    algo = rpt.Pelt(model=model, min_size=PELT_MIN_SIZE, jump=PELT_JUMP)
    return algo.fit(signal)


def _run_pelt(
    algo: rpt.Pelt,
    beta: float,
) -> tuple[int | None, float | None, list | None]:
    """Run PELT at a single penalty value and return the result.

    Parameters
    ----------
    algo:
        Fitted PELT instance (from ``build_pelt``).
    beta:
        Penalty value.

    Returns
    -------
    tuple
        ``(n_bkps, total_cost, bkps)`` on success.
        ``(None, None, None)`` on a ruptures failure; a ``RuntimeWarning``
        is issued and the caller is responsible for skipping the result.

    Notes
    -----
    ``sum_of_costs`` is an internal ruptures method. It is used here
    because no stable public alternative exists for retrieving the
    unpenalised segmentation cost after ``predict``. The result is used
    only for CROPS crossover-penalty computation and is not reported
    directly in any output.
    """
    try:
        bkps = algo.predict(pen=beta)
    except Exception as exc:  # noqa: BLE001
        # ruptures + scikit-learn (RBF cost) can raise a wider variety of
        # exceptions than BadSegmentationParameters / ValueError alone.
        # Observed cases include MemoryError on very long signals, generic
        # RuntimeError from the kernel cost machinery, and LinAlgError
        # from numerical degeneracies. A broad catch is justified here:
        # the CROPS sweep relies on continuing past localised PELT
        # failures, and the caller already guards on the (None, None, None)
        # sentinel returned below.
        warnings.warn(
            f"PELT failed at pen={beta:.6g}: {type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None, None
    cost = algo.cost.sum_of_costs(bkps)
    return len(bkps) - 1, float(cost), bkps


def resolve_beta_max(
    algo: rpt.Pelt,
    headroom: float = BETA_MAX_HEADROOM,
    retries: int = BETA_MAX_RETRIES,
) -> float:
    """Find a penalty large enough that PELT returns zero changepoints.

    The CROPS algorithm requires ``beta_max`` to produce the unsegmented
    (zero-changepoint) solution. Setting it to a fixed multiple of the
    unsegmented cost is not always sufficient for non-quadratic costs, so
    this function doubles the candidate until PELT confirms zero
    changepoints.

    Parameters
    ----------
    algo:
        Fitted PELT instance.
    headroom:
        Initial multiplier on the unsegmented-segment cost. The starting
        candidate is ``headroom * cost([n])``, where ``[n]`` is the
        single-segment breakpoint for a signal of length ``n``.
    retries:
        Maximum number of doublings before raising ``RuntimeError``.

    Returns
    -------
    float
        A penalty at which PELT has been confirmed to produce zero
        changepoints.

    Raises
    ------
    RuntimeError
        If no beta achieves the zero-changepoint solution within
        ``retries`` doublings. Surfacing failure here instead of returning
        a non-zero-changepoint penalty avoids confusing downstream errors
        from ``crops()`` complaining about a violated precondition.

    Notes
    -----
    The signal length is derived from ``algo.cost.signal`` so that no
    external ``n_grid`` argument is needed. This avoids the ambiguity of
    passing a length that may not match the fitted signal.
    """
    n = len(algo.cost.signal)
    unsegmented_cost = float(algo.cost.sum_of_costs([n]))

    # Guard against zero unsegmented cost. A perfectly flat signal has
    # cost zero under the l2 cost model, so ``headroom * 0.0`` would seed
    # beta at zero and the doubling loop ``beta *= 2.0`` would never make
    # progress. Use a unit fallback that scales with the doublings instead.
    if unsegmented_cost <= 0.0:
        warnings.warn(
            f"resolve_beta_max: unsegmented cost is {unsegmented_cost:.6g}. "
            f"Signal may be flat or degenerate. Using fallback beta=1.0 as "
            f"the starting point.",
            RuntimeWarning,
            stacklevel=2,
        )
        beta = 1.0
    else:
        beta = headroom * unsegmented_cost

    last_attempted: float = beta
    last_n_bkps: int | None = None
    for _ in range(retries):
        last_attempted = beta
        n_bkps, _, _ = _run_pelt(algo, beta)
        last_n_bkps = n_bkps
        if n_bkps == 0:
            return beta
        beta *= 2.0

    # Failure path. The previous implementation returned ``beta / 2.0``
    # here, which is the last value that *was* evaluated but did not
    # achieve zero changepoints. That confused downstream error reporting
    # in ``crops()``: the precondition check would fail with a misleading
    # numeric message instead of pointing the user at this function. By
    # raising explicitly we preserve the diagnostic chain.
    raise RuntimeError(
        f"resolve_beta_max: failed to reach zero changepoints after "
        f"{retries} doublings. Last beta={last_attempted:.6g} produced "
        f"{last_n_bkps} changepoints. Increase BETA_MAX_RETRIES or "
        f"BETA_MAX_HEADROOM in config, or verify that the cost model "
        f"is well-defined for this signal."
    )


# ── CROPS ─────────────────────────────────────────────────────────────────────

def _crossover_penalty(
    k_more: int,
    cost_more: float,
    k_fewer: int,
    cost_fewer: float,
) -> float | None:
    """Penalty at which two segmentations have equal penalised cost.

    For penalised cost ``C(k, beta) = cost(k) + beta * k``, two
    segmentations ``(k_more, cost_more)`` and ``(k_fewer, cost_fewer)``
    with ``k_more > k_fewer`` have equal penalised cost when::

        cost_more + beta * k_more = cost_fewer + beta * k_fewer
        beta = (cost_more - cost_fewer) / (k_fewer - k_more)

    Because ``k_more > k_fewer``, the denominator is negative and
    ``cost_more < cost_fewer`` (the finer segmentation has lower
    unpenalised cost), so ``beta`` is positive.

    Parameters
    ----------
    k_more:
        Number of changepoints for the higher-k (finer) segmentation.
    cost_more:
        Unpenalised cost of the higher-k segmentation.
    k_fewer:
        Number of changepoints for the lower-k (coarser) segmentation.
    cost_fewer:
        Unpenalised cost of the lower-k segmentation.

    Returns
    -------
    float or None
        Crossover penalty, or ``None`` if ``k_more == k_fewer``
        (undefined; avoids division by zero).
    """
    if k_more == k_fewer:
        return None
    return (cost_more - cost_fewer) / (k_fewer - k_more)


def crops(
    algo: rpt.Pelt,
    beta_min: float,
    beta_max: float,
) -> list[dict]:
    """CROPS algorithm (Haynes, Eckley & Fearnhead 2017, Algorithm 1).

    Finds all distinct optimal segmentations for penalties in
    ``[beta_min, beta_max]`` by recursively bisecting penalty intervals
    where the changepoint count jumps by more than one.

    Parameters
    ----------
    algo:
        Fitted PELT instance (from ``build_pelt``).
    beta_min:
        Lower penalty bound. Must produce the highest-k segmentation.
    beta_max:
        Upper penalty bound. Must produce the lowest-k segmentation.
        Must be strictly greater than ``beta_min``.

    Returns
    -------
    list of dict
        Sorted ascending by ``n_bkps``. Each dict has keys
        ``n_bkps``, ``cost``, ``penalty``, ``bkps``.
        Distinct segmentations with the same ``n_bkps`` but different
        breakpoint patterns are preserved as separate entries.

    Raises
    ------
    ValueError
        If ``beta_min >= beta_max``, or if the initial evaluations show
        that ``beta_min`` does not produce more changepoints than
        ``beta_max`` (violates the CROPS precondition).

    Notes
    -----
    Variable naming follows the mathematical convention of the paper:
    lower penalty (``b_lo``) produces more changepoints (higher k);
    higher penalty (``b_hi``) produces fewer changepoints (lower k).
    ``k_at_lo`` and ``k_at_hi`` always refer to the k values at those
    penalty bounds, never to "low k" or "high k" as magnitudes.
    """
    if beta_min >= beta_max:
        raise ValueError(
            f"beta_min ({beta_min:.6g}) must be strictly less than "
            f"beta_max ({beta_max:.6g})."
        )

    # Suppress known-harmless ruptures warnings within this function only.
    # Global suppression (warnings.filterwarnings at module level) hides
    # genuine numerical failures. We target only the specific category that
    # ruptures emits for valid but edge-case segmentations.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Numba.*",
            category=UserWarning,
            module="ruptures",
        )
        return _crops_inner(algo, beta_min, beta_max)


def _crops_inner(
    algo: rpt.Pelt,
    beta_min: float,
    beta_max: float,
) -> list[dict]:
    """Core CROPS recursion. Called by ``crops`` inside a warning filter."""
    # Keyed by (n_bkps, bkps_tuple) to preserve distinct segmentations
    # that share the same changepoint count but differ in breakpoint pattern.
    # CROPS guarantees each such pattern is genuinely distinct.
    evaluated: dict[float, tuple] = {
        beta_min: _run_pelt(algo, beta_min),
        beta_max: _run_pelt(algo, beta_max),
    }
    # Sorted list of evaluated penalties, kept in lockstep with the dict
    # above. The closeness test scans only the two neighbours of b_mid via
    # bisect, so each closeness check is O(log n_evaluated). Inserting a
    # new evaluated penalty into the sorted list is O(n) because Python
    # lists shift elements on insertion. For typical CROPS sweeps with
    # n_evaluated <= 50, the total insertion work is well below the cost
    # of a single PELT call and the data structure choice is not on the
    # hot path. If sweeps ever grow to thousands of penalties, replace
    # this with sortedcontainers.SortedList for O(log n) inserts.
    evaluated_sorted: list[float] = sorted(evaluated.keys())

    k_at_min, _, _ = evaluated[beta_min]
    k_at_max, _, _ = evaluated[beta_max]

    if k_at_min is None or k_at_max is None:
        warnings.warn(
            "PELT failed at one or both initial bounds. "
            "CROPS cannot proceed.",
            RuntimeWarning,
            stacklevel=3,
        )
        return []

    if k_at_min < k_at_max:
        raise ValueError(
            f"CROPS precondition violated: beta_min={beta_min:.6g} "
            f"produces k={k_at_min}, but beta_max={beta_max:.6g} "
            f"produces k={k_at_max}. "
            f"beta_min must produce at least as many changepoints as beta_max. "
            f"Decrease beta_min or increase beta_max."
        )

    intervals: list[tuple[float, float]] = [(beta_min, beta_max)]

    while intervals:
        b_lo, b_hi = intervals.pop()
        k_at_lo, c_at_lo, _ = evaluated[b_lo]
        k_at_hi, c_at_hi, _ = evaluated[b_hi]

        # Skip intervals where either bound failed.
        if k_at_lo is None or k_at_hi is None:
            continue

        # Monotonicity check. Lower penalty must produce at least as many
        # changepoints as higher penalty. A violation indicates numerical
        # instability or a cost model that does not satisfy PELT assumptions.
        if k_at_lo < k_at_hi:
            warnings.warn(
                f"Monotonicity violated at b_lo={b_lo:.6g} "
                f"(k={k_at_lo}) and b_hi={b_hi:.6g} (k={k_at_hi}). "
                f"Lower penalty should yield >= changepoints. "
                f"Skipping interval. Check cost model.",
                RuntimeWarning,
                stacklevel=3,
            )
            continue

        # Stopping criterion: k values differ by at most 1, so no
        # intermediate changepoint count is possible in this interval.
        if k_at_lo <= k_at_hi + 1:
            continue

        # Compute the crossover penalty: the beta at which the penalised
        # costs of the two bounding segmentations are equal. This is the
        # candidate midpoint for the next PELT evaluation.
        # b_lo has higher k (more changepoints, lower unpenalised cost).
        # b_hi has lower k (fewer changepoints, higher unpenalised cost).
        b_mid = _crossover_penalty(k_at_lo, c_at_lo, k_at_hi, c_at_hi)
        if b_mid is None:
            continue

        # Scale-aware guard: skip if b_mid is not strictly inside the
        # interval or is too close to an already-evaluated penalty.
        # Using machine epsilon here would be too tight for large penalties.
        if not (b_lo < b_mid < b_hi):
            continue
        # O(log n) closeness check: only the two evaluated penalties
        # bracketing b_mid can possibly be within _BETA_RTOL of it, given
        # that ``evaluated_sorted`` is monotonic ascending. Check both.
        insert_pos = bisect.bisect_left(evaluated_sorted, b_mid)
        too_close = False
        if insert_pos < len(evaluated_sorted) and _beta_close(
            b_mid, evaluated_sorted[insert_pos]
        ):
            too_close = True
        elif insert_pos > 0 and _beta_close(
            b_mid, evaluated_sorted[insert_pos - 1]
        ):
            too_close = True
        if too_close:
            continue

        evaluated[b_mid] = _run_pelt(algo, b_mid)
        bisect.insort(evaluated_sorted, b_mid)
        k_at_mid, _, _ = evaluated[b_mid]
        if k_at_mid is None:
            continue

        # CROPS recursion (Algorithm 1, Haynes et al. 2017).
        # Both sub-intervals are checked independently: both may require
        # further subdivision.
        if k_at_lo > k_at_mid + 1:
            intervals.append((b_lo, b_mid))
        if k_at_mid > k_at_hi + 1:
            intervals.append((b_mid, b_hi))

    # Collect results. Key by (n_bkps, bkps_tuple) to preserve distinct
    # segmentations with the same changepoint count but different patterns.
    # Within each unique pattern, keep the lowest penalty at which it was
    # first identified (consistent with CROPS paper convention).
    seen: dict[tuple, dict] = {}
    for beta, (k, cost, bkps) in evaluated.items():
        if k is None or bkps is None:
            continue
        key = (k, tuple(bkps))
        if key not in seen:
            seen[key] = {
                "n_bkps": k,
                "cost": cost,
                "penalty": beta,
                "bkps": bkps,
            }
        else:
            # Same segmentation pattern at a different penalty. Warn if
            # costs differ by more than floating-point noise: that would
            # indicate the cost function is not deterministic.
            if not np.isclose(cost, seen[key]["cost"], rtol=1e-6):
                warnings.warn(
                    f"Same segmentation (k={k}, bkps={bkps}) has "
                    f"inconsistent costs at different penalties: "
                    f"{seen[key]['cost']:.6g} vs {cost:.6g}. "
                    f"This may indicate numerical instability.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    return sorted(seen.values(), key=lambda x: x["n_bkps"])
