"""Build the 1-cm interpolated F14C profile used by PELT and the downcore plot.

Design notes
------------
The interpolated grid encodes physical depth spacing as observation count.
A 25-cm segment contributes 25 ordered cells; a gap contributes interpolated
cells. PELT therefore sees the core's physical depth structure.

The replicated-grid representation is computational rather than generative:
adjacent cells within the same segment are not statistically independent
observations. They represent one measurement replicated across its spatial
extent to encode depth-proportional weighting.

This design choice has consequences that the researcher should keep in mind:

* **Thickness-driven weighting.** A thick segment contributes proportionally
  more cells to the PELT objective than a thin one. This is equivalent to
  deterministic weighting by segment thickness through replication. Thick
  zones dominate the RSS; thin abrupt events are harder to detect.
  This is intentional (thickness approximates sediment volume), but it
  changes the optimisation geometry substantially relative to an unweighted
  analysis.

* **Interpolation across gaps.** Gap cells are filled by linear or PCHIP
  interpolation, not by measurements. Interpolated cells are computational
  support points rather than empirical observations. Interpolation implicitly
  assumes depositional continuity through gaps; that assumption is often
  unjustified for sediment cores with hiatuses or erosion surfaces. PELT may
  place breakpoints inside gaps where no measurement exists. The ``valid``
  mask records which cells are measured.

* **Piecewise-constant model vs. smooth interpolation.** The underlying
  inference model assumes abrupt regime shifts. Linear interpolation is less
  structurally contradictory than PCHIP because it imposes no curvature, but
  it still requires continuity and latent intermediate values across gaps. It
  is not strictly consistent with abrupt shifts. PCHIP imposes monotone
  smooth gradients with no empirical support in gap regions. For PELT, linear
  interpolation is the more defensible default; PCHIP should be used only
  when visual output is the primary concern.

* **Nearest-neighbour fallback.** Boundary cells that remain NaN after the
  primary interpolator are filled by the value of the nearest valid cell.
  This is a pragmatic fix, not a physical model: it creates artificial
  persistence and flat plateaus with no empirical support. A warning is
  always issued when this fallback fires.

* **1-cm discretisation.** All geometry is quantised to integer centimetre
  cells. The grid spans ``floor(min(tops))`` to ``ceil(max(bottoms))``;
  measured cells are those where ``tops[i] <= d < bottoms[i]`` for integer
  depth d. Sub-centimetre segment boundaries are not individually snapped:
  the grid is constructed globally and cells are assigned by inequality test.
  Breakpoint precision is limited to 1 cm. This is a modelling choice, not
  a numerical error.

Interval convention
-------------------
Segments are left-closed, right-open. A cell at integer depth d belongs to
segment i when ``SEGMENT_TOPS[i] <= d < SEGMENT_BOTTOMS[i]``. When no gaps
exist between adjacent segments, this assigns every 1-cm cell to exactly one
segment. Cells in gap regions are not assigned to any segment.

Statistical separation
----------------------
The statistical tests never use this grid. They operate on the original
measured segment F14C values with thickness weights. Using replicated grid
cells as independent observations would inflate the rank-test sample size
and invalidate inference.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

from changepoint_analysis.config import (
    INTERPOLATION_METHOD,
    LARGE_GAP_THRESHOLD_CM,
    SEGMENT_BOTTOMS,
    SEGMENT_FM,
    SEGMENT_TOPS,
)

# Fraction of total core length at which the cumulative unsampled extent
# becomes large enough to trigger an additional warning. A small core
# punctuated by several short gaps can pass every absolute-threshold
# check while still being mostly interpolated. This relative gate flags
# that condition. See PIPELINE.md Step 1 for rationale.
_LARGE_GAP_FRACTION: float = 0.10

_VALID_METHODS: frozenset[str] = frozenset({"linear", "pchip"})

# Validate INTERPOLATION_METHOD at import time so a misconfigured value fails
# immediately rather than at the first call to build_grid.
if INTERPOLATION_METHOD.lower() not in _VALID_METHODS:
    raise ValueError(
        f"config.INTERPOLATION_METHOD={INTERPOLATION_METHOD!r} is not valid. "
        f"Choose one of {sorted(_VALID_METHODS)}."
    )


def _warn_large_gaps(
    depth_grid: np.ndarray,
    valid: np.ndarray,
    threshold_cm: int = LARGE_GAP_THRESHOLD_CM,
    fraction_threshold: float = _LARGE_GAP_FRACTION,
) -> None:
    """Warn when contiguous unsampled gaps exceed absolute or relative limits.

    Large gaps are filled by interpolation, creating synthetic continuity
    through unsampled material. PELT may place breakpoints inside these
    regions even though no measurements exist there. This is the principal
    methodological limitation of the interpolation approach.

    Two complementary checks are applied. The absolute check flags any
    single gap whose extent meets or exceeds ``threshold_cm``. The
    relative check flags cores where the cumulative unsampled extent is
    at least ``fraction_threshold`` of the total grid length, even when
    no individual gap exceeds the absolute threshold.

    Gap extents are reported as inclusive integer cm ranges.

    Parameters
    ----------
    depth_grid:
        Integer depth values in cm (1-D, step 1).
    valid:
        Boolean mask; True where a cell falls inside a measured segment.
    threshold_cm:
        Minimum gap length (in cm) that triggers the absolute warning.
    fraction_threshold:
        Cumulative unsampled fraction that triggers the relative warning.
    """
    # Vectorised gap-run detection. Pad the valid mask with True at both
    # ends so np.diff captures gaps that touch the grid boundary, then
    # find indices where the mask transitions True to False (gap starts)
    # and False to True (gap ends).
    padded = np.concatenate(([True], valid, [True]))
    transitions = np.diff(padded.astype(np.int8))
    # transitions == -1 at indices where True changes to False (gap start)
    # transitions == +1 at indices where False changes to True (gap end)
    gap_starts = np.where(transitions == -1)[0]
    gap_ends = np.where(transitions == +1)[0]

    if len(gap_starts) != len(gap_ends):
        raise RuntimeError(
            "_warn_large_gaps: gap start/end transition count mismatch. "
            "This indicates a corrupted valid mask."
        )

    large_gaps: list[tuple[int, int]] = []
    grid_len = len(depth_grid)
    for s_idx, e_idx in zip(gap_starts, gap_ends):
        # s_idx is the first unsampled index. e_idx is one past the last
        # unsampled index. Inclusive depth range is depth_grid[s_idx] to
        # depth_grid[e_idx - 1].
        if s_idx >= grid_len:
            continue
        last_idx = min(e_idx - 1, grid_len - 1)
        gap_len = last_idx - s_idx + 1
        if gap_len >= threshold_cm:
            large_gaps.append((int(depth_grid[s_idx]),
                               int(depth_grid[last_idx])))

    if large_gaps:
        gap_str = ", ".join(f"{a}-{b} cm" for a, b in large_gaps)
        warnings.warn(
            f"Large unsampled gap(s) detected ({gap_str}). "
            f"Interpolation creates synthetic continuity through "
            f"unsampled material. PELT may place breakpoints in these "
            f"regions. Consider this when interpreting results.",
            UserWarning,
            stacklevel=2,
        )

    # Relative gate. Small cores with many short gaps can pass the
    # absolute threshold yet still be mostly interpolated.
    n_unsampled = int((~valid).sum())
    if grid_len > 0 and n_unsampled / grid_len >= fraction_threshold:
        warnings.warn(
            f"Cumulative unsampled extent is "
            f"{n_unsampled}/{grid_len} cells "
            f"({n_unsampled / grid_len:.1%} of the grid). "
            f"PELT cost computation relies substantially on interpolated "
            f"values that have no empirical support.",
            UserWarning,
            stacklevel=2,
        )


def build_grid(
    method: str = INTERPOLATION_METHOD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 1-cm F14C grid with gap interpolation.

    Segment geometry has already been validated by ``config._validate_core``
    on import. This function does not re-validate; it trusts the frozen
    config arrays.

    Parameters
    ----------
    method:
        Interpolation method for gap cells. Either ``"linear"`` or
        ``"pchip"`` (case-insensitive). Validated immediately on entry.

    Returns
    -------
    depth_grid:
        Integer depths in cm (dtype int) from ``floor(min(SEGMENT_TOPS))``
        up to but not including ``ceil(max(SEGMENT_BOTTOMS))``. Step is
        always exactly 1 cm.
    fm_interp:
        F14C values at each depth. Measured cells hold the segment F14C;
        gap cells are filled by ``method``; any remaining boundary cells
        are filled by nearest-neighbour with a warning.
    valid:
        Boolean mask. True where a cell falls inside a measured segment
        under the left-closed, right-open interval convention.

    Raises
    ------
    TypeError
        If ``method`` is not a string.
    ValueError
        If ``method`` is not ``"linear"`` or ``"pchip"``; if the active
        core contains fewer than 2 valid cells after grid construction; or
        if the grid is too short for PELT (fewer than 2 cells total); or
        if valid depths are non-monotonic before interpolation (can arise
        from pathological sub-centimetre segment geometry).
    """
    if not isinstance(method, str):
        raise TypeError(
            f"method must be a string, got {type(method).__name__!r}."
        )
    method = method.lower()
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown interpolation method {method!r}. "
            f"Choose one of {sorted(_VALID_METHODS)}."
        )

    # floor/ceil ensure non-integer segment boundaries (e.g. top=0.5) do not
    # truncate the grid. depth_grid is integer so that index and depth
    # arithmetic are identical under the fixed 1-cm spacing.
    core_top = int(np.floor(SEGMENT_TOPS.min()))
    core_bottom = int(np.ceil(SEGMENT_BOTTOMS.max()))
    depth_grid = np.arange(core_top, core_bottom, dtype=int)

    if len(depth_grid) < 2:
        raise ValueError(
            f"Grid has only {len(depth_grid)} cell(s) "
            f"(core_top={core_top}, core_bottom={core_bottom}). "
            f"PELT requires at least 2 cells. "
            f"Check that the core spans at least 2 cm."
        )

    # Assign each measured cell its segment FM value using a single
    # vectorised searchsorted. For each grid depth d, the candidate
    # segment is the largest i with SEGMENT_TOPS[i] <= d. The cell
    # belongs to that segment iff d < SEGMENT_BOTTOMS[i]. Cells outside
    # any segment remain NaN and are filled by interpolation below.
    seg_idx = np.searchsorted(SEGMENT_TOPS, depth_grid, side="right") - 1
    in_range = (seg_idx >= 0) & (seg_idx < len(SEGMENT_TOPS))
    fm_grid = np.full(depth_grid.shape, np.nan, dtype=float)
    if in_range.any():
        candidate_idx = seg_idx[in_range]
        candidate_d = depth_grid[in_range]
        inside = candidate_d < SEGMENT_BOTTOMS[candidate_idx]
        target = np.where(in_range)[0][inside]
        fm_grid[target] = SEGMENT_FM[candidate_idx[inside]]

    # Detect segments that contributed zero cells (sub-centimetre or
    # collapsed onto a single cell with rounding). Vectorised comparison
    # against floor/ceil of segment bounds preserves the original warning
    # semantics without a Python loop.
    contributed = np.zeros(len(SEGMENT_FM), dtype=bool)
    assigned_idx = candidate_idx[inside] if in_range.any() else np.array([], dtype=int)
    if assigned_idx.size:
        contributed[np.unique(assigned_idx)] = True
    for i in np.where(~contributed)[0]:
        warnings.warn(
            f"Segment [{SEGMENT_TOPS[i]}, {SEGMENT_BOTTOMS[i]}) contributes "
            f"zero cells to the 1-cm grid. The segment is thinner than 1 cm "
            f"and will not appear in the PELT objective. Statistical "
            f"inference on the original segment values is unaffected.",
            UserWarning,
            stacklevel=2,
        )

    valid = ~np.isnan(fm_grid)

    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError(
            "No valid cells in the interpolated grid. "
            "Check that SEGMENT_TOPS and SEGMENT_BOTTOMS produce at least "
            "one 1-cm cell inside the measured segments."
        )
    if n_valid < 2:
        raise ValueError(
            f"Only {n_valid} valid cell in the interpolated grid. "
            f"At least 2 are required for interpolation and changepoint "
            f"analysis. Check sub-centimetre segment geometry."
        )

    fm_interp = fm_grid.copy()

    _warn_large_gaps(depth_grid, valid)

    if not valid.all():
        # depth_grid is integer with step 1, so float conversion is exact
        # and index distance equals depth distance in cm.
        gap_depths = depth_grid[~valid].astype(float)
        valid_depths = depth_grid[valid].astype(float)
        valid_fm = fm_grid[valid]

        # Guard against duplicate valid depths, which can arise when two
        # sub-centimetre segments both collapse onto the same integer cell.
        # PchipInterpolator requires strictly increasing x.
        if np.any(np.diff(valid_depths) <= 0):
            raise ValueError(
                "Interpolation grid contains duplicate or non-monotonic "
                "valid depths. This can arise when multiple sub-centimetre "
                "segments collapse onto the same 1-cm cell. "
                "Check sub-centimetre segment geometry."
            )

        if method == "linear":
            fm_interp[~valid] = interp1d(
                valid_depths,
                valid_fm,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )(gap_depths)
        else:  # pchip
            # SciPy's PchipInterpolator accepts as few as 2 points but the
            # monotone cubic shape is uniquely determined only when at
            # least 3 distinct points are available. Below 3 points,
            # fall back to explicit linear interpolation.
            if len(valid_depths) < 3:
                warnings.warn(
                    f"PCHIP interpolation requires at least 3 valid grid "
                    f"cells for a well-defined monotone cubic; "
                    f"only {len(valid_depths)} present. Falling back "
                    f"to linear interpolation for this core.",
                    UserWarning,
                    stacklevel=2,
                )
                fm_interp[~valid] = interp1d(
                    valid_depths,
                    valid_fm,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )(gap_depths)
            else:
                fm_interp[~valid] = PchipInterpolator(
                    valid_depths,
                    valid_fm,
                    extrapolate=False,
                )(gap_depths)

    # Nearest-neighbour fallback for cells that remain NaN after the primary
    # interpolator. These are typically boundary cells outside the convex hull
    # of valid depths, but may also arise from degenerate interpolation
    # geometry (e.g. only one valid cluster, PCHIP numerical edge cases).
    # This fallback creates artificial flat plateaus with no empirical support.
    # A warning is always issued when this fallback fires.
    nan_mask = np.isnan(fm_interp)
    if nan_mask.any():
        valid_idx = np.where(~np.isnan(fm_interp))[0]
        nan_idx = np.where(nan_mask)[0]

        # Vectorised nearest-neighbour using searchsorted (O(n log n) total).
        # Distances are computed from depth_grid values (cm), not indices.
        # depth_grid currently uses fixed 1-cm spacing so index distance
        # equals depth distance in cm, but using depth values directly keeps
        # the code correct if spacing ever changes.
        ins = np.searchsorted(valid_idx, nan_idx)
        ins = np.clip(ins, 0, len(valid_idx) - 1)
        left = np.clip(ins - 1, 0, len(valid_idx) - 1)

        left_dist_cm = np.abs(
            depth_grid[nan_idx] - depth_grid[valid_idx[left]]
        )
        right_dist_cm = np.abs(
            depth_grid[nan_idx] - depth_grid[valid_idx[ins]]
        )
        min_dist_cm = np.minimum(left_dist_cm, right_dist_cm)
        nearest_pos = np.where(
            left_dist_cm <= right_dist_cm,
            valid_idx[left],
            valid_idx[ins],
        )
        fm_interp[nan_mask] = fm_interp[nearest_pos]

        n_filled = int(nan_mask.sum())
        max_dist_cm = int(np.max(min_dist_cm))
        n_far = int(np.sum(min_dist_cm >= LARGE_GAP_THRESHOLD_CM))
        far_note = (
            f" {n_far} of these are >= {LARGE_GAP_THRESHOLD_CM} cm from "
            f"the nearest measurement and have essentially no empirical "
            f"support."
            if n_far > 0 else ""
        )
        warnings.warn(
            f"Nearest-neighbour fallback filled {n_filled} boundary "
            f"cell(s); nearest measured value was up to {max_dist_cm} cm "
            f"away.{far_note} These filled values should not be "
            f"interpreted as measurements.",
            UserWarning,
            stacklevel=2,
        )

    return depth_grid, fm_interp, valid
