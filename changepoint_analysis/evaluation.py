"""Segmentation evaluation, zone assignment, snapping, and selection.

These functions bridge the CROPS/Kneedle detection layer and the
statistics layer. Breakpoints are converted from grid indices to depth
boundaries, snapped to measured segment edges, and used to partition
segments into zones. The pipeline then suggests a segmentation from the
evidence table.

Selection philosophy
--------------------
The pipeline identifies the maximal-resolution segmentation whose
boundaries are reproducible across cost models and tolerance settings.

Step 1 — Cross-model knee consensus.
    A k achieves consensus when it appears in the Kneedle knee set
    (union across the S sweep) of at least two cost models with mean
    detection frequency >= 0.5 across those models. Suggested
    immediately.

Step 2 — Maximal stable resolution (when Step 1 finds no consensus).
    Admissibility criterion: every breakpoint satisfies
    f(b) >= STABILITY_THRESHOLD. Among admissible candidates, select
    the highest k. Ties broken in this order:
        (a) non-plateau before plateau (plateau_flag == False first),
        (b) nested_in_next == True first,
        (c) n_models descending.
    If no stable candidate exists at any k, fall back to the
    highest-k row in the evidence table and label the result.

The plateau test (``MIN_DELTA_RSS_NORM``) is NOT a hard admissibility
gate. It appears as a descriptive column in the evidence table and as
a tiebreaker in Step 2 (a). Using it as a hard gate would force
low-k selections on cores where reproducible boundaries explain
modest but consistent additional variance, which conflicts with the
goal of maximum-resolution interpretable stratigraphy.

The omnibus p-value is descriptive and post-selection. It is reported
in the evidence table but does not gate selection.
"""
from __future__ import annotations

import warnings
from collections import Counter
from typing import TypedDict

import numpy as np
import pandas as pd

from changepoint_analysis.config import (
    BREAKPOINT_SNAP_TOL_CM,
    COST_MODELS,
    MIN_DELTA_RSS_NORM,
    MIN_SEGMENTS_PER_ZONE,
    NESTING_TOL_CM,
    N_SEGMENTS,
    S_VALUES,
    SEGMENT_FM,
    SEGMENT_TOPS,
    STABILITY_THRESHOLD,
    THICKNESSES,
    TOLERANCE_SWEEP,
)
from changepoint_analysis.kneedle import kneedle_knee
from changepoint_analysis.statistics import (
    weighted_pairwise_test,
    weighted_rank_permutation_anova,
)


# ── Type aliases ──────────────────────────────────────────────────────────────

UniqueBkpsKey = tuple[tuple[int, ...], tuple[int, ...]]
"""Composite key (seg_bkps, depth_bkps) for ``unique_bkps`` entries.

Two segmentations with identical ``seg_bkps`` but different
``depth_bkps`` are kept as separate entries because they originate from
different grid-level breakpoint patterns.
"""


class SegmentationEval(TypedDict):
    """Return type of ``evaluate_segmentation``."""

    H: float
    omni_p: float
    conover_df: pd.DataFrame | None
    zone_labels: np.ndarray
    min_zone_size: int


# ── Breakpoint clustering ─────────────────────────────────────────────────────

def _cluster_breakpoints(
    all_bkps: list[int],
    snap_tol_cm: int,
) -> dict[int, int]:
    """Cluster nearby breakpoint depths and return a representative mapping.

    Single-linkage chaining: a value joins the current cluster when its
    distance to the cluster's last (largest) element is at most
    ``snap_tol_cm``. The span of an entire cluster can therefore exceed
    ``snap_tol_cm`` when intermediate values bridge the gap. This
    matches the physical structure of transition zones where multiple
    cost models may walk through gradient boundaries.

    Each cluster is represented by its integer median.

    Parameters
    ----------
    all_bkps:
        List of breakpoint depths in cm. Not required to be sorted; the
        function sorts internally.
    snap_tol_cm:
        Linkage tolerance (cm).

    Returns
    -------
    dict
        Maps each raw breakpoint depth to its cluster representative.
    """
    clusters: list[list[int]] = []
    for b in sorted(all_bkps):
        if clusters and abs(b - clusters[-1][-1]) <= snap_tol_cm:
            clusters[-1].append(b)
        else:
            clusters.append([b])
    return {b: int(np.median(c)) for c in clusters for b in c}


def _nearest_stability(
    snapped_cm: tuple[int, ...],
    bp_freq: dict[int, float],
    bp_sigma: dict[int, float],
    tol_cm: int = BREAKPOINT_SNAP_TOL_CM,
) -> tuple[list[float], list[float]]:
    """Look up stability and sigma for each breakpoint via binary search.

    For breakpoints not found exactly in ``bp_freq``, finds the nearest
    representative within ``tol_cm``. Breakpoints with no representative
    within tolerance return NaN for both stability and sigma. NaN here
    means "stability is unknown", distinct from "stability is zero".
    Downstream gating must treat NaN as a fail condition.
    """
    known = np.array(sorted(bp_freq), dtype=int) if bp_freq else np.array([])
    stab_vals: list[float] = []
    sig_vals: list[float] = []
    for b in snapped_cm:
        if b in bp_freq:
            stab_vals.append(bp_freq[b])
            sig_vals.append(bp_sigma.get(b, 0.0))
            continue
        if len(known) == 0:
            stab_vals.append(float("nan"))
            sig_vals.append(float("nan"))
            continue
        pos = int(np.searchsorted(known, b))
        candidates = []
        if pos < len(known):
            candidates.append(int(known[pos]))
        if pos > 0:
            candidates.append(int(known[pos - 1]))
        nearest = min(candidates, key=lambda r: abs(r - b))
        if abs(nearest - b) <= tol_cm:
            stab_vals.append(bp_freq[nearest])
            sig_vals.append(bp_sigma.get(nearest, 0.0))
        else:
            stab_vals.append(float("nan"))
            sig_vals.append(float("nan"))
    return stab_vals, sig_vals


# ── Zone assignment ───────────────────────────────────────────────────────────

def assign_zones_to_segments(
    segment_bkps_indices: list[int] | tuple[int, ...],
    n_segments: int = N_SEGMENTS,
) -> np.ndarray:
    """Map each measured segment to an integer zone label.

    Parameters
    ----------
    segment_bkps_indices:
        Segment indices at which zone boundaries fall.
    n_segments:
        Total number of measured segments.

    Returns
    -------
    np.ndarray
        Integer zone labels, one per segment. Labels start at 0.
    """
    bkps = sorted(int(b) for b in segment_bkps_indices)
    zones = np.zeros(n_segments, dtype=int)
    current_zone = 0
    bkp_iter = iter(bkps)
    next_bkp = next(bkp_iter, None)
    for i in range(n_segments):
        while next_bkp is not None and i >= next_bkp:
            current_zone += 1
            next_bkp = next(bkp_iter, None)
        zones[i] = current_zone
    return zones


# ── Breakpoint conversion and snapping ────────────────────────────────────────

def pelt_bkps_to_depths(
    pelt_bkps: list[int],
    depth_grid: np.ndarray,
) -> tuple[int, ...]:
    """Convert PELT grid-index breakpoints to depths in cm.

    Drops the terminal index returned by ruptures.
    """
    return tuple(
        int(depth_grid[b]) for b in pelt_bkps[:-1] if b < len(depth_grid)
    )


def depths_to_segment_indices(
    depth_bkps: tuple[int, ...],
    segment_tops: np.ndarray = SEGMENT_TOPS,
) -> tuple[int, ...]:
    """Convert depth breakpoints to segment-index breakpoints.

    Forward-snaps each depth to the first segment top >= depth. Two
    distinct input depths may snap to the same segment top (a
    'collision'); the duplicate is dropped. Breakpoints deeper than the
    deepest candidate segment top cannot be snapped and are also
    dropped. Both kinds of drop emit a ``RuntimeWarning``.

    Returns
    -------
    tuple of int
        Sorted, deduplicated segment indices. The length may be less
        than ``len(depth_bkps)`` when drops occur.
    """
    candidate_tops = segment_tops[1:]
    if len(candidate_tops) == 0:
        return ()
    seg_indices: list[int] = []
    dropped_offend: list[int] = []
    dropped_collision: list[tuple[int, int]] = []
    for d in depth_bkps:
        pos = int(np.searchsorted(candidate_tops, d, side="left"))
        if pos >= len(candidate_tops):
            dropped_offend.append(int(d))
            continue
        nearest_idx = pos + 1
        if nearest_idx in seg_indices:
            target_cm = int(candidate_tops[pos])
            dropped_collision.append((int(d), target_cm))
            continue
        seg_indices.append(nearest_idx)
    if dropped_offend:
        warnings.warn(
            f"depths_to_segment_indices: dropped "
            f"{len(dropped_offend)} breakpoint(s) at depths "
            f"{dropped_offend} cm beyond the deepest segment top "
            f"({int(candidate_tops[-1])} cm). Resulting segmentation "
            f"has {len(seg_indices)} breakpoints instead of "
            f"{len(depth_bkps)}.",
            RuntimeWarning,
            stacklevel=2,
        )
    if dropped_collision:
        coll_str = ", ".join(f"{d}cm->{t}cm" for d, t in dropped_collision)
        warnings.warn(
            f"depths_to_segment_indices: dropped "
            f"{len(dropped_collision)} breakpoint(s) due to snap "
            f"collisions ({coll_str}). Two distinct input depths "
            f"snapped to the same segment top. Resulting segmentation "
            f"has {len(seg_indices)} breakpoints instead of "
            f"{len(depth_bkps)}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return tuple(sorted(seg_indices))


# ── Segmentation scoring ──────────────────────────────────────────────────────

def evaluate_segmentation(
    segment_bkps_indices: list[int] | tuple[int, ...],
    fm_values: np.ndarray = SEGMENT_FM,
    thicknesses: np.ndarray = THICKNESSES,
) -> SegmentationEval:
    """Run weighted rank ANOVA and pairwise post-hoc on a segmentation.

    The omnibus test is post-selection and conditional on the chosen
    segmentation. It quantifies zone separation given the boundaries,
    not the existence or location of those boundaries.
    """
    zones = assign_zones_to_segments(segment_bkps_indices, len(fm_values))
    unique_zones, counts = np.unique(zones, return_counts=True)
    min_zone_size = int(counts.min()) if counts.size else 0

    if len(unique_zones) < 2:
        return {
            "H": np.nan,
            "omni_p": np.nan,
            "conover_df": None,
            "zone_labels": zones,
            "min_zone_size": min_zone_size,
        }

    H, omni_p = weighted_rank_permutation_anova(
        fm_values, zones, thicknesses)
    conover_df = weighted_pairwise_test(fm_values, zones, thicknesses)

    return {
        "H": H,
        "omni_p": omni_p,
        "conover_df": conover_df,
        "zone_labels": zones,
        "min_zone_size": min_zone_size,
    }


def _compute_rss(
    seg_bkps: tuple[int, ...],
    fm: np.ndarray,
    thicknesses: np.ndarray,
    zones: np.ndarray | None = None,
) -> float:
    """Within-zone residual sum of squares with thickness-weighted means."""
    if zones is None:
        zones = assign_zones_to_segments(seg_bkps, len(fm))
    rss = 0.0
    for z in np.unique(zones):
        mask = zones == z
        thick_z = thicknesses[mask]
        fm_z = fm[mask]
        zm = float(np.sum(fm_z * thick_z) / np.sum(thick_z))
        rss += float(np.sum((fm_z - zm) ** 2))
    return rss


def _rss_baseline(fm: np.ndarray, thicknesses: np.ndarray) -> float:
    """Unsegmented thickness-weighted RSS (k=0 baseline)."""
    grand_mean = float(np.sum(fm * thicknesses) / np.sum(thicknesses))
    return float(np.sum((fm - grand_mean) ** 2))


def _min_adjacent_effect_size(
    seg_bkps: tuple[int, ...],
    fm: np.ndarray,
    thicknesses: np.ndarray,
    zones: np.ndarray | None = None,
) -> float:
    """Minimum standardised difference between adjacent zones.

    For each consecutive zone pair (i, i+1), computes
    ``|mean_i - mean_j| / pooled_sd`` where means and the pooled SD are
    thickness-weighted. Reports the minimum across adjacent pairs.

    This has the algebraic form of Cohen's d but uses thickness-weighted
    population SDs rather than classical sample SDs. Treat it as a
    descriptive standardised distance, not a classical effect size. The
    weighting is consistent with the rest of the pipeline, which treats
    thickness as the representativeness weight.

    Returns NaN when fewer than two zones are present.
    """
    if zones is None:
        zones = assign_zones_to_segments(seg_bkps, len(fm))
    unique_zones = np.unique(zones)
    if len(unique_zones) < 2:
        return np.nan

    zone_means: list[float] = []
    zone_vars: list[float] = []
    zone_ws: list[float] = []
    for z in unique_zones:
        mask = zones == z
        thick_z = thicknesses[mask]
        fm_z = fm[mask]
        w_total = float(thick_z.sum())
        if w_total <= 0:
            zone_means.append(np.nan)
            zone_vars.append(np.nan)
            zone_ws.append(0.0)
            continue
        w_norm = thick_z / w_total
        zm = float((w_norm * fm_z).sum())
        zv = float((w_norm * (fm_z - zm) ** 2).sum())
        zone_means.append(zm)
        zone_vars.append(zv)
        zone_ws.append(w_total)

    min_d = np.inf
    for i in range(len(unique_zones) - 1):
        mu_i, mu_j = zone_means[i], zone_means[i + 1]
        var_i, var_j = zone_vars[i], zone_vars[i + 1]
        w_i, w_j = zone_ws[i], zone_ws[i + 1]
        if np.isnan(mu_i) or np.isnan(mu_j):
            continue
        pooled_var = (w_i * var_i + w_j * var_j) / (w_i + w_j)
        pooled_sd = float(np.sqrt(pooled_var)) if pooled_var > 0 else 0.0
        d = abs(mu_i - mu_j) / pooled_sd if pooled_sd > 0 else np.inf
        min_d = min(min_d, d)

    return float(min_d) if np.isfinite(min_d) else np.nan


# ── Cross-model consensus scoring ─────────────────────────────────────────────

def compute_consensus_scores(
    crops_results: dict,
    max_k: int = N_SEGMENTS - 1,
) -> dict[int, dict]:
    """Score each k by cross-model detection frequency across the S sweep.

    A k achieves consensus when detected by at least two models with
    mean detection frequency >= 0.5 across S values for those models.

    Returns
    -------
    dict
        Maps k -> ``{n_models, mean_freq, models}``. Only consensus k
        values are included.
    """
    n_s = len(S_VALUES)
    model_k_counts: dict[str, dict[int, int]] = {}

    for model in COST_MODELS:
        records = crops_results[model]
        ks = np.array([r["n_bkps"] for r in records])
        costs = np.array([r["cost"] for r in records])
        # One Kneedle run per model produces knee detections at every S
        # in the sweep. The cap on max_k is enforced here.
        result = kneedle_knee(ks, costs, S_values=S_VALUES, max_k=max_k)
        counts: dict[int, int] = {}
        for knees_at_S in result.knee_ks_per_S.values():
            for k in knees_at_S:
                if k <= max_k:
                    counts[k] = counts.get(k, 0) + 1
        model_k_counts[model] = counts

    all_detected: set[int] = set()
    for c in model_k_counts.values():
        all_detected.update(c.keys())

    consensus: dict[int, dict] = {}
    for k in all_detected:
        supporting = [
            (model, cnt[k] / n_s)
            for model, cnt in model_k_counts.items()
            if cnt.get(k, 0) > 0
        ]
        if len(supporting) < 2:
            continue
        mean_freq = float(np.mean([f for _, f in supporting]))
        if mean_freq < 0.5:
            continue
        consensus[k] = {
            "n_models": len(supporting),
            "mean_freq": round(mean_freq, 3),
            "models": [m for m, _ in supporting],
        }
    return consensus


# ── Breakpoint stability ──────────────────────────────────────────────────────

def compute_breakpoint_stability(
    crops_results: dict,
    depth_grid: np.ndarray,
    tolerance_grid: tuple[float, ...] = TOLERANCE_SWEEP,
    max_k: int = N_SEGMENTS - 1,
    snap_tol_cm: int = BREAKPOINT_SNAP_TOL_CM,
) -> tuple[dict[int, float], dict[int, float]]:
    """Compute f(b) and sigma_b for every breakpoint across all candidates.

    f(b) is the fraction of unique (model, tolerance) combinations
    containing boundary b. This makes f(b) a property of reproducibility
    across analytical settings and prevents inflation by the number of k
    values that survive within any single combination.

    sigma_b is the standard deviation of snapped boundary positions
    across all combinations containing b.

    Returns
    -------
    tuple of (dict, dict)
        ``(freq_dict, sigma_dict)``. Both map representative depth (cm)
        to the metric value.
    """
    records_index = group_records_by_k(crops_results)

    # Run Kneedle once per model. The result is reused across the
    # tolerance grid via region_at_tolerance().
    model_results = {
        model: kneedle_knee(
            np.array([r["n_bkps"] for r in crops_results[model]]),
            np.array([r["cost"] for r in crops_results[model]]),
            S_values=S_VALUES,
        )
        for model in COST_MODELS
    }

    combo_sets: dict[tuple[str, float], frozenset] = {}
    for model in COST_MODELS:
        result = model_results[model]
        model_records = records_index.get(model, {})
        for tol in tolerance_grid:
            knee_region_ks = result.region_at_tolerance(
                tolerance=tol, max_k=max_k)
            combo_bkps: set[int] = set()
            for k in knee_region_ks:
                matching = model_records.get(k, [])
                if not matching:
                    continue
                # CROPS can return alternative breakpoint patterns at
                # the same k. Use the first (lowest-penalty) only.
                depth_bkps = pelt_bkps_to_depths(
                    matching[0]["bkps"], depth_grid)
                seg_bkps = depths_to_segment_indices(depth_bkps)
                snapped_cm = tuple(int(SEGMENT_TOPS[i]) for i in seg_bkps)
                combo_bkps.update(snapped_cm)
            combo_sets[(model, tol)] = frozenset(combo_bkps)

    n_combos = len(combo_sets)
    if n_combos == 0:
        return {}, {}

    all_bkps = sorted(set(b for s in combo_sets.values() for b in s))
    reps = _cluster_breakpoints(all_bkps, snap_tol_cm)

    freq: Counter = Counter()
    raw_by_rep: dict[int, list[int]] = {rep: [] for rep in set(reps.values())}
    for combo_bkps in combo_sets.values():
        for rep in {reps.get(b, b) for b in combo_bkps}:
            freq[rep] += 1
        for b in combo_bkps:
            rep = reps.get(b, b)
            raw_by_rep[rep].append(b)

    sigma: dict[int, float] = {}
    for rep, positions in raw_by_rep.items():
        sigma[rep] = (
            round(float(np.std(positions, ddof=1)), 2)
            if len(positions) >= 2 else 0.0
        )

    freq_dict = {rep: round(freq[rep] / n_combos, 3) for rep in sorted(freq)}
    sigma_dict = {rep: sigma.get(rep, 0.0) for rep in sorted(freq)}
    return freq_dict, sigma_dict


# ── Nesting consistency (tiebreaker only) ─────────────────────────────────────

def _is_nested(
    bkps_lo: tuple[int, ...],
    bkps_hi: tuple[int, ...],
    tol: int = NESTING_TOL_CM,
) -> bool:
    """Return True if every boundary in bkps_lo appears within tol cm in bkps_hi."""
    if not bkps_lo:
        return True
    if not bkps_hi:
        return False
    hi_arr = np.asarray(sorted(bkps_hi), dtype=int)
    lo_arr = np.asarray(bkps_lo, dtype=int)
    pos = np.searchsorted(hi_arr, lo_arr)
    right = np.clip(pos, 0, len(hi_arr) - 1)
    left = np.clip(pos - 1, 0, len(hi_arr) - 1)
    dist_right = np.abs(hi_arr[right] - lo_arr)
    dist_left = np.abs(hi_arr[left] - lo_arr)
    return bool(np.all(np.minimum(dist_left, dist_right) <= tol))


def compute_nesting(df: pd.DataFrame) -> dict[tuple, bool | None]:
    """Check whether each segmentation nests into the next-k candidates.

    Returns dict mapping breakpoint tuple to True (nested), False
    (reorganised), or None (no higher-k candidate exists).
    """
    result: dict[tuple, bool | None] = {}
    bkps_by_k: dict[int, list[tuple[int, ...]]] = {}
    for k, rows in df.groupby("k_snapped"):
        bkps_by_k[int(k)] = [
            tuple(b) for b in rows["breakpoints_snapped_cm"]
        ]
    k_vals = sorted(bkps_by_k)
    for i, k in enumerate(k_vals):
        bkps_at_k = bkps_by_k[k]
        if i + 1 >= len(k_vals):
            for bkps in bkps_at_k:
                result[bkps] = None
            continue
        next_bkps_list = bkps_by_k[k_vals[i + 1]]
        for bkps in bkps_at_k:
            result[bkps] = any(
                _is_nested(bkps, nb) for nb in next_bkps_list
            )
    return result


# ── Candidate collection ──────────────────────────────────────────────────────

def group_records_by_k(crops_results: dict) -> dict[str, dict[int, list]]:
    """Index CROPS records by cost model and breakpoint count.

    Pre-grouping allows O(1) lookup per (model, k) instead of an
    O(n_records) scan.
    """
    result: dict[str, dict[int, list]] = {}
    for model in COST_MODELS:
        by_k: dict[int, list] = {}
        for record in crops_results.get(model, []):
            by_k.setdefault(record["n_bkps"], []).append(record)
        result[model] = by_k
    return result


def collect_unique_bkps(
    crops_results: dict,
    knee_region: list[int],
    depth_grid: np.ndarray,
    records_by_k: dict[str, dict[int, list]] | None = None,
) -> dict[UniqueBkpsKey, list[str]]:
    """Collect unique snapped segmentations within a knee region.

    Returns
    -------
    dict
        Maps ``(seg_bkps, depth_bkps)`` to the list of source model
        names that produced this segmentation.
    """
    index = (
        records_by_k if records_by_k is not None
        else group_records_by_k(crops_results)
    )
    knee_set = set(knee_region)
    unique_bkps: dict[UniqueBkpsKey, list[str]] = {}
    for model in COST_MODELS:
        model_index = index.get(model, {})
        for k in knee_set:
            for record in model_index.get(k, []):
                depth_bkps = pelt_bkps_to_depths(record["bkps"], depth_grid)
                seg_bkps = depths_to_segment_indices(depth_bkps)
                key = (seg_bkps, depth_bkps)
                unique_bkps.setdefault(key, [])
                if model not in unique_bkps[key]:
                    unique_bkps[key].append(model)
    return unique_bkps


# ── Results table ─────────────────────────────────────────────────────────────

def build_results_table(
    unique_bkps: dict,
    crops_results: dict | None = None,
    depth_grid: np.ndarray | None = None,
    precomputed_stability: tuple[dict, dict] | None = None,
) -> pd.DataFrame:
    """Evaluate all knee-region segmentations and return an evidence table.

    Columns are documented in the module docstring of ``main``. The table
    is sorted by (k_snapped DESC, n_models DESC).
    """
    rss_0 = _rss_baseline(SEGMENT_FM, THICKNESSES)

    bp_freq: dict[int, float] = {}
    bp_sigma: dict[int, float] = {}
    if precomputed_stability is not None:
        bp_freq, bp_sigma = precomputed_stability
    elif crops_results is not None and depth_grid is not None:
        bp_freq, bp_sigma = compute_breakpoint_stability(
            crops_results, depth_grid)

    # Sort by descending k so highest-resolution candidates appear
    # first. Both passes below iterate this sorted list.
    sorted_items: list[tuple[UniqueBkpsKey, list[str]]] = sorted(
        unique_bkps.items(),
        key=lambda x: len(x[0][0]),
        reverse=True,
    )

    # Pass 1: best (lowest) RSS at each k, plus cached zones.
    rss_cache: dict[tuple[int, ...], float] = {}
    zones_cache: dict[tuple[int, ...], np.ndarray] = {}
    rss_by_k: dict[int, float] = {}
    for (seg_bkps, _), _ in sorted_items:
        if seg_bkps in rss_cache:
            continue
        zones = assign_zones_to_segments(seg_bkps, len(SEGMENT_FM))
        zones_cache[seg_bkps] = zones
        rss_k = _compute_rss(seg_bkps, SEGMENT_FM, THICKNESSES, zones=zones)
        rss_cache[seg_bkps] = rss_k
        k = len(seg_bkps)
        if k not in rss_by_k or rss_k < rss_by_k[k]:
            rss_by_k[k] = rss_k

    # Pass 2: build the rows.
    rows = []
    for (seg_bkps, depth_bkps), source_models in sorted_items:
        zones = zones_cache[seg_bkps]
        result = evaluate_segmentation(seg_bkps)
        rss_k = rss_cache[seg_bkps]
        k = len(seg_bkps)

        # Marginal delta normalised by RSS(0) and the k gap. Non-
        # consecutive k values (e.g. k=3 then k=5) are scaled by the
        # gap so the value is comparable to a unit step.
        prev_k = k - 1
        if prev_k == 0:
            rss_prev, k_gap = rss_0, k
        elif prev_k in rss_by_k:
            rss_prev, k_gap = rss_by_k[prev_k], 1
        else:
            lower_ks = [kk for kk in rss_by_k if kk < k]
            if lower_ks:
                nearest_lower = max(lower_ks)
                rss_prev, k_gap = rss_by_k[nearest_lower], k - nearest_lower
            else:
                rss_prev, k_gap = rss_0, k

        delta_rss = rss_prev - rss_k
        delta_rss_norm = (
            (delta_rss / k_gap) / rss_0 if rss_0 > 0 else np.nan
        )
        plateau_flag = (
            delta_rss_norm < MIN_DELTA_RSS_NORM
            if not np.isnan(delta_rss_norm) else True
        )

        snapped_cm = tuple(int(SEGMENT_TOPS[i]) for i in seg_bkps)

        if bp_freq and snapped_cm:
            stab_vals, sig_vals = _nearest_stability(
                snapped_cm, bp_freq, bp_sigma)
            stab_arr = np.asarray(stab_vals, dtype=float)
            sig_arr = np.asarray(sig_vals, dtype=float)
            if np.all(np.isnan(stab_arr)):
                min_stab = np.nan
                mean_sigma = np.nan
            else:
                min_stab = float(np.nanmin(stab_arr))
                mean_sigma = (
                    float(np.nanmean(sig_arr))
                    if not np.all(np.isnan(sig_arr)) else np.nan
                )
        elif snapped_cm:
            min_stab = np.nan
            mean_sigma = np.nan
        else:
            # k=0 segmentation: trivially stable.
            min_stab = 1.0
            mean_sigma = 0.0

        min_adj_d = _min_adjacent_effect_size(
            seg_bkps, SEGMENT_FM, THICKNESSES, zones=zones)

        rows.append({
            "models":                 ", ".join(source_models),
            "n_models":               len(source_models),
            "k_snapped":              k,
            "breakpoints_seg_idx":    list(seg_bkps),
            "breakpoints_grid_cm":    list(depth_bkps),
            "breakpoints_snapped_cm": list(snapped_cm),
            "min_zone_size":          result["min_zone_size"],
            "singleton_flag":         result["min_zone_size"] < MIN_SEGMENTS_PER_ZONE,
            "omni_p": (round(result["omni_p"], 4)
                       if not np.isnan(result["omni_p"]) else np.nan),
            "delta_rss_norm":  round(delta_rss_norm, 4),
            "plateau_flag":    plateau_flag,
            "min_stability":   (round(min_stab, 3)
                                if not np.isnan(min_stab) else np.nan),
            "stability_flag":  (min_stab >= STABILITY_THRESHOLD
                                if not np.isnan(min_stab) else None),
            "mean_sigma_b":    (round(mean_sigma, 2)
                                if not np.isnan(mean_sigma) else np.nan),
            "min_adjacent_d":  (round(min_adj_d, 3)
                                if not np.isnan(min_adj_d) else np.nan),
            "nested_in_next":  None,   # filled in below
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(
        ["k_snapped", "n_models"], ascending=[False, False],
    ).reset_index(drop=True)

    nesting = compute_nesting(df)
    df["nested_in_next"] = df["breakpoints_snapped_cm"].apply(
        lambda b: nesting.get(tuple(b))
    )
    return df


# ── Selection ─────────────────────────────────────────────────────────────────

def _admissible(
    df: pd.DataFrame,
    require_stable: bool = True,
) -> pd.DataFrame:
    """Return rows passing the requested admissibility criteria.

    Stability is the primary admissibility gate. A row is admissible
    when every breakpoint has ``f(b) >= STABILITY_THRESHOLD``, which is
    encoded in the precomputed ``stability_flag`` column.

    The plateau flag is NOT used as an admissibility gate. It is a
    descriptive marker reported in the evidence table and used as a
    tiebreaker by ``suggest_segmentation`` when multiple candidates
    tie at the chosen k.

    Parameters
    ----------
    df:
        Evidence table from ``build_results_table``.
    require_stable:
        When True, only return rows with ``stability_flag == True``.
        When False, return every row. Used by the progressive
        relaxation path when no stable candidate exists at any k.

    Returns
    -------
    pd.DataFrame
        A copy of the passing rows.
    """
    if not require_stable:
        return df.copy()
    # ``.eq(True)`` treats None (stability not computable) as non-passing
    # rather than coercing to False via truthiness.
    return df[df["stability_flag"].eq(True)].copy()


def _sort_candidates(pool: pd.DataFrame) -> pd.DataFrame:
    """Sort candidates by maximum-resolution principle.

    Primary key:    highest k_snapped first.
    Secondary key:  non-plateau first (plateau_flag == False before
                    plateau_flag == True). Among candidates at the same
                    k, prefer the one with substantive marginal RSS
                    gain. This is the plateau gate demoted to a
                    tiebreaker.
    Tertiary key:   nested-into-next first.
    Quaternary key: most source models first.

    The previous version used plateau as an admissibility gate, which
    forced low-k selections on cores where reproducible boundaries
    explain only modest additional variance. The current ordering
    treats reproducibility (stability) as primary evidence and uses
    explanatory gain (plateau) only to break ties at the chosen k.
    """
    nest_order = {True: 0, None: 1, False: 2}
    pool = pool.copy()
    # Plateau rank: False (non-plateau) sorts before True (plateau)
    # because of ascending=True on the boolean. None entries fall to
    # the high end via the explicit mapping below.
    plateau_order = {False: 0, None: 1, True: 2}
    pool["_plateau_rank"] = pool["plateau_flag"].map(
        lambda v: plateau_order.get(v, 1))
    pool["_nest_rank"] = pool["nested_in_next"].map(
        lambda v: nest_order.get(v, 1))
    return pool.sort_values(
        ["k_snapped", "_plateau_rank", "_nest_rank", "n_models"],
        ascending=[False, True, True, False],
    ).drop(columns=["_plateau_rank", "_nest_rank"])


def suggest_segmentation(
    df: pd.DataFrame,
    consensus_scores: dict[int, dict] | None = None,
) -> tuple[pd.Series, str]:
    """Suggest a segmentation using a two-step sequential rule.

    Selection philosophy
    --------------------
    The pipeline identifies the maximal-resolution segmentation whose
    boundaries are reproducible across cost models and tolerance
    settings.

    Step 1 — Cross-model knee consensus.
        A k achieves consensus when it appears in the Kneedle knee set
        of at least two cost models with mean detection frequency
        >= 0.5. The k=consensus candidate with the most source models
        is suggested immediately.

    Step 2 — Maximal stable resolution.
        Stability is the primary gate. Among candidates whose every
        breakpoint satisfies ``f(b) >= STABILITY_THRESHOLD`` (the
        ``stability_flag = True`` rows), select the highest k. The
        plateau test (``MIN_DELTA_RSS_NORM``) is used only to break
        ties at the chosen k, never to exclude candidates. If no
        stable candidate exists at any k, the function relaxes to
        the highest-k row available and labels the result accordingly.

    The omnibus p-value is descriptive and post-selection; it does
    not gate selection. Nesting consistency is a tiebreaker.

    Parameters
    ----------
    df:
        Evidence table from ``build_results_table``. Must be non-empty.
    consensus_scores:
        Output of ``compute_consensus_scores``. When None or empty,
        Step 1 is skipped.

    Returns
    -------
    tuple of (pd.Series, str)
        ``(best_row, selection_mode_label)``. The label encodes which
        step fired and, for Step 2, whether the stability gate was
        relaxed.

    Raises
    ------
    ValueError
        If ``df`` is empty.
    """
    if df.empty:
        raise ValueError("No segmentations in the knee region.")

    # Step 1: cross-model consensus.
    if consensus_scores:
        best_k = max(
            consensus_scores,
            key=lambda k: (consensus_scores[k]["mean_freq"], k),
        )
        info = consensus_scores[best_k]
        cands = df[df["k_snapped"] == best_k]
        if not cands.empty:
            best = _sort_candidates(cands).iloc[0]
            n_total = len(COST_MODELS)
            mode = (
                f"Step 1 — cross-model consensus "
                f"(k={best_k}, {info['n_models']}/{n_total} models, "
                f"mean S-freq={info['mean_freq']:.2f})"
            )
            return best, mode

    # Step 2: maximal stable resolution. Stability is the primary
    # gate. If no stable candidate exists at any k, relax to the
    # highest-k row available.
    stable = _admissible(df, require_stable=True)
    if not stable.empty:
        best = _sort_candidates(stable).iloc[0]
        mode = (
            f"Step 2 — maximal stable resolution "
            f"(stability \u2265 {STABILITY_THRESHOLD:.2f}; "
            f"plateau used as tiebreaker only)"
        )
        return best, mode

    # Stability could not be assessed or no candidate met the threshold.
    # Fall back to the highest-k row regardless.
    best = _sort_candidates(df).iloc[0]
    return best, (
        "Step 2 — fallback: no candidate met the stability gate; "
        "highest-k row returned"
    )
