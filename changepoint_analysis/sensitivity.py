"""Knee tolerance sensitivity analysis.

Reruns Kneedle knee detection and segmentation selection across a grid
of ELBOW_TOLERANCE values using the CROPS results already computed by
the main pipeline. No additional PELT runs are needed.

The analysis answers: does the suggested k change when the tolerance
band is widened or narrowed? Stability across the sweep supports the
chosen tolerance as a robust operating point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from changepoint_analysis.config import (
    ACTIVE_CORE,
    COST_MODELS,
    ELBOW_TOLERANCE,
    MIN_DELTA_RSS_NORM,
    N_SEGMENTS,
    S_VALUES,
    TOLERANCE_SWEEP,
)
from changepoint_analysis.evaluation import (
    build_results_table,
    collect_unique_bkps,
    compute_breakpoint_stability,
    compute_consensus_scores,
    group_records_by_k,
    suggest_segmentation,
)
from changepoint_analysis.kneedle import kneedle_knee


def run_tolerance_sweep(
    crops_results: dict,
    depth_grid: np.ndarray,
    tolerance_grid: tuple[float, ...] = TOLERANCE_SWEEP,
    max_k: int = N_SEGMENTS - 1,
) -> pd.DataFrame:
    """Run segmentation selection across a grid of knee tolerances.

    Returns
    -------
    pd.DataFrame
        One row per tolerance value with columns: ``tolerance``,
        ``suggested_k``, ``breakpoints_snapped_cm``, ``selection_step``,
        ``omni_p``, ``delta_rss_norm``, ``plateau_flag``,
        ``singleton_flag``, ``knee_region``.
    """
    # Consensus scores are tolerance-independent.
    consensus_scores = compute_consensus_scores(crops_results, max_k=max_k)

    # Stability defined over the same tolerance grid as the sweep.
    bp_freq, bp_sigma = compute_breakpoint_stability(
        crops_results, depth_grid,
        tolerance_grid=tolerance_grid,
        max_k=max_k,
    )

    records_by_k = group_records_by_k(crops_results)

    # Single Kneedle run per model; tolerance variation handled by the
    # cached ``region_at_tolerance`` method on each result.
    model_results = {
        model: kneedle_knee(
            np.array([r["n_bkps"] for r in crops_results[model]]),
            np.array([r["cost"] for r in crops_results[model]]),
            S_values=S_VALUES,
        )
        for model in COST_MODELS
    }

    rows = []
    for tol in tolerance_grid:
        knee_region = sorted(set().union(*(
            set(model_results[m].region_at_tolerance(
                tolerance=tol, max_k=max_k))
            for m in COST_MODELS
        )))

        if not knee_region:
            rows.append({
                "tolerance":              tol,
                "suggested_k":            None,
                "breakpoints_snapped_cm": None,
                "selection_step":         "empty knee region",
                "omni_p":                 None,
                "delta_rss_norm":         None,
                "plateau_flag":           None,
                "singleton_flag":         None,
                "knee_region":            [],
            })
            continue

        unique_bkps = collect_unique_bkps(
            crops_results, knee_region, depth_grid,
            records_by_k=records_by_k,
        )
        df = build_results_table(
            unique_bkps,
            crops_results=crops_results,
            depth_grid=depth_grid,
            precomputed_stability=(bp_freq, bp_sigma),
        )

        if df.empty:
            rows.append({
                "tolerance":              tol,
                "suggested_k":            None,
                "breakpoints_snapped_cm": None,
                "selection_step":         "no candidates",
                "omni_p":                 None,
                "delta_rss_norm":         None,
                "plateau_flag":           None,
                "singleton_flag":         None,
                "knee_region":            knee_region,
            })
            continue

        best, mode = suggest_segmentation(df, consensus_scores)
        if mode.startswith("Step 1"):
            step = "Step 1"
        elif mode.startswith("Step 2"):
            step = "Step 2"
        else:
            step = "Fallback"

        rows.append({
            "tolerance":              tol,
            "suggested_k":            int(best["k_snapped"]),
            "breakpoints_snapped_cm": list(best["breakpoints_snapped_cm"]),
            "selection_step":         step,
            "omni_p":                 best["omni_p"],
            "delta_rss_norm":         best["delta_rss_norm"],
            "plateau_flag":           best["plateau_flag"],
            "singleton_flag":         best["singleton_flag"],
            "knee_region":            knee_region,
        })

    return pd.DataFrame(rows)


def _fmt(value, spec: str, na_label: str = "n/a") -> str:
    """Format a numeric value, returning ``na_label`` for None/NaN.

    ``pd.isna`` raises on list-like input. The try/except catches that
    and treats anything untestable as missing.
    """
    if value is None:
        return na_label
    try:
        if pd.isna(value):
            return na_label
    except (TypeError, ValueError):
        return na_label
    return format(value, spec)


def _fmt_flag(value, na_label: str = "n/a") -> str:
    """Format a boolean cell value, returning ``na_label`` for None/NaN."""
    if value is None:
        return na_label
    try:
        if pd.isna(value):
            return na_label
    except (TypeError, ValueError):
        return na_label
    return "yes" if value else "no"


def print_tolerance_sweep(sweep_df: pd.DataFrame) -> None:
    """Print tolerance sweep results to stdout."""
    print("=" * 78)
    print("KNEE TOLERANCE SENSITIVITY ANALYSIS")
    print(
        f"Core: {ACTIVE_CORE}   "
        f"Nominal tolerance: {ELBOW_TOLERANCE}   "
        f"RSS plateau threshold: {MIN_DELTA_RSS_NORM:.0%}"
    )
    print("=" * 78)

    header = (
        f"{'Tol':>5}  {'k':>3}  {'Step':>6}  "
        f"{'omni_p':>7}  {chr(948) + '(k)':>7}  "
        f"{'plateau':>7}  {'singleton':>9}  Knee region"
    )
    print(header)
    print("-" * 78)

    for _, row in sweep_df.iterrows():
        tol = f"{row['tolerance']:.2f}"
        k_val = row["suggested_k"]
        k = (str(int(k_val))
             if k_val is not None and not pd.isna(k_val) else "n/a")
        nom = (" <-- nominal"
               if abs(row["tolerance"] - ELBOW_TOLERANCE) < 1e-9 else "")
        print(
            f"{tol:>5}  {k:>3}  {str(row['selection_step']):>6}  "
            f"{_fmt(row['omni_p'], '.4f'):>7}  "
            f"{_fmt(row['delta_rss_norm'], '.4f'):>7}  "
            f"{_fmt_flag(row['plateau_flag']):>7}  "
            f"{_fmt_flag(row['singleton_flag']):>9}  "
            f"{row['knee_region']}{nom}"
        )

    print()
    _print_stability_summary(sweep_df)


def _print_stability_summary(sweep_df: pd.DataFrame) -> None:
    """Print a one-line stability verdict for the tolerance sweep."""
    valid = sweep_df[sweep_df["suggested_k"].notna()]
    if valid.empty:
        print("Stability: no valid suggestion at any tolerance.")
        return
    unique_ks = valid["suggested_k"].unique()
    if len(unique_ks) == 1:
        print(
            f"Stability: suggested k = {int(unique_ks[0])} is stable "
            f"across all {len(valid)} tolerance values tested."
        )
    else:
        modal_k = int(valid["suggested_k"].mode().iloc[0])
        stable_range = valid[valid["suggested_k"] == modal_k]["tolerance"]
        print(
            f"Stability: suggested k varies. "
            f"Modal k = {modal_k} (tolerance {stable_range.min():.2f} "
            f"to {stable_range.max():.2f}). "
            f"All unique k values: "
            f"{sorted(int(k) for k in unique_ks)}."
        )
