"""Pipeline orchestration for changepoint detection.

Two entry points:

* ``run_pipeline()`` returns a ``PipelineResult`` with no side effects.
  Library callers and notebooks should use this.
* ``main()`` calls ``run_pipeline()``, prints progress, and writes CSV
  outputs. Suitable for scripted runs from the command line.

Both functions share a single computation path. The ``main()`` entry
point only adds printing and file writing on top.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np
import pandas as pd

from changepoint_analysis.config import (
    ACTIVE_CORE,
    COST_MODELS,
    ELBOW_TOLERANCE,
    INTERPOLATION_METHOD,
    KNEEDLE_SMOOTHING,
    MIN_DELTA_RSS_NORM,
    N_PERMUTATIONS,
    N_SEGMENTS,
    OUTPUT_PATHS,
    PIPELINE_VERSION,
    SEGMENT_BOTTOMS,
    SEGMENT_FM,
    SEGMENT_SD,
    SEGMENT_TOPS,
    STABILITY_THRESHOLD,
    STABILITY_THRESHOLD_MODERATE,
    S_VALUES,
    THICKNESSES,
)
from changepoint_analysis.crops import build_pelt, crops, resolve_beta_max
from changepoint_analysis.evaluation import (
    SegmentationEval,
    build_results_table,
    collect_unique_bkps,
    compute_breakpoint_stability,
    compute_consensus_scores,
    evaluate_segmentation,
    suggest_segmentation,
)
from changepoint_analysis.interpolation import build_grid
from changepoint_analysis.kneedle import KneedleResult, kneedle_knee
from changepoint_analysis.plotting import (
    plot_downcore,
    plot_knees,
    plot_tolerance_sweep,
)
from changepoint_analysis.sensitivity import (
    print_tolerance_sweep,
    run_tolerance_sweep,
)
from changepoint_analysis.statistics import zone_summary_table


class CropsRecord(TypedDict):
    """One CROPS sweep output record."""

    n_bkps: int
    cost: float
    penalty: float
    bkps: list[int]


CropsResults = dict[str, list[CropsRecord]]
"""Per-model CROPS sweep output.

Maps cost model name (e.g. ``"l1"``, ``"l2"``, ``"rbf"``) to a list of
records sorted ascending by ``n_bkps``.
"""


@dataclass
class PipelineResult:
    """Container for all artefacts produced by ``run_pipeline()``.

    Attributes
    ----------
    df:
        Evidence table of all knee-region segmentations.
    knee_info:
        Per-model KneedleResult from the primary tolerance.
    crops_results:
        Per-model list of CROPS records.
    best_eval:
        Output of evaluate_segmentation for the suggested segmentation.
    best_depth_bkps:
        Snapped breakpoint depths of the suggested segmentation (cm).
    selected_k:
        Number of breakpoints in the suggested segmentation.
    selection_mode:
        Human-readable label describing which selection rule fired.
    depth_grid, fm_interp, valid:
        Outputs of ``interpolation.build_grid``.
    zone_df:
        Zone summary table with weighted statistics.
    sweep_df:
        Tolerance sensitivity sweep results.
    bp_freq, bp_sigma:
        Per-boundary stability frequency and position dispersion.
    beta_ranges:
        Per-model CROPS penalty range ``(beta_min, beta_max)`` actually
        swept. Recorded for reproducibility so the printed and saved
        output can show what penalty interval CROPS explored for each
        cost model.
    """

    df: pd.DataFrame
    knee_info: dict[str, KneedleResult]
    crops_results: CropsResults
    best_eval: SegmentationEval
    best_depth_bkps: list[int]
    selected_k: int
    selection_mode: str
    depth_grid: np.ndarray
    fm_interp: np.ndarray
    valid: np.ndarray
    zone_df: pd.DataFrame
    sweep_df: pd.DataFrame
    bp_freq: dict[int, float] = field(default_factory=dict)
    bp_sigma: dict[int, float] = field(default_factory=dict)
    beta_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)


# ── Computation core (no side effects) ────────────────────────────────────────

def run_pipeline() -> PipelineResult:
    """Run the full changepoint detection pipeline.

    Performs grid construction, CROPS detection, knee detection,
    candidate evaluation, segmentation suggestion, zone statistics, and
    the tolerance sensitivity analysis. Does NOT print progress or
    write files. For console output and CSVs, use ``main()``.

    Returns
    -------
    PipelineResult
        Dataclass containing every artefact needed for downstream
        plotting or reporting.

    Raises
    ------
    RuntimeError
        If the knee region is empty or no candidate segmentation is
        found.
    """
    depth_grid, fm_interp, valid = build_grid(INTERPOLATION_METHOD)
    max_k = N_SEGMENTS - 1

    crops_results: CropsResults = {}
    knee_info: dict[str, KneedleResult] = {}
    beta_ranges: dict[str, tuple[float, float]] = {}
    for model in COST_MODELS:
        algo = build_pelt(model, fm_interp)
        beta_min = 0.5 * float(np.var(fm_interp))
        beta_max = resolve_beta_max(algo)
        beta_ranges[model] = (beta_min, beta_max)
        records = crops(algo, beta_min, beta_max)
        crops_results[model] = records
        ks = np.array([r["n_bkps"] for r in records])
        costs = np.array([r["cost"] for r in records])
        knee_info[model] = kneedle_knee(ks, costs, max_k=max_k)

    knee_region = sorted(set().union(
        *(set(knee_info[m].knee_region_ks) for m in COST_MODELS)
    ))
    if not knee_region:
        raise RuntimeError("Knee region is empty.")

    bp_freq, bp_sigma = compute_breakpoint_stability(
        crops_results, depth_grid, max_k=max_k)

    unique_bkps = collect_unique_bkps(
        crops_results, knee_region, depth_grid)
    df = build_results_table(
        unique_bkps,
        crops_results=crops_results,
        depth_grid=depth_grid,
        precomputed_stability=(bp_freq, bp_sigma),
    )
    if df.empty:
        raise RuntimeError("No segmentations in the knee region.")

    consensus_scores = compute_consensus_scores(crops_results, max_k=max_k)
    best, selection_mode = suggest_segmentation(df, consensus_scores)
    best_seg_bkps = list(best["breakpoints_seg_idx"])
    best_depth_bkps = best["breakpoints_snapped_cm"]
    selected_k = int(best["k_snapped"])

    best_eval = evaluate_segmentation(best_seg_bkps, SEGMENT_FM, THICKNESSES)

    # Zone summary issues warnings for small zones (n < 5) by design.
    # Suppress only inside the call site rather than at the module level
    # so genuine warnings in other code paths remain visible.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        zone_df = zone_summary_table(
            fm_values=SEGMENT_FM,
            sd_values=SEGMENT_SD,
            thicknesses=THICKNESSES,
            tops=SEGMENT_TOPS,
            bottoms=SEGMENT_BOTTOMS,
            zone_labels=best_eval["zone_labels"],
        )
    zone_df["singleton"] = zone_df["n_segments"] < 2

    sweep_df = run_tolerance_sweep(crops_results, depth_grid, max_k=max_k)

    return PipelineResult(
        df=df,
        knee_info=knee_info,
        crops_results=crops_results,
        best_eval=best_eval,
        best_depth_bkps=best_depth_bkps,
        selected_k=selected_k,
        selection_mode=selection_mode,
        depth_grid=depth_grid,
        fm_interp=fm_interp,
        valid=valid,
        zone_df=zone_df,
        sweep_df=sweep_df,
        bp_freq=bp_freq,
        bp_sigma=bp_sigma,
        beta_ranges=beta_ranges,
    )


# ── Side-effecting entry point ────────────────────────────────────────────────

def main() -> PipelineResult:
    """Run the pipeline with progress prints and CSV writing.

    Wraps ``run_pipeline()`` with console output and writes the CSVs
    listed in ``config.OUTPUT_PATHS``. The returned ``PipelineResult``
    is identical to what ``run_pipeline()`` produces.
    """
    _print_header()
    result = run_pipeline()
    _print_summary(result)
    _write_csv_outputs(result)
    return result


# ── Output: CSV writing ───────────────────────────────────────────────────────

def _write_csv_outputs(result: PipelineResult) -> None:
    """Write CSV outputs to the paths in ``config.OUTPUT_PATHS``.

    Every output CSV is stamped with ``pipeline_version`` and
    ``active_core`` so that downstream consumers can identify the run
    that produced the file even after it has moved from the working
    directory.
    """
    stamp = {
        "pipeline_version": PIPELINE_VERSION,
        "active_core": ACTIVE_CORE,
    }

    zone_df = result.zone_df.copy()
    for key, value in stamp.items():
        zone_df[key] = value
    zone_df.to_csv(OUTPUT_PATHS["zone_summary_csv"], index=False)

    display = _display_df(result.df)
    for key, value in stamp.items():
        display[key] = value
    display.to_csv(OUTPUT_PATHS["evidence_table_csv"], index=False)

    sweep_df = result.sweep_df.copy()
    for key, value in stamp.items():
        sweep_df[key] = value
    sweep_df.to_csv(OUTPUT_PATHS["tolerance_sweep_csv"], index=False)


# ── Output: console printing ──────────────────────────────────────────────────

def _print_header() -> None:
    """Print the pipeline configuration header."""
    print("=" * 78)
    print("CHANGEPOINT DETECTION VIA CROPS + KNEEDLE")
    print("=" * 78)
    print(f"Pipeline version      {PIPELINE_VERSION}")
    print(f"Core                  {ACTIVE_CORE}")
    print(f"Segments              {N_SEGMENTS}")
    print(f"Total core length     {int(SEGMENT_BOTTOMS.max())} cm")
    print(f"Interpolation         {INTERPOLATION_METHOD}")
    print(f"Kneedle smoothing     {KNEEDLE_SMOOTHING}")
    print(f"Sensitivity sweep     {tuple(S_VALUES)}")
    print(f"Knee tolerance        {ELBOW_TOLERANCE}")
    print(f"Cost models           {tuple(COST_MODELS)}")
    print(f"Permutations          {N_PERMUTATIONS}")
    print(f"Stability gate        f(b) \u2265 {STABILITY_THRESHOLD:.2f}")
    print(
        f"Plateau threshold     "
        f"{MIN_DELTA_RSS_NORM:.0%} of RSS\u2080 "
        f"(descriptive / tiebreaker)"
    )
    print(f"Max k (segments - 1)  {N_SEGMENTS - 1}")
    print()


def _print_summary(result: PipelineResult) -> None:
    """Print results from a completed ``run_pipeline`` run."""
    n_gap = int((~result.valid).sum())
    print(
        f"1-cm grid             {len(result.depth_grid)} points "
        f"(measured {int(result.valid.sum())}, gap {n_gap})"
    )
    print()

    for model in COST_MODELS:
        info = result.knee_info[model]
        ks = info.ks
        print(f"  CROPS [{model}]")
        if model in result.beta_ranges:
            b_lo, b_hi = result.beta_ranges[model]
            print(f"    beta range       {b_lo:.4g} to {b_hi:.4g}")
        print(f"    k range          {int(ks.min())} to {int(ks.max())}")
        print(f"    knee (S sweep)   {info.knee_ks}")
        print(
            f"    knee region      {info.knee_region_ks} "
            f"(tolerance {info.tolerance})"
        )
    print()

    knee_region = sorted(set().union(
        *(set(result.knee_info[m].knee_region_ks) for m in COST_MODELS)
    ))
    print(
        f"Combined knee region (union over cost models): k = {knee_region}"
    )
    print()

    _print_results_table(result.df)
    _print_breakpoint_stability(result.bp_freq, result.bp_sigma)
    _print_suggested(result)

    print()
    print("=" * 78)
    print("ZONE SUMMARY (thickness-weighted, 95% bootstrap CI)")
    print("=" * 78)
    print(result.zone_df.to_string(index=False))
    print()

    print_tolerance_sweep(result.sweep_df)


def _format_bkps_cell(value) -> str:
    """Format a breakpoints list cell for display.

    Returns ``"none"`` for None, NaN, or empty sequences; otherwise a
    comma-separated string. ``pd.isna`` raises on lists in some pandas
    versions, so the check is wrapped in try/except.
    """
    if value is None:
        return "none"
    try:
        if pd.isna(value):
            return "none"
    except (TypeError, ValueError):
        pass
    try:
        items = list(value)
    except TypeError:
        return "none"
    return ", ".join(str(v) for v in items) if items else "none"


def _display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a display-friendly copy of the evidence table for CSV export.

    Formats list columns as comma-separated strings and renames
    ``omni_p`` to make its post-selection status explicit. The
    ``breakpoints_seg_idx`` column is dropped because it is an internal
    integer-index list with no user-facing meaning.
    """
    out = df.drop(columns=[
        c for c in ["breakpoints_seg_idx"] if c in df.columns
    ]).copy()
    for col in ("breakpoints_grid_cm", "breakpoints_snapped_cm"):
        if col in out.columns:
            out[col] = out[col].apply(_format_bkps_cell)
    for col in ("singleton_flag", "plateau_flag", "stability_flag"):
        if col in out.columns:
            out[col] = out[col].map({True: "yes", False: "no", None: "n/a"})
    if "omni_p" in out.columns:
        out = out.rename(
            columns={"omni_p": "omni_p_post_selection_descriptive"})
    return out


def _print_results_table(df: pd.DataFrame) -> None:
    """Print the knee-region evidence table to stdout."""
    print("=" * 78)
    print("KNEE-REGION SEGMENTATIONS (evidence table)")
    print("=" * 78)
    print(
        "Note: omni_p is post-selection and descriptive. It quantifies "
        "zone separation given the suggested boundaries, not the "
        "existence or location of those boundaries."
    )
    disp = _display_df(df)
    cols = [
        "models", "n_models", "k_snapped",
        "breakpoints_snapped_cm",
        "min_zone_size", "singleton_flag",
        "omni_p_post_selection_descriptive",
        "delta_rss_norm", "plateau_flag",
        "min_stability", "stability_flag",
    ]
    print(disp[[c for c in cols if c in disp.columns]].to_string(index=False))
    print()


def _print_breakpoint_stability(
    bp_freq: dict[int, float],
    bp_sigma: dict[int, float],
) -> None:
    """Print the breakpoint stability frequency table to stdout.

    A boundary is tiered as HIGH, MODERATE, or LOW based on its f(b)
    value. Only HIGH boundaries pass the selection gate. MODERATE
    boundaries are shown for context but do not gate any decision.
    """
    if not bp_freq:
        return
    print("=" * 78)
    print("BREAKPOINT STABILITY FREQUENCIES")
    print(
        "f(b)    = fraction of unique (model, tolerance) "
        "combinations containing boundary b"
    )
    print("sigma_b = SD of boundary position across combinations (cm)")
    print(
        f"Stability gate: "
        f"HIGH = f(b) >= {STABILITY_THRESHOLD:.2f}  |  "
        f"MODERATE = f(b) >= {STABILITY_THRESHOLD_MODERATE:.2f}  |  "
        f"LOW = f(b) < {STABILITY_THRESHOLD_MODERATE:.2f}"
    )
    print("=" * 78)
    print(f"  {'bkp (cm)':>9}  {'f(b)':>6}  {'sigma_b':>8}  stability")
    print(f"  {'-' * 9}  {'-' * 6}  {'-' * 8}  {'-' * 10}")
    for b in sorted(bp_freq):
        f = bp_freq[b]
        s = bp_sigma.get(b, 0.0)
        if f >= STABILITY_THRESHOLD:
            stab = "HIGH"
        elif f >= STABILITY_THRESHOLD_MODERATE:
            stab = "MODERATE"
        else:
            stab = "LOW"
        print(f"  {b:>9}  {f:>6.3f}  {s:>8.2f}  {stab}")
    print()


def _print_suggested(result: PipelineResult) -> None:
    """Print the suggested segmentation summary to stdout.

    Locates the chosen evidence-table row by matching BOTH the selected
    k and the selected breakpoint depths. Matching on k alone fails when
    the knee region contains multiple candidate segmentations at the
    same k (e.g. one from L1 and one from L2 with different snapped
    boundaries). In that case ``iloc[0]`` would pick whichever row
    happened to sort to position 0 in the evidence table, not the one
    ``suggest_segmentation`` actually returned. The printed summary
    would then show fields from the wrong row.
    """
    same_k = result.df[result.df["k_snapped"] == result.selected_k]
    target = list(result.best_depth_bkps)
    matches = same_k[same_k["breakpoints_snapped_cm"].apply(
        lambda b: list(b) == target)]
    if matches.empty:
        # Defensive fallback: should never trigger because suggest_segmentation
        # returns a row that exists in df, but if the data structures ever
        # drift, fall back to the first matching k row rather than crash.
        best_row = same_k.iloc[0]
    else:
        best_row = matches.iloc[0]

    print("=" * 78)
    print("SUGGESTED SEGMENTATION")
    print("=" * 78)
    print(f"  Selection:           {result.selection_mode}")
    print(f"  k (snapped):         {result.selected_k}")
    print(f"  Breakpoints (cm):    {result.best_depth_bkps}")
    print(f"  Source models:       {best_row['models']}")
    print(
        f"  omni_p (post-selection, descriptive): "
        f"{best_row['omni_p']}"
    )

    ms = best_row.get("min_stability", float("nan"))
    ms_str = f"{ms:.3f}" if not pd.isna(ms) else "n/a"
    sg = best_row.get("mean_sigma_b", float("nan"))
    sg_str = f"{sg:.2f} cm" if not pd.isna(sg) else "n/a"
    md = best_row.get("min_adjacent_d", float("nan"))
    md_str = f"{md:.3f}" if not pd.isna(md) else "n/a"

    delta = best_row["delta_rss_norm"]
    print(
        f"  delta_rss_norm:      {delta:.4f}  ({delta:.1%} of RSS\u2080)"
    )
    print(f"  Min breakpt stab.:   {ms_str}")
    print(f"  Mean sigma_b:        {sg_str}")
    print(f"  Min adjacent d:      {md_str}")
    print(
        f"  Singleton zone:      "
        f"{'yes' if best_row['singleton_flag'] else 'no'}"
    )
    print(f"  Min zone size:       {best_row['min_zone_size']}")
    print()


# ── Command-line entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    result = main()
    plot_knees(result.knee_info, OUTPUT_PATHS["crops_knee_png"])
    plot_downcore(
        result.best_depth_bkps,
        result.selected_k,
        OUTPUT_PATHS["downcore_png"],
        depth_grid=result.depth_grid,
        fm_interp=result.fm_interp,
        valid=result.valid,
    )
    plot_tolerance_sweep(result.sweep_df, OUTPUT_PATHS["tolerance_sweep_png"])
    print("Output files written to paths in config.OUTPUT_PATHS:")
    for label, path in OUTPUT_PATHS.items():
        print(f"  {label:>28}  {path}")
