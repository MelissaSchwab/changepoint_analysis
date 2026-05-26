"""Knee detection and downcore profile plots."""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

from changepoint_analysis.config import (
    ACTIVE_CORE,
    COST_MODELS,
    ELBOW_TOLERANCE,
    INTERPOLATION_METHOD,
    N_SEGMENTS,
    S_VALUES,
    SEGMENT_BOTTOMS,
    SEGMENT_FM,
    SEGMENT_TOPS,
)
from changepoint_analysis.interpolation import build_grid
from changepoint_analysis.kneedle import KneedleResult


# ── Matplotlib style ──────────────────────────────────────────────────────────

# rcParams used by every plot in this module. Applied via a context
# manager so callers' global style is not silently overwritten on import.
# Arial is standard for many journals. On Linux HPC systems without Arial
# installed, matplotlib substitutes a fallback sans-serif silently.
_RC_PARAMS: dict = {
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Arial", "DejaVu Sans", "Helvetica"],
    # Illustrator-compatible PDF/PS output.
    "pdf.fonttype":          42,
    "ps.fonttype":           42,
    "font.size":             8,
    "axes.linewidth":        0.5,
    "axes.labelpad":         5,
    "axes.grid":             False,
    "xtick.direction":       "in",
    "ytick.direction":       "in",
    "xtick.minor.visible":   True,
    "ytick.minor.visible":   True,
    "xtick.major.width":     0.5,
    "ytick.major.width":     0.5,
    "xtick.minor.width":     0.5,
    "ytick.minor.width":     0.5,
}


@contextmanager
def _plot_style() -> Iterator[None]:
    """Apply the module's matplotlib style for the duration of a block.

    Using ``plt.rc_context`` keeps style changes scoped to each plot
    function rather than mutating the global ``plt.rcParams`` at import
    time. Library imports should not have visible side-effects on the
    caller's plotting environment.
    """
    with plt.rc_context(_RC_PARAMS):
        yield


# Colour scheme. Points and drop lines inside the knee region use a
# single accent colour; everything outside is black.
_COLOUR_KNEE_REGION: str = "#4c9be8"
_COLOUR_OUTSIDE: str = "black"
_COLOUR_CURVE: str = "black"

# Maximum number of integer y-axis ticks before switching to automatic
# tick spacing in the tolerance sweep plot.
_MAX_YTICKS: int = 20

# Default axis limits for the downcore plot. Override per-call when a
# core extends beyond these ranges.
_DOWNCORE_FM_LIMITS: tuple[float, float] = (0.0, 1.4)
_DOWNCORE_DEPTH_LIMITS_CM: tuple[float, float] = (200.0, 0.0)


def _ensure_dir(savepath: str) -> None:
    """Create parent directories of ``savepath`` if they do not exist."""
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)


def _perpendicular_foot(kn: float, cn: float) -> tuple[float, float]:
    """Return the foot of the perpendicular from (kn, cn) to the diagonal.

    After the Kneedle inversion the normalised curve runs from (0, 0) to
    (1, 1), so the reference line is the main diagonal ``cn = kn``.
    The perpendicular projection of point (kn, cn) onto that line is::

        foot = ((kn + cn) / 2,  (kn + cn) / 2)

    The Kneedle difference value ``diff = cost_norm - k_norm`` is exactly
    the signed vertical distance from the point to this diagonal, so the
    drop lines correctly visualise the quantity being maximised.
    """
    mid = (kn + cn) / 2.0
    return mid, mid


# ── Knee plot ─────────────────────────────────────────────────────────────────

def plot_knees(
    knee_info: dict[str, KneedleResult],
    savepath: str,
    show: bool = True,
) -> None:
    """Produce a per-model CROPS Kneedle plot.

    One panel per cost model. Follows Tufte design principles: minimal ink,
    maximum data. The normalised curve runs from (0, 0) to (1, 1) after
    the Kneedle cost inversion; the reference diagonal is the main diagonal
    cn = kn. Elements inside the knee region are blue; outside elements and
    the curve are black. The detected knee is a black star. Drop lines show
    the perpendicular distance from each point to the reference diagonal,
    which equals the Kneedle difference value diff = cost_norm - k_norm.

    Parameters
    ----------
    knee_info:
        Dict mapping cost model name to ``KneedleResult``. Models absent
        from this dict (e.g., because CROPS failed) are removed from the
        figure layout entirely.
    savepath:
        File path for the saved figure.
    show:
        If ``True`` (default), call ``plt.show()`` after saving so the
        figure appears in the current display window or notebook cell.
        Set to ``False`` for a silent save in batch scripts.
    """
    _ensure_dir(savepath)

    # Only plot models that have results. Remove axes for failed models
    # rather than leaving empty whitespace.
    available_models = [m for m in COST_MODELS if m in knee_info]
    skipped = [m for m in COST_MODELS if m not in knee_info]
    if skipped:
        warnings.warn(
            f"plot_knees: no knee_info for model(s) {skipped}. "
            f"Panel(s) omitted. CROPS may have failed for these models.",
            UserWarning,
            stacklevel=2,
        )

    n_panels = len(available_models)
    if n_panels == 0:
        warnings.warn(
            "plot_knees: no models with knee_info. Nothing to plot.",
            UserWarning,
            stacklevel=2,
        )
        return

    with _plot_style():
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(6 * n_panels, 5),
            constrained_layout=True,
        )
        if n_panels == 1:
            axes = [axes]

        fig.suptitle(
            f"CROPS Kneedle plots "
            f"(S sweep = {tuple(S_VALUES)},  "
            f"knee tolerance = {ELBOW_TOLERANCE})",
            fontsize=11,
            fontweight="bold",
        )

        for ax, model in zip(axes, available_models):
            info = knee_info[model]
            ks = info.ks.astype(int)
            k_norm = info.k_norm
            cost_norm = info.cost_norm
            knee_region_set = set(info.knee_region_ks)
            knee_set = set(info.knee_ks)

            if not (len(ks) == len(k_norm) == len(cost_norm)):
                raise ValueError(
                    f"KneedleResult arrays are misaligned for model "
                    f"{model!r}: len(ks)={len(ks)}, "
                    f"len(k_norm)={len(k_norm)}, "
                    f"len(cost_norm)={len(cost_norm)}."
                )

            ax.plot([0, 1], [0, 1], "--", color="gray",
                    linewidth=0.8, alpha=0.5)
            ax.plot(k_norm, cost_norm, "-", color=_COLOUR_CURVE,
                    linewidth=1.5, zorder=2)

            # Single pass over points: drop line, scatter, and annotation.
            # Previously this was two separate loops, doubling the matplotlib
            # call count and adding interpreter overhead per point.
            for k_int, kn, cn in zip(ks, k_norm, cost_norm):
                in_region = k_int in knee_region_set
                colour = _COLOUR_KNEE_REGION if in_region else _COLOUR_OUTSIDE

                # Perpendicular drop line from the point to the main diagonal
                # (cn = kn). The vertical distance to this line equals
                # diff[i], the quantity Kneedle maximises to find the knee.
                fx, fy = _perpendicular_foot(kn, cn)
                ax.plot(
                    [kn, fx], [cn, fy],
                    color=colour, linewidth=1.0, alpha=0.7,
                )

                ax.scatter(
                    kn, cn,
                    color=colour, marker="o",
                    s=100 if in_region else 25,
                    zorder=4,
                )

                # Annotate only knee-region points to avoid clutter.
                if in_region:
                    ax.annotate(
                        f"k={k_int}", (kn, cn),
                        textcoords="offset points",
                        xytext=(6, 5),
                        fontsize=7.5,
                        color="black",
                    )

            if knee_set:
                for k_int in sorted(knee_set):
                    matches = np.where(ks == k_int)[0]
                    if len(matches) == 0:
                        warnings.warn(
                            f"plot_knees: knee k={k_int} not found in ks "
                            f"array for model {model!r}. Skipping star "
                            f"marker.",
                            UserWarning,
                            stacklevel=2,
                        )
                        continue
                    idx = int(matches[0])
                    ax.scatter(
                        k_norm[idx], cost_norm[idx],
                        color="black", marker="*", s=200, zorder=6,
                    )
                knee_label = f"Knee k={sorted(knee_set)}"
            else:
                knee_label = "Knee: none detected"

            ax.set_xlabel("k (normalized)", fontsize=9)
            ax.set_ylabel("Cost (inverted, normalized)", fontsize=9)
            ax.set_title(
                f"Model: {model}  |  {knee_label}  |  "
                f"Knee region k={sorted(knee_region_set)}",
                fontsize=9,
            )
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect("equal")
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

            legend_elements = [
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=_COLOUR_KNEE_REGION, markersize=8,
                       label=f"Knee region k={sorted(knee_region_set)}"),
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=_COLOUR_OUTSIDE, markersize=5,
                       label="Outside knee region"),
                Line2D([0], [0], marker="*", color="w",
                       markerfacecolor="black", markersize=10,
                       label=knee_label),
            ]
            ax.legend(handles=legend_elements, fontsize=7, loc="lower right")

        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)


# ── Downcore plot ─────────────────────────────────────────────────────────────

def plot_downcore(
    best_depth_bkps: list[int],
    selected_k: int,
    savepath: str,
    show: bool = True,
    depth_grid: np.ndarray | None = None,
    fm_interp: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    fm_limits: tuple[float, float] = _DOWNCORE_FM_LIMITS,
    depth_limits_cm: tuple[float, float] = _DOWNCORE_DEPTH_LIMITS_CM,
) -> None:
    """Plot the downcore F14C profile with breakpoints and gap shading.

    Depth is shown as positive values increasing downward, following
    sediment and soil science convention. Interpolated gaps are drawn with
    a dashed line to distinguish them from measured segments and avoid
    implying empirical support in gap regions.

    The default axis limits (F14C ``[0, 1.4]``, depth ``[200, 0]``)
    accommodate the cores currently in ``config.CORES``. Override via
    ``fm_limits`` and ``depth_limits_cm`` for cores that fall outside
    these ranges. The depth limits are in matplotlib order, so the second
    value is the top of the axis.

    Parameters
    ----------
    best_depth_bkps:
        Snapped breakpoint depths in cm.
    selected_k:
        Number of breakpoints (used in the title only).
    savepath:
        File path for the saved figure.
    show:
        If ``True`` (default), call ``plt.show()`` after saving so the
        figure appears in the current display window or notebook cell.
        Set to ``False`` for a silent save in batch scripts.
    depth_grid, fm_interp, valid:
        Optional pre-built outputs of ``interpolation.build_grid``. When
        all three are supplied the function skips rebuilding the grid.
        Pass these from the pipeline result to avoid redundant
        interpolation work.
    fm_limits:
        ``(min, max)`` for the F14C axis. Defaults to ``(0.0, 1.4)``.
    depth_limits_cm:
        ``(bottom, top)`` for the depth axis. Defaults to ``(200.0, 0.0)``.
        Note the convention: positive depth increases downward, so the
        first value is the deepest plotted depth.
    """
    _ensure_dir(savepath)

    # Reuse the precomputed grid when supplied. Rebuilding here costs an
    # avoidable interpolation pass and risks inconsistency if the caller
    # has changed any grid-affecting config in the meantime.
    supplied = sum(x is not None for x in (depth_grid, fm_interp, valid))
    if supplied == 3:
        # Type narrowing — the callers always pass arrays when supplied.
        depth_grid = np.asarray(depth_grid)
        fm_interp = np.asarray(fm_interp)
        valid = np.asarray(valid, dtype=bool)
    elif supplied == 0:
        depth_grid, fm_interp, valid = build_grid(INTERPOLATION_METHOD)
    else:
        raise ValueError(
            "plot_downcore: depth_grid, fm_interp, and valid must be "
            "supplied together (or all None). Got "
            f"{supplied} of 3 provided."
        )

    if len(depth_grid) != len(valid):
        raise ValueError(
            f"plot_downcore: depth_grid and valid mask have different "
            f"lengths ({len(depth_grid)} vs {len(valid)})."
        )
    if not np.all(np.diff(depth_grid) > 0):
        raise ValueError(
            "plot_downcore: depth_grid is not strictly monotonically "
            "increasing."
        )

    with _plot_style():
        fig, ax = plt.subplots(figsize=(4, 6), constrained_layout=True)

        # Draw the profile in two passes: solid for measured cells, dashed
        # for gap cells. This makes the distinction between measurements
        # and interpolated support points visually explicit.
        measured_fm = np.where(valid, fm_interp, np.nan)
        gap_fm = np.where(~valid, fm_interp, np.nan)
        ax.plot(measured_fm, depth_grid,
                color="grey", linewidth=1.0, zorder=1)
        if (~valid).any():
            ax.plot(gap_fm, depth_grid, color="grey", linewidth=1.0,
                    linestyle="--", alpha=0.6, zorder=1)

        for i in range(N_SEGMENTS):
            ax.vlines(
                SEGMENT_FM[i],
                ymin=SEGMENT_TOPS[i], ymax=SEGMENT_BOTTOMS[i],
                color="0.5", linewidth=5, alpha=0.85, zorder=2,
            )

        for d in best_depth_bkps:
            # Check the breakpoint matches a segment top within tolerance.
            matches = np.where(np.isclose(SEGMENT_TOPS, d))[0]
            if len(matches) == 0:
                warnings.warn(
                    f"plot_downcore: breakpoint at {d} cm does not match "
                    f"any segment top within floating-point tolerance. "
                    f"The breakpoint line will still be drawn, but zone "
                    f"assignment may be incorrect.",
                    UserWarning,
                    stacklevel=2,
                )
            ax.axhline(d, color="red", linestyle="--",
                       linewidth=1.5, zorder=4)
            # x=0.02 is in axes-fraction coordinates; y=d is in data units.
            # get_yaxis_transform() is the standard matplotlib idiom for
            # labels that track data-space depth while staying at a fixed
            # horizontal position inside the axes frame.
            ax.text(
                0.02, d, f" {d} cm",
                color="red", fontsize=8, va="bottom",
                transform=ax.get_yaxis_transform(),
            )

        # Vectorised gap-boundary detection. np.diff on the boolean mask
        # produces +1 at False→True transitions and -1 at True→False
        # transitions. Indexing yields gap start/end pairs in O(n) NumPy
        # time instead of an interpreted per-cell loop.
        valid_int = valid.astype(np.int8)
        transitions = np.diff(valid_int, prepend=1, append=1)
        # prepend=1 / append=1 treats out-of-range as "measured", so a gap
        # at either end is bounded by a virtual True on the outside.
        gap_start_idx = np.where(transitions == -1)[0]  # measured→gap
        gap_end_idx = np.where(transitions == 1)[0]     # gap→measured
        # Each start has a matching end thanks to the bracketing prepend
        # and append; assert this is consistent for safety.
        if len(gap_start_idx) != len(gap_end_idx):
            raise RuntimeError(
                "plot_downcore: gap start/end transition mismatch. "
                "This indicates a corrupted valid mask."
            )
        for s_idx, e_idx in zip(gap_start_idx, gap_end_idx):
            gap_start_cm = int(depth_grid[s_idx])
            # e_idx points at the first measured cell after the gap. The
            # gap occupies [s_idx, e_idx), so the shading should end at
            # depth_grid[e_idx] when that index is in range, otherwise
            # one past the last grid cell.
            if e_idx < len(depth_grid):
                gap_end_cm = int(depth_grid[e_idx])
            else:
                gap_end_cm = int(depth_grid[-1]) + 1
            ax.axhspan(gap_start_cm, gap_end_cm,
                       color="lightgrey", alpha=0.4, zorder=0)

        ax.set_xlim(*fm_limits)
        ax.set_ylim(*depth_limits_cm)

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.set_xlabel("F14C (fraction modern)", fontsize=9, labelpad=8)
        ax.set_ylabel("Depth (cm)", fontsize=9)
        ax.set_title(
            f"{ACTIVE_CORE}  |  Downcore F14C  |  k = {selected_k} "
            f"breakpoints  ({INTERPOLATION_METHOD})",
            fontsize=10, fontweight="bold", pad=15,
        )

        legend_elements = [
            Line2D([0], [0], color="grey", linewidth=1,
                   label=f"Interpolated ({INTERPOLATION_METHOD})"),
            Line2D([0], [0], color="0.5", linewidth=5,
                   label="Measured segment"),
            Line2D([0], [0], color="red", linewidth=1.5,
                   linestyle="--", label="Breakpoint"),
        ]
        if (~valid).any():
            legend_elements.append(
                Line2D([0], [0], color="grey", linewidth=1,
                       linestyle="--", alpha=0.6,
                       label="Interpolated gap"),
            )
            legend_elements.append(
                Line2D([0], [0], color="lightgrey", linewidth=6,
                       label="Unsampled region"),
            )
        ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

        plt.savefig(savepath, dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)


# ── Tolerance sensitivity plot ────────────────────────────────────────────────

def plot_tolerance_sweep(
    sweep_df: pd.DataFrame,
    savepath: str,
    show: bool = True,
) -> None:
    """Plot suggested k as a function of knee tolerance.

    A flat line across the sweep demonstrates that the chosen knee tolerance
    is a robust operating point. Variation indicates sensitivity that must
    be reported in the methods section.

    Parameters
    ----------
    sweep_df:
        Output of ``sensitivity.run_tolerance_sweep``.
    savepath:
        File path for the saved figure.
    show:
        If ``True`` (default), call ``plt.show()`` after saving so the
        figure appears in the current display window or notebook cell.
        Set to ``False`` for a silent save in batch scripts.
    """
    _ensure_dir(savepath)

    tolerances = sweep_df["tolerance"].to_numpy(dtype=float)
    ks = sweep_df["suggested_k"].to_numpy(dtype=float)
    # Renamed from `valid` to avoid confusion with the boolean grid mask
    # used elsewhere in this module. These are sweep rows that produced
    # a non-null suggested_k.
    valid_rows = sweep_df[sweep_df["suggested_k"].notna()].copy()

    with _plot_style():
        fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)

        ax.plot(
            tolerances, ks,
            color="steelblue", linewidth=1.5, marker="o",
            markersize=6, zorder=3,
        )

        if not valid_rows.empty:
            mode_vals = valid_rows["suggested_k"].mode()
            if len(mode_vals) > 1:
                warnings.warn(
                    f"plot_tolerance_sweep: multiple modal k values "
                    f"({mode_vals.tolist()}). The modal reference line is "
                    f"suppressed because no single k dominates across "
                    f"tolerances.",
                    UserWarning,
                    stacklevel=2,
                )
                # Do not draw a modal line when the distribution is
                # multimodal; a single line would misrepresent the data.
            else:
                modal_k = int(round(mode_vals.iloc[0]))
                ax.axhline(
                    modal_k, color="steelblue", linewidth=0.8,
                    linestyle="--", alpha=0.5, zorder=2,
                )

        ax.axvline(
            ELBOW_TOLERANCE, color="red", linewidth=1.0,
            linestyle=":", zorder=4,
            label=f"Nominal knee tolerance ({ELBOW_TOLERANCE})",
        )

        k_vals = [int(round(k)) for k in ks if not np.isnan(k)]
        if k_vals:
            y_min = max(0, min(k_vals) - 1)
            y_max = max(k_vals) + 1
            n_ticks = y_max - y_min + 1
            if n_ticks <= _MAX_YTICKS:
                ax.set_yticks(range(y_min, y_max + 1))
            else:
                warnings.warn(
                    f"plot_tolerance_sweep: k range spans {n_ticks} integer "
                    f"values. Using automatic tick spacing for readability.",
                    UserWarning,
                    stacklevel=2,
                )
                ax.locator_params(axis="y", nbins=_MAX_YTICKS)
            ax.set_ylim(y_min - 0.5, y_max + 0.5)

        ax.set_xlabel("Knee tolerance", fontsize=9)
        ax.set_ylabel("Selected k (changepoints)", fontsize=9)
        ax.set_title(
            f"{ACTIVE_CORE}  |  Knee tolerance sensitivity",
            fontsize=10, fontweight="bold",
        )
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.legend(fontsize=8, loc="best")

        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
