"""Configuration constants and core sediment data.

All tuneable parameters live here. Changing a value in this file
propagates automatically to every module that imports from config.

To analyse a different core, set ``ACTIVE_CORE`` to one of the keys in
``CORES``. No other file needs to change.

The five active-core arrays (``SEGMENT_TOPS``, ``SEGMENT_BOTTOMS``,
``SEGMENT_FM``, ``SEGMENT_SD``, ``THICKNESSES``) are write-protected on
import. Attempting to mutate them raises a ``ValueError``. All other
names in this module are constants by convention only. Do not reassign
them after import; downstream arrays are computed once at import time
and will not reflect reassignments.

Units
-----
All depths are in cm. All F14C values are unitless fraction modern.
SD values are 1-sigma AMS measurement uncertainties in the same units
as F14C (i.e. also unitless fraction modern).

Reproducibility
---------------
All stochastic procedures (permutation tests, bootstrap CIs) use
deterministic seeds via ``PERMUTATION_SEED``. Exact numerical results
are reproducible across runs with the same seed. Changing the seed
should not materially alter the suggested segmentation; if it does,
the result is sensitive to random variation and should be treated with
caution.

Version
-------
PIPELINE_VERSION records the pipeline version. It is printed in the
run header and included in all CSV output to support reproducibility
across pipeline versions.
"""
from __future__ import annotations

import warnings

import numpy as np

# RBF cost requires ruptures with scikit-learn kernel support compiled in.
# Checking hasattr(CostRbf) is not sufficient: the class can exist but fail
# at instantiation if the kernel dependency is absent. We therefore attempt
# an actual instantiation and treat any exception as unavailability.
# The result gates COST_MODELS below so that no reassignment is needed later.
def _rbf_is_available() -> bool:
    """Return True only if ruptures CostRbf instantiates without error."""
    try:
        from ruptures.costs import CostRbf  # noqa: F401
        CostRbf()
        return True
    except Exception:  # noqa: BLE001
        return False


_RBF_AVAILABLE: bool = _rbf_is_available()

if not _RBF_AVAILABLE:
    warnings.warn(
        "ruptures CostRbf is not available or failed to instantiate. "
        "'rbf' has been excluded from COST_MODELS. "
        "Install ruptures with scikit-learn kernel support to enable it.",
        ImportWarning,
        stacklevel=2,
    )

# ── Pipeline version ──────────────────────────────────────────────────────────

PIPELINE_VERSION: str = "0.1.0"
"""Semantic version of the pipeline. Increment when behaviour changes."""


# ── Algorithm parameters ──────────────────────────────────────────────────────

INTERPOLATION_METHOD: str = "linear"
"""Interpolation method for gap cells. Either ``"linear"`` or ``"pchip"``.

Used only for PELT cost computation and visualisation. All statistical
inference uses the original measured segment values.
"""

KNEEDLE_SMOOTHING: bool = False
"""If ``True``, apply a smoothing spline before Kneedle knee detection."""

S_VALUES: tuple[float, ...] = (0.0, 0.5, 1.0)
"""Kneedle sensitivity sweep values."""

ELBOW_TOLERANCE: float = 0.20
"""Near-knee region half-width as a fraction of the peak difference (0 < tol < 1)."""

TOLERANCE_SWEEP: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25)
"""Tolerance grid for the knee-region sensitivity analysis."""

COST_MODELS: tuple[str, ...] = (
    ("l2", "l1", "rbf") if _RBF_AVAILABLE
    else ("l2", "l1")
)
"""ruptures cost model names passed to PELT.

L2 detects mean shifts under Gaussian assumptions. L1 detects median
shifts and is robust to outliers. RBF detects more general kernel
shifts; for sparse stratigraphic data its smoothness assumption is
weakly motivated, so RBF results should be interpreted as a supporting
cross-check rather than primary inference. The selection algorithm
treats all three models uniformly through the cross-model consensus
rule, which requires at least two models to agree on a candidate k
before suggesting it.

RBF requires ruptures with scikit-learn kernel support. It is excluded
at import time if instantiation fails, with an ``ImportWarning``.
"""

N_PERMUTATIONS: int = 5000
"""Number of permutations for the null distribution."""

PELT_MIN_SIZE: int = 2
"""Minimum PELT segment length in grid cells (1-cm units)."""

PELT_JUMP: int = 1
"""PELT search jump parameter."""

PERMUTATION_SEED: int = 42
"""Random seed for all permutation and bootstrap procedures.

All stochastic procedures use this seed deterministically. Changing it
should not materially alter the suggested segmentation.
"""

MIN_SEGMENTS_PER_ZONE: int = 2
"""Singleton-zone flag threshold.

Zones with fewer segments than this value are flagged in output but are
not excluded. Scientific justification is required when zones are flagged.
"""

MIN_DELTA_RSS_NORM: float = 0.02
"""Minimum normalised marginal RSS reduction required to avoid the plateau gate.

Formerly named RSS_PLATEAU_THRESHOLD. Renamed to reflect what is actually
being thresholded: the marginal explanatory gain per additional changepoint.

    delta(k) = [RSS(k-1) - RSS(k)] / [k_gap * RSS(0)]

where ``k_gap = k - (k-1) = 1`` under the integer-step sweep used here,
and ``RSS(0)`` is the unsegmented (single-zone) residual sum of squares.
The full definition and computation appear in ``evaluation.py``
(``_normalised_marginal_rss``).

This is a heuristic reproducibility filter, not a formal statistical test.
It operationalises diminishing returns: an additional changepoint must explain
at least this fraction of total unsegmented variance per unit increase in k to
justify the added complexity. The threshold is intended to exclude candidates
with negligible explanatory gain rather than to provide a significance
criterion. Verify robustness via the tolerance sensitivity sweep.
"""

STABILITY_THRESHOLD: float = 0.60
"""Minimum f(b) for a boundary to pass the stability gate (Tier 1).

f(b) is the fraction of unique (model, tolerance) combinations in which
boundary b appears. This threshold is a heuristic operational filter
intended to require strong parameter persistence across analytical
settings. It does not provide a formal significance criterion.

The value 0.60 was chosen so that only Tier 1 boundaries pass the gate.
Empirical calibration across four cores shows that all geologically
plausible boundaries achieve f(b) >= 0.60, while Tier 2 boundaries
(0.40 <= f(b) < 0.60) are model- or tolerance-selective and
scientifically more ambiguous. Verify robustness via the tolerance
sensitivity sweep.

Note: ``STABILITY_THRESHOLD_MODERATE`` must be strictly less than this
value. An assertion at import enforces this constraint.
"""

STABILITY_THRESHOLD_MODERATE: float = 0.40
"""Minimum f(b) for the Tier 2 (moderate persistence) tier (display only).

Boundaries with f(b) >= STABILITY_THRESHOLD_MODERATE but below
STABILITY_THRESHOLD are classified as Tier 2. They are reported in
the breakpoint stability table but do not pass the selection gate.

Must be strictly less than ``STABILITY_THRESHOLD``.
"""

if not (0.0 < STABILITY_THRESHOLD_MODERATE < STABILITY_THRESHOLD <= 1.0):
    raise ValueError(
        f"Stability thresholds are inconsistent. "
        f"Required: 0 < STABILITY_THRESHOLD_MODERATE "
        f"({STABILITY_THRESHOLD_MODERATE}) < STABILITY_THRESHOLD "
        f"({STABILITY_THRESHOLD}) <= 1."
    )

BETA_MAX_HEADROOM: float = 2.0
"""Initial multiplier on the unsegmented cost when resolving beta_max.

Must be strictly positive. The CROPS sweep multiplies the unsegmented
RSS by this value to set an initial upper bound on the penalty range.
"""

BETA_MAX_RETRIES: int = 4
"""Maximum number of doublings of beta_max before a warning is issued.

Must be >= 1. Increase if CROPS fails to find a valid upper bound on
the first attempt for very short or low-variance cores.
"""

if BETA_MAX_HEADROOM <= 0:
    raise ValueError(
        f"BETA_MAX_HEADROOM must be strictly positive; "
        f"got {BETA_MAX_HEADROOM}."
    )
if BETA_MAX_RETRIES < 1:
    raise ValueError(
        f"BETA_MAX_RETRIES must be >= 1; got {BETA_MAX_RETRIES}."
    )


# ── Core validation ───────────────────────────────────────────────────────────

def _validate_core(name: str, core: dict) -> None:
    """Validate a core data dict on import.

    Hard errors (``ValueError``) are raised for conditions that make the
    data geometrically or numerically unusable. Soft warnings
    (``UserWarning``) are issued for conditions that are physically
    implausible but not necessarily wrong (e.g. negative depths, extreme
    F14C values, large SD values).

    Parameters
    ----------
    name:
        Key used to identify the core in error messages.
    core:
        Dict with keys ``tops``, ``bottoms``, ``fm``, ``sd``. All arrays
        must be 1-D with consistent length.

    Raises
    ------
    ValueError
        If array lengths are inconsistent; any segment has non-positive
        thickness; tops or bottoms are not strictly monotonically
        increasing (up to floating-point tolerance); segments overlap;
        F14C values are non-finite; or SD values are non-positive.

    Warns
    -----
    UserWarning
        If any depth is negative (unusual but not impossible for some
        datum conventions); if any F14C value is outside [0, 1.5]
        (physically implausible for natural mineral soil); or if any SD
        value exceeds 0.05 (likely a data entry error given typical AMS
        precision of 0.001-0.003).
    """
    tops = np.asarray(core["tops"], dtype=float)
    bottoms = np.asarray(core["bottoms"], dtype=float)
    fm = np.asarray(core["fm"], dtype=float)
    sd = np.asarray(core["sd"], dtype=float)

    if tops.ndim != 1 or bottoms.ndim != 1 or fm.ndim != 1 or sd.ndim != 1:
        raise ValueError(f"{name}: all arrays must be 1-D.")

    n = len(fm)
    if not (len(tops) == len(bottoms) == len(sd) == n):
        raise ValueError(
            f"{name}: inconsistent array lengths. "
            f"tops={len(tops)}, bottoms={len(bottoms)}, "
            f"fm={n}, sd={len(sd)}."
        )

    # Soft check: negative depths are unusual but valid under some datum
    # conventions (e.g. elevation above a reference surface). Warn rather
    # than error so that non-standard cores are not silently excluded.
    if np.any(tops < 0) or np.any(bottoms < 0):
        warnings.warn(
            f"{name}: negative depth values detected. "
            f"Depths are assumed to increase downward from the surface. "
            f"Verify that the datum convention is intentional.",
            UserWarning,
            stacklevel=3,
        )

    if np.any(bottoms <= tops):
        raise ValueError(
            f"{name}: non-positive segment thickness detected. "
            f"All segments must satisfy top < bottom."
        )

    # Strict monotonicity for both tops and bottoms. Use isclose to avoid
    # rejecting valid data that differs only by floating-point rounding.
    _atol: float = 1e-9  # 0.01 µm: negligible for cm-scale depth data
    if np.any(np.diff(tops) < 0):
        raise ValueError(
            f"{name}: segment tops are not non-decreasing."
        )
    if np.any(np.isclose(np.diff(tops), 0, atol=_atol)):
        raise ValueError(
            f"{name}: duplicate segment tops detected (within {_atol} cm). "
            f"Each segment must have a unique top depth."
        )
    if np.any(np.diff(bottoms) < 0):
        raise ValueError(
            f"{name}: segment bottoms are not non-decreasing."
        )
    if np.any(np.isclose(np.diff(bottoms), 0, atol=_atol)):
        raise ValueError(
            f"{name}: duplicate segment bottoms detected (within {_atol} cm). "
            f"Each segment must have a unique bottom depth."
        )

    if np.any(bottoms[:-1] > tops[1:]):
        raise ValueError(
            f"{name}: overlapping segments detected. "
            f"Segment bottoms must not exceed the following segment top."
        )
    if np.any(~np.isfinite(fm)):
        raise ValueError(f"{name}: non-finite F14C values detected.")
    if np.any(sd <= 0):
        raise ValueError(f"{name}: SD values must be strictly positive.")

    # Soft check: F14C outside [0, 1.5] is physically implausible for
    # natural mineral soil. Bomb-carbon samples (post-1950 surface soils)
    # can exceed 1.0 but should not approach 1.5 without contamination.
    _fm_max: float = 1.5
    if np.any(fm < 0) or np.any(fm > _fm_max):
        bad = np.where((fm < 0) | (fm > _fm_max))[0]
        warnings.warn(
            f"{name}: F14C values outside [0, {_fm_max}] at segment "
            f"indices {bad.tolist()}. Values: {fm[bad].tolist()}. "
            f"Verify these are not data entry errors.",
            UserWarning,
            stacklevel=3,
        )

    # Soft check: SD > 0.05 is roughly 15-100x larger than typical AMS
    # precision (0.001-0.003). Flag as a likely data entry error without
    # blocking the pipeline.
    _sd_warn: float = 0.05
    if np.any(sd > _sd_warn):
        bad = np.where(sd > _sd_warn)[0]
        warnings.warn(
            f"{name}: SD values > {_sd_warn} at segment indices "
            f"{bad.tolist()}. Values: {sd[bad].tolist()}. "
            f"Typical AMS precision is 0.001-0.003. "
            f"Verify these are not data entry errors.",
            UserWarning,
            stacklevel=3,
        )


# ── Core data ─────────────────────────────────────────────────────────────────
# Each entry is a dict with keys: tops, bottoms, fm, sd.
#
# SD values are 1-sigma AMS measurement uncertainties in F14C units. They
# are used for plotting and for propagated measurement-uncertainty summaries
# in zone_summary_table. Inferential weighting uses thickness only.
#
# Asymmetry rationale: SD reflects laboratory analytical precision, not
# environmental or stratigraphic representativeness. Thickness approximates
# the sediment volume each measurement integrates and is therefore the
# appropriate representativeness weight for stratigraphic inference.
# Ignoring SD in segmentation is intentional, not an oversight.
#
# Set ACTIVE_CORE to the key of the core to analyse.

CORES: dict[str, dict] = {
    "HCCN": {
        "tops": np.array([
            0, 4, 8, 16, 19, 24, 30, 37, 53, 68, 93,
            113, 133, 150, 170, 190,
        ]),
        "bottoms": np.array([
            4, 8, 16, 19, 24, 30, 37, 53, 68, 93, 113,
            133, 150, 170, 190, 200,
        ]),
        "fm": np.array([
            1.0658, 1.1136, 1.1741, 1.0633, 1.0087, 0.9256,
            0.9256, 0.8775, 0.7939, 0.5500, 0.2437, 0.1612,
            0.4086, 0.1245, 0.2856, 0.0832,
        ]),
        "sd": np.array([
            0.0022, 0.0029, 0.0027, 0.0032, 0.0028, 0.0019,
            0.0019, 0.0019, 0.0020, 0.0027, 0.0005, 0.0005,
            0.0006, 0.0012, 0.0010, 0.0005,
        ]),
    },
    "HC03": {
        # Segment [0, 0.5) collapses to the single integer-cm cell at depth 0
        # on the 1-cm grid. Statistical inference uses the raw 0.5-cm thickness
        # and is unaffected.
        "tops": np.array([
            0, 0.5, 4, 12, 22, 34, 54, 74, 90, 114, 123, 140, 160, 175,
        ]),
        "bottoms": np.array([
            0.5, 4, 12, 22, 34, 54, 74, 90, 114, 123, 140, 160, 175, 182,
        ]),
        "fm": np.array([
            1.1541, 1.2297, 0.9914, 0.7661, 0.6911, 0.4893,
            0.1049, 0.0633, 0.0565, 0.0715, 0.0639, 0.0498, 0.0460, 0.0363,
        ]),
        "sd": np.array([
            0.0021, 0.0022, 0.0019, 0.0014, 0.0014, 0.0011,
            0.0006, 0.0006, 0.0006, 0.0006, 0.0010, 0.0006, 0.0006, 0.0006,
        ]),
    },
    "HC67": {
        "tops": np.array([20, 25, 30, 35, 45, 62, 84, 104, 115, 134, 173]),
        "bottoms": np.array([25, 30, 35, 40, 62, 84, 104, 115, 134, 154, 185]),
        "fm": np.array([
            0.6686, 0.7804, 0.8754, 0.6231, 0.3233,
            0.3523, 0.1744, 0.1566, 0.0754, 0.0637, 0.0286,
        ]),
        "sd": np.array([
            0.0009, 0.0010, 0.0010, 0.0011, 0.0008,
            0.0007, 0.0011, 0.0007, 0.0009, 0.0006, 0.0005,
        ]),
    },
    "Toolik": {
        # This core contains unsampled gaps between measured segments
        # (e.g. 1-2 cm, 5-7 cm, 19-24 cm). Gaps are filled by interpolation
        # on the 1-cm grid, creating synthetic continuity through unsampled
        # material. Statistical inference uses only the measured segments and
        # is not affected by gap interpolation.
        "tops": np.array([
            0, 2, 3, 4, 7, 9, 14, 24, 34, 39, 44, 49,
            51, 61, 81, 91, 101, 121, 141, 151, 157,
        ]),
        "bottoms": np.array([
            1, 3, 4, 5, 9, 14, 19, 29, 39, 44, 49, 51,
            61, 71, 91, 101, 111, 131, 151, 157, 164,
        ]),
        "fm": np.array([
            1.1270, 1.0830, 1.1920, 1.2090, 1.1590, 0.9310,
            0.9040, 0.7280, 0.7280, 0.7030, 0.8270, 0.7200,
            0.6990, 0.7110, 0.4690, 0.4160, 0.3740, 0.3200,
            0.3230, 0.3380, 0.3120,
        ]),
        "sd": np.array([
            0.0019, 0.0024, 0.0026, 0.0025, 0.0019, 0.0015,
            0.0019, 0.0015, 0.0018, 0.0012, 0.0023, 0.0013,
            0.0013, 0.0013, 0.0013, 0.0011, 0.0010, 0.0011,
            0.0008, 0.0008, 0.0009,
        ]),
    },
}

# Validate all cores on import. Catches data entry errors before they
# propagate silently through the pipeline.
for _name, _core in CORES.items():
    _validate_core(_name, _core)

# Freeze every raw array inside the CORES catalogue. Without this, a caller
# could mutate ``CORES["Toolik"]["fm"][0] = 0`` and silently corrupt the
# data dictionary used by any module that later extracts a different core.
# The active-core arrays below are separately frozen after extraction, so
# this protects callers who consult CORES directly (e.g. for documentation
# or cross-core comparison).
for _name, _core in CORES.items():
    for _field in ("tops", "bottoms", "fm", "sd"):
        _core[_field].flags.writeable = False


# ── Active core selection ─────────────────────────────────────────────────────

ACTIVE_CORE: str = "HC03"
"""Key of the core to analyse. Must be one of: HCCN, HC03, HC67, Toolik."""

if ACTIVE_CORE not in CORES:
    raise ValueError(
        f"ACTIVE_CORE={ACTIVE_CORE!r} is not a valid key. "
        f"Choose one of {sorted(CORES)}."
    )


# ── Active core arrays (read-only) ───────────────────────────────────────────

_core = CORES[ACTIVE_CORE]

SEGMENT_TOPS: np.ndarray = _core["tops"].copy()
"""Segment top depths in cm (read-only)."""

SEGMENT_BOTTOMS: np.ndarray = _core["bottoms"].copy()
"""Segment bottom depths in cm (read-only)."""

SEGMENT_FM: np.ndarray = _core["fm"].copy()
"""Measured F14C values per segment (read-only)."""

SEGMENT_SD: np.ndarray = _core["sd"].copy()
"""1-sigma AMS measurement uncertainties in F14C units (read-only)."""

THICKNESSES: np.ndarray = SEGMENT_BOTTOMS - SEGMENT_TOPS
"""Segment thicknesses in cm (read-only)."""

N_SEGMENTS: int = len(SEGMENT_FM)
"""Number of measured segments in the active core."""

for _arr in (
    SEGMENT_TOPS, SEGMENT_BOTTOMS, SEGMENT_FM, SEGMENT_SD,
    THICKNESSES,
):
    _arr.flags.writeable = False

# Warn when PELT_MIN_SIZE exceeds the thinnest measured segment. PELT
# operates on 1-cm grid cells (assumed grid resolution; see
# interpolation.py). A segment thinner than PELT_MIN_SIZE cm occupies
# fewer grid cells than the minimum segment length and cannot be recovered
# as a standalone zone regardless of its F14C value. This is a limitation
# of the interpolation approach, not a bug.
#
# Note: this warning is only valid under the 1-cm grid assumption. If
# INTERPOLATION_METHOD or grid resolution ever changes, revisit this check.
_thicknesses_for_check = THICKNESSES[np.isfinite(THICKNESSES)]
if len(_thicknesses_for_check) > 0:
    _min_thickness: float = float(_thicknesses_for_check.min())
    if PELT_MIN_SIZE > _min_thickness:
        warnings.warn(
            f"PELT_MIN_SIZE={PELT_MIN_SIZE} grid cells (1-cm grid assumed) "
            f"exceeds the thinnest segment in {ACTIVE_CORE!r} "
            f"({_min_thickness:.4g} cm). "
            f"That segment cannot be recovered as a standalone zone. "
            f"See interpolation.py for details.",
            UserWarning,
            stacklevel=2,
        )


# ── Heuristic clustering and tolerance constants ──────────────────────────────
#
# These constants were previously embedded as magic numbers in
# interpolation.py and evaluation.py. They are centralised here so that
# every heuristic threshold appears in one place. Each carries a brief
# rationale; deeper context lives in the original module docstrings.

LARGE_GAP_THRESHOLD_CM: int = 10
"""Minimum contiguous unsampled gap length that triggers a UserWarning.

A heuristic operational threshold. Used by ``interpolation.build_grid``
to flag gaps large enough for PELT to potentially place changepoints
inside unsampled material.
"""

BREAKPOINT_SNAP_TOL_CM: int = 5
"""Tolerance for clustering nearby snapped breakpoints (cm).

Used by ``evaluation._cluster_breakpoints`` and downstream nesting
consistency checks. Two breakpoints within this distance are treated as
the same boundary for stability and nesting purposes.
"""

NESTING_TOL_CM: int = 5
"""Tolerance for matching boundaries across adjacent-k segmentations (cm).

A breakpoint in a lower-k segmentation is treated as 'nested' in a
higher-k segmentation when a boundary within this distance exists.
"""

# ── Output paths ──────────────────────────────────────────────────────────────
#
# All output filenames are routed through this dict so that callers
# running the pipeline from a non-default working directory can override
# the destinations without modifying ``main()``.

OUTPUT_PATHS: dict[str, str] = {
    "zone_summary_csv": "zone_summary.csv",
    "evidence_table_csv": "knee_region_combinations.csv",
    "tolerance_sweep_csv": "tolerance_sweep.csv",
    "crops_knee_png": "crops_knee.png",
    "downcore_png": "downcore_profile.png",
    "tolerance_sweep_png": "tolerance_sweep.png",
}
"""Output file paths for all pipeline products.

Keys
----
zone_summary_csv:
    Thickness-weighted zone statistics with bootstrap CIs.
evidence_table_csv:
    All knee-region candidate segmentations and their scores.
tolerance_sweep_csv:
    Suggested k as a function of knee tolerance.
crops_knee_png:
    Kneedle plots for each cost model.
downcore_png:
    Downcore F14C profile with suggested breakpoints.
tolerance_sweep_png:
    Suggested k versus knee tolerance.
"""


# ── Public API ────────────────────────────────────────────────────────────────

__all__ = [
    # Pipeline version
    "PIPELINE_VERSION",
    # Algorithm parameters
    "INTERPOLATION_METHOD",
    "KNEEDLE_SMOOTHING",
    "S_VALUES",
    "ELBOW_TOLERANCE",
    "TOLERANCE_SWEEP",
    "COST_MODELS",
    "N_PERMUTATIONS",
    "PELT_MIN_SIZE",
    "PELT_JUMP",
    "PERMUTATION_SEED",
    "MIN_SEGMENTS_PER_ZONE",
    "MIN_DELTA_RSS_NORM",
    "STABILITY_THRESHOLD",
    "STABILITY_THRESHOLD_MODERATE",
    "BETA_MAX_HEADROOM",
    "BETA_MAX_RETRIES",
    # Heuristic constants
    "LARGE_GAP_THRESHOLD_CM",
    "BREAKPOINT_SNAP_TOL_CM",
    "NESTING_TOL_CM",
    # Output paths
    "OUTPUT_PATHS",
    # Core catalogue
    "CORES",
    # Active core
    "ACTIVE_CORE",
    "SEGMENT_TOPS",
    "SEGMENT_BOTTOMS",
    "SEGMENT_FM",
    "SEGMENT_SD",
    "THICKNESSES",
    "N_SEGMENTS",
]
