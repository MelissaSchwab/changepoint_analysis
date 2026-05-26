"""Weighted rank permutation ANOVA and pairwise post-hoc tests.

All inference operates on the original measured segment F14C values with
segment thickness as sampling-support weights reflecting the represented
depth interval. The interpolated 1-cm grid is never used here.

Omnibus statistic
-----------------
The omnibus statistic is a weighted between-group rank dispersion of ANOVA
form. It is NOT a Kruskal-Wallis statistic. Its null distribution is built
entirely by permutation.

Null hypothesis
---------------
F14C values assigned to depth intervals are exchangeable under the null
segmentation model. Equivalently: the observed between-zone rank separation
could have arisen by random reassignment of F14C values to depth intervals
while holding zone boundaries and thicknesses fixed. Thickness is treated
as a sampling-support weight, not a property of the F14C value. Weights are
therefore held fixed while ranks permute. A joint permutation of (rank,
weight) would target a different null and is explicitly avoided.

Post-selection inference
------------------------
Changepoints are detected and k is selected before testing. The resulting
p-values are conditional on the selected segmentation and are not valid for
confirming that changepoints exist at those locations. They quantify the
degree of F14C separation among the zones identified by the changepoint model,
conditional on the selected segmentation. They should be interpreted as
descriptive support for the segmentation, not as independent inferential tests.

Pairwise tests
--------------
Pairwise tests use a pooled-rank statistic: the absolute difference of
thickness-weighted rank means, where ranks are computed across the full
dataset. The null distribution is built by permutation, structurally
identical to the omnibus mechanism. Multiple-comparison adjustment uses
the Holm (1979) step-down procedure.

Reference
---------
Holm S (1979) Scand J Stat 6(2), 65–70.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from changepoint_analysis.config import N_PERMUTATIONS, PERMUTATION_SEED


# ── Zone summary table ────────────────────────────────────────────────────────

def zone_summary_table(
    fm_values: np.ndarray,
    sd_values: np.ndarray,
    thicknesses: np.ndarray,
    tops: np.ndarray,
    bottoms: np.ndarray,
    zone_labels: np.ndarray,
    n_boot: int = 10000,
    seed: int = PERMUTATION_SEED,
    ci: float = 0.95,
) -> pd.DataFrame:
    """Compute thickness-weighted summary statistics per zone.

    For each zone identified by PELT/CROPS/Kneedle, computes the
    thickness-weighted mean F14C, weighted SD, and a bootstrap confidence
    interval on the weighted mean. Bootstrapping is used rather than a
    parametric CI because zone sample sizes are small and no distributional
    assumption is warranted.

    Bootstrap procedure
    -------------------
    Segments within a zone are resampled uniformly with replacement.
    Thickness weighting is applied only inside the estimator. Resampling
    with probability proportional to thickness and then re-weighting would
    double-weight thick intervals and distort uncertainty downward. The CI
    is the percentile interval of the bootstrap distribution of weighted
    means.

    Parameters
    ----------
    fm_values:
        Measured F14C for each segment.
    sd_values:
        1-sigma AMS measurement uncertainty for each segment.
    thicknesses:
        Segment thicknesses used as weights.
    tops:
        Segment top depths in cm.
    bottoms:
        Segment bottom depths in cm.
    zone_labels:
        Integer zone label for each segment, from
        ``assign_zones_to_segments``.
    n_boot:
        Number of bootstrap resamples.
    seed:
        Random seed for reproducibility.
    ci:
        Confidence level for the bootstrap interval.

    Returns
    -------
    pd.DataFrame
        One row per zone. Columns: ``zone``, ``depth_top_cm``,
        ``depth_bottom_cm``, ``n_segments``, ``weighted_mean_fm``,
        ``weighted_sd`` (descriptive weighted dispersion),
        ``ci{level}_lower``, ``ci{level}_upper``,
        ``sigma_propagated``, ``sigma_combined``, ``mean_ams_sd``.
        Zones with fewer than 2 segments return NaN for all uncertainty
        columns and emit a RuntimeWarning.
    """
    fm = np.asarray(fm_values, dtype=float)
    sd = np.asarray(sd_values, dtype=float)
    th = np.asarray(thicknesses, dtype=float)
    tp = np.asarray(tops, dtype=float)
    bt = np.asarray(bottoms, dtype=float)
    zl = np.asarray(zone_labels, dtype=int)

    rng = np.random.default_rng(seed)
    alpha_tail = (1.0 - ci) / 2.0
    ci_label = int(ci * 100)
    rows = []

    # Threshold below which the percentile bootstrap CI on the weighted
    # mean is known to be biased for skewed distributions and to have
    # poor finite-sample coverage. A BCa interval would correct for both
    # but is out of scope here. We warn so that the researcher can apply
    # additional caution to small-zone CIs.
    _small_zone_threshold: int = 5

    for zone in np.unique(zl):
        mask = zl == zone
        fm_z = fm[mask]
        sd_z = sd[mask]
        th_z = th[mask]
        tp_z = tp[mask]
        bt_z = bt[mask]
        n = int(mask.sum())

        if n < 2:
            warnings.warn(
                f"Zone {zone} contains only {n} segment(s). "
                f"Weighted SD, bootstrap CI, and propagated uncertainty "
                f"are unreliable. Reporting NaN for these columns.",
                RuntimeWarning,
                stacklevel=2,
            )
            rows.append({
                "zone": int(zone),
                "depth_top_cm": float(tp_z.min()),
                "depth_bottom_cm": float(bt_z.max()),
                "n_segments": n,
                "weighted_mean_fm": round(float(fm_z[0]), 4),
                "weighted_sd": float("nan"),
                f"ci{ci_label}_lower": float("nan"),
                f"ci{ci_label}_upper": float("nan"),
                "sigma_propagated": float("nan"),
                "sigma_combined": float("nan"),
                "mean_ams_sd": round(float(sd_z.mean()), 4),
            })
            continue

        if 2 <= n < _small_zone_threshold:
            warnings.warn(
                f"Zone {zone} contains {n} segments. The percentile "
                f"bootstrap CI has poor finite-sample coverage and may "
                f"be biased for skewed distributions when n < "
                f"{_small_zone_threshold}. Treat the reported CI as "
                f"approximate. A BCa interval would be preferable.",
                RuntimeWarning,
                stacklevel=2,
            )

        total_w = th_z.sum()
        w_norm = th_z / total_w

        w_mean = float((w_norm * fm_z).sum())
        w_var = float((w_norm * (fm_z - w_mean) ** 2).sum())
        w_sd = float(np.sqrt(w_var))

        # Vectorised bootstrap: draw all resamples in one call and compute
        # all weighted means with matrix operations. This replaces a Python
        # loop of n_boot iterations with two NumPy operations.
        # Memory: n_boot × n × 8 bytes. For n_boot=10000, n≤20: ~1.6 MB.
        idx_matrix = rng.integers(0, n, size=(n_boot, n))
        fm_boot = fm_z[idx_matrix]                          # (n_boot, n)
        th_boot = th_z[idx_matrix]                          # (n_boot, n)
        # Guard against zero-sum resamples. If a bootstrap row resamples only
        # segments with zero thickness (possible if thicknesses contain zeros,
        # though config._validate_core normally prevents this), the row sum
        # is zero and the division below would produce NaN. Replace any
        # zero row sum with 1.0 so the resulting w_boot row is all zeros and
        # contributes a 0.0 mean, which is then dropped by np.nanpercentile.
        row_sums = th_boot.sum(axis=1, keepdims=True)
        zero_rows = row_sums == 0
        if zero_rows.any():
            warnings.warn(
                f"Zone {zone}: {int(zero_rows.sum())} bootstrap resample(s) "
                f"drew only zero-thickness segments. These rows are dropped "
                f"from the CI computation.",
                RuntimeWarning,
                stacklevel=2,
            )
        safe_sums = np.where(zero_rows, 1.0, row_sums)
        w_boot = th_boot / safe_sums
        boot_means = (w_boot * fm_boot).sum(axis=1)         # (n_boot,)
        if zero_rows.any():
            boot_means = boot_means[~zero_rows.ravel()]

        if boot_means.size == 0:
            warnings.warn(
                f"Zone {zone}: all bootstrap resamples drew only "
                f"zero-thickness segments. Reporting NaN CI.",
                RuntimeWarning,
                stacklevel=2,
            )
            ci_lo = float("nan")
            ci_hi = float("nan")
            sigma_boot = float("nan")
        else:
            ci_lo = float(np.percentile(boot_means, 100 * alpha_tail))
            ci_hi = float(np.percentile(boot_means, 100 * (1.0 - alpha_tail)))
            sigma_boot = float(np.std(boot_means, ddof=1))

        # Propagated AMS measurement uncertainty on the weighted mean.
        # sigma_prop = sqrt(sum((w_i * SD_i)^2)). Much smaller than the
        # bootstrap CI but reported for completeness.
        sigma_prop = float(np.sqrt(np.sum((w_norm * sd_z) ** 2)))

        # Combined uncertainty in quadrature. sigma_boot is the SD of the
        # bootstrap distribution computed directly to avoid the Gaussian
        # symmetry assumption implicit in (ci_hi - ci_lo) / (2 * 1.96).
        # sigma_boot was set above (in either the success or zero-row branch).
        sigma_combined = float(np.sqrt(sigma_prop ** 2 + sigma_boot ** 2))

        rows.append({
            "zone": int(zone),
            "depth_top_cm": float(tp_z.min()),
            "depth_bottom_cm": float(bt_z.max()),
            "n_segments": n,
            "weighted_mean_fm": round(w_mean, 4),
            "weighted_sd": round(w_sd, 4),
            f"ci{ci_label}_lower": round(ci_lo, 4),
            f"ci{ci_label}_upper": round(ci_hi, 4),
            "sigma_propagated": round(sigma_prop, 4),
            "sigma_combined": round(sigma_combined, 4),
            "mean_ams_sd": round(float(sd_z.mean()), 4),
        })

    return pd.DataFrame(rows)


# ── Omnibus statistic ─────────────────────────────────────────────────────────

def _weighted_rank_anova_statistic(
    ranks: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute the weighted between-group rank dispersion statistic.

    .. math::

        H_w = \\frac{\\sum_g W_g (\\bar{r}_g - \\bar{r})^2}{V_r}

    where means and variance are thickness-weighted. This is a custom
    weighted extension of the standard between-group rank dispersion used
    in one-way ANOVA on ranks. Thickness weights are incorporated because
    segments represent unequal depth intervals. Dividing by the weighted
    rank variance places the statistic on a variance-normalised scale,
    making it invariant to linear scaling of the ranks.

    This is NOT a Kruskal-Wallis chi-squared statistic and NOT the
    Conover-Iman ANOVA statistic. Calibration is entirely by permutation;
    no distributional assumption on H_w is required.

    Parameters
    ----------
    ranks:
        Pooled ranks of F14C values across all segments.
    groups:
        Integer zone label for each segment.
    weights:
        Segment thicknesses.

    Returns
    -------
    float
        The weighted rank dispersion statistic H_w. Returns ``NaN`` when
        total weight is zero or rank variance is zero. NaN propagates
        through the permutation null and yields a NaN p-value, signalling
        that the test was not computable rather than that the observed
        separation was indistinguishable from chance.
    """
    total_w = weights.sum()
    if total_w <= 0:
        return float("nan")
    mean_rank = (weights * ranks).sum() / total_w
    var_rank = (weights * (ranks - mean_rank) ** 2).sum() / total_w
    if var_rank <= 0:
        return float("nan")
    H = 0.0
    for g in np.unique(groups):
        mask = groups == g
        W_g = weights[mask].sum()
        if W_g <= 0:
            continue
        mean_rank_g = (weights[mask] * ranks[mask]).sum() / W_g
        H += W_g * (mean_rank_g - mean_rank) ** 2
    return H / var_rank


def weighted_rank_permutation_anova(
    values: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    n_perm: int = N_PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> tuple[float, float]:
    """Run a permutation-based weighted rank ANOVA omnibus test.

    Parameters
    ----------
    values:
        F14C values for each measured segment.
    groups:
        Integer zone label for each segment.
    weights:
        Segment thicknesses. Held fixed during permutation.
    n_perm:
        Number of permutations for the null distribution.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    tuple of (float, float)
        ``(H_obs, p_value)``. H_obs is the observed statistic. p_value
        uses the +1/+1 continuity correction. Both are NaN when fewer
        than two unique groups are present.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups, dtype=int)
    weights = np.asarray(weights, dtype=float)

    if len(np.unique(groups)) < 2:
        return np.nan, np.nan

    ranks = rankdata(values, method="average")
    H_obs = _weighted_rank_anova_statistic(ranks, groups, weights)

    rng = np.random.default_rng(seed)
    n_obs = len(values)

    # Vectorised permutation: generate all permutation indices in one call,
    # then compute H for each row with a broadcasting-friendly formulation.
    # This avoids n_perm Python loop iterations at the cost of holding a
    # (n_perm, n_obs) index matrix in memory (~40 MB for n_perm=5000, n=200).
    # For the typical core sizes here (n ≤ 200) this is well within budget.
    perm_matrix = rng.permuted(
        np.broadcast_to(np.arange(n_obs), (n_perm, n_obs)).copy(),
        axis=1,
    )                                          # (n_perm, n_obs)

    # Fully vectorised null computation.
    #
    # For each permutation row p, H(p) = sum_g W_g * (mean_rank_g(p) - mean)^2 / V.
    # Both mean (the global weighted rank mean) and V (the global weighted rank
    # variance) are invariant under permutation of ranks. Only mean_rank_g(p)
    # changes. Build a one-hot group membership matrix M of shape (n_obs, n_groups);
    # then weighted per-group rank sums for all permutations are
    #     ((ranks[perm_matrix]) * weights[None, :]) @ M
    # of shape (n_perm, n_groups). Divide row-wise by W_g to get per-group
    # weighted means under each permutation.
    unique_groups, group_indices = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)
    membership = np.zeros((n_obs, n_groups), dtype=float)
    membership[np.arange(n_obs), group_indices] = 1.0
    W_g = weights @ membership                                 # (n_groups,)
    safe_W_g = np.where(W_g > 0, W_g, 1.0)

    total_w = weights.sum()
    if total_w <= 0 or n_obs == 0:
        return float(H_obs), float("nan")
    global_mean = float((weights * ranks).sum() / total_w)
    var_rank = float((weights * (ranks - global_mean) ** 2).sum() / total_w)
    if var_rank <= 0:
        return float(H_obs), float("nan")

    permuted_ranks = ranks[perm_matrix]                        # (n_perm, n_obs)
    weighted_perm_ranks = permuted_ranks * weights[None, :]    # (n_perm, n_obs)
    group_weighted_sums = weighted_perm_ranks @ membership     # (n_perm, n_groups)
    group_means = group_weighted_sums / safe_W_g[None, :]      # (n_perm, n_groups)
    centred = group_means - global_mean
    H_null = (W_g[None, :] * centred ** 2).sum(axis=1) / var_rank

    p_value = (np.sum(H_null >= H_obs) + 1) / (n_perm + 1)
    return float(H_obs), float(p_value)


# ── Multiple-comparison adjustment ───────────────────────────────────────────

def _holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Apply the Holm-Bonferroni step-down adjustment (Holm 1979).

    Sorts p-values ascending, multiplies the i-th by (n - i + 1), takes
    the running maximum to enforce monotonicity, and caps at 1.

    Parameters
    ----------
    p_values:
        Array of raw p-values.

    Returns
    -------
    np.ndarray
        Adjusted p-values in the same order as the input. Returns an
        empty array and emits a RuntimeWarning on empty input. Returns
        the input clipped to 1.0 for a single-element input.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        warnings.warn(
            "Holm-Bonferroni called on an empty p-value array.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.array([], dtype=float)
    if n == 1:
        return np.minimum(p, 1.0)
    order = np.argsort(p)
    p_sorted = p[order]
    multipliers = n - np.arange(n)
    raw_adj = np.minimum(p_sorted * multipliers, 1.0)
    p_adj_sorted = np.maximum.accumulate(raw_adj)
    p_adj = np.empty_like(p_adj_sorted)
    p_adj[order] = p_adj_sorted
    return p_adj


# ── Pairwise post-hoc ─────────────────────────────────────────────────────────

def weighted_pairwise_test(
    values: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    n_perm: int = N_PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> pd.DataFrame:
    """Run pairwise weighted permutation tests on rank means.

    For each zone pair, tests whether their thickness-weighted rank means
    differ. The statistic is the absolute difference of weighted rank means
    computed on ranks pooled across the full dataset. The null distribution
    is built by permuting pooled ranks with group labels fixed, structurally
    identical to the omnibus permutation mechanism.

    Multiple-comparison adjustment uses the Holm (1979) step-down procedure
    across all unique pairs.

    Parameters
    ----------
    values:
        F14C values for each measured segment.
    groups:
        Integer zone label for each segment.
    weights:
        Segment thicknesses treated as sampling-support weights.
    n_perm:
        Number of permutations per pair.
    seed:
        Random seed.

    Returns
    -------
    pd.DataFrame
        Square DataFrame of Holm-adjusted p-values, indexed by zone label.
        Diagonal entries are 1.0. P-values are conditional on the selected
        segmentation, not independent tests of zone separation.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups, dtype=int)
    weights = np.asarray(weights, dtype=float)
    unique_g = np.unique(groups)
    n_g = len(unique_g)

    ranks = rankdata(values, method="average")

    raw_p = np.full((n_g, n_g), np.nan)
    pair_indices: list[tuple[int, int]] = []

    for i in range(n_g):
        for j in range(i + 1, n_g):
            mask = (groups == unique_g[i]) | (groups == unique_g[j])
            r_pair = ranks[mask]
            g_pair = groups[mask]
            w_pair = weights[mask]
            mi = g_pair == unique_g[i]
            mj = g_pair == unique_g[j]
            wi_obs = w_pair[mi].sum()
            wj_obs = w_pair[mj].sum()
            pair_indices.append((i, j))

            if wi_obs <= 0 or wj_obs <= 0:
                continue

            mean_i = (w_pair[mi] * r_pair[mi]).sum() / wi_obs
            mean_j = (w_pair[mj] * r_pair[mj]).sum() / wj_obs
            stat_obs = abs(mean_i - mean_j)

            # Vectorised permutation: draw all permutations of r_pair at once.
            # perm_matrix[p] is r_pair permuted — same semantics as the
            # original r_perm = pair_rng.permutation(r_pair).
            # Cantor pairing produces a unique integer for every (i, j) pair
            # regardless of n_g. The earlier formula ``seed + i * n_g + j``
            # is injective only when ``0 <= j < n_g`` strictly holds. Cantor
            # pairing carries no such precondition and is robust to future
            # refactors that may iterate over a different index range.
            pair_seed = seed + ((i + j) * (i + j + 1)) // 2 + j
            pair_rng = np.random.default_rng(pair_seed)
            n_pair = len(r_pair)
            idx_matrix = pair_rng.permuted(
                np.broadcast_to(
                    np.arange(n_pair), (n_perm, n_pair),
                ).copy(),
                axis=1,
            )                                    # (n_perm, n_pair) index array
            r_perm_matrix = r_pair[idx_matrix]  # (n_perm, n_pair) rank values
            null = np.abs(
                (r_perm_matrix[:, mi] * w_pair[mi]).sum(axis=1) / wi_obs
                - (r_perm_matrix[:, mj] * w_pair[mj]).sum(axis=1) / wj_obs
            )                                    # (n_perm,) — fully vectorised

            p = (np.sum(null >= stat_obs) + 1) / (n_perm + 1)
            raw_p[i, j] = p
            raw_p[j, i] = p

    valid_pairs = [
        (i, j) for i, j in pair_indices if not np.isnan(raw_p[i, j])
    ]
    flat_p = np.array([raw_p[i, j] for i, j in valid_pairs])
    adj_flat = _holm_bonferroni(flat_p) if flat_p.size else flat_p

    adj_p = np.full((n_g, n_g), np.nan)
    for (i, j), p_adj in zip(valid_pairs, adj_flat):
        adj_p[i, j] = p_adj
        adj_p[j, i] = p_adj
    np.fill_diagonal(adj_p, 1.0)

    return pd.DataFrame(adj_p, index=unique_g, columns=unique_g)
