# Pipeline overview

This document explains why the pipeline works the way it does. For
instructions on how to run the pipeline, see `__init__.py`.

The pipeline suggests a segmentation of a downcore F14C profile into zones of
distinct radiocarbon activity. It does not identify the "true" or "best"
segmentation. The output is an automated operating point that the researcher
should evaluate in the context of the core's sampling design, depth
resolution, and scientific objectives.

---

## Step 1 — Grid construction (`interpolation.py`)

Each core is resampled to a uniform 1-cm depth grid. Within each measured
segment, F14C is treated as constant across all grid cells belonging to that
segment (left-closed, right-open interval). Gaps between measured segments
are filled by linear interpolation (default) or PCHIP. A nearest-neighbour
fallback fills any remaining boundary gaps. A boolean `valid` mask records
which cells are measured versus interpolated.

Gaps exceeding 10 cm trigger a `UserWarning` because PELT may place
breakpoints inside unsampled regions where no F14C measurement exists.

**Critical separation:** the interpolated grid is used only for PELT cost
computation and visualisation. All statistical inference operates on the
original measured segment values with segment thickness as weights.

---

## Step 2 — Penalised changepoint detection over a continuous penalty range (`crops.py`)

PELT (Killick et al. 2012) minimises a penalised segmentation cost on the
1-cm grid. Three cost functions are used independently: L2, L1, and RBF
(via the `ruptures` library; Truong et al. 2020). Running PELT at a single
fixed penalty requires an arbitrary choice. CROPS (Haynes, Eckley &
Fearnhead 2017) avoids this by recovering all distinct optimal segmentations
over a continuous penalty range [beta_min, beta_max]. It recursively bisects
intervals where the changepoint count jumps by more than one, yielding a
complete cost-versus-k curve for each cost function. Detected grid-index
breakpoints are forward-snapped to the nearest measured segment boundary.

CROPS reduces the number of PELT runs from O(K) to at most O(K log K) in
the number of distinct segmentations, making the sweep feasible even for
long cores with many candidate k values.

---

## Step 3 — Candidate region identification (`kneedle.py`)

The Kneedle algorithm (Satopaa et al. 2011) is applied to each
cost-versus-k curve across a sweep of sensitivity values (S) and tolerance
values. For each (model, S, tolerance) combination, Kneedle identifies the
primary detected knee — the k with the highest value in the normalised
difference curve — of the convex decreasing curve. All k values within a
tolerance-defined band around that knee are retained as candidates. The final
candidate region is the union of knee regions across all three cost models.

The knee region is an operationally defined heuristic band, not a derived
confidence region. The tolerance sensitivity analysis in Step 8 tests whether
the suggested k is stable across the sweep.

---

## Step 4 — Breakpoint stability scoring (`evaluation.py`)

For every candidate boundary position, f(b) is computed as:

    f(b) = number of unique (model, tolerance) combinations containing b
           ---------------------------------------------------------------
           total number of unique (model, tolerance) combinations

This definition makes f(b) a property of reproducibility across analytical
settings. It is not inflated by the number of k values that survive within
any single (model, tolerance) pair, because boundaries are unioned across k
before counting.

This metric does not fully separate cross-model reproducibility from
tolerance robustness; the two dimensions are combined into a single score.
The `models` column in the evidence table reports which cost functions support
each candidate segmentation, allowing the researcher to inspect the
distribution of support directly.

The spatial uncertainty sigma_b is the standard deviation of snapped
boundary positions across all combinations that contain b. Low sigma_b
indicates a tightly localised transition. High sigma_b indicates a diffuse
or uncertain boundary.

---

## Step 5 — Evidence table construction (`evaluation.py`)

All unique snapped segmentations in the candidate region are evaluated. For
each candidate segmentation, the following quantities are computed on the
original measured F14C values with thickness weights.

**Structural robustness (primary):**
- `min_stability`: minimum f(b) across all breakpoints in the segmentation.
  Reflects how reproducibly each boundary appears across models and
  tolerances.

**Explanatory gain (primary):**
- `delta_rss_norm`: marginal normalised RSS reduction per additional
  changepoint, relative to the unsegmented baseline. Reflects diminishing
  returns as k increases.

**Descriptive separability (secondary):**
- `omni_p`: post-selection omnibus permutation p-value (weighted rank ANOVA).
  Conditional on the selected segmentation. Not an independent test of
  whether changepoints exist. Reported as descriptive support only.
- `min_adjacent_d`: minimum standardised mean difference between adjacent
  zones. Flags micro-segmentation where neighbouring zones are not
  meaningfully separated.

**Interpretability:**
- `singleton_flag`: True when any zone contains fewer than 2 segments.
  Flagged in output but not used as a gate.
- `nested_in_next`: whether all boundaries in this segmentation appear
  within 5 cm of a boundary in the next-k candidate. Used as a tiebreaker
  only.

---

## Step 6 — Segmentation suggestion (`evaluation.py`)

The pipeline produces a *suggested* segmentation, not a definitive answer.
Selection proceeds in two sequential steps.

### Step 6a — Cross-model knee consensus

A k achieves consensus when Kneedle detects it as a knee in at least 2 of
3 cost models with mean S-detection frequency >= 0.5. If consensus exists,
that k is suggested immediately without applying further criteria.

### Step 6b — Maximal stable resolution

Applied only when Step 6a finds no consensus. Among knee-region candidates,
the highest k is selected whose every breakpoint satisfies:

    min f(b) >= stability threshold (default 0.60, HIGH stability)

This is the **only** admissibility gate. Stability is treated as the
primary evidence of boundary reality because reproducibility across cost
models and tolerance settings is a robustness check that a per-model RSS
measure cannot provide. A boundary with min f(b) >= 0.60 has appeared in
at least 60% of (cost model, tolerance) combinations and is therefore
defensible as a real stratigraphic transition.

If multiple candidates tie at the highest stable k, the tiebreakers are
applied in this order:

1. **Non-plateau first.** Among tied-k candidates, prefer the one whose
   marginal RSS gain meets the plateau threshold
   (`delta(k) >= MIN_DELTA_RSS_NORM`, default 2%). The plateau metric is
   demoted from a hard gate to a tiebreaker. The motivation is that two
   equally stable segmentations are preferred in the order of how much
   additional variance they explain, but the plateau test alone should
   not eliminate a reproducible boundary just because the F14C jump is
   modest. Smaller but consistent boundaries are still informative for
   stratigraphic interpretation.
2. **Nested-in-next first.** Prefer the candidate whose every boundary
   appears within 5 cm of a boundary in the next-k candidate.
3. **Most source models.** Prefer the candidate supported by the
   largest number of cost models.

If no candidate meets the stability threshold at any k, the pipeline
falls back to the highest-k row in the evidence table and reports the
fallback in the selection mode label so the researcher knows the gate
was bypassed.

**The omnibus p-value is always reported in the evidence table as
descriptive support, but it does not control which segmentation is
selected.**

**Plateau flag interpretation.** The `plateau_flag` column in the
evidence table is True when the marginal RSS gain at a candidate's k
falls below `MIN_DELTA_RSS_NORM`. It is informational. A plateau row can
still be selected if it is the most-stable candidate at the highest
admissible k. The flag flags the row for the researcher's attention,
not for automatic exclusion.

**Note on competing segmentation families.** Candidate segmentations from
different cost models are not guaranteed to form a nested hierarchy. The L2,
L1, and RBF cost functions may identify different boundaries at the same k,
and the knee regions of different models may not overlap. The
maximal-resolution criterion therefore operates across competing segmentation
families and should be interpreted as an operational heuristic rather than a
formal optimisation procedure. When competing families are present, the
evidence table reports which models support each candidate; the researcher
should inspect this column before accepting the automated suggestion. In such
cases, preference should generally be given to segmentations supported by
multiple cost models and containing highly reproducible boundaries, unless
finer-resolution candidates provide substantial additional explanatory gain.

---

## Step 7 — Zone summary and uncertainty quantification (`statistics.py`)

For the suggested segmentation, each zone receives:

- Thickness-weighted mean F14C and weighted SD.
- Bootstrap 95% CI on the weighted mean (10 000 resamples). Segments
  within a zone are resampled uniformly with replacement. Thickness
  weighting is applied only inside the estimator, not during resampling,
  to avoid double-weighting thick intervals.
- Propagated AMS measurement uncertainty: sigma_propagated =
  sqrt(sum((w_i * SD_i)^2)), where w_i are normalised thickness weights.
- Combined uncertainty in quadrature: sigma_combined =
  sqrt(sigma_propagated^2 + sigma_boot^2).

---

## Step 8 — Tolerance sensitivity analysis (`sensitivity.py`)

Steps 3 and 6 are re-run across a grid of knee tolerance values using the
CROPS results already computed in Step 2. No additional PELT runs are
required. The analysis reports whether the suggested k is stable across the
tolerance sweep. A stable result (same k at all tested tolerances) suggests
that the chosen tolerance is a robust operating point. Variation across the
sweep must be reported and discussed.

---

## Tests

The current version of the pipeline ships without an automated test
suite. End-to-end behaviour has been verified by running the full
pipeline on the four reference cores in `config.CORES` and checking
that the suggested segmentation, breakpoint depths, and zone summary
statistics are reproducible across runs at the configured permutation
seed.

A pytest-based test suite is planned for a future revision. The
intended coverage includes:

- `crops.py`: the closed-form crossover penalty, the `RuntimeError`
  contract of `resolve_beta_max`, and end-to-end CROPS on a synthetic
  piecewise-constant signal with known changepoints.
- `kneedle.py`: knee detection on a synthetic convex decreasing
  curve, the `region_at_tolerance` caching method, and degenerate
  inputs (fewer than three points, flat curves).
- `interpolation.py`: grid invariants (length, monotonicity, finite
  values), the large-gap warning, and the PCHIP-to-linear fallback.
- `evaluation.py`: `_cluster_breakpoints` single-linkage chaining,
  the snap-collision warning in `depths_to_segment_indices`, NaN
  propagation in `_nearest_stability`, and the two-step selection
  rule on a small hand-constructed evidence table.
- `statistics.py`: Holm-Bonferroni against a hand-computed example,
  the NaN return from `_weighted_rank_anova_statistic` on degenerate
  input, and the omnibus test on well-separated groups.

Until that suite is added, reviewers can reproduce the reference
output by setting `ACTIVE_CORE` in `config.py` and running `main()`.
The combination of `pipeline_version` and `active_core` columns in
every output CSV makes the run identifiable after the fact.

---

## Outputs

| File                            | Contents                                         |
| ------------------------------- | ------------------------------------------------ |
| `knee_region_combinations.csv`  | Evidence table for all candidate segmentations   |
| `zone_summary.csv`              | Thickness-weighted F14C statistics per zone      |
| `tolerance_sweep.csv`           | Suggested k as a function of knee tolerance      |
| `crops_knee.png`                | Kneedle plots for each cost model                |
| `downcore_profile.png`          | Downcore F14C profile with suggested breakpoints |
| `tolerance_sweep.png`           | Suggested k versus knee tolerance                |

---

## Assumptions and limitations

- F14C is treated as uniform within each measured segment. This is a
  piecewise-constant approximation that ignores within-segment variation.
- PELT operates on the interpolated 1-cm grid. Breakpoints may fall in
  interpolated gaps where no measurement exists. The `valid` mask and gap
  warnings flag this, but the researcher must interpret results accordingly.
- The omnibus test is post-selection and conditional. P-values do not
  confirm that changepoints exist at the detected locations.
- Candidate segmentations from different cost models are not guaranteed to
  be nested. The maximal-resolution criterion is an operational heuristic,
  not a formal optimisation.
- The suggested segmentation is an automated operating point. The researcher
  should evaluate it in the context of the core's sampling design, depth
  resolution, and scientific objectives.

**What this pipeline does not do:**

- Does not estimate posterior probabilities of changepoints.
- Does not produce confidence intervals for the number of changepoints k.
- Does not prove that detected zone boundaries correspond to real physical
  transitions.
- Does not replace geological or ecological interpretation by the researcher.
- Does not model autocorrelation within segments; observations within a
  segment are treated as exchangeable under the permutation null.
- Does not infer the temporal process or mechanism underlying the F14C
  depth profile.

---

## Glossary

| Term                   | Definition                                                                                   |
| ---------------------- | -------------------------------------------------------------------------------------------- |
| candidate region       | The set of k values retained after Kneedle detection, from which a suggestion is drawn.     |
| delta_rss_norm         | Marginal normalised RSS reduction per additional changepoint relative to the k=0 baseline.  |
| dominant knee          | The k with the highest difference-curve value in the Kneedle algorithm.                     |
| f(b)                   | Fraction of unique (model, tolerance) combinations in which boundary b appears.             |
| knee region            | Tolerance-defined band of k values around the dominant knee. A heuristic, not a CI.        |
| sigma_b                | Standard deviation of snapped boundary positions across combinations containing b (cm).      |
| snapped breakpoint     | A PELT grid-index breakpoint converted to depth and forward-snapped to a segment boundary.  |
| suggested segmentation | The automated operating point returned by the pipeline. Requires researcher evaluation.     |
