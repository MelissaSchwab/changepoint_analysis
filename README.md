# changepoint_analysis

Multi-model changepoint detection for downcore radiocarbon (F<sup>14</sup>C)
profiles in sediment and soil cores. The pipeline identifies stratigraphic
zone boundaries by combining penalised changepoint detection
(PELT / CROPS) with knee-point selection (Kneedle) across multiple cost
models, then quantifies boundary reproducibility across analytical
settings.

## What the pipeline does

Given a core's depth intervals, F<sup>14</sup>C values, and AMS
uncertainties, the pipeline:

1. Builds a 1-cm depth grid with linear or PCHIP interpolation across
   unsampled gaps.
2. Runs CROPS over a continuous penalty range for three cost models
   (L2 mean-shift, L1 median-shift, RBF kernel).
3. Detects the knee of each model's cost-vs-k curve with Kneedle.
4. Scores every candidate segmentation in the knee-region union by
   stability across (model, tolerance) combinations, marginal RSS gain,
   and a thickness-weighted permutation rank ANOVA.
5. Suggests a segmentation using a two-step rule: cross-model consensus
   first, then maximum-resolution selection gated on boundary
   reproducibility.
6. Produces zone summary statistics with bootstrap 95% CIs, an evidence
   table, a tolerance sensitivity sweep, and three diagnostic plots.

The selection rule, statistical assumptions, and design rationale are
documented in [PIPELINE.md](PIPELINE.md).

## Installation

The package requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

Optional: install in a virtual environment first.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```python
from changepoint_analysis.main import main

result = main()

print(f"Suggested segmentation: k={result.selected_k}")
print(f"Breakpoints (cm): {result.best_depth_bkps}")
print(result.zone_df)
```

`main()` prints a progress log, writes three CSV outputs (zone summary,
evidence table, tolerance sweep), and returns a `PipelineResult`
dataclass with every artefact needed for downstream plotting or
analysis.

For library use without console output or CSV writing, call
`run_pipeline()` instead of `main()`.

## Configuration

All tunable parameters live in `config.py`. The most common ones to
change are:

- `ACTIVE_CORE`: selects which core in the `CORES` catalogue to analyse.
- `INTERPOLATION_METHOD`: `"linear"` or `"pchip"`.
- `ELBOW_TOLERANCE`: knee region width (default 0.20).
- `STABILITY_THRESHOLD`: f(b) cutoff for the HIGH stability tier
  (default 0.60).
- `N_PERMUTATIONS`: permutations for the omnibus test (default 5000).
- `PERMUTATION_SEED`: fixed for reproducibility (default 42).

To add a new core, append a `CoreData` entry to the `CORES` dictionary
in `config.py` and set `ACTIVE_CORE` to its key.

## Outputs

Running `main()` writes the following files to the working directory:

- `zone_summary.csv` — thickness-weighted zone statistics with bootstrap CIs.
- `knee_region_combinations.csv` — every candidate segmentation evaluated.
- `tolerance_sweep.csv` — suggested k as a function of knee tolerance.

Running `python -m changepoint_analysis.main` additionally produces:

- `crops_knee.png` — per-model Kneedle plot.
- `downcore_profile.png` — F<sup>14</sup>C profile with breakpoints.
- `tolerance_sweep.png` — suggested k versus knee tolerance.

Output filenames are configurable via `config.OUTPUT_PATHS`.

## Algorithms and references

| Step | Algorithm | Reference |
|---|---|---|
| Changepoint detection | PELT | Killick, Fearnhead, Eckley (2012) *JASA* |
| Penalty sweep | CROPS | Haynes, Eckley, Fearnhead (2017) *JCGS* |
| Knee detection | Kneedle | Satopää et al. (2011) *ICDCS Workshops* |
| Multiple comparison | Holm step-down | Holm (1979) *Scand J Stat* |

## Reproducibility

Every output CSV is stamped with the pipeline version
(`config.PIPELINE_VERSION`) and the active core. The permutation seed
is fixed at import time. Running the pipeline on the same input data
with the same configuration will produce bit-identical outputs.

## License

MIT. See [LICENSE](LICENSE).

## Citing

If you use this code in published work, please cite both the original
methods papers above and this repository. Once the accompanying
manuscript is published, please cite the paper as well. See
[CITATION.cff](CITATION.cff) for a machine-readable citation block.
