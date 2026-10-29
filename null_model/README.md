# Steady-state radiocarbon null model

Tests whether the change points found by `changepoint_analysis` could arise in a
soil with no thaw history. Used for fig. S1 of Czimczik, Schwab, Pedron & Welker,
"Are we digging deep enough to get permafrost carbon?" (submitted to *Science Advances*).

The change-point pipeline itself is unchanged from v0.1.0. This folder adds a
forward model that generates synthetic F<sup>14</sup>C profiles, and scripts that
run the pipeline on them.

## Model

One-dimensional advection–diffusion–decay model of soil organic carbon (SOC) and
its <sup>14</sup>C content (after Elzein & Balesdent 1995):

    dC/dt  = d/dz(D dC/dz) - v dC/dz - k(z) C + I(z)
    dC*/dt = d/dz(D dC*/dz) - v dC*/dz - (k(z) + λ) C* + I(z) Fatm(t)

Carbon stocks are at steady state. <sup>14</sup>C is integrated from 48 ka to the
sampling year (Toolik 2019, Hess Creek 2007) under IntCal20 (Reimer et al. 2020)
and the Northern Hemisphere zone 1 post-bomb curve (Hua et al. 2022). All
parameters are constant in time, so the model contains no thermal history.

| variant | description | free parameters |
|---|---|---|
| N0 | mixing D and decomposition k decline exponentially with depth | 8 |
| N1 | N0 plus a step reduction in D and k below the field active-layer depth z_AL | 10 |
| N2 | N1 with the depth of that step fitted (diagnostic, not a null) | 11 |

Each variant is fitted to each core by multistart bounded least squares, with
residuals weighted by increment thickness.

Each fitted profile is then sampled at the measured increments, scatter is added,
and the result is run through the unchanged pipeline:

- `ams`: AMS uncertainty only (5 realizations);
- `fit`: Gaussian scatter equal to the thickness-weighted RMS misfit of the null fit;
- `zone`: scatter equal to the within-zone SD under the Table 1 segmentation.

## Files

| file | purpose |
|---|---|
| `forward.py` | forward model; reads `curves/` |
| `cores.py` | observed profiles (copied from `config.py`), field z_AL, Table 1 change points, sampling years |
| `fit.py` | fits one core and variant; writes `fits/CORE_VARIANT.json` |
| `simulate.py` | runs the pipeline on synthetic profiles; writes `sims/CORE_VARIANT_NOISE.json` |
| `run_one.py` | single pipeline call, used by `simulate.py` |
| `analyze.py` | summary tables in `results/` |
| `plot_null.py` | fig. S1, written to `results/FigS1_null_model.png` and `.pdf` |
| `curves/` | `3Col_intcal20.14C` (IntCal20) and `postbomb_NH1.14C` (NH1 bomb curve) |
| `fits/`, `sims/` | archived outputs used in the paper |

The pipeline reads synthetic profiles through an optional hook in
`changepoint_analysis/config.py`, controlled by three environment variables:

- `CPA_CORE_JSON`: path to a JSON file with `tops`, `bottoms`, `fm` and `sd`;
- `CPA_ACTIVE`: the core to analyse;
- `CPA_NPERM`: number of permutations.

When these variables are not set, the pipeline behaves exactly as in v0.1.0.

## Reproduce

From the repository root, after `pip install -r requirements.txt`:

```bash
cd null_model
python analyze.py      # tables from the archived fits/ and sims/
python plot_null.py    # fig. S1
```

`run_all.sh` regenerates the fits and simulations from scratch. This takes
several hours on one core, and the simulations run the full pipeline thousands
of times.

The fits in the paper were refined by restarting the optimizer from earlier
solutions (`fit.py` always restarts from an existing fit and from
`fits/Toolik_N1_seed.json`). A clean rerun should reach the same or a slightly
different local optimum; the archived `fits/` are the ones reported.

## Caveats

- The nulls have 8 to 11 free parameters for 11 to 21 increments. Some
  parameters sit at their bounds, so the fits are not calibrated turnover
  estimates.
- Atmospheric <sup>14</sup>C after mid-2019 is extrapolated.
- Noise is independent Gaussian per increment.
- p values come from 60 to 80 realizations and are approximate.

## References

- Elzein, A. & Balesdent, J. (1995) *Soil Sci. Soc. Am. J.* 59, 1328–1335.
- Hua, Q. et al. (2022) *Radiocarbon* 64, 723–745.
- Reimer, P. J. et al. (2020) *Radiocarbon* 62, 725–757.
