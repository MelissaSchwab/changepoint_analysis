"""Multi-model changepoint detection for downcore radiocarbon (F14C) profiles.

This package detects zone boundaries in sediment or soil cores by finding
changepoints in the F14C (fraction modern) depth profile. All statistical
summaries and post-selection analyses operate on the original measured
segment values with segment thickness as weights. The 1-cm interpolated
grid is used only for PELT cost computation and visualization.

Capabilities
------------
Exploratory analysis
- 1-cm depth grid construction with gap interpolation
- CROPS/PELT changepoint detection across a continuous penalty range
- Kneedle knee detection on cost-versus-k curves
- Breakpoint stability scoring across models and tolerances

Segmentation evaluation
- Automated segmentation suggestion (cross-model consensus or maximal stable resolution)
- Zone F14C statistics with bootstrap uncertainty quantification
- Knee tolerance sensitivity analysis

Quick start
-----------
    from changepoint_analysis.main import main

    result = main()

    print(
        f"Suggested segmentation: k={result.selected_k}, "
        f"breakpoints={result.best_depth_bkps}"
    )
    print(result.zone_df[[
        "zone", "depth_top_cm", "depth_bottom_cm",
        "weighted_mean_fm", "ci95_lower", "ci95_upper",
    ]])

``main()`` writes CSV and figure outputs (paths controlled by
``config.OUTPUT_PATHS``) and returns a ``PipelineResult`` dataclass
with named attributes. See ``main.PipelineResult`` for the field list.

Programmatic use without console side effects
---------------------------------------------
``main()`` prints a progress log and writes CSV files. For library
callers that want neither, use ``run_pipeline()`` instead. It returns
the same ``PipelineResult`` without printing or writing.

    from changepoint_analysis.main import run_pipeline
    result = run_pipeline()

Configuration
-------------
All tunable parameters and core data live in ``config.py``. Set
``ACTIVE_CORE`` to one of the available core keys; no other file needs
to change. Output file paths are controlled by ``config.OUTPUT_PATHS``.

Modules
-------
config.py         Tunable parameters and read-only core data arrays.
interpolation.py  1-cm depth grid construction with gap interpolation.
crops.py          PELT changepoint detection and CROPS penalty sweep.
kneedle.py        Knee detection on cost-versus-k curves.
evaluation.py     Breakpoint snapping, stability scoring, evidence table,
                  and automated segmentation suggestion.
statistics.py     Zone summary statistics and uncertainty quantification.
sensitivity.py    Knee tolerance sensitivity analysis.
plotting.py       Diagnostic and publication figures.
main.py           Pipeline orchestration.

See PIPELINE.md for a full description of the statistical pipeline,
design decisions, assumptions, and limitations.

Data
----
International Soil Radiocarbon Database (ISRaD):
https://www.soilradiocarbon.org/

References
----------
Haynes, K., Eckley, I. A., & Fearnhead, P. (2017). Computationally efficient
    changepoint detection for a range of penalties. Journal of Computational
    and Graphical Statistics, 26(1), 134-143.
    https://doi.org/10.1080/10618600.2015.1116445
Haynes, K., Fearnhead, P., & Eckley, I. A. (2017). A computationally efficient
    nonparametric approach for changepoint detection. Statistics and Computing,
    27(5), 1293-1305. https://doi.org/10.1007/s11222-016-9687-5
Holm, S. (1979). A simple sequentially rejective multiple test procedure.
    Scandinavian Journal of Statistics, 6(2), 65-70.
Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of
    changepoints with a linear computational cost. Journal of the American
    Statistical Association, 107(500), 1590-1598.
    https://doi.org/10.1080/01621459.2012.737745
Lavielle, M. (2005). Using penalized contrasts for the change-point problem.
    Signal Processing, 85(8), 1501-1510.
    https://doi.org/10.1016/j.sigpro.2005.01.012
Satopaa, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011). Finding a
    kneedle in a haystack: Detecting knee points in system behavior.
    Proceedings - International Conference on Distributed Computing Systems,
    166-171. https://doi.org/10.1109/ICDCSW.2011.20
Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline
    changepoint detection methods. Signal Processing, 167, 107299.
    https://doi.org/10.1016/j.sigpro.2019.107299
"""
