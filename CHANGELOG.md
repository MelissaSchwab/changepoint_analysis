# Changelog

## 0.2.0

- Added `null_model/`: steady-state radiocarbon null model, fitted to the four
  cores, with scripts that run the pipeline on synthetic profiles (fig. S1).
- `config.py`: added an optional hook (`CPA_CORE_JSON`, `CPA_ACTIVE`,
  `CPA_NPERM`) so the pipeline can read synthetic profiles. When the variables
  are unset, behaviour is identical to 0.1.0, and the pipeline still reproduces
  Table 1 for all four cores.
- `PIPELINE_VERSION` set to 0.2.0. Detection and selection algorithms are unchanged.
- Code style brought into line with PEP 8 and PEP 257 across
  `changepoint_analysis/` and `null_model/` (line length, imports, docstrings,
  variable names). No change in behaviour. Pipeline outputs for all four cores,
  the archived null-model tables and fig. S1 are identical before and after.

## 0.1.0

- Initial release.
