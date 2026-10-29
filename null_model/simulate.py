"""Run the change-point pipeline on synthetic null-model profiles.

Synthetic profiles are drawn from the fitted steady-state null model and
passed through the unchanged ``changepoint_analysis`` pipeline.

Usage: python simulate.py CORE VARIANT NOISE N_REAL

NOISE is one of

- ``ams``: AMS measurement uncertainty only.
- ``fit``: sigma equals the thickness-weighted RMS residual of the null
  fit. This is generous to the null, since all misfit counts as noise.
- ``zone``: sigma equals the pooled within-zone SD of the observed
  profile under the published segmentation. This is closer to the true
  sample scatter.

Writes sims/CORE_VARIANT_NOISE.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from cores import CORES

HERE = Path(__file__).resolve().parent
TOL = 5.0  # cm; same as BREAKPOINT_SNAP_TOL_CM in the pipeline


def seg_means(fm, w, tops, bottoms, bkps):
    """Return thickness-weighted zone means and zone labels per increment.

    Each increment is assigned to a zone by its midpoint depth relative
    to the breakpoint depths ``bkps``.
    """
    edges = [-np.inf] + list(bkps) + [np.inf]
    mid = 0.5 * (tops + bottoms)
    lab = np.digitize(mid, edges[1:-1])
    out = np.empty_like(fm)
    for z in np.unique(lab):
        m = lab == z
        out[m] = np.sum(w[m] * fm[m]) / np.sum(w[m])
    return out, lab


def run_pipeline(tops, bottoms, fm, sd):
    """Run the pipeline once on a synthetic profile in a subprocess.

    The profile is written to a temporary JSON file and read by the
    pipeline through the ``CPA_*`` hook in ``config.py``. Returns the
    parsed result, or ``{"error": message}`` if the run fails.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".json",
                                     delete=False) as fh:
        json.dump({"tops": tops.tolist(), "bottoms": bottoms.tolist(),
                   "fm": fm.tolist(), "sd": sd.tolist()}, fh)
        path = fh.name
    env = dict(os.environ, CPA_CORE_JSON=path, CPA_ACTIVE="CUSTOM",
               CPA_NPERM="50")
    try:
        r = subprocess.run([sys.executable, str(HERE / "run_one.py")],
                           env=env, cwd=HERE, capture_output=True,
                           text=True, timeout=300)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    finally:
        os.unlink(path)


def main(core_name, variant, noise, n_real, seed=7):
    """Simulate ``n_real`` null realizations and save the pipeline results."""
    core = CORES[core_name]
    fitres = json.loads(
        (HERE / f"fits/{core_name}_{variant}.json").read_text())
    fm_mod = np.array(fitres["fm_model"])
    tops, bottoms = core["tops"], core["bottoms"]
    fm_obs, sd = core["fm"], core["sd"]
    w = bottoms - tops
    if noise == "ams":
        sigma = 0.0
    elif noise == "fit":
        sigma = float(np.sqrt(np.sum(w * (fm_obs - fm_mod) ** 2)
                              / np.sum(w)))
    else:
        mu, lab = seg_means(fm_obs, w, tops, bottoms, core["cps"])
        sigma = float(np.sqrt(np.sum(w * (fm_obs - mu) ** 2) / np.sum(w)))
    rng = np.random.default_rng(seed)
    sims = []
    for i in range(n_real):
        fm = fm_mod + rng.normal(0, sigma, fm_mod.size) + rng.normal(0, sd)
        fm = np.clip(fm, 0.005, None)
        res = run_pipeline(tops, bottoms, fm, sd)
        if "bkps" in res:
            mu, _ = seg_means(fm, w, tops, bottoms, res["bkps"])
            res["wrss_seg"] = float(np.sum(w * (fm - mu) ** 2))
            res["wrss_null"] = float(np.sum(w * (fm - fm_mod) ** 2))
        res["fm"] = fm.tolist()
        sims.append(res)
        print(i, res.get("bkps"), flush=True)
    out = {"core": core_name, "variant": variant, "noise": noise,
           "sigma": sigma, "observed_cps": list(core["cps"]), "sims": sims}
    Path(HERE / "sims").mkdir(exist_ok=True)
    (HERE / f"sims/{core_name}_{variant}_{noise}.json").write_text(
        json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
