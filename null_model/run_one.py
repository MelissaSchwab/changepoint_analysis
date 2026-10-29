"""Run the changepoint_analysis pipeline once and print one JSON line.

The core is set by the ``CPA_CORE_JSON`` and ``CPA_ACTIVE`` environment
variables (see the hook in ``changepoint_analysis/config.py``). Called
by ``simulate.py``.
"""
import json
import sys
import time
import warnings
from pathlib import Path

# Make the repository root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from changepoint_analysis.main import run_pipeline  # noqa: E402

t = time.time()
r = run_pipeline()
print(json.dumps({"k": r.selected_k,
                  "bkps": [float(b) for b in r.best_depth_bkps],
                  "mode": r.selection_mode,
                  "sec": round(time.time() - t, 1)}))
