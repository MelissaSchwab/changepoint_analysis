#!/usr/bin/env bash
# Regenerate all null-model fits, simulations, tables and fig. S1.
# This takes several hours.
set -euo pipefail
cd "$(dirname "$0")"
for c in Toolik HCCN HC03 HC67; do
  # N1 and N2 warm-start from the simpler fit.
  for v in N0 N1 N2; do python fit.py "$c" "$v" 12; done
done
for c in Toolik HCCN HC03 HC67; do
  python simulate.py "$c" N0 ams 5
  python simulate.py "$c" N0 fit 60
  python simulate.py "$c" N1 ams 5
  python simulate.py "$c" N1 fit 80
  python simulate.py "$c" N1 zone 60
done
python analyze.py
python plot_null.py
