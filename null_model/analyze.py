"""Summarise null-model fits and pipeline runs on null realizations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cores import CORES
from simulate import seg_means

HERE = Path(__file__).resolve().parent
TOL = 5.0
ORDER = ["Toolik", "HCCN", "HC03", "HC67"]


def fit_table():
    """Compare the weighted RSS of each null fit with the segmentation."""
    rows = []
    for c in ORDER:
        co = CORES[c]
        w = co["bottoms"] - co["tops"]
        mu, _ = seg_means(co["fm"], w, co["tops"], co["bottoms"], co["cps"])
        seg = float(np.sum(w * (co["fm"] - mu) ** 2))
        row = {"core": c, "n_increments": len(w), "z_AL": co["z_al"],
               "change_points": ", ".join(str(int(b)) for b in co["cps"]),
               "wRSS_segmentation": seg,
               "n_par_segmentation": 2 * len(co["cps"]) + 1}
        for v in ["N0", "N1", "N2"]:
            f = HERE / f"fits/{c}_{v}.json"
            if f.exists():
                r = json.loads(f.read_text())
                row[f"wRSS_{v}"] = r["wrss"]
                row[f"n_par_{v}"] = r["n_params"]
                if v == "N2":
                    row["N2_floor_cm"] = r["params"]["zf"]
        rows.append(row)
    return pd.DataFrame(rows)


def sim_table():
    """Summarise boundary detection on each set of null realizations.

    ``T`` is the ratio of null-fit wRSS to segmentation wRSS. ``p_T`` is
    the fraction of realizations with ``T`` at least as large as the
    observed value, with the +1/+1 correction.
    """
    rows = []
    for f in sorted((HERE / "sims").glob("*.json")):
        d = json.loads(f.read_text())
        c = d["core"]
        co = CORES[c]
        sims = [s for s in d["sims"] if "bkps" in s]
        obs = co["cps"]
        w = co["bottoms"] - co["tops"]
        mu, _ = seg_means(co["fm"], w, co["tops"], co["bottoms"], obs)
        seg_obs = float(np.sum(w * (co["fm"] - mu) ** 2))
        fit = json.loads(
            (HERE / f"fits/{c}_{d['variant']}.json").read_text())
        T_obs = fit["wrss"] / seg_obs
        T_sim = np.array([s["wrss_null"] / s["wrss_seg"] for s in sims])
        p_T = float((np.sum(T_sim >= T_obs) + 1) / (len(T_sim) + 1))
        row = {"core": c, "null": d["variant"], "noise": d["noise"],
               "sigma": d["sigma"], "n_real": len(sims),
               "k_null_median": float(np.median([s["k"] for s in sims])),
               "T_obs": T_obs, "p_T": p_T}
        deep_hits = []
        for b in obs:
            hit = np.array([any(abs(x - b) <= TOL for x in s["bkps"])
                            for s in sims])
            row[f"P(detect {int(b)} cm)"] = float(hit.mean())
            if b > co["z_al"]:
                deep_hits.append(hit)
        if deep_hits:
            row["P(all boundaries below z_AL)"] = float(
                np.mean(np.all(deep_hits, axis=0)))
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    ft = fit_table()
    print(ft.round(4).to_string(index=False))
    (HERE / "results").mkdir(exist_ok=True)
    ft.to_csv(HERE / "results/null_fit_table.csv", index=False)
    if (HERE / "sims").exists():
        st = sim_table()
        print(st.round(3).to_string(index=False))
        st.to_csv(HERE / "results/null_sim_table.csv", index=False)
