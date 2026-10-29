"""Fig. S1: steady-state null model vs observed SOC-F14C profiles.

Top row: observed increments (black bars), best-fit null profiles on the 1-cm
grid (N0 smooth, N1 modern active-layer floor, N2 free floor), field z_AL
(orange line) and the published change points (dashed).
Bottom row: how often the pipeline, run on synthetic profiles drawn from the
N0 null with "fit" noise, places a boundary at each depth (5-cm bins);
black arrows mark boundaries on the AMS-noise-only null.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import forward as fw
from cores import CORES

HERE = Path(__file__).resolve().parent
ORDER = ["Toolik", "HCCN", "HC03", "HC67"]
TITLES = {"Toolik": "Toolik SF_C_1", "HCCN": "HCCN (no recent fire)",
          "HC03": "HC03 (fire in 2003)", "HC67": "HC67 (fire in 1967)"}
COL = {"N0": "#7a7a7a", "N1": "#1f5aa6", "N2": "#d4691e"}
LAB = {"N0": "N0 smooth steady state",
       "N1": "N1 + mixing floor at field $z_{AL}$",
       "N2": "N2 + free mixing floor (diagnostic)"}
LS = {"N0": (0, (4, 2)), "N1": "-", "N2": "-"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
})


def main(sim_variant="N0", sim_noise="fit", out="results/FigS1_null_model"):
    """Draw fig. S1 and save it as PNG and PDF under ``out``."""
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 6.2), sharey=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    for j, c in enumerate(ORDER):
        co = CORES[c]
        ax = axes[0, j]
        for t, b, f in zip(co["tops"], co["bottoms"], co["fm"]):
            ax.plot([f, f], [t, b], color="k", lw=2.2,
                    solid_capstyle="butt", zorder=4)
        for v in ["N0", "N1", "N2"]:
            fp = HERE / f"fits/{c}_{v}.json"
            if not fp.exists():
                continue
            r = json.loads(fp.read_text())
            F = np.array(r["grid_F"])
            m = fw.Z <= co["bottoms"].max() + 2
            ax.plot(F[m], fw.Z[m], color=COL[v], ls=LS[v], lw=1.1, zorder=3,
                    label=LAB[v] if j == 0 else None)
            if v == "N2":
                zf = r["params"]["zf"]
                ax.plot([1.42], [zf], marker="<", color=COL["N2"], ms=5,
                        clip_on=False)
        ax.axhline(co["z_al"], color="#e0a000", lw=1.2, zorder=2,
                   label="field $z_{AL}$" if j == 0 else None)
        for b in co["cps"]:
            first = j == 0 and b == co["cps"][0]
            ax.axhline(b, color="k", lw=0.7, ls=(0, (2, 2)), zorder=1,
                       label="change point (Table 1)" if first else None)
        ax.axvline(1.0, color="0.75", lw=0.5, zorder=0)
        ax.set_xlim(0, 1.45)
        ax.set_title(TITLES[c], fontsize=8, fontweight="bold")
        ax.set_xlabel("SOC-F$^{14}$C")
        ax.text(0.04, 0.02, "ABCD"[j], transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom")
        # Bottom row: detection frequency under the null.
        axb = axes[1, j]
        sp = HERE / f"sims/{c}_{sim_variant}_{sim_noise}.json"
        if sp.exists():
            d = json.loads(sp.read_text())
            sims = [s for s in d["sims"] if "bkps" in s]
            allb = (np.concatenate([s["bkps"] for s in sims]) if sims
                    else np.array([]))
            bins = np.arange(0, 205, 5)
            h, _ = np.histogram(allb, bins=bins)
            axb.barh(bins[:-1] + 2.5, h / max(len(sims), 1), height=4.2,
                     color=COL[sim_variant], edgecolor="none")
            axb.text(0.97, 0.03, f"n = {len(sims)}", transform=axb.transAxes,
                     ha="right", va="bottom", fontsize=7, color="0.3")
        ap = HERE / f"sims/{c}_{sim_variant}_ams.json"
        if ap.exists():
            ab = json.loads(ap.read_text())["sims"][0].get("bkps", [])
            axb.plot([0.97] * len(ab), ab, ls="none", marker="<", color="k",
                     ms=4,
                     label=("boundary on null, AMS noise only" if j == 0
                            else None))
        for b in co["cps"]:
            axb.axhline(b, color="k", lw=0.7, ls=(0, (2, 2)))
        axb.axhline(co["z_al"], color="#e0a000", lw=1.2)
        axb.set_xlim(0, 1.0)
        axb.set_xlabel("fraction of null runs\nwith a boundary here")
        axb.text(0.04, 0.02, "EFGH"[j], transform=axb.transAxes, fontsize=10,
                 fontweight="bold", va="bottom")
    for ax in axes[:, 0]:
        ax.set_ylabel("Soil depth (cm)")
    axes[0, 0].set_ylim(200, 0)
    axes[1, 0].set_ylim(200, 0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles2, labels2 = axes[1, 0].get_legend_handles_labels()
    handles, labels = handles + handles2, labels + labels2
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    (HERE / "results").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{out}.{ext}", dpi=600)
    print("saved", out)


if __name__ == "__main__":
    main()
