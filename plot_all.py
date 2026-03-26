#!/usr/bin/env python3
"""
AIConfigurator Plot
====================
Reads sweep CSV (from run_sweep.py), produces a multi-page PDF.

Each page = one (model, GPU) combo.
Each subplot = one workload (ISL, OSL).
X-axis = (TP, PP) configs.
Y-axis = peak throughput (best BS for that TP).

Usage:
    python plot_all.py --csv full_sweep_vllm.csv
    python plot_all.py --csv sweep_Qwen3-32B-FP8_h200_sxm.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── Style ────────────────────────────────────────────────────────────────
TXT, NOTE, BG, GRID = "#111111", "#777777", "#FAFAFA", "#E0E0E0"
BAR_COLORS = ["#2563EB", "#DC2626", "#16A34A", "#9333EA", "#EA580C", "#0891B2", "#BE185D", "#854D0E"]


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 14, "axes.titlesize": 16, "axes.titleweight": "bold",
        "axes.titlepad": 10, "axes.labelsize": 14, "axes.labelpad": 6,
        "axes.labelcolor": TXT, "axes.facecolor": BG, "axes.edgecolor": "#CCC",
        "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.5, "grid.alpha": 0.7,
        "legend.fontsize": 11, "legend.framealpha": 0.95,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "figure.facecolor": "#FFF", "figure.dpi": 150,
    })


def fmt_val(v):
    v = int(v)
    if v >= 1024 and v % 1024 == 0: return f"{v//1024}K"
    if v >= 1024: return f"{v/1024:.1f}K"
    return str(v)


def plot_one_page(df, model, system, metric="tokens/s/gpu"):
    """
    One page for (model, system).
    Grid of subplots: one per workload.
    Each subplot: bars for each (TP, PP), Y = peak metric across BS.
    """
    ms = model.split("/")[-1]
    gpu = system.upper().replace("_", " ")
    backend = df["backend"].iloc[0]
    gemm_q = df["gemm"].iloc[0] if "gemm" in df.columns else ""

    # Get workloads sorted by ISL then OSL
    wls = df.groupby(["isl", "osl"]).size().reset_index()[["isl", "osl"]]
    wls = wls.sort_values(["isl", "osl"]).reset_index(drop=True)
    n_wl = len(wls)

    # Grid layout
    if n_wl <= 4:
        nrows, ncols = 2, 2
    elif n_wl <= 6:
        nrows, ncols = 2, 3
    elif n_wl <= 9:
        nrows, ncols = 3, 3
    elif n_wl <= 12:
        nrows, ncols = 3, 4
    elif n_wl <= 16:
        nrows, ncols = 4, 4
    else:
        nrows, ncols = 5, 4

    fig = plt.figure(figsize=(ncols * 7, nrows * 5.5 + 2))
    fig.patch.set_facecolor("#FFF")
    fig.suptitle(f"{ms}  on  {gpu}", fontsize=24, fontweight="bold", color=TXT, y=0.98)
    fig.text(0.5, 0.965, f"{backend}  |  {gemm_q}  |  peak throughput (best BS) per TP",
             ha="center", fontsize=14, color=NOTE)

    gs = gridspec.GridSpec(nrows, ncols, hspace=0.45, wspace=0.35,
                           top=0.93, bottom=0.04, left=0.06, right=0.97)

    for idx in range(nrows * ncols):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])

        if idx >= n_wl:
            ax.set_visible(False)
            continue

        isl = int(wls.iloc[idx]["isl"])
        osl = int(wls.iloc[idx]["osl"])
        sub = df[(df["isl"] == isl) & (df["osl"] == osl)]

        # Get unique (TP, PP) combos
        configs = sub.groupby(["tp", "pp"]).size().reset_index()[["tp", "pp"]]
        configs = configs.sort_values(["tp", "pp"]).reset_index(drop=True)

        if configs.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", color=NOTE)
            ax.set_title(f"ISL={fmt_val(isl)}, OSL={fmt_val(osl)}")
            continue

        # For each (TP, PP), get peak throughput and the BS that achieved it
        labels = []
        peaks = []
        best_bs = []
        for _, cfg in configs.iterrows():
            tp, pp = int(cfg["tp"]), int(cfg["pp"])
            s = sub[(sub["tp"] == tp) & (sub["pp"] == pp)]
            peak_row = s.loc[s[metric].idxmax()]
            labels.append(f"TP{tp}" if pp == 1 else f"TP{tp}\nPP{pp}")
            peaks.append(peak_row[metric])
            best_bs.append(int(peak_row["bs"]))

        x = np.arange(len(labels))
        colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(labels))]
        bars = ax.bar(x, peaks, 0.6, color=colors, edgecolor="white", linewidth=0.8)

        # Annotate each bar with the best BS
        for bar, bs_val, peak_val in zip(bars, best_bs, peaks):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"bs={bs_val}", ha="center", va="bottom", fontsize=10, color=NOTE)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("tok/s/gpu")
        ax.set_title(f"ISL={fmt_val(isl)}, OSL={fmt_val(osl)}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot AIConfigurator sweep results")
    parser.add_argument("--csv", required=True, help="Sweep CSV from run_sweep.py")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"Error: {args.csv} not found")

    df = pd.read_csv(csv_path)
    output = str(csv_path.with_suffix(".pdf"))
    print(f"Loaded: {args.csv}  ({len(df)} rows)")

    combos = df.groupby(["model", "system"]).size().reset_index()[["model", "system"]]
    combos = [(r["model"], r["system"]) for _, r in combos.iterrows()]
    print(f"  {len(combos)} (model, system) combos")

    setup_style()
    n_pages = 0

    with PdfPages(output) as pdf:
        for i, (model, system) in enumerate(combos):
            ms = model.split("/")[-1]
            gpu = system.upper().replace("_", " ")
            sub = df[(df["model"] == model) & (df["system"] == system)]
            n_wl = sub.groupby(["isl", "osl"]).ngroups
            print(f"  [{i+1}/{len(combos)}] {ms} / {gpu}  ({len(sub)} rows, {n_wl} workloads)")

            fig = plot_one_page(sub, model, system)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            n_pages += 1

    print(f"\nSaved: {output} ({n_pages} pages)")


if __name__ == "__main__":
    main()
