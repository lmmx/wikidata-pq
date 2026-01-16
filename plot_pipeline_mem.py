#!/usr/bin/env python3
"""
Pipeline Memory Stats Plotter
Produces a clean, publication-quality visualization of memory usage over time.
"""

from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PARQUET_PATH = "pipeline_memory.parquet"

# Color palette - muted, professional
COLORS = {
    "rss": "#e63946",  # warm red
    "vms": "#457b9d",  # steel blue
    "sys_used": "#2a9d8f",  # teal
    "sys_avail": "#e9c46a",  # muted gold
    "grid": "#e0e0e0",
    "spine": "#cccccc",
    "text": "#333333",
    "background": "#fafafa",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Convert unix timestamp to datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    # Calculate elapsed time in minutes from start
    df["elapsed_min"] = (df["timestamp"] - df["timestamp"].iloc[0]) / 60
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────


def setup_style():
    """Apply custom styling - clean, modern, readable."""
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["background"],
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": COLORS["spine"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "legend.framealpha": 0.95,
            "legend.edgecolor": COLORS["spine"],
        }
    )


def plot_memory_stats(df: pd.DataFrame, save_path: str = None):
    setup_style()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.08},
    )

    x = df["elapsed_min"]

    # ─── Top Panel: Process Memory (RSS & VMS) ───────────────────────────────
    ax1 = axes[0]

    # VMS as filled area (it's always >= RSS)
    ax1.fill_between(x, 0, df["vms_gb"], alpha=0.25, color=COLORS["vms"], linewidth=0)
    ax1.plot(x, df["vms_gb"], color=COLORS["vms"], linewidth=2, label="VMS (Virtual)")

    # RSS as filled area on top
    ax1.fill_between(x, 0, df["rss_gb"], alpha=0.4, color=COLORS["rss"], linewidth=0)
    ax1.plot(x, df["rss_gb"], color=COLORS["rss"], linewidth=2, label="RSS (Resident)")

    ax1.set_ylabel("Process Memory (GB)")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, axis="y", linestyle="-", alpha=0.7)
    ax1.set_title("Pipeline Memory Profile", fontweight="medium", pad=12)

    # Add peak annotations
    rss_peak_idx = df["rss_gb"].idxmax()
    vms_peak_idx = df["vms_gb"].idxmax()

    ax1.annotate(
        f'RSS peak: {df.loc[rss_peak_idx, "rss_gb"]:.1f} GB',
        xy=(x.loc[rss_peak_idx], df.loc[rss_peak_idx, "rss_gb"]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
        color=COLORS["rss"],
        arrowprops=dict(arrowstyle="->", color=COLORS["rss"], lw=0.8),
    )

    # ─── Bottom Panel: System Memory ─────────────────────────────────────────
    ax2 = axes[1]

    # Stacked area: used + available = total
    ax2.fill_between(
        x,
        0,
        df["sys_used_gb"],
        alpha=0.5,
        color=COLORS["sys_used"],
        linewidth=0,
        label="System Used",
    )
    ax2.fill_between(
        x,
        df["sys_used_gb"],
        df["sys_used_gb"] + df["sys_avail_gb"],
        alpha=0.35,
        color=COLORS["sys_avail"],
        linewidth=0,
        label="System Available",
    )

    # Edge lines for clarity
    ax2.plot(x, df["sys_used_gb"], color=COLORS["sys_used"], linewidth=1.5)
    total = df["sys_used_gb"] + df["sys_avail_gb"]
    ax2.plot(
        x, total, color=COLORS["sys_avail"], linewidth=1, linestyle="--", alpha=0.7
    )

    # Secondary y-axis for percentage
    ax2_pct = ax2.twinx()
    ax2_pct.plot(
        x,
        df["sys_percent"],
        color=COLORS["text"],
        linewidth=1.2,
        linestyle=":",
        alpha=0.6,
        label="Usage %",
    )
    ax2_pct.set_ylabel("System Usage %", color=COLORS["text"], alpha=0.7)
    ax2_pct.set_ylim(0, 100)
    ax2_pct.tick_params(axis="y", colors=COLORS["text"], labelsize=8)
    ax2_pct.spines["right"].set_color(COLORS["spine"])

    ax2.set_ylabel("System Memory (GB)")
    ax2.set_xlabel("Elapsed Time (minutes)")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, axis="y", linestyle="-", alpha=0.7)

    # ─── Final Touches ───────────────────────────────────────────────────────
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.02)

    ax2_pct.spines["top"].set_visible(False)

    # Timestamp footer
    start_time = df["datetime"].iloc[0].strftime("%Y-%m-%d %H:%M:%S")
    end_time = df["datetime"].iloc[-1].strftime("%H:%M:%S")
    duration = df["elapsed_min"].iloc[-1]

    fig.text(
        0.99,
        0.01,
        f"Run: {start_time} → {end_time} ({duration:.1f} min)  •  {len(df)} samples",
        ha="right",
        va="bottom",
        fontsize=8,
        color=COLORS["text"],
        alpha=0.5,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["background"]
        )
        print(f"Saved to {save_path}")

    plt.savefig("pipeline_mem_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot pipeline memory statistics")
    parser.add_argument(
        "input", nargs="?", default=PARQUET_PATH, help="Path to parquet file"
    )
    parser.add_argument("-o", "--output", help="Save plot to file (png/pdf/svg)")
    args = parser.parse_args()

    df = load_data(args.input)

    # Quick stats to stdout
    print(f"{'─'*60}")
    print(f"Memory Stats Summary")
    print(f"{'─'*60}")
    print(f"Duration:     {df['elapsed_min'].iloc[-1]:.1f} minutes")
    print(
        f"RSS:          {df['rss_gb'].min():.2f} → {df['rss_gb'].max():.2f} GB (peak)"
    )
    print(
        f"VMS:          {df['vms_gb'].min():.2f} → {df['vms_gb'].max():.2f} GB (peak)"
    )
    print(f"System peak:  {df['sys_percent'].max():.1f}%")
    print(f"{'─'*60}\n")

    plot_memory_stats(df, save_path=args.output)
