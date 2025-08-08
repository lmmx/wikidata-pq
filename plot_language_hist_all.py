import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

# Set seaborn style without grid
sns.set_style("white")
plt.rcParams["font.size"] = 12

df = pl.read_parquet("results/claims")
lang_counts = (
    df["language"]
    .value_counts(sort=True, normalize=True)
    .with_columns((pl.col("proportion") * 100).alias("percentage"))
)

fig, ax = plt.subplots(figsize=(18, 10))

# Use a nice seaborn color palette
colors = sns.color_palette("husl", len(lang_counts))

bars = ax.bar(
    range(len(lang_counts)),
    lang_counts["percentage"],
    color=colors,
    edgecolor="white",
    linewidth=0.3,
    alpha=0.9,
)

# Set log scale for y-axis
ax.set_yscale("log")

# Clean styling - only show every 10th language on x-axis to avoid clutter
tick_indices = range(0, len(lang_counts), 10)
ax.set_xticks(tick_indices)
ax.set_xticklabels(
    [lang_counts["language"][i] for i in tick_indices],
    rotation=45,
    ha="right",
    fontsize=10,
    fontweight="bold",
)

ax.set_xlabel(
    "Language Code (every 10th shown)", fontsize=16, fontweight="bold", labelpad=15
)
ax.set_ylabel("Percentage of Claims (%)", fontsize=16, fontweight="bold", labelpad=15)
ax.set_title(
    f"Language Distribution in Wikidata Claims\nAll {len(lang_counts)} Languages (Log Scale)",
    fontsize=20,
    fontweight="bold",
    pad=30,
)

# Format y-axis to show percentages
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}%"))

# Remove all spines and grid for ultra-clean look
sns.despine(left=True, bottom=True, top=True, right=True)
ax.grid(False)

plt.tight_layout()
plt.savefig("language_hist_all.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
