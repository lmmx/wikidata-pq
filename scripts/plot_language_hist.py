import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

# Set seaborn style without grid
sns.set_style("white")
plt.rcParams["font.size"] = 12

df = pl.read_parquet("results/claims")
n_langs = 50
lang_counts = df["language"].value_counts(sort=True)

fig, ax = plt.subplots(figsize=(18, 10))

# Use a nice seaborn color palette
colors = sns.color_palette("husl", n_langs)

bars = ax.bar(
    range(n_langs),
    lang_counts["count"][:n_langs],
    color=colors,
    edgecolor="white",
    linewidth=0.8,
    alpha=0.9,
)

# Clean styling
ax.set_xticks(range(n_langs))
ax.set_xticklabels(
    lang_counts["language"][:n_langs],
    rotation=45,
    ha="right",
    fontsize=11,
    fontweight="bold",
)

ax.set_xlabel("Language Code", fontsize=16, fontweight="bold", labelpad=15)
ax.set_ylabel("Number of Claims", fontsize=16, fontweight="bold", labelpad=15)
ax.set_title(
    f"Language Distribution in Wikidata Claims\nTop {n_langs} Languages by Frequency",
    fontsize=20,
    fontweight="bold",
    pad=30,
)

# Remove the overlapping value labels entirely for cleaner look

# Format y-axis
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x/1000):.0f}k"))

# Remove all spines and grid for ultra-clean look
sns.despine(left=True, bottom=True, top=True, right=True)
ax.grid(False)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("language_hist.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
