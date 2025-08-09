import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

full_sizes = pl.read_csv("source_size/full_file_sizes.csv")

s = full_sizes["size_gb"].sort()  # ascending
sizes = s.to_numpy()
cum_files = np.arange(1, len(sizes) + 1) / len(sizes)
cum_bytes = np.cumsum(sizes) / sizes.sum()

plt.figure(figsize=(8, 8))
plt.plot(cum_files, cum_bytes, label="Lorenz curve", linewidth=2.5)
plt.plot([0, 1], [0, 1], linestyle="--", label="Equality line", linewidth=2)

# Mark 80% of bytes point
k = int(np.searchsorted(cum_bytes, 0.80)) + 1
plt.scatter(cum_files[k - 1], cum_bytes[k - 1], s=100, zorder=5, color='red')
plt.annotate(
    f"{k/len(sizes):.1%} of files → 80% of bytes",
    xy=(cum_files[k - 1], cum_bytes[k - 1]),
    xytext=(0.55, 0.2),
    arrowprops=dict(arrowstyle="->", lw=2, color="black"),
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
)

plt.xlabel("Cumulative share of files", fontsize=12)
plt.ylabel("Cumulative share of total size (GB)", fontsize=12)
plt.title("Lorenz Curve of File Sizes", fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("source_size/lorenz_curve.png", dpi=300, bbox_inches="tight")
plt.close()

# Optional: Gini coefficient
gini = 1 - 2 * np.trapezoid(cum_bytes, cum_files)
print(f"Gini (file size inequality): {gini:.3f}")
print("Lorenz curve saved to source_size/lorenz_curve.png")