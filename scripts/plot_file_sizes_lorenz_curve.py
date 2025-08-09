import matplotlib.pyplot as plt
import numpy as np
import polars as pl

full_sizes = pl.read_csv("source_size/full_file_sizes.csv")

s = full_sizes["size_gb"].sort()  # ascending
sizes = s.to_numpy()
cum_files = np.arange(1, len(sizes) + 1) / len(sizes)
cum_bytes = np.cumsum(sizes) / sizes.sum()

plt.figure(figsize=(6, 6))
plt.plot(cum_files, cum_bytes, label="Lorenz curve")
plt.plot([0, 1], [0, 1], linestyle="--", label="Equality line")

# Mark 80% of bytes point
k = int(np.searchsorted(cum_bytes, 0.80)) + 1
plt.scatter(cum_files[k - 1], cum_bytes[k - 1])
plt.annotate(
    f"{k/len(sizes):.1%} of files → 80% of bytes",
    xy=(cum_files[k - 1], cum_bytes[k - 1]),
    xytext=(0.55, 0.2),
    arrowprops=dict(arrowstyle="->"),
)

plt.xlabel("Cumulative share of files")
plt.ylabel("Cumulative share of total size (GB)")
plt.title("Lorenz Curve of File Sizes")
plt.legend()
plt.tight_layout()
plt.savefig("source_size/lorenz_curve.png", dpi=300, bbox_inches="tight")
plt.close()

# Optional: Gini coefficient
gini = 1 - 2 * np.trapezoid(cum_bytes, cum_files)
print(f"Gini (file size inequality): {gini:.3f}")
print("Lorenz curve saved to source_size/lorenz_curve.png")
