import polars as pl
import matplotlib.pyplot as plt

df = pl.read_parquet("repro_results/combined_sweep.parquet")

# Plot RSS vs file_idx, one line per rows_per_file setting
fig, ax = plt.subplots(figsize=(12, 6))
for rows in df["rows_per_file"].unique().sort():
    subset = df.filter(pl.col("rows_per_file") == rows)
    ax.plot(subset["file_idx"], subset["rss_gb"], label=f"n={rows}", alpha=0.7)

ax.set_xlabel("File index")
ax.set_ylabel("RSS (GB)")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_title("Memory usage vs iteration count at different row sizes")
plt.tight_layout()
plt.savefig("leak_sweep.png", dpi=150)