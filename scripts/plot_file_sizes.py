import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Load the data
full_sizes = pl.read_csv("source_size/full_file_sizes.csv")
chunk_totals = pl.read_csv("source_size/chunk_totals.csv")

print("Full file sizes:")
print(full_sizes.head())
print(f"\nChunk totals:")
print(chunk_totals)

# Plot 1: Chunk totals bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart of total size per chunk
chunk_data = chunk_totals.sort("chunk_index")
sns.barplot(data=chunk_data.to_pandas(), x="chunk_index", y="total_size_gb", ax=ax1)
ax1.set_xlabel("Chunk Index", fontsize=12)
ax1.set_ylabel("Total Size (GB)", fontsize=12)
ax1.set_title("Total Size per Chunk", fontsize=14, fontweight='bold')

# Show every 10th tick label
chunk_indices = chunk_data["chunk_index"].to_list()
ax1.set_xticks(range(0, len(chunk_indices), 10))
ax1.set_xticklabels([chunk_indices[i] for i in range(0, len(chunk_indices), 10)])

# File count per chunk
sns.barplot(data=chunk_data.to_pandas(), x="chunk_index", y="file_count", ax=ax2)
ax2.set_xlabel("Chunk Index", fontsize=12)
ax2.set_ylabel("Number of Files", fontsize=12)
ax2.set_title("File Count per Chunk", fontsize=14, fontweight='bold')

# Show every 10th tick label
ax2.set_xticks(range(0, len(chunk_indices), 10))
ax2.set_xticklabels([chunk_indices[i] for i in range(0, len(chunk_indices), 10)])

plt.tight_layout()
plt.savefig("source_size/chunk_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot 2: Distribution of individual file sizes
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Histogram of file sizes
sns.histplot(data=full_sizes.to_pandas(), x="size_gb", bins=50, kde=True, ax=ax1)
ax1.set_xlabel("File Size (GB)", fontsize=12)
ax1.set_ylabel("Frequency", fontsize=12)
ax1.set_title("Distribution of Individual File Sizes", fontsize=14, fontweight='bold')

# Box plot of file sizes for ALL chunks (sample every 10th chunk for readability)
sample_chunks = full_sizes.with_columns(
    pl.col("filename").str.extract(r"chunk_(\d+)", 1).cast(pl.Int32).alias("chunk_num")
).filter(
    pl.col("chunk_num") % 10 == 0  # Every 10th chunk
)
chunk_labels_df = sample_chunks.to_pandas()
sns.boxplot(data=chunk_labels_df, x="chunk_num", y="size_gb", ax=ax2)
ax2.set_xlabel("Chunk Index (Every 10th)", fontsize=12)
ax2.set_ylabel("File Size (GB)", fontsize=12)
ax2.set_title("File Size Distribution by Chunk (Every 10th)", fontsize=14, fontweight='bold')

# Scatter plot: file index vs size for first chunk
chunk_0_files = full_sizes.filter(pl.col("filename").str.contains("chunk_0")).to_pandas()
chunk_0_files['file_index'] = range(len(chunk_0_files))
sns.scatterplot(data=chunk_0_files, x="file_index", y="size_gb", alpha=0.6, ax=ax3)
ax3.set_xlabel("File Index within Chunk 0", fontsize=12)
ax3.set_ylabel("File Size (GB)", fontsize=12)
ax3.set_title("File Sizes within Chunk 0", fontsize=14, fontweight='bold')

# Violin plot showing distribution across all chunks (sample for readability)
sample_for_violin = full_sizes.with_columns(
    pl.col("filename").str.extract(r"chunk_(\d+)", 1).cast(pl.Int32).alias("chunk_num")
).filter(
    pl.col("chunk_num") < 20  # First 20 chunks
).to_pandas()
sns.violinplot(data=sample_for_violin, x="chunk_num", y="size_gb", ax=ax4)
ax4.set_xlabel("Chunk Index (First 20)", fontsize=12)
ax4.set_ylabel("File Size (GB)", fontsize=12)
ax4.set_title("File Size Distribution by Chunk (Violin Plot)", fontsize=14, fontweight='bold')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("source_size/file_size_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot 3: Line plot showing trends across all chunks
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Total size trend
chunk_pandas = chunk_data.to_pandas()
sns.lineplot(data=chunk_pandas, x="chunk_index", y="total_size_gb", marker='o', ax=ax1)
ax1.set_xlabel("Chunk Index", fontsize=12)
ax1.set_ylabel("Total Size (GB)", fontsize=12)
ax1.set_title("Total Size Trend Across All Chunks", fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# File count trend
sns.lineplot(data=chunk_pandas, x="chunk_index", y="file_count", marker='s', color='orange', ax=ax2)
ax2.set_xlabel("Chunk Index", fontsize=12)
ax2.set_ylabel("Number of Files", fontsize=12)
ax2.set_title("File Count Trend Across All Chunks", fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("source_size/chunk_trends.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSummary statistics:")
print(f"Total chunks: {len(chunk_totals)}")
print(f"Total files: {full_sizes.height}")
print(f"Average file size: {full_sizes['size_gb'].mean():.3f} GB")
print(f"Median file size: {full_sizes['size_gb'].median():.3f} GB")
print(f"Largest file: {full_sizes['size_gb'].max():.3f} GB")
print(f"Smallest file: {full_sizes['size_gb'].min():.3f} GB")
print(f"Average chunk size: {chunk_totals['total_size_gb'].mean():.2f} GB")
print(f"Largest chunk: {chunk_totals['total_size_gb'].max():.2f} GB")
print(f"Smallest chunk: {chunk_totals['total_size_gb'].min():.2f} GB")