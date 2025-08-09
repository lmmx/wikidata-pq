import polars as pl
import matplotlib.pyplot as plt

# Load the data
full_sizes = pl.read_csv("source_size/full_file_sizes.csv")
chunk_totals = pl.read_csv("source_size/chunk_totals.csv")

print("Full file sizes:")
print(full_sizes.head())
print(f"\nChunk totals:")
print(chunk_totals)

# Plot 1: Chunk totals bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Bar chart of total size per chunk
chunk_data = chunk_totals.sort("chunk_index")
ax1.bar(chunk_data["chunk_index"], chunk_data["total_size_gb"])
ax1.set_xlabel("Chunk Index")
ax1.set_ylabel("Total Size (GB)")
ax1.set_title("Total Size per Chunk")
ax1.tick_params(axis='x', rotation=45)

# Plot 2: File count per chunk
ax2.bar(chunk_data["chunk_index"], chunk_data["file_count"])
ax2.set_xlabel("Chunk Index")
ax2.set_ylabel("Number of Files")
ax2.set_title("File Count per Chunk")
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("source_size/chunk_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot 3: Distribution of individual file sizes
plt.figure(figsize=(12, 8))

# Histogram of file sizes
plt.subplot(2, 2, 1)
plt.hist(full_sizes["size_gb"], bins=50, alpha=0.7, edgecolor='black')
plt.xlabel("File Size (GB)")
plt.ylabel("Frequency")
plt.title("Distribution of Individual File Sizes")

# Box plot of file sizes per chunk (sample first few chunks if too many)
plt.subplot(2, 2, 2)
sample_chunks = full_sizes.filter(
    pl.col("filename").str.extract(r"chunk_(\d+)", 1).cast(pl.Int32) < 10
)
chunk_labels = sample_chunks.with_columns(
    pl.col("filename").str.extract(r"chunk_(\d+)", 1).alias("chunk")
)
chunks_list = chunk_labels["chunk"].unique().sort()
box_data = [chunk_labels.filter(pl.col("chunk") == c)["size_gb"].to_list() 
           for c in chunks_list]
plt.boxplot(box_data, labels=chunks_list)
plt.xlabel("Chunk Index")
plt.ylabel("File Size (GB)")
plt.title("File Size Distribution by Chunk (First 10)")

# Scatter plot: file index vs size for first chunk
plt.subplot(2, 1, 2)
chunk_0_files = full_sizes.filter(pl.col("filename").str.contains("chunk_0"))
file_indices = range(len(chunk_0_files))
plt.scatter(file_indices, chunk_0_files["size_gb"], alpha=0.6, s=20)
plt.xlabel("File Index within Chunk 0")
plt.ylabel("File Size (GB)")
plt.title("File Sizes within Chunk 0")

plt.tight_layout()
plt.savefig("source_size/file_size_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSummary statistics:")
print(f"Total chunks: {len(chunk_totals)}")
print(f"Total files: {full_sizes.height}")
print(f"Average file size: {full_sizes['size_gb'].mean():.3f} GB")
print(f"Median file size: {full_sizes['size_gb'].median():.3f} GB")
print(f"Largest file: {full_sizes['size_gb'].max():.3f} GB")
print(f"Smallest file: {full_sizes['size_gb'].min():.3f} GB")