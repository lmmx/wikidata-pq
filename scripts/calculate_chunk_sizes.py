import polars as pl
from huggingface_hub import HfApi

repo_id = "philippesaade/wikidata"
api = HfApi()

# Get directory listing with file details in one API call
files_info = list(
   api.list_repo_tree(repo_id=repo_id, path_in_repo="data", repo_type="dataset")
)

# Full file sizes table
df = (
   pl.DataFrame(
       [{"name": f.path, "size": f.size} for f in files_info if hasattr(f, "size")]
   )
   .lazy()
   .filter(pl.col("name").str.contains(r"chunk_.*\.parquet$"))
   .with_columns([
       pl.col("name").str.split("/").list.last().alias("filename"),
       (pl.col("size") / 1024**3).round(3).alias("size_gb"),
   ])
   .select(["filename", "size", "size_gb"])
   .sort("filename")
)

# Chunk totals table
chunk_totals = (
   df
   .with_columns([
       pl.col("filename")
       .str.split("_").list.get(1)
       .str.split("-").list.get(0)
       .alias("chunk_index")
   ])
   .group_by("chunk_index")
   .agg([
       pl.col("size").sum().alias("total_size_bytes"),
       pl.col("size_gb").sum().round(3).alias("total_size_gb"),
       pl.len().alias("file_count")
   ])
   .sort("chunk_index")
)

# Sink to CSV files
df.sink_csv("source_size/full_file_sizes.csv", mkdir=True)
chunk_totals.sink_csv("source_size/chunk_totals.csv", mkdir=True)

# Print results
print("Full file sizes:")
print(df.collect())

print("\nChunk totals:")
print(chunk_totals.collect())

total_size_bytes = df.select(pl.col("size").sum()).collect().item()
total_size_gb = total_size_bytes / (1024**3)
print(f"\nOverall total: {total_size_bytes:,} bytes ({total_size_gb:.2f} GB)")