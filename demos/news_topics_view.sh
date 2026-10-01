# uv pip install embedding-atlas
embedding-atlas demos/output/news_topics.parquet --text text \
    --query "SELECT *, string_split(outlets, '; ') AS outlet_list FROM data" \
    --features outlet_list \
    --umap-random-state 0
