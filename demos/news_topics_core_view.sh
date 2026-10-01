# uv pip install embedding-atlas
# The topics with a page on at least 5 news sites (demos/news_topics_export.sh first)
embedding-atlas demos/output/news_topics.parquet --text text \
    --query "SELECT *, string_split(outlets, '; ') AS outlet_list FROM data WHERE n_outlets >= 5" \
    --features outlet_list \
    --umap-random-state 0
