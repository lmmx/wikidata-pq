# uv pip install embedding-atlas
# The items with a defining formula (demos/formulas_export.sh first): colour by `field`,
# the first field that studies each, or pick fields in the features list
embedding-atlas demos/output/formulas.parquet --text text \
    --query "SELECT *, string_split(\"studied by\", '; ') AS fields, string_split(\"studied by\", '; ')[1] AS field FROM data" \
    --features fields \
    --umap-random-state 0
