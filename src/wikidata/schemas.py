import polars as pl

__all__ = [
    "str_snak_values",
    "struct_snak_values",
    "common_schema",
    "struct_schemata",
    "total_schema",
    "coalesced_schema",
    "coalesced_cols",
    "final_schema",
]

# Some entity IDs inferred to be str dtype datavalue from parser at:
# https://github.com/maxlath/wikibase-sdk/blob/2447f7d1355d3461adb2676b0c5336534932feda/src/types/snakvalue.ts#L68

str_snak_values = [
    "commonsMedia",
    "geo-shape",
    "tabular-data",
    "url",
    "external-id",
    "string",
    "musical-notation",
    "math",
    "entity-schema",
    "wikibase-lexeme",
    "wikibase-form",
    "wikibase-sense",
]
struct_snak_values = {
    "wikibase-item": [
        "id",
        "labels",  # categorical
    ],
    "wikibase-item": [
        "id",
        "labels",  # categorical
    ],
    "globe-coordinate": [
        "latitude",
        "longitude",
        "altitude",
        "precision",
        "globe",
    ],
    "quantity": [
        "amount",
        "unit",
        "upperBound",
        "lowerBound",
        "unit-labels",
    ],
    "time": [
        "time",
        "timezone",
        "before",
        "after",
        "precision",
        "calendarmodel",
    ],
    "monolingualtext": [
        "text",
        "language",
    ],
}

common_schema = {
    "id": pl.String,
    "rank": pl.String,
    "property": pl.String,
    "property-label-lang": pl.String,
    "property-label": pl.String,
    "datatype": pl.String,
    "datavalue": pl.String,  # Now only used for string dtype datavalues
    # The struct schemata cols should go here
}

struct_schemata = {
    "wikibase-item": {
        "wikibase-id": pl.String,
        "wikibase-label": pl.String,
        "wikibase-label-lang": pl.String,
    },
    "wikibase-property": {
        # overload all of these to match wikibase-item's
        # (we can still distinguish a row on the datatype column)
        "wikibase-id": pl.String,
        "wikibase-label": pl.String,
        "wikibase-label-lang": pl.String,
    },
    "globe-coordinate": {
        "latitude": pl.Float64,
        "longitude": pl.Float64,
        "altitude": pl.Float64,
        "precision": pl.Float64,
        "globe": pl.String,
    },
    "quantity": {
        "amount": pl.String,
        "unit": pl.String,
        "upperBound": pl.String,
        "lowerBound": pl.String,
        "unit-label": pl.String,
        "unit-label-lang": pl.String,
    },
    "time": {
        "time": pl.String,
        "timezone": pl.Int64,
        "before": pl.Int64,
        "after": pl.Int64,
        "ts_precision": pl.Int64,
        "calendarmodel": pl.String,
    },
    "monolingualtext": {
        "datavalue": pl.String,  # act like it was a regular simple type
        "mlt-language": pl.String,  # single universal value, we can't filter on it
    },
}

total_schema = {
    **common_schema,
    **{k: v for d in struct_schemata.values() for k, v in d.items()},
}

coalesced_schema = {
    "language": pl.String,
}
coalesced_cols = {
    "language": ["wikibase-label-lang", "unit-label-lang", "property-label-lang"]
}

final_schema = {
    **coalesced_schema,
    **{
        k: v
        for k, v in total_schema.items()
        if k not in [v for lst in coalesced_cols.values() for v in lst]
    },
}
