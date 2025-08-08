# wikidata-pq

Processed Wikidata entity IDs, properties, descriptions, and claims in Parquet format with multilingual support.

## Updates

- 7th August 2025: Finished developing the pipeline to handle various datavalue types (strings, timestamps, entity references, etc.). The final dataset and Hugging Face model card will be released once processing is complete.

## Background

This project processes the dense Wikidata parquet files from [philippesaade/wikidata](https://huggingface.co/datasets/philippesaade/wikidata)
into a more accessible format for analysis and machine learning applications.

In particular, it expands out the single row per entity ID, in which multiple languages are crammed,
and mean that 'claims' fields get very big, to the point they break typical data ingestion routes.
Besides which, most people will simply not need all that metadata, or would be happy to
cross-reference from a dataset with clear language subsets.

Put simply, the original Wikidata dataset presents several challenges:

1. **Massive JSON objects**: Claims data can exceed 1M characters in a single field, breaking Polars' JSON decoding ([bug report](https://github.com/pola-rs/polars/issues/23891))
2. **Nested multilingual structures**: Entity and property labels are deeply nested in language-specific dictionaries, and awkward to fish out despite representing fairly simple scalar values
3. **Complex schema**: Mixed datatypes (strings, timestamps, entity references) within the same fields, whose schema is probably available somewhere but not easy to hunt down
4. **Poor queryability**: The raw format requires extensive preprocessing for most use cases, which
   will be a barrier to wider bulk use of Wikidata (for instance, to pretrain LLMs on)

### Note on coverage

This dataset does not intend to perfectly reproduce the original, and note that subsets will be lost
during processing where the language of datavalue and property label does not match. This is
probably an acceptable loss for most users. Feel free to modify the code if the edge cases are of interest.

### Note on schema

The labels, descriptions, aliases and links were all fairly straightforward and have 3 fields each.

The claims field was significantly more complex: both nested subschemas, implicitly union dtypes
(e.g. scalar string and mappings from language to string) and all of this had to be ironed out to a
single common flat schema. This made it grow to many columns, so to combat this the claims table
schema was coalesced as follows:

- The "datavalue" field is the field that makes sense as the primary value for the claim. If the
  datavalue was already a scalar string it will remain so, but where it was a struct and there was a
  particular field which was the main value that becomes the coalesced datavalue. For datatype "wikibase-item"
  that is "wikibase-id" (but see also the "wikibase-label"), for datatype "quantity" it is "amount",
  for datatype "time" it was the "time" field and for datatype "monolingualtext" it is the
  "mlt-text" field.
- The "language" field is the coalesced union of the unit label language (for claims of datatype = "quantity"),
  wikibase label language (datatype "wikibase-item") and property label language (common to all, and
  which the other two were matched against). Monolingual text always has the same 'universal' language
  and so was not coalesced.

See the schema module for the mappings used here.

## Terminology

- An entity ID is the thing being described, starting with a Q plus some numbers
- A property is something like "instance of" (P31), the connecting part of a 'triple' statement,
  starting with a P plus some numbers
- A datavalue is a piece of metadata about the thing being described, and can have different types:
  - a `wikibase-item` type is a mapping of languages to labels e.g. all the translations of
    `{"en": "Wikimedia disambiguation page", ...}` for language codes like "en", "fr", and so on.
  - an `external-id` type is a scalar string identifier e.g. `/m/077yw_` whose property
    labels tell you that this is a "Freebase ID", again as a mapping over languages
  - a `time` type is a scalar time

For more info see [this page](https://doc.wikimedia.org/Wikibase/master/php/docs_topics_json.html#json_snaks).

There is also some odd terminology of "snaks" which mean something like "bite-sized pieces of info", but
can be interpreted as some fact or property of an entity.

Statements are composed of snaks,
qualifiers, and references. Qualifiers qualify the fact with context e.g. with a point in time,
and references provide provenance/authority info for the main Snak and qualifiers of an individual
Statement.

Site links ("sitelinks") are given as records with site, title, 'badges' (like "featured article")
and optionally a full URL.

To see a full example of an entity's JSON representation click [here](https://doc.wikimedia.org/Wikibase/master/php/docs_topics_json.html#json_example).

### Approach

This project transforms the raw Wikidata into structured Parquet files split into language subsets:

- **Expanding multilingual labels**: Both entity datavalues and property labels are unpivoted into separate language rows
- **Language matching**: Filtering to combinations where both the entity description and property label exist in the same language
- **Type-aware processing**: Handling different datavalue types (strings, timestamps, entity references) appropriately
- **Partitioned output**: Language-partitioned Parquet files allow downloading only the subset you need

## Data Structure

The processed dataset expands each Wikidata claim ("mainsnak") into multilingual rows. For example, a single claim about an entity might become 100+ rows covering different languages where both the entity description and property label are available.

### Key Fields

- `id`: Wikidata entity ID (e.g., Q398520)
- `property`: Property ID (e.g., P31 for "instance of")
- `datavalue-id`: Referenced entity ID (for entity-type properties)
- `datavalue-label`: Entity description in the target language
- `property-label`: Property description in the target language
- `datavalue-label-lang` / `property-label-lang`: Language codes (filtered to matching pairs)
- `datatype`: Type of the property value (wikibase-item, string, time, etc.)
- `rank`: Claim ranking (normal, preferred, deprecated)

### Language Coverage

The filtering process retains language combinations where both entity and property labels exist. While this loses some language-specific data, it ensures semantic consistency and covers the majority of well-documented languages in Wikidata (typically 50-150 languages per entity).

## Output Format

- **Format**: Parquet files partitioned by language
- **Partitioning**: `language=en/`, `language=fr/`, etc.
- **Size**: Manageable chunks allowing selective download of specific languages
- **Schema**: Consistent across all partitions with explicit typing

## Source Data

Original dataset: [philippesaade/wikidata](https://huggingface.co/datasets/philippesaade/wikidata)

The source provides a Wikidata dump in Parquet format but with an entire entity ID's metadata packed into a single record (row), containing entity descriptions, properties, claims, and multilingual labels for millions of entities. This reprocessing effort's main aim is to split this data into language subsets.
