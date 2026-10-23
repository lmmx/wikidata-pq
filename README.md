# Benchmark data v1

The input data of wikidata-pq's public benchmark (`bench/` on master): 11 chunk files of the
2026-10-12 Wikidata JSON dump, as wikidata-pq's chunking writes them. Each holds 7 columns
of JSON text (`id`, `labels`, `descriptions`, `aliases`, `sitelinks`, `claims`, `entity`),
60,000 entities in all.

| file | entities | from |
|---|---|---|
| `c0.parquet` | 10,000 | chunk 0 |
| `r10.parquet`, `r30.parquet`, `r50.parquet`, `r70.parquet`, `r90.parquet` | 6,541, 2,699, 7,000, 3,764, 1,081 | chunks 10 to 90 of the routed entities |
| `s10.parquet`, `s30.parquet`, `s50.parquet`, `s70.parquet`, `s90.parquet` | 3,459, 7,301, 3,000, 6,236, 8,919 | chunks 10 to 90 of the scholarly-article entities |

`SHA256SUMS` holds each file's SHA-256; the benchmark checks them before it runs. This
branch is data only and is never merged. The benchmark names the commit it reads, so a
new version of the data is a new commit, and old results keep pointing at their own.

Wikidata's data is CC0.
