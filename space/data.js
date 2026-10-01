// Reads the dataset's Parquet files from the Hub with hyparquet, a few byte ranges at a time:
// a file's footer once, then only the row groups a lookup can need, found by the footer's
// min/max statistics (the files are sorted by the column looked up).
//
// The hyparquet functions and compressors are passed in, so the same code runs in the page
// (from a CDN) and in Node (from node_modules).

export function makeData({ hyparquet, compressors, base }) {
  const { asyncBufferFromUrl, cachedAsyncBuffer, parquetMetadataAsync, parquetReadObjects } =
    hyparquet;
  const files = new Map();

  // A file's buffer and footer, fetched once
  function open(name) {
    if (!files.has(name)) {
      files.set(name, (async () => {
        const file = cachedAsyncBuffer(await asyncBufferFromUrl({ url: base + name }));
        const metadata = await parquetMetadataAsync(file, { initialFetchSize: 1 << 20 });
        let start = 0;
        const groups = metadata.row_groups.map((rg) => {
          const rows = Number(rg.num_rows);
          const g = { start, end: start + rows, stats: {} };
          for (const col of rg.columns) {
            const m = col.meta_data;
            if (m?.statistics) g.stats[m.path_in_schema.join(".")] = m.statistics;
          }
          start += rows;
          return g;
        });
        return { file, metadata, groups };
      })());
    }
    return files.get(name);
  }

  // The row groups whose [min, max] of `column` overlaps [lo, hi]
  function overlapping(groups, column, lo, hi) {
    return groups.filter((g) => {
      const s = g.stats[column];
      if (!s) return true;
      const min = s.min_value ?? s.min, max = s.max_value ?? s.max;
      return !(max < lo || min > hi);
    });
  }

  // Row groups read in parallel, rows in file order, 64-bit integers as numbers
  async function readGroups(name, groups, columns) {
    const { file, metadata } = await open(name);
    const parts = await Promise.all(groups.map((g) => parquetReadObjects({
      file, metadata, columns, compressors, rowStart: g.start, rowEnd: g.end,
    })));
    return parts.flat().map(numbers);
  }

  return {
    // The whole features table (one small file)
    async features() {
      const { file, metadata } = await open("features.parquet");
      return (await parquetReadObjects({ file, metadata, compressors })).map(numbers);
    },

    // Fetch a file's footer ahead of its first lookup
    warm(name) { open(name).catch(() => {}); },

    // Items whose lowercased label starts with `q`: exact matches first, then by Wikipedias.
    // Reads at most `maxGroups` row groups of names, from the first that can hold `q`.
    async search(q, { limit = 12, maxGroups = 3 } = {}) {
      const { groups } = await open("names.parquet");
      const hit = overlapping(groups, "key", q, q + "￿").slice(0, maxGroups);
      const found = (await readGroups("names.parquet", hit,
        ["key", "label", "id", "description", "wikipedias"]))
        .filter((r) => r.key.startsWith(q));
      found.sort((a, b) => (b.key === q) - (a.key === q) || b.wikipedias - a.wikipedias);
      return found.slice(0, limit);
    },

    // Items by id: one row group each (items is sorted by id)
    async items(ids, columns = ["id", "label", "features", "weights", "norm"]) {
      const { groups } = await open("items.parquet");
      const want = new Set(ids);
      const hit = new Set(ids.flatMap((id) => overlapping(groups, "id", id, id)));
      const rows = await readGroups("items.parquet", [...hit], columns);
      return rows.filter((r) => want.has(r.id));
    },

    // A feature's postings, heaviest first: the row groups that can hold it
    async postings(feature, { limit = Infinity, columns = ["feature", "id", "weight", "norm"] } = {}) {
      const { groups } = await open("postings.parquet");
      let hit = overlapping(groups, "feature", feature, feature);
      if (limit < Infinity) hit = hit.slice(0, 1);  // rank order: the first row group leads
      const rows = await readGroups("postings.parquet", hit, columns);
      return rows.filter((r) => r.feature === feature).slice(0, limit);
    },
  };
}

function numbers(row) {
  for (const k in row) {
    const v = row[k];
    if (typeof v === "bigint") row[k] = Number(v);
    else if (Array.isArray(v) && typeof v[0] === "bigint") row[k] = v.map(Number);
  }
  return row;
}

// The items most like `item`, among those with any of its `top` heaviest features, by cosine
// of the weighted features (over those features). Returns the seed features ([feature,
// weight], heaviest first) and the neighbours, each with `shared`: which of them it has.
export async function neighbours(data, item, { top = 8, limit = 30 } = {}) {
  const seed = item.features.map((f, i) => [f, item.weights[i]])
    .sort((a, b) => b[1] - a[1]).slice(0, top);
  const lists = await Promise.all(seed.map(([f]) => data.postings(f)));
  const acc = new Map();
  seed.forEach(([, w], i) => {
    for (const p of lists[i]) {
      if (p.id === item.id) continue;
      let a = acc.get(p.id);
      if (!a) acc.set(p.id, a = { id: p.id, dot: 0, norm: p.norm, shared: [] });
      a.dot += p.weight * w;
      a.shared.push(i);
    }
  });
  const near = [...acc.values()]
    .map((a) => ({ ...a, similarity: a.dot / (a.norm * item.norm) }))
    .sort((a, b) => b.similarity - a.similarity || (a.id < b.id ? -1 : 1))
    .slice(0, limit);
  return { seed, near };
}
