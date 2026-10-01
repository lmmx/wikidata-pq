// Reads the dataset's Parquet files from the Hub with hyparquet, a few byte ranges at a time:
// a file's footer once, then only the row groups a lookup can need, found by the footer's
// min/max statistics (the files are sorted by the column looked up).
//
// The hyparquet functions and compressors are passed in, so the same code runs in the page
// (from a CDN) and in Node (from node_modules).

export function makeData({ hyparquet, compressors, base }) {
  const { asyncBufferFromUrl, cachedAsyncBuffer, parquetMetadataAsync, parquetReadObjects,
    parquetSchema } = hyparquet;
  const files = new Map();
  const postingsOf = new Map();

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
        const names = new Set(parquetSchema(metadata).children.map((c) => c.element.name));
        return { file, metadata, groups, names };
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

  // Row groups read in parallel, rows in file order, 64-bit integers as numbers; columns the
  // file lacks (an older run's) are left out
  async function readGroups(name, groups, columns) {
    const { file, metadata, names } = await open(name);
    columns = columns.filter((c) => names.has(c));
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

    // The class hierarchy, whole (one small file), or an empty list for a run without it
    async classes() {
      try {
        const { file, metadata } = await open("classes.parquet");
        return (await parquetReadObjects({ file, metadata, compressors })).map(numbers);
      } catch {
        return [];
      }
    },

    // Fetch a file's footer ahead of its first lookup
    warm(name) { open(name).catch(() => {}); },

    // Items whose lowercased label starts with `q`: exact matches first, then by Wikipedias.
    // With several words and few such labels, also those whose label starts with the first
    // words and whose description has the rest ("transformer machine learning"). Reads at
    // most `maxGroups` row groups of names per label prefix.
    async search(q, { limit = 40, maxGroups = 3 } = {}) {
      const { groups } = await open("names.parquet");
      const columns = ["key", "label", "id", "description", "wikipedias", "coded"];
      const byPrefix = async (prefix) => (await readGroups("names.parquet",
        overlapping(groups, "key", prefix, prefix + "\uffff").slice(0, maxGroups), columns))
        .filter((r) => r.key.startsWith(prefix));
      const rank = (rows, exact) => rows.sort((a, b) =>
        (b.key === exact) - (a.key === exact) || b.wikipedias - a.wikipedias);
      const found = rank(await byPrefix(q), q);
      const words = q.split(/\s+/).filter(Boolean);
      for (let k = words.length - 1; k >= 1 && found.length < limit; k--) {
        const prefix = words.slice(0, k).join(" ");
        const rest = words.slice(k);
        const seen = new Set(found.map((r) => r.id));
        found.push(...rank((await byPrefix(prefix)).filter((r) => !seen.has(r.id) &&
          rest.every((w) => (r.description ?? "").toLowerCase().includes(w))), prefix));
      }
      return found.slice(0, limit);
    },

    // An item's description: names is sorted by lowercased label, so its label finds it
    async description(id, label) {
      if (!label) return null;
      const key = label.toLowerCase();
      const { groups } = await open("names.parquet");
      const rows = await readGroups("names.parquet", overlapping(groups, "key", key, key),
        ["key", "id", "description"]);
      return rows.find((r) => r.id === id)?.description ?? null;
    },

    // Items by id: one row group each (items is sorted by id)
    async items(ids, columns = ["id", "label", "kinds", "is_class", "features", "weights", "norm"]) {
      const { groups } = await open("items.parquet");
      const want = new Set(ids);
      const hit = new Set(ids.flatMap((id) => overlapping(groups, "id", id, id)));
      const rows = await readGroups("items.parquet", [...hit], columns);
      return rows.filter((r) => want.has(r.id));
    },

    // A feature's postings, each with `id` ("Q…"), `unit` (its weight for the feature over
    // its norm) and `kinds`: the row groups that can hold it (kept for the page's life, as
    // filters re-rank the same postings). Older runs store `weight` and `norm`, heaviest
    // first; newer ones `unit`, by id.
    async postings(feature, { limit = Infinity } = {}) {
      const columns = ["feature", "id", "unit", "weight", "norm", "kinds"];
      if (limit === Infinity && postingsOf.has(feature)) return postingsOf.get(feature);
      const { groups } = await open("postings.parquet");
      let hit = overlapping(groups, "feature", feature, feature);
      if (limit < Infinity) hit = hit.slice(0, 1);  // rank order: the first row group leads
      const read = readGroups("postings.parquet", hit, columns).then((rows) => rows
        .filter((r) => r.feature === feature)
        .slice(0, limit)
        .map((r) => ({
          id: typeof r.id === "number" ? `Q${r.id}` : r.id,
          unit: r.unit ?? r.weight / r.norm,
          kinds: r.kinds,
        })));
      if (limit === Infinity) postingsOf.set(feature, read);
      return read;
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

// The items most like `item`, by cosine of the weighted features, over the item's features
// that `use` accepts (e.g. those on at most so many items: a feature on a million items
// costs megabytes to read and weighs little), at most `top` of them, heaviest first, among
// the candidates `keep` accepts (each with its `kinds`). Returns those features ([feature,
// weight]) and the neighbours, each with `shared`: which of them it has.
export async function neighbours(data, item,
  { use = () => true, keep = () => true, top = 12, limit = 30 } = {}) {
  const seed = item.features.map((f, i) => [f, item.weights[i]])
    .filter(([f]) => use(f))
    .sort((a, b) => b[1] - a[1]).slice(0, top);
  const lists = await Promise.all(seed.map(([f]) => data.postings(f)));
  const acc = new Map();
  seed.forEach(([, w], i) => {
    for (const p of lists[i]) {
      if (p.id === item.id) continue;
      let a = acc.get(p.id);
      if (!a) acc.set(p.id, a = { id: p.id, dot: 0, kinds: p.kinds, shared: [] });
      a.dot += p.unit * w;
      a.shared.push(i);
    }
  });
  const near = [...acc.values()]
    .filter(keep)
    .map((a) => ({ ...a, similarity: a.dot / item.norm }))
    .sort((a, b) => b.similarity - a.similarity || (a.id < b.id ? -1 : 1))
    .slice(0, limit);
  return { seed, near };
}
