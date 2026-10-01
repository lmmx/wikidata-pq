// Reads the dataset's Parquet files from the Hub with hyparquet, a few byte ranges at a time:
// a file's footer once, then only the row groups a lookup can need, found by the footer's
// min/max statistics (the files are sorted by the column looked up), adjacent ones in one
// request, straight from the CDN.
//
// The hyparquet functions and compressors are passed in, so the same code runs in the page
// (from a CDN) and in Node (from node_modules).

const TAIL = 1 << 20;  // the bytes fetched from a file's end on opening: its footer, usually

// A file on the Hub as hyparquet's AsyncBuffer, read straight from the CDN. The Hub's URL
// redirects each request to a signed CDN URL (about 0.15 s a time, an hour's validity), so
// the first request, for the file's last TAIL bytes (its footer), finds the CDN URL and the
// file's length, and later ranges go to the CDN URL, found again when it expires or is
// refused. Spans fetched ahead (`prefetch`) serve the slices inside them, so a run of
// adjacent row groups costs one request.
async function hubBuffer(url) {
  let target = url, expires = 0;
  const spans = [];  // { start, end, data: Promise<ArrayBuffer> }
  const get = async (range) => {
    for (let tries = 0; ; tries++) {
      const res = await fetch(target, { headers: { Range: range } });
      if (res.ok || tries) {
        if (!res.ok) throw new Error(`fetch failed ${res.status}`);
        if (res.redirected || target === url) {
          target = res.url;
          const m = /[?&]Expires=(\d+)/.exec(target);
          expires = m ? +m[1] * 1000 : Date.now() + 30 * 60e3;
        }
        return res;
      }
      target = url;  // the CDN URL expired or was refused: through the Hub again
    }
  };
  const first = await get(`bytes=-${TAIL}`);
  const total = Number((first.headers.get("content-range") ?? "").split("/")[1]);
  const byteLength = total || Number(first.headers.get("content-length"));
  const tailStart = Math.max(0, byteLength - TAIL);
  spans.push({ start: tailStart, end: byteLength, data: first.arrayBuffer() });
  const fetchSpan = (start, end) => {
    if (Date.now() > expires - 60e3) target = url;
    const data = get(`bytes=${start}-${end - 1}`).then(async (res) => {
      const buf = await res.arrayBuffer();
      return res.status === 206 ? buf : buf.slice(start, end);  // a 200 is the whole file
    });
    const span = { start, end, data };
    spans.push(span);
    return span;
  };
  return {
    byteLength,
    prefetch(start, end) {
      if (!spans.some((x) => x.start <= start && end <= x.end)) fetchSpan(start, end);
    },
    async slice(start, end = byteLength) {
      const span = spans.find((x) => x.start <= start && end <= x.end) ?? fetchSpan(start, end);
      const buf = await span.data;
      return buf.slice(start - span.start, end - span.start);
    },
  };
}

export function makeData({ hyparquet, compressors, base }) {
  const { parquetMetadataAsync, parquetReadObjects, parquetSchema } = hyparquet;
  const files = new Map();
  const postingsOf = new Map();

  // A file's buffer and footer, fetched once
  function open(name) {
    if (!files.has(name)) {
      files.set(name, (async () => {
        const file = await hubBuffer(base + name);
        const metadata = await parquetMetadataAsync(file, { initialFetchSize: TAIL });
        let start = 0;
        const groups = metadata.row_groups.map((rg) => {
          const rows = Number(rg.num_rows);
          const g = { start, end: start + rows, stats: {}, from: Infinity, to: 0, bytes: {} };
          for (const col of rg.columns) {
            const m = col.meta_data;
            if (m?.statistics) g.stats[m.path_in_schema.join(".")] = m.statistics;
            if (!m) continue;
            const at = Number(m.dictionary_page_offset ?? m.data_page_offset);
            const top = m.path_in_schema[0];
            g.bytes[top] = (g.bytes[top] ?? 0) + Number(m.total_compressed_size);
            g.from = Math.min(g.from, at);
            g.to = Math.max(g.to, at + Number(m.total_compressed_size));
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
    // Adjacent row groups' bytes in one request each run, when the columns read are most of
    // their bytes (else hyparquet's own requests, column by column, fetch less)
    const share = (g) => columns.reduce((n, c) => n + (g.bytes[c] ?? 0), 0) / (g.to - g.from);
    const sorted = groups.filter((g) => share(g) >= 0.8).sort((a, b) => a.from - b.from);
    for (let i = 0; i < sorted.length; ) {
      let j = i;
      while (j + 1 < sorted.length && sorted[j + 1].from <= sorted[j].to) j++;
      if (sorted[i].from < sorted[j].to) file.prefetch(sorted[i].from, sorted[j].to);
      i = j + 1;
    }
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

    // A class's direct members among the coded items: [{ id: "Q…", subclass }], or [] for a
    // run without the members table
    async members(cls) {
      try {
        const { groups } = await open("members.parquet");
        const rows = await readGroups("members.parquet",
          overlapping(groups, "class", cls, cls), ["class", "id", "subclass"]);
        return rows.filter((r) => r.class === cls).map((r) => ({ id: `Q${r.id}`, subclass: r.subclass }));
      } catch {
        return [];
      }
    },

    // Fetch a file's footer ahead of its first lookup
    warm(name) { open(name).catch(() => {}); },

    // Items whose lowercased label starts with `q`: exact matches first, then coded items,
    // then by Wikipedias.
    // With several words and few such labels, also those whose label starts with the first
    // words and whose description has the rest ("transformer machine learning"). Reads at
    // most `maxGroups` row groups of names per label prefix.
    async search(q, { limit = 40, maxGroups = 3 } = {}) {
      const { groups } = await open("names.parquet");
      const columns = ["key", "label", "id", "description", "wikipedias", "coded"];
      const byPrefix = async (prefix) => (await readGroups("names.parquet",
        overlapping(groups, "key", prefix, prefix + "\uffff").slice(0, maxGroups), columns))
        .filter((r) => r.key.startsWith(prefix));
      // Exact names first, then items with a code (they can be explored), then by Wikipedias
      const rank = (rows, exact) => rows.sort((a, b) => (b.key === exact) - (a.key === exact)
        || (b.coded !== false) - (a.coded !== false) || b.wikipedias - a.wikipedias);
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
    // filters re-rank the same postings). Older layouts store `weight` and `norm`, heaviest
    // first, or a float `unit`; the current one a 16-bit `unit16`, by id.
    async postings(feature, { limit = Infinity } = {}) {
      const columns = ["feature", "id", "unit16", "unit", "weight", "norm", "kinds"];
      if (limit === Infinity && postingsOf.has(feature)) return postingsOf.get(feature);
      const { groups } = await open("postings.parquet");
      let hit = overlapping(groups, "feature", feature, feature);
      if (limit < Infinity) hit = hit.slice(0, 1);  // rank order: the first row group leads
      const read = readGroups("postings.parquet", hit, columns).then((rows) => rows
        .filter((r) => r.feature === feature)
        .slice(0, limit)
        .map((r) => ({
          id: typeof r.id === "number" ? `Q${r.id}` : r.id,
          unit: r.unit16 != null ? r.unit16 / 65535 : r.unit ?? r.weight / r.norm,
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
// weight]) and the neighbours, each with `shared`: which of them it has. `onRead(done, total)`
// is called as each feature's postings arrive.
export async function neighbours(data, item,
  { use = () => true, keep = () => true, top = 12, limit = 30, onRead = () => {} } = {}) {
  const seed = item.features.map((f, i) => [f, item.weights[i]])
    .filter(([f]) => use(f))
    .sort((a, b) => b[1] - a[1]).slice(0, top);
  let done = 0;
  onRead(0, seed.length);
  const lists = await Promise.all(seed.map(([f]) =>
    data.postings(f).then((list) => (onRead(++done, seed.length), list))));
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
