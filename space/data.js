// Reads the dataset's Parquet files from the Hub with hyparquet, a few byte ranges at a time:
// a file's footer once, then only the row groups a lookup can need, found by the footer's
// min/max statistics (the files are sorted by the column looked up), adjacent ones in one
// request, straight from the CDN.
//
// The hyparquet functions and compressors are passed in, so the same code runs in the page
// (from a CDN) and in Node (from node_modules).

const TAIL = 1 << 20;  // the bytes fetched from a file's end on opening: its footer, usually
const KEEP_BYTES = 64 << 20;  // the most bytes kept from a file's earlier requests

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
    // The oldest spans let go past KEEP_BYTES (the footer's stays); a slice already waiting on
    // one still gets it
    let kept = 0;
    for (let i = spans.length - 1; i >= 1; i--) {
      kept += spans[i].end - spans[i].start;
      if (kept > KEEP_BYTES && i < spans.length - 1) { spans.splice(1, i); break; }
    }
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
  const { parquetMetadataAsync, parquetRead, parquetReadObjects, parquetSchema } = hyparquet;
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

  // Adjacent row groups' bytes in one request each run, when the columns read are most of
  // their bytes (else hyparquet's own requests, column by column, fetch less)
  function prefetch(file, groups, columns) {
    const share = (g) => columns.reduce((n, c) => n + (g.bytes[c] ?? 0), 0) / (g.to - g.from);
    const sorted = groups.filter((g) => share(g) >= 0.8).sort((a, b) => a.from - b.from);
    for (let i = 0; i < sorted.length; ) {
      let j = i;
      while (j + 1 < sorted.length && sorted[j + 1].from <= sorted[j].to) j++;
      if (sorted[i].from < sorted[j].to) file.prefetch(sorted[i].from, sorted[j].to);
      i = j + 1;
    }
  }

  // Row groups read in parallel, rows in file order, 64-bit integers as numbers; columns the
  // file lacks (an older run's) are left out
  async function readGroups(name, groups, columns) {
    const { file, metadata, names } = await open(name);
    columns = columns.filter((c) => names.has(c));
    prefetch(file, groups, columns);
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
    // With `aliases`, an item's aliases match too (rows with `alias` set); an item found
    // more than once is listed once.
    async search(q, { limit = 40, maxGroups = 3, aliases = false } = {}) {
      const { groups } = await open("names.parquet");
      const columns = ["key", "label", "alias", "id", "description", "wikipedias", "coded"];
      const byPrefix = async (prefix) => (await readGroups("names.parquet",
        overlapping(groups, "key", prefix, prefix + "\uffff").slice(0, maxGroups), columns))
        .filter((r) => r.key.startsWith(prefix) && (aliases || r.alias == null));
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
      const once = new Set();
      return found.filter((r) => !once.has(r.id) && once.add(r.id)).slice(0, limit);
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

    // A feature's postings, as columns: `ids` (the number of each "Q…"), `units` (its weight
    // for the feature over its norm), and its `kinds`, those of posting i being
    // kinds[kindAt[i]] to kinds[kindAt[i + 1]]. Typed arrays, at about 17 bytes a posting:
    // one object per posting took ten times that, and a few items' worth of families ran a
    // browser tab out of memory. The latest are kept (filters re-rank the same postings), up
    // to KEEP_POSTINGS in all. With `limit`, the first `limit` as objects ({ id, unit, kinds }).
    // Older layouts store `weight` and `norm`, heaviest first, or a float `unit`; the current
    // one a 16-bit `unit16`, by id.
    async postings(feature, { limit = Infinity } = {}) {
      if (limit === Infinity && postingsOf.has(feature)) {
        const kept = postingsOf.get(feature);
        postingsOf.delete(feature);  // most recently used last
        postingsOf.set(feature, kept);
        return kept;
      }
      const { groups } = await open("postings.parquet");
      let hit = overlapping(groups, "feature", feature, feature);
      if (limit < Infinity) hit = hit.slice(0, 1);  // rank order: the first row group leads
      const read = readColumns("postings.parquet", hit,
        ["feature", "id", "unit16", "unit", "weight", "norm", "kinds"])
        .then((parts) => packPostings(parts, feature));
      if (limit < Infinity) {
        const p = await read;
        return Array.from({ length: Math.min(limit, p.length) }, (_, i) =>
          ({ id: `Q${p.ids[i]}`, unit: p.units[i], kinds: p.kindsOf(i) }));
      }
      postingsOf.set(feature, read);
      read.then(() => trimPostings(), () => postingsOf.delete(feature));
      return read;
    },
  };

  // The oldest postings dropped while more than KEEP_POSTINGS are kept
  async function trimPostings() {
    let total = 0;
    const sizes = [];
    for (const [f, p] of postingsOf) {
      const n = await Promise.race([p.then((x) => x.length, () => 0), 0]);
      sizes.push([f, n]);
      total += n;
    }
    for (const [f, n] of sizes) {
      if (total <= KEEP_POSTINGS || sizes.length < 2) break;
      postingsOf.delete(f);
      total -= n;
    }
  }

  // Row groups read in parallel as columns, without an object per row: for each group,
  // { column: values }, typed arrays where hyparquet decodes to them
  async function readColumns(name, groups, columns) {
    const { file, metadata, names } = await open(name);
    columns = columns.filter((c) => names.has(c));
    prefetch(file, groups, columns);
    return Promise.all(groups.map(async (g) => {
      const chunks = Object.fromEntries(columns.map((c) => [c, []]));
      await parquetRead({
        file, metadata, columns, compressors, rowStart: g.start, rowEnd: g.end,
        onChunk: ({ columnName, columnData, rowStart, rowEnd }) => {
          const from = Math.max(g.start, rowStart), to = Math.min(g.end, rowEnd);
          if (from < to) chunks[columnName].push([from, columnData.slice(from - rowStart, to - rowStart)]);
        },
      });
      const part = {};
      for (const c of columns) {
        const list = chunks[c].sort((a, b) => a[0] - b[0]).map(([, d]) => d);
        part[c] = list.length === 1 ? list[0] : list.flatMap((d) => Array.from(d));
      }
      return part;
    }));
  }
}

// The most postings kept between lookups, over all features (about 100 MB)
const KEEP_POSTINGS = 6_000_000;

// One feature's rows from row groups of postings, packed into typed arrays
function packPostings(parts, feature) {
  let n = 0, nk = 0;
  for (const p of parts) {
    for (let i = 0; i < p.feature.length; i++) {
      if (Number(p.feature[i]) !== feature || p.id[i] == null) continue;  // a property: no Q number
      n++;
      nk += p.kinds?.[i]?.length ?? 0;
    }
  }
  const ids = new Uint32Array(n), units = new Float32Array(n);
  const kindAt = new Uint32Array(n + 1), kinds = new Uint32Array(nk);
  let r = 0, k = 0;
  for (const p of parts) {
    for (let i = 0; i < p.feature.length; i++) {
      if (Number(p.feature[i]) !== feature || p.id[i] == null) continue;  // a property: no Q number
      const id = p.id[i];
      ids[r] = typeof id === "string" ? +id.slice(1) : Number(id);
      units[r] = p.unit16 ? Number(p.unit16[i]) / 65535
        : p.unit ? Number(p.unit[i]) : Number(p.weight[i]) / Number(p.norm[i]);
      kindAt[r] = k;
      for (const c of p.kinds?.[i] ?? []) kinds[k++] = Number(c);
      r++;
    }
  }
  kindAt[n] = k;
  return {
    length: n, ids, units, kindAt, kinds,
    kindsOf: (i) => Array.from(kinds.subarray(kindAt[i], kindAt[i + 1])),
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
  // Each candidate's dot product, which seed features it has (a bit each), and where its
  // kinds are (the first list it is in, and its row there), in typed arrays: hundreds of
  // thousands of candidates are common
  const self = +item.id.slice(1);
  const most = lists.reduce((n, p) => n + p.length, 0);
  const slot = new Map();
  const ids = new Uint32Array(most), dot = new Float64Array(most), mask = new Uint32Array(most);
  const from = new Uint8Array(most), row = new Uint32Array(most);
  let n = 0;
  seed.forEach(([, w], i) => {
    const p = lists[i];
    for (let r = 0; r < p.length; r++) {
      const id = p.ids[r];
      if (id === self) continue;
      let s = slot.get(id);
      if (s === undefined) {
        slot.set(id, s = n++);
        ids[s] = id;
        from[s] = i;
        row[s] = r;
      }
      dot[s] += p.units[r] * w;
      mask[s] |= 1 << i;
    }
  });
  // A light object per candidate; its id, kinds and shared features are worked out when read
  class Candidate {
    constructor(s) { this.s = s; this.similarity = dot[s] / item.norm; }
    get id() { return `Q${ids[this.s]}`; }
    get kinds() { return lists[from[this.s]].kindsOf(row[this.s]); }
    get shared() { return seed.map((_, i) => i).filter((i) => mask[this.s] & (1 << i)); }
  }
  const order = new Uint32Array(n).map((_, s) => s)
    .sort((a, b) => dot[b] - dot[a] || ids[a] - ids[b]);
  const near = [];
  for (const s of order) {
    const c = new Candidate(s);
    if (keep(c)) near.push(c);
    if (near.length >= limit) break;
  }
  return { seed, near };
}
