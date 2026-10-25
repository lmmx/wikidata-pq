// The benchmark history dashboard (docs/guide/bench-history.md): a hero chart of every
// figure at once (#bench-hero), then a line chart of each figure, from
// docs/assets/bench-history.json (scripts/bench_history.py), in plain SVG.
// Runs on any page with a #bench-history element, on load and on each of Material's
// instant navigations (document$).
(() => {
  "use strict";
  const SVG = "http://www.w3.org/2000/svg";
  const GROUPS = [
    ["measured", "Measured on the sample", "The bench's own workload (33,220 entities), nothing modelled. The first two rows are the end-to-end total (bench 11.0.0 on)."],
    ["eta", "Modelled release ETAs", "The sample's rates scaled to a release (bench/README.md, \"What the numbers mean\"). A dashed line is an earlier model's figure, recomputed by a later harness on the same runs. The first row is the total, the next three its stages. When a change moves work from one stage to another (shift-left: #134 and #148 moved work from the pipeline into the split, by design), the stage it moved to rises while the total falls: read a stage's rise beside the total's row."],
    ["memory", "Peak RSS", "Measured on the sample, except the two rows modelled at a release's size."],
    ["disk", "Disk", ""],
    ["stages", "CPU by stage", "When a change moves work from one stage to another, the stage it moved to rises while the total falls: #134 and #148 moved work from the pipeline into the split (shift-left, by design), and #140 moved the partitioning of a table split by key from each chunk to its group's close (partition down 80%, group merges up). Read a stage's rise beside the totals under \"Measured on the sample\"."],
    ["phases", "Where the CPU goes", "Each stage's own CPU time (bench/phases.py, bench 6.0.0 on): the largest stages of the latest run, by span. A stage renamed or split by a change starts a new line."],
  ];
  // Line styles by the harness's major version: the latest five each a categorical slot
  // (with a marker shape of its own, a second channel), the earlier ones muted
  // (SLOTS and OLDEST are set from the data: latest major -> slot 1, and so on)
  let SLOTS = {};
  let OLDEST = 0;
  const SHAPES = { 1: "circle", 2: "square", 3: "diamond", 4: "triangle", 5: "cross" };
  const VIEWS = [
    ["cpu", "One runner CPU"],
    ["absolute", "All runners (mixed)"],
    ["paired", "Paired change"],
    ["cumulative", "Cumulative change"],
  ];
  // The default view is one runner CPU, the one with the most runs: "All runners" averages
  // over whichever CPUs a commit drew, which differ by 20-40% in speed, so it moves with
  // the runner mix. The cumulative change starts by default from #133 (bench 12's first
  // record of its own, the campaign done).
  const SINCE_PR = 133;
  // (heroAll and heroHidden: the hero chart's range and the figures its legend hid)
  const state = { view: "cpu", cpu: null, from: 8, since: null, sinceSha: null, backfill: false, heroAll: false, heroHidden: [] };
  // (a new key: the earlier one kept "All runners" as most visitors' saved view)
  const STORE = "bench-history-2";
  let data = null;
  let root = null;
  let tip = null;

  const el = (tag, attrs = {}, parent = null, ns = null) => {
    const e = ns ? document.createElementNS(ns, tag) : document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined) continue;
      if (k === "text") e.textContent = v;
      else e.setAttribute(k, v);
    }
    if (parent) parent.appendChild(e);
    return e;
  };
  const svg = (tag, attrs, parent) => el(tag, attrs, parent, SVG);

  // Values and their formats
  const duration = (s) => {
    const a = Math.abs(s);
    if (a < 60) return `${s.toFixed(a < 10 ? 2 : 1)}s`;
    const t = Math.round(a);
    const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), sec = t % 60;
    const out = h ? `${h}h${String(m).padStart(2, "0")}m` : `${m}m${String(sec).padStart(2, "0")}s`;
    return s < 0 ? `−${out}` : out;
  };
  const size = (mb) => (Math.abs(mb) >= 1000 ? `${(mb / 1000).toFixed(Math.abs(mb) >= 10000 ? 1 : 2)} GB` : `${Math.round(mb)} MB`);
  // (unit "n": a count, such as files)
  const count = (v) => Math.round(v).toLocaleString("en");
  const fmt = (unit, v) => (unit === "%" ? `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(1)}%` : unit === "MB" ? size(v) : unit === "n" ? count(v) : duration(v));
  const fmtE = (unit, e) => (unit === "%" ? `${e.toFixed(1)}%` : unit === "MB" ? size(e) : unit === "n" ? count(e) : e < 60 ? `${e.toFixed(e < 1 ? 2 : 1)}s` : duration(e));
  const tickFmt = (unit, v) => {
    if (unit === "%") return `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v)}%`;
    if (unit === "MB") return v >= 1000 ? `${+(v / 1000).toFixed(2)} GB` : `${v} MB`;
    if (unit === "n") return count(v);
    if (v >= 60) return duration(v);
    return `${v}s`;
  };
  const niceTicks = (lo, hi, count = 4) => {
    if (!(hi > lo)) hi = lo + 1;
    const raw = (hi - lo) / count, mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((s) => s >= raw);
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
    return out;
  };

  const commitLabel = (c) => (c.pr ? `#${c.pr}` : c.sha.slice(0, 7));
  // The Actions runs a commit was measured in: as either arm of a paired record
  const REPO = "https://github.com/lmmx/wikidata-pq";
  const runUrl = (r) => (/^\d+$/.test(String(r)) ? `${REPO}/actions/runs/${r}` : String(r));
  const runsOf = (c) => [...new Set(data.records.filter((r) => r.run && (r.after === c || r.before === c)).map((r) => runUrl(r.run)))];
  const styleOf = (major) => {
    const slot = SLOTS[major];
    return slot ? { color: `var(--bh-s${slot})`, shape: SHAPES[slot] } : { color: "var(--bh-old)", shape: "circle" };
  };
  const lineName = (l) => {
    if (l.computed) return `bench ${l.model}'s model, recomputed by bench ${l.measured_by}`;
    return SLOTS[l.model] ? `bench ${l.model}` : `bench 1–${OLDEST} (each its own segment)`;
  };

  // The points a chart shows under the current view: [{line, pts: [{c, m, e, n}]}]
  function series(fig) {
    if (state.view === "paired") {
      const out = [];
      const by = {};
      for (const ch of fig.changes) {
        if (ch.bench < state.from) continue;
        if (ch.source === "backfill" && !state.backfill) continue;
        const k = ch.source === "backfill" ? "backfill" : ch.bench;
        (by[k] ||= []).push(ch);
      }
      // The backfill's reference (master's head as it ran) is its line's zero
      if (by.backfill) {
        for (const ref of new Set(by.backfill.map((p) => p.vs))) by.backfill.push({ c: ref, m: 0, e: 0, n: 0, ref: true, bench: by.backfill[0].bench });
        by.backfill.sort((a, b) => a.c - b.c);
      }
      for (const [k, pts] of Object.entries(by)) {
        const major = k === "backfill" ? 0 : +k;
        out.push({
          name: k === "backfill" ? `backfill: each older commit against ${[...new Set(pts.map((p) => commitLabel(data.commits[p.vs ?? p.c])))].join(", ")}, not a step` : SLOTS[major] ? `bench ${major}: against the commit before` : `bench 1–${OLDEST}: against the commit before`,
          style: k === "backfill" ? styleOf(pts[0].bench) : styleOf(major),
          joined: k === "backfill", key: `p${k}`, pts,
        });
      }
      // drawn oldest first, the backfill's line last (on top); the legend lists them reversed
      return out.sort((a, b) => (a.key === "pbackfill" ? 1 : b.key === "pbackfill" ? -1 : +a.key.slice(1) - +b.key.slice(1)));
    }
    if (state.view === "cumulative") {
      const ch = chain(fig, state.since);
      if (!ch.pts.length) return [];
      return [{ name: `master since ${commitLabel(data.commits[state.since])}: its merges' paired changes, chained`, style: { color: "var(--bh-s1)", shape: "circle" }, joined: true, key: "cumulative", pts: ch.pts }];
    }
    const ci = state.view === "cpu" ? data.cpus.indexOf(state.cpu) : -1;
    const out = [];
    for (const l of fig.lines) {
      if (l.measured_by < state.from) continue;
      let pts = l.points;
      if (ci >= 0) pts = pts.filter((p) => p.k && p.k[ci]).map((p) => ({ c: p.c, m: p.k[ci][0], e: p.k[ci][1], n: p.k[ci][2] }));
      if (!pts.length) continue;
      out.push({ name: lineName(l), style: styleOf(l.model), dashed: l.computed, joined: true, key: `${l.model}-${l.measured_by}-${l.computed}`, pts, line: l });
    }
    return out;
  }

  // Master's merges' own paired changes (the ledger's records, each against the commit
  // before it on the same runners; never the backfill, whose changes are against #132)
  // chained from the commit `since`: the product of the ratios, its 95% interval each
  // step's half-width (on the log scale) added in quadrature, the steps' runs independent.
  // Commits with no record of their own (the bench skips docs-only pushes) count as no
  // change; a record without the figure ends the chain there.
  const ledger = () => data.records.filter((r) => r.source === "ledger").sort((a, b) => a.after - b.after);
  function chain(fig, since) {
    const by = new Map(fig.changes.filter((c) => c.source === "ledger").map((c) => [`${c.c}:${c.vs}`, c]));
    const pts = [];
    let L = 0, V = 0, steps = 0, broken = null;
    for (const r of ledger()) {
      if (r.after <= since) continue;
      const c = by.get(`${r.after}:${r.before}`);
      if (!c) { broken = r.after; break; }
      if (!pts.length) pts.push({ c: since, m: 0, e: 0, lo: 0, hi: 0, steps: 0, since, start: true });
      L += Math.log1p(c.m / 100);
      V += (c.e / (100 + c.m)) ** 2;
      steps += 1;
      const h = Math.sqrt(V);
      pts.push({ c: r.after, m: 100 * Math.expm1(L), e: 0, lo: 100 * Math.expm1(L - h), hi: 100 * Math.expm1(L + h), steps, since });
    }
    return { pts, broken };
  }
  // A cross-check without pairing: one runner CPU's means, from `since` to the last commit,
  // spliced where a harness's major version starts (at the commit both versions measured:
  // the ratio within each version's line, multiplied). No interval: each end is one
  // commit's mean on one CPU.
  function spliced(fig, since, cpu) {
    const ci = String(data.cpus.indexOf(cpu));
    const lines = fig.lines.filter((l) => !l.computed).sort((a, b) => a.measured_by - b.measured_by)
      .map((l) => new Map(l.points.filter((p) => p.k && p.k[ci]).map((p) => [p.c, p.k[ci][0]])));
    let i = lines.map((l) => l.has(since)).lastIndexOf(true);
    if (i < 0) return null;
    let at = since, ratio = 1;
    for (let j = i + 1; j < lines.length; j++) {
      const join = [...lines[j].keys()].filter((c) => c >= at && lines[i].has(c)).sort((a, b) => a - b)[0];
      if (join === undefined) continue;
      ratio *= lines[i].get(join) / lines[i].get(at);
      at = join; i = j;
    }
    const last = Math.max(...lines[i].keys());
    return last > since ? { m: 100 * (ratio * lines[i].get(last) / lines[i].get(at) - 1), c: last } : null;
  }
  const loOf = (p) => (p.lo ?? p.m - p.e);
  const hiOf = (p) => (p.hi ?? p.m + p.e);
  const hasRange = (p) => (p.lo !== undefined ? p.hi !== p.lo : p.e > 0);
  const valueText = (unit, p) =>
    p.lo !== undefined
      ? `${fmt(unit, p.m)}${p.hi !== p.lo ? ` (95%: ${fmt(unit, p.lo)} to ${fmt(unit, p.hi)})` : ""}`
      : `${fmt(unit, p.m)}${p.e ? ` ± ${fmtE(unit, p.e)}` : ""}`;
  const ofText = (p, long) =>
    p.start ? "the start" : p.steps !== undefined ? `${p.steps} merge${p.steps > 1 ? "s" : ""} since ${commitLabel(data.commits[p.since])}`
      : p.ref ? "the reference" : p.vs !== undefined ? `${p.n} round${p.n > 1 ? "s" : ""} ${long ? "against" : "vs"} ${commitLabel(data.commits[p.vs])}` : `${p.n} run${p.n > 1 ? "s" : ""}`;

  function marker(shape, x, y, color, parent) {
    const r = 4;
    let m;
    if (shape === "square") m = svg("rect", { x: x - r, y: y - r, width: 2 * r, height: 2 * r, rx: 1 }, parent);
    else if (shape === "diamond") m = svg("path", { d: `M${x} ${y - r - 1}L${x + r + 1} ${y}L${x} ${y + r + 1}L${x - r - 1} ${y}Z` }, parent);
    else if (shape === "triangle") m = svg("path", { d: `M${x} ${y - r - 1}L${x + r + 1} ${y + r}L${x - r - 1} ${y + r}Z` }, parent);
    else if (shape === "cross") m = svg("path", { d: `M${x - 1.6} ${y - r}h3.2v${r - 1.6}h${r - 1.6}v3.2h${1.6 - r}v${r - 1.6}h-3.2v${1.6 - r}h${1.6 - r}v-3.2h${r - 1.6}Z` }, parent);
    else m = svg("circle", { cx: x, cy: y, r }, parent);
    m.setAttribute("class", "bh-marker");
    m.style.fill = color;
    return m;
  }

  function chart(fig, card) {
    const lines = series(fig);
    const unit = state.view === "paired" || state.view === "cumulative" ? "%" : fig.unit;
    const body = card.querySelector(".bh-body");
    body.textContent = "";
    if (!lines.length) {
      el("p", { class: "bh-empty", text: state.view === "cpu" ? `No run of this figure on an ${state.cpu} runner in the versions shown.` : state.view === "cumulative" ? `No record of master's measures this figure after ${commitLabel(data.commits[state.since])}.` : "Not measured in the versions shown." }, body);
      card.querySelector(".bh-legend").textContent = "";
      card.querySelector(".bh-latest").textContent = "";
      return;
    }
    const cs = [...new Set(lines.flatMap((l) => l.pts.map((p) => p.c)))].sort((a, b) => a - b);
    const xi = new Map(cs.map((c, i) => [c, i]));
    const W = Math.max(280, body.clientWidth || 600), H = 230;
    const M = { l: 56, r: 14, t: 18, b: 34 };
    // The means set the scale; an interval reaches at most half as far again (one-run
    // records of bench 1.x have huge ones), and is clipped to the plot
    let lo = Infinity, hi = -Infinity, mlo = Infinity, mhi = -Infinity;
    for (const l of lines) for (const p of l.pts) { mlo = Math.min(mlo, p.m); mhi = Math.max(mhi, p.m); }
    const span = Math.max(mhi - mlo, Math.abs(mhi) * 0.05, 1e-9);
    for (const l of lines) for (const p of l.pts) { lo = Math.min(lo, Math.max(loOf(p), mlo - span / 2)); hi = Math.max(hi, Math.min(hiOf(p), mhi + span / 2)); }
    if (unit !== "%") lo = 0;
    else { lo = Math.min(0, lo); hi = Math.max(0, hi); }
    const ticks = niceTicks(lo, hi);
    lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]);
    const x = (c) => M.l + (cs.length === 1 ? (W - M.l - M.r) / 2 : (xi.get(c) * (W - M.l - M.r)) / (cs.length - 1));
    const y = (v) => M.t + ((hi - v) * (H - M.t - M.b)) / (hi - lo || 1);

    const names = [...new Set(lines.map((l) => l.name))];
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": `${fig.label}: ${names.join(", ")}` }, body);
    const clipId = `bh-clip-${fig.key.replace(/[^\w-]/g, "_")}`;
    svg("rect", { x: 0, y: M.t, width: W, height: H - M.t - M.b }, svg("clipPath", { id: clipId }, svg("defs", {}, s)));
    const grid = svg("g", { class: "bh-grid" }, s);
    for (const t of ticks) {
      svg("line", { x1: M.l, x2: W - M.r, y1: y(t), y2: y(t), class: t === 0 ? "bh-base" : "bh-gridline" }, grid);
      svg("text", { x: M.l - 6, y: y(t) + 4, "text-anchor": "end", class: "bh-tick", text: tickFmt(unit, t) }, grid);
    }
    // x labels: a few commits, evenly spread
    // x labels: the last commit, the milestones (each with a tick), then others, where they fit
    const placed = [];
    const fits = (px) => placed.every((q) => Math.abs(q - px) >= 34);
    const label = (c) => { placed.push(x(c)); svg("text", { x: x(c), y: H - M.b + 16, "text-anchor": "middle", class: "bh-tick", text: commitLabel(data.commits[c]) }, grid); };
    label(cs[cs.length - 1]);
    for (const c of cs.slice().reverse()) {
      if (!data.commits[c].milestone) continue;
      svg("line", { x1: x(c), x2: x(c), y1: H - M.b, y2: H - M.b + 4, class: "bh-base" }, grid);
      if (fits(x(c))) label(c);
    }
    if (fits(x(cs[0]))) label(cs[0]);
    // A harness's major version's first commit here: a hairline and its number
    if (state.view === "absolute" || state.view === "cpu") {
      const shown = new Set(lines.filter((l) => !l.line.computed).map((l) => l.line.model));
      const starts = new Map(Object.entries(data.starts || {}).map(([m, c]) => [+m, c]).filter(([m, c]) => shown.has(m) && xi.has(c)));
      const minMajor = Math.min(...shown);
      let lastLabel = -Infinity;
      for (const [mj, c0] of [...starts].sort((a, b) => a[1] - b[1])) {
        if (mj === minMajor) continue;
        svg("line", { x1: x(c0), x2: x(c0), y1: M.t - 6, y2: H - M.b, class: "bh-break" }, grid);
        if (x(c0) - lastLabel < 22) continue;
        lastLabel = x(c0);
        svg("text", { x: x(c0) + 3, y: M.t - 8, class: "bh-tick bh-break-label", text: `b${mj}` }, grid);
      }
    }

    // Bands, then lines, then markers
    const g = svg("g", { "clip-path": `url(#${clipId})` }, s);
    // Markers where the commits are far enough apart to show them, dots where not
    const gap = cs.length > 1 ? (W - M.l - M.r) / (cs.length - 1) : 99;
    for (const l of lines) {
      if (!l.joined || l.pts.length < 2) continue;
      if (l.pts.some(hasRange)) {
        const top = l.pts.map((p) => `${x(p.c)},${y(hiOf(p))}`), bot = l.pts.slice().reverse().map((p) => `${x(p.c)},${y(loOf(p))}`);
        const band = svg("polygon", { points: top.concat(bot).join(" "), class: "bh-band" }, g);
        band.style.fill = l.style.color;
      }
    }
    for (const l of lines) {
      if (!l.joined || l.pts.length < 2) continue;
      const path = svg("polyline", { points: l.pts.map((p) => `${x(p.c)},${y(p.m)}`).join(" "), class: l.dashed ? "bh-line bh-dashed" : "bh-line" }, g);
      path.style.stroke = l.style.color;
    }
    for (const l of lines) {
      for (const p of l.pts) {
        if (!l.joined || l.pts.length === 1) {
          const w = svg("line", { x1: x(p.c), x2: x(p.c), y1: y(loOf(p)), y2: y(hiOf(p)), class: "bh-whisker" }, g);
          w.style.stroke = l.style.color;
        }
        if (gap >= 9 || l.pts.length === 1 || !l.joined) marker(l.style.shape, x(p.c), y(p.m), l.style.color, g);
      }
    }

    // The hover layer: a crosshair snapping to the nearest commit, one tooltip for every line
    const cross = svg("line", { y1: M.t, y2: H - M.b, class: "bh-cross", visibility: "hidden" }, s);
    const hit = svg("rect", { x: M.l - 10, y: 0, width: W - M.l - M.r + 20, height: H, class: "bh-hit", tabindex: 0 }, s);
    const show = (c) => {
      cross.setAttribute("x1", x(c)); cross.setAttribute("x2", x(c)); cross.setAttribute("visibility", "visible");
      tooltip(fig, lines, unit, c, s, x(c));
    };
    const nearest = (px) => cs.reduce((a, c) => (Math.abs(x(c) - px) < Math.abs(x(a) - px) ? c : a), cs[0]);
    hit.addEventListener("pointermove", (ev) => {
      const r = s.getBoundingClientRect();
      show(nearest(((ev.clientX - r.left) * W) / r.width));
    });
    hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tip.hidden = true; });
    let focusI = cs.length - 1;
    hit.addEventListener("focus", () => show(cs[focusI]));
    hit.addEventListener("blur", () => { cross.setAttribute("visibility", "hidden"); tip.hidden = true; });
    hit.addEventListener("keydown", (ev) => {
      if (ev.key === "ArrowLeft") focusI = Math.max(0, focusI - 1);
      else if (ev.key === "ArrowRight") focusI = Math.min(cs.length - 1, focusI + 1);
      else return;
      ev.preventDefault();
      show(cs[focusI]);
    });

    // Legend (every line), and the latest value of the headline line
    const legend = card.querySelector(".bh-legend");
    legend.textContent = "";
    const seen = new Set();
    for (const l of lines.slice().reverse()) {
      if (seen.has(l.name)) continue;
      seen.add(l.name);
      const item = el("span", { class: "bh-key" }, legend);
      const k = svg("svg", { width: 26, height: 10, "aria-hidden": "true" }, item);
      if (l.joined) {
        const ln = svg("line", { x1: 1, x2: 25, y1: 5, y2: 5, class: l.dashed ? "bh-line bh-dashed" : "bh-line" }, k);
        ln.style.stroke = l.style.color;
      }
      marker(l.style.shape, 13, 5, l.style.color, k);
      el("span", { text: l.name }, item);
    }
    const latest = card.querySelector(".bh-latest");
    const own = lines.filter((l) => !l.dashed && l.key !== "pbackfill");
    const head = (own.length ? own : lines).reduce((a, l) => (l.pts[l.pts.length - 1].c >= a.pts[a.pts.length - 1].c ? l : a));
    const lp = head.pts[head.pts.length - 1];
    latest.textContent = `${valueText(unit, lp)} at ${commitLabel(data.commits[lp.c])}`;

    // The table view: every point
    const det = card.querySelector("details");
    const fill = () => {
      const tbody = det.querySelector("tbody");
      tbody.textContent = "";
      for (const c of cs.slice().reverse()) {
        for (const l of lines) {
          const p = l.pts.find((q) => q.c === c);
          if (!p) continue;
          const cm = data.commits[c];
          const tr = el("tr", {}, tbody);
          const where = el("td", {}, tr);
          el("a", { href: `${REPO}/commit/${cm.sha}`, text: cm.sha.slice(0, 7) }, where);
          if (cm.pr) {
            where.appendChild(document.createTextNode(" ("));
            el("a", { href: `${REPO}/pull/${cm.pr}`, text: `#${cm.pr}` }, where);
            where.appendChild(document.createTextNode(")"));
          }
          el("td", { text: cm.date.slice(0, 10) }, tr);
          el("td", { text: l.name }, tr);
          el("td", { text: valueText(unit, p), class: "bh-num" }, tr);
          el("td", { text: ofText(p, true), class: "bh-num" }, tr);
          const ran = el("td", {}, tr);
          runsOf(c).forEach((u, i) => {
            if (i) ran.appendChild(document.createTextNode(" "));
            el("a", { href: u, text: u.split("/").pop() }, ran);
          });
        }
      }
    };
    det.ontoggle = () => det.open && fill();
    if (det.open) fill();
  }

  function tooltip(fig, lines, unit, c, s, px) {
    const cm = data.commits[c];
    tip.textContent = "";
    el("div", { class: "bh-tip-title", text: `${cm.sha.slice(0, 7)}${cm.pr ? ` · #${cm.pr}` : ""} · ${cm.date.slice(0, 10)}${cm.milestone ? " · milestone" : ""}` }, tip);
    el("div", { class: "bh-tip-subject", text: cm.subject.length > 90 ? `${cm.subject.slice(0, 88)}…` : cm.subject }, tip);
    for (const l of lines) {
      const p = l.pts.find((q) => q.c === c);
      if (!p) continue;
      const row = el("div", { class: "bh-tip-row" }, tip);
      const k = svg("svg", { width: 14, height: 10, "aria-hidden": "true" }, row);
      const ln = svg("line", { x1: 1, x2: 13, y1: 5, y2: 5, class: l.dashed ? "bh-line bh-dashed" : "bh-line" }, k);
      ln.style.stroke = l.style.color;
      el("strong", { text: valueText(unit, p) }, row);
      el("span", { text: ` ${l.name}, ${ofText(p, false)}` }, row);
      if (state.view === "absolute" && l.line) {
        const full = l.line.points.find((q) => q.c === c);
        if (full && full.k) {
          const parts = Object.entries(full.k).map(([i, v]) => `${data.cpus[i]} ${fmt(unit, v[0])} (${v[2]})`);
          el("div", { class: "bh-tip-cpu", text: parts.join(" · ") }, tip);
        }
      }
    }
    tip.hidden = false;
    const r = s.getBoundingClientRect(), host = root.getBoundingClientRect();
    const left = r.left - host.left + (px * r.width) / s.viewBox.baseVal.width;
    const wide = tip.offsetWidth;
    tip.style.left = `${Math.max(0, Math.min(host.width - wide, left + 12 > host.width - wide ? left - wide - 12 : left + 12))}px`;
    tip.style.top = `${r.top - host.top + 8}px`;
  }

  function controls() {
    const bar = el("div", { class: "bh-controls" }, root);
    const views = el("div", { class: "bh-seg", role: "radiogroup", "aria-label": "View" }, bar);
    for (const [v, label] of VIEWS) {
      const b = el("button", { type: "button", role: "radio", "aria-checked": String(state.view === v), text: label }, views);
      b.addEventListener("click", () => { state.view = v; save(); render(); });
    }
    const cpuL = el("label", { class: "bh-field" }, bar);
    el("span", { text: "Runner CPU" }, cpuL);
    const cpuS = el("select", { disabled: state.view === "cpu" ? null : "" }, cpuL);
    for (const c of data.cpus) el("option", { value: c, text: c, ...(c === state.cpu ? { selected: "" } : {}) }, cpuS);
    cpuS.addEventListener("change", () => { state.cpu = cpuS.value; save(); render(); });
    const fromL = el("label", { class: "bh-field" }, bar);
    el("span", { text: "Harness versions" }, fromL);
    const fromS = el("select", { disabled: state.view === "cumulative" ? "" : null }, fromL);
    const latest = Math.max(...Object.keys(SLOTS).map(Number));
    const froms = [[1, "all (1.0.0 on)"], [7, "7.0.0 on (the split)"], [8, "8.0.0 on"], [latest - 1, `${latest - 1}.0.0 on`], [latest, `${latest}.0.0 only`]];
    for (const [v, label] of froms.filter(([v], i) => froms.findIndex(([w]) => w === v) === i))
      el("option", { value: v, text: label, ...(v === state.from ? { selected: "" } : {}) }, fromS);
    fromS.addEventListener("change", () => { state.from = +fromS.value; save(); render(); });
    const sinceL = el("label", { class: "bh-field" }, bar);
    el("span", { text: "Cumulative since" }, sinceL);
    const sinceS = el("select", {}, sinceL);
    for (const c of sinceOptions()) {
      const cm = data.commits[c];
      el("option", { value: c, text: `${commitLabel(cm)} · ${cm.date.slice(0, 10)}`, ...(c === state.since ? { selected: "" } : {}) }, sinceS);
    }
    sinceS.addEventListener("change", () => { state.since = +sinceS.value; state.sinceSha = data.commits[state.since].sha; save(); render(); });
    if (state.view === "paired") {
      const bfL = el("label", { class: "bh-field" }, bar);
      const bf = el("input", { type: "checkbox", ...(state.backfill ? { checked: "" } : {}) }, bfL);
      el("span", { text: "Backfill (older commits against #132)" }, bfL);
      bf.addEventListener("change", () => { state.backfill = bf.checked; save(); render(); });
    }
  }

  // The commits a cumulative change can start from: the ledger's, from bench 8 on, newest first
  function sinceOptions() {
    const out = new Set();
    for (const r of ledger()) {
      if (major(r.bench) < 8) continue;
      out.add(r.before);
      out.add(r.after);
    }
    out.delete(data.commits.length - 1);
    return [...out].sort((a, b) => b - a);
  }
  const major = (v) => +String(v).split(".")[0];

  // What the view shows, in a line under the controls
  function viewNote() {
    if (state.view === "cpu") {
      const counts = runCounts();
      const total = Object.values(counts).reduce((a, b) => a + b, 0);
      const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
      return `Only the runs on ${state.cpu} runners (${counts[state.cpu] || 0} of the ${total} runs of the release ETA; ${top && top[0] === state.cpu ? "the CPU with the most runs, on which every commit from bench 2 on was measured" : `${top[0]} has the most`}). GitHub's runner CPUs differ by 20–40% in speed, so one CPU's runs compare like with like; the bands are wider for it, from fewer runs.`;
    }
    if (state.view === "absolute") return "Every run of each commit, whatever CPU it drew. The CPUs differ by 20–40% in speed, so a commit that happened to draw slower runners jumps by itself: the hover lists each CPU's mean. \"One runner CPU\" compares like with like.";
    if (state.view === "paired") return `Each merge's change against the commit before it, both run side by side on the same runners, so the runner mix cannot move it.${state.backfill ? " The backfill's points are older commits each against #132, not steps from their predecessor: a positive one means that commit was slower than #132." : " The backfill (older commits against #132, not steps) is left out: tick it to show it."}`;
    return `Master's merges' paired changes chained from ${commitLabel(data.commits[state.since])}: each a ledger record against the commit before it on the same runners (never the backfill), the ratios multiplied, the 95% band each step's half-width added in quadrature on the log scale (the steps' runs independent). Commits the bench skipped (docs-only) count as no change.`;
  }
  // Each runner CPU's runs, over the release ETA's own lines (its most-measured figure)
  function runCounts() {
    const fig = data.figures.find((f) => f.key === "eta") || data.figures[0];
    const out = {};
    for (const l of fig.lines) {
      if (l.computed) continue;
      for (const p of l.points) for (const [i, v] of Object.entries(p.k || {})) out[data.cpus[i]] = (out[data.cpus[i]] || 0) + v[2];
    }
    return out;
  }

  // The headline figures' cumulative change since state.since, as a table above the charts
  const SUMMARY = ["eta", "measured_cpu", "measured_wall", "peak_sum_rss_mb", "disk_peak_mb", "split_eta", "pipeline_eta", "finalise_eta"];
  function summaryTable() {
    const wrap = el("div", { class: "bh-summary" }, root);
    const head = data.commits[data.commits.length - 1];
    el("h2", { class: "bh-group", id: "bench-since", text: `Master since ${commitLabel(data.commits[state.since])}` }, wrap);
    el("p", { class: "bh-note", text: `Each figure's change from ${commitLabel(data.commits[state.since])} to ${commitLabel(head)}: master's merges' own paired changes (the ledger's records, each against the commit before it on the same runners; not the backfill) chained, with the 95% interval of each step added in quadrature on the log scale. The last column is a cross-check without pairing: ${state.cpu}'s means at the two ends, spliced where a harness version starts (no interval: one commit's mean each). The split's ETA rising while the pipeline's falls is the shift-left (#134, #148): work moved, the total fell. "Cumulative change" charts it.` }, wrap);
    const table = el("table", {}, el("div", { style: "overflow-x:auto" }, wrap));
    const hr = el("tr", {}, el("thead", {}, table));
    for (const h of ["figure", "change", "95% interval", "merges", "to", `${state.cpu} alone`]) el("th", { text: h }, hr);
    const tbody = el("tbody", {}, table);
    for (const key of SUMMARY) {
      const fig = data.figures.find((f) => f.key === key);
      if (!fig) continue;
      const { pts, broken } = chain(fig, state.since);
      const tr = el("tr", {}, tbody);
      el("td", { text: fig.label }, tr);
      if (!pts.length) {
        el("td", { text: "not measured", colspan: 5 }, tr);
        continue;
      }
      const p = pts[pts.length - 1];
      el("strong", { text: fmt("%", p.m) }, el("td", { class: "bh-num" }, tr));
      el("td", { class: "bh-num", text: `${fmt("%", p.lo)} to ${fmt("%", p.hi)}` }, tr);
      el("td", { class: "bh-num", text: String(p.steps) }, tr);
      el("td", { class: "bh-num", text: `${commitLabel(data.commits[p.c])}${broken !== null ? ` (not measured at ${commitLabel(data.commits[broken])})` : ""}` }, tr);
      const one = spliced(fig, state.since, state.cpu);
      el("td", { class: "bh-num", text: one ? `${fmt("%", one.m)}${one.c !== p.c ? ` (to ${commitLabel(data.commits[one.c])})` : ""}` : "–" }, tr);
    }
  }

  const save = () => { try { localStorage.setItem(STORE, JSON.stringify(state)); } catch (e) { /* none */ } };
  const restore = () => {
    try { Object.assign(state, JSON.parse(localStorage.getItem(STORE) || "{}")); } catch (e) { /* none */ }
    if (!VIEWS.some(([v]) => v === state.view)) state.view = "cpu";
    state.heroAll = state.heroAll === true;
    if (!Array.isArray(state.heroHidden)) state.heroHidden = [];
  };

  // The hero chart (#bench-hero, first on the page): every figure on one chart, one point
  // per master commit, each line on its own y-scale (its own range over the commits shown,
  // padded), so every figure's shape shows at once; the left axis is the highlighted
  // line's. A point is the commit's mean on the runner CPU chosen below (state.cpu: the
  // runner CPUs differ by 20-40% in speed, so never mixed), under the newest harness
  // version that measured it (its own figure, not one recomputed for an earlier model).
  const HERO_FROM = 12;
  const HERO_DEFAULT = "eta";
  const hero = { pinned: null, hover: null, commit: null };
  let heroRoot = null;
  const groupColor = (g) => {
    const i = GROUPS.findIndex(([k]) => k === g);
    return i >= 0 ? `var(--bh-s${i + 1})` : "var(--bh-old)";
  };
  const groupTitle = (g) => (GROUPS.find(([k]) => k === g) || [g, g])[1];
  // Ticks for a line's own axis: whole minutes and hours for durations
  const TIME_STEPS = [30, 60, 120, 300, 600, 900, 1200, 1800, 3600, 7200, 10800, 21600, 43200, 86400];
  function heroScale(unit, lo, hi, count) {
    if (!(hi > lo)) hi = lo + 1;
    const raw = (hi - lo) / count;
    let step;
    if (unit === "s" && raw >= 25) step = TIME_STEPS.find((t) => t >= raw) || TIME_STEPS[TIME_STEPS.length - 1];
    else {
      const mag = 10 ** Math.floor(Math.log10(raw));
      step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((t) => t >= raw);
    }
    const a = Math.floor(lo / step + 1e-9) * step, b = Math.ceil(hi / step - 1e-9) * step;
    const ticks = [];
    for (let v = a; v <= b + step * 1e-9; v += step) ticks.push(+v.toPrecision(12));
    return { lo: ticks[0], hi: ticks[ticks.length - 1], ticks };
  }
  const heroTick = (unit, v) => {
    if (unit !== "s") return tickFmt(unit, v);
    if (v < 60) return `${v}s`;
    if (v % 3600 === 0) return `${v / 3600}h`;
    if (v >= 3600 && v % 60 === 0) return `${Math.floor(v / 3600)}h${String((v % 3600) / 60).padStart(2, "0")}m`;
    if (v % 60 === 0) return `${v / 60}m`;
    return duration(v);
  };

  // The commits shown and each figure's value at each: [{f, vals: Map(commit -> {v, bench})}]
  function heroData() {
    const ci = String(data.cpus.indexOf(state.cpu));
    const shown = data.commits.map((c, i) => i).filter((i) => state.heroAll || data.commits[i].benches.some((b) => b >= HERO_FROM));
    const keep = new Set(shown);
    const figs = [];
    for (const f of data.figures) {
      const vals = new Map();
      // oldest harness first, so the newest that measured a commit has the last word
      for (const l of f.lines.filter((q) => !q.computed).sort((a, b) => a.measured_by - b.measured_by))
        for (const p of l.points) if (keep.has(p.c) && p.k && p.k[ci]) vals.set(p.c, { v: p.k[ci][0], bench: l.measured_by });
      if (vals.size) figs.push({ f, vals });
    }
    const used = new Set(figs.flatMap((h) => [...h.vals.keys()]));
    return { cs: shown.filter((c) => used.has(c)), figs };
  }

  function drawHero() {
    if (!heroRoot || !heroRoot.isConnected) return;
    // a redraw keeps keyboard focus on the control that had it
    const ae = document.activeElement;
    const refocus = ae && heroRoot.contains(ae) ? ae.getAttribute("data-hero-id") : null;
    heroRoot.textContent = "";
    const { cs, figs } = heroData();
    const hidden = new Set(state.heroHidden);
    const vis = figs.filter((h) => !hidden.has(h.f.key));
    const byKey = new Map(vis.map((h) => [h.f.key, h]));
    if (hero.pinned && !byKey.has(hero.pinned)) hero.pinned = null;
    if (hero.hover && !byKey.has(hero.hover)) hero.hover = null;
    if (!cs.includes(hero.commit)) hero.commit = null;
    const current = () => hero.pinned || hero.hover || (byKey.has(HERO_DEFAULT) ? HERO_DEFAULT : vis.length ? vis[0].f.key : null);

    const head = el("div", { class: "bh-hero-head" }, heroRoot);
    el("h2", { class: "bh-hero-title", id: "bench-every-figure", text: "Every figure, every commit" }, head);
    const bar = el("div", { class: "bh-controls" }, head);
    const seg = el("div", { class: "bh-seg", role: "radiogroup", "aria-label": "Commits shown" }, bar);
    for (const [v, label] of [[false, `Bench ${HERO_FROM} on`], [true, "All history"]]) {
      const b = el("button", { type: "button", role: "radio", "aria-checked": String(state.heroAll === v), text: label, "data-hero-id": `range-${v}` }, seg);
      b.addEventListener("click", () => { state.heroAll = v; save(); drawHero(); });
    }
    const all = el("button", { type: "button", class: "bh-btn", text: "Show all", "data-hero-id": "show-all", ...(hidden.size ? {} : { disabled: "" }) }, bar);
    all.addEventListener("click", () => { state.heroHidden = []; save(); drawHero(); });
    el("span", { class: "bh-field", text: `Runner: ${state.cpu}` }, bar);
    const focusP = el("p", { class: "bh-hero-focus" }, heroRoot);
    const body = el("div", { class: "bh-hero-chart" }, heroRoot);
    const link = el("p", { class: "bh-hero-commit" }, heroRoot);
    const legend = el("div", { class: "bh-hero-legend" }, heroRoot);
    el("p", { class: "bh-note", text: `Each line is scaled to its own range over the commits shown, so every figure's shape is visible at once; the axis on the left is the highlighted line's. Point at a line to highlight it, click to keep it highlighted and click again to release it (on the keyboard: ←/→ a commit, ↑/↓ a line, Enter to keep). Colour is the figure's group; the legend's entries hide and show figures. A point is the commit's mean on ${state.cpu} runners (the runner CPU chosen below: GitHub's runner CPUs differ by 20–40% in speed, so they are not mixed), from the newest harness version that measured it; a dashed rule marks a new bench version, where a figure's model can change. Shift-left: when a change moves work from one stage to another (#134 and #148 moved work from the pipeline into the split, by design), the line of the stage it moved to rises while the total's falls, so read a stage's rise beside the total.` }, heroRoot);
    const tipH = el("div", { class: "bh-tip bh-hero-tip", hidden: "" }, heroRoot);

    const refocusNow = () => {
      if (!refocus) return;
      const t = heroRoot.querySelector(`[data-hero-id="${CSS.escape(refocus)}"]`);
      // ("Show all" disables itself once used: focus moves to the range control)
      if (t && !t.disabled) t.focus();
      else heroRoot.querySelector('[data-hero-id^="range-"][aria-checked="true"]').focus();
    };
    if (!cs.length || !vis.length) {
      el("p", { class: "bh-empty", text: figs.length ? "Every figure is hidden: \"Show all\" brings them back." : `No run on an ${state.cpu} runner in the commits shown.` }, body);
      heroLegend(legend, figs, hidden, () => null, () => {});
      refocusNow();
      return;
    }

    const W = Math.max(300, Math.round(body.clientWidth) || 700), H = W < 560 ? 340 : 440;
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": `${vis.length} bench figures over ${cs.length} master commits on ${state.cpu} runners, each line on its own scale` }, body);
    for (const h of vis) {
      const vs = [...h.vals.values()].map((o) => o.v);
      const lo = Math.min(...vs), hi = Math.max(...vs);
      const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.05 || 1;
      h.sc = heroScale(h.f.unit, lo >= 0 ? Math.max(0, lo - pad) : lo - pad, hi + pad, H < 400 ? 4 : 5);
    }
    // The left margin fits the widest tick label of any line, so the plot never shifts
    const meas = svg("text", { class: "bh-tick bh-hero-tick", x: 0, y: -50 }, s);
    const textW = (t) => { meas.textContent = t; return meas.getComputedTextLength(); };
    let tw = 24;
    for (const h of vis) for (const t of h.sc.ticks) tw = Math.max(tw, textW(heroTick(h.f.unit, t)));
    const M = { l: Math.ceil(tw) + 14, r: 12, t: 26, b: 30 };
    const inner = W - M.l - M.r, ih = H - M.t - M.b;
    const xi = new Map(cs.map((c, i) => [c, i]));
    const pad = cs.length > 1 ? Math.min(14, inner / (cs.length - 1) / 2) : 0;
    const x = (c) => (cs.length === 1 ? M.l + inner / 2 : M.l + pad + (xi.get(c) * (inner - 2 * pad)) / (cs.length - 1));
    const y = (h, v) => M.t + ((h.sc.hi - v) * ih) / (h.sc.hi - h.sc.lo || 1);

    const gridG = svg("g", {}, s);
    const deco = svg("g", {}, s);
    svg("line", { x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, class: "bh-base" }, deco);
    // A new bench version: a dashed rule before the first commit it measured
    let lastRule = -Infinity;
    for (const [m, c0] of Object.entries(data.starts || {}).map(([k, c]) => [+k, c]).sort((a, b) => a[1] - b[1])) {
      const i = xi.get(c0);
      if (!i) continue;
      const xx = (x(cs[i - 1]) + x(c0)) / 2;
      svg("line", { x1: xx, x2: xx, y1: M.t - 6, y2: H - M.b, class: "bh-hero-rule" }, deco);
      const text = state.heroAll || W < 560 ? `b${m}` : `bench ${m}`;
      const w = textW(text);
      if (xx - lastRule < w + 8) continue;
      lastRule = xx;
      svg("text", { x: Math.min(xx + 3, W - w), y: M.t - 9, class: "bh-tick", text }, deco);
    }
    // x labels, each a link to its pull request (or commit): the last, then others where they fit
    const placed = [];
    for (const c of cs.slice().reverse()) {
      const cm = data.commits[c];
      if (state.heroAll && cm.milestone) svg("line", { x1: x(c), x2: x(c), y1: H - M.b, y2: H - M.b + 4, class: "bh-base" }, deco);
      const text = commitLabel(cm), w = textW(text);
      const lx = Math.max(M.l + w / 2, Math.min(W - w / 2, x(c)));
      if (!placed.every((q) => Math.abs(q - lx) >= w + 12)) continue;
      placed.push(lx);
      const a = svg("a", { href: cm.pr ? `${REPO}/pull/${cm.pr}` : `${REPO}/commit/${cm.sha}`, class: "bh-hero-xlink", "aria-label": `${text}: ${cm.subject}` }, deco);
      svg("text", { x: lx, y: H - M.b + 17, "text-anchor": "middle", class: "bh-tick", text }, a);
    }
    meas.remove();

    const lineG = svg("g", {}, s);
    const paths = new Map();
    for (const h of vis) {
      const pts = cs.filter((c) => h.vals.has(c)).map((c) => `${x(c).toFixed(1)} ${y(h, h.vals.get(c).v).toFixed(1)}`);
      const p = svg("path", { d: `M${pts.join("L")}${pts.length === 1 ? "h0" : ""}`, class: "bh-hero-line" }, lineG);
      p.style.stroke = groupColor(h.f.group);
      paths.set(h.f.key, p);
    }
    const axisG = svg("g", {}, s);
    const cross = svg("line", { y1: M.t, y2: H - M.b, class: "bh-cross", visibility: "hidden" }, s);
    const dots = svg("g", {}, s);
    const hit = svg("rect", { x: M.l - 8, y: M.t - 6, width: inner + 16, height: ih + 6, class: "bh-hit bh-hero-hit", tabindex: 0, "data-hero-id": "plot", "aria-label": "The chart: ←/→ moves between commits, ↑/↓ between lines, Enter keeps the highlighted line, Escape releases it" }, s);

    const dot = (h, c, r) => {
      const d = svg("circle", { cx: x(c), cy: y(h, h.vals.get(c).v), r, class: "bh-marker" }, dots);
      d.style.fill = groupColor(h.f.group);
    };
    const paint = () => {
      const key = current(), h = byKey.get(key), active = !!(hero.pinned || hero.hover);
      for (const [k, p] of paths) {
        p.classList.toggle("is-on", k === key);
        p.classList.toggle("is-dim", active && k !== key);
      }
      lineG.appendChild(paths.get(key));
      gridG.textContent = ""; axisG.textContent = ""; dots.textContent = "";
      const col = groupColor(h.f.group);
      svg("line", { x1: M.l - 0.5, x2: M.l - 0.5, y1: M.t, y2: H - M.b, class: "bh-hero-axis" }, axisG).style.stroke = col;
      for (const t of h.sc.ticks) {
        svg("line", { x1: M.l, x2: W - M.r, y1: y(h, t), y2: y(h, t), class: "bh-gridline" }, gridG);
        svg("line", { x1: M.l - 5, x2: M.l - 0.5, y1: y(h, t), y2: y(h, t), class: "bh-hero-axis" }, axisG).style.stroke = col;
        svg("text", { x: M.l - 8, y: y(h, t) + 4, "text-anchor": "end", class: "bh-tick bh-hero-tick", text: heroTick(h.f.unit, t) }, axisG);
      }
      const lastC = cs.filter((c) => h.vals.has(c)).pop();
      dot(h, lastC, 4);
      if (hero.commit !== null && h.vals.has(hero.commit) && cross.getAttribute("visibility") === "visible") dot(h, hero.commit, 4.5);
      const lv = h.vals.get(lastC);
      focusP.textContent = "Axis: ";
      el("strong", { text: h.f.label }, focusP);
      focusP.appendChild(document.createTextNode(` · ${groupTitle(h.f.group)} · ${fmt(h.f.unit, lv.v)} at ${commitLabel(data.commits[lastC])} (bench ${lv.bench})${hero.pinned ? " · kept: click the chart again to release it" : ""}`));
      for (const b of legend.querySelectorAll(".bh-hero-item")) b.classList.toggle("is-focus", b.dataset.key === key);
    };

    // The commit the pointer (or keyboard) is on: its links under the chart, and the tooltip
    const commitLinks = (c) => {
      const cm = data.commits[c];
      link.textContent = "";
      el("strong", { text: `${commitLabel(cm)}` }, link);
      link.appendChild(document.createTextNode(` · ${cm.date.slice(0, 10)} · ${cm.subject} · `));
      el("a", { href: `${REPO}/commit/${cm.sha}`, text: `commit ${cm.sha.slice(0, 7)}` }, link);
      if (cm.pr) {
        link.appendChild(document.createTextNode(" · "));
        el("a", { href: `${REPO}/pull/${cm.pr}`, text: `pull request #${cm.pr}` }, link);
      }
      const runs = runsOf(c);
      if (runs.length) {
        link.appendChild(document.createTextNode(` · run${runs.length > 1 ? "s" : ""} `));
        runs.forEach((u, i) => {
          if (i) link.appendChild(document.createTextNode(" "));
          el("a", { href: u, text: u.split("/").pop() }, link);
        });
      }
    };
    const hideTip = () => { tipH.hidden = true; cross.setAttribute("visibility", "hidden"); };
    const show = (c, py) => {
      hero.commit = c;
      commitLinks(c);
      cross.setAttribute("x1", x(c)); cross.setAttribute("x2", x(c)); cross.setAttribute("visibility", "visible");
      paint();
      const key = current(), h = byKey.get(key), cm = data.commits[c];
      tipH.textContent = "";
      const own = h.vals.get(c);
      el("div", { class: "bh-tip-title", text: `${commitLabel(cm)} · ${cm.sha.slice(0, 7)}${own ? ` · bench ${own.bench}` : ""}` }, tipH);
      el("div", { class: "bh-tip-subject", text: cm.subject.length > 90 ? `${cm.subject.slice(0, 88)}…` : cm.subject }, tipH);
      el("div", { class: "bh-hero-tip-group", text: groupTitle(h.f.group) }, tipH);
      for (const v of vis) {
        if (v.f.group !== h.f.group || !v.vals.has(c)) continue;
        const row = el("div", { class: `bh-hero-tip-row${v.f.key === key ? " is-on" : ""}` }, tipH);
        const k = svg("svg", { width: 12, height: 8, "aria-hidden": "true" }, row);
        svg("line", { x1: 1, x2: 11, y1: 4, y2: 4, class: "bh-line" }, k).style.stroke = groupColor(v.f.group);
        el("span", { class: "bh-num", text: fmt(v.f.unit, v.vals.get(c).v) }, row);
        el("span", { text: v.f.label }, row);
      }
      if (!own) el("div", { class: "bh-hero-tip-group", text: `${h.f.label}: not measured here` }, tipH);
      tipH.hidden = false;
      const r = s.getBoundingClientRect(), host = heroRoot.getBoundingClientRect(), k = r.width / W;
      const px = r.left - host.left + x(c) * k, tw2 = tipH.offsetWidth, th = tipH.offsetHeight;
      let left = px + 14;
      if (left + tw2 > host.width) left = px - tw2 - 14;
      tipH.style.left = `${Math.max(0, Math.min(host.width - tw2, left))}px`;
      // above or below the pointer, whichever half of the plot it is not in; on a narrow
      // screen, under the chart, which it would otherwise mostly cover
      const top = r.top - host.top + M.t * k, bottom = r.top - host.top + (H - M.b) * k;
      if (W < 560) tipH.style.top = `${r.bottom - host.top + 4}px`;
      else tipH.style.top = `${py < M.t + ih / 2 ? Math.max(top, bottom - th) : top}px`;
    };
    // The nearest line to a point, by pixel distance at that commit
    const nearestLine = (c, py) => {
      let best = null, bd = Infinity;
      for (const h of vis) {
        if (!h.vals.has(c)) continue;
        const d = Math.abs(y(h, h.vals.get(c).v) - py);
        if (d < bd) { bd = d; best = h.f.key; }
      }
      return best;
    };
    const at = (ev) => {
      const r = s.getBoundingClientRect();
      const px = ((ev.clientX - r.left) * W) / r.width, py = ((ev.clientY - r.top) * H) / r.height;
      const c = cs.reduce((a, q) => (Math.abs(x(q) - px) < Math.abs(x(a) - px) ? q : a), cs[0]);
      return { c, py, key: nearestLine(c, py) };
    };
    const move = (ev) => {
      const { c, py, key } = at(ev);
      if (!hero.pinned && key) hero.hover = key;
      show(c, py);
    };
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerdown", move);
    hit.addEventListener("click", (ev) => {
      const { c, py, key } = at(ev);
      if (hero.pinned) { hero.pinned = null; hero.hover = key; } else hero.pinned = key;
      show(c, py);
    });
    hit.addEventListener("pointerleave", () => { hideTip(); hero.hover = null; paint(); });
    let ki = cs.length - 1;
    const keyShow = () => {
      const h = byKey.get(current());
      show(cs[ki], h.vals.has(cs[ki]) ? y(h, h.vals.get(cs[ki]).v) : M.t);
    };
    hit.addEventListener("focus", () => { if (hero.commit !== null) ki = xi.get(hero.commit); keyShow(); });
    hit.addEventListener("blur", () => { hideTip(); hero.hover = null; paint(); });
    hit.addEventListener("keydown", (ev) => {
      if (ev.key === "ArrowLeft") ki = Math.max(0, ki - 1);
      else if (ev.key === "ArrowRight") ki = Math.min(cs.length - 1, ki + 1);
      else if (ev.key === "ArrowUp" || ev.key === "ArrowDown") {
        // the next line above or below the highlighted one, at this commit
        const c = cs[ki];
        const order = vis.filter((h) => h.vals.has(c)).sort((a, b) => y(a, a.vals.get(c).v) - y(b, b.vals.get(c).v)).map((h) => h.f.key);
        if (!order.length) return;
        const i = order.indexOf(current());
        const next = order[i < 0 ? 0 : Math.max(0, Math.min(order.length - 1, i + (ev.key === "ArrowUp" ? -1 : 1)))];
        if (hero.pinned) hero.pinned = next; else hero.hover = next;
      } else if (ev.key === "Enter" || ev.key === " ") {
        if (hero.pinned) { hero.hover = hero.pinned; hero.pinned = null; } else hero.pinned = current();
      } else if (ev.key === "Escape") { hero.pinned = null; hero.hover = null; }
      else return;
      ev.preventDefault();
      keyShow();
    });

    heroLegend(legend, figs, hidden, current, paint);
    commitLinks(hero.commit !== null ? hero.commit : cs[cs.length - 1]);
    paint();
    refocusNow();
  }

  // The legend under the hero chart, by group: an entry hides or shows its figure, a
  // group's header the whole group; pointing at (or focusing) an entry highlights its line
  function heroLegend(legend, figs, hidden, current, paint) {
    legend.textContent = "";
    const key = current();
    for (const [g] of GROUPS.concat([...new Set(figs.map((h) => h.f.group))].filter((g) => !GROUPS.some(([k]) => k === g)).map((g) => [g]))) {
      const items = figs.filter((h) => h.f.group === g);
      if (!items.length) continue;
      const box = el("div", { class: "bh-hero-lgroup", role: "group", "aria-label": groupTitle(g) }, legend);
      const allHidden = items.every((h) => hidden.has(h.f.key));
      const gb = el("button", { type: "button", class: "bh-hero-item bh-hero-ghead", "aria-pressed": String(!allHidden), "data-hero-id": `group-${g}`, title: allHidden ? "Show the group" : "Hide the group" }, box);
      el("span", { class: "bh-hero-sw", style: `background:${groupColor(g)}` }, gb);
      el("span", { text: `${groupTitle(g)} (${items.length})` }, gb);
      gb.addEventListener("click", () => {
        const keys = new Set(state.heroHidden);
        for (const h of items) (allHidden ? keys.delete(h.f.key) : keys.add(h.f.key));
        state.heroHidden = [...keys]; save(); drawHero();
      });
      for (const h of items) {
        const k = h.f.key, off = hidden.has(k);
        const b = el("button", { type: "button", class: `bh-hero-item${k === key ? " is-focus" : ""}`, "aria-pressed": String(!off), "data-key": k, "data-hero-id": `fig-${k}` }, box);
        el("span", { class: "bh-hero-sw", style: `background:${groupColor(g)}` }, b);
        el("span", { text: h.f.label }, b);
        b.addEventListener("click", () => {
          const keys = new Set(state.heroHidden);
          if (off) keys.delete(k); else keys.add(k);
          if (!off && hero.pinned === k) hero.pinned = null;
          if (!off && hero.hover === k) hero.hover = null;
          state.heroHidden = [...keys]; save(); drawHero();
        });
        const enter = () => { if (!hidden.has(k)) { hero.hover = k; paint(); } };
        const leave = () => { if (hero.hover === k) { hero.hover = null; paint(); } };
        b.addEventListener("pointerenter", enter);
        b.addEventListener("focus", enter);
        b.addEventListener("pointerleave", leave);
        b.addEventListener("blur", leave);
      }
    }
  }

  function render() {
    root.textContent = "";
    controls();
    const meta = el("p", { class: "bh-meta" }, root);
    const last = data.commits[data.commits.length - 1];
    meta.textContent = `${data.commits.length} commits of master, to ${last.sha.slice(0, 7)} (${last.date.slice(0, 10)}); ${data.records.length} paired runs. Generated ${data.generated.replace("T", " ").replace("+00:00", " UTC")}.`;
    el("p", { class: "bh-note bh-view-note", text: viewNote() }, root);
    summaryTable();
    for (const [g, title, note] of GROUPS) {
      const figs = data.figures.filter((f) => f.group === g);
      if (!figs.length) continue;
      el("h2", { class: "bh-group", text: title, id: `bench-${g}` }, root);
      if (note) el("p", { class: "bh-note", text: note }, root);
      const grid = el("div", { class: "bh-grid-cards" }, root);
      for (const f of figs) {
        const card = el("figure", { class: "bh-card" }, grid);
        const cap = el("figcaption", {}, card);
        el("span", { class: "bh-title", text: f.label }, cap);
        el("span", { class: "bh-latest" }, cap);
        el("div", { class: "bh-legend" }, card);
        el("div", { class: "bh-body" }, card);
        const det = el("details", {}, card);
        el("summary", { text: "Table" }, det);
        const table = el("table", {}, det);
        const hr = el("tr", {}, el("thead", {}, table));
        for (const h of ["commit", "date", "line", "value", "of", "runs"]) el("th", { text: h }, hr);
        el("tbody", {}, table);
        chart(f, card);
      }
    }
    tip = el("div", { class: "bh-tip", role: "status", hidden: "" }, root);
    drawHero();
  }

  let resizeT = null;
  const onResize = () => {
    clearTimeout(resizeT);
    resizeT = setTimeout(() => { if (root && root.isConnected && data) render(); }, 150);
  };

  async function init() {
    root = document.getElementById("bench-history");
    if (!root || root.dataset.ready) return;
    root.dataset.ready = "1";
    root.classList.add("bh-root");
    root.textContent = "Loading the benchmark history…";
    heroRoot = document.getElementById("bench-hero");
    if (heroRoot) heroRoot.classList.add("bh-root", "bh-hero");
    try {
      const res = await fetch(root.dataset.src, { cache: "no-cache" });
      if (!res.ok) throw new Error(`${res.status}`);
      data = await res.json();
    } catch (e) {
      root.textContent = `The benchmark history (${root.dataset.src}) could not be loaded: build it with scripts/bench_history.py.`;
      if (heroRoot) heroRoot.textContent = "";
      return;
    }
    const majors = [...new Set(data.commits.flatMap((c) => c.benches))].sort((a, b) => b - a);
    SLOTS = Object.fromEntries(majors.slice(0, 5).map((m, i) => [m, i + 1]));
    OLDEST = majors.length > 5 ? majors[5] : 0;
    restore();
    if (!data.cpus.includes(state.cpu)) {
      const counts = runCounts();
      state.cpu = data.cpus.slice().sort((a, b) => (counts[b] || 0) - (counts[a] || 0))[0];
    }
    // (kept by sha: a backfilled older commit moves every index)
    const sinces = sinceOptions();
    state.since = data.commits.findIndex((c) => c.sha === state.sinceSha);
    if (!sinces.includes(state.since)) {
      const pr = data.commits.findIndex((c) => c.pr === SINCE_PR);
      state.since = sinces.includes(pr) ? pr : sinces[Math.min(sinces.length - 1, 9)];
    }
    render();
  }

  window.addEventListener("resize", onResize);
  if (window.document$ && typeof window.document$.subscribe === "function") window.document$.subscribe(init);
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
