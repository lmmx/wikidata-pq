// The benchmark history dashboard (docs/guide/bench-history.md): line charts of the
// figures in docs/assets/bench-history.json (scripts/bench_history.py), in plain SVG.
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
  const state = { view: "cpu", cpu: null, from: 8, since: null, sinceSha: null, backfill: false };
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
  const fmt = (unit, v) => (unit === "%" ? `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(1)}%` : unit === "MB" ? size(v) : duration(v));
  const fmtE = (unit, e) => (unit === "%" ? `${e.toFixed(1)}%` : unit === "MB" ? size(e) : e < 60 ? `${e.toFixed(e < 1 ? 2 : 1)}s` : duration(e));
  const tickFmt = (unit, v) => {
    if (unit === "%") return `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v)}%`;
    if (unit === "MB") return v >= 1000 ? `${+(v / 1000).toFixed(2)} GB` : `${v} MB`;
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
  };

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
    try {
      const res = await fetch(root.dataset.src, { cache: "no-cache" });
      if (!res.ok) throw new Error(`${res.status}`);
      data = await res.json();
    } catch (e) {
      root.textContent = `The benchmark history (${root.dataset.src}) could not be loaded: build it with scripts/bench_history.py.`;
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
