//! Split-side prototype: reshape's entity_row (copied from reshape/src/lib.rs at
//! origin/master) timed as it is, without its claims text, and with the claims appended
//! from the parsed line straight into columnar builders laid out as process.claims_schema
//! (offsets, validity bitmaps, values; no arrow-rs). The builder approximates genson's
//! rules (kv maps, empty_as_null, promoted scalars) closely enough to do the same work;
//! leaf checksums are printed so its coverage can be checked against genson's output.
#![allow(dead_code)]
include!("reshape_inc.rs");

use std::time::Instant;

// ---------- columnar builders ----------
#[derive(Default)]
struct Bits { bytes: Vec<u8>, len: usize }
impl Bits {
    #[inline] fn push(&mut self, b: bool) {
        if self.len % 8 == 0 { self.bytes.push(0); }
        if b { *self.bytes.last_mut().unwrap() |= 1 << (self.len % 8); }
        self.len += 1;
    }
    fn ones(&self) -> usize { self.bytes.iter().map(|b| b.count_ones() as usize).sum() }
}

enum Kind { Str, I64, F64, Null, Struct(Vec<(&'static str, B)>), List(Box<B>), Map(Box<B>) }
// a promoted scalar: a string, int or number in place of the record goes to field `name__type`
#[derive(Clone, Copy, PartialEq)]
enum Promote { No, Yes }

struct B {
    kind: Kind,
    valid: Bits,
    off: Vec<i64>,      // Str, List, Map
    data: Vec<u8>,      // Str
    ints: Vec<i64>,
    nums: Vec<f64>,
    keys: Option<Box<B>>, // Map keys (Str)
    promote: Promote,
    nulls: usize,
    hint: usize,
    phantom: bool,
}

fn mk(kind: Kind) -> B {
    let off = match kind { Kind::Str | Kind::List(_) | Kind::Map(_) => vec![0], _ => vec![] };
    let keys = match kind { Kind::Map(_) => Some(Box::new(mk(Kind::Str))), _ => None };
    B { kind, valid: Bits::default(), off, data: vec![], ints: vec![], nums: vec![], keys, promote: Promote::No, nulls: 0, hint: 0, phantom: false }
}
fn s() -> B { mk(Kind::Str) }
fn i() -> B { mk(Kind::I64) }
fn f() -> B { mk(Kind::F64) }
fn rec(fields: Vec<(&'static str, B)>) -> B { mk(Kind::Struct(fields)) }
fn list(b: B) -> B { mk(Kind::List(Box::new(b))) }
fn map(b: B) -> B { mk(Kind::Map(Box::new(b))) }
fn promoted(mut b: B) -> B { b.promote = Promote::Yes; b }
fn phantom(mut b: B) -> B { b.phantom = true; b }

fn snak() -> B {
    let numint = |n: &'static str, m: &'static str| promoted(rec(vec![(n, f()), (m, phantom(i()))]));
    let dv = promoted(rec(vec![
        ("id", s()), ("datavalue__string", s()),
        ("precision", promoted(rec(vec![("precision__integer", i()), ("precision__number", f())]))),
        ("text", s()), ("language", s()), ("amount", s()), ("unit", s()), ("upperBound", s()),
        ("lowerBound", s()), ("time", s()), ("timezone", i()), ("before", i()), ("after", i()),
        ("calendarmodel", s()), ("latitude", numint("latitude__number", "latitude__integer")),
        ("longitude", numint("longitude__number", "longitude__integer")), ("altitude", mk(Kind::Null)),
        ("globe", s()), ("entity-type", s()), ("numeric-id", i()),
    ]));
    rec(vec![("snaktype", s()), ("property", s()), ("hash", s()), ("datavalue", dv), ("datavalue_type", s()), ("datatype", s())])
}
fn claims_builder() -> B {
    let statement = rec(vec![
        ("mainsnak", snak()), ("type", s()), ("id", s()), ("rank", s()),
        ("references", list(rec(vec![("hash", s()), ("snaks", map(list(snak()))), ("snaks-order", list(s()))]))),
        ("qualifiers", map(list(snak()))), ("qualifiers-order", list(s())),
    ]);
    map(list(statement))
}

impl B {
    fn null(&mut self) {
        self.valid.push(false);
        self.nulls += 1;
        match &mut self.kind {
            Kind::Str | Kind::List(_) | Kind::Map(_) => { let l = *self.off.last().unwrap(); self.off.push(l); }
            Kind::I64 => self.ints.push(0),
            Kind::F64 => self.nums.push(0.0),
            Kind::Null => {}
            Kind::Struct(fs) => for (_, b) in fs.iter_mut() { b.null() },
        }
    }
    fn str(&mut self, x: &str) {
        self.data.extend_from_slice(x.as_bytes());
        self.off.push(self.data.len() as i64);
        self.valid.push(true);
    }
    /// Append `v` (None: absent). A value this schema does not hold is Err (genson: no fit).
    fn put(&mut self, v: Option<&Value>) -> R {
        let v = match v { Some(v) if !v.is_null() => v, _ => { self.null(); return Ok(()) } };
        let promote = self.promote;
        match &mut self.kind {
            Kind::Str => { let x = v.as_str().ok_or(Fallback)?; self.data.extend_from_slice(x.as_bytes()); self.off.push(self.data.len() as i64); self.valid.push(true) }
            Kind::I64 => { let raw = v.as_raw_number().ok_or(Fallback)?; self.ints.push(raw.as_str().parse().map_err(|_| Fallback)?); self.valid.push(true) }
            Kind::F64 => {
                let raw = v.as_raw_number().ok_or(Fallback)?;
                // as genson reads a number: serde_json's Number (its fast float path, which is
                // not always the nearest double), then as_f64
                let x = if std::env::var_os("NEAREST").is_some() { raw.as_str().parse().map_err(|_| Fallback)? } else {
                    raw.as_str().parse::<serde_json::Number>().map_err(|_| Fallback)?.as_f64().ok_or(Fallback)? };
                self.nums.push(x); self.valid.push(true) }
            Kind::Null => return Err(Fallback),
            Kind::Struct(fs) => {
                if let Some(o) = v.as_object() {
                    if o.is_empty() { self.null(); return Ok(()) } // empty_as_null (approx.)
                    // members in order: a field matched by the hint (the next field) or a scan
                    let mut seen: u64 = 0;
                    let mut hint = 0usize;
                    for (k, x) in o.iter() {
                        let j = if fs.get(hint).map(|f| f.0) == Some(k) { hint } else {
                            fs.iter().position(|f| f.0 == k).ok_or(Fallback)?
                        };
                        if fs[j].1.phantom { return Err(Fallback) }
                        fs[j].1.put(Some(x))?;
                        seen |= 1 << j;
                        hint = j + 1;
                    }
                    for (j, (_, b)) in fs.iter_mut().enumerate() { if seen & (1 << j) == 0 { b.null() } }
                } else if promote == Promote::Yes {
                    // a scalar in place of the record: to its `name__type` field
                    let want = match v.get_type() {
                        JsonType::String => "string",
                        JsonType::Number => { let r = v.as_raw_number().ok_or(Fallback)?; if r.as_str().bytes().any(|c| matches!(c, b'.' | b'e' | b'E')) { "number" } else { "integer" } }
                        _ => return Err(Fallback),
                    };
                    // the field for the scalar's type in the decode schema (claims.avsc); an
                    // integer goes to a number field where the record has no integer one
                    let pick = |want: &str| fs.iter().position(|(n, b)| !b.phantom && n.ends_with(want) && n[..n.len() - want.len()].ends_with("__"));
                    let j = pick(want).or_else(|| if want == "integer" { pick("number") } else { None }).ok_or(Fallback)?;
                    for (k, (_, b)) in fs.iter_mut().enumerate() { if k == j { b.put(Some(v))? } else { b.null() } }
                } else { return Err(Fallback) }
                self.valid.push(true);
            }
            Kind::List(kid) => {
                let a = v.as_array().ok_or(Fallback)?;
                if a.is_empty() { self.null(); return Ok(()) } // empty_as_null
                for x in a.iter() { kid.put(Some(x))? }
                let l = *self.off.last().unwrap(); self.off.push(l + a.len() as i64); self.valid.push(true);
            }
            Kind::Map(kid) => {
                let o = v.as_object().ok_or(Fallback)?;
                if o.is_empty() { self.null(); return Ok(()) }
                let keys = self.keys.as_mut().unwrap();
                for (k, x) in o.iter() { keys.str(k); kid.put(Some(x))? }
                let l = *self.off.last().unwrap(); self.off.push(l + o.len() as i64); self.valid.push(true);
            }
        }
        Ok(())
    }
    fn mark(&self) -> usize { self.valid.len }
    /// leaf checksums: path -> (values, non-null, bytes or sum)
    fn sums(&self, path: &str, out: &mut Vec<(String, usize, usize, i128)>) {
        let nn = self.valid.ones();
        match &self.kind {
            Kind::Str => out.push((path.into(), self.valid.len, nn, self.data.len() as i128)),
            Kind::I64 => out.push((path.into(), self.valid.len, nn, self.ints.iter().map(|&x| x as i128).sum())),
            Kind::F64 => out.push((path.into(), self.valid.len, nn, 0)),
            Kind::Null => {}
            Kind::Struct(fs) => { out.push((format!("{path}<>"), self.valid.len, nn, 0)); for (n, b) in fs { b.sums(&format!("{path}.{n}"), out) } }
            Kind::List(k) => { out.push((format!("{path}[]"), self.valid.len, nn, 0)); k.sums(&format!("{path}[]"), out) }
            Kind::Map(k) => { out.push((format!("{path}{{}}"), self.valid.len, nn, 0)); self.keys.as_ref().unwrap().sums(&format!("{path}{{}}.key"), out); k.sums(&format!("{path}{{}}.value"), out) }
        }
    }
}

impl B {
    fn export(&self, out: &mut Vec<u8>) {
        let put = |out: &mut Vec<u8>, b: &[u8]| { out.extend_from_slice(&(b.len() as u64).to_le_bytes()); out.extend_from_slice(b) };
        out.extend_from_slice(&(self.valid.len as u64).to_le_bytes());
        out.extend_from_slice(&(self.nulls as u64).to_le_bytes());
        put(out, &self.valid.bytes);
        let off: Vec<u8> = self.off.iter().flat_map(|o| o.to_le_bytes()).collect();
        put(out, &off);
        put(out, &self.data);
        let ints: Vec<u8> = self.ints.iter().flat_map(|o| o.to_le_bytes()).collect();
        put(out, &ints);
        let nums: Vec<u8> = self.nums.iter().flat_map(|o| o.to_le_bytes()).collect();
        put(out, &nums);
        match &self.kind {
            Kind::Struct(fs) => for (_, b) in fs { b.export(out) },
            Kind::List(k) => k.export(out),
            Kind::Map(k) => { self.keys.as_ref().unwrap().export(out); k.export(out) }
            _ => {}
        }
    }
}

// ---------- the dump line's claims, as the reshape makes them, into the builder ----------
// dump._snak: the datavalue's value as `datavalue`, its type as `datavalue_type`
fn put_snak(b: &mut B, sv: &Value) -> R {
    let o = sv.as_object().ok_or(Fallback)?;
    let fs = match &mut b.kind { Kind::Struct(fs) => fs, _ => unreachable!() };
    let mut seen: u64 = 0;
    let dv = sv.get("datavalue");
    for (k, x) in o.iter() {
        if k == "datavalue" { continue }
        let j = fs.iter().position(|f| f.0 == k).ok_or(Fallback)?;
        fs[j].1.put(Some(x))?; seen |= 1 << j;
    }
    if let Some(dv) = dv {
        fs[3].1.put(dv.get("value"))?; seen |= 1 << 3;
        fs[4].1.put(dv.get("type"))?; seen |= 1 << 4;
    }
    for (j, (_, c)) in fs.iter_mut().enumerate() { if seen & (1 << j) == 0 { c.null() } }
    b.valid.push(true);
    Ok(())
}
fn put_snaks(b: &mut B, v: &Value) -> R { // map(list(snak))
    let o = match v.as_object() { Some(o) if !o.is_empty() => o, _ => { b.null(); return Ok(()) } };
    let n = o.len();
    let (keys, kid) = match &mut b.kind { Kind::Map(k) => (b.keys.as_mut().unwrap(), k), _ => unreachable!() };
    for (p, l) in o.iter() {
        keys.str(p);
        let a = l.as_array().ok_or(Fallback)?;
        if a.is_empty() { kid.null(); continue }
        let inner = match &mut kid.kind { Kind::List(k) => k, _ => unreachable!() };
        for sv in a.iter() { put_snak(inner, sv)? }
        let last = *kid.off.last().unwrap(); kid.off.push(last + a.len() as i64); kid.valid.push(true);
    }
    let last = *b.off.last().unwrap(); b.off.push(last + n as i64); b.valid.push(true);
    Ok(())
}
fn put_statement(b: &mut B, st: &Value) -> R {
    let o = st.as_object().ok_or(Fallback)?;
    let fs = match &mut b.kind { Kind::Struct(fs) => fs, _ => unreachable!() };
    let mut seen: u64 = 0;
    for (k, x) in o.iter() {
        let j = fs.iter().position(|f| f.0 == k).ok_or(Fallback)?;
        seen |= 1 << j;
        match k {
            "mainsnak" => put_snak(&mut fs[j].1, x)?,
            "qualifiers" => put_snaks(&mut fs[j].1, x)?,
            "references" => {
                let rb = &mut fs[j].1;
                let a = x.as_array().ok_or(Fallback)?;
                if a.is_empty() { rb.null(); continue }
                let rec_b = match &mut rb.kind { Kind::List(k) => k, _ => unreachable!() };
                for r in a.iter() {
                    let ro = r.as_object().ok_or(Fallback)?;
                    let rfs = match &mut rec_b.kind { Kind::Struct(fs) => fs, _ => unreachable!() };
                    let mut rseen: u64 = 0;
                    for (rk, rx) in ro.iter() {
                        let rj = rfs.iter().position(|f| f.0 == rk).ok_or(Fallback)?;
                        rseen |= 1 << rj;
                        if rk == "snaks" { put_snaks(&mut rfs[rj].1, rx)? } else { rfs[rj].1.put(Some(rx))? }
                    }
                    for (rj, (_, c)) in rfs.iter_mut().enumerate() { if rseen & (1 << rj) == 0 { c.null() } }
                    rec_b.valid.push(true);
                }
                let last = *rb.off.last().unwrap(); rb.off.push(last + a.len() as i64); rb.valid.push(true);
            }
            _ => fs[j].1.put(Some(x))?,
        }
    }
    for (j, (_, c)) in fs.iter_mut().enumerate() { if seen & (1 << j) == 0 { c.null() } }
    b.valid.push(true);
    Ok(())
}
fn put_claims(b: &mut B, v: Option<&Value>) -> R { // map(list(statement))
    let o = match v.and_then(|v| v.as_object()) { Some(o) if !o.is_empty() => o, _ => { b.null(); return Ok(()) } };
    let n = o.len();
    let (keys, kid) = match &mut b.kind { Kind::Map(k) => (b.keys.as_mut().unwrap(), k), _ => unreachable!() };
    for (p, l) in o.iter() {
        keys.str(p);
        let a = l.as_array().ok_or(Fallback)?;
        if a.is_empty() { kid.null(); continue }
        let inner = match &mut kid.kind { Kind::List(k) => k, _ => unreachable!() };
        for st in a.iter() { put_statement(inner, st)? }
        let last = *kid.off.last().unwrap(); kid.off.push(last + a.len() as i64); kid.valid.push(true);
    }
    let last = *b.off.last().unwrap(); b.off.push(last + n as i64); b.valid.push(true);
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode { Parse, AsIs, NoClaims, Typed, ClaimsTextOnly }

/// reshape's entity_row (as in lib.rs), with its claims written as text (AsIs), left
/// out (NoClaims), or appended to `typed` (Typed); Parse only parses the line
fn entity_row_mode(line: &[u8], cols: &mut Columns, stats: &mut Stats, mode: Mode, typed: &mut B) -> R {
    let e: Value = sonic_rs::Deserializer::from_slice(line).use_rawnumber().deserialize().map_err(|_| Fallback)?;
    if mode == Mode::Parse { std::hint::black_box(&e); return Ok(()) }
    let o = e.as_object().ok_or(Fallback)?;
    let id = e.get("id").and_then(|v| v.as_str()).ok_or(Fallback)?;
    let mut mismatched = 0;
    if mode == Mode::ClaimsTextOnly {
        // only the claims written as text
    } else {
    cols.data[0].extend_from_slice(id.as_bytes());
    cols.end_cell(0);
    write_terms(&mut cols.data[1], e.get("labels"), &mut mismatched)?;
    cols.end_cell(1);
    write_terms(&mut cols.data[2], e.get("descriptions"), &mut mismatched)?;
    cols.end_cell(2);
    write_alias_terms(&mut cols.data[3], e.get("aliases"), &mut mismatched)?;
    cols.end_cell(3);
    let mut sitelink_fields = Vec::new();
    {
        let out = &mut cols.data[4];
        out.push(b'{');
        let mut first = true;
        for (site, link) in map_entries(e.get("sitelinks")) {
            let lo = link.as_object().ok_or(Fallback)?;
            sitelink_fields.extend(lo.iter().map(|(k, _)| k));
            key(out, &mut first, site);
            write_value(out, link)?;
        }
        out.push(b'}');
    }
    cols.end_cell(4);
    stats.sitelink_fields.extend(sitelink_fields.into_iter().map(String::from));
    }
    match mode {
        Mode::AsIs | Mode::ClaimsTextOnly => {
            let out = &mut cols.data[5];
            out.push(b'{');
            let mut first = true;
            for (p, list) in map_entries(e.get("claims")) {
                key(out, &mut first, p);
                out.push(b'[');
                for (i, st) in list.as_array().ok_or(Fallback)?.iter().enumerate() {
                    if i > 0 { out.push(b','); }
                    write_statement(out, st)?;
                }
                out.push(b']');
            }
            out.push(b'}');
        }
        Mode::Typed => put_claims(typed, e.get("claims"))?,
        _ => {}
    }
    cols.end_cell(5);
    if mode == Mode::ClaimsTextOnly { return Ok(()) }
    let mut entity_fields = Vec::new();
    {
        let out = &mut cols.data[6];
        out.push(b'{');
        let mut first = true;
        for (k, x) in o.iter() {
            if !ENTITY_PARTS.contains(&k) {
                entity_fields.push(k);
                key(out, &mut first, k);
                write_value(out, x)?;
            }
        }
        out.push(b'}');
    }
    cols.end_cell(6);
    stats.mismatched += mismatched;
    stats.entity_fields.extend(entity_fields.into_iter().map(String::from));
    Ok(())
}

fn cpu() -> f64 {
    let t = std::fs::read_to_string("/proc/thread-self/stat").unwrap();
    let f: Vec<&str> = t.rsplit(')').next().unwrap().split_whitespace().collect();
    (f[11].parse::<f64>().unwrap() + f[12].parse::<f64>().unwrap()) / 100.0
}

fn hwm() -> String { std::fs::read_to_string("/proc/self/status").unwrap().lines().find(|l| l.starts_with("VmHWM")).unwrap().to_string() }
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read(&args[1]).unwrap();
    let reps: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(5);
    let lines: Vec<&[u8]> = text.split(|&c| c == b'\n').filter(|l| !l.is_empty()).collect();
    let bytes: usize = lines.iter().map(|l| l.len()).sum();
    let all = [Mode::Parse, Mode::AsIs, Mode::NoClaims, Mode::Typed, Mode::ClaimsTextOnly];
    let only = std::env::var("ONLY").ok();
    let modes: Vec<Mode> = all.into_iter().filter(|m| only.as_deref().map_or(true, |o| format!("{m:?}") == o)).collect();
    let mut best = vec![(f64::MAX, f64::MAX); modes.len()];
    let mut fallbacks = vec![0usize; modes.len()];
    let mut claims_text = 0usize;
    for _ in 0..reps {
        for (mi, &mode) in modes.iter().enumerate() {
            let mut cols = Columns::with_room(lines.len(), bytes);
            let mut stats = Stats { mismatched: 0, entity_fields: BTreeSet::new(), sitelink_fields: BTreeSet::new() };
            let mut typed = claims_builder();
            let (c0, t0) = (cpu(), Instant::now());
            let mut fb = 0;
            for line in &lines {
                let m = cols.mark();
                if entity_row_mode(line, &mut cols, &mut stats, mode, &mut typed).is_err() { cols.truncate(&m); fb += 1; }
            }
            let (c, w) = (cpu() - c0, t0.elapsed().as_secs_f64());
            if mode == Mode::AsIs { claims_text = cols.data[5].len() }
            fallbacks[mi] = fb;
            if c < best[mi].0 { best[mi] = (c, w) }
            if mode == Mode::Typed && std::env::var("SUMS").is_ok() {
                let mut out = vec![];
                typed.sums("claims", &mut out);
                for (p, n, nn, x) in out { println!("SUM\t{p}\t{n}\t{nn}\t{x}") }
                std::env::remove_var("SUMS");
            }
            drop(typed); drop(cols);
        }
    }
    if let Ok(path) = std::env::var("EXPORT") {
        let batch: usize = std::env::var("BATCH").ok().map(|b| b.parse().unwrap()).unwrap_or(1000);
        use std::io::Write;
        let mut file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        let mut cols = Columns::with_room(0, 0);
        let mut stats = Stats { mismatched: 0, entity_fields: BTreeSet::new(), sitelink_fields: BTreeSet::new() };
        let mut ids = Vec::new();
        for part in lines.chunks(batch) {
            let mut typed = claims_builder();
            for line in part {
                if entity_row_mode(line, &mut cols, &mut stats, Mode::Typed, &mut typed).is_err() { panic!("misfit: the chunk would take genson's path") }
                let e: Value = sonic_rs::from_slice(line).unwrap();
                ids.push(e.get("id").unwrap().as_str().unwrap().to_string());
            }
            let mut out = (part.len() as u64).to_le_bytes().to_vec();
            typed.export(&mut out);
            drop(typed);
            file.write_all(&out).unwrap();
            cols = Columns::with_room(0, 0);
        }
        file.flush().unwrap();
        std::fs::write(format!("{path}.ids"), ids.join("\n")).unwrap();
        eprintln!("{}", hwm());
        return;
    }
    println!("lines\t{}\tbytes\t{}\tclaims_text_bytes\t{}", lines.len(), bytes, claims_text);
    eprintln!("{}", hwm());
    for (mi, m) in modes.iter().enumerate() {
        println!("{:?}\tcpu {:.3}\twall {:.3}\tfallbacks {}", m, best[mi].0, best[mi].1, fallbacks[mi]);
    }
}
