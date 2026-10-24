//! dump.entity_row in one compiled pass: each dump line parsed once (sonic-rs), and its
//! row written straight from the parsed line as orjson writes it: strings escaped as
//! orjson escapes them, floats by ryu (as orjson writes them), keys in the line's order.
//! A line outside what the reshape expects (a field of the wrong type, say) is left for
//! dump.entity_row to reshape, so the rows are the same whatever the dump holds.
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use sonic_rs::{JsonContainerTrait, JsonType, JsonValueTrait, Value};
use std::collections::BTreeSet;

const ENTITY_PARTS: [&str; 6] = [
    "id",
    "labels",
    "descriptions",
    "aliases",
    "sitelinks",
    "claims",
];

/// A line the fast path leaves to dump.entity_row
struct Fallback;
type R<T = ()> = Result<T, Fallback>;

fn write_str(out: &mut Vec<u8>, s: &str) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out.push(b'"');
    let b = s.as_bytes();
    let mut start = 0;
    for (i, &c) in b.iter().enumerate() {
        let esc: &[u8] = match c {
            b'"' => b"\\\"",
            b'\\' => b"\\\\",
            b'\n' => b"\\n",
            b'\r' => b"\\r",
            b'\t' => b"\\t",
            0x08 => b"\\b",
            0x0c => b"\\f",
            0..=0x1f => b"",
            _ => continue,
        };
        out.extend_from_slice(&b[start..i]);
        if esc.is_empty() {
            out.extend_from_slice(&[
                b'\\',
                b'u',
                b'0',
                b'0',
                HEX[(c >> 4) as usize],
                HEX[(c & 15) as usize],
            ]);
        } else {
            out.extend_from_slice(esc);
        }
        start = i + 1;
    }
    out.extend_from_slice(&b[start..]);
    out.push(b'"');
}

/// A number as orjson writes what it parses it as (the line's numbers are kept as their
/// text): an integer in 64 bits as it is (but -0 as 0), any other number as a double,
/// which orjson writes as ryu does but with a + on a positive exponent (1e+16)
fn write_number(out: &mut Vec<u8>, raw: &str) -> R {
    let int = !raw.bytes().any(|c| matches!(c, b'.' | b'e' | b'E'));
    if int && (raw.parse::<i64>().is_ok() || raw.parse::<u64>().is_ok()) {
        out.extend_from_slice(if raw == "-0" { b"0" } else { raw.as_bytes() });
        return Ok(());
    }
    let f: f64 = raw.parse().map_err(|_| Fallback)?;
    if !f.is_finite() {
        return Err(Fallback); // orjson raises
    }
    let mut buf = ryu::Buffer::new();
    let text = buf.format_finite(f).as_bytes();
    match text.iter().position(|&c| c == b'e') {
        Some(e) if text.get(e + 1) != Some(&b'-') => {
            out.extend_from_slice(&text[..=e]);
            out.push(b'+');
            out.extend_from_slice(&text[e + 1..]);
        }
        _ => out.extend_from_slice(text),
    }
    Ok(())
}

fn write_value(out: &mut Vec<u8>, v: &Value) -> R {
    match v.get_type() {
        JsonType::Null => out.extend_from_slice(b"null"),
        JsonType::Boolean => out.extend_from_slice(if v.as_bool() == Some(true) {
            b"true"
        } else {
            b"false"
        }),
        JsonType::Number => write_number(out, v.as_raw_number().ok_or(Fallback)?.as_str())?,
        JsonType::String => write_str(out, v.as_str().ok_or(Fallback)?),
        JsonType::Array => {
            out.push(b'[');
            for (i, x) in v.as_array().ok_or(Fallback)?.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_value(out, x)?;
            }
            out.push(b']');
        }
        JsonType::Object => {
            out.push(b'{');
            for (i, (k, x)) in v.as_object().ok_or(Fallback)?.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_str(out, k);
                out.push(b':');
                write_value(out, x)?;
            }
            out.push(b'}');
        }
    }
    Ok(())
}

/// An object's key (after a comma unless it is the object's first)
fn key(out: &mut Vec<u8>, first: &mut bool, k: &str) {
    if !std::mem::take(first) {
        out.push(b',');
    }
    write_str(out, k);
    out.push(b':');
}

/// dump._map: the object, or none (Wikibase writes an empty map as [])
fn map_entries(v: Option<&Value>) -> impl Iterator<Item = (&str, &Value)> {
    v.and_then(|v| v.as_object())
        .into_iter()
        .flat_map(|o| o.iter())
}

fn plain_term(term: &Value, lang: &str) -> bool {
    term.as_object().map(|o| o.len()) == Some(2)
        && term.get("language").and_then(|l| l.as_str()) == Some(lang)
}

fn write_terms(out: &mut Vec<u8>, v: Option<&Value>, mismatched: &mut u64) -> R {
    out.push(b'{');
    let mut first = true;
    for (lang, term) in map_entries(v) {
        if !term.is_object() {
            return Err(Fallback);
        }
        if !plain_term(term, lang) {
            *mismatched += 1;
        }
        key(out, &mut first, lang);
        write_value(out, term.get("value").ok_or(Fallback)?)?;
    }
    out.push(b'}');
    Ok(())
}

fn write_alias_terms(out: &mut Vec<u8>, v: Option<&Value>, mismatched: &mut u64) -> R {
    out.push(b'{');
    let mut first = true;
    for (lang, list) in map_entries(v) {
        key(out, &mut first, lang);
        out.push(b'[');
        for (i, term) in list.as_array().ok_or(Fallback)?.iter().enumerate() {
            if !term.is_object() {
                return Err(Fallback);
            }
            if !plain_term(term, lang) {
                *mismatched += 1;
            }
            if i > 0 {
                out.push(b',');
            }
            write_value(out, term.get("value").ok_or(Fallback)?)?;
        }
        out.push(b']');
    }
    out.push(b'}');
    Ok(())
}

/// dump._snak: the snak's fields but its datavalue, then the datavalue's value and type
fn write_snak(out: &mut Vec<u8>, s: &Value) -> R {
    let o = s.as_object().ok_or(Fallback)?;
    let dv = s.get("datavalue");
    if dv.is_some() && s.get("datavalue_type").is_some() {
        return Err(Fallback); // dump._snak would set it in place
    }
    out.push(b'{');
    let mut first = true;
    for (k, x) in o.iter() {
        if k != "datavalue" {
            key(out, &mut first, k);
            write_value(out, x)?;
        }
    }
    if let Some(dv) = dv {
        if !dv.is_object() {
            return Err(Fallback);
        }
        for (k, name) in [("datavalue", "value"), ("datavalue_type", "type")] {
            key(out, &mut first, k);
            match dv.get(name) {
                Some(x) => write_value(out, x)?,
                None => out.extend_from_slice(b"null"),
            }
        }
    }
    out.push(b'}');
    Ok(())
}

fn write_snaks(out: &mut Vec<u8>, v: Option<&Value>) -> R {
    out.push(b'{');
    let mut first = true;
    for (p, list) in map_entries(v) {
        key(out, &mut first, p);
        out.push(b'[');
        for (i, s) in list.as_array().ok_or(Fallback)?.iter().enumerate() {
            if i > 0 {
                out.push(b',');
            }
            write_snak(out, s)?;
        }
        out.push(b']');
    }
    out.push(b'}');
    Ok(())
}

/// dump._statement: its mainsnak, qualifiers and references' snaks reshaped, in place
fn write_statement(out: &mut Vec<u8>, st: &Value) -> R {
    let o = st.as_object().ok_or(Fallback)?;
    if st.get("mainsnak").is_none() {
        return Err(Fallback);
    }
    out.push(b'{');
    let mut first = true;
    for (k, x) in o.iter() {
        key(out, &mut first, k);
        match k {
            "mainsnak" => write_snak(out, x)?,
            "qualifiers" => write_snaks(out, Some(x))?,
            "references" => {
                out.push(b'[');
                for (i, r) in x.as_array().ok_or(Fallback)?.iter().enumerate() {
                    let ro = r.as_object().ok_or(Fallback)?;
                    if i > 0 {
                        out.push(b',');
                    }
                    out.push(b'{');
                    let mut rfirst = true;
                    for (rk, rx) in ro.iter() {
                        key(out, &mut rfirst, rk);
                        if rk == "snaks" {
                            write_snaks(out, Some(rx))?
                        } else {
                            write_value(out, rx)?
                        }
                    }
                    if r.get("snaks").is_none() {
                        key(out, &mut rfirst, "snaks");
                        out.extend_from_slice(b"{}");
                    }
                    out.push(b'}');
                }
                out.push(b']');
            }
            _ => write_value(out, x)?,
        }
    }
    out.push(b'}');
    Ok(())
}

/// The 7 columns' buffers (Arrow large-string layout: the data and its offsets)
struct Columns {
    data: [Vec<u8>; 7],
    offsets: [Vec<i64>; 7],
}

impl Columns {
    /// Room for `rows` rows of `bytes` bytes of lines, so the buffers are not regrown (a
    /// reshaped column is never longer than its lines; untouched room costs no memory)
    fn with_room(rows: usize, bytes: usize) -> Self {
        let mut offsets: [Vec<i64>; 7] = std::array::from_fn(|_| Vec::with_capacity(rows + 1));
        for o in &mut offsets {
            o.push(0);
        }
        Columns {
            data: std::array::from_fn(|_| Vec::with_capacity(bytes)),
            offsets,
        }
    }
    fn end_cell(&mut self, c: usize) {
        self.offsets[c].push(self.data[c].len() as i64);
    }
    fn mark(&self) -> [(usize, usize); 7] {
        std::array::from_fn(|c| (self.data[c].len(), self.offsets[c].len()))
    }
    fn truncate(&mut self, marks: &[(usize, usize); 7]) {
        for ((data, offsets), &(d, o)) in self.data.iter_mut().zip(&mut self.offsets).zip(marks) {
            data.truncate(d);
            offsets.truncate(o);
        }
    }
}

struct Stats {
    mismatched: u64,
    entity_fields: BTreeSet<String>,
    sitelink_fields: BTreeSet<String>,
}

fn entity_row(line: &[u8], cols: &mut Columns, stats: &mut Stats) -> R {
    let e: Value = sonic_rs::Deserializer::from_slice(line)
        .use_rawnumber()
        .deserialize()
        .map_err(|_| Fallback)?;
    let o = e.as_object().ok_or(Fallback)?;
    let id = e.get("id").and_then(|v| v.as_str()).ok_or(Fallback)?;
    let mut mismatched = 0;
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
    {
        let out = &mut cols.data[5];
        out.push(b'{');
        let mut first = true;
        for (p, list) in map_entries(e.get("claims")) {
            key(out, &mut first, p);
            out.push(b'[');
            for (i, st) in list.as_array().ok_or(Fallback)?.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_statement(out, st)?;
            }
            out.push(b']');
        }
        out.push(b'}');
    }
    cols.end_cell(5);
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
    stats
        .entity_fields
        .extend(entity_fields.into_iter().map(String::from));
    stats
        .sitelink_fields
        .extend(sitelink_fields.into_iter().map(String::from));
    Ok(())
}

/// Reshape a run of lines; returns its columns, stats, and the indices left to Python
fn run(lines: &[&[u8]], base: usize) -> (Columns, Stats, Vec<usize>) {
    let mut cols = Columns::with_room(lines.len(), lines.iter().map(|l| l.len()).sum());
    let mut stats = Stats {
        mismatched: 0,
        entity_fields: BTreeSet::new(),
        sitelink_fields: BTreeSet::new(),
    };
    let mut left = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let marks = cols.mark();
        let before = stats.mismatched;
        if entity_row(line, &mut cols, &mut stats).is_err() {
            cols.truncate(&marks);
            stats.mismatched = before;
            for c in 0..7 {
                cols.end_cell(c); // an empty cell, filled in by Python
            }
            left.push(base + i);
        }
    }
    (cols, stats, left)
}

type Out<'py> = (
    Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)>,
    u64,
    Vec<String>,
    Vec<String>,
    Vec<usize>,
);

/// The chunk's rows as 7 columns (dump.COLUMNS), each as (data, int64 offsets) bytes;
/// the mismatched terms; the entities' own field names and their sitelinks' field names;
/// and the indices of the lines left for dump.entity_row (their cells empty here).
#[pyfunction]
#[pyo3(signature = (lines, threads=1))]
fn entity_rows<'py>(py: Python<'py>, lines: Vec<Bound<'py, PyBytes>>, threads: usize) -> Out<'py> {
    let slices: Vec<&[u8]> = lines.iter().map(|b| b.as_bytes()).collect();
    let parts = py.allow_threads(|| {
        if threads <= 1 {
            vec![run(&slices, 0)]
        } else {
            let size = slices.len().div_ceil(threads).max(1);
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                slices
                    .par_chunks(size)
                    .enumerate()
                    .map(|(i, part)| run(part, i * size))
                    .collect()
            })
        }
    });
    if parts.len() == 1 {
        return out(py, parts.into_iter().next().unwrap());
    }
    let mut all = Columns::with_room(slices.len(), slices.iter().map(|l| l.len()).sum());
    let mut stats = Stats {
        mismatched: 0,
        entity_fields: BTreeSet::new(),
        sitelink_fields: BTreeSet::new(),
    };
    let mut left = Vec::new();
    for (cols, s, l) in parts {
        for c in 0..7 {
            let shift = all.data[c].len() as i64;
            all.offsets[c].extend(cols.offsets[c][1..].iter().map(|o| o + shift));
            all.data[c].extend_from_slice(&cols.data[c]);
        }
        stats.mismatched += s.mismatched;
        stats.entity_fields.extend(s.entity_fields);
        stats.sitelink_fields.extend(s.sitelink_fields);
        left.extend(l);
    }
    out(py, (all, stats, left))
}

fn out<'py>(py: Python<'py>, (all, stats, left): (Columns, Stats, Vec<usize>)) -> Out<'py> {
    let columns = (0..7)
        .map(|c| {
            let offsets: Vec<u8> = all.offsets[c]
                .iter()
                .flat_map(|o| o.to_le_bytes())
                .collect();
            (PyBytes::new(py, &all.data[c]), PyBytes::new(py, &offsets))
        })
        .collect();
    (
        columns,
        stats.mismatched,
        stats.entity_fields.into_iter().collect(),
        stats.sitelink_fields.into_iter().collect(),
        left,
    )
}

#[pymodule]
fn wikidata_reshape(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(entity_rows, m)?)
}
