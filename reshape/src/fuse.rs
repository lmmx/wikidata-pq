//! The split's term and sitelink tables, built and partitioned where the split has the
//! rows (stage 1 of "fusion A": not yet called by dump.py). `fused_partitions` takes a
//! chunk's (or a routed part's) id, labels, descriptions, aliases and sitelinks columns,
//! as the split's reshape writes them (Arrow large strings, JSON text), and writes each
//! table's partition files as `partition_parquet` lays them out after process.py's
//! normalisation (see terms.rs, links.rs, partition.rs). Python's side, with the
//! sidecars, is `wikidata.partitioning.fused`.

#[path = "links.rs"]
pub mod links;
#[path = "partition.rs"]
pub mod partition;
#[path = "terms.rs"]
pub mod terms;

use partition::{Group, Shape, Writer, LINKS, ROW_GROUP_ROWS, TERMS};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use std::collections::HashSet;
use std::ffi::OsStr;
use std::hash::BuildHasherDefault;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use terms::{Fx, Misfit, Rows, Terms};

/// A Python object's bytes, through the buffer protocol (any format, read as bytes)
struct View(ffi::Py_buffer);

impl View {
    fn get(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut view = std::mem::MaybeUninit::<ffi::Py_buffer>::zeroed();
        // SAFETY: a zeroed Py_buffer filled in by the call, released on drop
        unsafe {
            if ffi::PyObject_GetBuffer(obj.as_ptr(), view.as_mut_ptr(), ffi::PyBUF_SIMPLE) == -1 {
                return Err(PyErr::fetch(obj.py()));
            }
            Ok(View(view.assume_init()))
        }
    }
    fn bytes(&self) -> &[u8] {
        if self.0.len == 0 {
            return &[];
        }
        // SAFETY: the exporter keeps buf..buf + len valid while the view is held
        unsafe { std::slice::from_raw_parts(self.0.buf as *const u8, self.0.len as usize) }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        // SAFETY: the view was filled in by PyObject_GetBuffer; dropped with the GIL held
        unsafe { ffi::PyBuffer_Release(&mut self.0) }
    }
}

/// An Arrow large-string array: its data, its int64 offsets, and the array's offset
struct Strings<'a> {
    data: &'a [u8],
    offsets: Vec<i64>,
}

impl<'a> Strings<'a> {
    fn new(data: &'a [u8], offsets: &[u8], offset: usize, n: usize) -> PyResult<Self> {
        let all: Vec<i64> = offsets
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let offsets = all
            .get(offset..offset + n + 1)
            .ok_or_else(|| PyValueError::new_err("an offsets buffer shorter than the column"))?
            .to_vec();
        let ok = offsets.windows(2).all(|w| 0 <= w[0] && w[0] <= w[1]) && offsets[n] as usize <= data.len();
        if !ok {
            return Err(PyValueError::new_err("offsets outside the data buffer"));
        }
        Ok(Strings { data, offsets })
    }
    fn get(&self, i: usize) -> &'a [u8] {
        &self.data[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }
}

/// One table's partition files, by key in the keys' byte order: (key, rows, min id, max
/// id, file bytes, first row group, row groups) — in the one-file layout the bytes are
/// the key's row groups' (see partition::Writer::write_file), else its file's
type KeyFile = (String, u64, String, String, u64, usize, usize);

/// The rows of each key, in the keys' byte order: (key, its rows in their order)
fn by_key(rows: &Rows) -> Vec<(usize, Vec<u32>)> {
    let mut counts = vec![0usize; rows.keys.len()];
    for &k in &rows.key {
        counts[k as usize] += 1;
    }
    let mut order: Vec<usize> = (0..rows.keys.len()).collect();
    order.sort_unstable_by(|&a, &b| rows.keys[a].cmp(&rows.keys[b]));
    let mut slots: Vec<Vec<u32>> = counts.iter().map(|&c| Vec::with_capacity(c)).collect();
    for (r, &k) in rows.key.iter().enumerate() {
        slots[k as usize].push(r as u32);
    }
    order.into_iter().map(|k| (k, std::mem::take(&mut slots[k]))).collect()
}

/// A key's rows as row groups of at most ROW_GROUP_ROWS rows
fn groups<'a>(rows: &'a Rows, key: usize, members: &[u32], ids: &Ids<'a>, shape: &Shape) -> Vec<Group<'a>> {
    members
        .chunks(ROW_GROUP_ROWS)
        .map(|part| Group {
            ids: part.iter().map(|&r| ids.get(rows.entity[r as usize] as usize)).collect(),
            key: &rows.keys[key],
            values: part.iter().map(|&r| rows.value(r as usize)).collect(),
            badges: if shape.badges {
                part.iter()
                    .map(|&r| rows.has_badges(r as usize).then(|| rows.badges(r as usize).collect()))
                    .collect()
            } else {
                Vec::new()
            },
        })
        .collect()
}

/// The distinct ids of the entities a table has a row for (as Polars' n_unique of its
/// file's id column counts them): each entity marked once, then its id hashed once
fn distinct_ids(rows: &Rows, ids: &Ids, n: usize) -> u64 {
    let mut seen = vec![false; n];
    for &e in &rows.entity {
        seen[e as usize] = true;
    }
    let mut set: HashSet<&[u8], BuildHasherDefault<Fx>> = HashSet::default();
    for (e, _) in seen.iter().enumerate().filter(|(_, &s)| s) {
        set.insert(ids.get(e));
    }
    set.len() as u64
}

fn id_bounds(rows: &Rows, members: &[u32], ids: &Ids) -> (String, String) {
    let mut it = members.iter().map(|&r| ids.get(rows.entity[r as usize] as usize));
    let first = it.next().unwrap_or_default();
    let (lo, hi) = it.fold((first, first), |(lo, hi), v| (lo.min(v), hi.max(v)));
    (String::from_utf8_lossy(lo).into_owned(), String::from_utf8_lossy(hi).into_owned())
}

/// Write a table's partition files: one a key at `dst/{key}.parquet`, or with `one_file`
/// one file at `dst`, its row groups by key
fn write_table(
    w: &mut Writer,
    rows: &Rows,
    ids: &Ids,
    shape: &Shape,
    dst: &Path,
    one_file: bool,
) -> std::io::Result<Vec<KeyFile>> {
    let keys = by_key(rows);
    let mut out = Vec::with_capacity(keys.len());
    if one_file {
        let mut all = Vec::new();
        let mut spans = Vec::with_capacity(keys.len());
        for (k, members) in &keys {
            let gs = groups(rows, *k, members, ids, shape);
            spans.push((all.len(), gs.len()));
            all.extend(gs);
        }
        if all.is_empty() {
            return Ok(out); // no rows, no file, as the sink
        }
        let (_, sizes) = w.write_file(dst, shape, &all)?;
        for ((k, members), (first, n)) in keys.iter().zip(spans) {
            let (lo, hi) = id_bounds(rows, members, ids);
            let bytes = sizes[first..first + n].iter().sum();
            let key = String::from_utf8_lossy(&rows.keys[*k]).into_owned();
            out.push((key, members.len() as u64, lo, hi, bytes, first, n));
        }
        return Ok(out);
    }
    let mut path = PathBuf::from(dst);
    let mut name = Vec::new();
    for (k, members) in &keys {
        let gs = groups(rows, *k, members, ids, shape);
        name.clear();
        name.extend_from_slice(&rows.keys[*k]);
        name.extend_from_slice(b".parquet");
        path.push(OsStr::from_bytes(&name));
        let (size, _) = w.write_file(&path, shape, &gs)?;
        path.pop();
        let (lo, hi) = id_bounds(rows, members, ids);
        let key = String::from_utf8_lossy(&rows.keys[*k]).into_owned();
        out.push((key, members.len() as u64, lo, hi, size, 0, gs.len()));
    }
    Ok(out)
}

/// A column passed from Python: (data buffer, int64 offsets buffer, the array's offset)
type ColArg<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>, usize);

/// What each table came to: its key files and its distinct ids, or the first cell that
/// did not fit
type Outcome = Result<(Vec<KeyFile>, u64), Misfit>;

/// The rows a call builds: the chunk's ids, and the positions in the chunk's columns of
/// the part's rows (all of them, in order, if none are given)
struct Ids<'a> {
    ids: Strings<'a>,
    rows: Option<Vec<usize>>,
}

impl<'a> Ids<'a> {
    fn pos(&self, i: usize) -> usize {
        self.rows.as_ref().map_or(i, |r| r[i])
    }
    /// The id of the part's i-th row
    fn get(&self, i: usize) -> &'a [u8] {
        self.ids.get(self.pos(i))
    }
}

/// Build the tables of a chunk's part and write their partition files.
///
/// `len` the chunk's rows; `ids` and the four tables' columns (labels, descriptions,
/// aliases, sitelinks) each as (data, offsets, offset) over the chunk's rows, a table's
/// None to leave it out; `rows` the part's rows (int64 positions in the chunk's columns,
/// in order), or None for all of them; `dsts` where each table's files go (see
/// write_table), a table's None to build it without writing it.
///
/// Every table is built before any file is written: if any cell of any table does not
/// fit, nothing is written, and the result is ("misfit", [(table index, row of the part,
/// why)]). Otherwise it is ("ok", [each table's key files, or None], [each table's
/// distinct ids, or None]): its key files (key, rows, min id, max id, bytes, first row
/// group, row groups), by key; its distinct ids those of the entities it has a row for
/// (for process's id check). The GIL is released throughout.
#[pyfunction]
#[pyo3(signature = (len, ids, columns, dsts, one_file=false, rows=None))]
pub fn fused_partitions<'py>(
    py: Python<'py>,
    len: usize,
    ids: ColArg<'py>,
    columns: Vec<Option<ColArg<'py>>>,
    dsts: Vec<Option<String>>,
    one_file: bool,
    rows: Option<Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    if columns.len() != 4 || dsts.len() != 4 {
        return Err(PyValueError::new_err("four columns and four destinations: labels, descriptions, aliases, sitelinks"));
    }
    let id_views = (View::get(&ids.0)?, View::get(&ids.1)?);
    let mut views = Vec::with_capacity(4);
    for c in &columns {
        views.push(match c {
            Some((d, o, off)) => Some((View::get(d)?, View::get(o)?, *off)),
            None => None,
        });
    }
    let positions = match &rows {
        Some(r) => {
            let v = View::get(r)?;
            let pos: Vec<usize> = v
                .bytes()
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .map(|p| usize::try_from(p).ok().filter(|&p| p < len))
                .collect::<Option<_>>()
                .ok_or_else(|| PyValueError::new_err("a row position outside the chunk"))?;
            Some(pos)
        }
        None => None,
    };
    let n = positions.as_ref().map_or(len, |p| p.len());
    let ids = Ids { ids: Strings::new(id_views.0.bytes(), id_views.1.bytes(), ids.2, len)?, rows: positions };
    let cols = views
        .iter()
        .map(|v| v.as_ref().map(|(d, o, off)| Strings::new(d.bytes(), o.bytes(), *off, len)).transpose())
        .collect::<PyResult<Vec<_>>>()?;
    let result: Result<Vec<Option<Outcome>>, std::io::Error> = py.allow_threads(|| {
        let built: Vec<Option<Result<Rows, Misfit>>> = cols
            .iter()
            .enumerate()
            .map(|(t, c)| {
                c.as_ref().map(|c| {
                    let cell = |i| c.get(ids.pos(i));
                    match t {
                        0 | 1 => terms::build(Terms::Map, n, cell),
                        2 => terms::build(Terms::ListMap, n, cell),
                        _ => links::build(n, cell),
                    }
                })
            })
            .collect();
        if built.iter().flatten().any(|b| b.is_err()) {
            return Ok(built.into_iter().map(|b| b.map(|b| b.map(|_| (Vec::new(), 0)))).collect());
        }
        let mut w = Writer::default();
        let mut out = Vec::with_capacity(4);
        for (t, b) in built.into_iter().enumerate() {
            out.push(match (b, &dsts[t]) {
                (Some(Ok(rows)), Some(dst)) => {
                    let shape = if t == 3 { &LINKS } else { &TERMS };
                    let distinct = distinct_ids(&rows, &ids, n);
                    Some(Ok((write_table(&mut w, &rows, &ids, shape, Path::new(dst), one_file)?, distinct)))
                }
                _ => None,
            });
        }
        Ok(out)
    });
    let outcomes = result.map_err(|e| PyOSError::new_err(e.to_string()))?;
    let misfits: Vec<(usize, usize, &str)> = outcomes
        .iter()
        .enumerate()
        .filter_map(|(t, o)| match o {
            Some(Err(m)) => Some((t, m.row, m.why)),
            _ => None,
        })
        .collect();
    if !misfits.is_empty() {
        return Ok(("misfit", misfits).into_pyobject(py)?.into_any().unbind());
    }
    let (files, distinct): (Vec<Option<Vec<KeyFile>>>, Vec<Option<u64>>) =
        outcomes.into_iter().map(|o| o.map(|r| r.unwrap())).map(|o| o.map_or((None, None), |(f, d)| (Some(f), Some(d)))).unzip();
    Ok(("ok", files, distinct).into_pyobject(py)?.into_any().unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fused_partitions, m)?)
}
