//! Partition files for the term and sitelink tables, written straight from the split's
//! rows (see fuse.rs): Parquet in the Parquet schema and Arrow schema Polars' sink writes
//! them in (`partitioning.core.partition_parquet`), the same rows in the same order and
//! the same row groups (at most ROW_GROUP_ROWS rows each), zstd at Polars' default level.
//!
//! The bytes are not Polars' (its encoder chooses dictionaries and pages its own way):
//! the files are logically identical, which `scripts/test_fused_terms.py` checks on every
//! bench chunk. What makes them cheap is what the sink pays for each file and this does
//! not: one zstd context for every page of every file (the sink sets one up for each
//! column chunk), and a file built in memory and written with one call.
//!
//! The format written is the plain core of Parquet: data pages v1, PLAIN values, levels
//! in RLE runs, the key column (one value a file) as a one-entry dictionary, statistics
//! (null count, min and max) on each column chunk, and the footer in Thrift's compact
//! protocol.

use std::fs;
use std::io::{self, Write};
use std::path::Path;

/// Rows per row group: as many as Polars' sink puts in one
/// (`partitioning.core.ARROW_ROW_GROUP_ROWS`)
pub const ROW_GROUP_ROWS: usize = 122_880;
/// A data page holds about this many bytes of values before the next starts
const PAGE_BYTES: usize = 1024 * 1024;
/// Polars' default zstd level
const ZSTD_LEVEL: i32 = 3;
/// Statistics are left out for a column chunk whose min or max is longer than this (Polars
/// truncates longer ones; nothing downstream reads them but the ids', which are short)
const STAT_BYTES: usize = 64;
const CREATED_BY: &str = "wikidata-reshape 0.1.0 (fused partition writer)";

/// The Arrow schemas Polars' sink stores in the files (its `ARROW:schema` footer key, as
/// written by Polars 2.0.0), so that every reader decodes these files to the same Arrow
/// types as the sink's: large strings, and a large list of them for the badges
const TERMS_ARROW_SCHEMA: &str = "/////8cAAAAEAAAA8v///xQAAAAEAAEAAAAKAAsACAAKAAQA+P///wwAAAAIAAgAAAAEAAMAAABgAAAAMAAAAAQAAACY////GAAAABAAAAAIAAAAARQAAAAAAACY////BQAAAHZhbHVlAAAAwP///xgAAAAQAAAACAAAAAEUAAAAAAAAwP///wgAAABsYW5ndWFnZQAAAADs////LAAAACAAAAAYAAAAARQAABAAEgAEABAAEQAIAAAADAAAAAAA/P///wQABAACAAAAaWQA";
const LINKS_ARROW_SCHEMA: &str = "/////xsBAAAEAAAA8v///xQAAAAEAAEAAAAKAAsACAAKAAQA+P///wwAAAAIAAgAAAAEAAQAAAC0AAAAiAAAAFwAAAAEAAAASP///0QAAAA8AAAACAAAAAEVAAABAAAABAAAAGT///8YAAAAEAAAAAgAAAABFAAAAAAAAGT///8EAAAAaXRlbQAAAAB0////BgAAAGJhZGdlcwAAnP///xgAAAAQAAAACAAAAAEUAAAAAAAAnP///wUAAAB0aXRsZQAAAMT///8YAAAAEAAAAAgAAAABFAAAAAAAAMT///8EAAAAc2l0ZQAAAADs////LAAAACAAAAAYAAAAARQAABAAEgAEABAAEQAIAAAADAAAAAAA/P///wQABAACAAAAaWQA";

/// The shape of a table's files: `id`, the key column (`language` or `site`), the value
/// column (`value` or `title`), and for the sitelinks a `badges` list of strings
#[derive(Clone, Copy)]
pub struct Shape {
    pub key: &'static str,
    pub value: &'static str,
    pub badges: bool,
}

pub const TERMS: Shape = Shape { key: "language", value: "value", badges: false };
pub const LINKS: Shape = Shape { key: "site", value: "title", badges: true };

impl Shape {
    fn arrow_schema(&self) -> &'static str {
        if self.badges {
            LINKS_ARROW_SCHEMA
        } else {
            TERMS_ARROW_SCHEMA
        }
    }
}

/// One row group's columns: the ids, the key (one value for every row), the values, and
/// (for the sitelinks) each row's badges, None for a null list
pub struct Group<'a> {
    pub ids: Vec<&'a [u8]>,
    pub key: &'a [u8],
    pub values: Vec<&'a [u8]>,
    pub badges: Vec<Option<Vec<&'a [u8]>>>,
}

// Parquet's enums (parquet.thrift)
const BYTE_ARRAY: i32 = 6;
const OPTIONAL: i32 = 1;
const REPEATED: i32 = 2;
const UTF8: i32 = 0;
const LIST: i32 = 3;
const PLAIN: i32 = 0;
const RLE: i32 = 3;
const RLE_DICTIONARY: i32 = 8;
const ZSTD: i32 = 6;
const DATA_PAGE: i32 = 0;
const DICTIONARY_PAGE: i32 = 2;

/// Thrift's compact protocol, as much of it as a Parquet footer and page header use
mod thrift {
    pub const BOOL_TRUE: u8 = 1;
    pub const I16: u8 = 4;
    pub const I32: u8 = 5;
    pub const I64: u8 = 6;
    pub const BINARY: u8 = 8;
    pub const LIST: u8 = 9;
    pub const STRUCT: u8 = 12;

    pub struct W<'a> {
        pub out: &'a mut Vec<u8>,
        last: Vec<i16>,
        cur: i16,
    }

    impl<'a> W<'a> {
        pub fn new(out: &'a mut Vec<u8>) -> Self {
            W { out, last: Vec::with_capacity(8), cur: 0 }
        }
        pub fn varint(&mut self, mut v: u64) {
            while v >= 0x80 {
                self.out.push(v as u8 | 0x80);
                v >>= 7;
            }
            self.out.push(v as u8);
        }
        fn zigzag(&mut self, v: i64) {
            self.varint(((v << 1) ^ (v >> 63)) as u64);
        }
        fn field(&mut self, id: i16, ty: u8) {
            let delta = id - self.cur;
            if (1..=15).contains(&delta) {
                self.out.push((delta as u8) << 4 | ty);
            } else {
                self.out.push(ty);
                self.zigzag(id as i64);
            }
            self.cur = id;
        }
        pub fn i32(&mut self, id: i16, v: i32) {
            self.field(id, I32);
            self.zigzag(v as i64);
        }
        pub fn i16(&mut self, id: i16, v: i16) {
            self.field(id, I16);
            self.zigzag(v as i64);
        }
        pub fn i64(&mut self, id: i16, v: i64) {
            self.field(id, I64);
            self.zigzag(v);
        }
        pub fn bool_true(&mut self, id: i16) {
            self.field(id, BOOL_TRUE);
        }
        pub fn binary(&mut self, id: i16, v: &[u8]) {
            self.field(id, BINARY);
            self.bytes(v);
        }
        /// A binary as a list element (no field header)
        pub fn bytes(&mut self, v: &[u8]) {
            self.varint(v.len() as u64);
            self.out.extend_from_slice(v);
        }
        /// An i32 as a list element
        pub fn elem_i32(&mut self, v: i32) {
            self.zigzag(v as i64);
        }
        pub fn begin(&mut self, id: i16) {
            self.field(id, STRUCT);
            self.enter();
        }
        /// A struct as a list element
        pub fn enter(&mut self) {
            self.last.push(self.cur);
            self.cur = 0;
        }
        pub fn end(&mut self) {
            self.out.push(0);
            self.cur = self.last.pop().unwrap_or(0);
        }
        pub fn list(&mut self, id: i16, elem: u8, n: usize) {
            self.field(id, LIST);
            if n < 15 {
                self.out.push((n as u8) << 4 | elem);
            } else {
                self.out.push(0xf0 | elem);
                self.varint(n as u64);
            }
        }
        /// The top-level struct's stop
        pub fn stop(&mut self) {
            self.out.push(0);
        }
    }
}

/// A column chunk's metadata, for the footer
struct Chunk {
    path: &'static [&'static str],
    encodings: &'static [i32],
    num_values: i64,
    uncompressed: i64,
    compressed: i64,
    data_page_offset: i64,
    dictionary_page_offset: Option<i64>,
    null_count: i64,
    min_max: Option<(Vec<u8>, Vec<u8>)>,
}

impl Chunk {
    fn start(&self) -> i64 {
        self.dictionary_page_offset.unwrap_or(self.data_page_offset)
    }
}

struct RowGroupMeta {
    chunks: Vec<Chunk>,
    num_rows: i64,
}

/// Levels as RLE runs (Parquet's RLE/bit-packing hybrid, its RLE runs only), each value
/// in one byte (bit widths 1 and 2)
fn rle_runs(out: &mut Vec<u8>, levels: impl Iterator<Item = u8>) {
    let mut levels = levels.peekable();
    while let Some(v) = levels.next() {
        let mut n: u64 = 1;
        while levels.peek() == Some(&v) {
            levels.next();
            n += 1;
        }
        let mut h = n << 1;
        while h >= 0x80 {
            out.push(h as u8 | 0x80);
            h >>= 7;
        }
        out.push(h as u8);
        out.push(v);
    }
}

/// `n` levels of one value, as one RLE run
fn rle_run(out: &mut Vec<u8>, n: usize, v: u8) {
    if n > 0 {
        rle_runs(out, std::iter::repeat_n(v, n));
    }
}

/// A v1 data page's level block: its length (4 bytes, little-endian), then its runs
fn level_block(out: &mut Vec<u8>, runs: impl FnOnce(&mut Vec<u8>)) {
    let at = out.len();
    out.extend_from_slice(&[0; 4]);
    runs(out);
    let len = (out.len() - at - 4) as u32;
    out[at..at + 4].copy_from_slice(&len.to_le_bytes());
}

fn plain(out: &mut Vec<u8>, v: &[u8]) {
    out.extend_from_slice(&(v.len() as u32).to_le_bytes());
    out.extend_from_slice(v);
}

/// The min and max of `values`, by their bytes (Parquet's order for UTF-8 strings), or
/// None if there are none or one is longer than STAT_BYTES
fn min_max<'a>(values: impl Iterator<Item = &'a [u8]>) -> Option<(Vec<u8>, Vec<u8>)> {
    let mut lo: Option<&[u8]> = None;
    let mut hi: Option<&[u8]> = None;
    for v in values {
        if lo.is_none_or(|l| v < l) {
            lo = Some(v);
        }
        if hi.is_none_or(|h| v > h) {
            hi = Some(v);
        }
    }
    let (lo, hi) = (lo?, hi?);
    (lo.len() <= STAT_BYTES && hi.len() <= STAT_BYTES).then(|| (lo.to_vec(), hi.to_vec()))
}

/// Writes partition files, one after another, reusing its zstd context and buffers
pub struct Writer {
    cctx: zstd_safe::CCtx<'static>,
    /// the page being built, and its compressed form
    page: Vec<u8>,
    packed: Vec<u8>,
    /// the file being built
    file: Vec<u8>,
}

impl Default for Writer {
    fn default() -> Self {
        Writer {
            cctx: zstd_safe::CCtx::create(),
            page: Vec::new(),
            packed: Vec::new(),
            file: Vec::new(),
        }
    }
}

impl Writer {
    /// Compress the page built in `self.page` and append it to the file with its header
    /// (its type and sizes, then what `header` writes);
    /// returns its offset and its header's and body's sizes (uncompressed, compressed)
    fn flush_page(&mut self, page_type: i32, header: impl FnOnce(&mut thrift::W)) -> io::Result<(i64, i64, i64)> {
        let bound = zstd_safe::compress_bound(self.page.len());
        self.packed.clear();
        self.packed.reserve(bound);
        self.cctx
            .compress(&mut self.packed, &self.page, ZSTD_LEVEL)
            .map_err(|code| io::Error::other(zstd_safe::get_error_name(code)))?;
        let offset = self.file.len() as i64;
        let (raw, packed) = (self.page.len(), self.packed.len());
        let mut w = thrift::W::new(&mut self.file);
        w.i32(1, page_type);
        w.i32(2, raw as i32);
        w.i32(3, packed as i32);
        header(&mut w);
        w.stop();
        let header_len = self.file.len() as i64 - offset;
        self.file.extend_from_slice(&self.packed);
        Ok((offset, header_len + raw as i64, header_len + packed as i64))
    }

    /// A column of strings, none null, as PLAIN data pages
    fn strings(&mut self, path: &'static [&'static str], values: &[&[u8]]) -> io::Result<Chunk> {
        let mut chunk = Chunk {
            path,
            encodings: &[PLAIN, RLE],
            num_values: values.len() as i64,
            uncompressed: 0,
            compressed: 0,
            data_page_offset: -1,
            dictionary_page_offset: None,
            null_count: 0,
            min_max: min_max(values.iter().copied()),
        };
        let mut start = 0;
        while start < values.len() {
            let mut end = start;
            let mut bytes = 0;
            while end < values.len() && (end == start || bytes < PAGE_BYTES) {
                bytes += values[end].len() + 4;
                end += 1;
            }
            let n = end - start;
            self.page.clear();
            level_block(&mut self.page, |o| rle_run(o, n, 1));
            for v in &values[start..end] {
                plain(&mut self.page, v);
            }
            let (offset, raw, packed) = self.flush_page(DATA_PAGE, |w| {
                w.begin(5);
                w.i32(1, n as i32);
                w.i32(2, PLAIN);
                w.i32(3, RLE);
                w.i32(4, RLE);
                w.end();
            })?;
            if chunk.data_page_offset < 0 {
                chunk.data_page_offset = offset;
            }
            chunk.uncompressed += raw;
            chunk.compressed += packed;
            start = end;
        }
        Ok(chunk)
    }

    /// A column holding one value in each of its `n` rows: a one-entry dictionary, and a
    /// data page of its indices, all 0
    fn constant(&mut self, path: &'static [&'static str], value: &[u8], n: usize) -> io::Result<Chunk> {
        self.page.clear();
        plain(&mut self.page, value);
        let (dict_offset, raw0, packed0) = self.flush_page(DICTIONARY_PAGE, |w| {
            w.begin(7);
            w.i32(1, 1);
            w.i32(2, PLAIN);
            w.end();
        })?;
        self.page.clear();
        level_block(&mut self.page, |o| rle_run(o, n, 1));
        self.page.push(1); // the indices' bit width
        rle_run(&mut self.page, n, 0);
        let (data_offset, raw1, packed1) = self.flush_page(DATA_PAGE, |w| {
            w.begin(5);
            w.i32(1, n as i32);
            w.i32(2, RLE_DICTIONARY);
            w.i32(3, RLE);
            w.i32(4, RLE);
            w.end();
        })?;
        Ok(Chunk {
            path,
            encodings: &[PLAIN, RLE, RLE_DICTIONARY],
            num_values: n as i64,
            uncompressed: raw0 + raw1,
            compressed: packed0 + packed1,
            data_page_offset: data_offset,
            dictionary_page_offset: Some(dict_offset),
            null_count: 0,
            min_max: (value.len() <= STAT_BYTES).then(|| (value.to_vec(), value.to_vec())),
        })
    }

    /// A list of strings a row (None a null list), its elements never null: repetition
    /// levels (bit width 1) and definition levels (bit width 2: 0 a null list, 3 an
    /// element) and the elements PLAIN, the pages split between rows. The null count is
    /// of null elements (none), as Polars counts it, not of null lists.
    fn string_lists(&mut self, path: &'static [&'static str], rows: &[Option<Vec<&[u8]>>]) -> io::Result<Chunk> {
        let levels = |r: &Option<Vec<&[u8]>>| r.as_ref().map_or(1, |l| l.len().max(1));
        let mut chunk = Chunk {
            path,
            encodings: &[PLAIN, RLE],
            num_values: rows.iter().map(levels).sum::<usize>() as i64,
            uncompressed: 0,
            compressed: 0,
            data_page_offset: -1,
            dictionary_page_offset: None,
            null_count: 0,
            min_max: min_max(rows.iter().flatten().flatten().copied()),
        };
        let mut start = 0;
        while start < rows.len() {
            let mut end = start;
            let mut bytes = 0;
            while end < rows.len() && (end == start || bytes < PAGE_BYTES) {
                bytes += rows[end].iter().flatten().map(|v| v.len() + 4).sum::<usize>() + 2;
                end += 1;
            }
            let part = &rows[start..end];
            let n: usize = part.iter().map(levels).sum();
            self.page.clear();
            level_block(&mut self.page, |o| {
                rle_runs(o, part.iter().flat_map(|r| {
                    let k = r.as_ref().map_or(1, |l| l.len().max(1));
                    (0..k).map(|i| (i > 0) as u8)
                }))
            });
            level_block(&mut self.page, |o| {
                rle_runs(o, part.iter().flat_map(|r| {
                    let (k, d) = match r {
                        None => (1, 0),
                        Some(l) if l.is_empty() => (1, 1),
                        Some(l) => (l.len(), 3),
                    };
                    std::iter::repeat_n(d, k)
                }))
            });
            for v in part.iter().flatten().flatten() {
                plain(&mut self.page, v);
            }
            let (offset, raw, packed) = self.flush_page(DATA_PAGE, |w| {
                w.begin(5);
                w.i32(1, n as i32);
                w.i32(2, PLAIN);
                w.i32(3, RLE);
                w.i32(4, RLE);
                w.end();
            })?;
            if chunk.data_page_offset < 0 {
                chunk.data_page_offset = offset;
            }
            chunk.uncompressed += raw;
            chunk.compressed += packed;
            start = end;
        }
        Ok(chunk)
    }

    fn row_group(&mut self, shape: &Shape, g: &Group) -> io::Result<RowGroupMeta> {
        let n = g.ids.len();
        let key_path: &'static [&'static str] = if shape.key == "site" { &["site"] } else { &["language"] };
        let value_path: &'static [&'static str] = if shape.value == "title" { &["title"] } else { &["value"] };
        let mut chunks = vec![
            self.strings(&["id"], &g.ids)?,
            self.constant(key_path, g.key, n)?,
            self.strings(value_path, &g.values)?,
        ];
        if shape.badges {
            chunks.push(self.string_lists(&["badges", "list", "element"], &g.badges)?);
        }
        Ok(RowGroupMeta { chunks, num_rows: n as i64 })
    }

    fn footer(&mut self, shape: &Shape, groups: &[RowGroupMeta]) {
        let start = self.file.len();
        let num_rows: i64 = groups.iter().map(|g| g.num_rows).sum();
        let mut w = thrift::W::new(&mut self.file);
        w.i32(1, 1); // version
        // the schema, depth first: root, id, key, value[, badges, list, element]
        let leaves = 3 + shape.badges as usize;
        let n_elements = if shape.badges { 7 } else { 4 };
        w.list(2, thrift::STRUCT, n_elements);
        w.enter();
        w.binary(4, b"root");
        w.i32(5, leaves as i32);
        w.end();
        for name in ["id", shape.key, shape.value] {
            string_element(&mut w, name.as_bytes());
        }
        if shape.badges {
            w.enter();
            w.i32(3, OPTIONAL);
            w.binary(4, b"badges");
            w.i32(5, 1);
            w.i32(6, LIST);
            w.begin(10);
            w.begin(3); // LogicalType.LIST
            w.end();
            w.end();
            w.end();
            w.enter();
            w.i32(3, REPEATED);
            w.binary(4, b"list");
            w.i32(5, 1);
            w.end();
            string_element(&mut w, b"element");
        }
        w.i64(3, num_rows);
        w.list(4, thrift::STRUCT, groups.len());
        for (ordinal, g) in groups.iter().enumerate() {
            w.enter();
            w.list(1, thrift::STRUCT, g.chunks.len());
            for c in &g.chunks {
                w.enter();
                w.i64(2, c.start());
                w.begin(3);
                w.i32(1, BYTE_ARRAY);
                w.list(2, thrift::I32, c.encodings.len());
                for &e in c.encodings {
                    w.elem_i32(e);
                }
                w.list(3, thrift::BINARY, c.path.len());
                for p in c.path {
                    w.bytes(p.as_bytes());
                }
                w.i32(4, ZSTD);
                w.i64(5, c.num_values);
                w.i64(6, c.uncompressed);
                w.i64(7, c.compressed);
                w.i64(9, c.data_page_offset);
                if let Some(d) = c.dictionary_page_offset {
                    w.i64(11, d);
                }
                w.begin(12);
                w.i64(3, c.null_count);
                if let Some((lo, hi)) = &c.min_max {
                    w.binary(5, hi);
                    w.binary(6, lo);
                    w.bool_true(7);
                    w.bool_true(8);
                }
                w.end();
                w.end(); // meta_data
                w.end(); // column chunk
            }
            w.i64(2, g.chunks.iter().map(|c| c.uncompressed).sum());
            w.i64(3, g.num_rows);
            w.i64(5, g.chunks[0].start());
            w.i64(6, g.chunks.iter().map(|c| c.compressed).sum());
            w.i16(7, ordinal as i16);
            w.end();
        }
        w.list(5, thrift::STRUCT, 1);
        w.enter();
        w.binary(1, b"ARROW:schema");
        w.binary(2, shape.arrow_schema().as_bytes());
        w.end();
        w.binary(6, CREATED_BY.as_bytes());
        // column_orders: TypeDefinedOrder for each leaf, so readers trust min/max
        w.list(7, thrift::STRUCT, leaves);
        for _ in 0..leaves {
            w.enter();
            w.begin(1);
            w.end();
            w.end();
        }
        w.stop();
        let len = (self.file.len() - start) as u32;
        self.file.extend_from_slice(&len.to_le_bytes());
        self.file.extend_from_slice(b"PAR1");
    }

    /// Write `groups` (each at most ROW_GROUP_ROWS rows) as one file at `path`, made in
    /// memory and written in one call (its directory made if missing). Returns its size
    /// and each row group's bytes (the row group's compressed column chunks; the first
    /// also counts the file's magic and the last its footer, so they add up to the file).
    pub fn write_file(&mut self, path: &Path, shape: &Shape, groups: &[Group]) -> io::Result<(u64, Vec<u64>)> {
        self.file.clear();
        self.file.extend_from_slice(b"PAR1");
        let mut metas = Vec::with_capacity(groups.len());
        let mut sizes = Vec::with_capacity(groups.len());
        for g in groups {
            let at = self.file.len();
            metas.push(self.row_group(shape, g)?);
            sizes.push((self.file.len() - at) as u64);
        }
        let body = self.file.len();
        self.footer(shape, &metas);
        if let Some(first) = sizes.first_mut() {
            *first += 4;
        }
        if let Some(last) = sizes.last_mut() {
            *last += (self.file.len() - body) as u64;
        }
        write_all(path, &self.file)?;
        Ok((self.file.len() as u64, sizes))
    }
}

fn string_element(w: &mut thrift::W, name: &[u8]) {
    w.enter();
    w.i32(1, BYTE_ARRAY);
    w.i32(3, OPTIONAL);
    w.binary(4, name);
    w.i32(6, UTF8);
    w.begin(10);
    w.begin(1); // LogicalType.STRING
    w.end();
    w.end();
    w.end();
}

/// Write a file, making its directory only when it is missing (one call in the usual case)
fn write_all(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut f = match fs::File::create(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            if let Some(dir) = path.parent() {
                fs::create_dir_all(dir)?;
            }
            fs::File::create(path)?
        }
        Err(e) => return Err(e),
    };
    f.write_all(bytes)
}
