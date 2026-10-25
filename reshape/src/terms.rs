//! The labels, descriptions and aliases tables, built from the split's reshaped cells as
//! the pipeline builds them from the chunk's JSON columns: genson's normalisation
//! (`process.normalise_map_direct`: a map of language to string, or to a list of strings,
//! as a list of {key, value}, an empty map null) and then the explode for partitioning
//! (`partitioning.transforms.prepare_for_partition`: rows with a null map dropped, one row
//! per entry, and for the aliases one per string). A row is (entity, language, value), in
//! the entities' order and each map's order.
//!
//! A cell is JSON as orjson writes it (the split's form: dump.entity_row, or reshape's
//! compiled copy of it): no whitespace between tokens, strings UTF-8. Anything other than
//! what these tables hold in the pipeline's schema is a misfit (see `Misfit`), and the
//! chunk is left to the pipeline's path, whose genson check decides it as it does now:
//! a value that is not a string (or for aliases, a list of strings), an empty list of
//! aliases, a null, a key that is empty or not a plain path component (the partition
//! file's directory), anything but an object.

use std::collections::HashMap;

/// A cell outside what the builders reproduce exactly: the row (the entity's place in the
/// chunk) and why
#[derive(Debug, Clone)]
pub struct Misfit {
    pub row: usize,
    pub why: &'static str,
}

pub type R<T = ()> = Result<T, &'static str>;

/// A table's rows: for each, its entity (its index in the chunk), its partition key (an
/// index into `keys`), its value; and for the sitelinks its badges (an empty list where
/// they are null or empty, which genson makes null)
#[derive(Default)]
pub struct Rows {
    pub entity: Vec<u32>,
    pub key: Vec<u32>,
    pub keys: Vec<Vec<u8>>,
    index: HashMap<Vec<u8>, u32>,
    /// the values, row r's at data[ends[r]..ends[r + 1]]
    pub data: Vec<u8>,
    pub ends: Vec<usize>,
    /// the badges: row r's are badges[badge_rows[r]..badge_rows[r + 1]], each at
    /// badge_data[badge_ends[i]..badge_ends[i + 1]]
    pub badge_rows: Vec<u32>,
    pub badge_data: Vec<u8>,
    pub badge_ends: Vec<usize>,
}

impl Rows {
    pub fn new() -> Self {
        Rows { ends: vec![0], badge_rows: vec![0], badge_ends: vec![0], ..Default::default() }
    }
    pub fn len(&self) -> usize {
        self.entity.len()
    }
    pub fn value(&self, r: usize) -> &[u8] {
        &self.data[self.ends[r]..self.ends[r + 1]]
    }
    pub fn badges(&self, r: usize) -> impl Iterator<Item = &[u8]> {
        (self.badge_rows[r] as usize..self.badge_rows[r + 1] as usize)
            .map(move |i| &self.badge_data[self.badge_ends[i]..self.badge_ends[i + 1]])
    }
    pub fn has_badges(&self, r: usize) -> bool {
        self.badge_rows[r + 1] > self.badge_rows[r]
    }
    /// The key's index, interned
    pub fn intern(&mut self, key: &[u8]) -> R<u32> {
        if let Some(&k) = self.index.get(key) {
            return Ok(k);
        }
        check_key(key)?;
        let k = self.keys.len() as u32;
        self.keys.push(key.to_vec());
        self.index.insert(key.to_vec(), k);
        Ok(k)
    }
    /// End the row whose value was just decoded into `data`
    pub fn push(&mut self, entity: usize, key: u32) {
        self.entity.push(entity as u32);
        self.key.push(key);
        self.ends.push(self.data.len());
        self.badge_rows.push(self.badge_ends.len() as u32 - 1);
    }
}

/// A partition key becomes a directory name: it must be a plain path component
fn check_key(key: &[u8]) -> R {
    if key.is_empty() || key == b"." || key == b".." || key.iter().any(|&c| c == b'/' || c == 0) {
        return Err("a key that is not a plain path component");
    }
    Ok(())
}

/// A cell's JSON, read in orjson's compact form
pub struct Cell<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Cell<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Cell { b, i: 0 }
    }
    #[inline]
    pub fn peek(&self) -> R<u8> {
        self.b.get(self.i).copied().ok_or("a cell cut short")
    }
    #[inline]
    pub fn eat(&mut self, c: u8) -> R {
        if self.peek()? != c {
            return Err("not the JSON the split writes");
        }
        self.i += 1;
        Ok(())
    }
    pub fn end(&self) -> R {
        if self.i != self.b.len() {
            return Err("content after the cell's value");
        }
        Ok(())
    }
    /// `{`, and whether the object is empty (its `}` eaten)
    pub fn open(&mut self, open: u8, close: u8) -> R<bool> {
        self.eat(open)?;
        if self.peek()? == close {
            self.i += 1;
            return Ok(true);
        }
        Ok(false)
    }
    /// After a member or element: whether another follows (`,` eaten) or not (`close` eaten)
    pub fn more(&mut self, close: u8) -> R<bool> {
        match self.peek()? {
            b',' => {
                self.i += 1;
                Ok(true)
            }
            c if c == close => {
                self.i += 1;
                Ok(false)
            }
            _ => Err("not the JSON the split writes"),
        }
    }
    /// An object's key, its `:` eaten: borrowed from the cell, or decoded into `scratch`
    /// if it holds an escape
    pub fn key<'s>(&mut self, scratch: &'s mut Vec<u8>) -> R<&'s [u8]>
    where
        'a: 's,
    {
        scratch.clear();
        let plain = self.string(scratch)?;
        self.eat(b':')?;
        Ok(match plain {
            Some(s) => s,
            None => scratch,
        })
    }
    /// A string, decoded: returned as the cell's own bytes if it holds no escape (and
    /// nothing written to `out`), else decoded onto the end of `out`
    pub fn string(&mut self, out: &mut Vec<u8>) -> R<Option<&'a [u8]>> {
        self.eat(b'"')?;
        let start = self.i;
        let b = self.b;
        let mut run = start;
        let mut escaped = false;
        loop {
            let k = memchr::memchr2(b'"', b'\\', &b[self.i..]).ok_or("a string cut short")?;
            let j = self.i + k;
            if b[run..j].iter().any(|&c| c < 0x20) {
                return Err("a control character in a string");
            }
            if b[j] == b'"' {
                self.i = j + 1;
                if !escaped {
                    return Ok(Some(&b[start..j]));
                }
                out.extend_from_slice(&b[run..j]);
                return Ok(None);
            }
            escaped = true;
            out.extend_from_slice(&b[run..j]);
            let c = *b.get(j + 1).ok_or("a string cut short")?;
            self.i = j + 2;
            match c {
                b'"' => out.push(b'"'),
                b'\\' => out.push(b'\\'),
                b'/' => out.push(b'/'),
                b'b' => out.push(8),
                b'f' => out.push(12),
                b'n' => out.push(b'\n'),
                b'r' => out.push(b'\r'),
                b't' => out.push(b'\t'),
                b'u' => {
                    let mut cp = self.hex4()?;
                    if (0xd800..0xdc00).contains(&cp) {
                        if b.get(self.i) != Some(&b'\\') || b.get(self.i + 1) != Some(&b'u') {
                            return Err("a lone surrogate");
                        }
                        self.i += 2;
                        let lo = self.hex4()?;
                        if !(0xdc00..0xe000).contains(&lo) {
                            return Err("a lone surrogate");
                        }
                        cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                    } else if (0xdc00..0xe000).contains(&cp) {
                        return Err("a lone surrogate");
                    }
                    let ch = char::from_u32(cp).ok_or("an invalid escape")?;
                    let mut buf = [0; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                _ => return Err("an invalid escape"),
            }
            run = self.i;
        }
    }
    fn hex4(&mut self) -> R<u32> {
        let h = self.b.get(self.i..self.i + 4).ok_or("an invalid escape")?;
        let mut v = 0;
        for &c in h {
            let d = match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                b'A'..=b'F' => c - b'A' + 10,
                _ => return Err("an invalid escape"),
            };
            v = v << 4 | d as u32;
        }
        self.i += 4;
        Ok(v)
    }
    /// A string value decoded onto the end of `out`
    pub fn string_into(&mut self, out: &mut Vec<u8>) -> R {
        if self.peek()? != b'"' {
            return Err("a value that is not a string");
        }
        if let Some(s) = self.string(out)? {
            out.extend_from_slice(s);
        }
        Ok(())
    }
}

/// One entity's labels or descriptions cell ({lang: value}) appended to `rows`
fn map_cell(rows: &mut Rows, entity: usize, cell: &[u8], scratch: &mut Vec<u8>) -> R {
    let mut p = Cell::new(cell);
    if !p.open(b'{', b'}')? {
        loop {
            let k = rows.intern(p.key(scratch)?)?;
            p.string_into(&mut rows.data)?;
            rows.push(entity, k);
            if !p.more(b'}')? {
                break;
            }
        }
    }
    p.end()
}

/// One entity's aliases cell ({lang: [value]}) appended to `rows`
fn list_map_cell(rows: &mut Rows, entity: usize, cell: &[u8], scratch: &mut Vec<u8>) -> R {
    let mut p = Cell::new(cell);
    if !p.open(b'{', b'}')? {
        loop {
            let k = rows.intern(p.key(scratch)?)?;
            if p.peek()? != b'[' {
                return Err("aliases that are not a list");
            }
            if p.open(b'[', b']')? {
                // genson makes an empty list null, which the explode keeps as a row with a
                // null value: not reproduced here
                return Err("an empty list of aliases");
            }
            loop {
                p.string_into(&mut rows.data)?;
                rows.push(entity, k);
                if !p.more(b']')? {
                    break;
                }
            }
            if !p.more(b'}')? {
                break;
            }
        }
    }
    p.end()
}

/// Which of the term tables a column is
#[derive(Clone, Copy)]
pub enum Terms {
    /// labels, descriptions: {lang: value}
    Map,
    /// aliases: {lang: [value]}
    ListMap,
}

/// A chunk's cells (`cell(i)` the i-th entity's) as the table's rows, or the first cell
/// that does not fit
pub fn build<'a>(kind: Terms, n: usize, cell: impl Fn(usize) -> &'a [u8]) -> Result<Rows, Misfit> {
    let mut rows = Rows::new();
    let mut scratch = Vec::new();
    for i in 0..n {
        let res = match kind {
            Terms::Map => map_cell(&mut rows, i, cell(i), &mut scratch),
            Terms::ListMap => list_map_cell(&mut rows, i, cell(i), &mut scratch),
        };
        res.map_err(|why| Misfit { row: i, why })?;
    }
    Ok(rows)
}
