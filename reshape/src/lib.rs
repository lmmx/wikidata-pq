//! dump.entity_row in one compiled pass over each dump line's bytes: the line is
//! scanned once (validated as orjson.loads validates it), and its row written as
//! orjson writes it, straight from the line. Nearly all of a row's bytes are the line's
//! own (keys, structure, unescaped strings, integers, true/false/null), so they are
//! copied as runs of the line; only what orjson writes differently is rewritten:
//! a string holding escapes (the dumps write all non-ASCII as \uXXXX and / as \/; orjson
//! writes UTF-8 and escapes only ", \ and control characters) and a number orjson
//! rewrites (any but an integer in 64 bits: floats by ryu, -0 as 0). The reshape's
//! changes of shape (terms, datavalues) are made as the line is scanned.
//!
//! A line outside what the reshape expects (a field of the wrong type, whitespace
//! between tokens, an escape in a key, a repeated entity field, deep nesting) is left
//! for dump.entity_row to reshape (or reject), so the rows are the same whatever the
//! dump holds.
use memchr::{memchr, memchr2};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::os::raw::c_int;

mod bz2; // the bz2 block scanner of the split's sweep reader (bz2.rs)
mod claims;
mod claims_arrow;
mod fuse; // the term and sitelink tables built and partitioned in the split (fuse.rs)

/// A line the fast path leaves to dump.entity_row
struct Fallback;
type R<T = ()> = Result<T, Fallback>;

/// Nesting deeper than this (an entity's is about 8) is left to dump.entity_row
/// (orjson.dumps refuses nesting past 254)
const MAX_DEPTH: u32 = 128;

/// Write `c` (a decoded character) as orjson does: UTF-8, but ", \ and the control
/// characters escaped (\b \f \n \r \t by name, the rest as \u00xx)
#[inline]
fn push_char(out: &mut Vec<u8>, c: u32) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    match c {
        0x22 => out.extend_from_slice(b"\\\""),
        0x5c => out.extend_from_slice(b"\\\\"),
        0x0a => out.extend_from_slice(b"\\n"),
        0x0d => out.extend_from_slice(b"\\r"),
        0x09 => out.extend_from_slice(b"\\t"),
        0x08 => out.extend_from_slice(b"\\b"),
        0x0c => out.extend_from_slice(b"\\f"),
        0..=0x1f => out.extend_from_slice(&[
            b'\\',
            b'u',
            b'0',
            b'0',
            HEX[(c >> 4) as usize],
            HEX[(c & 15) as usize],
        ]),
        0x20..=0x7f => out.push(c as u8),
        _ => {
            let mut buf = [0; 4];
            // c is a scalar value: surrogates are paired or refused before this
            let ch = char::from_u32(c).unwrap_or('\u{fffd}');
            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        }
    }
}

#[inline]
fn hex4(s: &[u8], i: usize) -> R<u32> {
    let h = s.get(i..i + 4).ok_or(Fallback)?;
    let mut v = 0;
    for &c in h {
        let d = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => return Err(Fallback),
        };
        v = v << 4 | d as u32;
    }
    Ok(v)
}

/// A string's content `s` (between its quotes, holding escapes) as orjson writes the
/// string it decodes to, quoted. Refuses what orjson.loads refuses: an unknown escape,
/// a \u without 4 hex digits, a high surrogate not followed by \u and a low one, a low
/// surrogate alone.
fn transcode(out: &mut Vec<u8>, s: &[u8]) -> R {
    out.push(b'"');
    let mut i = 0;
    while let Some(k) = memchr(b'\\', &s[i..]) {
        let j = i + k;
        out.extend_from_slice(&s[i..j]);
        let c = *s.get(j + 1).ok_or(Fallback)?;
        i = j + 2;
        match c {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            b'/' => out.push(b'/'),
            b'b' => out.extend_from_slice(b"\\b"),
            b'f' => out.extend_from_slice(b"\\f"),
            b'n' => out.extend_from_slice(b"\\n"),
            b'r' => out.extend_from_slice(b"\\r"),
            b't' => out.extend_from_slice(b"\\t"),
            b'u' => {
                let mut cp = hex4(s, i)?;
                i += 4;
                if (0xd800..0xdc00).contains(&cp) {
                    if s.get(i) != Some(&b'\\') || s.get(i + 1) != Some(&b'u') {
                        return Err(Fallback);
                    }
                    let lo = hex4(s, i + 2)?;
                    if !(0xdc00..0xe000).contains(&lo) {
                        return Err(Fallback);
                    }
                    cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                    i += 6;
                } else if (0xdc00..0xe000).contains(&cp) {
                    return Err(Fallback);
                }
                push_char(out, cp);
            }
            _ => return Err(Fallback),
        }
    }
    out.extend_from_slice(&s[i..]);
    out.push(b'"');
    Ok(())
}

/// A number as orjson writes what it parses it as: an integer in 64 bits as it is (but
/// -0 as 0), any other number as a double, which orjson writes as ryu does but with a +
/// on a positive exponent (1e+16)
fn write_number(out: &mut Vec<u8>, raw: &[u8]) -> R {
    // raw is ASCII (checked by the scan)
    let raw = std::str::from_utf8(raw).map_err(|_| Fallback)?;
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

/// The first " or \\ in `b` from `from`: most strings are short (keys, ids, languages),
/// so their first 16-byte blocks are looked at here (SSE2, which every x86-64 has),
/// before memchr2's search (whose call costs more than a short string's scan)
#[inline(always)]
fn find_end(b: &[u8], from: usize) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        let mut i = from;
        // SAFETY: SSE2 is part of x86-64; each load reads b[i..i + 16], within b
        unsafe {
            let quote = _mm_set1_epi8(b'"' as i8);
            let slash = _mm_set1_epi8(b'\\' as i8);
            while i + 16 <= b.len() && i < from + 64 {
                let v = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
                let hit = _mm_or_si128(_mm_cmpeq_epi8(v, quote), _mm_cmpeq_epi8(v, slash));
                let mask = _mm_movemask_epi8(hit) as u32;
                if mask != 0 {
                    return Some(i + mask.trailing_zeros() as usize);
                }
                i += 16;
            }
        }
        memchr2(b'"', b'\\', &b[i..]).map(|k| i + k)
    }
    #[cfg(not(target_arch = "x86_64"))]
    memchr2(b'"', b'\\', &b[from..]).map(|k| from + k)
}

/// The scan of one line (UTF-8, with no byte below 0x20: checked before the scan)
struct Line<'a> {
    b: &'a [u8],
    i: usize,
    /// The keys of the objects open in the scan (where each key is in the line), and
    /// where each open object's keys start: a key repeated in an object is left to
    /// dump.entity_row (orjson keeps the last value, in the first's place)
    keys: Vec<Key>,
    bases: Vec<usize>,
    /// Where the innermost open object's keys start in `keys`
    base: usize,
}

/// A key kept for the check for repeats: where it is in the line, its length, and its
/// first 8 bytes (so most comparisons are of two integers)
#[derive(Clone, Copy)]
struct Key {
    start: u32,
    len: u32,
    head: u64,
}

impl Key {
    #[inline(always)]
    fn new(b: &[u8], start: usize, end: usize) -> Self {
        let len = end - start;
        // a key is followed by at least `":` and a value, so 8 bytes are nearly always there
        let head = match b.get(start..start + 8) {
            Some(h) => {
                let h = u64::from_le_bytes(h.try_into().unwrap());
                if len >= 8 {
                    h
                } else {
                    h & ((1u64 << (8 * len)) - 1)
                }
            }
            None => {
                let mut h = [0; 8];
                h[..len.min(8)].copy_from_slice(&b[start..start + len.min(8)]);
                u64::from_le_bytes(h)
            }
        };
        Key {
            start: start as u32,
            len: len as u32,
            head,
        }
    }
    #[inline]
    fn text<'b>(&self, b: &'b [u8]) -> &'b [u8] {
        &b[self.start as usize..(self.start + self.len) as usize]
    }
    #[inline(always)]
    fn same(&self, other: &Key, b: &[u8]) -> bool {
        self.head == other.head
            && self.len == other.len
            && (self.len <= 8 || self.text(b) == other.text(b))
    }
}

/// An object's keys are each compared with those before them until it has this many;
/// an object with more (a map of terms, sitelinks or claims) has its keys sorted once it
/// closes, and any repeat is found then
const FEW_KEYS: usize = 8;

/// A string token's content: b[start..end], between its quotes; `esc` if it holds a \
struct Str {
    start: usize,
    end: usize,
    esc: bool,
}

impl<'a> Line<'a> {
    #[inline]
    fn peek(&self) -> R<u8> {
        self.b.get(self.i).copied().ok_or(Fallback)
    }

    #[inline]
    fn eat(&mut self, c: u8) -> R {
        if self.peek()? == c {
            self.i += 1;
            Ok(())
        } else {
            Err(Fallback)
        }
    }

    /// After an object's or array's member: true at a comma (another member follows),
    /// false at `close`
    #[inline]
    fn more(&mut self, close: u8) -> R<bool> {
        let c = self.peek()?;
        self.i += 1;
        if c == b',' {
            Ok(true)
        } else if c == close {
            Ok(false)
        } else {
            Err(Fallback)
        }
    }

    /// Past an opening `open`; true if the object or array is empty (and past its close)
    #[inline]
    fn open(&mut self, open: u8, close: u8) -> R<bool> {
        self.eat(open)?;
        if self.peek()? == close {
            self.i += 1;
            return Ok(true);
        }
        Ok(false)
    }

    /// The string token here (its escapes are checked only when it is transcoded)
    #[inline]
    fn string(&mut self) -> R<Str> {
        self.eat(b'"')?;
        let start = self.i;
        let mut esc = false;
        loop {
            let j = find_end(self.b, self.i).ok_or(Fallback)?;
            if self.b[j] == b'"' {
                self.i = j + 1;
                return Ok(Str { start, end: j, esc });
            }
            esc = true;
            self.i = j + 2; // past the escaped character
            if self.i > self.b.len() {
                return Err(Fallback);
            }
        }
    }

    /// Past an object's opening brace; true if it is empty (and past its close)
    #[inline]
    fn obj(&mut self) -> R<bool> {
        if self.open(b'{', b'}')? {
            return Ok(true);
        }
        self.bases.push(self.base);
        self.base = self.keys.len();
        Ok(false)
    }

    /// After an object's member: true at a comma, false at its close (past it, and its
    /// keys checked: see obj_end)
    #[inline]
    fn obj_more(&mut self) -> R<bool> {
        if self.peek()? == b'}' {
            self.obj_end()?;
            self.i += 1;
            return Ok(false);
        }
        self.eat(b',')?;
        Ok(true)
    }

    /// At an object's close: its keys checked for a repeat, and let go
    #[inline]
    fn obj_end(&mut self) -> R {
        let base = self.base;
        self.base = self.bases.pop().ok_or(Fallback)?;
        let b = self.b;
        let keys = &mut self.keys[base..];
        if keys.len() > FEW_KEYS {
            keys.sort_unstable_by(|x, y| {
                (x.head, x.len)
                    .cmp(&(y.head, y.len))
                    .then_with(|| x.text(b).cmp(y.text(b)))
            });
            if keys.windows(2).any(|w| w[0].same(&w[1], b)) {
                return Err(Fallback);
            }
        }
        self.keys.truncate(base);
        Ok(())
    }

    /// An object key and its colon: the key's text (a key holding an escape is left to
    /// dump.entity_row), and where the key's opening quote is, so `"key":` can be copied.
    /// The key is kept with its object's (see obj_end): one already there while the
    /// object has fewer than FEW_KEYS is found here
    #[inline]
    fn key(&mut self) -> R<(&'a [u8], usize)> {
        let at = self.i;
        let s = self.string()?;
        if s.esc {
            return Err(Fallback);
        }
        self.eat(b':')?;
        let b = self.b;
        let key = Key::new(b, s.start, s.end);
        let earlier = &self.keys[self.base..];
        if earlier.len() < FEW_KEYS && earlier.iter().any(|e| e.same(&key, b)) {
            return Err(Fallback);
        }
        self.keys.push(key);
        Ok((&b[s.start..s.end], at))
    }

    /// A string value without escapes (an id, a term's language)
    #[inline]
    fn plain_string(&mut self) -> R<&'a [u8]> {
        let s = self.string()?;
        if s.esc {
            return Err(Fallback);
        }
        Ok(&self.b[s.start..s.end])
    }

    /// The value here, written to `out` as orjson writes it
    #[inline]
    fn copy(&mut self, out: &mut Vec<u8>) -> R {
        let mut run = self.i;
        self.value(out, &mut run, 0)?;
        out.extend_from_slice(&self.b[run..self.i]);
        Ok(())
    }

    /// The value here, checked but not kept
    fn skip(&mut self, sink: &mut Vec<u8>) -> R {
        sink.clear();
        self.copy(sink)
    }

    /// The value here, its bytes from `run` on left to copy as they are, but each span
    /// orjson writes differently written out in its place (`run` moved past it)
    fn value(&mut self, out: &mut Vec<u8>, run: &mut usize, depth: u32) -> R {
        match self.peek()? {
            b'"' => {
                let at = self.i;
                let s = self.string()?;
                if s.esc {
                    out.extend_from_slice(&self.b[*run..at]);
                    transcode(out, &self.b[s.start..s.end])?;
                    *run = self.i;
                }
            }
            b'{' => {
                if depth >= MAX_DEPTH {
                    return Err(Fallback);
                }
                if !self.obj()? {
                    loop {
                        self.key()?;
                        self.value(out, run, depth + 1)?;
                        if !self.obj_more()? {
                            break;
                        }
                    }
                }
            }
            b'[' => {
                if depth >= MAX_DEPTH {
                    return Err(Fallback);
                }
                if !self.open(b'[', b']')? {
                    loop {
                        self.value(out, run, depth + 1)?;
                        if !self.more(b']')? {
                            break;
                        }
                    }
                }
            }
            b't' => self.literal(b"true")?,
            b'f' => self.literal(b"false")?,
            b'n' => self.literal(b"null")?,
            b'-' | b'0'..=b'9' => {
                let at = self.i;
                if !self.number()? {
                    out.extend_from_slice(&self.b[*run..at]);
                    write_number(out, &self.b[at..self.i])?;
                    *run = self.i;
                }
            }
            _ => return Err(Fallback),
        }
        Ok(())
    }

    #[inline]
    fn literal(&mut self, word: &[u8]) -> R {
        if self.b.get(self.i..self.i + word.len()) != Some(word) {
            return Err(Fallback);
        }
        self.i += word.len();
        Ok(())
    }

    /// A number (JSON's grammar); true if orjson writes it as it is: an integer of at
    /// most 18 digits, but not -0 (others are written by write_number, which may still
    /// write them as they are)
    #[inline]
    fn number(&mut self) -> R<bool> {
        let (end, verbatim) = scan_number(self.b, self.i)?;
        self.i = end;
        Ok(verbatim)
    }
}

/// The number at b[start..] (JSON's grammar): where it ends, and whether orjson writes it
/// as it is (see Line::number)
#[inline]
pub(crate) fn scan_number(b: &[u8], start: usize) -> R<(usize, bool)> {
    let mut i = start;
    let neg = b.get(i) == Some(&b'-');
    if neg {
        i += 1;
    }
    let digit = |i: usize| b.get(i).is_some_and(|c| c.is_ascii_digit());
    match b.get(i) {
        Some(b'0') => i += 1,
        Some(b'1'..=b'9') => {
            while digit(i) {
                i += 1;
            }
        }
        _ => return Err(Fallback),
    }
    let int_end = i;
    let mut int = true;
    if b.get(i) == Some(&b'.') {
        i += 1;
        if !digit(i) {
            return Err(Fallback);
        }
        while digit(i) {
            i += 1;
        }
        int = false;
    }
    if matches!(b.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(b.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        if !digit(i) {
            return Err(Fallback);
        }
        while digit(i) {
            i += 1;
        }
        int = false;
    }
    let digits = int_end - start - neg as usize;
    Ok((i, int && digits <= 18 && !(neg && &b[start..i] == b"-0")))
}

/// Scratch buffers reused across lines: a datavalue's value and type (when rewritten),
/// and a sink for values checked but not kept
#[derive(Default)]
struct Scratch {
    value: Vec<u8>,
    kind: Vec<u8>,
    sink: Vec<u8>,
}

// The reshape's writers below share value()'s convention: the row written so far is
// `out` followed by the line's bytes from `run` to where the scan is. A writer leaves
// what the row shares with the line in the run, and writes out only where the row
// departs from it (flushing the run first): so a row is mostly copied in long runs.

/// A value of the line, kept for writing later: its bytes in the line, or (if any of
/// them is written differently) written to a scratch buffer
enum Kept {
    Line(usize, usize),
    Scratch,
}

impl<'a> Line<'a> {
    /// Write the line's bytes from `run` up to `to`
    #[inline]
    fn flush(&self, out: &mut Vec<u8>, run: &mut usize, to: usize) {
        out.extend_from_slice(&self.b[*run..to]);
        *run = to;
    }

    /// The value here, kept (see Kept); `scratch` is cleared first
    #[inline]
    fn keep(&mut self, scratch: &mut Vec<u8>) -> R<Kept> {
        scratch.clear();
        let start = self.i;
        let mut run = start;
        self.value(scratch, &mut run, 0)?;
        if scratch.is_empty() {
            return Ok(Kept::Line(start, self.i));
        }
        scratch.extend_from_slice(&self.b[run..self.i]);
        Ok(Kept::Scratch)
    }

    #[inline]
    fn write_kept(&self, out: &mut Vec<u8>, kept: &Option<Kept>, scratch: &[u8]) {
        match kept {
            Some(Kept::Line(s, e)) => out.extend_from_slice(&self.b[*s..*e]),
            Some(Kept::Scratch) => out.extend_from_slice(scratch),
            None => out.extend_from_slice(b"null"),
        }
    }
}

/// dump._map: an object, or else (Wikibase writes an empty map as []) none: false if the
/// value here is not an object (it is checked and passed, and `{}` written in its place)
#[inline]
fn map_or_empty(p: &mut Line, out: &mut Vec<u8>, run: &mut usize, sc: &mut Scratch) -> R<bool> {
    if p.peek()? == b'{' {
        return Ok(true);
    }
    p.flush(out, run, p.i);
    p.skip(&mut sc.sink)?;
    out.extend_from_slice(b"{}");
    *run = p.i;
    Ok(false)
}

/// A map (see map_or_empty), `each` called at each of its values with its key
#[inline]
fn map<'a>(
    p: &mut Line<'a>,
    out: &mut Vec<u8>,
    run: &mut usize,
    sc: &mut Scratch,
    mut each: impl FnMut(&mut Line<'a>, &mut Vec<u8>, &mut usize, &mut Scratch, &'a [u8]) -> R,
) -> R {
    if map_or_empty(p, out, run, sc)? && !p.obj()? {
        loop {
            let (k, _) = p.key()?;
            each(p, out, run, sc, k)?;
            if !p.obj_more()? {
                break;
            }
        }
    }
    Ok(())
}

/// An array (anything else is left to dump.entity_row), `each` called at each item
#[inline]
fn list<'a>(
    p: &mut Line<'a>,
    out: &mut Vec<u8>,
    run: &mut usize,
    sc: &mut Scratch,
    mut each: impl FnMut(&mut Line<'a>, &mut Vec<u8>, &mut usize, &mut Scratch) -> R,
) -> R {
    if !p.open(b'[', b']')? {
        loop {
            each(p, out, run, sc)?;
            if !p.more(b']')? {
                break;
            }
        }
    }
    Ok(())
}

/// A term {language, value} as its value; counts it in `mismatched` unless its language
/// is `lang` and it has only those two fields
fn term(
    p: &mut Line,
    out: &mut Vec<u8>,
    run: &mut usize,
    sc: &mut Scratch,
    lang: &[u8],
    mismatched: &mut u64,
) -> R {
    p.flush(out, run, p.i);
    let mut fields = 0;
    let (mut language, mut value) = (None, false);
    if !p.obj()? {
        loop {
            let (k, _) = p.key()?;
            fields += 1;
            match k {
                b"language" => {
                    if language.is_some() {
                        return Err(Fallback);
                    }
                    language = Some(if p.peek()? == b'"' {
                        p.plain_string()? == lang
                    } else {
                        p.skip(&mut sc.sink)?;
                        false
                    });
                }
                b"value" => {
                    if value {
                        return Err(Fallback);
                    }
                    value = true;
                    *run = p.i;
                    p.value(out, run, 0)?;
                    p.flush(out, run, p.i);
                }
                _ => p.skip(&mut sc.sink)?,
            }
            if !p.obj_more()? {
                break;
            }
        }
    }
    if !value {
        return Err(Fallback); // entity_row raises (KeyError)
    }
    if !(language == Some(true) && fields == 2) {
        *mismatched += 1;
    }
    *run = p.i;
    Ok(())
}

/// dump._terms: {lang: {language, value}} as {lang: value}
fn terms(
    p: &mut Line,
    out: &mut Vec<u8>,
    run: &mut usize,
    sc: &mut Scratch,
    mismatched: &mut u64,
) -> R {
    map(p, out, run, sc, |p, out, run, sc, lang| {
        term(p, out, run, sc, lang, mismatched)
    })
}

/// dump._alias_terms: {lang: [{language, value}]} as {lang: [value]}
fn alias_terms(
    p: &mut Line,
    out: &mut Vec<u8>,
    run: &mut usize,
    sc: &mut Scratch,
    mismatched: &mut u64,
) -> R {
    map(p, out, run, sc, |p, out, run, sc, lang| {
        list(p, out, run, sc, |p, out, run, sc| {
            term(p, out, run, sc, lang, mismatched)
        })
    })
}

/// The sitelinks as they are, each link an object, its field names added to `fields`
fn sitelinks<'a>(
    p: &mut Line<'a>,
    out: &mut Vec<u8>,
    run: &mut usize,
    sc: &mut Scratch,
    fields: &mut Vec<&'a [u8]>,
) -> R {
    map(p, out, run, sc, |p, out, run, _, _| {
        if !p.obj()? {
            loop {
                let (k, _) = p.key()?;
                if !fields.contains(&k) {
                    fields.push(k);
                }
                p.value(out, run, 0)?;
                if !p.obj_more()? {
                    break;
                }
            }
        }
        Ok(())
    })
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

#[derive(Default)]
struct Stats {
    mismatched: u64,
    entity_fields: BTreeSet<String>,
    sitelink_fields: BTreeSet<String>,
}

fn add_fields(set: &mut BTreeSet<String>, names: &[&[u8]]) {
    for name in names {
        // a key is UTF-8: the line is checked to be, and its quotes are ASCII
        let name = String::from_utf8_lossy(name);
        if !set.contains(name.as_ref()) {
            set.insert(name.into_owned());
        }
    }
}

/// The entity's parts, by their column (dump.COLUMNS): 0 is the id, 6 its own fields
fn part(k: &[u8]) -> usize {
    match k {
        b"id" => 0,
        b"labels" => 1,
        b"descriptions" => 2,
        b"aliases" => 3,
        b"sitelinks" => 4,
        b"claims" => 5,
        _ => 6,
    }
}

/// The line's row into `cols`; its claims, with `built`, built from the line into it (and
/// none written to the claims column) rather than written as text (see claims_arrow.rs)
fn entity_row(
    line: &[u8],
    cols: &mut Columns,
    stats: &mut Stats,
    sc: &mut Scratch,
    mut built: Option<&mut claims_arrow::RawBuild>,
) -> R {
    // A byte below 0x20 is whitespace between tokens (the dumps have none) or a control
    // character in a string (which orjson refuses): either is left to dump.entity_row
    if line.iter().fold(false, |any, &c| any | (c < 0x20)) {
        return Err(Fallback);
    }
    if simdutf8::basic::from_utf8(line).is_err() {
        return Err(Fallback); // orjson refuses it
    }
    let p = &mut Line {
        b: line,
        i: 0,
        keys: Vec::with_capacity(64),
        bases: Vec::with_capacity(16),
        base: 0,
    };
    let mut seen = [false; 6];
    let mut mismatched = 0;
    let mut entity_fields: Vec<&[u8]> = Vec::new();
    let mut sitelink_fields: Vec<&[u8]> = Vec::new();
    cols.data[6].push(b'{');
    if !p.obj()? {
        loop {
            let (k, at) = p.key()?;
            let c = part(k);
            if c < 6 {
                seen[c] = true; // (a repeated key is left to entity_row: see Line::key)
            }
            let out = &mut cols.data[c];
            let mut run = p.i;
            match c {
                0 => out.extend_from_slice(p.plain_string()?),
                1 | 2 => terms(p, out, &mut run, sc, &mut mismatched)?,
                3 => alias_terms(p, out, &mut run, sc, &mut mismatched)?,
                4 => sitelinks(p, out, &mut run, sc, &mut sitelink_fields)?,
                5 => match built.as_deref_mut() {
                    Some(b) => {
                        p.i = b.row(line, p.i).ok_or(Fallback)?;
                        run = p.i;
                    }
                    None => claims::claims(p, out, &mut run, sc)?,
                },
                _ => {
                    if !entity_fields.is_empty() {
                        out.push(b',');
                    }
                    entity_fields.push(k);
                    out.extend_from_slice(&line[at..run]);
                    p.value(out, &mut run, 0)?;
                }
            }
            if c != 0 {
                p.flush(out, &mut run, p.i);
            }
            if !p.obj_more()? {
                break;
            }
        }
    }
    if p.i != line.len() || !seen[0] {
        return Err(Fallback); // content after the entity, or no id
    }
    cols.data[6].push(b'}');
    for c in 1..6 {
        if !seen[c] {
            match (c, built.as_deref_mut()) {
                (5, Some(b)) => b.no_claims(),
                _ => cols.data[c].extend_from_slice(b"{}"),
            }
        }
    }
    for c in 0..7 {
        cols.end_cell(c);
    }
    stats.mismatched += mismatched;
    add_fields(&mut stats.entity_fields, &entity_fields);
    add_fields(&mut stats.sitelink_fields, &sitelink_fields);
    Ok(())
}

/// Reshape a run of lines; returns its columns, stats, and the indices left to Python
fn run(lines: &[&[u8]], base: usize) -> (Columns, Stats, Vec<usize>) {
    let mut cols = Columns::with_room(lines.len(), lines.iter().map(|l| l.len()).sum());
    let mut stats = Stats::default();
    let mut sc = Scratch::default();
    let mut left = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let marks = cols.mark();
        if entity_row(line, &mut cols, &mut stats, &mut sc, None).is_err() {
            cols.truncate(&marks);
            for c in 0..7 {
                cols.end_cell(c); // an empty cell, filled in by Python
            }
            left.push(base + i);
        }
    }
    (cols, stats, left)
}

type Out<'py> = (
    Vec<(Bound<'py, Buffer>, Bound<'py, Buffer>)>,
    u64,
    Vec<String>,
    Vec<String>,
    Vec<usize>,
);

/// The chunk's rows as 7 columns (dump.COLUMNS), each as (data, int64 offsets) buffers
/// (Arrow's large-string layout, handed over without a copy: see Buffer); the mismatched
/// terms; the entities' own field names and their sitelinks' field names; and the indices
/// of the lines left for dump.entity_row (their cells empty here).
#[pyfunction]
#[pyo3(signature = (lines, threads=1))]
fn entity_rows<'py>(
    py: Python<'py>,
    lines: Vec<Bound<'py, PyBytes>>,
    threads: usize,
) -> PyResult<Out<'py>> {
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
    let mut stats = Stats::default();
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

/// A chunk's rows as entity_rows gives them, with its claims built from its lines (no
/// claims text: the claims column's cells are empty) and each row's set, by its routing
/// (0 main, 1 scholarly: see claims_arrow.rs RawBuild), as
/// (rows, (flags, [the main set's batches, the scholarly set's], each batch's arrays as
/// Col::export lists them), None), each batch about `batch_bytes` of the lines' claims; or (None, None, why) if the chunk takes the text path (entity_rows, then
/// claims_batches): a line left to entity_row, or a misfit
#[pyfunction]
fn entity_rows_built<'py>(
    py: Python<'py>,
    lines: Vec<Bound<'py, PyBytes>>,
    plan: &Bound<'py, PyAny>,
    scholarly: Vec<String>,
    batch_bytes: usize,
) -> PyResult<(Option<Out<'py>>, Option<(Bound<'py, PyBytes>, [Vec<Vec<PyObject>>; 2])>, Option<String>)> {
    let plan = claims_arrow::plan_of(plan)?;
    let ids = scholarly.into_iter().map(String::into_bytes).collect();
    let slices: Vec<&[u8]> = lines.iter().map(|b| b.as_bytes()).collect();
    let mut b = claims_arrow::RawBuild::new(plan, ids, batch_bytes);
    let done = py.allow_threads(|| {
        let mut cols = Columns::with_room(slices.len(), slices.iter().map(|l| l.len()).sum());
        let mut stats = Stats::default();
        let mut sc = Scratch::default();
        for line in &slices {
            if entity_row(line, &mut cols, &mut stats, &mut sc, Some(&mut b)).is_err() {
                b.misfit.get_or_insert("a line left to entity_row");
                return None;
            }
            if b.misfit.is_some() {
                return None;
            }
        }
        Some((cols, stats))
    });
    match done {
        None => Ok((None, None, Some(b.misfit.unwrap_or("not built").to_string()))),
        Some((cols, stats)) => {
            let flags = PyBytes::new(py, &b.flags);
            let rows = out(py, (cols, stats, Vec::new()))?;
            Ok((Some(rows), Some((flags, b.export(py)?)), None))
        }
    }
}

/// A column buffer handed to Python as it is (no copy), through the buffer protocol:
/// pyarrow's py_buffer keeps it alive as long as the arrays (and Polars' frame) built
/// on it
#[pyclass(frozen)]
struct Buffer(Data);

enum Data {
    Bytes(Vec<u8>),
    Offsets(Vec<i64>),
    Floats(Vec<f64>),
}

#[pymethods]
impl Buffer {
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let (ptr, len) = match &slf.get().0 {
            Data::Bytes(v) => (v.as_ptr(), v.len()),
            Data::Offsets(v) => (v.as_ptr() as *const u8, std::mem::size_of_val(v.as_slice())),
            Data::Floats(v) => (v.as_ptr() as *const u8, std::mem::size_of_val(v.as_slice())),
        };
        // read-only; the view holds a reference to slf
        if ffi::PyBuffer_FillInfo(
            view,
            slf.as_ptr(),
            ptr as *mut _,
            len as ffi::Py_ssize_t,
            1,
            flags,
        ) == -1
        {
            return Err(PyErr::fetch(slf.py()));
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
}

fn out<'py>(
    py: Python<'py>,
    (all, stats, left): (Columns, Stats, Vec<usize>),
) -> PyResult<Out<'py>> {
    let mut columns = Vec::with_capacity(7);
    for (mut data, mut offsets) in all.data.into_iter().zip(all.offsets) {
        // let go of the room not used (with_room's), as the buffers live on in Python
        data.shrink_to_fit();
        offsets.shrink_to_fit();
        columns.push((
            Bound::new(py, Buffer(Data::Bytes(data)))?,
            Bound::new(py, Buffer(Data::Offsets(offsets)))?,
        ));
    }
    Ok((
        columns,
        stats.mismatched,
        stats.entity_fields.into_iter().collect(),
        stats.sitelink_fields.into_iter().collect(),
        left,
    ))
}

#[pymodule]
fn wikidata_reshape(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Buffer>()?;
    m.add_function(wrap_pyfunction!(claims_arrow::claims_batches, m)?)?;
    m.add_function(wrap_pyfunction!(claims_arrow::rayon_threads, m)?)?;
    fuse::register(m)?;
    m.add_function(wrap_pyfunction!(entity_rows_built, m)?)?;
    m.add_function(wrap_pyfunction!(bz2::bz2_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(entity_rows, m)?)
}
