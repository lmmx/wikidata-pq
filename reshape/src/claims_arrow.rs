//! The claims column decoded into Arrow in the split: the arrays polars-genson's claims call
//! (process.normalise_claims_direct) gives for a chunk whose claims all fit `claims.avsc`
//! strictly, built from the claims text the reshape has just written (the same text genson
//! would read back from the chunk file), so the chunk file needs no claims column and the
//! pipeline no claims call.
//!
//! It is genson's strict decode (genson-core's `normalise/typed.rs`, `to_arrow_strict`)
//! against the decode plan Python compiles from `claims.avsc` and the target schema (see
//! claims_build.py), then conformed to the target as genson's `conform` does: the target's
//! fields in its order, those `claims.avsc` lacks ("phantom") all null, 64-bit offsets.
//!
//! It is stricter than genson: whatever it does not decode exactly as genson does is a
//! misfit, and a misfit sends the whole part (as genson does a whole chunk) down today's
//! path, its claims written as JSON for genson to decode. Beyond genson's own misfits (a
//! member with no field, a value of another type, a non-null value where the schema saw
//! only nulls, a null at a record that is not nullable), so are: any absent or null value
//! where `claims.avsc` does not allow null; a boolean; a key holding an escape; a repeated
//! key; an empty object where a record is; whitespace between tokens. Genson's rules kept
//! exactly:
//!
//! - empty arrays and maps are null (`empty_as_null`);
//! - maps are lists of {key, value} (`map_encoding="kv"`), in the text's order;
//! - a scalar in place of a record goes to its promoted field (`name__type`): a string to
//!   `__string`; an integer to `__int`/`__integer`/`__long` where the record has one,
//!   else to the float field (`__number`); a fraction to the float field;
//! - numbers are read as genson reads them (`Decoder::number`): up to 18 digits as an
//!   i64, anything else by serde_json's `Number` (whose float parsing is not always the
//!   nearest double: the published claims hold its values);
//! - strings are unescaped to UTF-8.
use crate::{Buffer, Data};
use memchr::memchr2;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::str::FromStr;

/// A value not decoded exactly as genson's strict decode decodes it
pub(crate) struct Misfit(pub(crate) &'static str);
type M<T = ()> = Result<T, Misfit>;

#[derive(Clone, Copy, PartialEq)]
enum Promoted {
    No,
    String,
    Boolean,
    Int,
    Float,
}

impl Promoted {
    /// As genson's `Promoted::of` reads a field's name
    fn of(name: &str) -> Self {
        if !name.contains("__") {
            return Promoted::No;
        }
        match name.split("__").last().unwrap_or("") {
            "string" => Promoted::String,
            "boolean" => Promoted::Boolean,
            "int" | "integer" | "long" => Promoted::Int,
            "float" | "double" | "number" => Promoted::Float,
            _ => Promoted::No,
        }
    }
}

/// The decode plan: a node of the target schema, with what claims.avsc says of it
pub(crate) enum Kind {
    Str,
    Int,
    Float,
    /// `["null"]` in claims.avsc (a column of nulls in the target)
    Null,
    /// In the target but not in claims.avsc: always null, and a member for it a misfit
    Phantom(Box<Kind>),
    Record {
        fields: Vec<Field>,
        has_int_field: bool,
    },
    Array(Box<Node>),
    Map(Box<Node>),
}

pub(crate) struct Node {
    kind: Kind,
    nullable: bool,
}

pub(crate) struct Field {
    name: String,
    promoted: Promoted,
    node: Node,
}

/// The plan from Python's nested tuples (claims_build.plan): ("str"|"int"|"float"|"null",
/// nullable), ("phantom", nullable, node), ("record", nullable, [(name, node)]),
/// ("array"|"map", nullable, node)
pub(crate) fn plan_of(obj: &Bound<'_, PyAny>) -> PyResult<Node> {
    let t = obj.downcast::<PyTuple>()?;
    let kind: String = t.get_item(0)?.extract()?;
    let nullable: bool = t.get_item(1)?.extract()?;
    let kind = match kind.as_str() {
        "str" => Kind::Str,
        "int" => Kind::Int,
        "float" => Kind::Float,
        "null" => Kind::Null,
        "phantom" => Kind::Phantom(Box::new(plan_of(&t.get_item(2)?)?.kind)),
        "array" => Kind::Array(Box::new(plan_of(&t.get_item(2)?)?)),
        "map" => Kind::Map(Box::new(plan_of(&t.get_item(2)?)?)),
        "record" => {
            let list = t.get_item(2)?;
            let list = list.downcast::<PyList>()?;
            let mut fields = Vec::with_capacity(list.len());
            for item in list.iter() {
                let pair = item.downcast::<PyTuple>()?;
                let name: String = pair.get_item(0)?.extract()?;
                let node = plan_of(&pair.get_item(1)?)?;
                fields.push(Field {
                    promoted: Promoted::of(&name),
                    name,
                    node,
                });
            }
            // As genson's plan: whether the record (as claims.avsc has it) holds an
            // integer field
            if fields.len() > 64 {
                return Err(pyo3::exceptions::PyValueError::new_err("a record of over 64 fields"));
            }
            let has_int_field = fields
                .iter()
                .any(|f| !matches!(f.node.kind, Kind::Phantom(_)) && f.promoted == Promoted::Int);
            Kind::Record {
                fields,
                has_int_field,
            }
        }
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown plan node {other}"
            )))
        }
    };
    Ok(Node { kind, nullable })
}

/// A validity bitmap (Arrow's: least significant bit first)
#[derive(Default)]
struct Bits {
    bytes: Vec<u8>,
    len: usize,
    nulls: usize,
}

impl Bits {
    #[inline]
    fn push(&mut self, valid: bool) {
        if self.len % 8 == 0 {
            self.bytes.push(0);
        }
        if valid {
            *self.bytes.last_mut().unwrap() |= 1 << (self.len % 8);
        } else {
            self.nulls += 1;
        }
        self.len += 1;
    }
}

/// A column being built, its shape the plan node's
enum Col {
    Str {
        valid: Bits,
        offsets: Vec<i64>,
        data: Vec<u8>,
    },
    Int {
        valid: Bits,
        values: Vec<i64>,
    },
    Float {
        valid: Bits,
        values: Vec<f64>,
    },
    Null(usize),
    Struct {
        valid: Bits,
        children: Vec<Col>,
    },
    List {
        valid: Bits,
        offsets: Vec<i64>,
        items: Box<Col>,
    },
    /// A list of {key, value} entries (no entry is null)
    Map {
        valid: Bits,
        offsets: Vec<i64>,
        keys: Box<Col>,
        values: Box<Col>,
    },
}

impl Col {
    fn new(kind: &Kind) -> Col {
        match kind {
            Kind::Str => Col::Str {
                valid: Bits::default(),
                offsets: vec![0],
                data: Vec::new(),
            },
            Kind::Int => Col::Int {
                valid: Bits::default(),
                values: Vec::new(),
            },
            Kind::Float => Col::Float {
                valid: Bits::default(),
                values: Vec::new(),
            },
            Kind::Null => Col::Null(0),
            Kind::Phantom(inner) => Col::new(inner),
            Kind::Record { fields, .. } => Col::Struct {
                valid: Bits::default(),
                children: fields.iter().map(|f| Col::new(&f.node.kind)).collect(),
            },
            Kind::Array(item) => Col::List {
                valid: Bits::default(),
                offsets: vec![0],
                items: Box::new(Col::new(&item.kind)),
            },
            Kind::Map(value) => Col::Map {
                valid: Bits::default(),
                offsets: vec![0],
                keys: Box::new(Col::new(&Kind::Str)),
                values: Box::new(Col::new(&value.kind)),
            },
        }
    }

    /// A null (a struct's children each take one, as genson pads them)
    fn null(&mut self) {
        match self {
            Col::Str { valid, offsets, .. } => {
                valid.push(false);
                offsets.push(*offsets.last().unwrap());
            }
            Col::Int { valid, values } => {
                valid.push(false);
                values.push(0);
            }
            Col::Float { valid, values } => {
                valid.push(false);
                values.push(0.0);
            }
            Col::Null(n) => *n += 1,
            Col::Struct { valid, children } => {
                valid.push(false);
                children.iter_mut().for_each(Col::null);
            }
            Col::List { valid, offsets, .. } | Col::Map { valid, offsets, .. } => {
                valid.push(false);
                offsets.push(*offsets.last().unwrap());
            }
        }
    }

    /// The column's buffers, depth first (see claims_build.arrays): each node's length,
    /// null count, validity (if it has a null), offsets and values
    fn export(self, py: Python<'_>, out: &mut Vec<PyObject>) -> PyResult<()> {
        match self {
            Col::Str {
                valid,
                offsets,
                data,
            } => node(py, out, valid, Some(offsets), Some(Data::Bytes(data))),
            Col::Int { valid, values } => node(py, out, valid, None, Some(Data::Offsets(values))),
            Col::Float { valid, values } => node(py, out, valid, None, Some(Data::Floats(values))),
            Col::Null(n) => {
                let valid = Bits {
                    bytes: Vec::new(),
                    len: n,
                    nulls: n,
                };
                node(py, out, valid, None, None)
            }
            Col::Struct { valid, children } => {
                node(py, out, valid, None, None)?;
                children.into_iter().try_for_each(|c| c.export(py, out))
            }
            Col::List {
                valid,
                offsets,
                items,
            } => {
                node(py, out, valid, Some(offsets), None)?;
                items.export(py, out)
            }
            Col::Map {
                valid,
                offsets,
                keys,
                values,
            } => {
                node(py, out, valid, Some(offsets), None)?;
                keys.export(py, out)?;
                values.export(py, out)
            }
        }
    }
}

/// One node of `Col::export`: (length, null count, validity or None, offsets or None,
/// values or None), each buffer handed over without a copy
fn node(
    py: Python<'_>,
    out: &mut Vec<PyObject>,
    valid: Bits,
    offsets: Option<Vec<i64>>,
    values: Option<Data>,
) -> PyResult<()> {
    let buf = |d: Data| -> PyResult<PyObject> { Ok(Bound::new(py, Buffer(d))?.into_any().unbind()) };
    let (len, nulls) = (valid.len, valid.nulls);
    let valid = if nulls > 0 && !valid.bytes.is_empty() {
        buf(Data::Bytes(valid.bytes))?
    } else {
        py.None()
    };
    let offsets = match offsets {
        Some(o) => buf(Data::Offsets(o))?,
        None => py.None(),
    };
    let values = match values {
        Some(v) => buf(v)?,
        None => py.None(),
    };
    let item = (len, nulls, valid, offsets, values).into_pyobject(py)?;
    out.push(item.into_any().unbind());
    Ok(())
}

/// A number as genson's `Decoder::number` reads it
#[derive(Clone, Copy)]
enum Num {
    Pos(u64),
    Neg(i64),
    Float(f64),
}

impl Num {
    fn read(text: &str) -> M<Num> {
        let (negative, digits) = match text.strip_prefix('-') {
            Some(d) => (true, d),
            None => (false, text),
        };
        if !digits.is_empty()
            && digits.len() <= 18
            && digits.bytes().all(|d| d.is_ascii_digit())
            && (digits.len() == 1 || !digits.starts_with('0'))
            && !(negative && digits == "0")
        {
            let n: i64 = digits.parse().map_err(|_| Misfit("number"))?;
            return Ok(if negative { Num::Neg(-n) } else { Num::Pos(n as u64) });
        }
        let n = serde_json::Number::from_str(text).map_err(|_| Misfit("number"))?;
        Ok(if let Some(u) = n.as_u64() {
            Num::Pos(u)
        } else if let Some(i) = n.as_i64() {
            Num::Neg(i)
        } else {
            Num::Float(n.as_f64().ok_or(Misfit("number"))?)
        })
    }
    fn is_i64(self) -> bool {
        match self {
            Num::Pos(u) => u <= i64::MAX as u64,
            Num::Neg(_) => true,
            Num::Float(_) => false,
        }
    }
    fn is_u64(self) -> bool {
        matches!(self, Num::Pos(_))
    }
    fn is_f64(self) -> bool {
        matches!(self, Num::Float(_))
    }
    fn as_i64(self) -> Option<i64> {
        match self {
            Num::Pos(u) if u <= i64::MAX as u64 => Some(u as i64),
            Num::Neg(i) => Some(i),
            _ => None,
        }
    }
    fn as_f64(self) -> f64 {
        match self {
            Num::Pos(u) => u as f64,
            Num::Neg(i) => i as f64,
            Num::Float(f) => f,
        }
    }
}

/// A JSON value's kind, from its first byte
#[derive(Clone, Copy, PartialEq)]
enum Tok {
    Null,
    Str,
    Num,
    Obj,
    Arr,
}

/// The decoding of one cell of claims text
struct P<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> P<'a> {
    #[inline]
    fn peek(&self) -> M<u8> {
        self.b.get(self.i).copied().ok_or(Misfit("truncated"))
    }
    #[inline]
    fn eat(&mut self, c: u8) -> M {
        if self.peek()? == c {
            self.i += 1;
            Ok(())
        } else {
            Err(Misfit("syntax"))
        }
    }
    fn tok(&self) -> M<Tok> {
        Ok(match self.peek()? {
            b'n' => Tok::Null,
            b'"' => Tok::Str,
            b'-' | b'0'..=b'9' => Tok::Num,
            b'{' => Tok::Obj,
            b'[' => Tok::Arr,
            b't' | b'f' => return Err(Misfit("a boolean")),
            _ => return Err(Misfit("syntax")),
        })
    }
    fn null(&mut self) -> M {
        if self.b.get(self.i..self.i + 4) == Some(b"null") {
            self.i += 4;
            Ok(())
        } else {
            Err(Misfit("syntax"))
        }
    }
    /// The number here, as its text
    fn number_text(&mut self) -> M<&'a str> {
        let start = self.i;
        while let Some(&c) = self.b.get(self.i) {
            if matches!(c, b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                self.i += 1;
            } else {
                break;
            }
        }
        // ASCII; its grammar is checked by Num::read
        std::str::from_utf8(&self.b[start..self.i]).map_err(|_| Misfit("number"))
    }
    /// The string here, unescaped, appended to `out`
    fn string_into(&mut self, out: &mut Vec<u8>) -> M {
        self.eat(b'"')?;
        loop {
            let k = memchr2(b'"', b'\\', &self.b[self.i..]).ok_or(Misfit("truncated"))?;
            out.extend_from_slice(&self.b[self.i..self.i + k]);
            self.i += k;
            if self.b[self.i] == b'"' {
                self.i += 1;
                return Ok(());
            }
            let c = *self.b.get(self.i + 1).ok_or(Misfit("truncated"))?;
            self.i += 2;
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
                        if self.b.get(self.i..self.i + 2) != Some(b"\\u") {
                            return Err(Misfit("a lone surrogate"));
                        }
                        self.i += 2;
                        let lo = self.hex4()?;
                        if !(0xdc00..0xe000).contains(&lo) {
                            return Err(Misfit("a lone surrogate"));
                        }
                        cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                    } else if (0xdc00..0xe000).contains(&cp) {
                        return Err(Misfit("a lone surrogate"));
                    }
                    let ch = char::from_u32(cp).ok_or(Misfit("an escape"))?;
                    let mut buf = [0; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                _ => return Err(Misfit("an escape")),
            }
        }
    }
    fn hex4(&mut self) -> M<u32> {
        let h = self.b.get(self.i..self.i + 4).ok_or(Misfit("an escape"))?;
        let s = std::str::from_utf8(h).map_err(|_| Misfit("an escape"))?;
        let v = u32::from_str_radix(s, 16).map_err(|_| Misfit("an escape"))?;
        self.i += 4;
        Ok(v)
    }
    /// An object key (one holding an escape is a misfit): its text
    fn key(&mut self) -> M<&'a [u8]> {
        self.eat(b'"')?;
        let k = memchr2(b'"', b'\\', &self.b[self.i..]).ok_or(Misfit("truncated"))?;
        if self.b[self.i + k] == b'\\' {
            return Err(Misfit("a key holding an escape"));
        }
        let key = &self.b[self.i..self.i + k];
        self.i += k + 1;
        self.eat(b':')?;
        Ok(key)
    }
    /// Past `open`; true if the container is empty (and past its close)
    fn open(&mut self, open: u8, close: u8) -> M<bool> {
        self.eat(open)?;
        if self.peek()? == close {
            self.i += 1;
            return Ok(true);
        }
        Ok(false)
    }
    /// After a member or item: true at a comma, false past the close
    fn more(&mut self, close: u8) -> M<bool> {
        let c = self.peek()?;
        self.i += 1;
        if c == b',' {
            Ok(true)
        } else if c == close {
            Ok(false)
        } else {
            Err(Misfit("syntax"))
        }
    }

    /// Decode the value here against `node` into `col`
    fn put(&mut self, node: &Node, col: &mut Col) -> M {
        let tok = self.tok()?;
        if tok == Tok::Null {
            self.null()?;
            return self.absent(node, col);
        }
        match (&node.kind, col) {
            (Kind::Phantom(_), _) => return Err(Misfit("a field claims.avsc lacks")),
            (Kind::Null, _) => return Err(Misfit("a value where claims.avsc has only nulls")),
            (Kind::Str, Col::Str { valid, offsets, data }) => {
                if tok != Tok::Str {
                    return Err(Misfit("a non-string at a string"));
                }
                self.string_into(data)?;
                offsets.push(data.len() as i64);
                valid.push(true);
            }
            (Kind::Int, Col::Int { valid, values }) => {
                if tok != Tok::Num {
                    return Err(Misfit("a non-number at an integer"));
                }
                let n = Num::read(self.number_text()?)?;
                values.push(n.as_i64().ok_or(Misfit("a fraction or too large an integer"))?);
                valid.push(true);
            }
            (Kind::Float, Col::Float { valid, values }) => {
                if tok != Tok::Num {
                    return Err(Misfit("a non-number at a number"));
                }
                values.push(Num::read(self.number_text()?)?.as_f64());
                valid.push(true);
            }
            (
                Kind::Record {
                    fields,
                    has_int_field,
                },
                Col::Struct { valid, children },
            ) => {
                match tok {
                    Tok::Obj => self.record(fields, children)?,
                    Tok::Str | Tok::Num => self.promoted(fields, *has_int_field, children, tok)?,
                    _ => return Err(Misfit("an array at a record")),
                }
                valid.push(true);
            }
            (Kind::Array(item), Col::List { valid, offsets, items }) => {
                if tok != Tok::Arr {
                    return Err(Misfit("a non-array at an array"));
                }
                let mut n = 0;
                if self.open(b'[', b']')? {
                    // empty_as_null
                    valid.push(false);
                    offsets.push(*offsets.last().unwrap());
                    return Ok(());
                }
                loop {
                    self.put(item, items)?;
                    n += 1;
                    if !self.more(b']')? {
                        break;
                    }
                }
                offsets.push(offsets.last().unwrap() + n);
                valid.push(true);
            }
            (
                Kind::Map(value),
                Col::Map {
                    valid,
                    offsets,
                    keys,
                    values,
                },
            ) => {
                if tok != Tok::Obj {
                    return Err(Misfit("a non-object at a map"));
                }
                if self.open(b'{', b'}')? {
                    // empty_as_null
                    valid.push(false);
                    offsets.push(*offsets.last().unwrap());
                    return Ok(());
                }
                let Col::Str {
                    valid: kvalid,
                    offsets: koffsets,
                    data: kdata,
                } = &mut **keys
                else {
                    unreachable!("a map's keys are strings")
                };
                let mut seen: Vec<&[u8]> = Vec::new();
                let mut set: std::collections::HashSet<&[u8]> = std::collections::HashSet::new();
                let mut n = 0;
                loop {
                    let k = self.key()?;
                    // a repeated key (genson keeps its first place and its last value)
                    let repeat = if seen.len() < 16 {
                        let r = seen.contains(&k);
                        seen.push(k);
                        if seen.len() == 16 {
                            set.extend(seen.iter().copied());
                        }
                        r
                    } else {
                        !set.insert(k)
                    };
                    if repeat {
                        return Err(Misfit("a repeated key"));
                    }
                    kdata.extend_from_slice(k);
                    koffsets.push(kdata.len() as i64);
                    kvalid.push(true);
                    self.put(value, values)?;
                    n += 1;
                    if !self.more(b'}')? {
                        break;
                    }
                }
                offsets.push(offsets.last().unwrap() + n);
                valid.push(true);
            }
            _ => unreachable!("a column is its node's"),
        }
        Ok(())
    }

    /// A null, or an absent member: null where claims.avsc allows it
    fn absent(&mut self, node: &Node, col: &mut Col) -> M {
        if !node.nullable && !matches!(node.kind, Kind::Phantom(_) | Kind::Null) {
            return Err(Misfit("a null where claims.avsc has no null"));
        }
        col.null();
        Ok(())
    }

    /// An object's members into its record's fields (by name, mostly in order)
    fn record(&mut self, fields: &[Field], children: &mut [Col]) -> M {
        if self.open(b'{', b'}')? {
            return Err(Misfit("an empty object at a record"));
        }
        let mut seen: u64 = 0;
        let mut next = 0;
        loop {
            let k = self.key()?;
            let j = if fields.get(next).is_some_and(|f| f.name.as_bytes() == k) {
                next
            } else {
                fields
                    .iter()
                    .position(|f| f.name.as_bytes() == k)
                    .ok_or(Misfit("a member with no field"))?
            };
            if seen & (1 << j) != 0 {
                return Err(Misfit("a repeated key"));
            }
            seen |= 1 << j;
            self.put(&fields[j].node, &mut children[j])?;
            next = j + 1;
            if !self.more(b'}')? {
                break;
            }
        }
        for (j, (f, c)) in fields.iter().zip(children.iter_mut()).enumerate() {
            if seen & (1 << j) == 0 {
                self.absent(&f.node, c)?;
            }
        }
        Ok(())
    }

    /// A scalar in place of a record: to the promoted field genson's `field_input` gives
    /// it, every other field null
    fn promoted(&mut self, fields: &[Field], has_int_field: bool, children: &mut [Col], tok: Tok) -> M {
        let start = self.i;
        let num = if tok == Tok::Num {
            Some(Num::read(self.number_text()?)?)
        } else {
            None
        };
        self.i = start;
        let matches = |p: Promoted| match (tok, p, num) {
            (Tok::Str, Promoted::String, _) => true,
            (Tok::Num, Promoted::Int, Some(n)) => n.is_i64() || n.is_u64(),
            (Tok::Num, Promoted::Float, Some(n)) => n.is_f64() || !has_int_field,
            _ => false,
        };
        let mut taken = None;
        for (j, f) in fields.iter().enumerate() {
            if f.promoted != Promoted::No && !matches!(f.node.kind, Kind::Phantom(_)) && matches(f.promoted) {
                if taken.is_some() {
                    return Err(Misfit("a scalar two promoted fields take"));
                }
                taken = Some(j);
            }
        }
        let j = taken.ok_or(Misfit("a scalar at a record no promoted field takes"))?;
        for (k, (f, c)) in fields.iter().zip(children.iter_mut()).enumerate() {
            if k == j {
                self.put(&f.node, c)?;
            } else {
                self.absent(&f.node, c)?;
            }
        }
        Ok(())
    }
}

/// The claims of rows `rows` (each a cell of claims text, the map {property: [statement]})
/// as one batch's column against `plan`
fn batch(plan: &Node, rows: &[&[u8]]) -> M<Col> {
    let mut col = Col::new(&plan.kind);
    for row in rows {
        let mut p = P { b: row, i: 0 };
        p.put(plan, &mut col)?;
        if p.i != row.len() {
            return Err(Misfit("text after the claims"));
        }
    }
    Ok(col)
}

/// The claims column of a part (its text: `data`, and `offsets` from `first` on; its rows
/// those at `index`, int64 positions from `first`, if given, else the `rows` from `first`)
/// decoded in batches of `batch_rows` rows, each batch's arrays as `Col::export`
/// lists them (and None), or None and the reason a value does not fit, for the part to
/// take today's path
#[pyfunction]
#[pyo3(signature = (plan, data, offsets, first, rows, batch_rows, index=None))]
pub(crate) fn claims_batches<'py>(
    py: Python<'py>,
    plan: &Bound<'py, PyAny>,
    data: pyo3::buffer::PyBuffer<i8>,
    offsets: pyo3::buffer::PyBuffer<i8>,
    first: usize,
    rows: usize,
    batch_rows: usize,
    index: Option<pyo3::buffer::PyBuffer<i8>>,
) -> PyResult<(Option<Vec<Vec<PyObject>>>, Option<String>)> {
    let plan = plan_of(plan)?;
    if !data.is_c_contiguous() || !offsets.is_c_contiguous() {
        return Err(pyo3::exceptions::PyValueError::new_err("a buffer not contiguous"));
    }
    // SAFETY: C-contiguous buffers of bytes (pyarrow's are signed, format "b"), held for
    // the call
    let data: &[u8] = unsafe { std::slice::from_raw_parts(data.buf_ptr() as *const u8, data.len_bytes()) };
    let raw: &[u8] = unsafe { std::slice::from_raw_parts(offsets.buf_ptr() as *const u8, offsets.len_bytes()) };
    let offs: Vec<i64> = raw
        .chunks_exact(8)
        .skip(first)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let positions: Vec<usize> = match &index {
        Some(ix) => {
            if !ix.is_c_contiguous() {
                return Err(pyo3::exceptions::PyValueError::new_err("an index not contiguous"));
            }
            // SAFETY: as above
            let raw: &[u8] = unsafe { std::slice::from_raw_parts(ix.buf_ptr() as *const u8, ix.len_bytes()) };
            raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize).collect()
        }
        None => (0..rows).collect(),
    };
    if positions.len() != rows
        || batch_rows == 0
        || positions.iter().any(|&r| r + 1 >= offs.len())
        || offs.iter().any(|&o| o < 0 || o as usize > data.len())
    {
        return Err(pyo3::exceptions::PyValueError::new_err("rows out of range"));
    }
    let cells: Vec<&[u8]> = positions
        .iter()
        .map(|&r| &data[offs[r] as usize..offs[r + 1] as usize])
        .collect();
    let built = py.allow_threads(|| {
        cells
            .chunks(batch_rows)
            .map(|part| batch(&plan, part))
            .collect::<M<Vec<Col>>>()
    });
    match built {
        Err(Misfit(why)) => Ok((None, Some(why.to_string()))),
        Ok(cols) => {
            let mut out = Vec::with_capacity(cols.len());
            for col in cols {
                let mut nodes = Vec::new();
                col.export(py, &mut nodes)?;
                out.push(nodes);
            }
            Ok((Some(out), None))
        }
    }
}

/// The threads in reshape's rayon pool: genson's (in the pipeline's worker) are as many
/// (rayon's default: RAYON_NUM_THREADS, else the CPUs available), and its batches of a
/// chunk's claims are `rows.div_ceil(threads * 4)` rows
#[pyfunction]
pub(crate) fn rayon_threads() -> usize {
    rayon::current_num_threads()
}
