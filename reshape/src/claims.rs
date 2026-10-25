//! The claims column: dump.entity_row's claims, {property: [statement]}, each statement's
//! mainsnak, qualifiers and references' snaks reshaped (dump._statement), written by the
//! run model of lib.rs (the row copied from the line in runs; see Line::value). Kept to
//! this module, entered only through `claims`, so it can be replaced on its own.
use crate::{list, map, Fallback, Kept, Line, Scratch, R};

/// dump._snak: the snak's fields but its datavalue, then the datavalue's value and type
/// (as `datavalue` and `datavalue_type`)
fn snak(p: &mut Line, out: &mut Vec<u8>, run: &mut usize, sc: &mut Scratch) -> R {
    // whether a field other than the datavalue has been seen, and the datavalue's
    let mut others = false;
    let mut datavalue_type = false;
    let mut datavalue: Option<(Option<Kept>, Option<Kept>)> = None;
    if !p.obj()? {
        loop {
            let at = p.i;
            let (k, _) = p.key()?;
            if k == b"datavalue" {
                if datavalue.is_some() {
                    return Err(Fallback);
                }
                // the row has the fields before it (without the comma before it)
                p.flush(out, run, if others { at - 1 } else { at });
                let (mut value, mut kind) = (None, None);
                // a datavalue that is not an object: entity_row raises (.get)
                if !p.obj()? {
                    loop {
                        let (dk, _) = p.key()?;
                        match dk {
                            b"value" if value.is_none() => value = Some(p.keep(&mut sc.value)?),
                            b"type" if kind.is_none() => kind = Some(p.keep(&mut sc.kind)?),
                            b"value" | b"type" => return Err(Fallback),
                            _ => p.skip(&mut sc.sink)?,
                        }
                        if !p.obj_more()? {
                            break;
                        }
                    }
                }
                datavalue = Some((value, kind));
                // the fields after it: with the comma after it only if a field came before
                *run = if !others && p.peek()? == b',' {
                    p.i + 1
                } else {
                    p.i
                };
            } else {
                others = true;
                datavalue_type |= k == b"datavalue_type";
                p.value(out, run, 0)?;
            }
            if p.peek()? == b'}' {
                break;
            }
            p.eat(b',')?;
        }
    } else {
        return Ok(()); // {} as it is
    }
    // at the snak's closing brace
    p.obj_end()?;
    if let Some((value, kind)) = datavalue {
        if datavalue_type {
            return Err(Fallback); // dump._snak would set it in place
        }
        p.flush(out, run, p.i);
        out.extend_from_slice(if others {
            b",\"datavalue\":"
        } else {
            b"\"datavalue\":"
        });
        p.write_kept(out, &value, &sc.value);
        out.extend_from_slice(b",\"datavalue_type\":");
        p.write_kept(out, &kind, &sc.kind);
    }
    p.i += 1;
    Ok(())
}

/// dump._snaks: {property: [snak]}, each snak reshaped
fn snaks(p: &mut Line, out: &mut Vec<u8>, run: &mut usize, sc: &mut Scratch) -> R {
    map(p, out, run, sc, |p, out, run, sc, _| {
        list(p, out, run, sc, snak)
    })
}

/// A statement's references, each with its snaks reshaped (and `"snaks":{}` added last
/// to one without)
fn references(p: &mut Line, out: &mut Vec<u8>, run: &mut usize, sc: &mut Scratch) -> R {
    list(p, out, run, sc, |p, out, run, sc| {
        let mut has_snaks = false;
        if p.obj()? {
            p.flush(out, run, p.i - 1);
            out.extend_from_slice(b"\"snaks\":{}");
            *run = p.i - 1;
            return Ok(());
        }
        loop {
            let (k, _) = p.key()?;
            if k == b"snaks" {
                if has_snaks {
                    return Err(Fallback);
                }
                has_snaks = true;
                snaks(p, out, run, sc)?;
            } else {
                p.value(out, run, 0)?;
            }
            if p.peek()? == b'}' {
                break;
            }
            p.eat(b',')?;
        }
        p.obj_end()?;
        if !has_snaks {
            p.flush(out, run, p.i);
            out.extend_from_slice(b",\"snaks\":{}");
        }
        p.i += 1;
        Ok(())
    })
}

/// dump._statement: its mainsnak, qualifiers and references' snaks reshaped, in place
fn statement(p: &mut Line, out: &mut Vec<u8>, run: &mut usize, sc: &mut Scratch) -> R {
    let mut mainsnak = false;
    if !p.obj()? {
        loop {
            let (k, _) = p.key()?;
            match k {
                b"mainsnak" => {
                    if mainsnak {
                        return Err(Fallback);
                    }
                    mainsnak = true;
                    // a mainsnak that is not an object: entity_row raises
                    if p.peek()? != b'{' {
                        return Err(Fallback);
                    }
                    snak(p, out, run, sc)?;
                }
                b"qualifiers" => snaks(p, out, run, sc)?,
                b"references" => references(p, out, run, sc)?,
                _ => p.value(out, run, 0)?,
            }
            if !p.obj_more()? {
                break;
            }
        }
    }
    if !mainsnak {
        return Err(Fallback); // entity_row raises (KeyError)
    }
    Ok(())
}

/// The claims: {property: [statement]}, each statement reshaped
pub(crate) fn claims(p: &mut Line, out: &mut Vec<u8>, run: &mut usize, sc: &mut Scratch) -> R {
    map(p, out, run, sc, |p, out, run, sc, _| {
        list(p, out, run, sc, statement)
    })
}
