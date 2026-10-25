//! The links table, built from the split's reshaped sitelinks cells as the pipeline
//! builds it from the chunk's `sitelinks` column: genson's normalisation
//! (`process.normalise_sitelinks`: a map of site to {site, title, badges}, decoded to
//! SITELINK_SCHEMA, an empty map or list null) and the explode for partitioning
//! (`partitioning.transforms.prepare_map_record`: rows with a null map dropped, one row
//! per link, the map's key dropped). A row is (entity, site, title, badges), the site
//! being the link's own `site` field (the partition key), in the entities' order and
//! each map's order.
//!
//! Misfits (see terms.rs): a link without a string `site` or `title`, a field other than
//! site, title and badges (dump.EXPECTED_FIELDS: the split's manifest halts on one
//! anyway), badges that are not a list of strings or null, a site that is not a plain
//! path component.

use super::terms::{Cell, Misfit, Rows, R};

const SITE: u8 = 1;
const TITLE: u8 = 2;
const BADGES: u8 = 4;

/// One link's record ({site, title, badges}) appended to `rows`
fn link(p: &mut Cell, rows: &mut Rows, entity: usize, scratch: &mut Vec<u8>, site: &mut Vec<u8>) -> R {
    let mut seen = 0u8;
    if p.open(b'{', b'}')? {
        return Err("a link without a site and title");
    }
    loop {
        let field = match p.key(scratch)? {
            b"site" => SITE,
            b"title" => TITLE,
            b"badges" => BADGES,
            _ => return Err("a link field other than site, title and badges"),
        };
        if seen & field != 0 {
            return Err("a link field repeated");
        }
        seen |= field;
        match field {
            SITE => {
                site.clear();
                p.string_into(site)?;
            }
            TITLE => p.string_into(&mut rows.data)?,
            _ => match p.peek()? {
                b'n' => {
                    for &c in b"null" {
                        p.eat(c)?;
                    }
                }
                b'[' => {
                    if !p.open(b'[', b']')? {
                        loop {
                            p.string_into(&mut rows.badge_data)
                                .map_err(|_| "a badge that is not a string")?;
                            rows.badge_ends.push(rows.badge_data.len());
                            if !p.more(b']')? {
                                break;
                            }
                        }
                    }
                }
                _ => return Err("badges that are not a list"),
            },
        }
        if !p.more(b'}')? {
            break;
        }
    }
    if seen & (SITE | TITLE) != SITE | TITLE {
        return Err("a link without a site and title");
    }
    let k = rows.intern(site)?;
    rows.push(entity, k);
    Ok(())
}

fn sitelinks_cell(rows: &mut Rows, entity: usize, cell: &[u8], scratch: &mut Vec<u8>, site: &mut Vec<u8>) -> R {
    let mut p = Cell::new(cell);
    if !p.open(b'{', b'}')? {
        loop {
            p.key(scratch)?; // the map's key is dropped: the link's own site is its key
            if p.peek()? != b'{' {
                return Err("a link that is not an object");
            }
            link(&mut p, rows, entity, scratch, site)?;
            if !p.more(b'}')? {
                break;
            }
        }
    }
    p.end()
}

/// A chunk's sitelinks cells as the links table's rows, or the first cell that does not fit
pub fn build<'a>(n: usize, cell: impl Fn(usize) -> &'a [u8]) -> Result<Rows, Misfit> {
    let mut rows = Rows::new();
    let (mut scratch, mut site) = (Vec::new(), Vec::new());
    for i in 0..n {
        sitelinks_cell(&mut rows, i, cell(i), &mut scratch, &mut site).map_err(|why| Misfit { row: i, why })?;
    }
    Ok(rows)
}
