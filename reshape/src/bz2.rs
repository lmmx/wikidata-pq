//! bz2's block magic found at any bit alignment, for the split's sweep reader
//! (src/wikidata/sweeps.py). A bz2 stream's blocks follow one another bit to bit, so a
//! block's 48-bit magic (0x314159265359, the digits of pi) can start at any bit of a
//! byte. The magic at bit s of a 7-byte window always covers that window's bytes 1 to 5
//! whole, so each of the 8 alignments is one 5-byte needle (memchr's memmem), and each
//! hit is checked whole, with what follows the magic in a real block: the block's
//! 32-bit CRC, a randomised bit of 0 (bzip2 and lbzip2 have not set it since 0.9.5)
//! and a 24-bit BWT origin below 900,000 (the largest block). A run of compressed data
//! that matches all of it by chance (about once in 3 * 10^4 dumps) is still caught: its
//! CRC breaks the stream CRC the sweep reader checks.
use memchr::memmem::Finder;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

const BLOCK: u64 = 0x3141_5926_5359;
/// The bytes after a hit's first byte that its checks read (magic, CRC, randomised bit,
/// origin: 105 bits from bit 7 at most, in 16 bytes)
pub const WINDOW: usize = 16;

fn scan(buf: &[u8]) -> Vec<(u64, u32)> {
    let mut out = Vec::new();
    let n = buf.len();
    for s in 0..8u32 {
        // the magic at bit s of a 7-byte window (the low 7 bytes of a u64)
        let w = (BLOCK << (8 - s)).to_be_bytes();
        let finder = Finder::new(&w[2..7]);
        for i in finder.find_iter(buf) {
            if i == 0 || i - 1 + WINDOW > n {
                continue;
            }
            let start = i - 1;
            let mut b = [0u8; WINDOW];
            b.copy_from_slice(&buf[start..start + WINDOW]);
            let x = u128::from_be_bytes(b);
            if (x >> (128 - s - 48)) as u64 & 0xFFFF_FFFF_FFFF != BLOCK {
                continue;
            }
            let crc = (x >> (128 - s - 80)) as u32;
            let randomised = (x >> (128 - s - 81)) & 1;
            let origin = (x >> (128 - s - 105)) & 0xFF_FFFF;
            if randomised != 0 || origin >= 900_000 {
                continue;
            }
            out.push((start as u64 * 8 + s as u64, crc));
        }
    }
    out.sort_unstable();
    out
}

/// The bit offsets in `buf` (from its first byte's top bit) at which a bz2 block starts,
/// at any alignment, each with its block CRC, ascending: each whose first WINDOW bytes
/// are in `buf` (so a caller reading a file in pieces keeps the last WINDOW bytes of one
/// piece before the next). The GIL is released while it scans.
#[pyfunction]
pub fn bz2_blocks(py: Python<'_>, buf: &Bound<'_, PyBytes>) -> Vec<(u64, u32)> {
    let b = buf.as_bytes();
    py.allow_threads(|| scan(b))
}
