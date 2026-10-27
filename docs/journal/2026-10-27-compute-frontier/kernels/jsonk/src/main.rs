//! jsonk FILE.jsonl: per byte of the lines, the cycles (TSC, 2.1 GHz) of
//! - simdutf8 validation (the UTF-8 floor);
//! - memchr2 over every string (a quote/backslash scan as the decoder makes it, but over
//!   the whole line: the scanning floor without per-token calls);
//! - stage1: the structural characters and the in-string mask, 64 bytes at a time
//!   (AVX2 compares, PCLMULQDQ prefix-XOR), counting structurals outside strings;
//! - simd-json's `to_tape` (stage 1 and 2: a general SIMD parse into its tape);
//! - a copy of the line (memcpy, the floor of touching every byte once).
use std::time::Instant;

fn rdtsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,pclmulqdq")]
unsafe fn stage1(b: &[u8]) -> (u64, u64) {
    use std::arch::x86_64::*;
    let mut structurals = 0u64;
    let mut in_string_carry = 0u64; // all ones if the previous block ended inside a string
    let mut prev_escaped = 0u64;
    let mut quotes_total = 0u64;
    let n = b.len() / 64 * 64;
    let q = _mm256_set1_epi8(b'"' as i8);
    let bs = _mm256_set1_epi8(b'\\' as i8);
    let (lb, rb, lc, rc, co, cl) = (
        _mm256_set1_epi8(b'{' as i8), _mm256_set1_epi8(b'}' as i8), _mm256_set1_epi8(b'[' as i8),
        _mm256_set1_epi8(b']' as i8), _mm256_set1_epi8(b',' as i8), _mm256_set1_epi8(b':' as i8));
    let mask = |v: __m256i, c: __m256i| _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, c)) as u32 as u64;
    let mut i = 0;
    while i < n {
        let v0 = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        let v1 = _mm256_loadu_si256(b.as_ptr().add(i + 32) as *const __m256i);
        let m64 = |c: __m256i| mask(v0, c) | mask(v1, c) << 32;
        let backslash = m64(bs);
        // escaped characters (simdjson's odd-length backslash runs)
        let escaped = {
            let bsl = backslash & !prev_escaped;
            let follows = bsl << 1;
            let odd_starts = bsl & !(bsl << 1) & 0xAAAA_AAAA_AAAA_AAAA ^ (bsl & !(bsl << 1) & 0x5555_5555_5555_5555);
            // (a simplified escape model: a backslash escapes the next byte unless itself escaped)
            let _ = odd_starts;
            let mut e = 0u64;
            let mut m = backslash;
            let mut carry = prev_escaped;
            // scalar over the (rare) backslashes
            while m != 0 {
                let t = m.trailing_zeros();
                let bit = 1u64 << t;
                if carry & bit == 0 {
                    e |= bit << 1;
                    carry |= bit << 1;
                }
                m &= m - 1;
            }
            prev_escaped = if t63(backslash, e) { 1 } else { 0 };
            let _ = follows;
            e
        };
        let quotes = m64(q) & !escaped;
        quotes_total += quotes.count_ones() as u64;
        // in-string mask: prefix XOR of the quotes, by carry-less multiply with all ones
        let inside = _mm_cvtsi128_si64(_mm_clmulepi64_si128(_mm_set_epi64x(0, quotes as i64), _mm_set1_epi8(-1), 0)) as u64
            ^ in_string_carry;
        in_string_carry = ((inside as i64) >> 63) as u64;
        let st = m64(lb) | m64(rb) | m64(lc) | m64(rc) | m64(co) | m64(cl);
        structurals += (st & !inside).count_ones() as u64;
        i += 64;
    }
    (structurals, quotes_total)
}

#[inline]
fn t63(backslash: u64, e: u64) -> bool {
    // the block's last byte is a backslash that escapes the next block's first byte
    backslash >> 63 == 1 && e & (1 << 63) == 0 && (e >> 63) == 0
}

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let text = std::fs::read(&path).unwrap();
    let lines: Vec<&[u8]> = text.split(|&c| c == b'\n').filter(|l| !l.is_empty()).collect();
    let bytes: usize = lines.iter().map(|l| l.len()).sum();
    let per = |c: u64| c as f64 / bytes as f64;
    let best = |f: &mut dyn FnMut() -> u64| (0..3).map(|_| { let t = rdtsc(); let r = f(); std::hint::black_box(r); rdtsc() - t }).min().unwrap();
    let utf8 = best(&mut || lines.iter().map(|l| simdutf8::basic::from_utf8(l).is_ok() as u64).sum());
    let mm2 = best(&mut || lines.iter().map(|l| memchr::memchr2_iter(b'"', b'\\', l).count() as u64).sum());
    let mut st = (0, 0);
    let s1 = best(&mut || { st = (0, 0); for l in &lines { let (a, b) = unsafe { stage1(l) }; st.0 += a; st.1 += b; } st.0 });
    let mut bufs: Vec<Vec<u8>> = lines.iter().map(|l| l.to_vec()).collect();
    let copy = best(&mut || { let mut n = 0; for (b, l) in bufs.iter_mut().zip(&lines) { b.copy_from_slice(l); n += b.len() as u64; } n });
    let mut tape_len = 0u64;
    let mut buffers = simd_json::Buffers::default();
    let t0 = Instant::now();
    let tape = best(&mut || {
        tape_len = 0;
        for (b, l) in bufs.iter_mut().zip(&lines) {
            b.copy_from_slice(l);
            let t = simd_json::to_tape_with_buffers(b, &mut buffers).unwrap();
            tape_len += t.0.len() as u64;
        }
        tape_len
    });
    let _ = t0;
    println!("{{\"lines\":{},\"bytes\":{},\"structurals_outside_strings\":{},\"quotes\":{},\"tape_nodes\":{},\
              \"cy_per_byte\":{{\"utf8_validate\":{:.3},\"memchr2_all\":{:.3},\"stage1_avx2\":{:.3},\"copy\":{:.3},\"simd_json_to_tape_incl_copy\":{:.3}}}}}",
             lines.len(), bytes, st.0, st.1, tape_len, per(utf8), per(mm2), per(s1), per(copy), per(tape));
}
