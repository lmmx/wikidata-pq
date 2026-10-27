//! pqk FILE.parquet MODE [THREADS] [COLUMN=ENCODING...]
//!
//! Decodes FILE (a claims file from the split) to Arrow, then encodes it again with
//! pq-glue's writer: MODE `split` (zstd 1, as genson's write_parquet in the split), `none`
//! (no compression), `final` (zstd 3 and content-defined chunking, as finalise's
//! _write_file_parallel), `final_nocdc` (zstd 3 alone). THREADS bounds the writer's pool
//! (0: the default, every core). The output is read back and compared with the input,
//! batch by batch. Prints one JSON line: bytes, wall and CPU (all threads) of each step.
use arrow_array::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::{Compression, Encoding, ZstdLevel};
use parquet::file::properties::CdcOptions;
use pq_glue::write::{write_batches, WriteOptions};
use std::time::Instant;

fn cpu() -> f64 {
    let mut r: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut r) };
    let t = |v: libc::timeval| v.tv_sec as f64 + v.tv_usec as f64 / 1e6;
    t(r.ru_utime) + t(r.ru_stime)
}

fn read(path: &str) -> Vec<RecordBatch> {
    let f = std::fs::File::open(path).unwrap();
    let r = ParquetRecordBatchReaderBuilder::try_new(f).unwrap().with_batch_size(1 << 20).build().unwrap();
    r.map(|b| b.unwrap()).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (path, mode) = (&a[1], a[2].as_str());
    let threads: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(0);
    let encodings: Vec<(String, Encoding)> = a[4.min(a.len())..]
        .iter()
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap();
            (k.to_string(), v.parse().unwrap())
        })
        .collect();
    let in_bytes = std::fs::metadata(path).unwrap().len();
    let (c0, t0) = (cpu(), Instant::now());
    let batches = read(path);
    let (dec_wall, dec_cpu) = (t0.elapsed().as_secs_f64(), cpu() - c0);
    let schema = batches[0].schema();
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    let arrow_bytes: usize = batches.iter().map(|b| b.get_array_memory_size()).sum();
    let compression = match mode {
        "none" => Compression::UNCOMPRESSED,
        "split" => Compression::ZSTD(ZstdLevel::try_new(1).unwrap()),
        _ => Compression::ZSTD(ZstdLevel::try_new(3).unwrap()),
    };
    let opts = WriteOptions {
        max_row_group_rows: Some(122_880),
        spec_names: true,
        compression,
        content_defined_chunking: if mode == "final" { Some(CdcOptions::default()) } else { None },
        max_workers: if threads == 0 { None } else { Some(threads) },
        column_encodings: encodings,
        ..Default::default()
    };
    let out = format!("{}.pqk.{}.parquet", path, mode);
    let (c0, t0) = (cpu(), Instant::now());
    write_batches(std::path::Path::new(&out), schema.clone(), batches.clone(), &opts).unwrap();
    let (enc_wall, enc_cpu) = (t0.elapsed().as_secs_f64(), cpu() - c0);
    let out_bytes = std::fs::metadata(&out).unwrap().len();
    // equality: the same rows back
    let back = read(&out);
    let cat = |bs: &[RecordBatch]| arrow_select_concat(&schema, bs);
    let equal = cat(&batches) == cat(&back);
    std::fs::remove_file(&out).unwrap();
    println!(
        "{{\"file\":\"{}\",\"mode\":\"{}\",\"threads\":{},\"rows\":{},\"in_parquet_bytes\":{},\"arrow_bytes\":{},\"out_parquet_bytes\":{},\
         \"decode_wall\":{:.3},\"decode_cpu\":{:.3},\"encode_wall\":{:.3},\"encode_cpu\":{:.3},\"equal\":{}}}",
        path, mode, threads, rows, in_bytes, arrow_bytes, out_bytes, dec_wall, dec_cpu, enc_wall, enc_cpu, equal
    );
}

fn arrow_select_concat(schema: &arrow_schema::SchemaRef, bs: &[RecordBatch]) -> RecordBatch {
    // (one batch: comparing batch lists would depend on how the reader cut them)
    let cols = (0..schema.fields().len())
        .map(|i| {
            let parts: Vec<&dyn arrow_array::Array> = bs.iter().map(|b| b.column(i).as_ref()).collect();
            arrow_select::concat::concat(&parts).unwrap()
        })
        .collect();
    RecordBatch::try_new(schema.clone(), cols).unwrap()
}
