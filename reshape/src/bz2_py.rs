//! An in-process bz2 decoder for Python: `Bz2Decoder`, a decoder process's two pipes in one
//! object. `write` and `close` are its stdin: dump._feed writes the dump into it from its
//! own thread, as into lbzip2's. `readinto` is its stdout, wrapped in an io.BufferedReader,
//! so the split reads its lines exactly as it reads lbzip2's. Every wait lets go of the GIL.
//!
//! The decoding itself is a `Backend`'s: a small interface any in-process, block-parallel
//! Rust decoder can be put behind (feed compressed chunks, read decoded bytes in order,
//! close, cancel, join). The split's flag (WIKIDATA_BZ2_DECODER), its restart contract and
//! the validation around it are the same for every backend. One exists: `native`, the
//! mibz2 crate (crates/mibz2). Another is a `Backend` impl and a name in `backend()`
//! (and in dump.IN_PROCESS_BZ2).

use mibz2::{Kernels, StreamDecoder};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyBrokenPipeError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// An in-process bz2 decoder, used from two threads at once: one writing, one reading
trait Backend: Send + Sync {
    /// Queue compressed bytes, blocking while the queue is full. Err once the decode has
    /// stopped (an error found, or cancelled), as a closed pipe's writer gets
    fn write(&self, data: Vec<u8>) -> Result<(), ()>;
    /// The input's end
    fn close(&self);
    /// Decoded bytes into `buf`, in order (0 only at the end). Err(message) once the
    /// stream is found bad (after every byte of the blocks before the bad one), or after
    /// `cancel`
    fn read(&self, buf: &mut [u8]) -> Result<usize, String>;
    /// Stop: the writer's next write fails, the reader's next read fails
    fn cancel(&self);
    /// Cancel, and wait for its threads
    fn join(&self);
    fn threads(&self) -> usize;
    /// What it runs on (its SIMD kernels, say)
    fn describe(&self) -> String;
}

/// The mibz2 crate's decoder (`WIKIDATA_BZ2_DECODER=native`)
struct Native {
    dec: StreamDecoder,
    kernels: Kernels,
}

impl Backend for Native {
    fn write(&self, data: Vec<u8>) -> Result<(), ()> {
        self.dec.write(data).map_err(|_| ())
    }
    fn close(&self) {
        self.dec.close()
    }
    fn read(&self, buf: &mut [u8]) -> Result<usize, String> {
        self.dec.read(buf).map_err(|e| e.to_string())
    }
    fn cancel(&self) {
        self.dec.cancel()
    }
    fn join(&self) {
        self.dec.join()
    }
    fn threads(&self) -> usize {
        self.dec.threads()
    }
    fn describe(&self) -> String {
        match (self.kernels.avx2, self.kernels.clmul) {
            (true, true) => "avx2+pclmulqdq".into(),
            (true, false) => "avx2".into(),
            (false, true) => "pclmulqdq".into(),
            (false, false) => "scalar".into(),
        }
    }
}

/// The backend `name` names, on `threads` workers with at most `slots` blocks in flight
fn backend(name: &str, threads: usize, slots: usize) -> PyResult<Box<dyn Backend>> {
    match name {
        "native" => {
            // (WIKIDATA_BZ2_SIMD=off, or 0: the portable kernels, whatever the CPU has)
            let kernels = match std::env::var("WIKIDATA_BZ2_SIMD").as_deref() {
                Ok("0") | Ok("off") => Kernels::scalar(),
                _ => Kernels::detect(),
            };
            Ok(Box::new(Native { dec: StreamDecoder::new(threads, slots, kernels), kernels }))
        }
        _ => Err(PyValueError::new_err(format!("no in-process bz2 decoder {name:?} (there is: native)"))),
    }
}

#[pyclass(frozen, name = "Bz2Decoder")]
struct Bz2Decoder {
    dec: Box<dyn Backend>,
    name: String,
}

#[pymethods]
impl Bz2Decoder {
    /// The decoder `backend` on `threads` worker threads, with at most `slots` blocks in
    /// flight (2 x threads by default)
    #[new]
    #[pyo3(signature = (threads, slots=None, backend="native"))]
    fn new(threads: usize, slots: Option<usize>, backend: &str) -> PyResult<Self> {
        if threads == 0 {
            return Err(PyValueError::new_err("threads must be at least 1"));
        }
        let dec = self::backend(backend, threads, slots.unwrap_or(2 * threads))?;
        Ok(Bz2Decoder { dec, name: backend.to_string() })
    }

    /// Compressed bytes in, blocking while the input queued is full. BrokenPipeError once
    /// the decode has stopped (an error found, or the reader gone), as a pipe's writer gets
    fn write(&self, py: Python<'_>, data: &[u8]) -> PyResult<usize> {
        let n = data.len();
        let chunk = data.to_vec();
        py.allow_threads(|| self.dec.write(chunk))
            .map_err(|()| PyBrokenPipeError::new_err("the bz2 decoder has stopped"))?;
        Ok(n)
    }

    /// The input's end
    fn close(&self) {
        self.dec.close()
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, _args: &Bound<'_, pyo3::types::PyTuple>) -> bool {
        self.dec.close();
        false
    }

    /// Decoded bytes into `buffer` (a writable buffer of bytes), in order: how many, 0
    /// only at the end. RuntimeError if the stream is bad (every block before the bad one
    /// read first), naming what and where.
    fn readinto(&self, py: Python<'_>, buffer: &Bound<'_, PyAny>) -> PyResult<usize> {
        let buf = PyBuffer::<u8>::get(buffer)?;
        if buf.readonly() || !buf.is_c_contiguous() {
            return Err(PyValueError::new_err("readinto needs a writable contiguous buffer"));
        }
        let (ptr, len) = (buf.buf_ptr() as usize, buf.len_bytes());
        let res = py.allow_threads(|| {
            // SAFETY: the buffer is writable, contiguous and `len` bytes long, and `buf`
            // holds it (and its exporter) until after this returns
            let out = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, len) };
            self.dec.read(out)
        });
        drop(buf);
        res.map_err(|e| PyRuntimeError::new_err(format!("{} bz2 decoder: {e}", self.name)))
    }

    /// Stop the decode: a writer's next write fails, the reader's next read fails
    fn cancel(&self) {
        self.dec.cancel()
    }

    /// Stop the decode and wait for its threads
    fn join(&self, py: Python<'_>) {
        py.allow_threads(|| self.dec.join())
    }

    #[getter]
    fn threads(&self) -> usize {
        self.dec.threads()
    }

    #[getter]
    fn backend(&self) -> String {
        self.name.clone()
    }

    /// What the backend runs on (native: "avx2+pclmulqdq", or either, or "scalar")
    #[getter]
    fn kernels(&self) -> String {
        self.dec.describe()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Bz2Decoder>()
}
