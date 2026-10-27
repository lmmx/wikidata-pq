//! The native bz2 decoder (the bunzip crate, crates/bunzip) for Python: `Bz2Decoder`, a decoder process's two
//! pipes in one object. `write` and `close` are its stdin (dump._feed writes the dump
//! into it, from its own thread, as into lbzip2's), `readinto` its stdout (wrapped in an
//! io.BufferedReader, so the split reads its lines exactly as it reads lbzip2's). Every
//! wait lets go of the GIL.

use bunzip::{Kernels, StreamDecoder};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyBrokenPipeError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[pyclass(frozen, name = "Bz2Decoder")]
struct Bz2Decoder {
    dec: StreamDecoder,
    kernels: Kernels,
}

#[pymethods]
impl Bz2Decoder {
    /// A decoder on `threads` worker threads (and one reading the stream in order), with
    /// at most `slots` blocks in flight (2 x threads by default)
    #[new]
    #[pyo3(signature = (threads, slots=None))]
    fn new(threads: usize, slots: Option<usize>) -> PyResult<Self> {
        if threads == 0 {
            return Err(PyValueError::new_err("threads must be at least 1"));
        }
        // (WIKIDATA_BZ2_SIMD=off, or 0: the portable kernels, whatever the CPU has)
        let kernels = match std::env::var("WIKIDATA_BZ2_SIMD").as_deref() {
            Ok("0") | Ok("off") => Kernels::scalar(),
            _ => Kernels::detect(),
        };
        Ok(Bz2Decoder { dec: StreamDecoder::new(threads, slots.unwrap_or(2 * threads), kernels), kernels })
    }

    /// Compressed bytes in, blocking while the input queued is full. BrokenPipeError once
    /// the decode has stopped (an error found, or the reader gone), as a pipe's writer gets
    fn write(&self, py: Python<'_>, data: &[u8]) -> PyResult<usize> {
        let n = data.len();
        let chunk = data.to_vec();
        py.allow_threads(|| self.dec.write(chunk))
            .map_err(|_| PyBrokenPipeError::new_err("the bz2 decoder has stopped"))?;
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
        res.map_err(|e| PyRuntimeError::new_err(format!("native bz2 decoder: {e}")))
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

    /// The kernels in use: "avx2+pclmulqdq" (or either), or "scalar"
    #[getter]
    fn kernels(&self) -> String {
        let k = self.kernels;
        match (k.avx2, k.clmul) {
            (true, true) => "avx2+pclmulqdq".into(),
            (true, false) => "avx2".into(),
            (false, true) => "pclmulqdq".into(),
            (false, false) => "scalar".into(),
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Bz2Decoder>()
}
