//! `lamb_core` -- native hot-path kernels for LAMb, exposed to Python via PyO3.
//!
//! Three responsibilities, chosen because they are the hottest and/or most
//! correctness-critical paths in the training loop:
//!   * `arith`      -- the exact verifier that produces the self-play reward,
//!   * `curriculum` -- the per-step problem generator,
//!   * `store`      -- the exact top-k retrieval memory.
//!
//! Everything here has a pure-Python fallback in `lamb/_fallback.py`, so the
//! package runs (more slowly) even when this extension is not compiled.

use pyo3::prelude::*;

mod arith;
mod curriculum;
mod store;

/// Evaluate an integer arithmetic expression exactly. Returns `None` on
/// malformed input or `i128` overflow.
#[pyfunction]
fn evaluate(expr: &str) -> Option<i128> {
    arith::evaluate(expr)
}

/// Verify that `answer` (decimal string) equals the exact value of `expr`.
#[pyfunction]
fn verify(expr: &str, answer: &str) -> bool {
    arith::verify(expr, answer)
}

/// Sample a `(problem, answer)` pair at the requested difficulty.
/// `op` is one of "+", "-", "*".
#[pyfunction]
fn sample_problem(op: &str, a_digits: u32, b_digits: u32, seed: u64) -> (String, String) {
    let opc = op.chars().next().unwrap_or('+');
    curriculum::sample_problem(opc, a_digits, b_digits, seed)
}

/// Exact top-k associative memory keyed by dot-product similarity.
#[pyclass]
struct TopKStore {
    inner: store::Store,
}

#[pymethods]
impl TopKStore {
    #[new]
    fn new(dim: usize) -> Self {
        TopKStore {
            inner: store::Store::new(dim),
        }
    }

    /// Add a single key vector with an integer id. Raises on dim mismatch.
    fn add(&mut self, key: Vec<f32>, id: i64) -> PyResult<()> {
        if self.inner.add(&key, id) {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "key has length {} but store dim is {}",
                key.len(),
                self.inner.dim()
            )))
        }
    }

    /// Add many key vectors at once.
    fn add_batch(&mut self, keys: Vec<Vec<f32>>, ids: Vec<i64>) -> PyResult<()> {
        if keys.len() != ids.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "keys and ids must have equal length",
            ));
        }
        for (k, id) in keys.iter().zip(ids) {
            if !self.inner.add(k, id) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "key dimensionality mismatch",
                ));
            }
        }
        Ok(())
    }

    /// Return up to `k` `(id, score)` pairs by descending dot-product.
    fn query(&self, query: Vec<f32>, k: usize) -> Vec<(i64, f32)> {
        self.inner.query(&query, k)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[pymodule]
fn lamb_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(verify, m)?)?;
    m.add_function(wrap_pyfunction!(sample_problem, m)?)?;
    m.add_class::<TopKStore>()?;
    m.add("USING_RUST", true)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
