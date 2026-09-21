//! Top-k associative vector store.
//!
//! An explicit, growable key->id memory queried by dot-product similarity. It
//! complements LAMb's differentiable neural memory: where the neural memory is
//! a fixed-size compressive state, this store is the exact retrieval tier used
//! by the long-context recall evaluations and by any future retrieval
//! augmentation. Brute-force search is intentional -- it is exact, cache
//! friendly, and trivially fast in native code for the sizes involved here.

pub struct Store {
    dim: usize,
    keys: Vec<f32>, // row-major, len == dim * ids.len()
    ids: Vec<i64>,
}

impl Store {
    pub fn new(dim: usize) -> Self {
        Store {
            dim,
            keys: Vec::new(),
            ids: Vec::new(),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    #[allow(dead_code)] // part of the public store API; not used internally yet
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Append one key vector with an associated id. Silently ignores vectors of
    /// the wrong dimensionality (the Python wrapper validates and raises).
    pub fn add(&mut self, key: &[f32], id: i64) -> bool {
        if key.len() != self.dim {
            return false;
        }
        self.keys.extend_from_slice(key);
        self.ids.push(id);
        true
    }

    /// Return up to `k` `(id, score)` pairs with the highest dot-product against
    /// `query`, sorted by descending score.
    pub fn query(&self, query: &[f32], k: usize) -> Vec<(i64, f32)> {
        if query.len() != self.dim || self.ids.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut scored: Vec<(i64, f32)> = self
            .ids
            .iter()
            .enumerate()
            .map(|(row, &id)| {
                let base = row * self.dim;
                let mut dot = 0.0f32;
                for j in 0..self.dim {
                    dot += self.keys[base + j] * query[j];
                }
                (id, dot)
            })
            .collect();

        let k = k.min(scored.len());
        // Partial selection then sort the retained prefix.
        scored.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retrieves_nearest() {
        let mut s = Store::new(3);
        s.add(&[1.0, 0.0, 0.0], 10);
        s.add(&[0.0, 1.0, 0.0], 20);
        s.add(&[0.9, 0.1, 0.0], 30);
        let out = s.query(&[1.0, 0.0, 0.0], 2);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, 10);
        assert_eq!(out[1].0, 30);
    }

    #[test]
    fn dim_mismatch_is_safe() {
        let mut s = Store::new(3);
        assert!(!s.add(&[1.0, 2.0], 1));
        assert!(s.query(&[1.0, 2.0], 1).is_empty());
    }
}
