//! Self-play curriculum sampler.
//!
//! Generates concrete arithmetic problem instances at a requested difficulty
//! (operation + per-operand digit counts). This is on the hot path: the
//! self-play loop draws a fresh batch of problems every training step, so the
//! sampler is deterministic-by-seed and allocation-light.
//!
//! We deliberately avoid an external RNG crate (keeps the build offline and
//! dependency-free) and use SplitMix64, which is fast and statistically fine
//! for curriculum generation.

pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        // Avoid the all-zero fixed point degeneracy.
        SplitMix64 {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform-ish integer in [0, n). Modulo bias is negligible for the small
    /// ranges used here.
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            0
        } else {
            self.next_u64() % n
        }
    }
}

/// Draw a base-10 integer with exactly `digits` decimal digits.
/// `digits == 1` yields 0..=9; otherwise [10^(d-1), 10^d).
fn int_with_digits(rng: &mut SplitMix64, digits: u32) -> i128 {
    let d = digits.clamp(1, 18);
    if d == 1 {
        rng.below(10) as i128
    } else {
        let lo = 10u64.pow(d - 1);
        let hi = 10u64.pow(d); // exclusive upper bound
        (lo + rng.below(hi - lo)) as i128
    }
}

/// Sample a `(problem_expr, exact_answer)` pair for the given operation and
/// operand widths. `op` is one of '+', '-', '*'.
pub fn sample_problem(op: char, a_digits: u32, b_digits: u32, seed: u64) -> (String, String) {
    let mut rng = SplitMix64::new(seed);
    let a = int_with_digits(&mut rng, a_digits);
    let b = int_with_digits(&mut rng, b_digits);
    let (expr, ans) = match op {
        '-' => (format!("{}-{}", a, b), a - b),
        '*' => (format!("{}*{}", a, b), a * b),
        _ => (format!("{}+{}", a, b), a + b),
    };
    (expr, ans.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith;

    #[test]
    fn produces_solvable_problems() {
        for op in ['+', '-', '*'] {
            for seed in 0..200u64 {
                let (expr, ans) = sample_problem(op, 3, 2, seed);
                assert!(
                    arith::verify(&expr, &ans),
                    "op={op} expr={expr} ans={ans} failed verification"
                );
            }
        }
    }

    #[test]
    fn deterministic_by_seed() {
        assert_eq!(sample_problem('+', 4, 4, 123), sample_problem('+', 4, 4, 123));
    }

    #[test]
    fn digit_widths_respected() {
        let (expr, _) = sample_problem('+', 3, 1, 7);
        let a = expr.split('+').next().unwrap();
        assert_eq!(a.len(), 3);
    }
}
