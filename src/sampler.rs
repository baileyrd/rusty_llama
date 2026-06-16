//! Token sampling: greedy, temperature, and top-p (nucleus).
//!
//! The RNG is the same xorshift used by llama2.c, so a given seed reproduces
//! the reference implementation's draws.

use std::cmp::Ordering;

use crate::math::softmax;

/// Draws the next token from a logit distribution.
pub struct Sampler {
    temperature: f32,
    topp: f32,
    rng_state: u64,
    /// Scratch reused by top-p sampling: `(probability, token_id)`.
    probindex: Vec<(f32, usize)>,
}

impl Sampler {
    /// Create a sampler.
    ///
    /// * `temperature == 0` → greedy (argmax).
    /// * `topp <= 0 || topp >= 1` → plain multinomial sampling.
    /// * otherwise → nucleus sampling over the smallest set of tokens whose
    ///   cumulative probability exceeds `topp`.
    pub fn new(vocab_size: usize, temperature: f32, topp: f32, seed: u64) -> Self {
        Sampler {
            temperature,
            topp,
            // xorshift must not start at zero.
            rng_state: if seed == 0 { 1 } else { seed },
            probindex: Vec::with_capacity(vocab_size),
        }
    }

    /// Sample the next token id from `logits` (mutated in place when scaling).
    pub fn sample(&mut self, logits: &mut [f32]) -> usize {
        if self.temperature == 0.0 {
            return argmax(logits);
        }
        for l in logits.iter_mut() {
            *l /= self.temperature;
        }
        softmax(logits);
        let coin = self.random_f32();
        if self.topp <= 0.0 || self.topp >= 1.0 {
            sample_mult(logits, coin)
        } else {
            self.sample_topp(logits, coin)
        }
    }

    /// xorshift64; returns the top 32 bits as in llama2.c.
    fn random_u32(&mut self) -> u32 {
        let mut s = self.rng_state;
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        self.rng_state = s;
        (s.wrapping_mul(0x2545F491_4F6CDD1D) >> 32) as u32
    }

    /// Uniform float in `[0, 1)`.
    fn random_f32(&mut self) -> f32 {
        (self.random_u32() >> 8) as f32 / 16_777_216.0
    }

    /// Nucleus (top-p) sampling.
    fn sample_topp(&mut self, probs: &[f32], coin: f32) -> usize {
        let n = probs.len();
        // Pre-filter: ignore tokens that can't possibly be in the nucleus.
        let cutoff = (1.0 - self.topp) / (n - 1).max(1) as f32;
        self.probindex.clear();
        for (i, &p) in probs.iter().enumerate() {
            if p >= cutoff {
                self.probindex.push((p, i));
            }
        }
        if self.probindex.is_empty() {
            return argmax(probs);
        }
        self.probindex
            .sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        // Truncate to the smallest prefix whose mass exceeds topp.
        let mut cumulative = 0.0;
        let mut last = self.probindex.len() - 1;
        for (k, &(p, _)) in self.probindex.iter().enumerate() {
            cumulative += p;
            if cumulative > self.topp {
                last = k;
                break;
            }
        }

        // Sample within the truncated, renormalized distribution.
        let target = coin * cumulative;
        let mut cdf = 0.0;
        for &(p, idx) in &self.probindex[..=last] {
            cdf += p;
            if target < cdf {
                return idx;
            }
        }
        self.probindex[last].1
    }
}

/// Index of the maximum element.
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Sample from a normalized distribution by inverse-CDF.
fn sample_mult(probs: &[f32], coin: f32) -> usize {
    let mut cdf = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cdf += p;
        if coin < cdf {
            return i;
        }
    }
    probs.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let mut s = Sampler::new(4, 0.0, 0.9, 1);
        let mut logits = vec![0.1, 0.2, 5.0, 0.3];
        assert_eq!(s.sample(&mut logits), 2);
    }

    #[test]
    fn same_seed_is_reproducible() {
        let draw = || {
            let mut s = Sampler::new(4, 1.0, 0.0, 12345);
            let mut logits = vec![1.0, 2.0, 3.0, 4.0];
            s.sample(&mut logits)
        };
        assert_eq!(draw(), draw());
    }

    #[test]
    fn topp_stays_in_range() {
        let mut s = Sampler::new(4, 1.0, 0.9, 7);
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let id = s.sample(&mut logits);
        assert!(id < 4);
    }
}
