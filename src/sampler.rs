//! Token sampling as a composable chain of [`SamplerStage`]s.
//!
//! The default chain ([`SamplerChain::new`]) reproduces the llama2.c reference
//! exactly — greedy / temperature / top-p nucleus, with the same xorshift RNG,
//! so a given seed reproduces the reference draws. Phase 1 hangs top-k / min-p /
//! penalties off this seam; Phase 5 adds grammar-constrained sampling.

use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};

/// One candidate token: id + (current) logit + (filled) probability.
/// Mirrors llama.cpp `llama_token_data`.
#[derive(Clone, Copy, Debug)]
pub struct TokenData {
    pub id: u32,
    pub logit: f32,
    pub p: f32,
}

/// The unit of work every [`SamplerStage`] mutates in place. `size` is the live
/// prefix (`data[..size]`): a filter truncates by shrinking `size`, never by
/// reallocating. `selected` is an INDEX into `data` (not a token id). `sorted`
/// records that `data[..size]` is sorted descending by `p` AND that `p` holds a
/// valid probability distribution. Mirrors `llama_token_data_array`.
pub struct TokenDataArray {
    pub data: Vec<TokenData>,
    pub size: usize,
    pub selected: Option<usize>,
    pub sorted: bool,
}

impl TokenDataArray {
    fn with_capacity(vocab_size: usize) -> Self {
        TokenDataArray {
            data: Vec::with_capacity(vocab_size),
            size: 0,
            selected: None,
            sorted: false,
        }
    }

    /// Reload the candidate set from a fresh logit vector: `id = index`,
    /// `logit = value`, full live prefix, nothing selected or sorted.
    fn fill_from(&mut self, logits: &[f32]) {
        self.data.clear();
        self.data.extend(logits.iter().enumerate().map(|(i, &l)| TokenData {
            id: i as u32,
            logit: l,
            p: 0.0,
        }));
        self.size = self.data.len();
        self.selected = None;
        self.sorted = false;
    }

    /// Numerically-stable softmax `logit -> p` over the live prefix. Identical
    /// math to [`crate::math::softmax`] (subtract max, exp, normalize).
    fn softmax_p(&mut self) {
        let live = &mut self.data[..self.size];
        if live.is_empty() {
            return;
        }
        let max = live.iter().fold(f32::NEG_INFINITY, |m, c| m.max(c.logit));
        let mut sum = 0.0f32;
        for c in live.iter_mut() {
            c.p = (c.logit - max).exp();
            sum += c.p;
        }
        let inv = 1.0 / sum;
        for c in live.iter_mut() {
            c.p *= inv;
        }
    }
}

/// Index (into the slice) of the maximum `logit`: last of equal maxima, NaN
/// treated as Equal. Matches the legacy llama2.c `argmax` tie/NaN rule verbatim.
fn argmax_live(data: &[TokenData]) -> usize {
    data.iter()
        .enumerate()
        .max_by(|a, b| a.1.logit.partial_cmp(&b.1.logit).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// A composable sampling step. `apply` is the only required hook; `accept` lets
/// stateful stages (Phase-1 penalties, Phase-5 grammar) observe the chosen
/// token; `reset` clears per-sequence state. Mirrors `llama_sampler_i`.
pub trait SamplerStage: Send {
    fn apply(&mut self, cur: &mut TokenDataArray);
    fn accept(&mut self, _token: u32) {}
    fn reset(&mut self) {}
    fn name(&self) -> &'static str;
}

/// Divide every live logit by the temperature. Maps to `*l /= temperature`.
pub struct Temp {
    pub t: f32,
}

impl SamplerStage for Temp {
    fn apply(&mut self, cur: &mut TokenDataArray) {
        for c in cur.data[..cur.size].iter_mut() {
            c.logit /= self.t;
        }
        cur.sorted = false;
    }
    fn name(&self) -> &'static str {
        "temp"
    }
}

/// Nucleus filter: keep the smallest set of highest-probability tokens whose
/// cumulative probability exceeds `p`. Reproduces the legacy `sample_topp`
/// filter/sort/truncate; the inverse-CDF draw is the selector's job.
pub struct TopP {
    pub p: f32,
}

impl SamplerStage for TopP {
    fn apply(&mut self, cur: &mut TokenDataArray) {
        let n = cur.size;
        cur.softmax_p();
        // Pre-filter: drop tokens that can't be in the nucleus, compacting the
        // survivors to the front of the live prefix (ascending index order is
        // preserved, matching the legacy `probindex` build order).
        let cutoff = (1.0 - self.p) / (n - 1).max(1) as f32;
        let mut kept = 0;
        for i in 0..n {
            if cur.data[i].p >= cutoff {
                cur.data.swap(kept, i);
                kept += 1;
            }
        }
        if kept == 0 {
            // Degenerate distribution: collapse to the argmax candidate so the
            // selector returns it (legacy `return argmax(probs)`), still drawing
            // its coin so the RNG advances identically.
            let am = argmax_live(&cur.data[..n]);
            cur.data.swap(0, am);
            cur.size = 1;
            cur.sorted = true;
            return;
        }
        // Sort survivors descending by probability (same comparator + algorithm
        // as the legacy code, so tie order is identical).
        cur.data[..kept].sort_unstable_by(|a, b| b.p.partial_cmp(&a.p).unwrap_or(Ordering::Equal));
        // Truncate to the smallest prefix whose cumulative mass exceeds `p`.
        let mut cumulative = 0.0;
        let mut last = kept - 1;
        for (k, c) in cur.data[..kept].iter().enumerate() {
            cumulative += c.p;
            if cumulative > self.p {
                last = k;
                break;
            }
        }
        cur.size = last + 1;
        cur.sorted = true;
    }
    fn name(&self) -> &'static str {
        "top_p"
    }
}

/// Stochastic selector (inverse-CDF). Carries the llama2.c xorshift RNG, so a
/// given seed reproduces the reference draws. Reproduces `sample_mult` (full
/// distribution, `!sorted`) and the `sample_topp` tail (sorted/truncated
/// nucleus, `sorted`).
pub struct Dist {
    rng: u64,
}

impl Dist {
    /// xorshift64; top 32 bits, as in llama2.c.
    fn random_u32(&mut self) -> u32 {
        let mut s = self.rng;
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        self.rng = s;
        (s.wrapping_mul(0x2545F491_4F6CDD1D) >> 32) as u32
    }
    /// Uniform float in `[0, 1)`.
    fn random_f32(&mut self) -> f32 {
        (self.random_u32() >> 8) as f32 / 16_777_216.0
    }
}

impl SamplerStage for Dist {
    fn apply(&mut self, cur: &mut TokenDataArray) {
        if !cur.sorted {
            cur.softmax_p();
        }
        let coin = self.random_f32();
        let size = cur.size;
        // Sorted/truncated nucleus renormalizes via `coin * cumulative`; the full
        // distribution (already summing to ~1) uses `coin` directly.
        let target = if cur.sorted {
            let cumulative: f32 = cur.data[..size].iter().map(|c| c.p).sum();
            coin * cumulative
        } else {
            coin
        };
        let mut cdf = 0.0;
        let mut sel = size - 1;
        for (k, c) in cur.data[..size].iter().enumerate() {
            cdf += c.p;
            if target < cdf {
                sel = k;
                break;
            }
        }
        cur.selected = Some(sel);
    }
    fn name(&self) -> &'static str {
        "dist"
    }
}

/// Greedy selector: pick the argmax logit (last of ties, NaN treated as Equal),
/// matching the legacy `argmax`. Draws no random number.
pub struct Greedy;

impl SamplerStage for Greedy {
    fn apply(&mut self, cur: &mut TokenDataArray) {
        cur.selected = Some(argmax_live(&cur.data[..cur.size]));
    }
    fn name(&self) -> &'static str {
        "greedy"
    }
}

/// Keep only the `k` highest-logit candidates. Off when `k == 0` or `k >= size`.
pub struct TopK {
    pub k: usize,
}

impl SamplerStage for TopK {
    fn apply(&mut self, cur: &mut TokenDataArray) {
        if self.k == 0 || self.k >= cur.size {
            return;
        }
        // Partition the top-k by logit to the front (O(n)); order within is
        // irrelevant (top-p/dist re-softmaxes). Breaks any prior p-sort.
        cur.data[..cur.size].select_nth_unstable_by(self.k - 1, |a, b| {
            b.logit.partial_cmp(&a.logit).unwrap_or(Ordering::Equal)
        });
        cur.size = self.k;
        cur.sorted = false;
    }
    fn name(&self) -> &'static str {
        "top_k"
    }
}

/// Min-p filter: keep candidates whose probability is at least `p`× the max,
/// never fewer than `min_keep`. Compared in logit space (`logit >= max + ln p`)
/// to avoid a softmax. Off when `p <= 0`.
pub struct MinP {
    pub p: f32,
    pub min_keep: usize,
}

impl SamplerStage for MinP {
    fn apply(&mut self, cur: &mut TokenDataArray) {
        let n = cur.size;
        if self.p <= 0.0 || n < 2 {
            return;
        }
        let max_logit = cur.data[..n]
            .iter()
            .fold(f32::NEG_INFINITY, |m, c| m.max(c.logit));
        let thresh = max_logit + self.p.ln();
        // Sort desc by logit so survivors form a prefix and `min_keep` is exact.
        cur.data[..n]
            .sort_unstable_by(|a, b| b.logit.partial_cmp(&a.logit).unwrap_or(Ordering::Equal));
        let above = cur.data[..n].iter().take_while(|c| c.logit >= thresh).count();
        cur.size = above.max(self.min_keep).max(1).min(n);
        cur.sorted = false;
    }
    fn name(&self) -> &'static str {
        "min_p"
    }
}

/// Repetition / frequency / presence penalties over a ring of the last `last_n`
/// accepted tokens (llama.cpp semantics). Off when all three are neutral.
/// Stateful: `accept` feeds the ring, `reset` clears it.
pub struct Penalties {
    last_n: usize,
    repeat: f32,
    freq: f32,
    present: f32,
    ring: VecDeque<u32>,
}

impl Penalties {
    pub fn new(last_n: usize, repeat: f32, freq: f32, present: f32) -> Self {
        Penalties {
            last_n,
            repeat,
            freq,
            present,
            ring: VecDeque::with_capacity(last_n.min(1024)),
        }
    }

    fn is_off(&self) -> bool {
        self.repeat == 1.0 && self.freq == 0.0 && self.present == 0.0
    }
}

impl SamplerStage for Penalties {
    fn apply(&mut self, cur: &mut TokenDataArray) {
        if self.ring.is_empty() || self.is_off() {
            return;
        }
        // Occurrence count per token id over the window (O(last_n)).
        let mut counts: HashMap<u32, u32> = HashMap::new();
        for &t in &self.ring {
            *counts.entry(t).or_insert(0) += 1;
        }
        for c in cur.data[..cur.size].iter_mut() {
            if let Some(&cnt) = counts.get(&c.id) {
                // repeat: scale toward zero; freq: per-occurrence; presence: once.
                c.logit = if c.logit > 0.0 {
                    c.logit / self.repeat
                } else {
                    c.logit * self.repeat
                };
                c.logit -= cnt as f32 * self.freq;
                c.logit -= self.present;
            }
        }
        cur.sorted = false;
    }

    fn accept(&mut self, token: u32) {
        if self.last_n == 0 {
            return;
        }
        if self.ring.len() == self.last_n {
            self.ring.pop_front();
        }
        self.ring.push_back(token);
    }

    fn reset(&mut self) {
        self.ring.clear();
    }

    fn name(&self) -> &'static str {
        "penalties"
    }
}

/// Mirostat v1: targets a fixed output-entropy `tau` ("surprise", in bits) by
/// estimating the local Zipf exponent from the top `m` candidates' probability
/// ratios, solving for the top-k cutoff that keeps expected surprise near the
/// adaptive ceiling `mu` (initialized to `2*tau`), then nudging `mu` toward the
/// actually observed surprise by learning rate `eta`. Mirrors
/// `llama_sampler_mirostat`. A *selector* stage — like [`Dist`]/[`Greedy`] it
/// terminates the chain by setting `cur.selected`, so it must run last.
pub struct Mirostat {
    tau: f32,
    eta: f32,
    m: usize,
    mu: f32,
    rng: u64,
}

impl Mirostat {
    pub fn new(tau: f32, eta: f32, m: usize, seed: u64) -> Self {
        Mirostat {
            tau,
            eta,
            m,
            mu: 2.0 * tau,
            rng: if seed == 0 { 1 } else { seed },
        }
    }

    /// xorshift64 (same construction as [`Dist`]'s, but its own independent
    /// stream — mirostat's draws don't need to line up with the plain-`Dist`
    /// sequence, only be reproducible for a given seed).
    fn random_u32(&mut self) -> u32 {
        let mut s = self.rng;
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        self.rng = s;
        (s.wrapping_mul(0x2545F491_4F6CDD1D) >> 32) as u32
    }
    fn random_f32(&mut self) -> f32 {
        (self.random_u32() >> 8) as f32 / 16_777_216.0
    }
}

impl SamplerStage for Mirostat {
    fn apply(&mut self, cur: &mut TokenDataArray) {
        let n = cur.size;
        // Sort descending by logit, then softmax — matches llama.cpp's
        // `softmax_impl`, which sorts before normalizing, so adjacent slots in
        // `data` are adjacent probability ranks (the Zipf estimate below needs
        // that).
        cur.data[..n]
            .sort_unstable_by(|a, b| b.logit.partial_cmp(&a.logit).unwrap_or(Ordering::Equal));
        cur.sorted = true;
        cur.softmax_p();

        // Estimate the local Zipf exponent from the top `m` candidates' p ratios.
        let bound = self.m.saturating_sub(1).min(n.saturating_sub(1));
        let mut sum_ti_bi = 0.0f64;
        let mut sum_ti_sq = 0.0f64;
        for i in 0..bound {
            let t_i = ((i + 2) as f64 / (i + 1) as f64).ln();
            let b_i = (cur.data[i].p as f64 / cur.data[i + 1].p as f64).ln();
            sum_ti_bi += t_i * b_i;
            sum_ti_sq += t_i * t_i;
        }
        let s_hat = sum_ti_bi / sum_ti_sq;

        // Solve for the top-k cutoff that keeps expected surprise near `mu`.
        let epsilon_hat = s_hat - 1.0;
        let k = ((epsilon_hat * 2f64.powf(self.mu as f64))
            / (1.0 - (-epsilon_hat).powf(n as f64)))
        .powf(1.0 / s_hat);
        let keep = if k.is_finite() { k as i64 } else { 1 }.max(1).min(n as i64) as usize;
        cur.size = keep;
        // Renormalize over the truncated (still-sorted) set.
        cur.softmax_p();

        let coin = self.random_f32();
        let size = cur.size;
        let mut cdf = 0.0f32;
        let mut sel = size - 1;
        for (k, c) in cur.data[..size].iter().enumerate() {
            cdf += c.p;
            if coin < cdf {
                sel = k;
                break;
            }
        }
        cur.selected = Some(sel);

        let observed_surprise = -cur.data[sel].p.log2();
        let e = observed_surprise - self.tau;
        self.mu -= self.eta * e;
    }
    fn reset(&mut self) {
        self.mu = 2.0 * self.tau;
    }
    fn name(&self) -> &'static str {
        "mirostat"
    }
}

/// Mirostat v2: the simpler variant — no Zipf estimate, just truncate to the
/// prefix of candidates whose surprise doesn't exceed the adaptive ceiling
/// `mu`, sample from what's left, then nudge `mu` the same way as
/// [`Mirostat`]. Mirrors `llama_sampler_mirostat_v2`. Also a selector stage.
pub struct MirostatV2 {
    tau: f32,
    eta: f32,
    mu: f32,
    rng: u64,
}

impl MirostatV2 {
    pub fn new(tau: f32, eta: f32, seed: u64) -> Self {
        MirostatV2 {
            tau,
            eta,
            mu: 2.0 * tau,
            rng: if seed == 0 { 1 } else { seed },
        }
    }

    fn random_u32(&mut self) -> u32 {
        let mut s = self.rng;
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        self.rng = s;
        (s.wrapping_mul(0x2545F491_4F6CDD1D) >> 32) as u32
    }
    fn random_f32(&mut self) -> f32 {
        (self.random_u32() >> 8) as f32 / 16_777_216.0
    }
}

impl SamplerStage for MirostatV2 {
    fn apply(&mut self, cur: &mut TokenDataArray) {
        let n = cur.size;
        cur.data[..n]
            .sort_unstable_by(|a, b| b.logit.partial_cmp(&a.logit).unwrap_or(Ordering::Equal));
        cur.sorted = true;
        cur.softmax_p();

        // Truncate to the prefix whose surprise (-log2 p) doesn't exceed mu.
        let cutoff = cur.data[..n]
            .iter()
            .position(|c| -c.p.log2() > self.mu)
            .unwrap_or(n);
        cur.size = cutoff.max(1);
        cur.softmax_p();

        let coin = self.random_f32();
        let size = cur.size;
        let mut cdf = 0.0f32;
        let mut sel = size - 1;
        for (k, c) in cur.data[..size].iter().enumerate() {
            cdf += c.p;
            if coin < cdf {
                sel = k;
                break;
            }
        }
        cur.selected = Some(sel);

        let observed_surprise = -cur.data[sel].p.log2();
        let e = observed_surprise - self.tau;
        self.mu -= self.eta * e;
    }
    fn reset(&mut self) {
        self.mu = 2.0 * self.tau;
    }
    fn name(&self) -> &'static str {
        "mirostat_v2"
    }
}

/// Knobs for [`SamplerChain::from_config`] — one per stage; off-values omit the
/// stage. When every Phase-1 knob is at its off-value the builder produces the
/// exact Phase-0.1 chain (byte-identical output).
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub min_keep: usize,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub seed: u64,
    /// 0 = off, 1 = Mirostat v1, 2 = Mirostat v2. When on, this bypasses
    /// `top_k`/`top_p`/`min_p` entirely (mirostat's own truncation replaces
    /// them), matching llama.cpp's mutually-exclusive sampling modes.
    pub mirostat: u8,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
    /// Only used by Mirostat v1 (the Zipf-exponent estimate window).
    pub mirostat_m: usize,
}

/// An ordered chain of [`SamplerStage`]s ending in a selector. Replaces the
/// monolithic legacy sampler; [`SamplerChain::new`] builds the exact default.
pub struct SamplerChain {
    stages: Vec<Box<dyn SamplerStage>>,
    cur: TokenDataArray,
}

impl SamplerChain {
    /// Build the legacy default chain (byte-identical to the old `Sampler`):
    ///
    /// * `temperature == 0` → greedy (argmax).
    /// * `topp <= 0 || topp >= 1` → temperature + multinomial.
    /// * otherwise → temperature + nucleus (top-p) + multinomial.
    pub fn new(vocab_size: usize, temperature: f32, topp: f32, seed: u64) -> Self {
        // xorshift must not start at zero.
        let rng = if seed == 0 { 1 } else { seed };
        let stages: Vec<Box<dyn SamplerStage>> = if temperature == 0.0 {
            vec![Box::new(Greedy)]
        } else if topp <= 0.0 || topp >= 1.0 {
            vec![Box::new(Temp { t: temperature }), Box::new(Dist { rng })]
        } else {
            vec![
                Box::new(Temp { t: temperature }),
                Box::new(TopP { p: topp }),
                Box::new(Dist { rng }),
            ]
        };
        SamplerChain {
            stages,
            cur: TokenDataArray::with_capacity(vocab_size),
        }
    }

    /// Build a chain from a [`SamplerConfig`]. With every Phase-1 knob off this
    /// delegates to [`SamplerChain::new`] (byte-identical to the legacy default);
    /// otherwise it builds the canonical llama.cpp-order chain
    /// `penalties → top_k → top_p → min_p → temp → dist` (or `… → greedy`).
    pub fn from_config(cfg: &SamplerConfig, vocab_size: usize) -> Self {
        let penalties_on = cfg.repeat_penalty != 1.0
            || cfg.frequency_penalty != 0.0
            || cfg.presence_penalty != 0.0;
        if cfg.mirostat != 0 {
            // Mirostat is its own mode: penalties (if on) + temp + mirostat,
            // bypassing top_k/top_p/min_p entirely (mirrors llama.cpp's
            // if/else-if mirostat/non-mirostat chain construction).
            let mut stages: Vec<Box<dyn SamplerStage>> = Vec::new();
            if penalties_on {
                stages.push(Box::new(Penalties::new(
                    cfg.repeat_last_n,
                    cfg.repeat_penalty,
                    cfg.frequency_penalty,
                    cfg.presence_penalty,
                )));
            }
            stages.push(Box::new(Temp { t: cfg.temperature }));
            if cfg.mirostat == 1 {
                stages.push(Box::new(Mirostat::new(
                    cfg.mirostat_tau,
                    cfg.mirostat_eta,
                    cfg.mirostat_m,
                    cfg.seed,
                )));
            } else {
                stages.push(Box::new(MirostatV2::new(
                    cfg.mirostat_tau,
                    cfg.mirostat_eta,
                    cfg.seed,
                )));
            }
            return SamplerChain {
                stages,
                cur: TokenDataArray::with_capacity(vocab_size),
            };
        }
        let new_knobs = cfg.top_k != 0 || cfg.min_p > 0.0 || penalties_on;
        if !new_knobs {
            // Byte-identical Phase-0.1 chain (temp → top_p → dist, or greedy).
            return SamplerChain::new(vocab_size, cfg.temperature, cfg.top_p, cfg.seed);
        }
        let rng = if cfg.seed == 0 { 1 } else { cfg.seed };
        let mut stages: Vec<Box<dyn SamplerStage>> = Vec::new();
        if penalties_on {
            stages.push(Box::new(Penalties::new(
                cfg.repeat_last_n,
                cfg.repeat_penalty,
                cfg.frequency_penalty,
                cfg.presence_penalty,
            )));
        }
        if cfg.top_k != 0 {
            stages.push(Box::new(TopK { k: cfg.top_k }));
        }
        if cfg.top_p > 0.0 && cfg.top_p < 1.0 {
            stages.push(Box::new(TopP { p: cfg.top_p }));
        }
        if cfg.min_p > 0.0 {
            stages.push(Box::new(MinP {
                p: cfg.min_p,
                min_keep: cfg.min_keep.max(1),
            }));
        }
        if cfg.temperature == 0.0 {
            stages.push(Box::new(Greedy));
        } else {
            stages.push(Box::new(Temp { t: cfg.temperature }));
            stages.push(Box::new(Dist { rng }));
        }
        SamplerChain {
            stages,
            cur: TokenDataArray::with_capacity(vocab_size),
        }
    }

    /// Sample the next token id from `logits` (read-only — the chain works on its
    /// own candidate buffer). Runs every stage's `apply` in order, reads the
    /// selector's choice, then runs every stage's `accept`.
    pub fn sample(&mut self, logits: &[f32]) -> usize {
        self.cur.fill_from(logits);
        for stage in self.stages.iter_mut() {
            stage.apply(&mut self.cur);
        }
        let sel = self
            .cur
            .selected
            .expect("a sampler chain must end in a selector stage");
        let id = self.cur.data[sel].id;
        for stage in self.stages.iter_mut() {
            stage.accept(id);
        }
        id as usize
    }

    /// Insert `stage` at the front of the chain (runs first in both `apply` and
    /// `accept`). Used to add a grammar constraint ahead of the truncation /
    /// selector stages, so they only ever see grammar-valid candidates.
    pub fn prepend(&mut self, stage: Box<dyn SamplerStage>) {
        self.stages.insert(0, stage);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `TokenDataArray` loaded from `logits` (test helper).
    fn tda(logits: &[f32]) -> TokenDataArray {
        let mut t = TokenDataArray::with_capacity(logits.len());
        t.fill_from(logits);
        t
    }

    #[test]
    fn greedy_picks_argmax() {
        let mut s = SamplerChain::new(4, 0.0, 0.9, 1);
        assert_eq!(s.sample(&[0.1, 0.2, 5.0, 0.3]), 2);
    }

    #[test]
    fn greedy_last_of_ties() {
        // `argmax_live` returns the last of equal maxima (verbatim legacy rule).
        let mut s = SamplerChain::new(3, 0.0, 0.9, 1);
        assert_eq!(s.sample(&[1.0, 1.0, 1.0]), 2);
    }

    #[test]
    fn greedy_ignores_non_terminal_nan() {
        // A NaN compares Equal, so a later real value replaces it and the true
        // max wins — matches the legacy `argmax` comparator.
        let mut s = SamplerChain::new(3, 0.0, 0.9, 1);
        assert_eq!(s.sample(&[f32::NAN, 0.0, 5.0]), 2);
    }

    #[test]
    fn same_seed_is_reproducible() {
        let draw = || {
            let mut s = SamplerChain::new(4, 1.0, 0.0, 12345);
            s.sample(&[1.0, 2.0, 3.0, 4.0])
        };
        assert_eq!(draw(), draw());
    }

    #[test]
    fn topp_stays_in_range() {
        let mut s = SamplerChain::new(4, 1.0, 0.9, 7);
        assert!(s.sample(&[1.0, 2.0, 3.0, 4.0]) < 4);
    }

    /// Golden stochastic stream captured (PR 0.1a) from the chain while it was
    /// proven token-for-token equal to the legacy `Sampler`; pins byte-identical
    /// behavior now that the legacy code is deleted.
    #[test]
    fn chain_stochastic_golden() {
        let logits = [2.0f32, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6];
        let mut sc = SamplerChain::new(8, 1.0, 0.9, 12345);
        let stream: Vec<usize> = (0..8).map(|_| sc.sample(&logits)).collect();
        assert_eq!(stream, vec![2, 4, 0, 4, 0, 0, 6, 2]);
    }

    #[test]
    fn temp_scales_logits() {
        let mut cur = tda(&[2.0, 4.0, 6.0, 8.0]);
        Temp { t: 2.0 }.apply(&mut cur);
        let got: Vec<f32> = cur.data.iter().map(|c| c.logit).collect();
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn top_p_truncates_to_nucleus() {
        // One logit dominates → nucleus is the single top token, marked sorted.
        let mut cur = tda(&[0.0, 0.0, 10.0, 0.0]);
        TopP { p: 0.9 }.apply(&mut cur);
        assert!(cur.sorted);
        assert_eq!(cur.size, 1);
        assert_eq!(cur.data[0].id, 2);
    }

    #[test]
    fn top_p_empty_collapses_to_argmax() {
        // Uniform dist + tiny top-p ⇒ every prob below the cutoff ⇒ empty
        // nucleus collapses to the argmax (last of ties).
        let mut cur = tda(&[1.0, 1.0, 1.0, 1.0]);
        TopP { p: 0.1 }.apply(&mut cur);
        assert_eq!(cur.size, 1);
        assert_eq!(cur.data[0].id, 3);
    }

    #[test]
    fn accept_and_reset_are_noops_for_stateless_stages() {
        // Guards the Phase-1 penalty/grammar hook: stateless stages ignore
        // accept/reset without perturbing selection.
        let mut t = Temp { t: 1.0 };
        t.accept(3);
        t.reset();
        let mut cur = tda(&[1.0, 9.0, 2.0]);
        t.apply(&mut cur);
        Greedy.apply(&mut cur);
        assert_eq!(cur.selected, Some(1));
    }

    #[test]
    fn vocab_of_one() {
        assert_eq!(SamplerChain::new(1, 0.0, 0.9, 1).sample(&[42.0]), 0);
        assert_eq!(SamplerChain::new(1, 1.0, 0.9, 1).sample(&[42.0]), 0);
    }

    #[test]
    fn top_k_keeps_k_highest() {
        let mut cur = tda(&[3.0, 1.0, 4.0, 1.0, 5.0]);
        TopK { k: 2 }.apply(&mut cur);
        assert_eq!(cur.size, 2);
        let ids: std::collections::HashSet<u32> = cur.data[..2].iter().map(|c| c.id).collect();
        assert_eq!(ids, [2, 4].into_iter().collect()); // logits 4 (id2) and 5 (id4)
    }

    #[test]
    fn top_k_zero_is_noop() {
        let mut cur = tda(&[3.0, 1.0, 4.0]);
        TopK { k: 0 }.apply(&mut cur);
        assert_eq!(cur.size, 3);
    }

    #[test]
    fn min_p_drops_low_probability_tail() {
        // thresh = 10 + ln(0.5) ≈ 9.31 ⇒ only the 10 (id 0) clears it.
        let mut cur = tda(&[10.0, 9.0, 2.0, 1.0]);
        MinP { p: 0.5, min_keep: 1 }.apply(&mut cur);
        assert_eq!(cur.size, 1);
        assert_eq!(cur.data[0].id, 0);
    }

    #[test]
    fn min_p_respects_min_keep() {
        let mut cur = tda(&[10.0, 9.0, 2.0, 1.0]);
        MinP { p: 0.5, min_keep: 3 }.apply(&mut cur);
        assert_eq!(cur.size, 3); // forced to keep 3 despite the threshold
    }

    #[test]
    fn penalty_demotes_repeated_token() {
        // id 2 is argmax (5.0); after a repeat penalty it falls below id 0 (4.5).
        let mut cur = tda(&[4.5, 1.0, 5.0]);
        let mut pen = Penalties::new(64, 1.3, 0.0, 0.0);
        pen.accept(2);
        pen.apply(&mut cur);
        assert!(cur.data[2].logit < cur.data[0].logit);
    }

    #[test]
    fn penalty_frequency_scales_with_count() {
        let logit_after = |n: usize| {
            let mut cur = tda(&[0.0, 0.0]);
            let mut pen = Penalties::new(64, 1.0, 1.0, 0.0);
            for _ in 0..n {
                pen.accept(0);
            }
            pen.apply(&mut cur);
            cur.data[0].logit
        };
        assert!(logit_after(2) < logit_after(1)); // seen twice penalized more
    }

    #[test]
    fn penalty_ring_evicts_past_window() {
        let mut pen = Penalties::new(2, 1.0, 0.0, 5.0); // presence-only
        pen.accept(0);
        pen.accept(1);
        pen.accept(2); // evicts 0
        let mut cur = tda(&[0.0, 0.0, 0.0]);
        pen.apply(&mut cur);
        assert_eq!(cur.data[0].logit, 0.0); // 0 evicted ⇒ unpenalized
        assert_eq!(cur.data[1].logit, -5.0);
        assert_eq!(cur.data[2].logit, -5.0);
    }

    #[test]
    fn stateless_stages_ignore_accept() {
        let mut tk = TopK { k: 2 };
        tk.accept(3); // no-op, must not panic or change behavior
        let mut cur = tda(&[1.0, 5.0, 2.0, 9.0]);
        tk.apply(&mut cur);
        assert_eq!(cur.size, 2);
    }

    #[test]
    fn mirostat_v2_truncates_and_updates_mu() {
        let mut m = MirostatV2::new(5.0, 0.1, 1);
        assert_eq!(m.mu, 10.0); // 2*tau
        // A dominant logit -> after softmax + truncation only token 0 survives
        // (its surprise is ~0, everything else's exceeds mu=10), so the
        // post-truncation softmax gives p=1 exactly: an exactly hand-computable
        // mu update.
        let mut cur = tda(&[10.0, 0.0, 0.0]);
        m.apply(&mut cur);
        assert_eq!(cur.size, 1);
        let sel = cur.selected.expect("mirostat_v2 must select");
        assert_eq!(cur.data[sel].id, 0);
        let want_mu = 10.0 - 0.1 * (0.0 - 5.0);
        assert!((m.mu - want_mu).abs() < 1e-4, "{} vs {want_mu}", m.mu);
    }

    #[test]
    fn mirostat_v2_reset_restores_mu() {
        let mut m = MirostatV2::new(3.0, 0.2, 42);
        let mut cur = tda(&[5.0, 1.0, 1.0]);
        m.apply(&mut cur);
        assert!((m.mu - 6.0).abs() > 1e-6, "mu should have moved from 2*tau");
        m.reset();
        assert_eq!(m.mu, 6.0);
    }

    #[test]
    fn mirostat_v1_collapses_and_updates_mu_hand_computed() {
        // A dominant logit drives the Zipf estimate's k formula out of its real
        // domain (negative base, fractional exponent -> NaN); the
        // implementation clamps that to a single-token keep rather than
        // propagating NaN, same end state as the v2 case above: p=1 exactly,
        // so the mu update is exactly hand-computable.
        let mut m = Mirostat::new(5.0, 0.1, 2, 7);
        assert_eq!(m.mu, 10.0);
        let mut cur = tda(&[10.0, 0.0, 0.0, 0.0]);
        m.apply(&mut cur);
        assert_eq!(cur.size, 1);
        let sel = cur.selected.expect("mirostat must select");
        assert_eq!(cur.data[sel].id, 0);
        let want_mu = 10.0 - 0.1 * (0.0 - 5.0);
        assert!((m.mu - want_mu).abs() < 1e-4, "{} vs {want_mu}", m.mu);
    }

    #[test]
    fn mirostat_v1_reset_restores_mu() {
        let mut m = Mirostat::new(4.0, 0.15, 100, 99);
        let mut cur = tda(&[3.0, 2.0, 1.0, 0.5, 0.1]);
        m.apply(&mut cur);
        m.reset();
        assert_eq!(m.mu, 8.0);
    }

    #[test]
    fn mirostat_chain_is_reproducible_and_valid() {
        let vocab = 16usize;
        let logits: Vec<f32> = (0..vocab).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        for mode in [1u8, 2u8] {
            let draw = || {
                let cfg = SamplerConfig {
                    temperature: 1.0,
                    top_k: 0,
                    top_p: 0.9,
                    min_p: 0.0,
                    min_keep: 1,
                    repeat_penalty: 1.0,
                    repeat_last_n: 64,
                    frequency_penalty: 0.0,
                    presence_penalty: 0.0,
                    seed: 4242,
                    mirostat: mode,
                    mirostat_tau: 5.0,
                    mirostat_eta: 0.1,
                    mirostat_m: 100,
                };
                let mut sc = SamplerChain::from_config(&cfg, vocab);
                (0..20).map(|_| sc.sample(&logits)).collect::<Vec<_>>()
            };
            let a = draw();
            let b = draw();
            assert_eq!(a, b, "mirostat mode {mode} must be reproducible for a fixed seed");
            assert!(a.iter().all(|&t| t < vocab), "mode {mode}: {a:?}");
        }
    }

    #[test]
    fn from_config_default_is_byte_identical() {
        // Every Phase-1 knob off ⇒ delegates to the legacy chain ⇒ the 0.1 golden.
        let cfg = SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.9,
            min_p: 0.0,
            min_keep: 1,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 12345,
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            mirostat_m: 100,
        };
        let logits = [2.0f32, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6];
        let mut sc = SamplerChain::from_config(&cfg, 8);
        let stream: Vec<usize> = (0..8).map(|_| sc.sample(&logits)).collect();
        assert_eq!(stream, vec![2, 4, 0, 4, 0, 0, 6, 2]);
    }
}
