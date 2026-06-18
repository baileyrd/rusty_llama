//! Test-only benchmarking helpers shared by the backend throughput benches.
//!
//! Brings the in-crate benches in line with `llama-bench`'s protocol: one (or
//! more) untimed warmup runs, then `-r` timed repetitions reported as
//! **mean ± sample-stddev** tok/s. Tokenize/sampling are excluded (the benches
//! feed token ids straight in). See
//! `docs/Architecture/08-testing-benchmarking-parity.md` for the run recipe and
//! the `llama-bench` invocation each figure is compared against.

use std::time::{Duration, Instant};

/// Timed-repetition count (`llama-bench -r`); default 5, overridable via the
/// `RUSTY_LLAMA_BENCH_R` environment variable. Always at least 1.
pub(crate) fn bench_reps() -> usize {
    std::env::var("RUSTY_LLAMA_BENCH_R")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&r| r >= 1)
        .unwrap_or(5)
}

/// Run `f` repeatedly for at least `warm` (untimed) to ramp the GPU boost clock,
/// then `reps` timed times; return the mean and sample standard deviation of
/// throughput (tok/s) given `tokens` processed per call. `warm == Duration::ZERO`
/// skips warmup (e.g. the CPU baseline); `reps` is clamped to at least 1.
///
/// Fast (sub-100 ms) GPU passes need a multi-second `warm`: the boost clock ramps
/// over several seconds of sustained load, so a brief warmup (or a short timed
/// window) samples a cold, under-boosted clock and under-reports throughput —
/// long passes self-ramp. See `docs/Architecture/08-testing-benchmarking-parity.md`.
pub(crate) fn bench_stat(
    tokens: usize,
    reps: usize,
    warm: Duration,
    mut f: impl FnMut(),
) -> (f64, f64) {
    if !warm.is_zero() {
        let t = Instant::now();
        loop {
            f();
            if t.elapsed() >= warm {
                break;
            }
        }
    }
    let reps = reps.max(1);
    let mut tps = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f();
        tps.push(tokens as f64 / t.elapsed().as_secs_f64());
    }
    let mean = tps.iter().sum::<f64>() / reps as f64;
    let var = tps.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / reps as f64;
    (mean, var.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn bench_stat_measures_throughput() {
        // "Process 1000 tokens" by sleeping 10 ms ⇒ ~100k tok/s. The warmup is
        // excluded; the spread across reps stays well under the mean.
        let reps = bench_reps();
        assert!(reps >= 1);
        let (mean, sd) = bench_stat(1000, reps, Duration::from_millis(30), || {
            std::thread::sleep(Duration::from_millis(10))
        });
        // 1000 / 0.01 s = 100_000 tok/s; allow generous scheduling slack.
        assert!((40_000.0..200_000.0).contains(&mean), "mean={mean}");
        assert!(sd < mean, "stddev {sd} should be well below mean {mean}");
    }
}
