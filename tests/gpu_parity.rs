//! End-to-end GPU↔CPU parity on a synthetic GGUF model.
//!
//! Builds a small synthetic `llama` GGUF (F32 and Q8_0), then checks that the
//! `GpuBackend` produces the same forward-pass logits and the same greedy token
//! stream as `CpuBackend`. The whole file is compiled only with `--features
//! gpu`; if no GPU adapter is present at run time the tests skip cleanly.
#![cfg(feature = "gpu")]

use rusty_llama::dummy::synthetic_gguf_typed;
use rusty_llama::{
    forward, forward_prefill, generate, Backend, Config, CpuBackend, GgmlType, Gguf, GpuBackend,
    Model, RunState, SamplerChain, Tokenizer,
};

/// dim/hidden are multiples of 32 so the same config serializes as Q8_0 too.
fn cfg() -> Config {
    Config {
        dim: 32,
        hidden_dim: 64,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2, // grouped-query attention
        vocab_size: 64,
        seq_len: 16,
        shared_weights: true,
        ..Default::default()
    }
}

/// A GPU backend, or `None` (with a note) when no adapter is available.
fn gpu() -> Option<GpuBackend> {
    match GpuBackend::new() {
        Ok(g) => {
            eprintln!("[gpu] adapter: {}", g.adapter_name());
            Some(g)
        }
        Err(e) => {
            eprintln!("skipping GPU e2e test: {e}");
            None
        }
    }
}

/// argmax of a logit vector (greedy's next token).
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

/// Single-step forward: GPU logits must track CPU logits, and pick the same
/// next token, for both F32 and Q8_0 weights.
#[test]
fn forward_logits_match_cpu() {
    let c = cfg();
    for ty in [GgmlType::F32, GgmlType::Q8_0] {
        // Fresh backend per model: the weight cache keys on data pointers.
        let Some(g) = gpu() else { return };
        let bytes = synthetic_gguf_typed(&c, ty);
        let gguf = Gguf::parse(&bytes).unwrap();
        let model = Model::from_gguf(&gguf).unwrap();

        let logits = |backend: &dyn Backend| -> Vec<f32> {
            let mut s = RunState::new(&model.config);
            forward(&model, &mut s, backend, 7, 0);
            s.logits().to_vec()
        };
        let cpu = logits(&CpuBackend::new());
        let gpu_l = logits(&g);

        assert!(gpu_l.iter().all(|v| v.is_finite()));
        let max_diff = cpu
            .iter()
            .zip(&gpu_l)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-2, "{ty:?}: logits diverge by {max_diff}");
        assert_eq!(argmax(&cpu), argmax(&gpu_l), "{ty:?}: greedy token differs");
    }
}

/// The whole batched prefill path (every batched GPU op, through all layers)
/// must track the CPU result and pick the same next token, for F32 and Q8_0.
#[test]
fn prefill_matches_cpu() {
    let c = cfg();
    for ty in [GgmlType::F32, GgmlType::Q8_0] {
        let Some(g) = gpu() else { return };
        let bytes = synthetic_gguf_typed(&c, ty);
        let gguf = Gguf::parse(&bytes).unwrap();
        let model = Model::from_gguf(&gguf).unwrap();
        let tokens: Vec<usize> = vec![3, 8, 1, 5, 9, 2, 40, 17];

        let logits = |backend: &dyn Backend| -> Vec<f32> {
            let mut s = RunState::new(&model.config);
            forward_prefill(&model, &mut s, backend, &tokens, 0);
            s.logits().to_vec()
        };
        let cpu = logits(&CpuBackend::new());
        let gpu_l = logits(&g);

        assert!(gpu_l.iter().all(|v| v.is_finite()));
        let max_diff = cpu
            .iter()
            .zip(&gpu_l)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-2, "{ty:?}: prefill logits diverge by {max_diff}");
        assert_eq!(argmax(&cpu), argmax(&gpu_l), "{ty:?}: prefill greedy token differs");
    }
}

/// Real Q4_K_M model parity — the synthetic builder only emits F32/Q8_0, so this is
/// the only coverage for the tiled **Q4_K and Q6_K** batch matmul shaders. Gated on
/// the GGUF being present (`RUSTY_LLAMA_GGUF` or the default TinyLlama path) and a
/// GPU; skips cleanly otherwise.
#[test]
fn prefill_matches_cpu_real_kquant() {
    let path = std::env::var("RUSTY_LLAMA_GGUF")
        .unwrap_or_else(|_| "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".into());
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping real-model k-quant parity: {path} not found");
        return;
    }
    let Some(g) = gpu() else { return };
    let bytes = std::fs::read(&path).unwrap();
    let gguf = Gguf::parse(&bytes).unwrap();
    let model = Model::from_gguf(&gguf).unwrap();
    let tokens: Vec<usize> = vec![1, 22, 333, 444, 555, 666, 777, 888];

    let logits = |backend: &dyn Backend| -> Vec<f32> {
        let mut s = RunState::new(&model.config);
        forward_prefill(&model, &mut s, backend, &tokens, 0);
        s.logits().to_vec()
    };
    let cpu = logits(&CpuBackend::new());
    let gpu_l = logits(&g);

    assert!(gpu_l.iter().all(|v| v.is_finite()));
    // Relative L2 across the vocab (real-model logits are large, so an absolute
    // bound isn't meaningful; cf. the CUDA coherence tests). A dequant/index bug in
    // the tiled k-quant shaders would corrupt the argmax, so assert that too.
    let num: f32 = cpu.iter().zip(&gpu_l).map(|(a, b)| (a - b) * (a - b)).sum();
    let den: f32 = cpu.iter().map(|a| a * a).sum::<f32>().max(1e-12);
    let rel = (num / den).sqrt();
    eprintln!(
        "real Q4_K_M prefill: argmax cpu={} gpu={} rel-L2={rel}",
        argmax(&cpu),
        argmax(&gpu_l)
    );
    // Regression guard for the tiled Q4_K/Q6_K batch shaders. NOTE: the wgpu *batched
    // prefill* already diverges from the CPU on real k-quant models (rel-L2 ~3.5%,
    // and the greedy token can differ) — a PRE-EXISTING issue, not caused by tiling.
    // (Verified: tiling preserves per-output FMA order, so tiled output is bit-
    // identical to the untiled shader — same rel-L2 to 8 sig figs.) So we can't
    // assert CPU argmax parity here; instead bound rel-L2 so a gross tiling bug
    // (NaN / wrong index / ~50% error) is still caught. Tightening this to argmax
    // parity is blocked on fixing the pre-existing wgpu k-quant prefill divergence.
    assert!(gpu_l.iter().all(|v| v.is_finite()));
    assert!(rel < 0.05, "real Q4_K_M prefill rel-L2 {rel} — tiling regression?");
}

/// Timing bench for the wgpu batched prefill on the real Q4_K_M model (Q4_K + Q6_K
/// shaders). Ignored by default; run with:
///   cargo test --release --features gpu --test gpu_parity -- --ignored --nocapture bench_prefill_gpu_real
/// A/B the tiling with `git stash push src/backend/gpu.rs`.
#[test]
#[ignore = "timing benchmark; needs the GGUF + a GPU"]
fn bench_prefill_gpu_real() {
    let path = std::env::var("RUSTY_LLAMA_GGUF")
        .unwrap_or_else(|_| "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".into());
    if !std::path::Path::new(&path).exists() {
        eprintln!("skip: {path} not found");
        return;
    }
    let Some(g) = gpu() else { return };
    let bytes = std::fs::read(&path).unwrap();
    let gguf = Gguf::parse(&bytes).unwrap();
    let model = Model::from_gguf(&gguf).unwrap();
    let n = 512usize;
    let tokens: Vec<usize> = (0..n).map(|i| (i * 7 + 3) % model.config.vocab_size).collect();
    let mut s = RunState::new(&model.config);
    // warmup (uploads + caches weights on device), then a timed prefill.
    forward_prefill(&model, &mut s, &g, &tokens, 0);
    let t = std::time::Instant::now();
    forward_prefill(&model, &mut s, &g, &tokens, 0);
    let dt = t.elapsed();
    eprintln!(
        "wgpu prefill pp{n}: {:?} -> {:.0} tok/s",
        dt,
        n as f64 / dt.as_secs_f64()
    );
}

/// The fused on-device decode step (with the prefill→decode KV sync) must track
/// the CPU forward and pick the same next token, for F32 and Q8_0.
#[test]
fn decode_step_matches_cpu() {
    let c = cfg();
    for ty in [GgmlType::F32, GgmlType::Q8_0] {
        // int8 decode has its own (looser) coherence tests; this one holds the
        // fused dequant path to exact CPU parity, so force the int8 path off.
        let Some(mut g) = gpu() else { return };
        g.set_int8_decode(false);
        let bytes = synthetic_gguf_typed(&c, ty);
        let gguf = Gguf::parse(&bytes).unwrap();
        let model = Model::from_gguf(&gguf).unwrap();
        let prompt: Vec<usize> = vec![3, 8, 1, 5, 9];
        let next = 7usize;

        // Batched prefill, then one fused decode step at the next position —
        // this exercises the one-time host→device KV-cache sync.
        let logits = |backend: &dyn Backend| -> Vec<f32> {
            let mut s = RunState::new(&model.config);
            forward_prefill(&model, &mut s, backend, &prompt, 0);
            backend.forward_step(&model, &mut s, next, prompt.len());
            s.logits().to_vec()
        };
        let cpu = logits(&CpuBackend::new());
        let gpu_l = logits(&g);

        assert!(gpu_l.iter().all(|v| v.is_finite()));
        let max_diff = cpu
            .iter()
            .zip(&gpu_l)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-2, "{ty:?}: decode logits diverge by {max_diff}");
        assert_eq!(argmax(&cpu), argmax(&gpu_l), "{ty:?}: decode greedy token differs");
    }
}

/// Multi-step greedy decoding must produce a byte-identical stream on GPU and
/// CPU (greedy is argmax, so matching logits => matching tokens).
#[test]
fn greedy_stream_matches_cpu() {
    let c = cfg();
    for ty in [GgmlType::F32, GgmlType::Q8_0] {
        // Fresh backend per model: the weight cache keys on data pointers.
        // int8 greedy decode has its own coherence test; this exact-parity test
        // pins the fused dequant path, so force the int8 path off.
        let Some(mut g) = gpu() else { return };
        g.set_int8_decode(false);
        let bytes = synthetic_gguf_typed(&c, ty);
        let gguf = Gguf::parse(&bytes).unwrap();
        let model = Model::from_gguf(&gguf).unwrap();
        let tk = Tokenizer::from_gguf(&gguf).unwrap();

        let run = |backend: &dyn Backend| -> Vec<u8> {
            let mut st = RunState::new(&model.config);
            let mut sm = SamplerChain::new(c.vocab_size, 0.0, 0.9, 1); // greedy
            let mut out = Vec::new();
            generate(&model, &mut st, backend, &tk, &mut sm, "", 12, |b| {
                out.extend_from_slice(b)
            });
            out
        };

        assert_eq!(run(&CpuBackend::new()), run(&g), "{ty:?}: greedy stream differs");
    }
}

/// The int8 (DP4A) fused decode stays COHERENT with the exact CPU forward: same
/// greedy next token, logits within a loose bound. int8 activation quantization
/// is lossy, so this is deliberately not exact parity — the gpu module's per-op
/// `matmul_q8_0_i8_*` tests pin the kernel against the exact integer dot.
#[test]
fn decode_step_int8_coherent_with_cpu() {
    let c = cfg();
    let Some(mut g) = gpu() else { return };
    g.set_int8_decode(true); // the synthetic model is Q8_0-eligible
    let bytes = synthetic_gguf_typed(&c, GgmlType::Q8_0);
    let gguf = Gguf::parse(&bytes).unwrap();
    let model = Model::from_gguf(&gguf).unwrap();
    let prompt: Vec<usize> = vec![3, 8, 1, 5, 9];
    let next = 7usize;

    // Prefill is the (exact) batched dequant path for both backends; only the
    // single fused decode step differs — so this isolates the int8 decode error.
    let decode = |backend: &dyn Backend| -> Vec<f32> {
        let mut s = RunState::new(&model.config);
        forward_prefill(&model, &mut s, backend, &prompt, 0);
        backend.forward_step(&model, &mut s, next, prompt.len());
        s.logits().to_vec()
    };
    let cpu = decode(&CpuBackend::new());
    let int8 = decode(&g);

    assert!(int8.iter().all(|v| v.is_finite()), "int8 logits must be finite");
    let max_diff = cpu
        .iter()
        .zip(&int8)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("[int8] decode logits maxdiff vs cpu = {max_diff}");
    // Loose, documents the scale: int8 activation quant perturbs the logits a
    // little (observed ~2e-4 on this tiny model; the prototype saw ~0.07 on
    // TinyLlama-1.1B). A wiring regression (bad scale/barrier) would diverge by
    // O(1), so this bound cleanly separates "coherent" from "broken".
    assert!(max_diff < 0.05, "int8 decode diverges from CPU by {max_diff}");
    assert_eq!(argmax(&cpu), argmax(&int8), "int8 decode greedy token differs");
}
