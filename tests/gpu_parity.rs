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
    // The wgpu prefill legitimately differs from the CPU on real k-quant models by a
    // few % (greedy token can differ): the GPU computes exact f32 dequant dots while
    // the CPU uses a Q8_K per-256-block int8 activation that loses ~0.3%/matmul,
    // compounding over depth (see R5.3 + `wgpu_kquant_matmul_matches_f32`, which proves
    // the GPU side is exact). So this asserts the strong, backend-agnostic facts —
    // finite, and bounded rel-L2 (a gross tiling/index bug would blow past it) — not
    // CPU argmax parity (which they don't share, and which isn't a GPU defect).
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

/// R5.3 quality probe: does the CPU's int8 (Q8_K per-256-block) activation actually
/// cost output quality vs the f32-exact GPU reference, or are the ~3.5% logit
/// divergences just near-tie flips? Measures perplexity (lower = better predicts real
/// text) and the argmax-flip rate on real English. Gated on the GGUF + a GPU.
#[test]
#[ignore = "R5.3 quality probe; needs GGUF + GPU"]
fn cpu_int8_quality_vs_f32() {
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
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let text = "The history of artificial intelligence began in antiquity, with myths and \
        stories of artificial beings endowed with intelligence by master craftsmen. Modern \
        machine learning lets computers improve at tasks through experience rather than \
        following explicit rules written by a programmer.";
    let tokens = tok.encode(text, true, false);
    let n = tokens.len();
    assert!(n >= 16, "need enough tokens, got {n}");

    // Mean negative log-likelihood of the actual next token (perplexity = exp(NLL))
    // + the per-position argmax, from a token-at-a-time forward.
    let run = |backend: &dyn Backend| -> (f64, Vec<usize>) {
        let mut s = RunState::new(&model.config);
        let mut nll = 0.0f64;
        let mut cnt = 0usize;
        let mut am = Vec::with_capacity(n);
        for pos in 0..n {
            forward(&model, &mut s, backend, tokens[pos], pos);
            let logits = s.logits();
            am.push(argmax(logits));
            if pos + 1 < n {
                let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let denom: f64 = logits.iter().map(|&l| ((l - mx) as f64).exp()).sum();
                let logp = ((logits[tokens[pos + 1]] - mx) as f64) - denom.ln();
                nll += -logp;
                cnt += 1;
            }
        }
        (nll / cnt as f64, am)
    };
    let (cpu_nll, cpu_am) = run(&CpuBackend::new());
    let (gpu_nll, gpu_am) = run(&g);
    let flips = cpu_am.iter().zip(&gpu_am).filter(|(a, b)| a != b).count();
    eprintln!(
        "n={n} | CPU(int8) ppl={:.3} (nll {:.4}) | GPU(f32) ppl={:.3} (nll {:.4})",
        cpu_nll.exp(),
        cpu_nll,
        gpu_nll.exp(),
        gpu_nll
    );
    eprintln!(
        "argmax CPU-vs-GPU differs at {}/{} positions ({:.1}%)",
        flips,
        n,
        100.0 * flips as f64 / n as f64
    );
    // R5.3 resolution: the int8 path is quality-neutral — perplexity matches the f32
    // reference within noise (measured ~0.2%). Guard against a future int8-accuracy
    // regression: CPU NLL must stay within 5% of the f32 NLL. (Greedy tokens can still
    // differ on near-ties — that's expected and not a quality loss.)
    let rel = (cpu_nll - gpu_nll).abs() / gpu_nll;
    assert!(
        rel < 0.05,
        "CPU int8 perplexity drifted {rel:.3} from f32 — int8 accuracy regression?"
    );
}

/// Proves the wgpu **Q4_K/Q6_K matmul is numerically exact**: it matches the true f32
/// dequant dot (`dequant_row` + f32 dot — the same dequant the shader does) to f32
/// precision. This is the R5.3 resolution — the GPU is correct; the GPU-vs-CPU
/// prefill divergence on real k-quant models is the CPU's Q8_K (per-256-block)
/// int8-activation approximation (~0.3% per matmul, compounding over depth to flip
/// the greedy token), NOT a GPU bug. Gated on the GGUF + a GPU.
#[test]
#[ignore = "needs the GGUF + a GPU"]
fn wgpu_kquant_matmul_matches_f32() {
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
    let cpu = CpuBackend::new();
    let rel = |a: &[f32], b: &[f32]| {
        let n: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let d: f32 = a.iter().map(|x| x * x).sum::<f32>().max(1e-12);
        (n / d).sqrt()
    };
    for w in [&model.weights.wq[0], &model.weights.w2[0]] {
        let (oc, ic) = (w.rows(), w.cols());
        let x: Vec<f32> = (0..ic).map(|i| ((i * 7 % 17) as f32 - 8.0) * 0.05).collect();
        let mut og = vec![0.0f32; oc];
        g.matmul(&mut og, &x, w);
        let mut oc_cpu = vec![0.0f32; oc];
        cpu.matmul(&mut oc_cpu, &x, w);
        let mut wf = vec![0.0f32; ic];
        let oref: Vec<f32> = (0..oc)
            .map(|j| {
                w.dequant_row(j, &mut wf);
                wf.iter().zip(&x).map(|(a, b)| a * b).sum()
            })
            .collect();
        let g_err = rel(&oref, &og);
        let c_err = rel(&oref, &oc_cpu);
        eprintln!("matmul {oc}x{ic}: gpu-vs-f32ref={g_err:.5}  cpu-vs-f32ref={c_err:.5}");
        // The GPU dequant-in-shader matmul must equal the f32 reference (only the
        // parallel-reduction order differs). The CPU's int8 path legitimately differs
        // more (Q8_K activation) — we don't bound it, just record it.
        assert!(
            g_err < 1e-3,
            "wgpu k-quant matmul should match the f32 reference, got rel-L2 {g_err}"
        );
    }
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
