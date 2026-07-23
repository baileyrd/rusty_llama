//! `rusty_llama` command-line interface.
//!
//! ```text
//! rusty_llama <checkpoint.bin> [options]
//!   -z <path>   tokenizer file              (default: tokenizer.bin)
//!   -t <float>  temperature, 0 = greedy     (default: 1.0)
//!   -p <float>  top-p / nucleus sampling    (default: 0.9)
//!   -s <int>    RNG seed                    (default: wall-clock time)
//!   -n <int>    number of steps to run      (default: 256)
//!   -i <text>   prompt                      (default: "")
//! ```

use std::error::Error;
use std::io::{self, Write};
use std::process::ExitCode;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rusty_llama::{
    forward_embed, generate_tokens, AdapterBackend, Backend, ChatTemplate, Checkpoint, ControlVector,
    CpuBackend, Gguf, GrammarStage, LoraAdapter, Message, Model, Pooling, Role, RunState,
    SamplerChain, SamplerConfig, Tokenizer,
};

const USAGE: &str = "\
Usage: rusty_llama <checkpoint.bin> [options]

Options:
  -z <path>     tokenizer file            (default: tokenizer.bin)
  -t <float>    temperature, 0 = greedy   (default: 1.0)
  -p <float>    top-p / nucleus sampling  (default: 0.9)
  -s <int>      RNG seed                  (default: wall-clock time)
  -n <int>      number of steps to run    (default: 256)
  -i <text>     prompt                    (default: \"\")
  --backend <b> compute backend: cpu|gpu|cuda  (default: cpu; gpu/cuda need their cargo feature)
  --top-k <int>           top-k sampling, 0 = off   (default: 0)
  --min-p <float>         min-p sampling, 0 = off   (default: 0.0)
  --repeat-penalty <f>    repetition penalty, 1=off (default: 1.0)
  --repeat-last-n <int>   penalty window            (default: 64)
  --frequency-penalty <f> frequency penalty         (default: 0.0)
  --presence-penalty <f>  presence penalty          (default: 0.0)
  --mirostat <int>        mirostat mode: 0|1|2, off/v1/v2 (default: 0)
  --mirostat-lr <float>   mirostat learning rate (eta)   (default: 0.1)
  --mirostat-ent <float>  mirostat target entropy (tau)  (default: 5.0)
  --xtc-probability <f>   XTC probability, 0 = off        (default: 0.0)
  --xtc-threshold <f>     XTC probability threshold       (default: 0.1)
  --dry-multiplier <f>    DRY penalty multiplier, 0 = off  (default: 0.0)
  --dry-base <f>          DRY penalty exponential base     (default: 1.75)
  --dry-allowed-length <n> DRY: shortest penalized repeat  (default: 2)
  --dry-penalty-last-n <n> DRY match window (tokens)       (default: 64)
  --dry-sequence-breaker <text>  DRY window restart token (repeatable, single-token only)
  --embedding             output an embedding vector instead of text
  --pooling <mode>        embedding pooling: mean|last|cls (default: mean)
  --embd-normalize <n>    embedding norm: 2 = L2, -1 = none     (default: 2)
  --chat                  treat -i as one user turn via the chat template
  --system <text>         system message for --chat
  --chat-template <name>  chatml|llama3|qwen2|gemma|phi3 (default: auto-detect)
  --lora <path>           apply a LoRA adapter GGUF
  --lora-scale <float>    LoRA strength multiplier         (default: 1.0)
  --control-vector <path> apply a control-vector GGUF
  --control-vector-scale <float>  control-vector strength      (default: 1.0)
  --serve                 run an OpenAI-compatible HTTP server (needs --features server)
  --host <addr>           server bind address       (default: 127.0.0.1)
  --port <int>            server port               (default: 8080)
  --grammar <gbnf>        constrain output to a GBNF grammar (inline)
  --grammar-file <path>   constrain output to a GBNF grammar (from a file)
  -h            show this help";

struct Args {
    checkpoint: String,
    tokenizer: String,
    temperature: f32,
    topp: f32,
    seed: u64,
    steps: usize,
    prompt: String,
    backend: String,
    top_k: usize,
    min_p: f32,
    repeat_penalty: f32,
    repeat_last_n: usize,
    frequency_penalty: f32,
    presence_penalty: f32,
    mirostat: u8,
    mirostat_lr: f32,
    mirostat_ent: f32,
    xtc_probability: f32,
    xtc_threshold: f32,
    dry_multiplier: f32,
    dry_base: f32,
    dry_allowed_length: usize,
    dry_penalty_last_n: usize,
    dry_sequence_breakers: Vec<String>,
    embedding: bool,
    pooling: String,
    embd_normalize: i32,
    chat: bool,
    system: String,
    chat_template: String,
    lora: String,
    lora_scale: f32,
    control_vector: String,
    control_vector_scale: f32,
    serve: bool,
    host: String,
    port: u16,
    grammar: String,
    grammar_file: String,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    if args.serve {
        #[cfg(feature = "server")]
        {
            let ct = (!args.chat_template.is_empty()).then_some(args.chat_template.as_str());
            return rusty_llama::serve(&args.checkpoint, &args.backend, &args.host, args.port, ct);
        }
        #[cfg(not(feature = "server"))]
        {
            return Err(format!(
                "built without server support (would serve {}:{}); rebuild with `--features server`",
                args.host, args.port
            )
            .into());
        }
    }
    let checkpoint = Checkpoint::open(&args.checkpoint)
        .map_err(|e| format!("failed to open checkpoint '{}': {e}", args.checkpoint))?;

    // GGUF files carry their own tokenizer; llama2.c checkpoints need -z.
    if Gguf::is_gguf(checkpoint.bytes()) {
        let gguf = checkpoint.gguf()?;
        let model = Model::from_gguf(&gguf)?;
        let tokenizer = Tokenizer::from_gguf(&gguf)
            .map_err(|e| format!("failed to read GGUF tokenizer: {e}"))?;
        let template = resolve_chat_template(&args, Some(&gguf))?;
        run_loaded(&model, &tokenizer, &args, template)
    } else {
        let model = checkpoint.model()?;
        let tokenizer = Tokenizer::load(&args.tokenizer, model.config.vocab_size)
            .map_err(|e| format!("failed to load tokenizer '{}': {e}", args.tokenizer))?;
        let template = resolve_chat_template(&args, None)?;
        run_loaded(&model, &tokenizer, &args, template)
    }
}

/// Dispatch a loaded model to the selected mode.
fn run_loaded(
    model: &Model,
    tokenizer: &Tokenizer,
    args: &Args,
    template: Option<ChatTemplate>,
) -> Result<(), Box<dyn Error>> {
    if args.embedding {
        run_embedding(model, tokenizer, args)
    } else if args.chat {
        run_chat(model, tokenizer, args, template)
    } else {
        run_generation(model, tokenizer, args)
    }
}

/// Resolve the chat template for `--chat`: explicit `--chat-template` wins, else
/// detect from the GGUF / arch. `Ok(None)` when not in chat mode.
fn resolve_chat_template(
    args: &Args,
    gguf: Option<&Gguf>,
) -> Result<Option<ChatTemplate>, Box<dyn Error>> {
    if !args.chat {
        return Ok(None);
    }
    if !args.chat_template.is_empty() {
        return ChatTemplate::from_name(&args.chat_template).map(Some).ok_or_else(|| {
            format!("unknown --chat-template '{}'", args.chat_template).into()
        });
    }
    if let Some(g) = gguf {
        let arch = g.meta_str("general.architecture").unwrap_or_default();
        if let Some(t) = ChatTemplate::detect(g, arch) {
            return Ok(Some(t));
        }
    }
    Err("could not detect a chat template; pass --chat-template (chatml|llama3|qwen2|gemma|phi3)".into())
}

/// The GBNF grammar source for constrained decoding: `--grammar-file` (read from
/// disk) wins, else inline `--grammar`, else `None`.
fn resolve_grammar(args: &Args) -> Result<Option<String>, Box<dyn Error>> {
    if !args.grammar_file.is_empty() {
        return std::fs::read_to_string(&args.grammar_file).map(Some).map_err(|e| {
            format!("failed to read grammar file '{}': {e}", args.grammar_file).into()
        });
    }
    if !args.grammar.is_empty() {
        return Ok(Some(args.grammar.clone()));
    }
    Ok(None)
}

fn run_embedding(model: &Model, tokenizer: &Tokenizer, args: &Args) -> Result<(), Box<dyn Error>> {
    let backend = make_backend(&args.backend)?;
    let mut state = RunState::new(&model.config);
    let tokens = tokenizer.encode(&args.prompt, tokenizer.add_bos(), false);
    if tokens.is_empty() {
        return Err("embedding mode needs a non-empty prompt (-i)".into());
    }
    if tokens.len() > model.config.seq_len {
        return Err(format!(
            "prompt of {} tokens exceeds context length {}",
            tokens.len(),
            model.config.seq_len
        )
        .into());
    }
    let pooling = match args.pooling.as_str() {
        "mean" => Pooling::Mean,
        "last" => Pooling::Last,
        "cls" => Pooling::Cls,
        other => return Err(format!("unknown --pooling '{other}' (mean|last|cls)").into()),
    };
    let emb = forward_embed(
        model,
        &mut state,
        backend.as_ref(),
        &tokens,
        pooling,
        args.embd_normalize != -1,
    );
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for v in &emb {
        writeln!(out, "{v}")?;
    }
    Ok(())
}

fn run_generation(model: &Model, tokenizer: &Tokenizer, args: &Args) -> Result<(), Box<dyn Error>> {
    let tokens = tokenizer.encode(&args.prompt, tokenizer.add_bos(), false);
    stream_generation(model, tokenizer, args, &tokens)
}

fn run_chat(
    model: &Model,
    tokenizer: &Tokenizer,
    args: &Args,
    template: Option<ChatTemplate>,
) -> Result<(), Box<dyn Error>> {
    let template = template.ok_or("chat mode needs a template (see --chat-template)")?;
    let mut msgs = Vec::new();
    if !args.system.is_empty() {
        msgs.push(Message {
            role: Role::System,
            content: args.system.clone(),
        });
    }
    msgs.push(Message {
        role: Role::User,
        content: args.prompt.clone(),
    });
    let prompt = template.render(&msgs, true);
    // Prepend BOS only if the model wants one (add_bos_token) and the template
    // doesn't already emit its own (llama-3's <|begin_of_text|>).
    let tokens = tokenizer.encode(&prompt, tokenizer.add_bos() && !template.emits_bos(), false);
    stream_generation(model, tokenizer, args, &tokens)
}

/// Shared decode loop: build the sampler, stream `generate_tokens`, report timing.
fn stream_generation(
    model: &Model,
    tokenizer: &Tokenizer,
    args: &Args,
    prompt_tokens: &[usize],
) -> Result<(), Box<dyn Error>> {
    if prompt_tokens.is_empty() {
        return Err("nothing to generate from (empty prompt)".into());
    }
    let backend = make_backend(&args.backend)?;

    // Optional adapters: LoRA + control vector, each from its own GGUF, matched
    // against this model. Binding either falls back to the per-op path (adapters
    // apply on every backend; the fused resident decode is bypassed).
    let lora_cp = if args.lora.is_empty() {
        None
    } else {
        Some(
            Checkpoint::open(&args.lora)
                .map_err(|e| format!("failed to open LoRA '{}': {e}", args.lora))?,
        )
    };
    let lora_gguf = match &lora_cp {
        Some(cp) => Some(cp.gguf()?),
        None => None,
    };
    let lora = match &lora_gguf {
        Some(g) => Some(LoraAdapter::from_gguf(g, model, args.lora_scale)?),
        None => None,
    };
    let cv_cp = if args.control_vector.is_empty() {
        None
    } else {
        Some(
            Checkpoint::open(&args.control_vector)
                .map_err(|e| format!("failed to open control vector '{}': {e}", args.control_vector))?,
        )
    };
    let cv_gguf = match &cv_cp {
        Some(cp) => Some(cp.gguf()?),
        None => None,
    };
    let control = match &cv_gguf {
        Some(g) => Some(ControlVector::from_gguf(g, model, args.control_vector_scale)?),
        None => None,
    };
    let adapter_backend = if lora.is_some() || control.is_some() {
        Some(AdapterBackend::new(
            backend.as_ref(),
            lora.as_ref(),
            control.as_ref(),
        ))
    } else {
        None
    };
    let active: &dyn Backend = match &adapter_backend {
        Some(ab) => ab,
        None => backend.as_ref(),
    };

    let mut state = RunState::new(&model.config);
    let mut dry_sequence_breakers = std::collections::HashSet::new();
    for text in &args.dry_sequence_breakers {
        let toks = tokenizer.encode(text, false, false);
        match toks.as_slice() {
            [t] => {
                dry_sequence_breakers.insert(*t as u32);
            }
            _ => eprintln!(
                "warning: --dry-sequence-breaker '{text}' tokenizes to {} tokens, \
                 not 1 — only single-token breakers are supported, skipping",
                toks.len()
            ),
        }
    }
    let sampler_cfg = SamplerConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.topp,
        min_p: args.min_p,
        min_keep: 1,
        repeat_penalty: args.repeat_penalty,
        repeat_last_n: args.repeat_last_n,
        frequency_penalty: args.frequency_penalty,
        presence_penalty: args.presence_penalty,
        seed: args.seed,
        mirostat: args.mirostat,
        mirostat_tau: args.mirostat_ent,
        mirostat_eta: args.mirostat_lr,
        mirostat_m: 100,
        xtc_probability: args.xtc_probability,
        xtc_threshold: args.xtc_threshold,
        dry_multiplier: args.dry_multiplier,
        dry_base: args.dry_base,
        dry_allowed_length: args.dry_allowed_length,
        dry_penalty_last_n: args.dry_penalty_last_n,
        dry_sequence_breakers,
    };
    let mut sampler = SamplerChain::from_config(&sampler_cfg, model.config.vocab_size);
    if let Some(src) = resolve_grammar(args)? {
        let stage = GrammarStage::from_source(&src, tokenizer)
            .map_err(|e| format!("grammar error: {e}"))?;
        sampler.prepend(Box::new(stage));
    }

    let stdout = io::stdout();
    let mut out = stdout.lock();
    let start = Instant::now();
    let generated = generate_tokens(
        model,
        &mut state,
        active,
        tokenizer,
        &mut sampler,
        prompt_tokens,
        args.steps,
        |bytes| {
            let _ = out.write_all(bytes);
            let _ = out.flush();
        },
    );
    let _ = writeln!(out);
    let secs = start.elapsed().as_secs_f64();
    if generated > 0 && secs > 0.0 {
        eprintln!("\n[{generated} tokens, {:.1} tok/s]", generated as f64 / secs);
    }
    Ok(())
}

/// Construct the compute backend named on the command line.
///
/// `gpu` requires the `gpu` cargo feature; without it (or without a usable
/// adapter) this returns a clear error rather than silently falling back.
fn make_backend(name: &str) -> Result<Box<dyn Backend>, Box<dyn Error>> {
    match name {
        "cpu" => Ok(Box::new(CpuBackend::new())),
        "gpu" => {
            #[cfg(feature = "gpu")]
            {
                let gpu = rusty_llama::GpuBackend::new()
                    .map_err(|e| format!("GPU backend unavailable: {e}"))?;
                eprintln!("[gpu] adapter: {}", gpu.adapter_name());
                Ok(Box::new(gpu))
            }
            #[cfg(not(feature = "gpu"))]
            {
                Err("this binary was built without GPU support; rebuild with \
                     `cargo build --release --features gpu`"
                    .into())
            }
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                let cuda = rusty_llama::CudaBackend::new()
                    .map_err(|e| format!("CUDA backend unavailable: {e}"))?;
                eprintln!("[cuda] device: {}", cuda.device_name());
                Ok(Box::new(cuda))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err("this binary was built without CUDA support; rebuild with \
                     `cargo build --release --features cuda`"
                    .into())
            }
        }
        other => Err(format!("unknown backend '{other}' (have cpu/gpu/cuda)").into()),
    }
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut raw = std::env::args().skip(1);
    let checkpoint = match raw.next() {
        Some(a) if a == "-h" || a == "--help" => {
            println!("{USAGE}");
            std::process::exit(0);
        }
        Some(a) => a,
        None => {
            return Err(format!("missing checkpoint path\n\n{USAGE}").into());
        }
    };

    let mut args = Args {
        checkpoint,
        tokenizer: "tokenizer.bin".to_string(),
        temperature: 1.0,
        topp: 0.9,
        seed: default_seed(),
        steps: 256,
        prompt: String::new(),
        backend: "cpu".to_string(),
        top_k: 0,
        min_p: 0.0,
        repeat_penalty: 1.0,
        repeat_last_n: 64,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        mirostat: 0,
        mirostat_lr: 0.1,
        mirostat_ent: 5.0,
        xtc_probability: 0.0,
        xtc_threshold: 0.1,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_penalty_last_n: 64,
        dry_sequence_breakers: Vec::new(),
        embedding: false,
        pooling: "mean".to_string(),
        embd_normalize: 2,
        chat: false,
        system: String::new(),
        chat_template: String::new(),
        lora: String::new(),
        lora_scale: 1.0,
        control_vector: String::new(),
        control_vector_scale: 1.0,
        serve: false,
        host: "127.0.0.1".to_string(),
        port: 8080,
        grammar: String::new(),
        grammar_file: String::new(),
    };

    let rest: Vec<String> = raw.collect();
    let mut i = 0;
    while i < rest.len() {
        let flag = rest[i].as_str();
        if flag == "--embedding" {
            args.embedding = true;
            i += 1;
            continue;
        }
        if flag == "--chat" {
            args.chat = true;
            i += 1;
            continue;
        }
        if flag == "--serve" {
            args.serve = true;
            i += 1;
            continue;
        }
        let value = || -> Result<&String, Box<dyn Error>> {
            rest.get(i + 1)
                .ok_or_else(|| format!("missing value for '{flag}'").into())
        };
        match flag {
            "-z" => args.tokenizer = value()?.clone(),
            "-t" => args.temperature = value()?.parse()?,
            "-p" => args.topp = value()?.parse()?,
            "-s" => args.seed = value()?.parse()?,
            "-n" => args.steps = value()?.parse()?,
            "-i" => args.prompt = value()?.clone(),
            "--backend" => args.backend = value()?.clone(),
            "--top-k" => args.top_k = value()?.parse()?,
            "--min-p" => args.min_p = value()?.parse()?,
            "--repeat-penalty" => args.repeat_penalty = value()?.parse()?,
            "--repeat-last-n" => args.repeat_last_n = value()?.parse()?,
            "--frequency-penalty" => args.frequency_penalty = value()?.parse()?,
            "--presence-penalty" => args.presence_penalty = value()?.parse()?,
            "--mirostat" => args.mirostat = value()?.parse()?,
            "--mirostat-lr" => args.mirostat_lr = value()?.parse()?,
            "--mirostat-ent" => args.mirostat_ent = value()?.parse()?,
            "--xtc-probability" => args.xtc_probability = value()?.parse()?,
            "--xtc-threshold" => args.xtc_threshold = value()?.parse()?,
            "--dry-multiplier" => args.dry_multiplier = value()?.parse()?,
            "--dry-base" => args.dry_base = value()?.parse()?,
            "--dry-allowed-length" => args.dry_allowed_length = value()?.parse()?,
            "--dry-penalty-last-n" => args.dry_penalty_last_n = value()?.parse()?,
            "--dry-sequence-breaker" => args.dry_sequence_breakers.push(value()?.clone()),
            "--pooling" => args.pooling = value()?.clone(),
            "--embd-normalize" => args.embd_normalize = value()?.parse()?,
            "--system" => args.system = value()?.clone(),
            "--chat-template" => args.chat_template = value()?.clone(),
            "--lora" => args.lora = value()?.clone(),
            "--lora-scale" => args.lora_scale = value()?.parse()?,
            "--control-vector" => args.control_vector = value()?.clone(),
            "--control-vector-scale" => args.control_vector_scale = value()?.parse()?,
            "--host" => args.host = value()?.clone(),
            "--port" => args.port = value()?.parse()?,
            "--grammar" => args.grammar = value()?.clone(),
            "--grammar-file" => args.grammar_file = value()?.clone(),
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown option '{other}'\n\n{USAGE}").into()),
        }
        i += 2;
    }
    Ok(args)
}

fn default_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1)
}
