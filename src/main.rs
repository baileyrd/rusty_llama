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
    generate, Backend, Checkpoint, CpuBackend, Gguf, Model, RunState, Sampler, Tokenizer,
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
  --backend <b> compute backend: cpu|gpu  (default: cpu; gpu needs --features gpu)
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

    let checkpoint = Checkpoint::open(&args.checkpoint)
        .map_err(|e| format!("failed to open checkpoint '{}': {e}", args.checkpoint))?;

    // GGUF files carry their own tokenizer; llama2.c checkpoints need -z.
    if Gguf::is_gguf(checkpoint.bytes()) {
        let gguf = Gguf::parse(checkpoint.bytes())?;
        let model = Model::from_gguf(&gguf)?;
        let tokenizer = Tokenizer::from_gguf(&gguf)
            .map_err(|e| format!("failed to read GGUF tokenizer: {e}"))?;
        run_generation(&model, &tokenizer, &args)
    } else {
        let model = checkpoint.model()?;
        let tokenizer = Tokenizer::load(&args.tokenizer, model.config.vocab_size)
            .map_err(|e| format!("failed to load tokenizer '{}': {e}", args.tokenizer))?;
        run_generation(&model, &tokenizer, &args)
    }
}

fn run_generation(model: &Model, tokenizer: &Tokenizer, args: &Args) -> Result<(), Box<dyn Error>> {
    let backend = make_backend(&args.backend)?;
    let mut state = RunState::new(&model.config);
    let mut sampler = Sampler::new(
        model.config.vocab_size,
        args.temperature,
        args.topp,
        args.seed,
    );

    let stdout = io::stdout();
    let mut out = stdout.lock();
    let start = Instant::now();
    let generated = generate(
        model,
        &mut state,
        backend.as_ref(),
        tokenizer,
        &mut sampler,
        &args.prompt,
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
        other => Err(format!("unknown backend '{other}' (have cpu/gpu)").into()),
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
    };

    let rest: Vec<String> = raw.collect();
    let mut i = 0;
    while i < rest.len() {
        let flag = rest[i].as_str();
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
