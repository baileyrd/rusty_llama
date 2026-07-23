//! Minimal OpenAI-compatible HTTP server (the `serve` subcommand), behind the
//! `server` cargo feature.
//!
//! The HTTP/1.1 layer is hand-rolled on [`std::net`] (no HTTP dependency); JSON
//! is serde. A single **generation worker thread** owns the model + backend +
//! tokenizer — so neither needs to be `Sync`, and the self-referential
//! `Model`↔`Gguf`↔`Checkpoint` borrow chain stays on one stack — and is fed
//! [`Job`]s over a channel by per-connection threads. Concurrent clients are
//! accepted in parallel but generation is serialized (one job at a time);
//! continuous batching is the Phase 4.2 follow-on.
//!
//! Endpoints: `POST /v1/chat/completions`, `POST /v1/completions`,
//! `GET /v1/models`, `GET /health`. Streaming (`"stream": true`) emits
//! Server-Sent Events; otherwise a single JSON body. Works with any backend.

use std::collections::VecDeque;
use std::error::Error;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::backend::make_backend;
use crate::chat::{ChatTemplate, Message, Role};
use crate::gguf::Gguf;
use crate::loader::Checkpoint;
use crate::model::{forward_prefill, Batch, Model, RunState};
use crate::grammar::{Grammar, GrammarStage, JSON_GRAMMAR};
use crate::sampler::{SamplerChain, SamplerConfig};
use crate::tokenizer::Tokenizer;

// ---------------------------------------------------------------------------
// Request / response wire types (a pragmatic subset of the OpenAI API).
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ApiMessage {
    role: String,
    #[serde(default)]
    content: String,
}

/// Sampling knobs shared by both endpoints (top-level OpenAI fields, flattened).
#[derive(Deserialize, Default)]
struct SamplingParams {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    max_tokens: Option<usize>,
    seed: Option<u64>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    mirostat: Option<u8>,
    mirostat_tau: Option<f32>,
    mirostat_eta: Option<f32>,
    xtc_probability: Option<f32>,
    xtc_threshold: Option<f32>,
}

/// OpenAI `response_format` — `{"type":"json_object"}` constrains output to JSON.
#[derive(Deserialize)]
struct ResponseFormat {
    #[serde(default, rename = "type")]
    kind: String,
}

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: String,
    messages: Vec<ApiMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(flatten)]
    params: SamplingParams,
    #[serde(default)]
    grammar: Option<String>,
    #[serde(default)]
    response_format: Option<ResponseFormat>,
}

#[derive(Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    stream: bool,
    #[serde(flatten)]
    params: SamplingParams,
    #[serde(default)]
    grammar: Option<String>,
    #[serde(default)]
    response_format: Option<ResponseFormat>,
}

#[derive(Serialize)]
struct RespMessage {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: RespMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct CompletionChoice {
    index: usize,
    text: String,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Serialize)]
struct ChunkChoice {
    index: usize,
    delta: Delta,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct ChatChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
}

// ---------------------------------------------------------------------------
// Worker job protocol.
// ---------------------------------------------------------------------------

enum JobKind {
    Chat(Vec<Message>),
    Completion(String),
}

/// A unit of work for the generation thread + the channel it streams back on.
struct Job {
    kind: JobKind,
    sampler: SamplerConfig,
    max_tokens: usize,
    grammar: Option<String>,
    reply: Sender<Event>,
}

/// One streamed piece of a response. `Token` carries a UTF-8-complete fragment
/// (the worker buffers partial multi-byte chars), `Done` the final token counts.
enum Event {
    Token(String),
    Done { prompt: usize, completion: usize },
    Error(String),
}

// ---------------------------------------------------------------------------
// Server entry point.
// ---------------------------------------------------------------------------

/// Serve `model_path` over HTTP on `host:port` using the named `backend`
/// (`"cpu"`/`"gpu"`/`"cuda"`). `chat_template` overrides auto-detection. Blocks
/// forever (the accept loop); returns only on a bind/load error.
pub fn serve(
    model_path: &str,
    backend: &str,
    host: &str,
    port: u16,
    chat_template: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let (jobs_tx, jobs_rx) = channel::<Job>();
    let (ready_tx, ready_rx) = channel::<Result<(), String>>();

    let (mp, bk, ct) = (
        model_path.to_string(),
        backend.to_string(),
        chat_template.map(str::to_string),
    );
    thread::Builder::new()
        .name("rusty-llama-worker".into())
        .spawn(move || worker(&mp, &bk, ct.as_deref(), jobs_rx, ready_tx))?;

    // Block until the worker has loaded the model (propagating load errors)
    // before we start accepting connections.
    ready_rx.recv().map_err(|_| "worker thread died during startup")??;

    let model_id: Arc<str> = Arc::from(model_label(model_path));
    let listener = TcpListener::bind((host, port))?;
    eprintln!("rusty_llama: serving '{model_id}' on http://{host}:{port} (backend: {backend})");

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("rusty_llama: accept error: {e}");
                continue;
            }
        };
        let tx = jobs_tx.clone();
        let id = model_id.clone();
        thread::spawn(move || {
            if let Err(e) = handle_connection(stream, &tx, &id) {
                // A dropped client mid-stream is normal; just note it.
                eprintln!("rusty_llama: connection closed: {e}");
            }
        });
    }
    Ok(())
}

/// Max concurrent sequences the scheduler batches. Override with
/// `RUSTY_LLAMA_BATCH` (clamped to >= 1).
fn batch_cap() -> usize {
    std::env::var("RUSTY_LLAMA_BATCH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8)
        .max(1)
}

/// An in-flight sequence occupying a decode slot.
struct Slot {
    reply: Sender<Event>,
    sampler: SamplerChain,
    prev: usize,      // previous token (decode renders `cur` as decode(prev, cur))
    cur: usize,       // current token: emit it, then feed it at `pos`
    pos: usize,       // position to feed `cur` at
    emitted: usize,   // generated tokens emitted so far
    max_new: usize,   // cap on `emitted`
    prompt_len: usize,
    pending: Vec<u8>, // UTF-8 carry across tokens
}

impl Slot {
    /// Whether this slot should stop: an end-of-generation token, the token
    /// budget reached, or the KV/context full.
    fn is_done(&self, tokenizer: &Tokenizer, seq_len: usize) -> bool {
        self.cur == 1
            || tokenizer.is_eog(self.cur)
            || self.emitted >= self.max_new
            || self.pos >= seq_len
    }

    /// Emit `cur`'s text (UTF-8-buffered so SSE never splits a code point).
    fn emit(&mut self, tokenizer: &Tokenizer) {
        self.pending
            .extend_from_slice(&tokenizer.decode(self.prev, self.cur));
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(s) => s.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid > 0 {
            let s = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
            let _ = self.reply.send(Event::Token(s));
            self.pending.drain(..valid);
        }
        self.emitted += 1;
    }

    /// Flush any carry and report completion.
    fn finish(self) {
        if !self.pending.is_empty() {
            let _ = self
                .reply
                .send(Event::Token(String::from_utf8_lossy(&self.pending).into_owned()));
        }
        let _ = self.reply.send(Event::Done {
            prompt: self.prompt_len,
            completion: self.emitted,
        });
    }
}

/// The generation thread: loads everything (kept resident on this stack) and
/// runs a **continuous-batching scheduler** — admit queued prompts into free
/// slots (prefill each), then batch-decode all active slots one step at a time,
/// streaming and evicting as sequences finish. Load failures go via `ready`.
fn worker(
    model_path: &str,
    backend_name: &str,
    template_override: Option<&str>,
    jobs: Receiver<Job>,
    ready: Sender<Result<(), String>>,
) {
    macro_rules! bail {
        ($r:expr, $ctx:expr) => {
            match $r {
                Ok(v) => v,
                Err(e) => {
                    let _ = ready.send(Err(format!("{}: {e}", $ctx)));
                    return;
                }
            }
        };
    }

    let cp = bail!(Checkpoint::open(model_path), "open model");
    let gguf = bail!(Gguf::parse(cp.bytes()), "parse GGUF");
    let model = bail!(Model::from_gguf(&gguf), "build model");
    let tokenizer = bail!(Tokenizer::from_gguf(&gguf), "load tokenizer");
    let backend = bail!(make_backend(backend_name), "init backend");
    let template = resolve_template(&gguf, template_override);
    let _ = ready.send(Ok(()));

    let cap = batch_cap();
    let vocab = model.config.vocab_size;
    let seq_len = model.config.seq_len;
    let mut batch = Batch::new(&model.config, cap);
    let mut states: Vec<RunState> = (0..cap).map(|_| RunState::new(&model.config)).collect();
    let mut slots: Vec<Option<Slot>> = (0..cap).map(|_| None).collect();
    let mut queue: VecDeque<Job> = VecDeque::new();
    // End-of-generation ids + a lazily-built token-piece table for grammar
    // constraints (a vocab-sized table, computed once on the first grammar job).
    let eog = tokenizer.eog_ids();
    let mut pieces: Option<Arc<Vec<Vec<u8>>>> = None;

    loop {
        let active_now = slots.iter().filter(|s| s.is_some()).count();
        // Block for work only when fully idle; otherwise drain without blocking.
        if active_now == 0 && queue.is_empty() {
            match jobs.recv() {
                Ok(j) => queue.push_back(j),
                Err(_) => return, // all senders dropped — shut down
            }
        }
        while let Ok(j) = jobs.try_recv() {
            queue.push_back(j);
        }

        // Admit queued prompts into free slots (serial prefill, per-op path so
        // the host KV the batched decode reads is consistent on every backend).
        for i in 0..cap {
            if slots[i].is_none() {
                if let Some(job) = queue.pop_front() {
                    slots[i] = admit(
                        &model,
                        &tokenizer,
                        backend.as_ref(),
                        template,
                        &mut states[i],
                        &mut pieces,
                        &eog,
                        job,
                    );
                }
            }
        }

        // Phase A: emit each live slot's current token, or evict finished slots.
        let mut active: Vec<usize> = Vec::new();
        for (i, slot_opt) in slots.iter_mut().enumerate() {
            if slot_opt.is_none() {
                continue;
            }
            if slot_opt.as_ref().unwrap().is_done(&tokenizer, seq_len) {
                slot_opt.take().unwrap().finish();
            } else {
                slot_opt.as_mut().unwrap().emit(&tokenizer);
                active.push(i);
            }
        }

        // Phase B: one batched decode step over the active slots → next token each.
        if !active.is_empty() {
            let tokens: Vec<usize> = active.iter().map(|&i| slots[i].as_ref().unwrap().cur).collect();
            let positions: Vec<usize> = active.iter().map(|&i| slots[i].as_ref().unwrap().pos).collect();
            let logits = batch.decode_step(&model, backend.as_ref(), &mut states, &active, &tokens, &positions);
            for (r, &i) in active.iter().enumerate() {
                let slot = slots[i].as_mut().unwrap();
                let next = slot.sampler.sample(&logits[r * vocab..(r + 1) * vocab]);
                slot.prev = slot.cur;
                slot.cur = next;
                slot.pos += 1;
            }
        }
    }
}

/// Render + tokenize a job's prompt, prefill it into `state`, and return the
/// occupied [`Slot`] ready to decode — or `None` after sending the client an
/// error. Uses the free `forward_prefill` (per-op path) so the host KV matches
/// what the batched decode reads on every backend.
#[allow(clippy::too_many_arguments)]
fn admit(
    model: &Model,
    tokenizer: &Tokenizer,
    backend: &dyn crate::Backend,
    template: Option<ChatTemplate>,
    state: &mut RunState,
    pieces: &mut Option<Arc<Vec<Vec<u8>>>>,
    eog: &[u32],
    job: Job,
) -> Option<Slot> {
    let prompt = match &job.kind {
        JobKind::Chat(msgs) => {
            let Some(t) = template else {
                let _ = job.reply.send(Event::Error(
                    "this model has no chat template; use /v1/completions or pass --chat-template"
                        .into(),
                ));
                return None;
            };
            let rendered = t.render(msgs, true);
            tokenizer.encode(&rendered, tokenizer.add_bos() && !t.emits_bos(), false)
        }
        JobKind::Completion(text) => tokenizer.encode(text, tokenizer.add_bos(), false),
    };
    if prompt.is_empty() {
        let _ = job.reply.send(Event::Error("empty prompt".into()));
        return None;
    }
    let prompt_len = prompt.len();
    if prompt_len >= model.config.seq_len {
        let _ = job.reply.send(Event::Error(format!(
            "prompt of {prompt_len} tokens exceeds context length {}",
            model.config.seq_len
        )));
        return None;
    }
    forward_prefill(model, state, backend, &prompt, 0);
    let mut sampler = SamplerChain::from_config(&job.sampler, model.config.vocab_size);
    if let Some(src) = job.grammar.as_deref().filter(|s| !s.is_empty()) {
        let grammar = match Grammar::parse(src) {
            Ok(g) => g,
            Err(e) => {
                let _ = job.reply.send(Event::Error(format!("grammar error: {e}")));
                return None;
            }
        };
        // Build the vocab-sized piece table once, then share it across requests.
        let table = pieces.get_or_insert_with(|| {
            Arc::new((0..tokenizer.vocab_size()).map(|i| tokenizer.token_piece(i)).collect())
        });
        sampler.prepend(Box::new(GrammarStage::new(
            grammar,
            table.clone(),
            eog.to_vec(),
            tokenizer.special_ids(),
        )));
    }
    let cur = sampler.sample(state.logits());
    Some(Slot {
        reply: job.reply,
        sampler,
        prev: *prompt.last().unwrap(),
        cur,
        pos: prompt_len,
        emitted: 0,
        max_new: job.max_tokens,
        prompt_len,
        pending: Vec::new(),
    })
}

fn resolve_template(gguf: &Gguf, override_name: Option<&str>) -> Option<ChatTemplate> {
    if let Some(name) = override_name {
        return ChatTemplate::from_name(name);
    }
    let arch = gguf.meta_str("general.architecture").unwrap_or_default();
    ChatTemplate::detect(gguf, arch)
}

// ---------------------------------------------------------------------------
// HTTP/1.1 (hand-rolled) + routing.
// ---------------------------------------------------------------------------

/// Reject request bodies larger than this — a memory-exhaustion guard. 16 MiB is
/// far above any real chat/completion payload.
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
/// Per-connection socket read/write timeout — a slowloris guard.
const SOCKET_TIMEOUT_SECS: u64 = 30;

fn body_within_limit(content_length: usize) -> bool {
    content_length <= MAX_BODY_BYTES
}

fn handle_connection(stream: TcpStream, jobs: &Sender<Job>, model_id: &str) -> io::Result<()> {
    // Bound how long one (stalled or slowloris) client can pin this worker thread:
    // without read/write timeouts a client that opens a socket and never finishes
    // its request would block the thread forever.
    let timeout = Some(std::time::Duration::from_secs(SOCKET_TIMEOUT_SECS));
    stream.set_read_timeout(timeout)?;
    stream.set_write_timeout(timeout)?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut stream = stream;

    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(()); // client hung up
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();

    let mut content_length = 0usize;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 {
            break;
        }
        if header.trim().is_empty() {
            break;
        }
        let lower = header.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    // Cap the body so a single request can't make us allocate an unbounded buffer
    // (a one-line memory-exhaustion DoS): reject oversized bodies with 413 rather
    // than `vec![0u8; client_supplied_len]`.
    if !body_within_limit(content_length) {
        return write_json(&mut stream, 413, &error_json("request body too large"));
    }
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;

    match (method.as_str(), path.as_str()) {
        ("POST", "/v1/chat/completions") => handle_chat(&mut stream, &body, jobs, model_id),
        ("POST", "/v1/completions") => handle_completion(&mut stream, &body, jobs, model_id),
        ("GET", "/v1/models") => write_json(&mut stream, 200, &models_json(model_id)),
        ("GET", "/health") | ("GET", "/") => write_text(&mut stream, 200, "ok\n"),
        _ => write_json(&mut stream, 404, &error_json("not found")),
    }
}

fn handle_chat(
    stream: &mut TcpStream,
    body: &[u8],
    jobs: &Sender<Job>,
    model_id: &str,
) -> io::Result<()> {
    let req: ChatRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => return write_json(stream, 400, &error_json(&format!("bad request: {e}"))),
    };
    let model = pick_model(&req.model, model_id);
    let msgs = req
        .messages
        .iter()
        .map(|m| Message {
            role: parse_role(&m.role),
            content: m.content.clone(),
        })
        .collect();
    let (reply, rx) = channel::<Event>();
    let job = Job {
        kind: JobKind::Chat(msgs),
        sampler: resolve_sampler(&req.params),
        max_tokens: req.params.max_tokens.unwrap_or(256),
        grammar: req_grammar(req.grammar, req.response_format),
        reply,
    };
    if jobs.send(job).is_err() {
        return write_json(stream, 500, &error_json("generation worker is gone"));
    }
    if req.stream {
        stream_chat(stream, &rx, &model)
    } else {
        buffer_chat(stream, &rx, &model)
    }
}

fn handle_completion(
    stream: &mut TcpStream,
    body: &[u8],
    jobs: &Sender<Job>,
    model_id: &str,
) -> io::Result<()> {
    let req: CompletionRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => return write_json(stream, 400, &error_json(&format!("bad request: {e}"))),
    };
    let model = pick_model(&req.model, model_id);
    let (reply, rx) = channel::<Event>();
    let job = Job {
        kind: JobKind::Completion(req.prompt),
        sampler: resolve_sampler(&req.params),
        max_tokens: req.params.max_tokens.unwrap_or(256),
        grammar: req_grammar(req.grammar, req.response_format),
        reply,
    };
    if jobs.send(job).is_err() {
        return write_json(stream, 500, &error_json("generation worker is gone"));
    }
    if req.stream {
        stream_completion(stream, &rx, &model)
    } else {
        buffer_completion(stream, &rx, &model)
    }
}

// ---------------------------------------------------------------------------
// Response writers.
// ---------------------------------------------------------------------------

fn buffer_chat(stream: &mut TcpStream, rx: &Receiver<Event>, model: &str) -> io::Result<()> {
    let mut content = String::new();
    let (mut prompt, mut completion) = (0, 0);
    for ev in rx {
        match ev {
            Event::Token(s) => content.push_str(&s),
            Event::Done { prompt: p, completion: c } => {
                prompt = p;
                completion = c;
                break;
            }
            Event::Error(e) => return write_json(stream, 400, &error_json(&e)),
        }
    }
    let resp = ChatResponse {
        id: gen_id("chatcmpl"),
        object: "chat.completion",
        created: now(),
        model: model.to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: RespMessage { role: "assistant", content },
            finish_reason: "stop",
        }],
        usage: usage(prompt, completion),
    };
    write_json(stream, 200, &serde_json::to_string(&resp).unwrap_or_default())
}

fn stream_chat(stream: &mut TcpStream, rx: &Receiver<Event>, model: &str) -> io::Result<()> {
    let id = gen_id("chatcmpl");
    let created = now();
    write_sse_headers(stream)?;
    // First chunk announces the assistant role.
    send_chunk(stream, &chat_chunk(&id, created, model, Delta { role: Some("assistant"), content: None }, None))?;
    for ev in rx {
        match ev {
            Event::Token(s) => send_chunk(
                stream,
                &chat_chunk(&id, created, model, Delta { role: None, content: Some(s) }, None),
            )?,
            Event::Done { .. } => {
                send_chunk(
                    stream,
                    &chat_chunk(&id, created, model, Delta { role: None, content: None }, Some("stop")),
                )?;
                break;
            }
            Event::Error(e) => {
                write_sse_event(stream, &error_json(&e))?;
                break;
            }
        }
    }
    write_sse_event(stream, "[DONE]")
}

fn buffer_completion(stream: &mut TcpStream, rx: &Receiver<Event>, model: &str) -> io::Result<()> {
    let mut text = String::new();
    let (mut prompt, mut completion) = (0, 0);
    for ev in rx {
        match ev {
            Event::Token(s) => text.push_str(&s),
            Event::Done { prompt: p, completion: c } => {
                prompt = p;
                completion = c;
                break;
            }
            Event::Error(e) => return write_json(stream, 400, &error_json(&e)),
        }
    }
    let resp = CompletionResponse {
        id: gen_id("cmpl"),
        object: "text_completion",
        created: now(),
        model: model.to_string(),
        choices: vec![CompletionChoice { index: 0, text, finish_reason: "stop" }],
        usage: usage(prompt, completion),
    };
    write_json(stream, 200, &serde_json::to_string(&resp).unwrap_or_default())
}

fn stream_completion(stream: &mut TcpStream, rx: &Receiver<Event>, model: &str) -> io::Result<()> {
    let id = gen_id("cmpl");
    let created = now();
    write_sse_headers(stream)?;
    for ev in rx {
        match ev {
            Event::Token(s) => {
                let chunk = serde_json::json!({
                    "id": id, "object": "text_completion", "created": created, "model": model,
                    "choices": [{ "index": 0, "text": s, "finish_reason": serde_json::Value::Null }],
                });
                write_sse_event(stream, &chunk.to_string())?;
            }
            Event::Done { .. } => break,
            Event::Error(e) => {
                write_sse_event(stream, &error_json(&e))?;
                break;
            }
        }
    }
    write_sse_event(stream, "[DONE]")
}

fn chat_chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: Delta,
    finish_reason: Option<&'static str>,
) -> ChatChunk {
    ChatChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice { index: 0, delta, finish_reason }],
    }
}

fn send_chunk(stream: &mut TcpStream, chunk: &ChatChunk) -> io::Result<()> {
    write_sse_event(stream, &serde_json::to_string(chunk).unwrap_or_default())
}

// ---------------------------------------------------------------------------
// Small helpers.
// ---------------------------------------------------------------------------

fn resolve_sampler(p: &SamplingParams) -> SamplerConfig {
    SamplerConfig {
        temperature: p.temperature.unwrap_or(1.0),
        top_k: p.top_k.unwrap_or(0),
        top_p: p.top_p.unwrap_or(1.0),
        min_p: p.min_p.unwrap_or(0.0),
        min_keep: 1,
        repeat_penalty: 1.0,
        repeat_last_n: 64,
        frequency_penalty: p.frequency_penalty.unwrap_or(0.0),
        presence_penalty: p.presence_penalty.unwrap_or(0.0),
        seed: p.seed.unwrap_or_else(time_seed),
        mirostat: p.mirostat.unwrap_or(0),
        mirostat_tau: p.mirostat_tau.unwrap_or(5.0),
        mirostat_eta: p.mirostat_eta.unwrap_or(0.1),
        mirostat_m: 100,
        xtc_probability: p.xtc_probability.unwrap_or(0.0),
        xtc_threshold: p.xtc_threshold.unwrap_or(0.1),
    }
}

/// The grammar source for a request: an explicit `grammar` (raw GBNF) wins, else
/// `response_format: {"type":"json_object"}` selects the bundled JSON grammar.
fn req_grammar(grammar: Option<String>, fmt: Option<ResponseFormat>) -> Option<String> {
    if let Some(g) = grammar {
        if !g.is_empty() {
            return Some(g);
        }
    }
    if fmt.is_some_and(|f| f.kind == "json_object") {
        return Some(JSON_GRAMMAR.to_string());
    }
    None
}

fn parse_role(s: &str) -> Role {
    match s {
        "system" => Role::System,
        "assistant" => Role::Assistant,
        _ => Role::User,
    }
}

fn pick_model(requested: &str, served: &str) -> String {
    if requested.is_empty() {
        served.to_string()
    } else {
        requested.to_string()
    }
}

fn usage(prompt: usize, completion: usize) -> Usage {
    Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
    }
}

fn models_json(model_id: &str) -> String {
    serde_json::json!({
        "object": "list",
        "data": [{ "id": model_id, "object": "model", "created": now(), "owned_by": "rusty_llama" }],
    })
    .to_string()
}

fn error_json(msg: &str) -> String {
    serde_json::json!({ "error": { "message": msg, "type": "invalid_request_error" } }).to_string()
}

fn model_label(path: &str) -> String {
    std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("rusty-llama")
        .to_string()
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn time_seed() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
}

fn gen_id(prefix: &str) -> String {
    format!("{prefix}-{}", SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0))
}

fn write_response(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) -> io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        _ => "OK",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()
}

fn write_json(stream: &mut TcpStream, status: u16, json: &str) -> io::Result<()> {
    write_response(stream, status, "application/json", json.as_bytes())
}

fn write_text(stream: &mut TcpStream, status: u16, text: &str) -> io::Result<()> {
    write_response(stream, status, "text/plain; charset=utf-8", text.as_bytes())
}

fn write_sse_headers(stream: &mut TcpStream) -> io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
    )?;
    stream.flush()
}

fn write_sse_event(stream: &mut TcpStream, data: &str) -> io::Result<()> {
    write!(stream, "data: {data}\n\n")?;
    stream.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_chat_request() {
        let json = r#"{"model":"x","messages":[{"role":"system","content":"s"},
            {"role":"user","content":"hi"}],"temperature":0.5,"max_tokens":10,"stream":true}"#;
        let req: ChatRequest = serde_json::from_slice(json.as_bytes()).unwrap();
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[1].content, "hi");
        assert!(req.stream);
        assert_eq!(req.params.temperature, Some(0.5));
        assert_eq!(req.params.max_tokens, Some(10));
    }

    #[test]
    fn sampler_defaults_when_unspecified() {
        let c = resolve_sampler(&SamplingParams::default());
        assert_eq!(c.temperature, 1.0);
        assert_eq!(c.top_p, 1.0);
        assert_eq!(c.top_k, 0);
        assert_eq!(c.min_keep, 1);
    }

    #[test]
    fn roles_map_with_user_fallback() {
        assert!(matches!(parse_role("system"), Role::System));
        assert!(matches!(parse_role("assistant"), Role::Assistant));
        assert!(matches!(parse_role("user"), Role::User));
        assert!(matches!(parse_role("weird"), Role::User));
    }

    #[test]
    fn helpers_emit_valid_json() {
        let m: serde_json::Value = serde_json::from_str(&models_json("foo")).unwrap();
        assert_eq!(m["data"][0]["id"], "foo");
        let e: serde_json::Value = serde_json::from_str(&error_json("oops")).unwrap();
        assert_eq!(e["error"]["message"], "oops");
    }

    #[test]
    fn chat_chunk_omits_absent_delta_fields() {
        let c = chat_chunk("id1", 7, "m", Delta { role: Some("assistant"), content: None }, None);
        let v: serde_json::Value = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["choices"][0]["delta"]["role"], "assistant");
        assert!(v["choices"][0]["delta"].get("content").is_none());
    }

    #[test]
    fn body_limit_rejects_oversized() {
        // The clamp that guards `vec![0u8; content_length]` against an unbounded
        // client-supplied length. At-limit is allowed; one byte over is rejected.
        assert!(body_within_limit(0));
        assert!(body_within_limit(MAX_BODY_BYTES));
        assert!(!body_within_limit(MAX_BODY_BYTES + 1));
        assert!(!body_within_limit(usize::MAX));
    }
}
