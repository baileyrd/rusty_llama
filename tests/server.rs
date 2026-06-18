//! End-to-end test of the OpenAI-compatible HTTP server: spawn it on a real
//! GGUF and drive `/v1/completions` over a raw socket. Skips when no test model
//! is on disk (CI has none), so it validates locally without flaking elsewhere.
#![cfg(feature = "server")]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::thread;
use std::time::Duration;

/// A small local GGUF to serve, or `None` (skip) when absent. Override with
/// `RUSTY_LLAMA_GGUF`; defaults to the Qwen2-0.5B used elsewhere in the suite.
fn test_model() -> Option<String> {
    let path = std::env::var("RUSTY_LLAMA_GGUF").unwrap_or_else(|_| "qwen2-0_5b-q8.gguf".into());
    std::path::Path::new(&path).exists().then_some(path)
}

fn post(addr: &str, path: &str, body: &str) -> String {
    let mut s = TcpStream::connect(addr).expect("connect");
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).expect("write request");
    let mut resp = String::new();
    s.read_to_string(&mut resp).expect("read response");
    resp
}

#[test]
fn server_serves_openai_endpoints() {
    let Some(model) = test_model() else {
        eprintln!("skipping server e2e: no test GGUF (set RUSTY_LLAMA_GGUF)");
        return;
    };
    // A fixed high port; the OS frees it when this test process exits.
    let port = 18931u16;
    let addr = format!("127.0.0.1:{port}");
    thread::spawn(move || {
        let _ = rusty_llama::serve(&model, "cpu", "127.0.0.1", port, None);
    });
    // serve() binds only after the model loads, so a successful connect means
    // the server is ready. Retry while it warms up.
    let mut up = false;
    for _ in 0..150 {
        if TcpStream::connect(&addr).is_ok() {
            up = true;
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
    assert!(up, "server never came up on {addr}");

    // /v1/completions — greedy, a few tokens.
    let resp = post(
        &addr,
        "/v1/completions",
        r#"{"prompt":"Hello,","max_tokens":4,"temperature":0}"#,
    );
    assert!(resp.contains("200 OK"), "completions status: {resp}");
    assert!(resp.contains("text_completion"), "completions object: {resp}");
    let json = resp.split("\r\n\r\n").nth(1).unwrap_or("");
    let v: serde_json::Value = serde_json::from_str(json).expect("valid completion JSON");
    assert!(v["choices"][0]["text"].is_string());
    assert!(v["usage"]["completion_tokens"].as_u64().unwrap() >= 1);

    // A malformed body is a clean 400, not a panic / hang.
    let bad = post(&addr, "/v1/chat/completions", "{ not json");
    assert!(bad.contains("400"), "bad-request status: {bad}");
}
