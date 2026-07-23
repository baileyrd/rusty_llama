//! Chat template rendering: [`ChatTemplate`] is 5 hardcoded families (no Jinja),
//! mirroring llama.cpp's `llama-chat.cpp`; [`ChatRenderer::resolve`] prefers
//! rendering the GGUF's *actual* embedded `tokenizer.chat_template` Jinja
//! source via `minijinja` instead, falling back to the hardcoded families only
//! when a GGUF has no template string at all.
//!
//! The special tokens the templates emit (`<|im_start|>`, `<|eot_id|>`,
//! `<|begin_of_text|>`, …) are matched verbatim by the tokenizer's special-token
//! splitter during `encode`, so a rendered prompt round-trips to the right ids.

use crate::error::{Error, Result};
use crate::gguf::Gguf;

/// A chat message role.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// One chat message.
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// A built-in chat template.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatTemplate {
    ChatMl,
    Llama3,
    Qwen2,
    Gemma,
    Phi3,
}

impl ChatTemplate {
    /// Detect from the GGUF `tokenizer.chat_template` markers, falling back to the
    /// architecture string. Returns `None` when neither matches.
    pub fn detect(gguf: &Gguf, arch: &str) -> Option<ChatTemplate> {
        let template = gguf.meta_str("tokenizer.chat_template").ok();
        Self::detect_from(template, arch)
    }

    /// Marker/arch detection split out for testing (no GGUF needed).
    fn detect_from(template: Option<&str>, arch: &str) -> Option<ChatTemplate> {
        if let Some(t) = template {
            if t.contains("<|start_header_id|>") {
                return Some(ChatTemplate::Llama3);
            }
            if t.contains("<|im_start|>") {
                return Some(ChatTemplate::ChatMl);
            }
            if t.contains("<start_of_turn>") {
                return Some(ChatTemplate::Gemma);
            }
            if t.contains("<|assistant|>") {
                return Some(ChatTemplate::Phi3);
            }
        }
        match arch {
            "qwen2" | "qwen" => Some(ChatTemplate::Qwen2),
            "gemma" | "gemma2" => Some(ChatTemplate::Gemma),
            "phi3" => Some(ChatTemplate::Phi3),
            _ => None,
        }
    }

    /// Parse a `--chat-template` override name.
    pub fn from_name(name: &str) -> Option<ChatTemplate> {
        match name {
            "chatml" => Some(ChatTemplate::ChatMl),
            "llama3" => Some(ChatTemplate::Llama3),
            "qwen2" => Some(ChatTemplate::Qwen2),
            "gemma" => Some(ChatTemplate::Gemma),
            "phi3" => Some(ChatTemplate::Phi3),
            _ => None,
        }
    }

    /// Whether the template emits its own beginning-of-text token, so `encode`
    /// should be called with `bos = false` to avoid a double-BOS.
    pub fn emits_bos(self) -> bool {
        matches!(self, ChatTemplate::Llama3)
    }

    /// Render messages to a prompt string. When `add_gen`, append the assistant
    /// generation header that primes the model to reply.
    pub fn render(&self, msgs: &[Message], add_gen: bool) -> String {
        match self {
            ChatTemplate::ChatMl | ChatTemplate::Qwen2 => render_chatml(msgs, add_gen),
            ChatTemplate::Llama3 => render_llama3(msgs, add_gen),
            ChatTemplate::Gemma => render_gemma(msgs, add_gen),
            ChatTemplate::Phi3 => render_phi3(msgs, add_gen),
        }
    }
}

/// Either one of the 5 hardcoded [`ChatTemplate`] families, or a Jinja
/// template rendered dynamically via `minijinja` from a GGUF's own
/// `tokenizer.chat_template` string.
pub enum ChatRenderer {
    Hardcoded(ChatTemplate),
    Jinja(String),
}

impl ChatRenderer {
    /// Resolve a renderer for `gguf`: an explicit `override_name` (CLI/server
    /// `--chat-template`) always wins and uses the matching hardcoded family;
    /// else the GGUF's own `tokenizer.chat_template` Jinja source is used if
    /// present (closing the gap a hardcoded-family guess can't: an
    /// unrecognized or customized template); else falls back to hardcoded-family
    /// detection ([`ChatTemplate::detect`]).
    pub fn resolve(gguf: &Gguf, arch: &str, override_name: Option<&str>) -> Option<Self> {
        Self::resolve_from(
            gguf.meta_str("tokenizer.chat_template").ok(),
            arch,
            override_name,
        )
    }

    /// Split out for testing (no real `Gguf` bytes needed) — mirrors
    /// [`ChatTemplate::detect_from`], which this delegates to for the final
    /// hardcoded-detection fallback.
    fn resolve_from(
        template: Option<&str>,
        arch: &str,
        override_name: Option<&str>,
    ) -> Option<Self> {
        if let Some(name) = override_name {
            return ChatTemplate::from_name(name).map(ChatRenderer::Hardcoded);
        }
        if let Some(src) = template {
            return Some(ChatRenderer::Jinja(src.to_string()));
        }
        // `template` is None here (the Some case already returned) — no
        // marker string to sniff, so this is arch-only detection.
        ChatTemplate::detect_from(None, arch).map(ChatRenderer::Hardcoded)
    }

    /// Render messages to a prompt string. `eos_token` feeds the Jinja context
    /// (ignored by the hardcoded path). A Jinja template that fails to parse
    /// or render is a hard error — never a silent fallback to a guessed
    /// family, which would risk producing a subtly wrong prompt.
    pub fn render(&self, msgs: &[Message], add_gen: bool, eos_token: &str) -> Result<String> {
        match self {
            ChatRenderer::Hardcoded(t) => Ok(t.render(msgs, add_gen)),
            ChatRenderer::Jinja(src) => render_jinja(src, msgs, add_gen, eos_token),
        }
    }

    /// Whether the caller should still ask the tokenizer to prepend BOS.
    /// Always `true` for a dynamic Jinja template: its context's `bos_token`
    /// is always the empty string (see [`render_jinja`]), so any `{{
    /// bos_token }}` the template emits is a no-op and the tokenizer is the
    /// single source of truth for BOS. Mirrors [`ChatTemplate::emits_bos`]
    /// for the hardcoded path.
    pub fn needs_tokenizer_bos(&self) -> bool {
        match self {
            ChatRenderer::Hardcoded(t) => !t.emits_bos(),
            ChatRenderer::Jinja(_) => true,
        }
    }
}

/// Render `template_src` (a GGUF's raw `tokenizer.chat_template` Jinja text)
/// via `minijinja`, with the same context shape Hugging Face's
/// `apply_chat_template`/llama.cpp's `minja` feed it: `messages` (a list of
/// `{role, content}`), `add_generation_prompt`, `bos_token`, `eos_token`.
///
/// `bos_token` is deliberately always `""`: whether a given template embeds it
/// (like the well-known Llama-3 template does, on the first message only)
/// isn't something this function can determine statically per-template, so
/// making it a no-op and always adding BOS via the tokenizer instead (see
/// [`ChatRenderer::needs_tokenizer_bos`]) avoids ever double-adding it — a
/// deliberate simplification, not a fidelity gap that matters: BOS is a single
/// fixed token regardless of which mechanism inserts it.
fn render_jinja(
    template_src: &str,
    msgs: &[Message],
    add_gen: bool,
    eos_token: &str,
) -> Result<String> {
    let mut env = minijinja::Environment::new();
    env.add_template("chat", template_src)
        .map_err(|e| Error::Format(format!("invalid chat template: {e}")))?;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| Error::Format(format!("invalid chat template: {e}")))?;

    let messages: Vec<minijinja::Value> = msgs
        .iter()
        .map(|m| {
            minijinja::Value::from_iter([
                ("role", m.role.as_str()),
                ("content", m.content.as_str()),
            ])
        })
        .collect();
    let ctx = minijinja::context! {
        messages => messages,
        add_generation_prompt => add_gen,
        bos_token => "",
        eos_token => eos_token,
    };
    tmpl.render(ctx)
        .map_err(|e| Error::Format(format!("chat template render error: {e}")))
}

fn render_chatml(msgs: &[Message], add_gen: bool) -> String {
    let mut s = String::new();
    for m in msgs {
        s.push_str("<|im_start|>");
        s.push_str(m.role.as_str());
        s.push('\n');
        s.push_str(&m.content);
        s.push_str("<|im_end|>\n");
    }
    if add_gen {
        s.push_str("<|im_start|>assistant\n");
    }
    s
}

fn render_llama3(msgs: &[Message], add_gen: bool) -> String {
    let mut s = String::from("<|begin_of_text|>");
    for m in msgs {
        s.push_str("<|start_header_id|>");
        s.push_str(m.role.as_str());
        s.push_str("<|end_header_id|>\n\n");
        s.push_str(&m.content);
        s.push_str("<|eot_id|>");
    }
    if add_gen {
        s.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }
    s
}

/// Gemma chat format: `<start_of_turn>{user|model}\n…<end_of_turn>`. Gemma has no
/// system role, so a system message is folded into the first user turn. The
/// tokenizer supplies `<bos>` (add_bos), so the template doesn't emit it.
fn render_gemma(msgs: &[Message], add_gen: bool) -> String {
    let mut s = String::new();
    let mut sys = String::new();
    for m in msgs {
        match m.role {
            Role::System => sys = m.content.clone(),
            Role::User => {
                s.push_str("<start_of_turn>user\n");
                if !sys.is_empty() {
                    s.push_str(&sys);
                    s.push_str("\n\n");
                    sys.clear();
                }
                s.push_str(&m.content);
                s.push_str("<end_of_turn>\n");
            }
            Role::Assistant => {
                s.push_str("<start_of_turn>model\n");
                s.push_str(&m.content);
                s.push_str("<end_of_turn>\n");
            }
        }
    }
    if add_gen {
        s.push_str("<start_of_turn>model\n");
    }
    s
}

/// Phi-3 chat format: `<|system|>\n…<|end|>\n<|user|>\n…<|end|>\n<|assistant|>\n`.
fn render_phi3(msgs: &[Message], add_gen: bool) -> String {
    let mut s = String::new();
    for m in msgs {
        s.push_str("<|");
        s.push_str(m.role.as_str());
        s.push_str("|>\n");
        s.push_str(&m.content);
        s.push_str("<|end|>\n");
    }
    if add_gen {
        s.push_str("<|assistant|>\n");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msgs() -> Vec<Message> {
        vec![
            Message { role: Role::System, content: "S".into() },
            Message { role: Role::User, content: "U".into() },
        ]
    }

    #[test]
    fn chatml_renders_expected() {
        assert_eq!(
            ChatTemplate::ChatMl.render(&msgs(), true),
            "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn llama3_renders_expected() {
        let one = [Message { role: Role::User, content: "hi".into() }];
        assert_eq!(
            ChatTemplate::Llama3.render(&one, true),
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn qwen2_shares_chatml_body() {
        assert_eq!(
            ChatTemplate::Qwen2.render(&msgs(), true),
            ChatTemplate::ChatMl.render(&msgs(), true)
        );
    }

    #[test]
    fn gemma_folds_system_into_user_and_maps_roles() {
        let m = vec![
            Message { role: Role::System, content: "S".into() },
            Message { role: Role::User, content: "U".into() },
            Message { role: Role::Assistant, content: "A".into() },
        ];
        assert_eq!(
            ChatTemplate::Gemma.render(&m, true),
            "<start_of_turn>user\nS\n\nU<end_of_turn>\n\
             <start_of_turn>model\nA<end_of_turn>\n\
             <start_of_turn>model\n"
        );
    }

    #[test]
    fn phi3_renders_role_tags() {
        assert_eq!(
            ChatTemplate::Phi3.render(&msgs(), true),
            "<|system|>\nS<|end|>\n<|user|>\nU<|end|>\n<|assistant|>\n"
        );
    }

    #[test]
    fn add_gen_false_omits_assistant_header() {
        assert!(!ChatTemplate::ChatMl.render(&msgs(), false).contains("assistant"));
    }

    #[test]
    fn from_name_parses_known_and_rejects_unknown() {
        assert_eq!(ChatTemplate::from_name("chatml"), Some(ChatTemplate::ChatMl));
        assert_eq!(ChatTemplate::from_name("llama3"), Some(ChatTemplate::Llama3));
        assert_eq!(ChatTemplate::from_name("qwen2"), Some(ChatTemplate::Qwen2));
        assert_eq!(ChatTemplate::from_name("jinja"), None);
        assert_eq!(ChatTemplate::from_name("gemma"), Some(ChatTemplate::Gemma));
        assert_eq!(ChatTemplate::from_name("phi3"), Some(ChatTemplate::Phi3));
    }

    #[test]
    fn detect_prefers_template_marker_then_arch() {
        assert_eq!(
            ChatTemplate::detect_from(Some("a <|start_header_id|> b"), "llama"),
            Some(ChatTemplate::Llama3)
        );
        assert_eq!(
            ChatTemplate::detect_from(Some("x <|im_start|> y"), "llama"),
            Some(ChatTemplate::ChatMl)
        );
        assert_eq!(ChatTemplate::detect_from(None, "qwen2"), Some(ChatTemplate::Qwen2));
        assert_eq!(ChatTemplate::detect_from(None, "llama"), None);
    }

    #[test]
    fn only_llama3_self_emits_bos() {
        assert!(ChatTemplate::Llama3.emits_bos());
        assert!(!ChatTemplate::ChatMl.emits_bos());
        assert!(!ChatTemplate::Qwen2.emits_bos());
    }

    // --- ChatRenderer / render_jinja ---------------------------------------

    #[test]
    fn render_jinja_threads_context_variables() {
        // A trivial template exercising each context key the render passes:
        // messages (as a sequence, via `length`), add_generation_prompt (bool
        // interpolation), and eos_token (string interpolation). bos_token is
        // always "" (see render_jinja's doc comment), so it's covered by the
        // Llama-3-shaped test below instead of asserted on directly here.
        let out = render_jinja(
            "{{ messages|length }} msgs, gen={{ add_generation_prompt }}, eos={{ eos_token }}",
            &msgs(),
            true,
            "<|eot|>",
        )
        .unwrap();
        assert_eq!(out, "2 msgs, gen=true, eos=<|eot|>");
    }

    #[test]
    fn render_jinja_llama3_shaped_template() {
        // A template with the same *shape* as the well-known, widely-published
        // Llama-3 chat template (for-loop over messages, an `{% if
        // loop.index0 == 0 %}` bos-prepend, a trailing generation-prompt
        // block) — written and hand-traced independently here, not copied
        // from a live GGUF (none available in this environment), so this
        // proves the rendering mechanism (loops/if/interpolation/loop.index0)
        // works, not byte-fidelity to any specific upstream file.
        const TEMPLATE: &str = "{% for message in messages %}\
             {% if loop.index0 == 0 %}{{ bos_token }}{% endif %}\
             <|start_header_id|>{{ message.role }}<|end_header_id|>\n\n\
             {{ message.content }}<|eot_id|>\
             {% endfor %}\
             {% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}";
        let out = render_jinja(TEMPLATE, &msgs(), true, "<|eot_id|>").unwrap();
        // bos_token is always "" (see render_jinja's doc comment), so unlike
        // ChatTemplate::Llama3's hardcoded render, no <|begin_of_text|> here —
        // ChatRenderer::needs_tokenizer_bos() is what covers BOS for this path.
        assert_eq!(
            out,
            "<|start_header_id|>system<|end_header_id|>\n\nS<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\nU<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn render_jinja_malformed_template_is_a_clear_error() {
        let err = render_jinja("{% for x in messages %}unclosed", &msgs(), true, "")
            .expect_err("malformed template must error, not silently fall back");
        assert!(
            format!("{err}").to_lowercase().contains("template"),
            "{err}"
        );
    }

    #[test]
    fn chat_renderer_override_name_wins_over_template_string() {
        let r = ChatRenderer::resolve_from(Some("{{ anything }}"), "llama", Some("chatml"));
        assert!(matches!(
            r,
            Some(ChatRenderer::Hardcoded(ChatTemplate::ChatMl))
        ));
    }

    #[test]
    fn chat_renderer_prefers_template_string_over_arch_detection() {
        // Even though arch="qwen2" would resolve to a known hardcoded family,
        // a present template string wins (closing the actual gap: qwen2 GGUFs
        // can carry a customized template too).
        let r = ChatRenderer::resolve_from(Some("{{ messages }}"), "qwen2", None);
        assert!(matches!(r, Some(ChatRenderer::Jinja(s)) if s == "{{ messages }}"));
    }

    #[test]
    fn chat_renderer_falls_back_to_hardcoded_detection_when_no_template_string() {
        let r = ChatRenderer::resolve_from(None, "qwen2", None);
        assert!(matches!(
            r,
            Some(ChatRenderer::Hardcoded(ChatTemplate::Qwen2))
        ));
        assert!(ChatRenderer::resolve_from(None, "unknown-arch", None).is_none());
    }

    #[test]
    fn chat_renderer_needs_tokenizer_bos() {
        assert!(ChatRenderer::Hardcoded(ChatTemplate::ChatMl).needs_tokenizer_bos());
        assert!(!ChatRenderer::Hardcoded(ChatTemplate::Llama3).needs_tokenizer_bos());
        assert!(ChatRenderer::Jinja("{{ x }}".into()).needs_tokenizer_bos());
    }
}
