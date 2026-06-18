//! Built-in chat templates (no Jinja): render OpenAI-style messages to a prompt
//! string, mirroring llama.cpp's hardcoded `llama-chat.cpp` templates. Phase 1
//! covers ChatML, Llama-3, and Qwen2 (Qwen2 shares the ChatML body).
//!
//! The special tokens the templates emit (`<|im_start|>`, `<|eot_id|>`,
//! `<|begin_of_text|>`, …) are matched verbatim by the tokenizer's special-token
//! splitter during `encode`, so a rendered prompt round-trips to the right ids.

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
}
