//! Real chat-template rendering (`close-tachyon-scope-audit-gaps` task
//! group 2): real Hugging Face `tokenizer_config.json` chat templates are
//! genuine Jinja2 templates (variable substitution, `{% for %}` over
//! `messages`, `{% if %}`/`{% elif %}` on loop/message state, `{% set %}`,
//! the `tojson` filter, the `defined` test, whitespace-control tags, and
//! both `message['role']` and `message.role` attribute-access forms -- all
//! exercised by real checkpoints' own templates, not a hypothetical
//! superset). `minijinja` is a small, actively maintained, spec-driven
//! Jinja2-compatible engine -- the de facto standard for this exact job in
//! the Rust LLM-serving ecosystem -- so this renders the real template
//! through it rather than hand-maintaining a parser for an ad hoc subset.
//!
//! Kept entirely outside `magnetar-runtime`: chat templates are a Hugging
//! Face convention, not a portable Magnetar concept, the same
//! externalization boundary this crate already applies to real tokenizer/
//! Safetensors/config parsing. This module implements `magnetar-runtime`'s
//! existing, unchanged [`ChatTemplateFormatter`] trait.

use magnetar_runtime::inference_api::{ChatMessage, ChatTemplateFormatter, InferenceApiError};
use magnetar_runtime::production_model_ingestion::ProductionIngestionError;

const TEMPLATE_NAME: &str = "chat";

fn environment(template: &str) -> Result<minijinja::Environment<'static>, minijinja::Error> {
    let mut env = minijinja::Environment::new();
    // Real templates reference context variables a caller may not always
    // supply (`tools`, `add_generation_prompt`, ...); Jinja2's own
    // (and therefore real templates' own authors') expectation is that a
    // missing one is falsy/empty rather than a hard error -- `Lenient` is
    // minijinja's default and matches that, so this is documentation of
    // the existing default, not a change to it.
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Lenient);
    env.add_template_owned(TEMPLATE_NAME, template.to_string())?;
    Ok(env)
}

/// Real Hugging Face chat-template rendering. Validates that `template`
/// is syntactically valid Jinja2 at construction time -- minijinja's own
/// compiler is the validator, so a genuinely malformed template is
/// rejected here rather than accepted and failing unpredictably during a
/// real generation request. A *semantically* unsupported construct a
/// syntactically valid template can still reference (an unknown filter,
/// for instance -- minijinja resolves filter names dynamically at call
/// time, not at compile time) is not detectable until the first real
/// render; it still fails closed there with a structured error naming it,
/// never a silently wrong render, just not as early as construction.
#[derive(Debug)]
pub struct HuggingFaceChatTemplateFormatter {
    template: String,
    add_generation_prompt: bool,
}

impl HuggingFaceChatTemplateFormatter {
    /// `add_generation_prompt` controls the template's own
    /// `add_generation_prompt` context variable (real templates use it to
    /// decide whether to append the assistant turn's opening markup, e.g.
    /// Qwen's trailing `<|im_start|>assistant\n`) -- `true` is the correct
    /// default for every real generation call this formatter serves: the
    /// whole point of formatting a prompt is to then generate a response
    /// to it.
    pub fn new(template: impl Into<String>) -> Result<Self, ProductionIngestionError> {
        let template = template.into();
        environment(&template).map_err(|error| ProductionIngestionError::MalformedMetadata {
            reason: format!("chat template is not valid Jinja2: {error}"),
        })?;
        Ok(Self {
            template,
            add_generation_prompt: true,
        })
    }

    pub fn with_add_generation_prompt(mut self, add_generation_prompt: bool) -> Self {
        self.add_generation_prompt = add_generation_prompt;
        self
    }
}

impl ChatTemplateFormatter for HuggingFaceChatTemplateFormatter {
    fn format(&self, messages: &[ChatMessage]) -> Result<String, InferenceApiError> {
        let env =
            environment(&self.template).map_err(|error| InferenceApiError::TokenizationFailed {
                reason: format!("chat template is not valid Jinja2: {error}"),
            })?;
        let template = env.get_template(TEMPLATE_NAME).map_err(|error| {
            InferenceApiError::TokenizationFailed {
                reason: format!("chat template lookup failed: {error}"),
            }
        })?;
        let rendered_messages: Vec<minijinja::Value> = messages
            .iter()
            .map(|message| {
                minijinja::context! {
                    role => message.role.clone(),
                    content => message.content.clone(),
                }
            })
            .collect();
        template
            .render(minijinja::context! {
                messages => rendered_messages,
                add_generation_prompt => self.add_generation_prompt,
            })
            .map_err(|error| InferenceApiError::TokenizationFailed {
                reason: format!("chat template rendering failed: {error}"),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: &str, content: &str) -> ChatMessage {
        ChatMessage::new(role, content)
    }

    #[test]
    fn rejects_a_syntactically_invalid_template_at_construction() {
        let error = HuggingFaceChatTemplateFormatter::new("{% for message in messages %}")
            .expect_err("an unterminated for-loop is not valid Jinja2");
        assert!(matches!(
            error,
            ProductionIngestionError::MalformedMetadata { .. }
        ));
    }

    #[test]
    fn renders_a_simple_role_content_loop() {
        let formatter = HuggingFaceChatTemplateFormatter::new(
            "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}",
        )
        .unwrap();
        let rendered = formatter
            .format(&[message("user", "hi"), message("assistant", "hello")])
            .unwrap();
        assert_eq!(rendered, "user: hi\nassistant: hello\n");
    }

    #[test]
    fn supports_dict_subscript_attribute_access() {
        // Real templates use both `message.role` and `message['role']`.
        let formatter = HuggingFaceChatTemplateFormatter::new(
            "{% for message in messages %}{{ message['role'] }}={{ message['content'] }};{% endfor %}",
        )
        .unwrap();
        let rendered = formatter.format(&[message("user", "hi")]).unwrap();
        assert_eq!(rendered, "user=hi;");
    }

    #[test]
    fn renders_the_real_qwen2_5_instruct_chat_template() {
        // The exact `chat_template` string from the real, public
        // `Qwen/Qwen2.5-0.5B-Instruct` checkpoint's own
        // `tokenizer_config.json` (revision `7ae55760...`), proving this
        // formatter handles a genuinely real template: `if`/`elif`/`else`,
        // whitespace-control tags, `{% set %}`, the `tojson` filter, the
        // `defined` test, `loop.first`/`loop.last`/`loop.index0`, and
        // string concatenation with `+`.
        let template = include_str!("../tests/fixtures/qwen2.5-instruct-chat-template.jinja");
        let formatter = HuggingFaceChatTemplateFormatter::new(template).unwrap();
        let rendered = formatter
            .format(&[message("user", "What is the capital of France?")])
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful \
             assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn real_qwen_template_respects_a_declared_system_message() {
        let template = include_str!("../tests/fixtures/qwen2.5-instruct-chat-template.jinja");
        let formatter = HuggingFaceChatTemplateFormatter::new(template).unwrap();
        let rendered = formatter
            .format(&[
                message("system", "Answer in French."),
                message("user", "Hi"),
            ])
            .unwrap();
        assert!(rendered.starts_with("<|im_start|>system\nAnswer in French.<|im_end|>\n"));
    }

    #[test]
    fn renders_the_real_phi_3_mini_4k_instruct_chat_template() {
        // The exact `chat_template` string from the real, public
        // `microsoft/Phi-3-mini-4k-instruct` checkpoint's own
        // `tokenizer_config.json`, materially different from Qwen's own
        // template above: `<|system|>`/`<|user|>`/`<|assistant|>`/`<|end|>`
        // markup instead of `<|im_start|>`/`<|im_end|>`, an `if`/`elif`
        // chain keyed on `message['role']` instead of a single loop body,
        // and no `loop.first`/`loop.last`/`tojson`/`defined` usage at all.
        let template = include_str!("../tests/fixtures/phi-3-mini-4k-instruct-chat-template.jinja");
        let formatter = HuggingFaceChatTemplateFormatter::new(template).unwrap();
        let rendered = formatter
            .format(&[
                message("system", "You are a helpful assistant."),
                message("user", "What is the capital of France?"),
            ])
            .unwrap();
        assert_eq!(
            rendered,
            "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\nWhat is the capital of \
             France?<|end|>\n<|assistant|>\n"
        );
    }

    #[test]
    fn rejects_an_unknown_filter_at_render_time() {
        // minijinja resolves a filter name dynamically at call time, not
        // at compile time, so an unknown filter is syntactically valid
        // Jinja2 and construction succeeds -- only rendering fails, with
        // a real, specific error naming the unknown filter. Still
        // fail-closed (never a silently wrong render), just not
        // detectable until the first real render rather than at
        // ingestion, unlike a genuine syntax error (see the test above).
        let formatter = HuggingFaceChatTemplateFormatter::new(
            "{{ messages | this_filter_does_not_exist_anywhere }}",
        )
        .expect("syntactically valid Jinja2; the unknown filter is a run-time concern");
        let error = formatter
            .format(&[message("user", "hi")])
            .expect_err("an unknown filter fails at render time");
        assert!(matches!(
            error,
            InferenceApiError::TokenizationFailed { .. }
        ));
    }

    #[test]
    fn add_generation_prompt_false_omits_the_trailing_assistant_markup() {
        let template = include_str!("../tests/fixtures/qwen2.5-instruct-chat-template.jinja");
        let formatter = HuggingFaceChatTemplateFormatter::new(template)
            .unwrap()
            .with_add_generation_prompt(false);
        let rendered = formatter.format(&[message("user", "hi")]).unwrap();
        assert!(!rendered.ends_with("<|im_start|>assistant\n"));
    }
}
