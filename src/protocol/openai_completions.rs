//! OpenAI Chat Completions API provider.

/// Default base URL for OpenAI Chat Completions API.
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const NON_VISION_USER_IMAGE_PLACEHOLDER: &str = "(image omitted: model does not support images)";
const NON_VISION_TOOL_IMAGE_PLACEHOLDER: &str =
    "(tool image omitted: model does not support images)";

use crate::protocol::LLMProtocol;
use crate::stream::{parse_streaming_json, AssistantMessageEventStream};
use crate::thinking::OpenAIThinkingOptions;
use crate::transform::{normalize_tool_call_id, transform_messages};
use crate::types::*;
use crate::types::{SimpleStreamOptions, StreamOptions};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// OpenAI Completions API provider.
pub struct OpenAICompletionsProtocol {
    client: Client,
    default_api_key: Option<String>,
}

impl OpenAICompletionsProtocol {
    /// Create a new OpenAI Completions provider.
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .connect_timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| Client::new()),
            default_api_key: None,
        }
    }

    /// Create a provider with a default API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::builder()
                .connect_timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| Client::new()),
            default_api_key: Some(api_key.into()),
        }
    }

    /// Get API key from options or environment.
    fn resolve_api_key(&self, options: &StreamOptions, provider: &Provider) -> String {
        // Priority: options.api_key > self.default_api_key > environment variable
        if let Some(ref key) = options.api_key {
            return key.clone();
        }
        if let Some(ref key) = self.default_api_key {
            return key.clone();
        }

        // Try environment variable based on provider
        let env_key = match provider {
            Provider::OpenAI => std::env::var("OPENAI_API_KEY").ok(),
            Provider::OpenAICompatible => std::env::var("OPENAI_API_KEY").ok(),
            Provider::Groq => std::env::var("GROQ_API_KEY").ok(),
            Provider::XAI => std::env::var("XAI_API_KEY").ok(),
            Provider::Cerebras => std::env::var("CEREBRAS_API_KEY").ok(),
            Provider::OpenRouter => std::env::var("OPENROUTER_API_KEY").ok(),
            Provider::VercelAiGateway => std::env::var("AI_GATEWAY_API_KEY").ok(),
            Provider::Mistral => std::env::var("MISTRAL_API_KEY").ok(),
            Provider::ZAI => std::env::var("ZAI_API_KEY").ok(),
            Provider::Ollama => return String::new(), // Ollama doesn't need API key
            _ => None,
        };

        env_key.unwrap_or_default()
    }
}

impl Default for OpenAICompletionsProtocol {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProtocol for OpenAICompletionsProtocol {
    fn provider_type(&self) -> Provider {
        Provider::OpenAI
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        let stream = AssistantMessageEventStream::new_assistant_stream();
        let stream_clone = stream.clone();

        let model = model.clone();
        let context = context.clone();
        let client = self.client.clone();
        let api_key = self.resolve_api_key(&options, &model.provider);
        let error_stream = stream_clone.clone();

        tokio::spawn(async move {
            if let Err(e) = run_stream(
                client,
                &model,
                &context,
                options,
                api_key,
                None,
                stream_clone,
            )
            .await
            {
                tracing::error!("Stream error: {}", e);
                super::common::emit_background_task_error(
                    &model,
                    Api::OpenAICompletions,
                    format!("OpenAI Completions stream error: {}", e),
                    &error_stream,
                );
            }
        });

        stream
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let thinking_options = options.reasoning.map(OpenAIThinkingOptions::from_level);

        let stream_options = options.base;

        let stream = AssistantMessageEventStream::new_assistant_stream();
        let stream_clone = stream.clone();

        let model = model.clone();
        let context = context.clone();
        let client = self.client.clone();
        let api_key = self.resolve_api_key(&stream_options, &model.provider);
        let error_stream = stream_clone.clone();

        tokio::spawn(async move {
            if let Err(e) = run_stream(
                client,
                &model,
                &context,
                stream_options,
                api_key,
                thinking_options,
                stream_clone,
            )
            .await
            {
                tracing::error!("Stream error: {}", e);
                super::common::emit_background_task_error(
                    &model,
                    Api::OpenAICompletions,
                    format!("OpenAI Completions stream error: {}", e),
                    &error_stream,
                );
            }
        });

        stream
    }
}

// ============================================================================
// Compat Resolution
// ============================================================================

/// Resolve compat settings for the model, using defaults when model.compat is None.
fn resolve_compat(model: &Model) -> OpenAICompletionsCompat {
    model.compat.clone().unwrap_or_else(|| detect_compat(model))
}

fn detect_compat(model: &Model) -> OpenAICompletionsCompat {
    let base_url = model.base_url.as_deref().unwrap_or("").to_ascii_lowercase();
    let is_zai = matches!(model.provider, Provider::ZAI) || base_url.contains("api.z.ai");
    let is_non_standard = matches!(
        model.provider,
        Provider::Cerebras
            | Provider::XAI
            | Provider::DeepSeek
            | Provider::ZAI
            | Provider::OpenCode
            | Provider::OpenAICompatible
            | Provider::Zenmux
    ) || base_url.contains("cerebras.ai")
        || base_url.contains("api.x.ai")
        || base_url.contains("chutes.ai")
        || base_url.contains("deepseek.com")
        || base_url.contains("api.z.ai")
        || base_url.contains("opencode.ai")
        || base_url.contains("zenmux.ai");
    let use_max_tokens = base_url.contains("chutes.ai");
    let is_grok = matches!(model.provider, Provider::XAI) || base_url.contains("api.x.ai");
    let is_groq = matches!(model.provider, Provider::Groq) || base_url.contains("groq.com");

    // NOTE: `reasoning_content_constrained` is intentionally left false here.
    // It is set by the provider's `default_compat()` (e.g. DeepSeekProvider) or
    // injected via catalog patches (patches.json).

    let reasoning_effort_map = if is_groq && model.id.eq_ignore_ascii_case("qwen/qwen3-32b") {
        HashMap::from([
            ("minimal".to_string(), "default".to_string()),
            ("low".to_string(), "default".to_string()),
            ("high".to_string(), "default".to_string()),
            ("xhigh".to_string(), "default".to_string()),
        ])
    } else {
        HashMap::new()
    };

    OpenAICompletionsCompat {
        supports_store: !is_non_standard,
        supports_developer_role: !is_non_standard,
        supports_reasoning_effort: !is_grok && !is_zai,
        reasoning_effort_map,
        supports_usage_in_streaming: true,
        max_tokens_field: if use_max_tokens {
            Some("max_tokens".to_string())
        } else {
            Some("max_completion_tokens".to_string())
        },
        requires_tool_result_name: false,
        requires_assistant_after_tool_result: false,
        requires_thinking_as_text: false,
        thinking_format: if is_zai {
            "zai".to_string()
        } else {
            "openai".to_string()
        },
        supports_strict_mode: true,
        open_router_routing: None,
        reasoning_content_constrained: false,
    }
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// OpenAI Chat Completions request.
#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptionsConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<bool>,
    /// ZAI / Qwen thinking format: top-level enable_thinking flag.
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_thinking: Option<bool>,
    /// Qwen chat-template thinking format.
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<serde_json::Value>,
    /// OpenRouter routing preferences.
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct StreamOptionsConfig {
    include_usage: bool,
}

/// OpenAI message format.
#[derive(Debug, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<OpenAIContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    /// Extra fields for provider-specific data (e.g. reasoning_content).
    /// Flattened into the top-level JSON object during serialization.
    #[serde(flatten, skip_serializing_if = "HashMap::is_empty", default)]
    extra_fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum OpenAIContent {
    Text(String),
    Parts(Vec<OpenAIContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIContentPart {
    #[serde(rename = "type")]
    content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<ImageUrl>,
    /// Cache control hint (for OpenRouter + Anthropic models).
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImageUrl {
    url: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OpenAIFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAIFunctionDef,
}

#[derive(Debug, Serialize)]
struct OpenAIFunctionDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

/// Streaming response chunk.
#[derive(Debug, Deserialize)]
struct ChatCompletionChunk {
    id: Option<String>,
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    usage: Option<ChunkUsage>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    #[allow(dead_code)]
    index: u32,
    delta: Option<ChunkDelta>,
    finish_reason: Option<String>,
    #[serde(default)]
    usage: Option<ChunkUsage>,
}

#[derive(Debug, Deserialize, Default)]
struct ChunkDelta {
    #[allow(dead_code)]
    role: Option<String>,
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    reasoning_text: Option<String>,
    #[serde(default)]
    reasoning_details: Option<Vec<serde_json::Value>>,
    #[serde(default, deserialize_with = "deserialize_null_as_default")]
    tool_calls: Vec<ChunkToolCall>,
}

#[derive(Debug, Deserialize)]
struct ChunkToolCall {
    index: Option<u32>,
    id: Option<String>,
    function: Option<ChunkFunction>,
}

#[derive(Debug, Deserialize)]
struct ChunkFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChunkUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    #[allow(dead_code)]
    prompt_tokens_details: Option<PromptTokensDetails>,
    #[allow(dead_code)]
    completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct PromptTokensDetails {
    #[allow(dead_code)]
    cached_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct CompletionTokensDetails {
    #[allow(dead_code)]
    reasoning_tokens: Option<u64>,
}

fn apply_openai_usage(output: &mut AssistantMessage, usage: &ChunkUsage) {
    let cached_tokens = usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.cached_tokens)
        .unwrap_or(0);
    let reasoning_tokens = usage
        .completion_tokens_details
        .as_ref()
        .and_then(|details| details.reasoning_tokens)
        .unwrap_or(0);

    let input_tokens = usage.prompt_tokens.unwrap_or(0);
    let completion_tokens = usage.completion_tokens.unwrap_or(0);

    output.usage.input = input_tokens.saturating_sub(cached_tokens);
    output.usage.output = completion_tokens + reasoning_tokens;
    output.usage.cache_read = cached_tokens;
    output.usage.total_tokens = output.usage.input + output.usage.output + output.usage.cache_read;
}

// ============================================================================
// Message Conversion
// ============================================================================

/// Convert context to OpenAI messages.
fn normalize_openai_tool_call_id(id: &str) -> String {
    if id.contains('|') {
        let call_id = id.split('|').next().unwrap_or(id);
        return call_id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .take(40)
            .collect();
    }

    normalize_tool_call_id(id, &Provider::OpenAI)
}

fn has_tool_history(messages: &[Message]) -> bool {
    messages.iter().any(|msg| match msg {
        Message::ToolResult(_) => true,
        Message::Assistant(assistant) => assistant.content.iter().any(|block| block.is_tool_call()),
        Message::User(_) => false,
    })
}

fn convert_messages(
    context: &Context,
    model: &Model,
    thinking_enabled: bool,
) -> Vec<OpenAIMessage> {
    let compat = resolve_compat(model);
    let mut messages = Vec::new();
    let transformed = transform_messages(
        &context.messages,
        model,
        Some(&normalize_openai_tool_call_id),
    );

    // Normalize reasoning content for constrained providers (e.g. DeepSeek)
    let default_url = "";
    let base_url = super::common::resolve_base_url(None, model.base_url.as_deref(), default_url);
    let transformed = super::common::normalize_reasoning_content(
        transformed,
        compat.reasoning_content_constrained,
        thinking_enabled,
        base_url,
    );

    // Add system prompt
    if let Some(ref prompt) = context.system_prompt {
        let use_developer = model.reasoning && compat.supports_developer_role;
        let role = if use_developer { "developer" } else { "system" };

        messages.push(OpenAIMessage {
            role: role.to_string(),
            content: Some(OpenAIContent::Text(sanitize_surrogates(prompt))),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            extra_fields: HashMap::new(),
        });
    }

    // Track last message role for requiresAssistantAfterToolResult logic
    let mut last_was_tool_result = false;

    // Convert messages
    for msg in &transformed {
        match msg {
            Message::User(user_msg) => {
                // P1-3: Insert synthetic assistant message between tool result and user
                if last_was_tool_result && compat.requires_assistant_after_tool_result {
                    messages.push(OpenAIMessage {
                        role: "assistant".to_string(),
                        content: Some(OpenAIContent::Text(
                            "I have processed the tool results.".to_string(),
                        )),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                        extra_fields: HashMap::new(),
                    });
                }

                messages.push(convert_user_message(user_msg, model));
                last_was_tool_result = false;
            }
            Message::Assistant(assistant_msg) => {
                let openai_msg = convert_assistant_message(assistant_msg, model, &compat);
                if let Some(msg) = openai_msg {
                    messages.push(msg);
                }
                last_was_tool_result = false;
            }
            Message::ToolResult(tool_result) => {
                messages.extend(convert_tool_result(tool_result, model));
                last_was_tool_result = true;
            }
        }
    }

    // P1-5: OpenRouter Anthropic cache_control injection
    maybe_add_openrouter_anthropic_cache_control(&mut messages, model);

    messages
}

fn text_part(text: impl Into<String>) -> OpenAIContentPart {
    OpenAIContentPart {
        content_type: "text".to_string(),
        text: Some(text.into()),
        image_url: None,
        cache_control: None,
    }
}

fn image_part(image: &ImageContent) -> OpenAIContentPart {
    OpenAIContentPart {
        content_type: "image_url".to_string(),
        text: None,
        image_url: Some(ImageUrl {
            url: format!("data:{};base64,{}", image.mime_type, image.data),
        }),
        cache_control: None,
    }
}

fn normalize_user_parts(blocks: &[ContentBlock], model: &Model) -> Vec<OpenAIContentPart> {
    let mut parts = Vec::new();
    let mut previous_was_placeholder = false;

    for block in blocks {
        match block {
            ContentBlock::Text(t) => {
                let text = sanitize_surrogates(&t.text);
                previous_was_placeholder = text == NON_VISION_USER_IMAGE_PLACEHOLDER;
                parts.push(text_part(text));
            }
            ContentBlock::Image(img) => {
                if model.supports_image() {
                    parts.push(image_part(img));
                    previous_was_placeholder = false;
                } else if !previous_was_placeholder {
                    parts.push(text_part(NON_VISION_USER_IMAGE_PLACEHOLDER));
                    previous_was_placeholder = true;
                }
            }
            _ => {}
        }
    }

    parts
}

fn build_user_content(parts: Vec<OpenAIContentPart>) -> OpenAIContent {
    if parts.len() == 1 && parts[0].content_type == "text" {
        return OpenAIContent::Text(parts[0].text.clone().unwrap_or_default());
    }

    OpenAIContent::Parts(parts)
}

fn convert_user_message(user_msg: &UserMessage, model: &Model) -> OpenAIMessage {
    match &user_msg.content {
        UserContent::Text(text) => OpenAIMessage {
            role: "user".to_string(),
            content: Some(OpenAIContent::Text(sanitize_surrogates(text))),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            extra_fields: HashMap::new(),
        },
        UserContent::Blocks(blocks) => {
            let parts = normalize_user_parts(blocks, model);

            OpenAIMessage {
                role: "user".to_string(),
                content: Some(if parts.is_empty() {
                    OpenAIContent::Text(String::new())
                } else {
                    build_user_content(parts)
                }),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                extra_fields: HashMap::new(),
            }
        }
    }
}

fn convert_assistant_message(
    assistant_msg: &AssistantMessage,
    _model: &Model,
    compat: &OpenAICompletionsCompat,
) -> Option<OpenAIMessage> {
    // Skip error/aborted messages
    if assistant_msg.stop_reason == StopReason::Error
        || assistant_msg.stop_reason == StopReason::Aborted
    {
        return None;
    }

    // Handle thinking blocks
    let thinking_blocks: Vec<_> = assistant_msg
        .content
        .iter()
        .filter_map(|b| b.as_thinking())
        .filter(|t| !t.thinking.trim().is_empty() && !t.redacted)
        .collect();

    let thinking_text = if !thinking_blocks.is_empty() {
        Some(
            thinking_blocks
                .iter()
                .map(|t| t.thinking.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        )
    } else {
        None
    };

    // Get text content
    let mut text_content: String = assistant_msg
        .content
        .iter()
        .filter_map(|b| b.as_text())
        .filter(|t| !t.text.trim().is_empty())
        .map(|t| sanitize_surrogates(&t.text))
        .collect::<Vec<_>>()
        .join("");

    // P0-2: If requires_thinking_as_text, prepend thinking to text content
    if compat.requires_thinking_as_text {
        if let Some(ref thinking) = thinking_text {
            let mut combined = thinking.clone();
            if !text_content.is_empty() {
                combined.push_str("\n\n");
            }
            combined.push_str(&text_content);
            text_content = combined;
        }
    }

    // Get tool calls
    let tool_calls: Vec<OpenAIToolCall> = assistant_msg
        .content
        .iter()
        .filter_map(|b| b.as_tool_call())
        .map(|tc| OpenAIToolCall {
            id: tc.id.clone(),
            call_type: "function".to_string(),
            function: OpenAIFunction {
                name: tc.name.clone(),
                arguments: serde_json::to_string(&tc.arguments).unwrap_or_default(),
            },
        })
        .collect();

    let reasoning_details: Vec<serde_json::Value> = assistant_msg
        .content
        .iter()
        .filter_map(|b| b.as_tool_call())
        .filter_map(|tc| tc.thought_signature.as_deref())
        .filter_map(|sig| serde_json::from_str::<serde_json::Value>(sig).ok())
        .collect();

    // Skip if no content and no tool calls
    if text_content.is_empty() && tool_calls.is_empty() && thinking_text.is_none() {
        return None;
    }

    let content = if text_content.is_empty() {
        // When reasoning_content exists but content is empty, send "" instead of null
        // for provider compatibility (some providers reject null content with reasoning)
        if thinking_text.is_some() {
            Some(OpenAIContent::Text(String::new()))
        } else {
            None
        }
    } else {
        Some(OpenAIContent::Text(text_content))
    };

    let mut extra_fields = HashMap::new();

    // P0-2: Add thinking as reasoning_content extra field.
    // Always use \"reasoning_content\" as the key — thinking_signature is
    // a cryptographic signature for Anthropic model verification; it must
    // never be used as a JSON field name in OpenAI-compatible requests.
    if !compat.requires_thinking_as_text {
        if let Some(ref thinking) = thinking_text {
            extra_fields.insert(
                "reasoning_content".to_string(),
                serde_json::Value::String(thinking.clone()),
            );
        }
    }

    if !reasoning_details.is_empty() {
        extra_fields.insert(
            "reasoning_details".to_string(),
            serde_json::Value::Array(reasoning_details),
        );
    }

    let msg = OpenAIMessage {
        role: "assistant".to_string(),
        content,
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        tool_call_id: None,
        name: None,
        extra_fields,
    };

    Some(msg)
}

fn convert_tool_result(tool_result: &ToolResultMessage, model: &Model) -> Vec<OpenAIMessage> {
    let text: String = tool_result
        .content
        .iter()
        .filter_map(|b| b.as_text())
        .map(|t| sanitize_surrogates(&t.text))
        .collect::<Vec<_>>()
        .join("\n");
    let text_is_empty = text.is_empty();

    let images: Vec<&ImageContent> = tool_result
        .content
        .iter()
        .filter_map(|b| b.as_image())
        .collect();

    let requires_name = model
        .compat
        .as_ref()
        .is_some_and(|c| c.requires_tool_result_name);

    let mut messages = vec![OpenAIMessage {
        role: "tool".to_string(),
        content: Some(OpenAIContent::Text(if text_is_empty {
            if images.is_empty() {
                "(no output)".to_string()
            } else if model.supports_image() {
                "(image output attached)".to_string()
            } else {
                NON_VISION_TOOL_IMAGE_PLACEHOLDER.to_string()
            }
        } else {
            text
        })),
        tool_calls: None,
        tool_call_id: Some(tool_result.tool_call_id.clone()),
        name: if requires_name {
            Some(tool_result.tool_name.clone())
        } else {
            None
        },
        extra_fields: HashMap::new(),
    }];

    if !images.is_empty() {
        if model.supports_image() {
            let parts = images.into_iter().map(image_part).collect::<Vec<_>>();
            messages.push(OpenAIMessage {
                role: "user".to_string(),
                content: Some(build_user_content(parts)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                extra_fields: HashMap::new(),
            });
        } else if text_is_empty {
            messages[0].content = Some(OpenAIContent::Text(
                NON_VISION_TOOL_IMAGE_PLACEHOLDER.to_string(),
            ));
        }
    }

    messages
}

fn convert_tool_choice(tool_choice: Option<&ToolChoice>) -> Option<serde_json::Value> {
    match tool_choice? {
        ToolChoice::Mode(ToolChoiceMode::Auto) => Some(serde_json::json!("auto")),
        ToolChoice::Mode(ToolChoiceMode::None) => Some(serde_json::json!("none")),
        ToolChoice::Mode(ToolChoiceMode::Any | ToolChoiceMode::Required) => {
            Some(serde_json::json!("required"))
        }
        ToolChoice::Named(ToolChoiceNamed::Tool { name }) => Some(serde_json::json!({
            "type": "function",
            "function": { "name": name }
        })),
        ToolChoice::Named(ToolChoiceNamed::Function { function }) => Some(serde_json::json!({
            "type": "function",
            "function": { "name": function.name }
        })),
    }
}

/// Convert tools to OpenAI format, respecting compat strict mode setting.
fn convert_tools(tools: &[Tool], compat: &OpenAICompletionsCompat) -> Vec<OpenAITool> {
    tools
        .iter()
        .map(|t| OpenAITool {
            tool_type: "function".to_string(),
            function: OpenAIFunctionDef {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.parameters.clone(),
                // P1-2: Only include strict if provider supports it
                strict: if compat.supports_strict_mode {
                    Some(false)
                } else {
                    None
                },
            },
        })
        .collect()
}

/// P1-5: Add cache_control to last user/assistant text part for OpenRouter + Anthropic models.
fn maybe_add_openrouter_anthropic_cache_control(messages: &mut [OpenAIMessage], model: &Model) {
    // Only apply for OpenRouter provider with Anthropic models
    if model.provider != Provider::OpenRouter || !model.id.starts_with("anthropic/") {
        return;
    }

    let cache_control_value = serde_json::json!({ "type": "ephemeral" });

    // Walk backwards to find last user or assistant message with text content
    for msg in messages.iter_mut().rev() {
        if msg.role != "user" && msg.role != "assistant" {
            continue;
        }

        match &mut msg.content {
            Some(OpenAIContent::Parts(parts)) => {
                // Find last text part and add cache_control
                for part in parts.iter_mut().rev() {
                    if part.content_type == "text" {
                        part.cache_control = Some(cache_control_value);
                        return;
                    }
                }
            }
            Some(OpenAIContent::Text(_text)) => {
                // Convert to Parts format so we can add cache_control
                let text_val = if let Some(OpenAIContent::Text(t)) = msg.content.take() {
                    t
                } else {
                    return;
                };
                msg.content = Some(OpenAIContent::Parts(vec![OpenAIContentPart {
                    content_type: "text".to_string(),
                    text: Some(text_val),
                    image_url: None,
                    cache_control: Some(cache_control_value),
                }]));
                return;
            }
            _ => continue,
        }
    }
}

/// Sanitize Unicode surrogates.
fn sanitize_surrogates(text: &str) -> String {
    text.replace(
        |c: char| {
            let cp = c as u32;
            (0xD800..=0xDFFF).contains(&cp)
        },
        "",
    )
}

// ============================================================================
// Thinking / Reasoning Effort Resolution
// ============================================================================

/// Resolve the reasoning_effort value, applying the compat reasoning_effort_map if present.
fn resolve_reasoning_effort(effort: &str, compat: &OpenAICompletionsCompat) -> String {
    // Check if there's a mapped value for this effort level
    if let Some(mapped) = compat.reasoning_effort_map.get(effort) {
        return mapped.clone();
    }
    effort.to_string()
}

// ============================================================================
// Streaming Implementation
// ============================================================================

async fn run_stream(
    client: Client,
    model: &Model,
    context: &Context,
    options: StreamOptions,
    api_key: String,
    thinking_options: Option<OpenAIThinkingOptions>,
    stream: AssistantMessageEventStream,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let limits = options.security_config();
    let compat = resolve_compat(model);
    let cancel_token = options.cancel_token.clone();

    let mut output = AssistantMessage::builder()
        .api(model.api.clone().unwrap_or(Api::OpenAICompletions))
        .provider(model.provider.clone())
        .model(model.id.clone())
        .stop_reason(StopReason::Stop)
        .usage(Usage::default())
        .build()?;

    let thinking_enabled = thinking_options.is_some();
    let messages = convert_messages(context, model, thinking_enabled);
    let tools = context
        .tools
        .as_ref()
        .map(|t| convert_tools(t, &compat))
        .or_else(|| has_tool_history(&context.messages).then(Vec::new));
    let clamped_max_tokens = super::common::clamp_openai_max_tokens(options.max_tokens);

    // Determine which max tokens field to use
    let max_tokens_field = compat.max_tokens_field.as_deref();
    let (max_tokens, max_completion_tokens) = match max_tokens_field {
        Some("max_tokens") => (clamped_max_tokens, None),
        _ => (None, clamped_max_tokens),
    };

    // P1-4: Conditional stream_options based on compat
    let stream_options = if compat.supports_usage_in_streaming {
        Some(StreamOptionsConfig {
            include_usage: true,
        })
    } else {
        None
    };

    // P1-1: store = false when provider supports it (privacy)
    let store = if compat.supports_store {
        Some(false)
    } else {
        None
    };

    // P0-1: Resolve reasoning / thinking parameters based on compat.thinking_format
    let mut reasoning_effort = None;
    let mut enable_thinking = None;
    let mut chat_template_kwargs = None;

    if model.reasoning {
        let has_thinking = thinking_options
            .as_ref()
            .and_then(|t| t.reasoning_effort.as_ref())
            .is_some();

        match compat.thinking_format.as_str() {
            "zai" => {
                // ZAI uses top-level enable_thinking: bool
                enable_thinking = Some(has_thinking);
            }
            "qwen" => {
                // Qwen uses top-level enable_thinking: bool
                enable_thinking = Some(has_thinking);
            }
            "qwen-chat-template" => {
                // Qwen chat template uses chat_template_kwargs.enable_thinking
                chat_template_kwargs = Some(serde_json::json!({
                    "enable_thinking": has_thinking
                }));
            }
            _ => {
                // Standard OpenAI format: use reasoning_effort
                if compat.supports_reasoning_effort {
                    if let Some(ref thinking_opts) = thinking_options {
                        if let Some(ref effort) = thinking_opts.reasoning_effort {
                            reasoning_effort = Some(resolve_reasoning_effort(effort, &compat));
                        }
                    }
                }
            }
        }
    }

    // P2-1: OpenRouter routing preferences
    let provider_routing = compat.open_router_routing.clone();

    let request = ChatCompletionRequest {
        model: model.id.clone(),
        messages,
        stream: true,
        temperature: options.temperature,
        max_tokens,
        max_completion_tokens,
        tools,
        tool_choice: convert_tool_choice(options.tool_choice.as_ref()),
        stream_options,
        reasoning_effort,
        store,
        enable_thinking,
        chat_template_kwargs,
        provider: provider_routing,
    };

    // Apply on_payload hook if set
    let body_string = super::common::apply_on_payload(&request, &options.on_payload, model).await?;

    let base = super::common::resolve_base_url(
        options.base_url.as_deref(),
        model.base_url.as_deref(),
        DEFAULT_BASE_URL,
    );
    let url = format!("{}/chat/completions", base);

    // H1: Validate base URL against security policy
    if !super::common::validate_url_or_error(base, &limits, &mut output, &stream) {
        return Ok(());
    }

    tracing::info!(
        url = %url,
        model = %model.id,
        provider = %model.provider,
        message_count = request.messages.len(),
        has_tools = request.tools.is_some(),
        "Sending OpenAI Completions request"
    );
    tracing::debug!(request_body = %super::common::debug_preview(&body_string, 500), "Request payload");

    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::AUTHORIZATION,
        format!("Bearer {}", api_key).parse()?,
    );
    headers.insert(reqwest::header::CONTENT_TYPE, "application/json".parse()?);

    // Add custom headers (H2: skip protected headers)
    super::common::apply_custom_headers(&mut headers, &options.headers, &limits.headers);

    let max_retries = options
        .max_retries
        .unwrap_or(super::common::DEFAULT_MAX_RETRIES);
    let max_retry_delay_ms = options
        .max_retry_delay_ms
        .unwrap_or(super::common::DEFAULT_MAX_RETRY_DELAY_MS);
    let request_headers = headers.clone();
    let request_body = body_string.clone();
    let Some(response) = super::common::send_request_with_retry(
        &client,
        &url,
        headers,
        body_string,
        limits.http.request_timeout(),
        max_retries,
        max_retry_delay_ms,
        cancel_token.as_ref(),
        &mut output,
        &stream,
    )
    .await?
    else {
        return Ok(());
    };

    if !response.status().is_success() {
        super::common::handle_error_response(
            response,
            &url,
            model,
            &limits,
            &mut output,
            &stream,
            "OpenAI Completions",
            &request_body,
        )
        .await;
        return Ok(());
    }

    // Send start event
    stream.push(AssistantMessageEvent::Start {
        partial: output.clone(),
    });
    let initial_output = output.clone();
    let mut emitted_semantic_event = false;
    let mut prelude_retry_attempt = 0;

    let mut current_block: Option<ContentBlock> = None;
    let mut partial_tool_args: HashMap<u32, String> = HashMap::new();
    let mut current_tool_index: Option<u32> = None;
    let mut line_buffer = String::new(); // Buffer for incomplete SSE lines
    let mut saw_finish_reason = false;
    let mut saw_done_sentinel = false;

    let finish_current_block = |output: &mut AssistantMessage,
                                stream: &AssistantMessageEventStream,
                                block: Option<ContentBlock>| {
        if let Some(block) = block {
            let content_index = output.content.len();
            match block {
                ContentBlock::Text(text) => {
                    let content = text.text.clone();
                    output.content.push(ContentBlock::Text(text));
                    stream.push(AssistantMessageEvent::TextEnd {
                        content_index,
                        content,
                        partial: output.clone(),
                    });
                }
                ContentBlock::Thinking(thinking) => {
                    let content = thinking.thinking.clone();
                    output.content.push(ContentBlock::Thinking(thinking));
                    stream.push(AssistantMessageEvent::ThinkingEnd {
                        content_index,
                        content,
                        partial: output.clone(),
                    });
                }
                ContentBlock::ToolCall(tool_call) => {
                    let tool_call_clone = tool_call.clone();
                    output.content.push(ContentBlock::ToolCall(tool_call));
                    stream.push(AssistantMessageEvent::ToolCallEnd {
                        content_index,
                        tool_call: tool_call_clone,
                        partial: output.clone(),
                    });
                }
                ContentBlock::Image(image) => {
                    output.content.push(ContentBlock::Image(image));
                }
            }
        }
    };

    let mut byte_stream = response.bytes_stream();
    while let Some(chunk_result) = super::common::next_stream_item_with_cancel(
        &mut byte_stream,
        cancel_token.as_ref(),
        &mut output,
        &stream,
    )
    .await
    {
        let chunk = match chunk_result {
            Ok(chunk) => chunk,
            Err(err)
                if !emitted_semantic_event
                    && prelude_retry_attempt < max_retries
                    && super::common::is_retryable_stream_error(&err) =>
            {
                let delay =
                    super::common::compute_retry_delay(prelude_retry_attempt, max_retry_delay_ms);
                tracing::warn!(
                    url = %url,
                    error = %err,
                    attempt = prelude_retry_attempt + 1,
                    max_retries = max_retries,
                    delay_ms = delay.as_millis() as u64,
                    "Retryable OpenAI Completions stream error before first semantic event, retrying request"
                );
                if super::common::sleep_with_cancel(delay, cancel_token.as_ref()).await {
                    super::common::emit_aborted(&mut output, &stream);
                    return Ok(());
                }
                prelude_retry_attempt += 1;
                output = initial_output.clone();
                current_block = None;
                partial_tool_args.clear();
                current_tool_index = None;
                line_buffer.clear();
                saw_finish_reason = false;
                saw_done_sentinel = false;

                let Some(response) = super::common::send_request_with_retry(
                    &client,
                    &url,
                    request_headers.clone(),
                    request_body.clone(),
                    limits.http.request_timeout(),
                    max_retries,
                    max_retry_delay_ms,
                    cancel_token.as_ref(),
                    &mut output,
                    &stream,
                )
                .await?
                else {
                    return Ok(());
                };

                if !response.status().is_success() {
                    super::common::handle_error_response(
                        response,
                        &url,
                        model,
                        &limits,
                        &mut output,
                        &stream,
                        "OpenAI Completions",
                        &request_body,
                    )
                    .await;
                    return Ok(());
                }

                byte_stream = response.bytes_stream();
                continue;
            }
            Err(err) => {
                // Close any open content block before emitting the error
                finish_current_block(&mut output, &stream, current_block.take());
                super::common::emit_terminal_error(
                    &mut output,
                    format!("OpenAI Completions stream transport error: {}", err),
                    limits.http.max_error_message_chars,
                    &stream,
                );
                return Ok(());
            }
        };
        let text = String::from_utf8_lossy(&chunk);
        line_buffer.push_str(&text);

        // C2: Check SSE line buffer limit
        if super::common::check_sse_buffer_overflow(
            line_buffer.len(),
            limits.http.max_sse_line_buffer_bytes,
            &mut output,
            &stream,
        ) {
            return Ok(());
        }

        // Process only complete lines (ending with \n), keep partial line in buffer
        while let Some(newline_pos) = line_buffer.find('\n') {
            let line = line_buffer[..newline_pos]
                .trim_end_matches('\r')
                .to_string();
            line_buffer = line_buffer[newline_pos + 1..].to_string();

            if !line.starts_with("data: ") {
                continue;
            }

            let data = &line[6..];
            if data == "[DONE]" {
                saw_done_sentinel = true;
                continue;
            }

            let parsed: Result<ChatCompletionChunk, _> = serde_json::from_str(data);
            if let Ok(chunk_data) = parsed {
                tracing::trace!(
                    choices = chunk_data.choices.len(),
                    raw = %super::common::debug_preview(data, 300),
                    "SSE chunk parsed"
                );
                if let Some(chunk_id) = &chunk_data.id {
                    output.response_id = Some(chunk_id.clone());
                }

                // Handle usage
                if let Some(usage) = &chunk_data.usage {
                    apply_openai_usage(&mut output, usage);
                }

                for choice in &chunk_data.choices {
                    // Handle finish reason
                    if let Some(ref reason) = choice.finish_reason {
                        saw_finish_reason = true;
                        let (stop_reason, error_message) = map_finish_reason(reason);
                        output.stop_reason = stop_reason;
                        if let Some(error_message) = error_message {
                            output.error_message = Some(error_message);
                        }
                    }

                    // Handle usage in choice (fallback for some providers)
                    if let Some(usage) = &choice.usage {
                        apply_openai_usage(&mut output, usage);
                    }

                    if let Some(ref delta) = choice.delta {
                        // Handle text content
                        if let Some(ref content) = delta.content {
                            if !content.is_empty() {
                                if current_block.as_ref().is_none_or(|b| !b.is_text()) {
                                    finish_current_block(
                                        &mut output,
                                        &stream,
                                        current_block.take(),
                                    );
                                    current_block = Some(ContentBlock::Text(TextContent::new("")));
                                    emitted_semantic_event = true;
                                    stream.push(AssistantMessageEvent::TextStart {
                                        content_index: output.content.len(),
                                        partial: output.clone(),
                                    });
                                }

                                if let Some(ContentBlock::Text(ref mut text_block)) = current_block
                                {
                                    text_block.text.push_str(content);
                                    emitted_semantic_event = true;
                                    stream.push(AssistantMessageEvent::TextDelta {
                                        content_index: output.content.len(),
                                        delta: content.clone(),
                                        partial: output.clone(),
                                    });
                                }
                            }
                        }

                        // Handle reasoning/thinking
                        let reasoning = delta
                            .reasoning_content
                            .as_ref()
                            .map(|content| (content, "reasoning_content"))
                            .or_else(|| {
                                delta
                                    .reasoning
                                    .as_ref()
                                    .map(|content| (content, "reasoning"))
                            })
                            .or_else(|| {
                                delta
                                    .reasoning_text
                                    .as_ref()
                                    .map(|content| (content, "reasoning_text"))
                            });

                        if let Some((content, _source_field)) = reasoning {
                            if !content.is_empty() {
                                if current_block.as_ref().is_none_or(|b| !b.is_thinking()) {
                                    finish_current_block(
                                        &mut output,
                                        &stream,
                                        current_block.take(),
                                    );
                                    current_block =
                                        Some(ContentBlock::Thinking(ThinkingContent::new("")));
                                    emitted_semantic_event = true;
                                    stream.push(AssistantMessageEvent::ThinkingStart {
                                        content_index: output.content.len(),
                                        partial: output.clone(),
                                    });
                                }

                                if let Some(ContentBlock::Thinking(ref mut thinking_block)) =
                                    current_block
                                {
                                    thinking_block.thinking.push_str(content);
                                    emitted_semantic_event = true;
                                    stream.push(AssistantMessageEvent::ThinkingDelta {
                                        content_index: output.content.len(),
                                        delta: content.clone(),
                                        partial: output.clone(),
                                    });
                                }
                            }
                        }

                        // Handle tool calls
                        for tc in &delta.tool_calls {
                            let index = tc.index.unwrap_or(0);

                            let is_new = (current_tool_index != Some(index))
                                || current_block.as_ref().is_none_or(|b| !b.is_tool_call());

                            if is_new {
                                // Finish previous block
                                finish_current_block(&mut output, &stream, current_block.take());

                                let id = tc.id.clone().unwrap_or_default();
                                let name = tc
                                    .function
                                    .as_ref()
                                    .and_then(|f| f.name.clone())
                                    .unwrap_or_default();

                                current_block = Some(ContentBlock::ToolCall(ToolCall::new(
                                    id,
                                    name,
                                    serde_json::Value::Object(serde_json::Map::new()),
                                )));
                                current_tool_index = Some(index);

                                partial_tool_args.insert(index, String::new());

                                emitted_semantic_event = true;
                                stream.push(AssistantMessageEvent::ToolCallStart {
                                    content_index: output.content.len(),
                                    partial: output.clone(),
                                });
                            }

                            if let Some(ContentBlock::ToolCall(ref mut tool_call)) = current_block {
                                if let Some(ref id) = tc.id {
                                    if !id.is_empty() {
                                        tool_call.id = id.clone();
                                    }
                                }
                                if let Some(ref func) = tc.function {
                                    if let Some(ref name) = func.name {
                                        if !name.is_empty() {
                                            tool_call.name = name.clone();
                                        }
                                    }
                                    if let Some(ref args) = func.arguments {
                                        let partial = partial_tool_args.entry(index).or_default();
                                        partial.push_str(args);
                                        tool_call.arguments = parse_streaming_json(partial);

                                        emitted_semantic_event = true;
                                        stream.push(AssistantMessageEvent::ToolCallDelta {
                                            content_index: output.content.len(),
                                            delta: args.clone(),
                                            partial: output.clone(),
                                        });
                                    }
                                }
                            }
                        }

                        if let Some(reasoning_details) = &delta.reasoning_details {
                            for detail in reasoning_details {
                                let detail_id = detail
                                    .get("id")
                                    .and_then(|id| id.as_str())
                                    .unwrap_or_default();
                                let detail_type = detail
                                    .get("type")
                                    .and_then(|kind| kind.as_str())
                                    .unwrap_or_default();
                                let has_data = detail.get("data").is_some();

                                if detail_type != "reasoning.encrypted"
                                    || detail_id.is_empty()
                                    || !has_data
                                {
                                    continue;
                                }

                                let detail_json = detail.to_string();

                                if let Some(ContentBlock::ToolCall(ref mut tool_call)) =
                                    current_block.as_mut()
                                {
                                    if tool_call.id == detail_id {
                                        tool_call.thought_signature = Some(detail_json.clone());
                                        continue;
                                    }
                                }

                                for block in &mut output.content {
                                    if let ContentBlock::ToolCall(tc) = block {
                                        if tc.id == detail_id {
                                            tc.thought_signature = Some(detail_json.clone());
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                tracing::warn!(
                    raw = %super::common::debug_preview(data, 500),
                    "SSE chunk JSON parse failed"
                );
            }
        }
    }

    let incomplete_detail = incomplete_openai_completions_stream_detail(
        saw_finish_reason,
        saw_done_sentinel,
        &partial_tool_args,
        &line_buffer,
    );

    // Finish current block
    finish_current_block(&mut output, &stream, current_block.take());

    if let Some(detail) = incomplete_detail {
        tracing::error!(
            url = %url,
            model = %model.id,
            detail = %detail,
            "OpenAI Completions stream ended before protocol completion"
        );
        super::common::emit_incomplete_stream_error(
            &mut output,
            "openai_completions",
            detail,
            limits.http.max_error_message_chars,
            &stream,
        );
        return Ok(());
    }

    // Tolerate missing finish_reason when [DONE] sentinel was received.
    // Many OpenAI-compatible providers omit finish_reason; the [DONE] sentinel
    // alone is sufficient evidence the stream completed normally.
    // When the response contains tool calls, infer ToolUse so the agent loop
    // continues to execute them (mirrors AI SDK behaviour where loop continuation
    // is driven by the presence of tool calls, not by finish_reason).
    if saw_done_sentinel && !saw_finish_reason {
        if output.has_tool_calls() {
            output.stop_reason = StopReason::ToolUse;
            tracing::warn!(
                url = %url,
                model = %model.id,
                "Provider omitted finish_reason with tool calls present; inferring ToolUse"
            );
        } else {
            tracing::warn!(
                url = %url,
                model = %model.id,
                "Provider omitted finish_reason but sent [DONE] sentinel; treating as normal stop"
            );
        }
    }

    tracing::debug!(
        url = %url,
        model = %model.id,
        stop_reason = ?output.stop_reason,
        content_blocks = output.content.len(),
        has_tool_calls = output.has_tool_calls(),
        content_summary = %output.content.iter().map(|b| match b {
            ContentBlock::Text(t) => format!("Text({}chars)", t.text.len()),
            ContentBlock::Thinking(t) => format!("Thinking({}chars)", t.thinking.len()),
            ContentBlock::ToolCall(tc) => format!("ToolCall(id={}, name={})", tc.id, tc.name),
            ContentBlock::Image(_) => "Image".to_string(),
        }).collect::<Vec<_>>().join(", "),
        "OpenAI Completions stream final output summary"
    );

    if output.stop_reason == StopReason::Error {
        if output.error_message.is_none() {
            output.error_message = Some("Provider returned an error stop reason".to_string());
        }
        stream.push(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: output,
        });
    } else {
        stream.push(AssistantMessageEvent::Done {
            reason: output.stop_reason,
            message: output,
        });
    }
    stream.end(None);

    Ok(())
}

/// Deserialize a field that may be `null` in JSON as the type's `Default` value.
/// Handles providers that send `"tool_calls": null` instead of omitting the field.
fn deserialize_null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + serde::Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

fn map_finish_reason(reason: &str) -> (StopReason, Option<String>) {
    match reason {
        "stop" | "end" => (StopReason::Stop, None),
        "length" => (StopReason::Length, None),
        "tool_calls" | "function_call" => (StopReason::ToolUse, None),
        "content_filter" | "network_error" => (
            StopReason::Error,
            Some(format!("Provider finish_reason: {}", reason)),
        ),
        other => (
            StopReason::Error,
            Some(format!("Provider finish_reason: {}", other)),
        ),
    }
}

fn incomplete_openai_completions_stream_detail(
    saw_finish_reason: bool,
    saw_done_sentinel: bool,
    partial_tool_args: &HashMap<u32, String>,
    line_buffer: &str,
) -> Option<String> {
    let mut reasons = Vec::new();

    if !saw_finish_reason && !saw_done_sentinel {
        reasons.push("missing finish_reason".to_string());
    }

    if !saw_done_sentinel {
        reasons.push("missing [DONE] sentinel".to_string());
    }

    let mut incomplete_tool_indexes: Vec<_> = partial_tool_args
        .iter()
        .filter_map(|(index, args)| {
            let trimmed = args.trim();
            (!trimmed.is_empty() && serde_json::from_str::<serde_json::Value>(trimmed).is_err())
                .then_some(*index)
        })
        .collect();
    incomplete_tool_indexes.sort_unstable();
    if !incomplete_tool_indexes.is_empty() {
        reasons.push(format!(
            "unfinished tool input JSON at indices [{}]",
            incomplete_tool_indexes
                .iter()
                .map(|index| index.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    if !line_buffer.trim().is_empty() {
        reasons.push("trailing partial SSE frame".to_string());
    }

    if reasons.is_empty() {
        None
    } else {
        Some(reasons.join("; "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_messages_basic() {
        let mut context = Context::with_system_prompt("You are helpful.");
        context.add_message(Message::User(UserMessage::text("Hello")));

        let model = Model::builder()
            .id("gpt-4o-mini")
            .name("GPT-4o Mini")
            .api(Api::OpenAICompletions)
            .provider(Provider::OpenAI)
            .base_url("https://api.openai.com/v1")
            .context_window(128000)
            .max_tokens(16384)
            .build()
            .unwrap();

        let messages = convert_messages(&context, &model, false);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
    }

    #[test]
    fn test_resolve_reasoning_effort_with_map() {
        let mut compat = OpenAICompletionsCompat::default();
        compat
            .reasoning_effort_map
            .insert("high".to_string(), "default".to_string());
        compat
            .reasoning_effort_map
            .insert("minimal".to_string(), "default".to_string());

        assert_eq!(resolve_reasoning_effort("high", &compat), "default");
        assert_eq!(resolve_reasoning_effort("minimal", &compat), "default");
        assert_eq!(resolve_reasoning_effort("medium", &compat), "medium"); // no mapping
    }

    #[test]
    fn test_store_and_strict_mode() {
        // Provider that supports store and strict
        let compat = OpenAICompletionsCompat::default();
        assert!(compat.supports_store);
        assert!(compat.supports_strict_mode);

        // Provider that doesn't support store/strict
        let compat = OpenAICompletionsCompat {
            supports_store: false,
            supports_strict_mode: false,
            ..Default::default()
        };

        let tools = vec![Tool {
            name: "test".to_string(),
            description: "A test tool".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        }];

        let converted = convert_tools(&tools, &compat);
        assert!(converted[0].function.strict.is_none());

        let compat_with_strict = OpenAICompletionsCompat::default();
        let converted = convert_tools(&tools, &compat_with_strict);
        assert_eq!(converted[0].function.strict, Some(false));
    }

    #[test]
    fn test_detect_compat_for_openai_compatible_chutes() {
        let model = Model::builder()
            .id("foo")
            .name("foo")
            .api(Api::OpenAICompletions)
            .provider(Provider::OpenAICompatible)
            .base_url("https://api.chutes.ai/v1")
            .context_window(128000)
            .max_tokens(8192)
            .build()
            .unwrap();

        let compat = resolve_compat(&model);
        assert!(!compat.supports_store);
        assert!(!compat.supports_developer_role);
        assert_eq!(compat.max_tokens_field.as_deref(), Some("max_tokens"));
    }

    #[test]
    fn test_has_tool_history_detects_assistant_tool_calls() {
        let assistant = AssistantMessage::builder()
            .api(Api::OpenAICompletions)
            .provider(Provider::OpenAI)
            .model("mock-model")
            .content(vec![ContentBlock::ToolCall(ToolCall::new(
                "call_1",
                "lookup",
                serde_json::json!({}),
            ))])
            .stop_reason(StopReason::ToolUse)
            .build()
            .unwrap();

        assert!(has_tool_history(&[Message::Assistant(assistant)]));
    }

    #[test]
    fn test_map_finish_reason_network_error_is_error() {
        let (reason, error_message) = map_finish_reason("network_error");
        assert_eq!(reason, StopReason::Error);
        assert_eq!(
            error_message.as_deref(),
            Some("Provider finish_reason: network_error")
        );
    }

    #[test]
    fn test_thinking_as_text_in_assistant_message() {
        let compat = OpenAICompletionsCompat {
            requires_thinking_as_text: true,
            ..Default::default()
        };

        let msg = AssistantMessage::builder()
            .api(Api::OpenAICompletions)
            .provider(Provider::OpenAI)
            .model("test")
            .stop_reason(StopReason::Stop)
            .usage(Usage::default())
            .content(vec![
                ContentBlock::Thinking(ThinkingContent::new("My reasoning")),
                ContentBlock::Text(TextContent::new("My answer")),
            ])
            .build()
            .unwrap();

        let model = Model::builder()
            .id("test")
            .name("Test")
            .api(Api::OpenAICompletions)
            .provider(Provider::OpenAI)
            .context_window(128000)
            .max_tokens(16384)
            .build()
            .unwrap();

        let converted = convert_assistant_message(&msg, &model, &compat).unwrap();
        if let Some(OpenAIContent::Text(text)) = &converted.content {
            assert_eq!(text, "My reasoning\n\nMy answer");
        } else {
            panic!("Expected text content");
        }
        assert!(converted.extra_fields.is_empty());
    }

    #[test]
    fn test_reasoning_content_in_assistant_message() {
        let compat = OpenAICompletionsCompat::default();

        let msg = AssistantMessage::builder()
            .api(Api::OpenAICompletions)
            .provider(Provider::OpenAI)
            .model("test")
            .stop_reason(StopReason::Stop)
            .usage(Usage::default())
            .content(vec![
                ContentBlock::Thinking(ThinkingContent::new("My reasoning")),
                ContentBlock::Text(TextContent::new("My answer")),
            ])
            .build()
            .unwrap();

        let model = Model::builder()
            .id("test")
            .name("Test")
            .api(Api::OpenAICompletions)
            .provider(Provider::OpenAI)
            .context_window(128000)
            .max_tokens(16384)
            .build()
            .unwrap();

        let converted = convert_assistant_message(&msg, &model, &compat).unwrap();
        assert!(converted.extra_fields.contains_key("reasoning_content"));
        assert_eq!(
            converted.extra_fields["reasoning_content"],
            serde_json::Value::String("My reasoning".to_string())
        );
    }

    #[test]
    fn test_incomplete_openai_completions_stream_detail_reports_missing_closure() {
        let mut partial_tool_args = HashMap::new();
        partial_tool_args.insert(1, "{\"path\":\"logs".to_string());

        let detail = incomplete_openai_completions_stream_detail(
            false,
            false,
            &partial_tool_args,
            "data: {",
        )
        .expect("detail");

        assert!(detail.contains("missing finish_reason"));
        assert!(detail.contains("missing [DONE] sentinel"));
        assert!(detail.contains("unfinished tool input JSON at indices [1]"));
        assert!(detail.contains("trailing partial SSE frame"));
    }

    #[test]
    fn test_incomplete_stream_detail_tolerates_missing_finish_reason_when_done_received() {
        // When [DONE] sentinel is received but finish_reason is missing,
        // this should NOT be considered an incomplete stream.
        let partial_tool_args = HashMap::new();

        let detail = incomplete_openai_completions_stream_detail(
            false, // no finish_reason
            true,  // [DONE] received
            &partial_tool_args,
            "",
        );

        assert!(
            detail.is_none(),
            "Expected None when [DONE] was received without finish_reason, got: {:?}",
            detail
        );
    }

    #[test]
    fn test_incomplete_stream_detail_still_reports_when_both_missing() {
        // When both [DONE] and finish_reason are missing, it IS incomplete.
        let partial_tool_args = HashMap::new();

        let detail = incomplete_openai_completions_stream_detail(
            false, // no finish_reason
            false, // no [DONE]
            &partial_tool_args,
            "",
        )
        .expect("should report incomplete");

        assert!(detail.contains("missing finish_reason"));
        assert!(detail.contains("missing [DONE] sentinel"));
    }

    // ========================================================================
    // normalize_reasoning_content tests (in super::super::common)
    // ========================================================================

    /// Build an assistant Message with the given content blocks.
    fn assistant_msg(content: Vec<ContentBlock>) -> Message {
        Message::Assistant(AssistantMessage {
            role: crate::types::Role::Assistant,
            content,
            api: Api::OpenAICompletions,
            provider: Provider::OpenAI,
            model: "test".to_string(),
            stop_reason: crate::types::StopReason::Stop,
            usage: crate::types::Usage::default(),
            error_message: None,
            response_id: None,
            timestamp: 0,
        })
    }

    /// Build a simple user Message.
    fn user_msg(text: &str) -> Message {
        Message::User(UserMessage::text(text.to_string()))
    }

    #[test]
    fn test_normalize_passthrough_for_non_constrained_provider() {
        let compat = OpenAICompletionsCompat::default(); // reasoning_content_constrained = false
        let messages = vec![
            user_msg("Hello"),
            assistant_msg(vec![
                ContentBlock::Thinking(ThinkingContent::new("thinking...")),
                ContentBlock::Text(TextContent::new("Hi!")),
            ]),
        ];

        let result = super::super::common::normalize_reasoning_content(
            messages.clone(),
            compat.reasoning_content_constrained,
            true,
            "",
        );
        assert_eq!(
            result, messages,
            "should pass through unchanged for non-constrained"
        );
    }

    #[test]
    fn test_normalize_thinking_enabled_backfills_missing_reasoning() {
        let compat = OpenAICompletionsCompat {
            reasoning_content_constrained: true,
            ..Default::default()
        };

        let thinking_block = ThinkingContent::new("previous thinking");

        let messages = vec![
            user_msg("Hello"),
            assistant_msg(vec![
                ContentBlock::Thinking(thinking_block.clone()),
                ContentBlock::Text(TextContent::new("Response 1")),
            ]),
            assistant_msg(vec![
                // No thinking block — should be backfilled
                ContentBlock::Text(TextContent::new("Response 2")),
            ]),
        ];

        let result = super::super::common::normalize_reasoning_content(
            messages,
            compat.reasoning_content_constrained,
            true,
            "",
        );

        // Second assistant should now have a thinking block
        if let Message::Assistant(ref msg) = result[2] {
            let has_thinking = msg
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::Thinking(_)));
            assert!(
                has_thinking,
                "second assistant should have backfilled thinking"
            );
        } else {
            panic!("expected assistant message");
        }
    }

    #[test]
    fn test_normalize_thinking_enabled_ensures_content_not_null() {
        let compat = OpenAICompletionsCompat {
            reasoning_content_constrained: true,
            ..Default::default()
        };

        // Assistant with thinking but no text — should get empty text block
        let messages = vec![
            user_msg("Hello"),
            assistant_msg(vec![ContentBlock::Thinking(ThinkingContent::new(
                "reasoning",
            ))]),
        ];

        let result = super::super::common::normalize_reasoning_content(
            messages,
            compat.reasoning_content_constrained,
            true,
            "",
        );

        if let Message::Assistant(ref msg) = result[1] {
            let has_text = msg
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::Text(_)));
            assert!(has_text, "should have inserted empty text block");
            // The empty text block should be empty string
            for block in &msg.content {
                if let ContentBlock::Text(t) = block {
                    assert!(t.text.is_empty(), "inserted text block should be empty");
                }
            }
        } else {
            panic!("expected assistant message");
        }
    }

    #[test]
    fn test_normalize_thinking_disabled_strips_all_thinking() {
        let compat = OpenAICompletionsCompat {
            reasoning_content_constrained: true,
            ..Default::default()
        };

        let messages = vec![
            user_msg("Hello"),
            assistant_msg(vec![
                ContentBlock::Thinking(ThinkingContent::new("should be stripped")),
                ContentBlock::Text(TextContent::new("Response")),
            ]),
        ];

        let result = super::super::common::normalize_reasoning_content(
            messages,
            compat.reasoning_content_constrained,
            false,
            "",
        );

        if let Message::Assistant(ref msg) = result[1] {
            let has_thinking = msg
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::Thinking(_)));
            assert!(!has_thinking, "all thinking blocks should be stripped");
            let has_text = msg
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::Text(_)));
            assert!(has_text, "text content should be preserved");
        } else {
            panic!("expected assistant message");
        }
    }

    #[test]
    fn test_normalize_thinking_enabled_preserves_existing_reasoning() {
        let compat = OpenAICompletionsCompat {
            reasoning_content_constrained: true,
            ..Default::default()
        };

        let thinking1 = ThinkingContent::new("thinking 1");
        let thinking2 = ThinkingContent::new("thinking 2");

        let messages = vec![
            user_msg("Hello"),
            assistant_msg(vec![
                ContentBlock::Thinking(thinking1.clone()),
                ContentBlock::Text(TextContent::new("Response 1")),
            ]),
            assistant_msg(vec![
                ContentBlock::Thinking(thinking2.clone()),
                ContentBlock::Text(TextContent::new("Response 2")),
            ]),
        ];

        let result = super::super::common::normalize_reasoning_content(
            messages,
            compat.reasoning_content_constrained,
            true,
            "",
        );

        // Both assistants should still have their original thinking
        if let Message::Assistant(ref msg) = result[1] {
            let thinkings: Vec<_> = msg
                .content
                .iter()
                .filter_map(|b| {
                    if let ContentBlock::Thinking(t) = b {
                        Some(t.thinking.clone())
                    } else {
                        None
                    }
                })
                .collect();
            assert_eq!(
                thinkings,
                vec!["thinking 1".to_string()],
                "first thinking should be preserved exactly"
            );
        }
        if let Message::Assistant(ref msg) = result[2] {
            let thinkings: Vec<_> = msg
                .content
                .iter()
                .filter_map(|b| {
                    if let ContentBlock::Thinking(t) = b {
                        Some(t.thinking.clone())
                    } else {
                        None
                    }
                })
                .collect();
            assert_eq!(
                thinkings,
                vec!["thinking 2".to_string()],
                "second thinking should be preserved exactly"
            );
        }
    }

    #[test]
    fn test_normalize_base_url_heuristic_triggers_constrained() {
        // Even without the compat flag, api.deepseek.com in base_url triggers normalization
        let compat = OpenAICompletionsCompat::default(); // reasoning_content_constrained = false

        let messages = vec![
            user_msg("Hello"),
            assistant_msg(vec![
                ContentBlock::Thinking(ThinkingContent::new("thinking")),
                ContentBlock::Text(TextContent::new("Response")),
            ]),
            assistant_msg(vec![
                // No thinking — should be backfilled because base_url matches
                ContentBlock::Text(TextContent::new("Response 2")),
            ]),
        ];

        let result = super::super::common::normalize_reasoning_content(
            messages,
            compat.reasoning_content_constrained,
            true,
            "https://api.deepseek.com/v1",
        );

        if let Message::Assistant(ref msg) = result[2] {
            let has_thinking = msg
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::Thinking(_)));
            assert!(
                has_thinking,
                "should backfill when base_url matches api.deepseek.com"
            );
        }
    }

    #[test]
    fn test_normalize_non_assistant_messages_passthrough() {
        let compat = OpenAICompletionsCompat {
            reasoning_content_constrained: true,
            ..Default::default()
        };

        let user = user_msg("Hello");
        let tool_result = Message::ToolResult(ToolResultMessage {
            role: crate::types::Role::ToolResult,
            tool_call_id: "call_1".to_string(),
            tool_name: "test".to_string(),
            content: vec![ContentBlock::Text(TextContent::new("result"))],
            details: None::<serde_json::Value>,
            is_error: false,
            timestamp: 0,
        });

        let messages = vec![user.clone(), tool_result.clone()];

        let result = super::super::common::normalize_reasoning_content(
            messages,
            compat.reasoning_content_constrained,
            true,
            "",
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], user);
        // Verify it's still a ToolResult
        assert!(matches!(result[1], Message::ToolResult(_)));
    }

    #[test]
    fn test_normalize_model_id_alone_does_not_trigger_constrained() {
        // Third-party DeepSeek model IDs should not trigger normalization by name alone.
        // Callers must use the compat flag/catalog patch or a DeepSeek base URL.
        let compat = OpenAICompletionsCompat::default(); // reasoning_content_constrained = false

        let messages = vec![
            user_msg("Hello"),
            assistant_msg(vec![
                ContentBlock::Thinking(ThinkingContent::new("thinking")),
                ContentBlock::Text(TextContent::new("Response")),
            ]),
            assistant_msg(vec![ContentBlock::Text(TextContent::new("Response 2"))]),
        ];

        let result = super::super::common::normalize_reasoning_content(
            messages,
            compat.reasoning_content_constrained,
            true,
            "https://openrouter.ai/api/v1",
        );

        if let Message::Assistant(ref msg) = result[2] {
            let has_thinking = msg
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::Thinking(_)));
            assert!(
                !has_thinking,
                "should not backfill based on a DeepSeek-looking model ID alone"
            );
        } else {
            panic!("expected assistant message");
        }
    }

    #[test]
    fn test_normalize_explicit_constraint_triggers_for_third_party_provider() {
        let compat = OpenAICompletionsCompat {
            reasoning_content_constrained: true,
            ..Default::default()
        };

        let messages = vec![
            user_msg("Hello"),
            assistant_msg(vec![
                ContentBlock::Thinking(ThinkingContent::new("thinking")),
                ContentBlock::Text(TextContent::new("Response")),
            ]),
            assistant_msg(vec![ContentBlock::Text(TextContent::new("Response 2"))]),
        ];

        let result = super::super::common::normalize_reasoning_content(
            messages,
            compat.reasoning_content_constrained,
            true,
            "https://openrouter.ai/api/v1",
        );

        if let Message::Assistant(ref msg) = result[2] {
            let has_thinking = msg
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::Thinking(_)));
            assert!(
                has_thinking,
                "should backfill when explicit reasoning_content_constrained is true"
            );
        } else {
            panic!("expected assistant message");
        }
    }
}
