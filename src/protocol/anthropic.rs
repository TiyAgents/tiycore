//! Anthropic Messages API provider.
//!
//! Implements streaming via Anthropic's SSE protocol with events:
//! message_start → content_block_start → content_block_delta → content_block_stop → message_delta → message_stop

/// Default base URL for Anthropic Messages API.
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";

use crate::protocol::LLMProtocol;
use crate::stream::{parse_streaming_json, AssistantMessageEventStream};
use crate::thinking::ThinkingLevel;
use crate::transform::{normalize_tool_call_id, transform_messages};
use crate::types::*;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

const CLAUDE_CODE_VERSION: &str = "2.1.75";
const CLAUDE_CODE_IDENTITY: &str = "You are Claude Code, Anthropic's official CLI for Claude.";

const CLAUDE_CODE_TOOLS: &[&str] = &[
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
];

/// Anthropic Messages API provider.
pub struct AnthropicProtocol {
    client: Client,
    default_api_key: Option<String>,
}

impl AnthropicProtocol {
    /// Create a new Anthropic provider.
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

    /// Resolve API key from options, default, or environment.
    fn resolve_api_key(&self, options: &StreamOptions) -> String {
        if let Some(ref key) = options.api_key {
            return key.clone();
        }
        if let Some(ref key) = self.default_api_key {
            return key.clone();
        }
        std::env::var("ANTHROPIC_API_KEY").unwrap_or_default()
    }
}

impl Default for AnthropicProtocol {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProtocol for AnthropicProtocol {
    fn provider_type(&self) -> Provider {
        Provider::Anthropic
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
        let api_key = self.resolve_api_key(&options);
        let error_stream = stream_clone.clone();

        tokio::spawn(async move {
            if let Err(e) = run_stream(
                client,
                &model,
                &context,
                options,
                api_key,
                None,
                None,
                stream_clone,
            )
            .await
            {
                tracing::error!("Anthropic stream error: {}", e);
                super::common::emit_background_task_error(
                    &model,
                    Api::AnthropicMessages,
                    format!("Anthropic stream error: {}", e),
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
        let stream_options = options.base;
        let (thinking, output_config) = build_thinking_options(
            model,
            options.reasoning,
            options.thinking_budget_tokens,
            options.thinking_display,
        );
        let stream = AssistantMessageEventStream::new_assistant_stream();
        let stream_clone = stream.clone();

        let model = model.clone();
        let context = context.clone();
        let client = self.client.clone();
        let api_key = self.resolve_api_key(&stream_options);
        let error_stream = stream_clone.clone();

        tokio::spawn(async move {
            if let Err(e) = run_stream(
                client,
                &model,
                &context,
                stream_options,
                api_key,
                thinking,
                output_config,
                stream_clone,
            )
            .await
            {
                tracing::error!("Anthropic stream error: {}", e);
                super::common::emit_background_task_error(
                    &model,
                    Api::AnthropicMessages,
                    format!("Anthropic stream error: {}", e),
                    &error_stream,
                );
            }
        });

        stream
    }
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// Anthropic Messages request.
#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Vec<AnthropicSystemBlock>>,
    max_tokens: u32,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<AnthropicMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AnthropicThinkingParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<AnthropicOutputConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicCacheControl {
    #[serde(rename = "type")]
    control_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl: Option<String>,
}

#[derive(Debug, Serialize)]
struct AnthropicSystemBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<AnthropicCacheControl>,
}

#[derive(Debug, Serialize)]
struct AnthropicMetadata {
    user_id: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicToolChoice {
    Auto,
    Any,
    None,
    Tool { name: String },
}

/// Anthropic thinking parameter for the request.
#[derive(Debug, Serialize)]
#[serde(untagged)]
#[allow(dead_code)]
enum AnthropicThinkingParam {
    Adaptive {
        #[serde(rename = "type")]
        param_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        display: Option<String>,
    },
    Budget {
        #[serde(rename = "type")]
        param_type: String,
        budget_tokens: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        display: Option<String>,
    },
    Disabled {
        #[serde(rename = "type")]
        param_type: String,
    },
}

#[derive(Debug, Serialize)]
struct AnthropicOutputConfig {
    effort: String,
}

fn supports_xhigh(model: &Model) -> bool {
    super::common::supports_gpt5_xhigh(&model.id)
        || model.id.contains("opus-4-6")
        || model.id.contains("opus-4.6")
        || model.id.contains("opus-4-7")
        || model.id.contains("opus-4.7")
}

fn supports_adaptive_thinking(model_id: &str) -> bool {
    model_id.contains("opus-4-6")
        || model_id.contains("opus-4.6")
        || model_id.contains("sonnet-4-6")
        || model_id.contains("sonnet-4.6")
        || model_id.contains("opus-4-7")
        || model_id.contains("opus-4.7")
}

/// Whether the model supports the native "xhigh" effort level in output_config.
fn supports_xhigh_effort(model_id: &str) -> bool {
    model_id.contains("opus-4-7") || model_id.contains("opus-4.7")
}

/// Whether the model requires explicit display opt-in for thinking content.
fn requires_thinking_display(model_id: &str) -> bool {
    model_id.contains("opus-4-7") || model_id.contains("opus-4.7")
}

fn clamp_reasoning(level: ThinkingLevel, model: &Model) -> ThinkingLevel {
    if matches!(level, ThinkingLevel::XHigh) && !supports_xhigh(model) {
        ThinkingLevel::High
    } else {
        level
    }
}

fn map_adaptive_effort(level: ThinkingLevel, model_id: &str) -> &'static str {
    match level {
        ThinkingLevel::Minimal | ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::XHigh => {
            if supports_xhigh_effort(model_id) {
                "xhigh"
            } else if model_id.contains("opus-4-6") || model_id.contains("opus-4.6") {
                "max"
            } else {
                "high"
            }
        }
        ThinkingLevel::Off => "high",
    }
}

fn build_thinking_options(
    model: &Model,
    level: Option<ThinkingLevel>,
    thinking_budget_tokens: Option<u32>,
    thinking_display: Option<crate::thinking::ThinkingDisplay>,
) -> (
    Option<AnthropicThinkingParam>,
    Option<AnthropicOutputConfig>,
) {
    let Some(level) = level else {
        return (None, None);
    };

    if !model.reasoning {
        return (None, None);
    }

    if matches!(level, ThinkingLevel::Off) {
        return (
            Some(AnthropicThinkingParam::Disabled {
                param_type: "disabled".to_string(),
            }),
            None,
        );
    }

    let level = clamp_reasoning(level, model);
    let display = Some(
        thinking_display
            .unwrap_or(crate::thinking::ThinkingDisplay::Summarized)
            .to_string(),
    );
    if supports_adaptive_thinking(&model.id) {
        let adaptive_display = if requires_thinking_display(&model.id) {
            display
        } else {
            None
        };
        (
            Some(AnthropicThinkingParam::Adaptive {
                param_type: "adaptive".to_string(),
                display: adaptive_display,
            }),
            Some(AnthropicOutputConfig {
                effort: map_adaptive_effort(level, &model.id).to_string(),
            }),
        )
    } else {
        let budget_level = if matches!(level, ThinkingLevel::XHigh) {
            ThinkingLevel::High
        } else {
            level
        };
        (
            Some(AnthropicThinkingParam::Budget {
                param_type: "enabled".to_string(),
                budget_tokens: thinking_budget_tokens.unwrap_or_else(|| {
                    crate::thinking::ThinkingConfig::default_budget(budget_level)
                }),
                display,
            }),
            None,
        )
    }
}

/// Anthropic message format.
#[derive(Debug, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

/// Anthropic content can be text or an array of content blocks.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

/// Anthropic content block in a message.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    #[serde(rename = "image")]
    Image {
        source: AnthropicImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

/// Anthropic tool definition.
#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

// ============================================================================
// SSE Event Types
// ============================================================================

/// SSE event from Anthropic's streaming API.
#[derive(Debug, Deserialize)]
struct MessageStartData {
    message: MessageStartMessage,
}

#[derive(Debug, Deserialize)]
struct MessageStartMessage {
    #[allow(dead_code)]
    id: String,
    model: String,
    usage: Option<MessageUsage>,
}

#[derive(Debug, Deserialize)]
struct MessageUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct ContentBlockStartData {
    index: usize,
    content_block: ContentBlockInfo,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ContentBlockInfo {
    #[serde(rename = "text")]
    Text {
        #[allow(dead_code)]
        text: String,
    },
    #[serde(rename = "thinking")]
    Thinking {
        #[allow(dead_code)]
        thinking: String,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: serde_json::Value,
    },
}

#[derive(Debug, Deserialize)]
struct ContentBlockDeltaData {
    index: usize,
    delta: DeltaInfo,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum DeltaInfo {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
}

#[derive(Debug, Deserialize)]
struct ContentBlockStopData {
    index: usize,
}

#[derive(Debug, Deserialize)]
struct MessageDeltaData {
    delta: MessageDelta,
    usage: Option<MessageDeltaUsage>,
}

#[derive(Debug, Deserialize)]
struct MessageDelta {
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessageDeltaUsage {
    #[serde(default)]
    output_tokens: u64,
}

// ============================================================================
// Message Conversion
// ============================================================================

/// Convert context to Anthropic messages.
fn normalize_anthropic_tool_call_id(id: &str) -> String {
    normalize_tool_call_id(id, &Provider::Anthropic)
}

fn to_claude_code_name(name: &str) -> String {
    CLAUDE_CODE_TOOLS
        .iter()
        .find(|tool| tool.eq_ignore_ascii_case(name))
        .copied()
        .unwrap_or(name)
        .to_string()
}

fn from_claude_code_name(name: &str, tools: Option<&[Tool]>) -> String {
    tools
        .and_then(|tools| {
            tools
                .iter()
                .find(|tool| tool.name.eq_ignore_ascii_case(name))
                .map(|tool| tool.name.clone())
        })
        .unwrap_or_else(|| name.to_string())
}

fn is_oauth_token(api_key: &str) -> bool {
    api_key.contains("sk-ant-oat")
}

fn resolve_cache_retention(retention: Option<CacheRetention>) -> CacheRetention {
    if let Some(retention) = retention {
        return retention;
    }
    match std::env::var("TIY_CACHE_RETENTION").ok().as_deref() {
        Some("long") => CacheRetention::Long,
        Some("none") => CacheRetention::None,
        _ => CacheRetention::Short,
    }
}

fn get_cache_control(
    base_url: &str,
    retention: Option<CacheRetention>,
) -> Option<AnthropicCacheControl> {
    match resolve_cache_retention(retention) {
        CacheRetention::None => None,
        CacheRetention::Short => Some(AnthropicCacheControl {
            control_type: "ephemeral".to_string(),
            ttl: None,
        }),
        CacheRetention::Long => Some(AnthropicCacheControl {
            control_type: "ephemeral".to_string(),
            ttl: if base_url.contains("api.anthropic.com") {
                Some("1h".to_string())
            } else {
                None
            },
        }),
    }
}

fn convert_messages(
    context: &Context,
    target_model: &Model,
    cache_control: Option<&AnthropicCacheControl>,
    use_claude_code_names: bool,
    thinking_enabled: bool,
) -> Vec<AnthropicMessage> {
    let mut messages = Vec::new();
    let transformed = transform_messages(
        &context.messages,
        target_model,
        Some(&normalize_anthropic_tool_call_id),
    );

    // Normalize reasoning content for constrained providers (e.g. DeepSeek via third-party)
    let default_url = "";
    let base_url =
        super::common::resolve_base_url(None, target_model.base_url.as_deref(), default_url);
    let transformed = super::common::normalize_reasoning_content(
        transformed,
        false, // Anthropic protocol has no OpenAI compat — rely on explicit constraints only
        thinking_enabled,
        base_url,
    );

    for msg in &transformed {
        match msg {
            Message::User(user_msg) => {
                let content = match &user_msg.content {
                    UserContent::Text(text) => AnthropicContent::Text(text.clone()),
                    UserContent::Blocks(blocks) => {
                        let parts: Vec<AnthropicContentBlock> = blocks
                            .iter()
                            .filter_map(|b| match b {
                                ContentBlock::Text(t) => Some(AnthropicContentBlock::Text {
                                    text: t.text.clone(),
                                    cache_control: None,
                                }),
                                ContentBlock::Image(img) => {
                                    target_model.supports_image().then(|| {
                                        AnthropicContentBlock::Image {
                                            source: AnthropicImageSource {
                                                source_type: "base64".to_string(),
                                                media_type: img.mime_type.clone(),
                                                data: img.data.clone(),
                                            },
                                            cache_control: None,
                                        }
                                    })
                                }
                                _ => None,
                            })
                            .collect();
                        AnthropicContent::Blocks(parts)
                    }
                };
                messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content,
                });
            }
            Message::Assistant(assistant_msg) => {
                let mut blocks = Vec::new();

                for block in &assistant_msg.content {
                    match block {
                        ContentBlock::Text(t) => {
                            if !t.text.trim().is_empty() {
                                blocks.push(AnthropicContentBlock::Text {
                                    text: t.text.clone(),
                                    cache_control: None,
                                });
                            }
                        }
                        ContentBlock::Thinking(t) => {
                            if t.redacted {
                                if let Some(signature) = &t.thinking_signature {
                                    if !signature.trim().is_empty() {
                                        blocks.push(AnthropicContentBlock::RedactedThinking {
                                            data: signature.clone(),
                                        });
                                    }
                                } else if !t.thinking.trim().is_empty() {
                                    blocks.push(AnthropicContentBlock::Text {
                                        text: t.thinking.clone(),
                                        cache_control: None,
                                    });
                                }
                            } else if !t.thinking.trim().is_empty() {
                                if t.thinking_signature
                                    .as_ref()
                                    .is_some_and(|sig| !sig.trim().is_empty())
                                {
                                    blocks.push(AnthropicContentBlock::Thinking {
                                        thinking: t.thinking.clone(),
                                        signature: t.thinking_signature.clone(),
                                    });
                                } else {
                                    blocks.push(AnthropicContentBlock::Text {
                                        text: t.thinking.clone(),
                                        cache_control: None,
                                    });
                                }
                            }
                        }
                        ContentBlock::ToolCall(tc) => {
                            blocks.push(AnthropicContentBlock::ToolUse {
                                id: tc.id.clone(),
                                name: if use_claude_code_names {
                                    to_claude_code_name(&tc.name)
                                } else {
                                    tc.name.clone()
                                },
                                input: tc.arguments.clone(),
                            });
                        }
                        _ => {}
                    }
                }

                if !blocks.is_empty() {
                    messages.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content: AnthropicContent::Blocks(blocks),
                    });
                }
            }
            Message::ToolResult(tool_result) => {
                // Anthropic: tool results go in a user message with tool_result blocks
                let text: String = tool_result
                    .content
                    .iter()
                    .filter_map(|b| b.as_text())
                    .map(|t| t.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");

                let block = AnthropicContentBlock::ToolResult {
                    tool_use_id: tool_result.tool_call_id.clone(),
                    content: if text.is_empty() { None } else { Some(text) },
                    is_error: if tool_result.is_error {
                        Some(true)
                    } else {
                        None
                    },
                    cache_control: None,
                };

                // Check if the last message is a user message; merge if so
                if let Some(last) = messages.last_mut() {
                    if last.role == "user" {
                        match &mut last.content {
                            AnthropicContent::Blocks(ref mut blocks) => {
                                blocks.push(block);
                                continue;
                            }
                            AnthropicContent::Text(text) => {
                                let text_block = AnthropicContentBlock::Text {
                                    text: text.clone(),
                                    cache_control: None,
                                };
                                last.content = AnthropicContent::Blocks(vec![text_block, block]);
                                continue;
                            }
                        }
                    }
                }

                messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: AnthropicContent::Blocks(vec![block]),
                });
            }
        }
    }

    if let Some(cache_control) = cache_control {
        apply_cache_control(&mut messages, cache_control.clone());
    }

    messages
}

/// Convert tools to Anthropic format.
fn convert_tools(tools: &[Tool], use_claude_code_names: bool) -> Vec<AnthropicTool> {
    tools
        .iter()
        .map(|t| AnthropicTool {
            name: if use_claude_code_names {
                to_claude_code_name(&t.name)
            } else {
                t.name.clone()
            },
            description: t.description.clone(),
            input_schema: t.parameters.clone(),
        })
        .collect()
}

fn apply_cache_control(messages: &mut [AnthropicMessage], cache_control: AnthropicCacheControl) {
    let Some(last_message) = messages.last_mut() else {
        return;
    };
    if last_message.role != "user" {
        return;
    }

    match &mut last_message.content {
        AnthropicContent::Text(text) => {
            let text = text.clone();
            last_message.content = AnthropicContent::Blocks(vec![AnthropicContentBlock::Text {
                text,
                cache_control: Some(cache_control),
            }]);
        }
        AnthropicContent::Blocks(blocks) => {
            for block in blocks.iter_mut().rev() {
                match block {
                    AnthropicContentBlock::Text {
                        cache_control: slot,
                        ..
                    }
                    | AnthropicContentBlock::Image {
                        cache_control: slot,
                        ..
                    }
                    | AnthropicContentBlock::ToolResult {
                        cache_control: slot,
                        ..
                    } => {
                        *slot = Some(cache_control);
                        return;
                    }
                    _ => {}
                }
            }
        }
    }
}

fn build_system_blocks(
    context: &Context,
    cache_control: Option<&AnthropicCacheControl>,
    use_claude_code_identity: bool,
) -> Option<Vec<AnthropicSystemBlock>> {
    let mut blocks = Vec::new();
    if use_claude_code_identity {
        blocks.push(AnthropicSystemBlock {
            block_type: "text".to_string(),
            text: CLAUDE_CODE_IDENTITY.to_string(),
            cache_control: cache_control.cloned(),
        });
    }
    if let Some(system_prompt) = context.system_prompt.as_ref() {
        blocks.push(AnthropicSystemBlock {
            block_type: "text".to_string(),
            text: system_prompt.clone(),
            cache_control: cache_control.cloned(),
        });
    }
    if blocks.is_empty() {
        None
    } else {
        Some(blocks)
    }
}

fn build_anthropic_metadata(
    metadata: Option<&std::collections::HashMap<String, serde_json::Value>>,
) -> Option<AnthropicMetadata> {
    metadata
        .and_then(|metadata| metadata.get("user_id"))
        .and_then(|value| value.as_str())
        .map(|user_id| AnthropicMetadata {
            user_id: user_id.to_string(),
        })
}

fn build_tool_choice(tool_choice: Option<&ToolChoice>) -> Option<AnthropicToolChoice> {
    match tool_choice {
        Some(ToolChoice::Mode(ToolChoiceMode::Auto)) => Some(AnthropicToolChoice::Auto),
        Some(ToolChoice::Mode(ToolChoiceMode::Any | ToolChoiceMode::Required)) => {
            Some(AnthropicToolChoice::Any)
        }
        Some(ToolChoice::Mode(ToolChoiceMode::None)) => Some(AnthropicToolChoice::None),
        Some(ToolChoice::Named(ToolChoiceNamed::Tool { name })) => {
            Some(AnthropicToolChoice::Tool { name: name.clone() })
        }
        Some(ToolChoice::Named(ToolChoiceNamed::Function { function })) => {
            Some(AnthropicToolChoice::Tool {
                name: function.name.clone(),
            })
        }
        None => None,
    }
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
    thinking: Option<AnthropicThinkingParam>,
    output_config: Option<AnthropicOutputConfig>,
    stream: AssistantMessageEventStream,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let limits = options.security_config();
    let cancel_token = options.cancel_token.clone();
    let base = super::common::resolve_base_url(
        options.base_url.as_deref(),
        model.base_url.as_deref(),
        DEFAULT_BASE_URL,
    );
    let oauth_token = is_oauth_token(&api_key);
    let cache_control = get_cache_control(base, options.cache_retention);
    let needs_interleaved_beta = thinking.is_some() && output_config.is_none();

    let mut output = AssistantMessage::builder()
        .api(Api::AnthropicMessages)
        .provider(model.provider.clone())
        .model(model.id.clone())
        .stop_reason(StopReason::Stop)
        .usage(Usage::default())
        .build()?;

    let messages = convert_messages(
        context,
        model,
        cache_control.as_ref(),
        oauth_token,
        thinking.is_some(),
    );
    let tools = context
        .tools
        .as_ref()
        .map(|t| convert_tools(t, oauth_token));

    let request = AnthropicRequest {
        model: model.id.clone(),
        messages,
        system: build_system_blocks(context, cache_control.as_ref(), oauth_token),
        max_tokens: options.max_tokens.unwrap_or(model.max_tokens),
        stream: true,
        temperature: if thinking.is_some() {
            None
        } else {
            options.temperature
        },
        tools,
        metadata: build_anthropic_metadata(options.metadata.as_ref()),
        tool_choice: build_tool_choice(options.tool_choice.as_ref()),
        thinking,
        output_config,
    };

    // Apply on_payload hook if set
    let body_string = super::common::apply_on_payload(&request, &options.on_payload, model).await?;

    let url = format!("{}/messages", base);

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
        "Sending Anthropic Messages request"
    );
    tracing::debug!(request_body = %super::common::debug_preview(&body_string, 500), "Request payload");

    let mut headers = reqwest::header::HeaderMap::new();
    if oauth_token {
        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", api_key).parse()?,
        );
        headers.insert(
            reqwest::header::USER_AGENT,
            format!("claude-cli/{}", CLAUDE_CODE_VERSION).parse()?,
        );
        headers.insert("x-app", "cli".parse()?);
        headers.insert(
            "anthropic-beta",
            if needs_interleaved_beta {
                "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14,interleaved-thinking-2025-05-14"
            } else {
                "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14"
            }
                .parse()?,
        );
    } else {
        headers.insert("x-api-key", api_key.parse()?);
        headers.insert(
            "anthropic-beta",
            if needs_interleaved_beta {
                "fine-grained-tool-streaming-2025-05-14,interleaved-thinking-2025-05-14"
            } else {
                "fine-grained-tool-streaming-2025-05-14"
            }
            .parse()?,
        );
    }
    headers.insert("anthropic-version", "2023-06-01".parse()?);
    headers.insert(reqwest::header::CONTENT_TYPE, "application/json".parse()?);

    // Add custom headers
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
            "Anthropic Messages",
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

    // Track content blocks by index
    let mut block_types: Vec<BlockType> = Vec::new();
    let mut open_blocks: HashSet<usize> = HashSet::new();
    let mut partial_tool_args: HashMap<usize, String> = HashMap::new();
    let mut line_buffer = String::new();
    let mut current_event_type = String::new();
    let mut saw_message_delta = false;
    let mut saw_message_stop = false;

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
                    "Retryable Anthropic stream error before first semantic event, retrying request"
                );
                if super::common::sleep_with_cancel(delay, cancel_token.as_ref()).await {
                    super::common::emit_aborted(&mut output, &stream);
                    return Ok(());
                }
                prelude_retry_attempt += 1;
                output = initial_output.clone();
                block_types.clear();
                open_blocks.clear();
                partial_tool_args.clear();
                line_buffer.clear();
                current_event_type.clear();
                saw_message_delta = false;
                saw_message_stop = false;

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
                        "Anthropic Messages",
                        &request_body,
                    )
                    .await;
                    return Ok(());
                }

                byte_stream = response.bytes_stream();
                continue;
            }
            Err(err) => {
                // Close any open thinking/text blocks before emitting the error
                let (thinking_indices, text_indices) =
                    find_open_block_indices(&open_blocks, &block_types);
                super::common::emit_pending_block_ends_multi(
                    &stream,
                    &output,
                    thinking_indices,
                    text_indices,
                );
                super::common::emit_terminal_error(
                    &mut output,
                    format!("Anthropic stream transport error: {}", err),
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

        while let Some(newline_pos) = line_buffer.find('\n') {
            let line = line_buffer[..newline_pos]
                .trim_end_matches('\r')
                .to_string();
            line_buffer = line_buffer[newline_pos + 1..].to_string();

            if let Some(stripped) = line.strip_prefix("event: ") {
                current_event_type = stripped.to_string();
                continue;
            }

            if !line.starts_with("data: ") {
                continue;
            }

            let data = &line[6..];
            if data == "[DONE]" {
                continue;
            }

            match current_event_type.as_str() {
                "message_start" => {
                    if let Ok(msg_start) = serde_json::from_str::<MessageStartData>(data) {
                        output.model = msg_start.message.model;
                        output.response_id = Some(msg_start.message.id);
                        if let Some(usage) = msg_start.message.usage {
                            output.usage.input = usage.input_tokens;
                            output.usage.output = usage.output_tokens;
                            output.usage.cache_read = usage.cache_read_input_tokens;
                            output.usage.cache_write = usage.cache_creation_input_tokens;
                            output.usage.total_tokens = output.usage.input
                                + output.usage.output
                                + output.usage.cache_read
                                + output.usage.cache_write;
                        }
                    }
                }
                "content_block_start" => {
                    if let Ok(block_start) = serde_json::from_str::<ContentBlockStartData>(data) {
                        let idx = block_start.index;
                        match block_start.content_block {
                            ContentBlockInfo::Text { .. } => {
                                while block_types.len() <= idx {
                                    block_types.push(BlockType::Unknown);
                                }
                                block_types[idx] = BlockType::Text;
                                open_blocks.insert(idx);
                                output
                                    .content
                                    .push(ContentBlock::Text(TextContent::new("")));
                                emitted_semantic_event = true;
                                stream.push(AssistantMessageEvent::TextStart {
                                    content_index: idx,
                                    partial: output.clone(),
                                });
                            }
                            ContentBlockInfo::Thinking { .. } => {
                                while block_types.len() <= idx {
                                    block_types.push(BlockType::Unknown);
                                }
                                block_types[idx] = BlockType::Thinking;
                                open_blocks.insert(idx);
                                output
                                    .content
                                    .push(ContentBlock::Thinking(ThinkingContent::new("")));
                                emitted_semantic_event = true;
                                stream.push(AssistantMessageEvent::ThinkingStart {
                                    content_index: idx,
                                    partial: output.clone(),
                                });
                            }
                            ContentBlockInfo::RedactedThinking { data } => {
                                while block_types.len() <= idx {
                                    block_types.push(BlockType::Unknown);
                                }
                                block_types[idx] = BlockType::RedactedThinking;
                                open_blocks.insert(idx);
                                let mut thinking = ThinkingContent::new("[Reasoning redacted]");
                                thinking.redacted = true;
                                thinking.thinking_signature = Some(data);
                                output.content.push(ContentBlock::Thinking(thinking));
                            }
                            ContentBlockInfo::ToolUse { id, name, input } => {
                                while block_types.len() <= idx {
                                    block_types.push(BlockType::Unknown);
                                }
                                block_types[idx] = BlockType::ToolUse;
                                open_blocks.insert(idx);
                                let initial_args = input;
                                let initial_args_str = match &initial_args {
                                    serde_json::Value::Null => String::new(),
                                    serde_json::Value::Object(map) if map.is_empty() => {
                                        String::new()
                                    }
                                    _ => serde_json::to_string(&initial_args).unwrap_or_default(),
                                };
                                partial_tool_args.insert(idx, initial_args_str);
                                output.content.push(ContentBlock::ToolCall(ToolCall::new(
                                    id,
                                    if oauth_token {
                                        from_claude_code_name(&name, context.tools.as_deref())
                                    } else {
                                        name
                                    },
                                    initial_args,
                                )));
                                emitted_semantic_event = true;
                                stream.push(AssistantMessageEvent::ToolCallStart {
                                    content_index: idx,
                                    partial: output.clone(),
                                });
                            }
                        }
                    }
                }
                "content_block_delta" => {
                    if let Ok(delta_data) = serde_json::from_str::<ContentBlockDeltaData>(data) {
                        let idx = delta_data.index;
                        match delta_data.delta {
                            DeltaInfo::TextDelta { text } => {
                                if let Some(ContentBlock::Text(ref mut t)) =
                                    output.content.get_mut(idx)
                                {
                                    t.text.push_str(&text);
                                }
                                emitted_semantic_event = true;
                                stream.push(AssistantMessageEvent::TextDelta {
                                    content_index: idx,
                                    delta: text,
                                    partial: output.clone(),
                                });
                            }
                            DeltaInfo::ThinkingDelta { thinking } => {
                                if let Some(ContentBlock::Thinking(ref mut t)) =
                                    output.content.get_mut(idx)
                                {
                                    t.thinking.push_str(&thinking);
                                }
                                emitted_semantic_event = true;
                                stream.push(AssistantMessageEvent::ThinkingDelta {
                                    content_index: idx,
                                    delta: thinking,
                                    partial: output.clone(),
                                });
                            }
                            DeltaInfo::InputJsonDelta { partial_json } => {
                                if let Some(ref mut args_str) = partial_tool_args.get_mut(&idx) {
                                    args_str.push_str(&partial_json);
                                    let parsed = parse_streaming_json(args_str);
                                    if let Some(ContentBlock::ToolCall(ref mut tc)) =
                                        output.content.get_mut(idx)
                                    {
                                        tc.arguments = parsed;
                                    }
                                }
                                emitted_semantic_event = true;
                                stream.push(AssistantMessageEvent::ToolCallDelta {
                                    content_index: idx,
                                    delta: partial_json,
                                    partial: output.clone(),
                                });
                            }
                            DeltaInfo::SignatureDelta { signature } => {
                                // Store signature on thinking blocks
                                if let Some(ContentBlock::Thinking(ref mut t)) =
                                    output.content.get_mut(idx)
                                {
                                    let existing =
                                        t.thinking_signature.get_or_insert_with(String::new);
                                    existing.push_str(&signature);
                                }
                            }
                        }
                    }
                }
                "content_block_stop" => {
                    if let Ok(stop_data) = serde_json::from_str::<ContentBlockStopData>(data) {
                        let idx = stop_data.index;
                        open_blocks.remove(&idx);
                        if let Some(block_type) = block_types.get(idx) {
                            match block_type {
                                BlockType::Text => {
                                    let text = output
                                        .content
                                        .get(idx)
                                        .and_then(|b| b.as_text())
                                        .map(|t| t.text.clone())
                                        .unwrap_or_default();
                                    emitted_semantic_event = true;
                                    stream.push(AssistantMessageEvent::TextEnd {
                                        content_index: idx,
                                        content: text,
                                        partial: output.clone(),
                                    });
                                }
                                BlockType::Thinking => {
                                    let text = output
                                        .content
                                        .get(idx)
                                        .and_then(|b| b.as_thinking())
                                        .map(|t| t.thinking.clone())
                                        .unwrap_or_default();
                                    emitted_semantic_event = true;
                                    stream.push(AssistantMessageEvent::ThinkingEnd {
                                        content_index: idx,
                                        content: text,
                                        partial: output.clone(),
                                    });
                                }
                                BlockType::ToolUse => {
                                    // Finalize tool call args from the accumulated partial JSON
                                    if let Some(args_str) = partial_tool_args.get(&idx) {
                                        if let Ok(parsed) =
                                            serde_json::from_str::<serde_json::Value>(args_str)
                                        {
                                            if let Some(ContentBlock::ToolCall(ref mut tc)) =
                                                output.content.get_mut(idx)
                                            {
                                                tc.arguments = parsed;
                                            }
                                        }
                                    }
                                    let tool_call = output
                                        .content
                                        .get(idx)
                                        .and_then(|b| b.as_tool_call())
                                        .cloned()
                                        .unwrap_or_else(|| {
                                            ToolCall::new("", "", serde_json::Value::Null)
                                        });
                                    emitted_semantic_event = true;
                                    stream.push(AssistantMessageEvent::ToolCallEnd {
                                        content_index: idx,
                                        tool_call,
                                        partial: output.clone(),
                                    });
                                    partial_tool_args.remove(&idx);
                                }
                                _ => {}
                            }
                        }
                    }
                }
                "message_delta" => {
                    if let Ok(delta_data) = serde_json::from_str::<MessageDeltaData>(data) {
                        saw_message_delta = true;
                        if let Some(ref reason) = delta_data.delta.stop_reason {
                            output.stop_reason = match reason.as_str() {
                                "end_turn" => StopReason::Stop,
                                "max_tokens" => StopReason::Length,
                                "tool_use" => StopReason::ToolUse,
                                "stop_sequence" => StopReason::Stop,
                                _ => StopReason::Stop,
                            };
                        }
                        if let Some(usage) = delta_data.usage {
                            output.usage.output = usage.output_tokens;
                            output.usage.total_tokens = output.usage.input
                                + output.usage.output
                                + output.usage.cache_read
                                + output.usage.cache_write;
                        }
                    }
                }
                "message_stop" => {
                    saw_message_stop = true;
                }
                "error" => {
                    // Parse error event
                    if let Ok(error_val) = serde_json::from_str::<serde_json::Value>(data) {
                        let error_msg = error_val
                            .get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                            .unwrap_or("Unknown Anthropic error");
                        output.stop_reason = StopReason::Error;
                        output.error_message = Some(error_msg.to_string());
                        // Close any open thinking/text blocks before emitting the error
                        let (thinking_indices, text_indices) =
                            find_open_block_indices(&open_blocks, &block_types);
                        super::common::emit_pending_block_ends_multi(
                            &stream,
                            &output,
                            thinking_indices,
                            text_indices,
                        );
                        stream.push(AssistantMessageEvent::Error {
                            reason: StopReason::Error,
                            error: output,
                        });
                        stream.end(None);
                        return Ok(());
                    }
                }
                "ping" => {
                    // Heartbeat, ignore
                }
                _ => {}
            }
        }
    }

    if let Some(detail) = incomplete_anthropic_stream_detail(
        saw_message_delta,
        saw_message_stop,
        &block_types,
        &open_blocks,
        &partial_tool_args,
        &line_buffer,
    ) {
        tracing::error!(
            url = %url,
            model = %model.id,
            detail = %detail,
            "Anthropic stream ended before protocol completion"
        );
        // Close any open thinking/text blocks before emitting the incomplete error
        let (thinking_indices, text_indices) = find_open_block_indices(&open_blocks, &block_types);
        super::common::emit_pending_block_ends_multi(
            &stream,
            &output,
            thinking_indices,
            text_indices,
        );
        super::common::emit_incomplete_stream_error(
            &mut output,
            "anthropic",
            detail,
            limits.http.max_error_message_chars,
            &stream,
        );
        return Ok(());
    }

    stream.push(AssistantMessageEvent::Done {
        reason: output.stop_reason,
        message: output,
    });
    stream.end(None);

    Ok(())
}

/// Track block types by index for content_block_stop handling.
#[derive(Debug, Clone, Copy)]
enum BlockType {
    Unknown,
    Text,
    Thinking,
    RedactedThinking,
    ToolUse,
}

/// Collect all open thinking and text block indices from the Anthropic
/// block state. Returns `(thinking_indices, text_indices)`.
fn find_open_block_indices(
    open_blocks: &HashSet<usize>,
    block_types: &[BlockType],
) -> (Vec<usize>, Vec<usize>) {
    let mut thinking_indices = Vec::new();
    let mut text_indices = Vec::new();
    for &idx in open_blocks {
        match block_types.get(idx) {
            Some(BlockType::Thinking) => thinking_indices.push(idx),
            Some(BlockType::Text) => text_indices.push(idx),
            _ => {}
        }
    }
    (thinking_indices, text_indices)
}

fn incomplete_anthropic_stream_detail(
    saw_message_delta: bool,
    saw_message_stop: bool,
    block_types: &[BlockType],
    open_blocks: &HashSet<usize>,
    partial_tool_args: &HashMap<usize, String>,
    line_buffer: &str,
) -> Option<String> {
    let mut reasons = Vec::new();

    // When message_stop was received, tolerate a missing message_delta —
    // some proxies / compatible endpoints omit it. This mirrors the
    // OpenAI Completions tolerance where [DONE] compensates for a missing
    // finish_reason.
    if !saw_message_delta && !saw_message_stop {
        reasons.push("missing message_delta".to_string());
    }
    if !saw_message_stop {
        reasons.push("missing message_stop".to_string());
    }

    if !open_blocks.is_empty() {
        let mut indexes: Vec<_> = open_blocks.iter().copied().collect();
        indexes.sort_unstable();
        reasons.push(format!(
            "unclosed content blocks at indices [{}]",
            indexes
                .iter()
                .map(|index| index.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    let mut incomplete_tool_indexes: Vec<_> = open_blocks
        .iter()
        .copied()
        .filter(|index| {
            matches!(block_types.get(*index), Some(BlockType::ToolUse))
                && partial_tool_args
                    .get(index)
                    .is_some_and(|args| !args.trim().is_empty())
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

    let trailing = line_buffer.trim();
    if !trailing.is_empty() {
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
    use std::sync::Mutex;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn test_provider_type() {
        let provider = AnthropicProtocol::new();
        assert_eq!(provider.provider_type(), Provider::Anthropic);
    }

    #[test]
    fn test_convert_messages_basic() {
        let mut context = Context::with_system_prompt("You are helpful.");
        context.add_message(Message::User(UserMessage::text("Hello")));

        let model = Model::builder()
            .id("claude-3-5-sonnet")
            .name("Claude 3.5 Sonnet")
            .api(Api::AnthropicMessages)
            .provider(Provider::Anthropic)
            .context_window(200000)
            .max_tokens(8192)
            .build()
            .unwrap();

        let messages = convert_messages(&context, &model, None, false, false);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    }

    #[test]
    fn test_convert_tool_results_merged() {
        let mut context = Context::new();
        context.add_message(Message::ToolResult(ToolResultMessage::text(
            "call_1", "tool_a", "result1", false,
        )));
        context.add_message(Message::ToolResult(ToolResultMessage::text(
            "call_2", "tool_b", "result2", false,
        )));

        let model = Model::builder()
            .id("claude-3-5-sonnet")
            .name("Claude 3.5 Sonnet")
            .api(Api::AnthropicMessages)
            .provider(Provider::Anthropic)
            .context_window(200000)
            .max_tokens(8192)
            .build()
            .unwrap();

        let messages = convert_messages(&context, &model, None, false, false);
        // Tool results should be merged into a single user message
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        match &messages[0].content {
            AnthropicContent::Blocks(blocks) => assert_eq!(blocks.len(), 2),
            _ => panic!("Expected blocks"),
        }
    }

    #[test]
    fn test_incomplete_stream_detail_reports_missing_closure() {
        let mut open_blocks = HashSet::new();
        open_blocks.insert(1);

        let mut partial_tool_args = HashMap::new();
        partial_tool_args.insert(1, "{\"path\":\"logs".to_string());

        let detail = incomplete_anthropic_stream_detail(
            false,
            false,
            &[BlockType::Unknown, BlockType::ToolUse],
            &open_blocks,
            &partial_tool_args,
            "event: content_block_delta\ndata: {\"partial_json\":\"{\\\"path\\\":\\\"logs\"}",
        )
        .expect("detail should be reported");

        assert!(detail.contains("missing message_delta"));
        assert!(detail.contains("missing message_stop"));
        assert!(detail.contains("unclosed content blocks at indices [1]"));
        assert!(detail.contains("unfinished tool input JSON at indices [1]"));
        assert!(detail.contains("trailing partial SSE frame"));
    }

    #[test]
    fn test_resolve_cache_retention_uses_tiy_env_prefix() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let old_tiy = std::env::var("TIY_CACHE_RETENTION").ok();

        std::env::remove_var("TIY_CACHE_RETENTION");
        assert_eq!(resolve_cache_retention(None), CacheRetention::Short);

        std::env::set_var("TIY_CACHE_RETENTION", "long");
        assert_eq!(resolve_cache_retention(None), CacheRetention::Long);

        if let Some(value) = old_tiy {
            std::env::set_var("TIY_CACHE_RETENTION", value);
        } else {
            std::env::remove_var("TIY_CACHE_RETENTION");
        }
    }
}
