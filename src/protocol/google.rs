//! Google Generative AI provider.
//!
//! Implements streaming via Google's SSE protocol with JSON chunks containing
//! response candidates with parts-based content format.

/// Default base URL for Google Generative AI API.
const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Default base URL for Google Vertex AI API.
const DEFAULT_VERTEX_BASE_URL: &str = "https://us-central1-aiplatform.googleapis.com";

const SKIP_THOUGHT_SIGNATURE_VALIDATOR: &str = "skip_thought_signature_validator";

use crate::protocol::LLMProtocol;
use crate::stream::AssistantMessageEventStream;
use crate::thinking::ThinkingLevel;
use crate::transform::transform_messages;
use crate::types::*;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// Google Generative AI provider.
pub struct GoogleProtocol {
    client: Client,
    default_api_key: Option<String>,
}

impl GoogleProtocol {
    /// Create a new Google provider.
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            default_api_key: None,
        }
    }

    /// Create a provider with a default API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
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
        std::env::var("GOOGLE_API_KEY")
            .or_else(|_| std::env::var("GEMINI_API_KEY"))
            .unwrap_or_default()
    }
}

impl Default for GoogleProtocol {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProtocol for GoogleProtocol {
    fn provider_type(&self) -> Provider {
        Provider::Google
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
                stream_clone,
            )
            .await
            {
                tracing::error!("Google stream error: {}", e);
                super::common::emit_background_task_error(
                    &model,
                    model.api.clone().unwrap_or(Api::GoogleGenerativeAi),
                    format!("Google stream error: {}", e),
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
        let thinking_config =
            build_thinking_config(model, options.reasoning, options.thinking_budget_tokens);
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
                thinking_config,
                stream_clone,
            )
            .await
            {
                tracing::error!("Google stream error: {}", e);
                super::common::emit_background_task_error(
                    &model,
                    model.api.clone().unwrap_or(Api::GoogleGenerativeAi),
                    format!("Google stream error: {}", e),
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

/// Google Generative AI request.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleRequest {
    contents: Vec<GoogleContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GoogleSystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GoogleGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GoogleTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_config: Option<GoogleToolConfig>,
}

#[derive(Debug, Serialize)]
struct GoogleSystemInstruction {
    parts: Vec<GooglePart>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GoogleContent {
    role: String,
    #[serde(default)]
    parts: Vec<GooglePart>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GooglePart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thought: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thought_signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<GoogleFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_response: Option<GoogleFunctionResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inline_data: Option<GoogleInlineData>,
}

impl GooglePart {
    fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            thought: None,
            thought_signature: None,
            function_call: None,
            function_response: None,
            inline_data: None,
        }
    }

    fn thinking(text: impl Into<String>, signature: Option<String>) -> Self {
        Self {
            text: Some(text.into()),
            thought: Some(true),
            thought_signature: signature,
            function_call: None,
            function_response: None,
            inline_data: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GoogleFunctionCall {
    name: String,
    args: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GoogleFunctionResponse {
    name: String,
    response: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parts: Option<Vec<GooglePart>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GoogleInlineData {
    mime_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<GoogleThinkingConfig>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleThinkingConfig {
    include_thoughts: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_level: Option<String>,
}

fn clamp_reasoning(level: ThinkingLevel) -> ThinkingLevel {
    if matches!(level, ThinkingLevel::XHigh) {
        ThinkingLevel::High
    } else {
        level
    }
}

fn is_gemini3_pro_model(model: &Model) -> bool {
    let id = model.id.to_lowercase();
    let normalized = id.strip_prefix("models/").unwrap_or(&id);
    normalized.contains("gemini-3") && normalized.contains("-pro")
}

fn is_gemini3_flash_model(model: &Model) -> bool {
    let id = model.id.to_lowercase();
    let normalized = id.strip_prefix("models/").unwrap_or(&id);
    normalized.contains("gemini-3") && normalized.contains("-flash")
}

fn get_gemini3_thinking_level(level: ThinkingLevel, model: &Model) -> String {
    if is_gemini3_pro_model(model) {
        match level {
            ThinkingLevel::Minimal | ThinkingLevel::Low => "LOW".to_string(),
            ThinkingLevel::Medium | ThinkingLevel::High | ThinkingLevel::XHigh => {
                "HIGH".to_string()
            }
            ThinkingLevel::Off => "LOW".to_string(),
        }
    } else {
        match level {
            ThinkingLevel::Minimal => "MINIMAL".to_string(),
            ThinkingLevel::Low => "LOW".to_string(),
            ThinkingLevel::Medium => "MEDIUM".to_string(),
            ThinkingLevel::High | ThinkingLevel::XHigh => "HIGH".to_string(),
            ThinkingLevel::Off => "LOW".to_string(),
        }
    }
}

fn default_google_budget(model: &Model, level: ThinkingLevel) -> i32 {
    let level = clamp_reasoning(level);
    if model.id.contains("2.5-pro") {
        return match level {
            ThinkingLevel::Minimal => 128,
            ThinkingLevel::Low => 2048,
            ThinkingLevel::Medium => 8192,
            ThinkingLevel::High | ThinkingLevel::XHigh => 32768,
            ThinkingLevel::Off => -1,
        };
    }

    if model.id.contains("2.5-flash") {
        return match level {
            ThinkingLevel::Minimal => 128,
            ThinkingLevel::Low => 2048,
            ThinkingLevel::Medium => 8192,
            ThinkingLevel::High | ThinkingLevel::XHigh => 24576,
            ThinkingLevel::Off => -1,
        };
    }

    -1
}

fn build_thinking_config(
    model: &Model,
    level: Option<ThinkingLevel>,
    thinking_budget_tokens: Option<u32>,
) -> Option<GoogleThinkingConfig> {
    let level = level?;

    if !model.reasoning {
        return None;
    }

    let level = clamp_reasoning(level);
    if is_gemini3_pro_model(model) || is_gemini3_flash_model(model) {
        return Some(GoogleThinkingConfig {
            include_thoughts: true,
            thinking_budget: None,
            thinking_level: Some(get_gemini3_thinking_level(level, model)),
        });
    }

    Some(GoogleThinkingConfig {
        include_thoughts: true,
        thinking_budget: Some(
            thinking_budget_tokens
                .map(|tokens| tokens as i32)
                .unwrap_or_else(|| default_google_budget(model, level)),
        ),
        thinking_level: None,
    })
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleTool {
    function_declarations: Vec<GoogleFunctionDeclaration>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleFunctionDeclaration {
    name: String,
    description: String,
    #[serde(rename = "parametersJsonSchema")]
    parameters_json_schema: serde_json::Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleToolConfig {
    function_calling_config: GoogleFunctionCallingConfig,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleFunctionCallingConfig {
    mode: String,
}

// ============================================================================
// SSE Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleStreamChunk {
    response_id: Option<String>,
    candidates: Option<Vec<GoogleCandidate>>,
    usage_metadata: Option<GoogleUsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleCandidate {
    content: Option<GoogleContent>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleUsageMetadata {
    #[serde(default)]
    prompt_token_count: u64,
    #[serde(default)]
    candidates_token_count: u64,
    #[serde(default)]
    cached_content_token_count: u64,
    #[serde(default)]
    total_token_count: u64,
    #[serde(default)]
    #[allow(dead_code)]
    thoughts_token_count: u64,
}

// ============================================================================
// Tool call ID generator
// ============================================================================

static TOOL_CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

fn generate_tool_call_id(name: &str, model_id: &str) -> String {
    let counter = TOOL_CALL_COUNTER.fetch_add(1, AtomicOrdering::SeqCst);
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    normalize_google_tool_call_id(
        &format!("{}_{}", name, timestamp + counter as u128),
        model_id,
    )
}

fn normalize_google_tool_call_id(id: &str, model_id: &str) -> String {
    if !requires_google_tool_call_id(model_id) {
        return id.to_string();
    }

    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect()
}

fn requires_google_tool_call_id(model_id: &str) -> bool {
    let model_id = model_id.to_ascii_lowercase();
    model_id.starts_with("claude-") || model_id.starts_with("gpt-oss-")
}

fn supports_multimodal_function_response(model_id: &str) -> bool {
    let model_id = model_id
        .to_ascii_lowercase()
        .trim_start_matches("models/")
        .to_string();
    if let Some(rest) = model_id.strip_prefix("gemini-") {
        if let Some(version) = rest.split(['-', '.']).next() {
            if let Ok(major) = version.parse::<u32>() {
                return major >= 3;
            }
        }
    }
    true
}

fn build_google_tool_config(
    tools: Option<&[Tool]>,
    tool_choice: Option<&ToolChoice>,
) -> Option<GoogleToolConfig> {
    if tools.is_none_or(|tools| tools.is_empty()) {
        return None;
    }

    let mode = match tool_choice {
        Some(ToolChoice::Mode(ToolChoiceMode::Auto)) => "AUTO",
        Some(ToolChoice::Mode(ToolChoiceMode::None)) => "NONE",
        Some(ToolChoice::Mode(ToolChoiceMode::Any | ToolChoiceMode::Required)) => "ANY",
        Some(ToolChoice::Named(_)) => "ANY",
        None => return None,
    };

    Some(GoogleToolConfig {
        function_calling_config: GoogleFunctionCallingConfig {
            mode: mode.to_string(),
        },
    })
}

fn resolve_google_tool_call_id(
    provided_id: Option<&str>,
    name: &str,
    model_id: &str,
    content: &[ContentBlock],
) -> String {
    if let Some(provided_id) = provided_id {
        let normalized = normalize_google_tool_call_id(provided_id, model_id);
        if !normalized.is_empty()
            && !content.iter().any(|block| {
                block
                    .as_tool_call()
                    .is_some_and(|tool_call| tool_call.id == normalized)
            })
        {
            return normalized;
        }
    }

    generate_tool_call_id(name, model_id)
}

// ============================================================================
// Message Conversion
// ============================================================================

fn convert_messages(context: &Context, target_model: &Model) -> Vec<GoogleContent> {
    let mut contents = Vec::new();
    let normalize_google_tool_call_id_for_model =
        |id: &str| normalize_google_tool_call_id(id, &target_model.id);
    let transformed = transform_messages(
        &context.messages,
        target_model,
        Some(&normalize_google_tool_call_id_for_model),
    );

    for msg in &transformed {
        match msg {
            Message::User(user_msg) => {
                let parts = match &user_msg.content {
                    UserContent::Text(text) => vec![GooglePart::text(text)],
                    UserContent::Blocks(blocks) => blocks
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::Text(t) => Some(GooglePart::text(&t.text)),
                            ContentBlock::Image(img) => Some(GooglePart {
                                text: None,
                                thought: None,
                                thought_signature: None,
                                function_call: None,
                                function_response: None,
                                inline_data: Some(GoogleInlineData {
                                    mime_type: img.mime_type.clone(),
                                    data: img.data.clone(),
                                }),
                            }),
                            _ => None,
                        })
                        .collect(),
                };
                contents.push(GoogleContent {
                    role: "user".to_string(),
                    parts,
                });
            }
            Message::Assistant(assistant_msg) => {
                let same_api = target_model
                    .api
                    .as_ref()
                    .is_none_or(|api| *api == assistant_msg.api);
                let is_same_model = assistant_msg.provider == target_model.provider
                    && same_api
                    && assistant_msg.model == target_model.id;
                let mut parts = Vec::new();
                for block in &assistant_msg.content {
                    match block {
                        ContentBlock::Text(t) if !t.text.trim().is_empty() => {
                            parts.push(GooglePart {
                                text: Some(t.text.clone()),
                                thought: None,
                                thought_signature: if is_same_model {
                                    t.text_signature.clone()
                                } else {
                                    None
                                },
                                function_call: None,
                                function_response: None,
                                inline_data: None,
                            });
                        }
                        ContentBlock::Thinking(t) if !t.thinking.trim().is_empty() => {
                            parts.push(GooglePart::thinking(
                                &t.thinking,
                                t.thinking_signature.clone(),
                            ));
                        }
                        ContentBlock::ToolCall(tc) => {
                            let effective_signature = tc.thought_signature.clone().or_else(|| {
                                if target_model.id.to_ascii_lowercase().contains("gemini-3") {
                                    Some(SKIP_THOUGHT_SIGNATURE_VALIDATOR.to_string())
                                } else {
                                    None
                                }
                            });
                            parts.push(GooglePart {
                                text: None,
                                thought: None,
                                thought_signature: effective_signature,
                                function_call: Some(GoogleFunctionCall {
                                    name: tc.name.clone(),
                                    args: tc.arguments.clone(),
                                    id: if requires_google_tool_call_id(&target_model.id) {
                                        Some(tc.id.clone())
                                    } else {
                                        None
                                    },
                                }),
                                function_response: None,
                                inline_data: None,
                            });
                        }
                        _ => {}
                    }
                }

                if !parts.is_empty() {
                    contents.push(GoogleContent {
                        role: "model".to_string(),
                        parts,
                    });
                }
            }
            Message::ToolResult(tool_result) => {
                let text: String = tool_result
                    .content
                    .iter()
                    .filter_map(|b| b.as_text())
                    .map(|t| t.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                let image_parts: Vec<GooglePart> = tool_result
                    .content
                    .iter()
                    .filter_map(|b| {
                        b.as_image().map(|img| GooglePart {
                            text: None,
                            thought: None,
                            thought_signature: None,
                            function_call: None,
                            function_response: None,
                            inline_data: Some(GoogleInlineData {
                                mime_type: img.mime_type.clone(),
                                data: img.data.clone(),
                            }),
                        })
                    })
                    .collect();
                let has_images = !image_parts.is_empty();
                let supports_multimodal =
                    has_images && supports_multimodal_function_response(&target_model.id);

                let response_value = if tool_result.is_error {
                    serde_json::json!({ "error": text })
                } else {
                    serde_json::json!({
                        "output": if !text.is_empty() {
                            text.clone()
                        } else if has_images {
                            "(see attached image)".to_string()
                        } else {
                            String::new()
                        }
                    })
                };

                let part = GooglePart {
                    text: None,
                    thought: None,
                    thought_signature: None,
                    function_call: None,
                    function_response: Some(GoogleFunctionResponse {
                        name: tool_result.tool_name.clone(),
                        response: response_value,
                        id: if requires_google_tool_call_id(&target_model.id) {
                            Some(tool_result.tool_call_id.clone())
                        } else {
                            None
                        },
                        parts: if supports_multimodal {
                            Some(image_parts.clone())
                        } else {
                            None
                        },
                    }),
                    inline_data: None,
                };

                // Merge with last user/function message if possible
                if let Some(last) = contents.last_mut() {
                    if last.role == "user" {
                        last.parts.push(part);
                        continue;
                    }
                }

                contents.push(GoogleContent {
                    role: "user".to_string(),
                    parts: vec![part],
                });

                if has_images && !supports_multimodal {
                    let mut parts = vec![GooglePart::text("Tool result image:")];
                    parts.extend(image_parts);
                    contents.push(GoogleContent {
                        role: "user".to_string(),
                        parts,
                    });
                }
            }
        }
    }

    contents
}

fn convert_tools(tools: &[Tool]) -> Vec<GoogleTool> {
    let declarations: Vec<GoogleFunctionDeclaration> = tools
        .iter()
        .map(|t| GoogleFunctionDeclaration {
            name: t.name.clone(),
            description: t.description.clone(),
            parameters_json_schema: t.parameters.clone(),
        })
        .collect();

    if declarations.is_empty() {
        Vec::new()
    } else {
        vec![GoogleTool {
            function_declarations: declarations,
        }]
    }
}

fn normalize_google_model_id(model_id: &str, is_vertex: bool) -> &str {
    let model_id = model_id.strip_prefix("models/").unwrap_or(model_id);

    let model_id = if is_vertex {
        model_id
            .strip_prefix("publishers/google/models/")
            .unwrap_or(model_id)
    } else {
        model_id
    };

    model_id.strip_prefix("google/").unwrap_or(model_id)
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
    thinking_config: Option<GoogleThinkingConfig>,
    stream: AssistantMessageEventStream,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let limits = options.security_config();
    let cancel_token = options.cancel_token.clone();

    let mut output = AssistantMessage::builder()
        .api(Api::GoogleGenerativeAi)
        .provider(model.provider.clone())
        .model(model.id.clone())
        .stop_reason(StopReason::Stop)
        .usage(Usage::default())
        .build()?;

    let contents = convert_messages(context, model);
    let tools = context.tools.as_ref().map(|t| convert_tools(t));

    let system_instruction = context
        .system_prompt
        .as_ref()
        .map(|prompt| GoogleSystemInstruction {
            parts: vec![GooglePart::text(prompt)],
        });

    let request = GoogleRequest {
        contents,
        system_instruction,
        generation_config: Some(GoogleGenerationConfig {
            temperature: options.temperature,
            max_output_tokens: options.max_tokens.or(Some(model.max_tokens)),
            thinking_config,
        }),
        tools,
        tool_config: build_google_tool_config(
            context.tools.as_deref(),
            options.tool_choice.as_ref(),
        ),
    };

    // Apply on_payload hook if set
    let body_string = super::common::apply_on_payload(&request, &options.on_payload, model).await?;

    // Determine if this is a Vertex AI request
    let is_vertex = model
        .api
        .as_ref()
        .map(|api| matches!(api, Api::GoogleVertex))
        .unwrap_or(false);

    let base = super::common::resolve_base_url(
        options.base_url.as_deref(),
        model.base_url.as_deref(),
        if is_vertex {
            DEFAULT_VERTEX_BASE_URL
        } else {
            DEFAULT_BASE_URL
        },
    );

    // H1: Validate base URL against security policy
    if !super::common::validate_url_or_error(base, &limits, &mut output, &stream) {
        return Ok(());
    }

    // Native Google APIs already encode publisher/models in the URL path, so
    // strip catalog-style prefixes such as `google/` from model IDs here.
    let request_model_id = normalize_google_model_id(&model.id, is_vertex);

    // Vertex AI URL: {base}/v1/publishers/google/models/{model}:streamGenerateContent?alt=sse
    // Generative AI URL: {base}/models/{model}:streamGenerateContent?alt=sse
    let url = if is_vertex {
        format!(
            "{}/v1/publishers/google/models/{}:streamGenerateContent?alt=sse",
            base, request_model_id
        )
    } else {
        format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            base, request_model_id
        )
    };

    tracing::info!(
        url = %url,
        model = %model.id,
        provider = %model.provider,
        content_count = request.contents.len(),
        has_tools = request.tools.is_some(),
        "Sending Google GenerativeAI request"
    );
    tracing::debug!(request_body = %super::common::debug_preview(&body_string, 500), "Request payload");

    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(reqwest::header::CONTENT_TYPE, "application/json".parse()?);

    // Vertex AI uses Authorization: Bearer header; Generative AI uses x-goog-api-key
    if is_vertex {
        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", api_key).parse()?,
        );
    } else {
        headers.insert("x-goog-api-key", api_key.parse()?);
    }

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
            "Google GenerativeAI",
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

    let mut current_text_index: Option<usize> = None;
    let mut current_thinking_index: Option<usize> = None;
    let mut line_buffer = String::new();
    let mut saw_candidate_finish_reason = false;
    let mut saw_usage_metadata = false;

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
                    "Retryable Google stream error before first semantic event, retrying request"
                );
                stream.push(AssistantMessageEvent::Retrying {
                    attempt: prelude_retry_attempt + 1,
                    max_retries,
                    delay_ms: delay.as_millis() as u64,
                    reason: err.to_string(),
                    status: None,
                });
                if super::common::sleep_with_cancel(delay, cancel_token.as_ref()).await {
                    super::common::emit_aborted(&mut output, &stream);
                    return Ok(());
                }
                prelude_retry_attempt += 1;
                output = initial_output.clone();
                current_text_index = None;
                current_thinking_index = None;
                line_buffer.clear();
                saw_candidate_finish_reason = false;
                saw_usage_metadata = false;

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
                        "Google GenerativeAI",
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
                super::common::emit_pending_block_ends(
                    &stream,
                    &output,
                    current_thinking_index,
                    current_text_index,
                );
                super::common::emit_terminal_error(
                    &mut output,
                    format!("Google stream transport error: {}", err),
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

            if !line.starts_with("data: ") {
                continue;
            }

            let data = &line[6..];
            if data.is_empty() || data == "[DONE]" {
                continue;
            }

            let parsed: Result<GoogleStreamChunk, _> = serde_json::from_str(data);
            match parsed {
                Ok(chunk_data) => {
                    if let Some(response_id) = &chunk_data.response_id {
                        output.response_id = Some(response_id.clone());
                    }

                    // Handle usage metadata
                    if let Some(ref usage) = chunk_data.usage_metadata {
                        saw_usage_metadata = true;
                        output.usage.input = usage.prompt_token_count;
                        output.usage.output =
                            usage.candidates_token_count + usage.thoughts_token_count;
                        output.usage.cache_read = usage.cached_content_token_count;
                        output.usage.total_tokens = if usage.total_token_count > 0 {
                            usage.total_token_count
                        } else {
                            output.usage.input + output.usage.output + output.usage.cache_read
                        };
                    }

                    if let Some(candidates) = chunk_data.candidates {
                        for candidate in &candidates {
                            // Handle finish reason
                            if let Some(ref reason) = candidate.finish_reason {
                                saw_candidate_finish_reason = true;
                                output.stop_reason = match reason.as_str() {
                                    "STOP" => StopReason::Stop,
                                    "MAX_TOKENS" => StopReason::Length,
                                    "SAFETY" | "RECITATION" | "BLOCKLIST" => StopReason::Error,
                                    _ => StopReason::Stop,
                                };
                            }

                            if let Some(ref content) = candidate.content {
                                for part in &content.parts {
                                    // Handle thinking content
                                    if part.thought == Some(true) {
                                        if let Some(ref thinking_text) = part.text {
                                            if !thinking_text.is_empty() {
                                                if current_thinking_index.is_none() {
                                                    let idx = output.content.len();
                                                    output.content.push(ContentBlock::Thinking(
                                                        ThinkingContent::new(""),
                                                    ));
                                                    current_thinking_index = Some(idx);
                                                    stream.push(
                                                        AssistantMessageEvent::ThinkingStart {
                                                            content_index: idx,
                                                            partial: output.clone(),
                                                        },
                                                    );
                                                }

                                                let idx = current_thinking_index.unwrap();
                                                if let Some(ContentBlock::Thinking(ref mut t)) =
                                                    output.content.get_mut(idx)
                                                {
                                                    t.thinking.push_str(thinking_text);
                                                    // Store thought signature if present
                                                    if let Some(ref sig) = part.thought_signature {
                                                        t.thinking_signature = Some(sig.clone());
                                                    }
                                                }
                                                emitted_semantic_event = true;
                                                stream.push(AssistantMessageEvent::ThinkingDelta {
                                                    content_index: idx,
                                                    delta: thinking_text.clone(),
                                                    partial: output.clone(),
                                                });
                                            }
                                        }
                                        continue;
                                    }

                                    // Handle function call (arrives complete, not streamed)
                                    if let Some(ref fc) = part.function_call {
                                        // End current thinking block if active
                                        if let Some(idx) = current_thinking_index.take() {
                                            let content = output
                                                .content
                                                .get(idx)
                                                .and_then(|b| b.as_thinking())
                                                .map(|t| t.thinking.clone())
                                                .unwrap_or_default();
                                            stream.push(AssistantMessageEvent::ThinkingEnd {
                                                content_index: idx,
                                                content,
                                                partial: output.clone(),
                                            });
                                        }
                                        // End current text block if active
                                        if let Some(idx) = current_text_index.take() {
                                            let content = output
                                                .content
                                                .get(idx)
                                                .and_then(|b| b.as_text())
                                                .map(|t| t.text.clone())
                                                .unwrap_or_default();
                                            stream.push(AssistantMessageEvent::TextEnd {
                                                content_index: idx,
                                                content,
                                                partial: output.clone(),
                                            });
                                        }

                                        let tool_call_id = resolve_google_tool_call_id(
                                            fc.id.as_deref(),
                                            &fc.name,
                                            &model.id,
                                            &output.content,
                                        );
                                        let mut tool_call =
                                            ToolCall::new(&tool_call_id, &fc.name, fc.args.clone());
                                        tool_call.thought_signature =
                                            part.thought_signature.clone();

                                        let idx = output.content.len();
                                        output
                                            .content
                                            .push(ContentBlock::ToolCall(tool_call.clone()));
                                        output.stop_reason = StopReason::ToolUse;

                                        emitted_semantic_event = true;
                                        stream.push(AssistantMessageEvent::ToolCallStart {
                                            content_index: idx,
                                            partial: output.clone(),
                                        });
                                        stream.push(AssistantMessageEvent::ToolCallEnd {
                                            content_index: idx,
                                            tool_call,
                                            partial: output.clone(),
                                        });
                                        continue;
                                    }

                                    // Handle text content
                                    if let Some(ref text_content) = part.text {
                                        if !text_content.is_empty() {
                                            // End thinking block if transitioning to text
                                            if let Some(idx) = current_thinking_index.take() {
                                                let content = output
                                                    .content
                                                    .get(idx)
                                                    .and_then(|b| b.as_thinking())
                                                    .map(|t| t.thinking.clone())
                                                    .unwrap_or_default();
                                                stream.push(AssistantMessageEvent::ThinkingEnd {
                                                    content_index: idx,
                                                    content,
                                                    partial: output.clone(),
                                                });
                                            }

                                            if current_text_index.is_none() {
                                                let idx = output.content.len();
                                                output
                                                    .content
                                                    .push(ContentBlock::Text(TextContent::new("")));
                                                current_text_index = Some(idx);
                                                stream.push(AssistantMessageEvent::TextStart {
                                                    content_index: idx,
                                                    partial: output.clone(),
                                                });
                                            }

                                            let idx = current_text_index.unwrap();
                                            if let Some(ContentBlock::Text(ref mut t)) =
                                                output.content.get_mut(idx)
                                            {
                                                t.text.push_str(text_content);
                                                if let Some(ref sig) = part.thought_signature {
                                                    t.text_signature = Some(sig.clone());
                                                }
                                            }
                                            emitted_semantic_event = true;
                                            stream.push(AssistantMessageEvent::TextDelta {
                                                content_index: idx,
                                                delta: text_content.clone(),
                                                partial: output.clone(),
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    let preview = if data.len() > 200 { &data[..200] } else { data };
                    tracing::warn!(error = %e, data_preview = %preview, "Failed to parse Google SSE JSON data");
                }
            }
        }
    }

    if let Some(detail) = incomplete_google_stream_detail(
        saw_candidate_finish_reason,
        saw_usage_metadata,
        &line_buffer,
    ) {
        tracing::error!(
            url = %url,
            model = %model.id,
            detail = %detail,
            "Google stream ended before protocol completion"
        );
        // Close any open thinking/text blocks before emitting the incomplete error
        super::common::emit_pending_block_ends(
            &stream,
            &output,
            current_thinking_index,
            current_text_index,
        );
        super::common::emit_incomplete_stream_error(
            &mut output,
            "google",
            detail,
            limits.http.max_error_message_chars,
            &stream,
        );
        return Ok(());
    }

    // End any active blocks
    if let Some(idx) = current_thinking_index {
        let content = output
            .content
            .get(idx)
            .and_then(|b| b.as_thinking())
            .map(|t| t.thinking.clone())
            .unwrap_or_default();
        stream.push(AssistantMessageEvent::ThinkingEnd {
            content_index: idx,
            content,
            partial: output.clone(),
        });
    }
    if let Some(idx) = current_text_index {
        let content = output
            .content
            .get(idx)
            .and_then(|b| b.as_text())
            .map(|t| t.text.clone())
            .unwrap_or_default();
        stream.push(AssistantMessageEvent::TextEnd {
            content_index: idx,
            content,
            partial: output.clone(),
        });
    }

    stream.push(AssistantMessageEvent::Done {
        reason: output.stop_reason,
        message: output,
    });
    stream.end(None);

    Ok(())
}

fn incomplete_google_stream_detail(
    saw_candidate_finish_reason: bool,
    saw_usage_metadata: bool,
    line_buffer: &str,
) -> Option<String> {
    let mut reasons = Vec::new();

    // When usage_metadata was received the server considers the response
    // complete, so tolerate a missing candidate finish_reason — some
    // proxy / gateway setups strip it while still delivering the final
    // usage chunk.
    if !saw_candidate_finish_reason && !saw_usage_metadata {
        reasons.push("missing candidate finish_reason".to_string());
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
    fn test_provider_type() {
        let provider = GoogleProtocol::new();
        assert_eq!(provider.provider_type(), Provider::Google);
    }

    #[test]
    fn test_convert_messages_basic() {
        let mut context = Context::new();
        context.add_message(Message::User(UserMessage::text("Hello")));

        let model = Model::builder()
            .id("gemini-2.0-flash")
            .name("Gemini 2.0 Flash")
            .api(Api::GoogleGenerativeAi)
            .provider(Provider::Google)
            .context_window(1048576)
            .max_tokens(8192)
            .build()
            .unwrap();

        let contents = convert_messages(&context, &model);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, "user");
        assert_eq!(contents[0].parts.len(), 1);
    }

    #[test]
    fn test_generate_tool_call_id() {
        let id1 = generate_tool_call_id("test_tool", "claude-3-7-sonnet");
        let id2 = generate_tool_call_id("test_tool", "claude-3-7-sonnet");
        assert_ne!(id1, id2);
        assert!(id1.starts_with("test_tool_"));
    }

    #[test]
    fn test_incomplete_google_stream_detail_reports_missing_termination() {
        let detail = incomplete_google_stream_detail(false, false, "data: {");

        assert_eq!(
            detail.as_deref(),
            Some("missing candidate finish_reason; trailing partial SSE frame")
        );
    }

    #[test]
    fn test_incomplete_google_stream_detail_usage_compensates_finish_reason() {
        // usage_metadata received but finish_reason missing — should only report trailing frame
        let detail = incomplete_google_stream_detail(false, true, "data: {");
        assert_eq!(detail.as_deref(), Some("trailing partial SSE frame"));

        // usage_metadata received, no trailing — should be None (complete)
        let detail = incomplete_google_stream_detail(false, true, "");
        assert!(detail.is_none());
    }
}
