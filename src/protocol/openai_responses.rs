//! OpenAI Responses API provider (new API for o1, o3, gpt-5 models).
//!
//! Implements streaming via typed SSE events:
//! response.output_item.added → response.output_text.delta / response.function_call_arguments.delta
//! → response.output_item.done → response.completed

/// Default base URL for OpenAI Responses API.
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const NON_VISION_USER_IMAGE_PLACEHOLDER: &str = "(image omitted: model does not support images)";
const NON_VISION_TOOL_IMAGE_PLACEHOLDER: &str =
    "(tool image omitted: model does not support images)";

use crate::protocol::LLMProtocol;
use crate::stream::{parse_streaming_json, AssistantMessageEventStream};
use crate::thinking::ThinkingLevel;
use crate::transform::transform_messages;
use crate::types::*;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

/// OpenAI Responses API provider.
pub struct OpenAIResponsesProtocol {
    client: Client,
    default_api_key: Option<String>,
}

impl OpenAIResponsesProtocol {
    /// Create a new OpenAI Responses provider.
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
        std::env::var("OPENAI_API_KEY").unwrap_or_default()
    }
}

impl Default for OpenAIResponsesProtocol {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProtocol for OpenAIResponsesProtocol {
    fn provider_type(&self) -> Provider {
        Provider::OpenAIResponses
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
                tracing::error!("OpenAI Responses stream error: {}", e);
                super::common::emit_background_task_error(
                    &model,
                    Api::OpenAIResponses,
                    format!("OpenAI Responses stream error: {}", e),
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
        let reasoning = build_reasoning(model, options.reasoning);
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
                reasoning,
                stream_clone,
            )
            .await
            {
                tracing::error!("OpenAI Responses stream error: {}", e);
                super::common::emit_background_task_error(
                    &model,
                    Api::OpenAIResponses,
                    format!("OpenAI Responses stream error: {}", e),
                    &error_stream,
                );
            }
        });

        stream
    }
}

// ============================================================================
// Request Types
// ============================================================================

/// OpenAI Responses API request.
#[derive(Debug, Serialize)]
struct ResponsesRequest {
    model: String,
    input: Vec<serde_json::Value>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_retention: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ResponsesTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ResponsesReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum ResponsesContent {
    Text(String),
    Parts(Vec<ResponsesContentPart>),
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum ResponsesContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "input_image")]
    InputImage { image_url: String },
    #[serde(rename = "output_text")]
    OutputText { text: String },
    #[serde(rename = "refusal")]
    Refusal { refusal: String },
}

#[derive(Debug, Serialize)]
struct ResponsesTool {
    #[serde(rename = "type")]
    tool_type: String,
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
struct ResponsesReasoning {
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<String>,
}

fn supports_xhigh(model: &Model) -> bool {
    super::common::supports_gpt5_xhigh(&model.id)
}

fn clamp_reasoning(level: ThinkingLevel, model: &Model) -> ThinkingLevel {
    if matches!(level, ThinkingLevel::XHigh) && !supports_xhigh(model) {
        ThinkingLevel::High
    } else {
        level
    }
}

fn build_reasoning(model: &Model, level: Option<ThinkingLevel>) -> Option<ResponsesReasoning> {
    if !model.reasoning {
        return None;
    }

    level.map(|level| ResponsesReasoning {
        effort: Some(clamp_reasoning(level, model).to_string()),
        summary: Some("auto".to_string()),
    })
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

fn get_prompt_cache_retention(base_url: &str, retention: CacheRetention) -> Option<String> {
    if retention == CacheRetention::Long && base_url.contains("api.openai.com") {
        Some("24h".to_string())
    } else {
        None
    }
}

fn map_service_tier(service_tier: OpenAIServiceTier) -> &'static str {
    match service_tier {
        OpenAIServiceTier::Auto => "auto",
        OpenAIServiceTier::Default => "default",
        OpenAIServiceTier::Flex => "flex",
        OpenAIServiceTier::Priority => "priority",
    }
}

// ============================================================================
// SSE Event Types
// ============================================================================

/// Parsed SSE event data from OpenAI Responses API.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ResponseEvent {
    #[serde(flatten)]
    data: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OutputItem {
    #[serde(rename = "type")]
    item_type: Option<String>,
    id: Option<String>,
    // For function_call items
    call_id: Option<String>,
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponseCompleted {
    response: Option<ResponseData>,
}

#[derive(Debug, Deserialize)]
struct ResponseData {
    id: Option<String>,
    status: Option<String>,
    usage: Option<ResponseUsage>,
    #[allow(dead_code)]
    output: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
struct ResponseUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    total_tokens: Option<u64>,
    #[serde(default)]
    input_tokens_details: Option<InputTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct InputTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
}

// ============================================================================
// Message Conversion
// ============================================================================

fn convert_messages(context: &Context, target_model: &Model) -> Vec<serde_json::Value> {
    // When `store` is false (the current default), server-side item IDs are not
    // persisted. Referencing them in subsequent requests causes 400 errors like
    // "Item with id 'rs_...' not found". We strip all server-side IDs from the
    // replayed conversation to avoid this.
    //
    // TODO: if `store` is ever made configurable, pass the flag in and
    // conditionally preserve IDs when `store == true`.
    let strip_server_ids = true;
    let transformed = transform_messages(&context.messages, target_model, None);
    let mut items = Vec::new();

    for msg in &transformed {
        match msg {
            Message::User(user_msg) => {
                let content = match &user_msg.content {
                    UserContent::Text(text) => ResponsesContent::Text(sanitize_surrogates(text)),
                    UserContent::Blocks(blocks) => ResponsesContent::Parts(
                        normalize_user_parts_for_responses(blocks, target_model),
                    ),
                };
                items.push(serde_json::json!({
                    "type": "message",
                    "role": "user",
                    "content": content,
                }));
            }
            Message::Assistant(assistant_msg) => {
                let is_same_model = is_same_openai_responses_model(assistant_msg, target_model);

                for block in &assistant_msg.content {
                    match block {
                        ContentBlock::Thinking(thinking) => {
                            if is_same_model && !strip_server_ids {
                                if let Some(signature) = &thinking.thinking_signature {
                                    if let Ok(reasoning_item) =
                                        serde_json::from_str::<serde_json::Value>(signature)
                                    {
                                        items.push(reasoning_item);
                                    }
                                }
                            }
                        }
                        ContentBlock::Text(text_block) => {
                            if text_block.text.trim().is_empty() {
                                continue;
                            }

                            let mut message = serde_json::json!({
                                "type": "message",
                                "role": "assistant",
                                "content": ResponsesContent::Parts(vec![ResponsesContentPart::OutputText {
                                    text: sanitize_surrogates(&text_block.text),
                                }]),
                                "status": "completed",
                            });

                            if !strip_server_ids {
                                if let Some(text_signature) =
                                    parse_text_signature(text_block.text_signature.as_deref())
                                {
                                    message["id"] = serde_json::Value::String(text_signature.id);
                                    if let Some(phase) = text_signature.phase {
                                        message["phase"] = serde_json::Value::String(phase);
                                    }
                                }
                            }

                            items.push(message);
                        }
                        ContentBlock::ToolCall(tc) => {
                            let call_id = normalize_responses_call_id(&tc.id);
                            let is_different_model = assistant_msg.model != target_model.id
                                && assistant_msg.provider == target_model.provider
                                && assistant_msg.api
                                    == target_model.api.clone().unwrap_or(Api::OpenAIResponses);
                            let item_id = responses_item_id_from_tool_call_id(&tc.id)
                                .map(normalize_responses_item_id)
                                .filter(|item_id| !item_id.is_empty());

                            let mut function_call = serde_json::json!({
                                "type": "function_call",
                                "call_id": call_id,
                                "name": tc.name,
                                "arguments": serde_json::to_string(&tc.arguments).unwrap_or_default(),
                            });

                            if !strip_server_ids {
                                if let Some(item_id) = item_id {
                                    if !is_different_model || !item_id.starts_with("fc_") {
                                        function_call["id"] = serde_json::Value::String(item_id);
                                    }
                                }
                            }

                            items.push(function_call);
                        }
                        ContentBlock::Image(_) => {}
                    }
                }
            }
            Message::ToolResult(tool_result) => {
                let call_id = normalize_responses_call_id(&tool_result.tool_call_id);

                items.push(serde_json::json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": tool_result_output_value(tool_result, target_model),
                }));
            }
        }
    }

    items
}

#[derive(Debug, Deserialize, Serialize)]
struct TextSignatureV1 {
    v: u8,
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase: Option<String>,
}

struct ParsedTextSignature {
    id: String,
    phase: Option<String>,
}

fn encode_text_signature_v1(id: &str, phase: Option<&str>) -> String {
    serde_json::to_string(&TextSignatureV1 {
        v: 1,
        id: id.to_string(),
        phase: phase.map(str::to_string),
    })
    .unwrap_or_else(|_| id.to_string())
}

fn parse_text_signature(signature: Option<&str>) -> Option<ParsedTextSignature> {
    let signature = signature?.trim();
    if signature.is_empty() {
        return None;
    }

    if signature.starts_with('{') {
        if let Ok(parsed) = serde_json::from_str::<TextSignatureV1>(signature) {
            return Some(ParsedTextSignature {
                id: parsed.id,
                phase: parsed.phase,
            });
        }
    }

    Some(ParsedTextSignature {
        id: signature.to_string(),
        phase: None,
    })
}

fn normalize_id_part(part: &str) -> String {
    let sanitized: String = part
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect();
    sanitized.trim_end_matches('_').to_string()
}

fn short_hash_hex(input: &str, hex_len: usize) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    let digest = hasher.finalize();
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    hex[..hex_len.min(hex.len())].to_string()
}

fn hashed_responses_call_id(raw_call_id: &str) -> String {
    format!("call_{}", short_hash_hex(raw_call_id, 35))
}

fn normalize_responses_call_id(id: &str) -> String {
    let raw_call_id = id.split('|').next().unwrap_or(id);
    let normalized = normalize_id_part(raw_call_id);
    if !normalized.is_empty() && normalized.len() <= 40 && normalized == raw_call_id {
        normalized
    } else {
        hashed_responses_call_id(raw_call_id)
    }
}

fn normalize_responses_item_id(item_id: &str) -> String {
    let normalized = normalize_id_part(item_id);
    if normalized.is_empty() || normalized != item_id {
        return format!("fc_{}", short_hash_hex(item_id, 24));
    }

    if normalized.starts_with("fc_") {
        return normalized;
    }

    if normalized.len() <= 61 {
        return format!("fc_{normalized}");
    }

    format!("fc_{}", short_hash_hex(item_id, 24))
}

fn responses_item_id_from_tool_call_id(id: &str) -> Option<&str> {
    id.split_once('|').map(|(_, item_id)| item_id)
}

fn is_same_openai_responses_model(assistant_msg: &AssistantMessage, target_model: &Model) -> bool {
    assistant_msg.provider == target_model.provider
        && assistant_msg.api == target_model.api.clone().unwrap_or(Api::OpenAIResponses)
        && assistant_msg.model == target_model.id
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
            "name": name,
        })),
        ToolChoice::Named(ToolChoiceNamed::Function { function }) => Some(serde_json::json!({
            "type": "function",
            "name": function.name,
        })),
    }
}

fn input_text_part(text: impl Into<String>) -> ResponsesContentPart {
    ResponsesContentPart::InputText { text: text.into() }
}

fn input_image_part(image: &ImageContent) -> ResponsesContentPart {
    ResponsesContentPart::InputImage {
        image_url: format!("data:{};base64,{}", image.mime_type, image.data),
    }
}

fn normalize_user_parts_for_responses(
    blocks: &[ContentBlock],
    target_model: &Model,
) -> Vec<ResponsesContentPart> {
    let mut parts = Vec::new();
    let mut previous_was_placeholder = false;

    for block in blocks {
        match block {
            ContentBlock::Text(t) => {
                let text = sanitize_surrogates(&t.text);
                previous_was_placeholder = text == NON_VISION_USER_IMAGE_PLACEHOLDER;
                parts.push(input_text_part(text));
            }
            ContentBlock::Image(img) => {
                if target_model.supports_image() {
                    parts.push(input_image_part(img));
                    previous_was_placeholder = false;
                } else if !previous_was_placeholder {
                    parts.push(input_text_part(NON_VISION_USER_IMAGE_PLACEHOLDER));
                    previous_was_placeholder = true;
                }
            }
            _ => {}
        }
    }

    parts
}

fn tool_result_output_value(
    tool_result: &ToolResultMessage,
    target_model: &Model,
) -> serde_json::Value {
    let text = tool_result
        .content
        .iter()
        .filter_map(|b| b.as_text())
        .map(|t| sanitize_surrogates(&t.text))
        .collect::<Vec<_>>()
        .join("\n");
    let images: Vec<&ImageContent> = tool_result
        .content
        .iter()
        .filter_map(|b| b.as_image())
        .collect();

    if !images.is_empty() && target_model.supports_image() {
        let mut output_parts = Vec::new();
        if !text.is_empty() {
            output_parts.push(serde_json::json!({
                "type": "input_text",
                "text": text,
            }));
        }
        for image in images {
            output_parts.push(serde_json::json!({
                "type": "input_image",
                "image_url": format!("data:{};base64,{}", image.mime_type, image.data),
            }));
        }
        serde_json::Value::Array(output_parts)
    } else {
        serde_json::Value::String(if text.is_empty() {
            if images.is_empty() {
                "(no output)".to_string()
            } else if target_model.supports_image() {
                "(see attached image)".to_string()
            } else {
                NON_VISION_TOOL_IMAGE_PLACEHOLDER.to_string()
            }
        } else {
            text
        })
    }
}

fn sanitize_surrogates(text: &str) -> String {
    text.replace(
        |c: char| {
            let cp = c as u32;
            (0xD800..=0xDFFF).contains(&cp)
        },
        "",
    )
}

fn convert_tools(tools: &[Tool]) -> Vec<ResponsesTool> {
    tools
        .iter()
        .map(|t| ResponsesTool {
            tool_type: "function".to_string(),
            name: t.name.clone(),
            description: t.description.clone(),
            parameters: t.parameters.clone(),
        })
        .collect()
}

fn ensure_response_text_item(
    output: &mut AssistantMessage,
    item_content_map: &mut HashMap<usize, ItemInfo>,
    open_output_items: &mut HashSet<usize>,
    output_index: usize,
    stream: &AssistantMessageEventStream,
) -> usize {
    if let Some(info) = item_content_map.get(&output_index) {
        return info.content_idx;
    }

    let content_idx = output.content.len();
    output
        .content
        .push(ContentBlock::Text(TextContent::new("")));
    item_content_map.insert(
        output_index,
        ItemInfo {
            content_idx,
            item_type: ItemType::Message,
            item_id: String::new(),
            call_id: None,
            name: None,
        },
    );
    open_output_items.insert(output_index);
    stream.push(AssistantMessageEvent::TextStart {
        content_index: content_idx,
        partial: output.clone(),
    });
    content_idx
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
    reasoning: Option<ResponsesReasoning>,
    stream: AssistantMessageEventStream,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let limits = options.security_config();
    let cancel_token = options.cancel_token.clone();
    let base = super::common::resolve_base_url(
        options.base_url.as_deref(),
        model.base_url.as_deref(),
        DEFAULT_BASE_URL,
    );
    let cache_retention = resolve_cache_retention(options.cache_retention);

    let mut output = AssistantMessage::builder()
        .api(Api::OpenAIResponses)
        .provider(model.provider.clone())
        .model(model.id.clone())
        .stop_reason(StopReason::Stop)
        .usage(Usage::default())
        .build()?;

    let mut input = convert_messages(context, model);
    let tools = context.tools.as_ref().map(|t| convert_tools(t));
    let max_output_tokens = super::common::clamp_openai_max_tokens(options.max_tokens);

    if model.reasoning && reasoning.is_none() && model.id.starts_with("gpt-5") {
        input.push(serde_json::json!({
            "type": "message",
            "role": "developer",
            "content": ResponsesContent::Parts(vec![ResponsesContentPart::InputText {
                text: "# Juice: 0 !important".to_string(),
            }]),
        }));
    }

    let request = ResponsesRequest {
        model: model.id.clone(),
        input,
        stream: true,
        store: Some(false),
        instructions: context.system_prompt.clone(),
        temperature: options.temperature,
        max_output_tokens,
        prompt_cache_key: if cache_retention == CacheRetention::None {
            None
        } else {
            options.session_id.clone()
        },
        prompt_cache_retention: get_prompt_cache_retention(base, cache_retention),
        tools,
        tool_choice: convert_tool_choice(options.tool_choice.as_ref()),
        reasoning: reasoning.clone(),
        include: reasoning
            .as_ref()
            .map(|_| vec!["reasoning.encrypted_content".to_string()]),
        service_tier: options
            .service_tier
            .map(|service_tier| map_service_tier(service_tier).to_string()),
    };

    // Apply on_payload hook if set
    let body_string = super::common::apply_on_payload(&request, &options.on_payload, model).await?;

    let url = format!("{}/responses", base);

    // H1: Validate base URL against security policy
    if !super::common::validate_url_or_error(base, &limits, &mut output, &stream) {
        return Ok(());
    }

    tracing::info!(
        url = %url,
        model = %model.id,
        provider = %model.provider,
        input_count = request.input.len(),
        has_tools = request.tools.is_some(),
        "Sending OpenAI Responses request"
    );
    tracing::debug!(request_body = %super::common::debug_preview(&body_string, 500), "Request payload");

    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::AUTHORIZATION,
        format!("Bearer {}", api_key).parse()?,
    );
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
            "OpenAI Responses",
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

    // Track output items by their index
    let mut item_content_map: HashMap<usize, ItemInfo> = HashMap::new();
    let mut open_output_items: HashSet<usize> = HashSet::new();
    let mut partial_tool_args: HashMap<usize, String> = HashMap::new();
    let mut line_buffer = String::new();
    let mut current_event_type = String::new();
    let mut item_counter: usize = 0;
    let mut saw_response_completion = false;

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
                    "Retryable OpenAI Responses stream error before first semantic event, retrying request"
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
                item_content_map.clear();
                open_output_items.clear();
                partial_tool_args.clear();
                line_buffer.clear();
                current_event_type.clear();
                item_counter = 0;
                saw_response_completion = false;

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
                        "OpenAI Responses",
                        &request_body,
                    )
                    .await;
                    return Ok(());
                }

                byte_stream = response.bytes_stream();
                continue;
            }
            Err(err) => {
                super::common::emit_transport_stream_error(
                    &mut output,
                    "openai_responses",
                    err.to_string(),
                    limits.http.max_error_message_chars,
                    &stream,
                );
                return Ok(());
            }
        };
        let text = String::from_utf8_lossy(&chunk);
        line_buffer.push_str(&text);

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

            // Determine event type: prefer JSON "type" field, fall back to SSE event: line.
            // Some proxies (e.g. Zenmux) may not forward the SSE "event:" line,
            // so we must always check the JSON "type" field as the primary source.
            let parsed = serde_json::from_str::<serde_json::Value>(data);
            let event_type = parsed
                .as_ref()
                .ok()
                .and_then(|v| v.get("type").and_then(|t| t.as_str()).map(String::from))
                .unwrap_or_else(|| current_event_type.clone());

            match event_type.as_str() {
                "response.created" => {
                    if let Ok(ref val) = parsed {
                        if let Some(response_id) = val
                            .get("response")
                            .and_then(|response| response.get("id"))
                            .and_then(|id| id.as_str())
                        {
                            output.response_id = Some(response_id.to_string());
                        }
                    }
                }

                "response.output_item.added" => {
                    if let Ok(ref val) = parsed {
                        let item = val.get("item");
                        let item_type = item
                            .and_then(|i| i.get("type"))
                            .and_then(|t| t.as_str())
                            .unwrap_or("");
                        let item_id = item
                            .and_then(|i| i.get("id"))
                            .and_then(|i| i.as_str())
                            .unwrap_or("")
                            .to_string();

                        let output_index = val
                            .get("output_index")
                            .and_then(|i| i.as_u64())
                            .unwrap_or(item_counter as u64)
                            as usize;

                        match item_type {
                            "message" => {
                                let content_idx = output.content.len();
                                output
                                    .content
                                    .push(ContentBlock::Text(TextContent::new("")));
                                item_content_map.insert(
                                    output_index,
                                    ItemInfo {
                                        content_idx,
                                        item_type: ItemType::Message,
                                        item_id,
                                        call_id: None,
                                        name: None,
                                    },
                                );
                                open_output_items.insert(output_index);
                                emitted_semantic_event = true;
                                stream.push(AssistantMessageEvent::TextStart {
                                    content_index: content_idx,
                                    partial: output.clone(),
                                });
                            }
                            "function_call" => {
                                let call_id = item
                                    .and_then(|i| i.get("call_id"))
                                    .and_then(|c| c.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let name = item
                                    .and_then(|i| i.get("name"))
                                    .and_then(|n| n.as_str())
                                    .unwrap_or("")
                                    .to_string();

                                // Composite ID: "{call_id}|{item_id}"
                                let composite_id = format!("{}|{}", call_id, item_id);
                                let content_idx = output.content.len();
                                output.content.push(ContentBlock::ToolCall(ToolCall::new(
                                    &composite_id,
                                    &name,
                                    serde_json::Value::Object(serde_json::Map::new()),
                                )));
                                partial_tool_args.insert(output_index, String::new());
                                open_output_items.insert(output_index);
                                item_content_map.insert(
                                    output_index,
                                    ItemInfo {
                                        content_idx,
                                        item_type: ItemType::FunctionCall,
                                        item_id,
                                        call_id: Some(call_id),
                                        name: Some(name),
                                    },
                                );
                                emitted_semantic_event = true;
                                stream.push(AssistantMessageEvent::ToolCallStart {
                                    content_index: content_idx,
                                    partial: output.clone(),
                                });
                            }
                            "reasoning" => {
                                let content_idx = output.content.len();
                                output
                                    .content
                                    .push(ContentBlock::Thinking(ThinkingContent::new("")));
                                open_output_items.insert(output_index);
                                item_content_map.insert(
                                    output_index,
                                    ItemInfo {
                                        content_idx,
                                        item_type: ItemType::Reasoning,
                                        item_id,
                                        call_id: None,
                                        name: None,
                                    },
                                );
                                emitted_semantic_event = true;
                                stream.push(AssistantMessageEvent::ThinkingStart {
                                    content_index: content_idx,
                                    partial: output.clone(),
                                });
                            }
                            _ => {}
                        }
                        item_counter += 1;
                    }
                }

                "response.output_text.delta" => {
                    if let Ok(ref val) = parsed {
                        let output_index = val
                            .get("output_index")
                            .and_then(|i| i.as_u64())
                            .unwrap_or(0) as usize;
                        let delta = val.get("delta").and_then(|d| d.as_str()).unwrap_or("");

                        let idx = ensure_response_text_item(
                            &mut output,
                            &mut item_content_map,
                            &mut open_output_items,
                            output_index,
                            &stream,
                        );
                        if let Some(ContentBlock::Text(ref mut t)) = output.content.get_mut(idx) {
                            t.text.push_str(delta);
                        }
                        emitted_semantic_event = true;
                        stream.push(AssistantMessageEvent::TextDelta {
                            content_index: idx,
                            delta: delta.to_string(),
                            partial: output.clone(),
                        });
                    }
                }

                "response.refusal.delta" => {
                    if let Ok(ref val) = parsed {
                        let output_index = val
                            .get("output_index")
                            .and_then(|i| i.as_u64())
                            .unwrap_or(0) as usize;
                        let delta = val.get("delta").and_then(|d| d.as_str()).unwrap_or("");

                        let idx = ensure_response_text_item(
                            &mut output,
                            &mut item_content_map,
                            &mut open_output_items,
                            output_index,
                            &stream,
                        );
                        if let Some(ContentBlock::Text(ref mut t)) = output.content.get_mut(idx) {
                            t.text.push_str(delta);
                        }
                        emitted_semantic_event = true;
                        stream.push(AssistantMessageEvent::TextDelta {
                            content_index: idx,
                            delta: delta.to_string(),
                            partial: output.clone(),
                        });
                    }
                }

                "response.function_call_arguments.delta" => {
                    if let Ok(ref val) = parsed {
                        let output_index = val
                            .get("output_index")
                            .and_then(|i| i.as_u64())
                            .unwrap_or(0) as usize;
                        let delta = val.get("delta").and_then(|d| d.as_str()).unwrap_or("");

                        // Auto-register if output_item.added was never received for this index
                        if let std::collections::hash_map::Entry::Vacant(e) =
                            item_content_map.entry(output_index)
                        {
                            let call_id = val
                                .get("call_id")
                                .or_else(|| val.get("item_id"))
                                .and_then(|c| c.as_str())
                                .unwrap_or("")
                                .to_string();
                            let name = val
                                .get("name")
                                .and_then(|n| n.as_str())
                                .unwrap_or("")
                                .to_string();
                            let item_id = val
                                .get("item_id")
                                .and_then(|i| i.as_str())
                                .unwrap_or("")
                                .to_string();
                            let composite_id = format!("{}|{}", call_id, item_id);
                            let content_idx = output.content.len();
                            output.content.push(ContentBlock::ToolCall(ToolCall::new(
                                &composite_id,
                                &name,
                                serde_json::Value::Object(serde_json::Map::new()),
                            )));
                            partial_tool_args.insert(output_index, String::new());
                            open_output_items.insert(output_index);
                            e.insert(ItemInfo {
                                content_idx,
                                item_type: ItemType::FunctionCall,
                                item_id,
                                call_id: Some(call_id),
                                name: Some(name),
                            });
                            emitted_semantic_event = true;
                            stream.push(AssistantMessageEvent::ToolCallStart {
                                content_index: content_idx,
                                partial: output.clone(),
                            });
                        }

                        if let Some(info) = item_content_map.get(&output_index) {
                            let idx = info.content_idx;
                            if let Some(ref mut args_str) = partial_tool_args.get_mut(&output_index)
                            {
                                args_str.push_str(delta);
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
                                delta: delta.to_string(),
                                partial: output.clone(),
                            });
                        }
                    }
                }

                "response.function_call_arguments.done" => {
                    if let Ok(ref val) = parsed {
                        let output_index = val
                            .get("output_index")
                            .and_then(|i| i.as_u64())
                            .unwrap_or(0) as usize;
                        let arguments = val
                            .get("arguments")
                            .and_then(|args| args.as_str())
                            .unwrap_or("");

                        if let std::collections::hash_map::Entry::Vacant(e) =
                            item_content_map.entry(output_index)
                        {
                            let call_id = val
                                .get("call_id")
                                .and_then(|c| c.as_str())
                                .unwrap_or("")
                                .to_string();
                            let name = val
                                .get("name")
                                .and_then(|n| n.as_str())
                                .unwrap_or("")
                                .to_string();
                            let item_id = val
                                .get("item_id")
                                .and_then(|i| i.as_str())
                                .unwrap_or("")
                                .to_string();
                            let composite_id = format!("{}|{}", call_id, item_id);
                            let content_idx = output.content.len();
                            output.content.push(ContentBlock::ToolCall(ToolCall::new(
                                &composite_id,
                                &name,
                                serde_json::Value::Object(serde_json::Map::new()),
                            )));
                            partial_tool_args.insert(output_index, String::new());
                            open_output_items.insert(output_index);
                            e.insert(ItemInfo {
                                content_idx,
                                item_type: ItemType::FunctionCall,
                                item_id,
                                call_id: Some(call_id),
                                name: Some(name),
                            });
                            emitted_semantic_event = true;
                            stream.push(AssistantMessageEvent::ToolCallStart {
                                content_index: content_idx,
                                partial: output.clone(),
                            });
                        }

                        if let Some(info) = item_content_map.get(&output_index) {
                            let idx = info.content_idx;
                            let previous = partial_tool_args
                                .get(&output_index)
                                .cloned()
                                .unwrap_or_default();
                            partial_tool_args.insert(output_index, arguments.to_string());
                            let parsed_args = parse_streaming_json(arguments);
                            if let Some(ContentBlock::ToolCall(ref mut tc)) =
                                output.content.get_mut(idx)
                            {
                                tc.arguments = parsed_args;
                            }
                            if let Some(delta) = arguments.strip_prefix(&previous) {
                                if !delta.is_empty() {
                                    emitted_semantic_event = true;
                                    stream.push(AssistantMessageEvent::ToolCallDelta {
                                        content_index: idx,
                                        delta: delta.to_string(),
                                        partial: output.clone(),
                                    });
                                }
                            }
                        }
                    }
                }

                "response.reasoning_summary_text.delta" => {
                    if let Ok(ref val) = parsed {
                        let output_index = val
                            .get("output_index")
                            .and_then(|i| i.as_u64())
                            .unwrap_or(0) as usize;
                        let delta = val.get("delta").and_then(|d| d.as_str()).unwrap_or("");

                        // Auto-register if output_item.added was never received for this index
                        if let std::collections::hash_map::Entry::Vacant(e) =
                            item_content_map.entry(output_index)
                        {
                            let content_idx = output.content.len();
                            output
                                .content
                                .push(ContentBlock::Thinking(ThinkingContent::new("")));
                            open_output_items.insert(output_index);
                            e.insert(ItemInfo {
                                content_idx,
                                item_type: ItemType::Reasoning,
                                item_id: String::new(),
                                call_id: None,
                                name: None,
                            });
                            emitted_semantic_event = true;
                            stream.push(AssistantMessageEvent::ThinkingStart {
                                content_index: content_idx,
                                partial: output.clone(),
                            });
                        }

                        if let Some(info) = item_content_map.get(&output_index) {
                            if info.item_type == ItemType::Reasoning {
                                let idx = info.content_idx;
                                if let Some(ContentBlock::Thinking(ref mut t)) =
                                    output.content.get_mut(idx)
                                {
                                    t.thinking.push_str(delta);
                                }
                                emitted_semantic_event = true;
                                stream.push(AssistantMessageEvent::ThinkingDelta {
                                    content_index: idx,
                                    delta: delta.to_string(),
                                    partial: output.clone(),
                                });
                            }
                        }
                    }
                }

                "response.reasoning_summary_part.added" => {}

                "response.reasoning_summary_part.done" => {
                    if let Ok(ref val) = parsed {
                        let output_index = val
                            .get("output_index")
                            .and_then(|i| i.as_u64())
                            .unwrap_or(0) as usize;

                        if let Some(info) = item_content_map.get(&output_index) {
                            if info.item_type == ItemType::Reasoning {
                                let idx = info.content_idx;
                                if let Some(ContentBlock::Thinking(ref mut t)) =
                                    output.content.get_mut(idx)
                                {
                                    t.thinking.push_str("\n\n");
                                }
                                emitted_semantic_event = true;
                                stream.push(AssistantMessageEvent::ThinkingDelta {
                                    content_index: idx,
                                    delta: "\n\n".to_string(),
                                    partial: output.clone(),
                                });
                            }
                        }
                    }
                }

                "response.output_item.done" => {
                    if let Ok(ref val) = parsed {
                        let output_index = val
                            .get("output_index")
                            .and_then(|i| i.as_u64())
                            .unwrap_or(0) as usize;
                        open_output_items.remove(&output_index);

                        if let Some(info) = item_content_map.get(&output_index) {
                            let idx = info.content_idx;
                            match info.item_type {
                                ItemType::Message => {
                                    if let Some(item) = val.get("item") {
                                        if let Some(response_text) =
                                            extract_response_message_text(item)
                                        {
                                            if let Some(ContentBlock::Text(ref mut t)) =
                                                output.content.get_mut(idx)
                                            {
                                                t.text = response_text;
                                                if let Some(item_id) =
                                                    item.get("id").and_then(|id| id.as_str())
                                                {
                                                    let phase = item
                                                        .get("phase")
                                                        .and_then(|phase| phase.as_str());
                                                    t.text_signature = Some(
                                                        encode_text_signature_v1(item_id, phase),
                                                    );
                                                }
                                            }
                                        }
                                    }

                                    let content = output
                                        .content
                                        .get(idx)
                                        .and_then(|b| b.as_text())
                                        .map(|t| t.text.clone())
                                        .unwrap_or_default();
                                    emitted_semantic_event = true;
                                    stream.push(AssistantMessageEvent::TextEnd {
                                        content_index: idx,
                                        content,
                                        partial: output.clone(),
                                    });
                                }
                                ItemType::FunctionCall => {
                                    if let Some(item) = val.get("item") {
                                        let final_arguments = partial_tool_args
                                            .get(&output_index)
                                            .and_then(|args| {
                                                if args.is_empty() {
                                                    None
                                                } else {
                                                    Some(args.as_str())
                                                }
                                            })
                                            .or_else(|| {
                                                item.get("arguments").and_then(|args| args.as_str())
                                            })
                                            .unwrap_or("{}");
                                        if let Some(ContentBlock::ToolCall(ref mut tc)) =
                                            output.content.get_mut(idx)
                                        {
                                            tc.arguments = parse_streaming_json(final_arguments);
                                            if let Some(name) =
                                                item.get("name").and_then(|name| name.as_str())
                                            {
                                                tc.name = name.to_string();
                                            }
                                            let call_id = item
                                                .get("call_id")
                                                .and_then(|call_id| call_id.as_str())
                                                .unwrap_or_default();
                                            let item_id = item
                                                .get("id")
                                                .and_then(|id| id.as_str())
                                                .unwrap_or_default();
                                            if !call_id.is_empty() || !item_id.is_empty() {
                                                tc.id = format!("{}|{}", call_id, item_id);
                                            }
                                        }
                                    } else if let Some(args_str) =
                                        partial_tool_args.get(&output_index)
                                    {
                                        if let Some(ContentBlock::ToolCall(ref mut tc)) =
                                            output.content.get_mut(idx)
                                        {
                                            tc.arguments = parse_streaming_json(args_str);
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
                                    partial_tool_args.remove(&output_index);
                                }
                                ItemType::Reasoning => {
                                    if let Some(item) = val.get("item") {
                                        if let Some(ContentBlock::Thinking(ref mut t)) =
                                            output.content.get_mut(idx)
                                        {
                                            let summary = item
                                                .get("summary")
                                                .and_then(|summary| summary.as_array())
                                                .map(|parts| {
                                                    parts
                                                        .iter()
                                                        .filter_map(|part| {
                                                            part.get("text")
                                                                .and_then(|text| text.as_str())
                                                        })
                                                        .collect::<Vec<_>>()
                                                        .join("\n\n")
                                                })
                                                .unwrap_or_default();
                                            if !summary.is_empty() {
                                                t.thinking = summary;
                                            }
                                            t.thinking_signature = Some(item.to_string());
                                        }
                                    }
                                    let content = output
                                        .content
                                        .get(idx)
                                        .and_then(|b| b.as_thinking())
                                        .map(|t| t.thinking.clone())
                                        .unwrap_or_default();
                                    emitted_semantic_event = true;
                                    stream.push(AssistantMessageEvent::ThinkingEnd {
                                        content_index: idx,
                                        content,
                                        partial: output.clone(),
                                    });
                                }
                            }
                        }
                    }
                }

                "response.completed" | "response.done" | "response.incomplete" => {
                    saw_response_completion = true;
                    // Try extracting from pre-parsed value, fall back to re-parsing data
                    let completed = parsed
                        .as_ref()
                        .ok()
                        .and_then(|v| serde_json::from_value::<ResponseCompleted>(v.clone()).ok())
                        .or_else(|| serde_json::from_str::<ResponseCompleted>(data).ok());
                    if let Some(completed) = completed {
                        if let Some(ref resp) = completed.response {
                            output.response_id = resp.id.clone().or(output.response_id);

                            // Update usage
                            if let Some(ref usage) = resp.usage {
                                let cached_tokens = usage
                                    .input_tokens_details
                                    .as_ref()
                                    .map(|details| details.cached_tokens)
                                    .unwrap_or(0);
                                output.usage.input =
                                    usage.input_tokens.saturating_sub(cached_tokens);
                                output.usage.output = usage.output_tokens;
                                output.usage.cache_read = cached_tokens;
                                output.usage.total_tokens = usage.total_tokens.unwrap_or(
                                    output.usage.input
                                        + output.usage.output
                                        + output.usage.cache_read,
                                );
                            }

                            // Update stop reason from status
                            if let Some(ref status) = resp.status {
                                output.stop_reason = match status.as_str() {
                                    "completed" => {
                                        if output.has_tool_calls() {
                                            StopReason::ToolUse
                                        } else {
                                            StopReason::Stop
                                        }
                                    }
                                    "incomplete" => StopReason::Length,
                                    "failed" | "cancelled" => StopReason::Error,
                                    _ => StopReason::Stop,
                                };
                            }
                        }
                    }
                }

                "error" | "response.failed" => {
                    if let Ok(ref val) = parsed {
                        let error_msg = val
                            .get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                            .or_else(|| val.get("message").and_then(|m| m.as_str()))
                            .unwrap_or("Unknown OpenAI error");
                        output.stop_reason = StopReason::Error;
                        output.error_message = Some(error_msg.to_string());
                        stream.push(AssistantMessageEvent::Error {
                            reason: StopReason::Error,
                            error: output,
                        });
                        stream.end(None);
                        return Ok(());
                    }
                }

                _ => {
                    // Ignore other events like response.created, etc.
                }
            }
        }
    }

    if let Some(detail) = incomplete_openai_responses_stream_detail(
        saw_response_completion,
        &open_output_items,
        &partial_tool_args,
        &line_buffer,
    ) {
        tracing::error!(
            url = %url,
            model = %model.id,
            detail = %detail,
            "OpenAI Responses stream ended before protocol completion"
        );
        super::common::emit_incomplete_stream_error(
            &mut output,
            "openai_responses",
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

/// Track information about output items.
#[derive(Debug, Clone)]
struct ItemInfo {
    content_idx: usize,
    item_type: ItemType,
    #[allow(dead_code)]
    item_id: String,
    #[allow(dead_code)]
    call_id: Option<String>,
    #[allow(dead_code)]
    name: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ItemType {
    Message,
    FunctionCall,
    Reasoning,
}

fn incomplete_openai_responses_stream_detail(
    saw_response_completion: bool,
    open_output_items: &HashSet<usize>,
    partial_tool_args: &HashMap<usize, String>,
    line_buffer: &str,
) -> Option<String> {
    let mut reasons = Vec::new();

    if !saw_response_completion {
        reasons.push("missing response.completed/response.done event".to_string());
    }

    // When response.completed/done was received, tolerate unfinished output
    // items — some proxies skip individual output_item.done events but still
    // deliver the terminal response event with full usage / status.
    if !saw_response_completion && !open_output_items.is_empty() {
        let mut indexes: Vec<_> = open_output_items.iter().copied().collect();
        indexes.sort_unstable();
        reasons.push(format!(
            "unfinished output items at indices [{}]",
            indexes
                .iter()
                .map(|index| index.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    // Tool-arg incompleteness is also gated on !saw_response_completion for
    // the same reason: a terminal event confirms the server considers the
    // response finished.
    let mut incomplete_tool_indexes: Vec<_> = if saw_response_completion {
        Vec::new()
    } else {
        open_output_items
            .iter()
            .copied()
            .filter(|index| {
                partial_tool_args.get(index).is_some_and(|args| {
                    let trimmed = args.trim();
                    !trimmed.is_empty()
                        && serde_json::from_str::<serde_json::Value>(trimmed).is_err()
                })
            })
            .collect()
    };
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

fn extract_response_message_text(item: &serde_json::Value) -> Option<String> {
    let content = item.get("content")?.as_array()?;
    Some(
        content
            .iter()
            .filter_map(|part| {
                match part
                    .get("type")
                    .and_then(|content_type| content_type.as_str())
                {
                    Some("output_text") => part.get("text").and_then(|text| text.as_str()),
                    Some("refusal") => part.get("refusal").and_then(|text| text.as_str()),
                    _ => None,
                }
            })
            .collect::<Vec<_>>()
            .join(""),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn test_provider_type() {
        let provider = OpenAIResponsesProtocol::new();
        assert_eq!(provider.provider_type(), Provider::OpenAIResponses);
    }

    #[test]
    fn test_convert_messages_basic() {
        let mut context = Context::with_system_prompt("You are helpful.");
        context.add_message(Message::User(UserMessage::text("Hello")));

        let model = Model::builder()
            .id("gpt-4o")
            .name("GPT-4o")
            .api(Api::OpenAIResponses)
            .provider(Provider::OpenAI)
            .context_window(128000)
            .max_tokens(16384)
            .build()
            .unwrap();

        let items = convert_messages(&context, &model);
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn test_convert_messages_strips_server_ids_when_store_is_false() {
        let model = Model::builder()
            .id("gpt-5")
            .name("GPT-5")
            .api(Api::OpenAIResponses)
            .provider(Provider::OpenAIResponses)
            .context_window(128000)
            .max_tokens(16384)
            .build()
            .unwrap();

        let assistant = AssistantMessage::builder()
            .api(Api::OpenAIResponses)
            .provider(Provider::OpenAIResponses)
            .model("gpt-5")
            .content(vec![ContentBlock::Text(TextContent {
                text: "hello".to_string(),
                text_signature: Some(encode_text_signature_v1("msg_123", Some("commentary"))),
            })])
            .build()
            .unwrap();

        let mut context = Context::new();
        context.add_message(Message::Assistant(assistant));
        let items = convert_messages(&context, &model);

        assert_eq!(items.len(), 1);
        // With store=false (default), server-side IDs should be stripped
        // to avoid "Item not found" errors.
        assert!(items[0].get("id").is_none() || items[0]["id"].is_null());
        assert!(items[0].get("phase").is_none() || items[0]["phase"].is_null());
    }

    #[test]
    fn test_convert_tool_call_composite_id() {
        let mut context = Context::new();
        context.add_message(Message::User(UserMessage::text("Hello")));

        // Create an assistant message with a tool call using composite ID
        let msg = AssistantMessage::builder()
            .api(Api::OpenAIResponses)
            .provider(Provider::OpenAI)
            .model("gpt-4o")
            .content(vec![ContentBlock::ToolCall(ToolCall::new(
                "call_abc|item_123",
                "get_weather",
                serde_json::json!({"city": "Tokyo"}),
            ))])
            .stop_reason(StopReason::ToolUse)
            .build()
            .unwrap();
        context.add_message(Message::Assistant(msg));

        // Add tool result
        context.add_message(Message::ToolResult(ToolResultMessage::text(
            "call_abc|item_123",
            "get_weather",
            "Sunny 25°C",
            false,
        )));

        let model = Model::builder()
            .id("gpt-4o")
            .name("GPT-4o")
            .api(Api::OpenAIResponses)
            .provider(Provider::OpenAI)
            .context_window(128000)
            .max_tokens(16384)
            .build()
            .unwrap();

        let items = convert_messages(&context, &model);
        assert_eq!(items.len(), 3); // user + function_call + function_call_output
        assert_eq!(items[1]["call_id"], serde_json::json!("call_abc"));
        assert_eq!(items[2]["call_id"], serde_json::json!("call_abc"));
        assert!(items[1].get("id").is_none());
    }

    #[test]
    fn test_convert_tool_call_composite_id_hashes_long_call_id() {
        let mut context = Context::new();
        let long_call_id = "helper_agentid_super_long_tool_call_id_1234567890_extra";
        let composite_id = format!("{}|fc_item_1234567890", long_call_id);
        context.add_message(Message::User(UserMessage::text("Hello")));
        context.add_message(Message::Assistant(
            AssistantMessage::builder()
                .api(Api::OpenAIResponses)
                .provider(Provider::OpenAI)
                .model("gpt-4o")
                .content(vec![ContentBlock::ToolCall(ToolCall::new(
                    &composite_id,
                    "get_weather",
                    serde_json::json!({"city": "Tokyo"}),
                ))])
                .stop_reason(StopReason::ToolUse)
                .build()
                .unwrap(),
        ));
        context.add_message(Message::ToolResult(ToolResultMessage::text(
            &composite_id,
            "get_weather",
            "Sunny 25°C",
            false,
        )));

        let model = Model::builder()
            .id("gpt-4o")
            .name("GPT-4o")
            .api(Api::OpenAIResponses)
            .provider(Provider::OpenAI)
            .context_window(128000)
            .max_tokens(16384)
            .build()
            .unwrap();

        let items = convert_messages(&context, &model);
        let function_call_call_id = items[1]["call_id"]
            .as_str()
            .expect("function_call call_id should be string");
        let function_call_output_call_id = items[2]["call_id"]
            .as_str()
            .expect("function_call_output call_id should be string");
        assert_eq!(function_call_call_id, function_call_output_call_id);
        assert!(function_call_call_id.starts_with("call_"));
        assert!(function_call_call_id.len() <= 40);
        assert_ne!(function_call_call_id, long_call_id);
        assert!(items[1].get("id").is_none());
    }

    #[test]
    fn test_normalize_responses_item_id_hashes_long_item_id() {
        let long_item_id =
            "foreign:item/with spaces/and/slashes/that/is/definitely/too/long/for/responses";
        let item_id = normalize_responses_item_id(long_item_id);
        assert!(item_id.starts_with("fc_"));
        assert!(item_id.len() <= 64);
        assert_ne!(item_id, long_item_id);
    }

    #[test]
    fn test_normalize_responses_item_id_hashes_long_prefixed_item_id_instead_of_truncating() {
        let long_prefixed_item_id = format!("fc_{}", "a".repeat(62));
        let item_id = normalize_responses_item_id(&long_prefixed_item_id);
        assert!(item_id.starts_with("fc_"));
        assert!(item_id.len() <= 64);
        assert_ne!(item_id, long_prefixed_item_id);
    }

    #[test]
    fn test_get_prompt_cache_retention_only_for_direct_openai() {
        assert_eq!(
            get_prompt_cache_retention("https://api.openai.com/v1", CacheRetention::Long),
            Some("24h".to_string())
        );
        assert_eq!(
            get_prompt_cache_retention("https://proxy.example.com/v1", CacheRetention::Long),
            None
        );
        assert_eq!(
            get_prompt_cache_retention("https://api.openai.com/v1", CacheRetention::Short),
            None
        );
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

    #[test]
    fn test_extract_response_message_text_includes_refusal_parts() {
        let item = serde_json::json!({
            "content": [
                { "type": "output_text", "text": "safe" },
                { "type": "refusal", "refusal": " no" }
            ]
        });

        assert_eq!(
            extract_response_message_text(&item).as_deref(),
            Some("safe no")
        );
    }

    #[test]
    fn test_incomplete_openai_responses_stream_detail_reports_missing_closure() {
        let mut open_output_items = HashSet::new();
        open_output_items.insert(2);
        let mut partial_tool_args = HashMap::new();
        partial_tool_args.insert(2, "{\"path\":\"logs".to_string());

        let detail = incomplete_openai_responses_stream_detail(
            false,
            &open_output_items,
            &partial_tool_args,
            "event: response.output_item.added",
        )
        .expect("detail");

        assert!(detail.contains("missing response.completed/response.done event"));
        assert!(detail.contains("unfinished output items at indices [2]"));
        assert!(detail.contains("unfinished tool input JSON at indices [2]"));
        assert!(detail.contains("trailing partial SSE frame"));
    }

    #[test]
    fn test_incomplete_openai_responses_completion_compensates_open_items() {
        let mut open_output_items = HashSet::new();
        open_output_items.insert(0);
        let mut partial_tool_args = HashMap::new();
        partial_tool_args.insert(0, "{\"path\":\"logs".to_string());

        // response.completed received — should tolerate open items and tool JSON
        let detail = incomplete_openai_responses_stream_detail(
            true,
            &open_output_items,
            &partial_tool_args,
            "",
        );
        assert!(
            detail.is_none(),
            "expected None when response.completed compensates, got: {:?}",
            detail
        );

        // trailing frame is still reported even with response.completed
        let detail = incomplete_openai_responses_stream_detail(
            true,
            &open_output_items,
            &partial_tool_args,
            "event: partial",
        );
        assert_eq!(detail.as_deref(), Some("trailing partial SSE frame"));
    }
}
