//! Shared infrastructure for protocol providers.
//!
//! Eliminates duplication across `openai_completions`, `anthropic`, `google`,
//! and `openai_responses` by extracting common patterns:
//! - Base URL resolution
//! - on_payload hook application
//! - URL validation with error event emission
//! - Debug preview truncation
//! - Custom header injection (H2)
//! - HTTP error response handling
//! - SSE line buffer limit checking
//! - Automatic retry with exponential backoff for transient HTTP errors

use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use futures::StreamExt;
use reqwest::header::HeaderMap;
use serde::Serialize;
use std::collections::HashMap;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Default maximum number of retries for transient HTTP errors.
pub const DEFAULT_MAX_RETRIES: u32 = 2;

/// Default maximum retry delay in milliseconds (30 seconds).
pub const DEFAULT_MAX_RETRY_DELAY_MS: u64 = 30_000;

/// Prefix used to mark provider-side streams that ended before protocol closure.
pub const INCOMPLETE_STREAM_ERROR_PREFIX: &str = "[incomplete_stream]";

/// Prefix used to mark transport-level stream errors (connection reset, body
/// decode failure, etc.) that interrupted an in-progress stream.
pub const TRANSPORT_ERROR_PREFIX: &str = "[transport_error]";

/// Base delay for exponential backoff in milliseconds.
const RETRY_BASE_DELAY_MS: u64 = 500;

/// Resolve the effective base URL using 3-level fallback:
/// `options.base_url` > `model.base_url` > `default`.
pub fn resolve_base_url<'a>(
    options_base_url: Option<&'a str>,
    model_base_url: Option<&'a str>,
    default: &'a str,
) -> &'a str {
    options_base_url.or(model_base_url).unwrap_or(default)
}

/// Apply the `on_payload` hook (if set) and serialize the request body.
///
/// When a hook is provided, the request is first serialized to `serde_json::Value`,
/// passed to the hook, and the (possibly modified) result is serialized to a JSON string.
/// Without a hook, the request is serialized directly to a JSON string.
pub async fn apply_on_payload<T: Serialize>(
    request: &T,
    hook: &Option<OnPayloadFn>,
    model: &Model,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    if let Some(ref hook) = hook {
        let request_json = serde_json::to_value(request)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?;
        match hook(request_json.clone(), model.clone()).await {
            Some(modified) => serde_json::to_string(&modified)
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) }),
            None => serde_json::to_string(&request_json)
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) }),
        }
    } else {
        serde_json::to_string(request)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })
    }
}

/// Validate the base URL against the security policy (H1).
///
/// On failure, pushes an `Error` event to the stream and returns `false`.
/// On success, returns `true`.
pub fn validate_url_or_error(
    base: &str,
    limits: &SecurityConfig,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
) -> bool {
    if let Err(e) = limits.url.validate(base) {
        tracing::error!(url = %base, error = %e, "Base URL validation failed");
        output.stop_reason = StopReason::Error;
        output.error_message = Some(format!("URL validation error: {}", e));
        stream.push(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: output.clone(),
        });
        stream.end(None);
        false
    } else {
        true
    }
}

/// Return a truncated preview of the body string for debug logging.
///
/// The truncation point is always clamped to a char boundary so multi-byte
/// characters (e.g. CJK) are never split.
pub fn debug_preview(body: &str, max_len: usize) -> &str {
    if body.len() <= max_len {
        return body;
    }
    // floor_char_boundary returns the nearest char boundary <= max_len.
    &body[..body.floor_char_boundary(max_len)]
}

/// OpenAI-style APIs reject very small output-token limits.
///
/// Clamp any explicit token limit below 16 up to 16 before serializing the
/// request payload. `None` is preserved as-is so providers can apply their own
/// defaults.
pub fn clamp_openai_max_tokens(max_tokens: Option<u32>) -> Option<u32> {
    max_tokens.map(|value| value.max(16))
}

/// OpenAI GPT-5.2 and later versioned GPT-5 models support the native `xhigh`
/// reasoning effort. Parse the dot-versioned model id by pattern instead of
/// enumerating every future release (for example `gpt-5.5`, `gpt-5.6`, ...).
pub(crate) fn supports_gpt5_xhigh(model_id: &str) -> bool {
    let normalized = model_id.to_ascii_lowercase();
    let mut search_start = 0;

    while let Some(relative_start) = normalized[search_start..].find("gpt-5") {
        let start = search_start + relative_start;
        let end = start + "gpt-5".len();

        let has_left_boundary = normalized[..start]
            .chars()
            .next_back()
            .map(|ch| !ch.is_ascii_alphanumeric())
            .unwrap_or(true);
        if !has_left_boundary {
            search_start = end;
            continue;
        }

        let Some(separator) = normalized[end..].chars().next() else {
            return false;
        };
        if separator != '.' {
            search_start = end;
            continue;
        }

        let minor_start = end + separator.len_utf8();
        let minor_digits = normalized[minor_start..]
            .chars()
            .take_while(|ch| ch.is_ascii_digit())
            .collect::<String>();
        if minor_digits.is_empty() {
            search_start = minor_start;
            continue;
        }

        if minor_digits.parse::<u32>().is_ok_and(|minor| minor >= 2) {
            return true;
        }

        search_start = minor_start + minor_digits.len();
    }

    false
}

/// Inject custom headers, skipping protected headers per security policy (H2).
pub fn apply_custom_headers(
    headers: &mut HeaderMap,
    custom: &Option<HashMap<String, String>>,
    policy: &HeaderPolicy,
) {
    if let Some(ref custom_headers) = custom {
        for (key, value) in custom_headers {
            if policy.is_protected(key) {
                tracing::warn!(header = %key, "Skipping protected header override");
                continue;
            }
            if let Ok(header_name) = reqwest::header::HeaderName::try_from(key.clone()) {
                if let Ok(header_value) = reqwest::header::HeaderValue::try_from(value.clone()) {
                    headers.insert(header_name, header_value);
                }
            }
        }
    }
}

/// Handle an HTTP error response: read the body (bounded), log it,
/// push an `Error` event to the stream.
///
/// Returns `true` to indicate that an error was handled (caller should return early).
pub async fn handle_error_response(
    response: reqwest::Response,
    url: &str,
    model: &Model,
    limits: &SecurityConfig,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
    provider_name: &str,
    request_body: &str,
) {
    let status = response.status();
    let body = crate::types::read_error_body(response, limits.http.max_error_body_bytes).await;
    tracing::error!(
        url = %url,
        model = %model.id,
        status = %status,
        response_body = %body,
        "{} request failed", provider_name
    );
    // Dump full request body on client errors (4xx) to aid debugging malformed payloads
    if status.is_client_error() {
        tracing::warn!(
            url = %url,
            model = %model.id,
            status = %status,
            request_body = %request_body,
            "{} client error request body dump", provider_name
        );
    }
    output.stop_reason = StopReason::Error;
    output.error_message = Some(crate::types::truncate_error_message(
        &format!("HTTP {}: {}", status, body),
        limits.http.max_error_message_chars,
    ));
    stream.push(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: output.clone(),
    });
    stream.end(None);
}

/// Check the SSE line buffer against the configured limit (C2).
///
/// On exceeding the limit, pushes an `Error` event to the stream and returns `true`
/// (indicating the stream should be aborted). Returns `false` if within limits.
pub fn check_sse_buffer_overflow(
    buffer_len: usize,
    max_bytes: usize,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
) -> bool {
    if buffer_len > max_bytes {
        tracing::error!(
            buffer_size = buffer_len,
            limit = max_bytes,
            "SSE line buffer exceeded limit, aborting stream"
        );
        output.stop_reason = StopReason::Error;
        output.error_message = Some("SSE line buffer exceeded maximum size".to_string());
        stream.push(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: output.clone(),
        });
        stream.end(None);
        true
    } else {
        false
    }
}

/// Emit an aborted terminal assistant message and close the stream.
pub fn emit_aborted(output: &mut AssistantMessage, stream: &AssistantMessageEventStream) {
    output.stop_reason = StopReason::Aborted;
    output.error_message = Some("Aborted".to_string());
    stream.push(AssistantMessageEvent::Error {
        reason: StopReason::Aborted,
        error: output.clone(),
    });
    stream.end(None);
}

/// Await an HTTP request, but abort early when the cancellation token fires.
pub async fn send_request_with_cancel(
    request: reqwest::RequestBuilder,
    cancel_token: Option<&CancellationToken>,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
) -> Result<Option<reqwest::Response>, reqwest::Error> {
    if let Some(cancel_token) = cancel_token {
        tokio::select! {
            _ = cancel_token.cancelled() => {
                emit_aborted(output, stream);
                Ok(None)
            }
            response = request.send() => response.map(Some),
        }
    } else {
        request.send().await.map(Some)
    }
}

/// Await the next item from a byte stream, but abort early when cancelled.
pub async fn next_stream_item_with_cancel<S, T, E>(
    source: &mut S,
    cancel_token: Option<&CancellationToken>,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
) -> Option<Result<T, E>>
where
    S: futures::Stream<Item = Result<T, E>> + Unpin,
{
    if let Some(cancel_token) = cancel_token {
        tokio::select! {
            _ = cancel_token.cancelled() => {
                emit_aborted(output, stream);
                None
            }
            item = source.next() => item,
        }
    } else {
        source.next().await
    }
}

// ============================================================================
// Retry infrastructure
// ============================================================================

/// Check whether an HTTP status code is transient and worth retrying.
pub fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 408 | 429 | 500 | 502 | 503 | 504)
}

/// Check whether a `reqwest::Error` represents a transient failure worth retrying.
pub fn is_retryable_error(err: &reqwest::Error) -> bool {
    err.is_timeout() || err.is_connect()
}

/// Check whether a streamed response body error is transient and worth retrying.
///
/// Before any semantic events have been emitted, body-read failures are treated
/// as retryable because they are typically transport interruptions rather than
/// application-level protocol errors.
pub fn is_retryable_stream_error(err: &reqwest::Error) -> bool {
    is_retryable_error(err) || err.is_body()
}

/// Parse the `Retry-After` header from an HTTP response.
///
/// Supports both "delay-seconds" (e.g. `5`) and HTTP-date formats.
/// Returns `None` if the header is missing or unparseable.
pub fn parse_retry_after(response: &reqwest::Response) -> Option<Duration> {
    let value = response.headers().get("retry-after")?.to_str().ok()?;

    // Try parsing as integer seconds first (most common for API rate limits).
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }

    // Try HTTP-date format: e.g. "Wed, 21 Oct 2025 07:28:00 GMT"
    if let Ok(date) = httpdate::parse_http_date(value) {
        let now = std::time::SystemTime::now();
        if let Ok(delta) = date.duration_since(now) {
            return Some(delta);
        }
        // Date is in the past — retry immediately.
        return Some(Duration::ZERO);
    }

    None
}

/// Compute the retry delay using exponential backoff with jitter.
///
/// Formula: `min(base_ms × 2^attempt, max_delay_ms)` + random jitter (0–25%).
pub fn compute_retry_delay(attempt: u32, max_delay_ms: u64) -> Duration {
    let exp_delay = RETRY_BASE_DELAY_MS.saturating_mul(1u64 << attempt.min(10));
    let capped = if max_delay_ms == 0 {
        exp_delay
    } else {
        exp_delay.min(max_delay_ms)
    };

    // Add 0–25% random jitter to avoid thundering herd.
    let jitter_range = capped / 4;
    let jitter = if jitter_range > 0 {
        // Simple deterministic-ish jitter using the attempt number and current
        // time nanoseconds. This is NOT cryptographically random, but perfectly
        // fine for retry jitter.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos() as u64;
        nanos % jitter_range
    } else {
        0
    };

    Duration::from_millis(capped + jitter)
}

/// Apply the optional retry-delay cap. A value of `0` disables the cap.
pub fn cap_retry_delay(delay: Duration, max_delay_ms: u64) -> Duration {
    if max_delay_ms == 0 {
        delay
    } else {
        delay.min(Duration::from_millis(max_delay_ms))
    }
}

/// Sleep for the given duration, but abort early if the cancellation token fires.
///
/// Returns `true` if cancelled, `false` if the sleep completed normally.
pub async fn sleep_with_cancel(
    duration: Duration,
    cancel_token: Option<&CancellationToken>,
) -> bool {
    if let Some(cancel_token) = cancel_token {
        tokio::select! {
            _ = cancel_token.cancelled() => true,
            _ = tokio::time::sleep(duration) => false,
        }
    } else {
        tokio::time::sleep(duration).await;
        false
    }
}

/// Emit a terminal error assistant message and close the stream.
pub fn emit_terminal_error(
    output: &mut AssistantMessage,
    error_message: impl Into<String>,
    max_error_message_chars: usize,
    stream: &AssistantMessageEventStream,
) {
    if output.content.is_empty() {
        output.content = vec![ContentBlock::Text(TextContent::new(""))];
    }
    output.stop_reason = StopReason::Error;
    output.error_message = Some(crate::types::truncate_error_message(
        &error_message.into(),
        max_error_message_chars,
    ));
    stream.push(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: output.clone(),
    });
    stream.end(None);
}

/// Emit a terminal error for a semantically incomplete provider stream.
pub fn emit_incomplete_stream_error(
    output: &mut AssistantMessage,
    provider: &str,
    detail: impl Into<String>,
    max_error_message_chars: usize,
    stream: &AssistantMessageEventStream,
) {
    let detail = detail.into();
    emit_terminal_error(
        output,
        format!(
            "{INCOMPLETE_STREAM_ERROR_PREFIX}{provider}: {}",
            crate::types::truncate_error_message(&detail, max_error_message_chars)
        ),
        max_error_message_chars,
        stream,
    );
}

/// Parse an incomplete-stream error marker back into `(provider, detail)`.
pub fn parse_incomplete_stream_error(error_message: &str) -> Option<(String, String)> {
    let payload = error_message.strip_prefix(INCOMPLETE_STREAM_ERROR_PREFIX)?;
    let (provider, detail) = payload.split_once(':')?;
    Some((provider.trim().to_string(), detail.trim().to_string()))
}

/// Emit a terminal error for a transport-level stream failure (connection
/// reset, body decode error, timeout mid-stream, etc.).
///
/// The error message is prefixed with [`TRANSPORT_ERROR_PREFIX`] so the agent
/// layer can distinguish transport errors from provider API errors and apply
/// turn-level retry logic.
pub fn emit_transport_stream_error(
    output: &mut AssistantMessage,
    provider: &str,
    detail: impl Into<String>,
    max_error_message_chars: usize,
    stream: &AssistantMessageEventStream,
) {
    let detail = detail.into();
    emit_terminal_error(
        output,
        format!(
            "{TRANSPORT_ERROR_PREFIX}{provider}: {}",
            crate::types::truncate_error_message(&detail, max_error_message_chars)
        ),
        max_error_message_chars,
        stream,
    );
}

/// Parse a transport-error marker back into `(provider, detail)`.
pub fn parse_transport_stream_error(error_message: &str) -> Option<(String, String)> {
    let payload = error_message.strip_prefix(TRANSPORT_ERROR_PREFIX)?;
    let (provider, detail) = payload.split_once(':')?;
    Some((provider.trim().to_string(), detail.trim().to_string()))
}

/// Emit `ThinkingEnd` and/or `TextEnd` events for any open content blocks
/// before an error or incomplete-stream event is pushed.
///
/// This ensures downstream consumers always see a matching end event for
/// every start event, even when the stream is interrupted by a transport
/// error, timeout, or protocol-level error.
///
/// The function is idempotent: passing `None` for an index means that block
/// type is not currently open.
pub fn emit_pending_block_ends(
    stream: &AssistantMessageEventStream,
    output: &AssistantMessage,
    open_thinking_index: Option<usize>,
    open_text_index: Option<usize>,
) {
    if let Some(idx) = open_thinking_index {
        emit_thinking_end(stream, output, idx);
    }
    if let Some(idx) = open_text_index {
        emit_text_end(stream, output, idx);
    }
}

/// Emit end events for multiple open thinking/text blocks (sorted by index).
///
/// Used when a protocol supports interleaved blocks (e.g., Anthropic's
/// interleaved thinking beta) where more than one block of the same type
/// may be open simultaneously.
pub fn emit_pending_block_ends_multi(
    stream: &AssistantMessageEventStream,
    output: &AssistantMessage,
    mut open_thinking_indices: Vec<usize>,
    mut open_text_indices: Vec<usize>,
) {
    open_thinking_indices.sort_unstable();
    open_text_indices.sort_unstable();

    for idx in open_thinking_indices {
        emit_thinking_end(stream, output, idx);
    }
    for idx in open_text_indices {
        emit_text_end(stream, output, idx);
    }
}

fn emit_thinking_end(stream: &AssistantMessageEventStream, output: &AssistantMessage, idx: usize) {
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

fn emit_text_end(stream: &AssistantMessageEventStream, output: &AssistantMessage, idx: usize) {
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

/// Emit a terminal background-task error unless the stream has already ended.
pub fn emit_background_task_error(
    model: &Model,
    fallback_api: Api,
    error_message: impl Into<String>,
    stream: &AssistantMessageEventStream,
) {
    if stream.is_done() {
        return;
    }

    let mut output = AssistantMessage::builder()
        .api(model.api.clone().unwrap_or(fallback_api))
        .provider(model.provider.clone())
        .model(model.id.clone())
        .usage(Usage::default())
        .stop_reason(StopReason::Error)
        .build()
        .expect("background task error message should be buildable");
    output.content = vec![ContentBlock::Text(TextContent::new(""))];
    emit_terminal_error(&mut output, error_message, 4096, stream);
}

/// Send an HTTP POST request with automatic retry on transient errors.
///
/// Rebuilds the request from components on each attempt. Retries on:
/// - HTTP 408 (Request Timeout), 429 (Too Many Requests), 500, 502, 503, 504
/// - `reqwest::Error` with `is_timeout()` or `is_connect()`
///
/// Uses exponential backoff with jitter, respects `Retry-After` headers, and
/// honours the cancellation token during both request sending and backoff sleep.
///
/// Returns:
/// - `Ok(Some(response))` — a response was received (may be success or a
///   non-retryable error status; the caller should check `.status()`)
/// - `Ok(None)` — request was cancelled via the token
/// - `Err(e)` — a non-retryable transport error
pub async fn send_request_with_retry(
    client: &reqwest::Client,
    url: &str,
    headers: HeaderMap,
    body: String,
    timeout: Duration,
    max_retries: u32,
    max_retry_delay_ms: u64,
    cancel_token: Option<&CancellationToken>,
    output: &mut AssistantMessage,
    stream: &AssistantMessageEventStream,
) -> Result<Option<reqwest::Response>, reqwest::Error> {
    let mut attempt: u32 = 0;

    loop {
        let request = client
            .post(url)
            .timeout(timeout)
            .headers(headers.clone())
            .body(body.clone());

        match send_request_with_cancel(request, cancel_token, output, stream).await {
            Ok(None) => {
                // Cancelled.
                return Ok(None);
            }
            Ok(Some(response)) => {
                let status = response.status();
                if is_retryable_status(status) && attempt < max_retries {
                    let delay = parse_retry_after(&response)
                        .map(|d| cap_retry_delay(d, max_retry_delay_ms))
                        .unwrap_or_else(|| compute_retry_delay(attempt, max_retry_delay_ms));

                    stream.push(AssistantMessageEvent::Retrying {
                        attempt: attempt + 1,
                        max_retries,
                        delay_ms: delay.as_millis() as u64,
                        reason: format!("HTTP {}", status),
                        status: Some(status.as_u16()),
                    });

                    tracing::warn!(
                        url = %url,
                        status = %status,
                        attempt = attempt + 1,
                        max_retries = max_retries,
                        delay_ms = delay.as_millis() as u64,
                        "Retryable HTTP status, backing off before retry"
                    );

                    if sleep_with_cancel(delay, cancel_token).await {
                        emit_aborted(output, stream);
                        return Ok(None);
                    }

                    attempt += 1;
                    continue;
                }

                // Either success or non-retryable status — return to caller.
                return Ok(Some(response));
            }
            Err(e) => {
                if is_retryable_error(&e) && attempt < max_retries {
                    let delay = compute_retry_delay(attempt, max_retry_delay_ms);

                    stream.push(AssistantMessageEvent::Retrying {
                        attempt: attempt + 1,
                        max_retries,
                        delay_ms: delay.as_millis() as u64,
                        reason: e.to_string(),
                        status: None,
                    });

                    tracing::warn!(
                        url = %url,
                        error = %e,
                        attempt = attempt + 1,
                        max_retries = max_retries,
                        delay_ms = delay.as_millis() as u64,
                        "Retryable transport error, backing off before retry"
                    );

                    if sleep_with_cancel(delay, cancel_token).await {
                        emit_aborted(output, stream);
                        return Ok(None);
                    }

                    attempt += 1;
                    continue;
                }

                // Non-retryable error or retries exhausted.
                return Err(e);
            }
        }
    }
}

// ============================================================================
// Reasoning Content Normalization (shared across protocols)
// ============================================================================

/// Normalize reasoning/thinking content in messages for constrained providers
/// (e.g., DeepSeek API and third-party providers forwarding DeepSeek models).
///
/// * `reasoning_content_constrained` — when true, enables normalization.
/// * `thinking_enabled` — when true, backfills missing thinking blocks from the
///   most recent assistant message that has one, and ensures every assistant
///   message with thinking also has a (possibly empty) text content.
/// * `thinking_enabled` — when false, strips all thinking blocks.
///
/// Normalization is applied when:
/// - `reasoning_content_constrained` is true (provider `default_compat()` or catalog patches), or
/// - the base_url matches `api.deepseek.com` (custom openai-compatible provider).
pub(crate) fn normalize_reasoning_content(
    messages: Vec<Message>,
    reasoning_content_constrained: bool,
    thinking_enabled: bool,
    base_url: &str,
) -> Vec<Message> {
    let constrained = reasoning_content_constrained || base_url.contains("api.deepseek.com");

    if !constrained {
        return messages;
    }

    let mut normalized = Vec::with_capacity(messages.len());
    let mut last_thinking: Option<ThinkingContent> = None;

    for msg in messages {
        match msg {
            Message::Assistant(mut assistant) => {
                if thinking_enabled {
                    // Track the most recent non-empty thinking block for backfilling
                    let has_thinking = assistant.content.iter().any(
                        |b| matches!(b, ContentBlock::Thinking(t) if !t.thinking.trim().is_empty()),
                    );

                    if has_thinking {
                        // Store the most recent non-empty thinking block as the backfill
                        // candidate for subsequent assistant messages that lack reasoning.
                        for block in &assistant.content {
                            if let ContentBlock::Thinking(t) = block {
                                if !t.thinking.trim().is_empty() {
                                    last_thinking = Some(t.clone());
                                    break;
                                }
                            }
                        }
                    } else if let Some(ref thinking) = last_thinking {
                        // Backfill: insert a copy of the most recent thinking block at the front
                        assistant
                            .content
                            .insert(0, ContentBlock::Thinking(thinking.clone()));
                    }

                    // Ensure content is not "null": if we have thinking but no text content,
                    // insert an empty text block for provider compatibility
                    let has_text = assistant
                        .content
                        .iter()
                        .any(|b| matches!(b, ContentBlock::Text(_)));
                    let has_any_thinking = assistant
                        .content
                        .iter()
                        .any(|b| matches!(b, ContentBlock::Thinking(_)));
                    if has_any_thinking && !has_text {
                        assistant
                            .content
                            .push(ContentBlock::Text(TextContent::new("")));
                    }
                } else {
                    // Thinking disabled: remove all thinking blocks
                    assistant
                        .content
                        .retain(|b| !matches!(b, ContentBlock::Thinking(_)));
                }
                normalized.push(Message::Assistant(assistant));
            }
            other => {
                normalized.push(other);
            }
        }
    }

    normalized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_gpt5_xhigh_matches_versioned_gpt5_2_and_later() {
        assert!(supports_gpt5_xhigh("gpt-5.2"));
        assert!(supports_gpt5_xhigh("gpt-5.5"));
        assert!(supports_gpt5_xhigh("openai/gpt-5.6-mini"));
        assert!(supports_gpt5_xhigh("GPT-5.10-CODEX"));

        assert!(!supports_gpt5_xhigh("gpt-5"));
        assert!(!supports_gpt5_xhigh("gpt-5.1"));
        assert!(!supports_gpt5_xhigh("gpt-5-6"));
        assert!(!supports_gpt5_xhigh("notgpt-5.6"));
    }

    #[test]
    fn test_compute_retry_delay_zero_cap_disables_capping() {
        let delay = compute_retry_delay(1, 0);
        assert!(
            delay >= Duration::from_millis(RETRY_BASE_DELAY_MS * 2),
            "zero cap should not collapse retry delay to zero"
        );
    }

    #[test]
    fn test_cap_retry_delay_zero_cap_is_unbounded() {
        let delay = Duration::from_secs(5);
        assert_eq!(cap_retry_delay(delay, 0), delay);
    }

    #[test]
    fn test_parse_transport_stream_error_extracts_provider_and_detail() {
        let msg = "[transport_error]anthropic: error decoding response body";
        let (provider, detail) = parse_transport_stream_error(msg).unwrap();
        assert_eq!(provider, "anthropic");
        assert_eq!(detail, "error decoding response body");
    }

    #[test]
    fn test_parse_transport_stream_error_returns_none_for_non_transport() {
        assert!(parse_transport_stream_error("Provider error: something").is_none());
        assert!(
            parse_transport_stream_error("[incomplete_stream]anthropic: missing message_stop")
                .is_none()
        );
    }

    #[test]
    fn test_parse_incomplete_and_transport_are_mutually_exclusive() {
        let transport_msg = "[transport_error]google: connection reset";
        let incomplete_msg = "[incomplete_stream]google: missing message_stop";

        // transport message should only match transport parser
        assert!(parse_transport_stream_error(transport_msg).is_some());
        assert!(parse_incomplete_stream_error(transport_msg).is_none());

        // incomplete message should only match incomplete parser
        assert!(parse_incomplete_stream_error(incomplete_msg).is_some());
        assert!(parse_transport_stream_error(incomplete_msg).is_none());
    }
}
