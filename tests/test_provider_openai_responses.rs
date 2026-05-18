//! Tests for OpenAI Responses API provider using wiremock for HTTP mocking.

use futures::StreamExt;
use parking_lot::Mutex;
use serde_json::json;
use std::sync::Arc;
use tiycore::protocol::openai_responses::OpenAIResponsesProtocol;
use tiycore::protocol::LLMProtocol;
use tiycore::types::*;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ============================================================================
// Helper functions
// ============================================================================

fn make_model(base_url: &str) -> Model {
    Model::builder()
        .id("gpt-4o")
        .name("GPT-4o")
        .api(Api::OpenAIResponses)
        .provider(Provider::OpenAI)
        .base_url(base_url)
        .input(vec![InputType::Text, InputType::Image])
        .context_window(128000)
        .max_tokens(16384)
        .build()
        .unwrap()
}

fn make_context(system_prompt: &str, user_msg: &str) -> Context {
    let mut ctx = Context::with_system_prompt(system_prompt);
    ctx.add_message(Message::User(UserMessage::text(user_msg)));
    ctx
}

fn make_options(api_key: &str) -> StreamOptions {
    StreamOptions {
        api_key: Some(api_key.to_string()),
        ..Default::default()
    }
}

fn make_options_with_capture(
    api_key: &str,
    captured: Arc<Mutex<Option<serde_json::Value>>>,
) -> StreamOptions {
    let mut options = make_options(api_key);
    options.on_payload = Some(Arc::new(move |payload, _model| {
        let captured = captured.clone();
        Box::pin(async move {
            *captured.lock() = Some(payload.clone());
            Some(payload)
        })
    }));
    options
}

/// Build an SSE body from a list of (event_type, data_json) pairs.
/// The OpenAI Responses API uses typed `event:` lines unlike the Completions API.
fn responses_sse(events: Vec<(&str, &str)>) -> String {
    events
        .iter()
        .map(|(event_type, data)| format!("event: {}\ndata: {}\n\n", event_type, data))
        .collect::<String>()
}

// ============================================================================
// Provider unit tests
// ============================================================================

#[test]
fn test_provider_type() {
    let provider = OpenAIResponsesProtocol::new();
    assert_eq!(provider.provider_type(), Provider::OpenAIResponses);
}

// ============================================================================
// Streaming integration tests with wiremock
// ============================================================================

#[tokio::test]
async fn test_stream_simple_text_response() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        (
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_01",
                    "role": "assistant",
                    "content": []
                }
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hello world!"
            })
            .to_string(),
        ),
        (
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_01"
                }
            })
            .to_string(),
        ),
        (
            "response.completed",
            &json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_01",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15
                    },
                    "output": [
                        {"type": "message", "id": "item_01"}
                    ]
                }
            })
            .to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("You are helpful.", "Hello");
    let options = make_options("test-key");

    let mut stream = provider.stream(&model, &context, options);

    // Collect all streamed events
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    // Should have: Start, TextStart, TextDelta, TextEnd, Done
    assert!(!events.is_empty());

    // Check Start event
    assert!(matches!(&events[0], AssistantMessageEvent::Start { .. }));

    // Check that text deltas are present
    let text_deltas: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, AssistantMessageEvent::TextDelta { .. }))
        .collect();
    assert!(!text_deltas.is_empty());

    // Verify via result
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Hello world!");
}

#[tokio::test]
async fn test_stream_reports_incomplete_stream_for_truncated_response() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        (
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_01",
                    "role": "assistant",
                    "content": []
                }
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hello world!"
            })
            .to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("You are helpful.", "Hello");
    let options = make_options("test-key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert!(result
        .error_message
        .as_deref()
        .is_some_and(|message| message.contains("[incomplete_stream]openai_responses:")));
    assert!(result
        .error_message
        .as_deref()
        .is_some_and(|message| message.contains("missing response.completed/response.done event")));
}

#[tokio::test]
async fn test_stream_retries_retryable_http_status_before_streaming() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        (
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_retry",
                    "role": "assistant",
                    "content": []
                }
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "Retried successfully"
            })
            .to_string(),
        ),
        (
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_retry"
                }
            })
            .to_string(),
        ),
        (
            "response.completed",
            &json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_retry",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15
                    },
                    "output": [
                        {"type": "message", "id": "item_retry"}
                    ]
                }
            })
            .to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(
            ResponseTemplate::new(503)
                .insert_header("retry-after", "0")
                .set_body_string("try again"),
        )
        .with_priority(1)
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .with_priority(2)
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("You are helpful.", "Hello");
    let mut options = make_options("test-key");
    options.max_retries = Some(1);
    options.max_retry_delay_ms = Some(10);

    let mut stream = provider.stream(&model, &context, options);
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    let retrying = events
        .iter()
        .find_map(|event| match event {
            AssistantMessageEvent::Retrying {
                attempt,
                max_retries,
                delay_ms,
                reason,
                status,
            } => Some((*attempt, *max_retries, *delay_ms, reason.clone(), *status)),
            _ => None,
        })
        .expect("expected retrying event after initial 503");
    assert_eq!(retrying.0, 1);
    assert_eq!(retrying.1, 1);
    assert_eq!(retrying.2, 0);
    assert!(retrying.3.contains("HTTP 503"));
    assert_eq!(retrying.4, Some(503));

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Retried successfully");

    let requests = server.received_requests().await.expect("received requests");
    assert_eq!(
        requests.len(),
        2,
        "expected one retry after the initial 503"
    );
}

#[tokio::test]
async fn test_stream_clamps_small_max_output_tokens_to_minimum() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        (
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_01",
                    "role": "assistant",
                    "content": []
                }
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "ok"
            })
            .to_string(),
        ),
        (
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_01"
                }
            })
            .to_string(),
        ),
        (
            "response.completed",
            &json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_01",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "total_tokens": 12
                    },
                    "output": [
                        {"type": "message", "id": "item_01"}
                    ]
                }
            })
            .to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_partial_json(json!({
            "max_output_tokens": 16
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("You are helpful.", "Hello");
    let mut options = make_options("test-key");
    options.max_tokens = Some(8);

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "ok");
}

#[tokio::test]
async fn test_stream_with_tool_call() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        (
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "item_02",
                    "call_id": "call_abc123",
                    "name": "get_weather",
                    "arguments": ""
                }
            })
            .to_string(),
        ),
        (
            "response.function_call_arguments.delta",
            &json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": "{\"city\":"
            })
            .to_string(),
        ),
        (
            "response.function_call_arguments.delta",
            &json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": " \"Tokyo\"}"
            })
            .to_string(),
        ),
        (
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "item_02",
                    "call_id": "call_abc123",
                    "name": "get_weather"
                }
            })
            .to_string(),
        ),
        (
            "response.completed",
            &json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_01",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 15,
                        "total_tokens": 35
                    },
                    "output": [
                        {
                            "type": "function_call",
                            "id": "item_02",
                            "call_id": "call_abc123",
                            "name": "get_weather"
                        }
                    ]
                }
            })
            .to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let mut context = make_context("You are helpful.", "What's the weather in Tokyo?");
    context.set_tools(vec![Tool::new(
        "get_weather",
        "Get weather",
        json!({"type": "object", "properties": {"city": {"type": "string"}}}),
    )]);
    let options = make_options("test-key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::ToolUse);
    assert!(result.has_tool_calls());
    let tool_calls = result.tool_calls();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].name, "get_weather");
    // The provider creates composite IDs: "{call_id}|{item_id}"
    assert!(tool_calls[0].id.contains("call_abc123"));
    assert_eq!(tool_calls[0].arguments["city"], "Tokyo");
}

#[tokio::test]
async fn test_stream_sends_service_tier_and_respects_cache_retention_none() {
    let server = MockServer::start().await;
    let captured = Arc::new(Mutex::new(None));

    let sse_body = responses_sse(vec![
        (
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_09",
                    "role": "assistant",
                    "content": []
                }
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "ok"
            })
            .to_string(),
        ),
        (
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_09"
                }
            })
            .to_string(),
        ),
        (
            "response.completed",
            &json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_09",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "total_tokens": 12
                    },
                    "output": [
                        {"type": "message", "id": "item_09"}
                    ]
                }
            })
            .to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("You are helpful.", "Hello");
    let mut options = make_options_with_capture("test-key", captured.clone());
    options.session_id = Some("sess_123".to_string());
    options.cache_retention = Some(CacheRetention::None);
    options.service_tier = Some(OpenAIServiceTier::Flex);

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);

    let payload = captured.lock().clone().expect("payload captured");
    assert_eq!(payload["service_tier"], json!("flex"));
    assert!(payload["prompt_cache_key"].is_null());
    assert!(payload["prompt_cache_retention"].is_null());
}

#[tokio::test]
async fn test_stream_http_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(401)
                .set_body_string(r#"{"error": {"message": "Invalid API key"}}"#),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("You are helpful.", "Hello");
    let options = make_options("invalid-key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert!(result.error_message.is_some());
    assert!(result.error_message.unwrap().contains("401"));
}

#[tokio::test]
async fn test_stream_with_thinking() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        // First: reasoning item added
        (
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "item_03"
                }
            })
            .to_string(),
        ),
        // Reasoning summary text delta
        (
            "response.reasoning_summary_text.delta",
            &json!({
                "type": "response.reasoning_summary_text.delta",
                "output_index": 0,
                "summary_index": 0,
                "delta": "Let me think"
            })
            .to_string(),
        ),
        (
            "response.reasoning_summary_text.delta",
            &json!({
                "type": "response.reasoning_summary_text.delta",
                "output_index": 0,
                "summary_index": 0,
                "delta": " about this..."
            })
            .to_string(),
        ),
        // Reasoning item done
        (
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "item_03"
                }
            })
            .to_string(),
        ),
        // Then: message item added
        (
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "type": "message",
                    "id": "item_04",
                    "role": "assistant",
                    "content": []
                }
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 1,
                "content_index": 0,
                "delta": "The answer is 42."
            })
            .to_string(),
        ),
        (
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "type": "message",
                    "id": "item_04"
                }
            })
            .to_string(),
        ),
        (
            "response.completed",
            &json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_01",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 20,
                        "total_tokens": 30
                    },
                    "output": [
                        {"type": "reasoning", "id": "item_03"},
                        {"type": "message", "id": "item_04"}
                    ]
                }
            })
            .to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("You are helpful.", "What is the meaning of life?");
    let options = make_options("test-key");

    let mut stream = provider.stream(&model, &context, options);

    // Collect all events to verify thinking events are emitted
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    // Verify thinking events are present
    let thinking_deltas: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, AssistantMessageEvent::ThinkingDelta { .. }))
        .collect();
    assert!(
        !thinking_deltas.is_empty(),
        "Should have thinking delta events"
    );

    // Verify thinking start/end events
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AssistantMessageEvent::ThinkingStart { .. })),
        "Should have ThinkingStart event"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AssistantMessageEvent::ThinkingEnd { .. })),
        "Should have ThinkingEnd event"
    );

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "The answer is 42.");
    assert!(result
        .thinking_content()
        .contains("Let me think about this..."));
}

#[tokio::test]
async fn test_stream_usage_tracking() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        (
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_01",
                    "role": "assistant",
                    "content": []
                }
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hi"
            })
            .to_string(),
        ),
        (
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_01"
                }
            })
            .to_string(),
        ),
        (
            "response.completed",
            &json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_01",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "total_tokens": 150
                    },
                    "output": [
                        {"type": "message", "id": "item_01"}
                    ]
                }
            })
            .to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.usage.input, 100);
    assert_eq!(result.usage.output, 50);
    assert_eq!(result.usage.total_tokens, 150);
}

#[tokio::test]
async fn test_stream_incomplete_stop_reason() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        (
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_01",
                    "role": "assistant",
                    "content": []
                }
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "truncated"
            })
            .to_string(),
        ),
        (
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_01"
                }
            })
            .to_string(),
        ),
        (
            "response.completed",
            &json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_01",
                    "status": "incomplete",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 100,
                        "total_tokens": 110
                    }
                }
            })
            .to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;

    assert_eq!(result.stop_reason, StopReason::Length);
    assert_eq!(result.text_content(), "truncated");
    assert_eq!(result.usage.input, 10);
    assert_eq!(result.usage.output, 100);
    assert_eq!(result.usage.total_tokens, 110);
}

#[tokio::test]
async fn test_stream_multiple_text_deltas() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        (
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_01",
                    "role": "assistant",
                    "content": []
                }
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hello"
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": " "
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "world"
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "!"
            })
            .to_string(),
        ),
        (
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "item_01"
                }
            })
            .to_string(),
        ),
        (
            "response.completed",
            &json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_01",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 8,
                        "total_tokens": 18
                    },
                    "output": [
                        {"type": "message", "id": "item_01"}
                    ]
                }
            })
            .to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("You are helpful.", "Hello");
    let options = make_options("test-key");

    let mut stream = provider.stream(&model, &context, options);

    // Collect all events
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    // Verify we got 4 text deltas
    let text_deltas: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AssistantMessageEvent::TextDelta { delta, .. } => Some(delta.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(text_deltas.len(), 4);
    assert_eq!(text_deltas[0], "Hello");
    assert_eq!(text_deltas[1], " ");
    assert_eq!(text_deltas[2], "world");
    assert_eq!(text_deltas[3], "!");

    // Verify final concatenated text
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Hello world!");
}

/// Some providers/proxies skip `response.output_item.added` and start directly
/// with `response.output_text.delta`. The parser must auto-register the item
/// and still emit TextStart + TextDelta events.
#[tokio::test]
async fn test_stream_text_without_output_item_added() {
    let server = MockServer::start().await;

    // SSE stream that skips response.output_item.added entirely
    let sse_body = responses_sse(vec![
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hello "
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "world!"
            })
            .to_string(),
        ),
        (
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": { "type": "message", "id": "item_01" }
            })
            .to_string(),
        ),
        (
            "response.completed",
            &json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_01",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15
                    },
                    "output": []
                }
            })
            .to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let model = make_model(&server.uri());
    let context = make_context("You are a test assistant.", "Hi");
    let options = make_options("test-key");
    let provider = OpenAIResponsesProtocol::new();

    let stream = provider.stream(&model, &context, options);

    // Collect all events (clone so we can still call result())
    let events: Vec<_> = stream.clone().collect().await;

    // Should have auto-generated TextStart
    let has_text_start = events
        .iter()
        .any(|e| matches!(e, AssistantMessageEvent::TextStart { .. }));
    assert!(
        has_text_start,
        "Expected TextStart event from auto-registration"
    );

    // Should have both TextDelta events
    let text_deltas: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            AssistantMessageEvent::TextDelta { delta, .. } => Some(delta.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(text_deltas.len(), 2);
    assert_eq!(text_deltas[0], "Hello ");
    assert_eq!(text_deltas[1], "world!");

    // Verify final result
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Hello world!");
}

/// Some proxies strip the SSE `event:` line and only forward `data:` lines.
/// The parser must extract the event type from the JSON `type` field.
#[tokio::test]
async fn test_stream_without_sse_event_lines() {
    let server = MockServer::start().await;

    // Build raw SSE body WITHOUT any "event:" lines — only "data:" lines.
    // Each data JSON has a "type" field that the parser should use.
    let sse_body = [
        format!(
            "data: {}\n\n",
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": { "type": "message", "id": "item_01", "role": "assistant", "content": [] }
            })
        ),
        format!(
            "data: {}\n\n",
            json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hello from "
            })
        ),
        format!(
            "data: {}\n\n",
            json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "data-only SSE!"
            })
        ),
        format!(
            "data: {}\n\n",
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": { "type": "message", "id": "item_01" }
            })
        ),
        format!(
            "data: {}\n\n",
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_01",
                    "status": "completed",
                    "usage": { "input_tokens": 12, "output_tokens": 8, "total_tokens": 20 },
                    "output": []
                }
            })
        ),
    ]
    .join("");

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let model = make_model(&server.uri());
    let context = make_context("You are a test assistant.", "Hi");
    let options = make_options("test-key");
    let provider = OpenAIResponsesProtocol::new();

    let stream = provider.stream(&model, &context, options);

    let events: Vec<_> = stream.clone().collect().await;

    // Should have TextStart, TextDelta events even without SSE event: lines
    let has_text_start = events
        .iter()
        .any(|e| matches!(e, AssistantMessageEvent::TextStart { .. }));
    assert!(has_text_start, "Expected TextStart from data-only SSE");

    let text_deltas: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            AssistantMessageEvent::TextDelta { delta, .. } => Some(delta.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(text_deltas.len(), 2);
    assert_eq!(text_deltas[0], "Hello from ");
    assert_eq!(text_deltas[1], "data-only SSE!");

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "Hello from data-only SSE!");
    assert_eq!(result.usage.input, 12);
    assert_eq!(result.usage.output, 8);
}

// ============================================================================
// Additional coverage: with_api_key, default, stream_simple, failed status,
// text + tool_call combined, multiple tool calls
// ============================================================================

#[test]
fn test_provider_with_api_key() {
    let provider = OpenAIResponsesProtocol::with_api_key("sk-test");
    assert_eq!(provider.provider_type(), Provider::OpenAIResponses);
}

#[test]
fn test_provider_default() {
    let provider = OpenAIResponsesProtocol::default();
    assert_eq!(provider.provider_type(), Provider::OpenAIResponses);
}

#[tokio::test]
async fn test_stream_simple_delegates_correctly() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        ("response.output_item.added", &json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"item_s","role":"assistant","content":[]}}).to_string()),
        ("response.output_text.delta", &json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"simple"}).to_string()),
        ("response.output_item.done", &json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"item_s"}}).to_string()),
        ("response.completed", &json!({"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":5,"output_tokens":1,"total_tokens":6},"output":[]}}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let stream = provider.stream_simple(
        &model,
        &context,
        SimpleStreamOptions {
            base: StreamOptions {
                api_key: Some("key".into()),
                ..Default::default()
            },
            reasoning: None,
            thinking_budget_tokens: None,
            thinking_display: None,
        },
    );
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "simple");
}

#[tokio::test]
async fn test_stream_failed_status() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        ("response.completed", &json!({"type":"response.completed","response":{"id":"r","status":"failed","usage":{"input_tokens":5,"output_tokens":0,"total_tokens":5},"output":[]}}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Error);
}

#[tokio::test]
async fn test_stream_text_then_tool_call() {
    let server = MockServer::start().await;

    // Message item with text, then function_call item
    let sse_body = responses_sse(vec![
        ("response.output_item.added", &json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"item_t","role":"assistant","content":[]}}).to_string()),
        ("response.output_text.delta", &json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Searching..."}).to_string()),
        ("response.output_item.done", &json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"item_t"}}).to_string()),
        ("response.output_item.added", &json!({"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"item_fc","call_id":"call_1","name":"search","arguments":""}}).to_string()),
        ("response.function_call_arguments.delta", &json!({"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"q\": \"test\"}"}).to_string()),
        ("response.output_item.done", &json!({"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"item_fc","call_id":"call_1","name":"search"}}).to_string()),
        ("response.completed", &json!({"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":10,"output_tokens":10,"total_tokens":20},"output":[{"type":"message","id":"item_t"},{"type":"function_call","id":"item_fc","call_id":"call_1","name":"search"}]}}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let mut context = make_context("test", "find info");
    context.set_tools(vec![Tool::new(
        "search",
        "Search",
        json!({"type":"object","properties":{"q":{"type":"string"}}}),
    )]);
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::ToolUse);
    assert_eq!(result.text_content(), "Searching...");
    assert_eq!(result.tool_calls().len(), 1);
    assert_eq!(result.tool_calls()[0].name, "search");
}

#[tokio::test]
async fn test_stream_multiple_function_calls() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        // First function call
        ("response.output_item.added", &json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc1","call_id":"c1","name":"fn_a","arguments":""}}).to_string()),
        ("response.function_call_arguments.delta", &json!({"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"x\":1}"}).to_string()),
        ("response.output_item.done", &json!({"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc1","call_id":"c1","name":"fn_a"}}).to_string()),
        // Second function call
        ("response.output_item.added", &json!({"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc2","call_id":"c2","name":"fn_b","arguments":""}}).to_string()),
        ("response.function_call_arguments.delta", &json!({"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"y\":2}"}).to_string()),
        ("response.output_item.done", &json!({"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"fc2","call_id":"c2","name":"fn_b"}}).to_string()),
        // Completed
        ("response.completed", &json!({"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":10,"output_tokens":20,"total_tokens":30},"output":[]}}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "use tools");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::ToolUse);
    let tcs = result.tool_calls();
    assert_eq!(tcs.len(), 2);
    assert_eq!(tcs[0].name, "fn_a");
    assert_eq!(tcs[1].name, "fn_b");
}

// ============================================================================
// Message conversion coverage: multi-turn, images, tool results, error events
// ============================================================================

#[tokio::test]
async fn test_stream_multiturn_with_tool_calls_and_results() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        ("response.output_item.added", &json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"item_r","role":"assistant","content":[]}}).to_string()),
        ("response.output_text.delta", &json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"continued"}).to_string()),
        ("response.output_item.done", &json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"item_r"}}).to_string()),
        ("response.completed", &json!({"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":50,"output_tokens":5,"total_tokens":55},"output":[]}}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let mut ctx = Context::with_system_prompt("system");
    ctx.add_message(Message::User(UserMessage::text("search for info")));
    // Assistant with text + tool call (composite ID)
    let asst = AssistantMessage::builder()
        .api(Api::OpenAIResponses)
        .provider(Provider::OpenAIResponses)
        .model("gpt-4o")
        .content(vec![
            ContentBlock::Text(TextContent {
                text: "Searching".to_string(),
                text_signature: None,
            }),
            ContentBlock::ToolCall(ToolCall {
                id: "call_1|fc_item1".to_string(),
                name: "search".to_string(),
                arguments: json!({"q": "test"}),
                thought_signature: None,
            }),
        ])
        .stop_reason(StopReason::ToolUse)
        .build()
        .unwrap();
    ctx.add_message(Message::Assistant(asst));
    // Tool result with composite ID
    ctx.add_message(Message::ToolResult(ToolResultMessage::text(
        "call_1|fc_item1",
        "search",
        "found data",
        false,
    )));
    // Errored assistant (should be skipped)
    let asst_err = AssistantMessage::builder()
        .api(Api::OpenAIResponses)
        .provider(Provider::OpenAIResponses)
        .model("gpt-4o")
        .content(vec![ContentBlock::Text(TextContent {
            text: "err".to_string(),
            text_signature: None,
        })])
        .stop_reason(StopReason::Error)
        .build()
        .unwrap();
    ctx.add_message(Message::Assistant(asst_err));
    ctx.add_message(Message::User(UserMessage::text("go on")));
    ctx.set_tools(vec![Tool::new(
        "search",
        "Search",
        json!({"type":"object","properties":{"q":{"type":"string"}}}),
    )]);

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let options = make_options("key");
    let stream = provider.stream(&model, &ctx, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "continued");
}

#[tokio::test]
async fn test_stream_with_image_user_content() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        ("response.output_item.added", &json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"item_img","role":"assistant","content":[]}}).to_string()),
        ("response.output_text.delta", &json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"I see a picture"}).to_string()),
        ("response.output_item.done", &json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"item_img"}}).to_string()),
        ("response.completed", &json!({"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":30,"output_tokens":3,"total_tokens":33},"output":[]}}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let mut ctx = Context::with_system_prompt("test");
    ctx.add_message(Message::User(UserMessage {
        role: Role::User,
        content: UserContent::Blocks(vec![
            ContentBlock::Text(TextContent {
                text: "What is this?".to_string(),
                text_signature: None,
            }),
            ContentBlock::Image(ImageContent {
                mime_type: "image/jpeg".to_string(),
                data: "/9j/4AA=".to_string(),
            }),
        ]),
        timestamp: 0,
    }));

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let options = make_options("key");
    let stream = provider.stream(&model, &ctx, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "I see a picture");
}

#[tokio::test]
async fn test_stream_non_composite_tool_call_id() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        ("response.output_item.added", &json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"item_nc","role":"assistant","content":[]}}).to_string()),
        ("response.output_text.delta", &json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"ok"}).to_string()),
        ("response.output_item.done", &json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"item_nc"}}).to_string()),
        ("response.completed", &json!({"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":20,"output_tokens":1,"total_tokens":21},"output":[]}}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let mut ctx = Context::with_system_prompt("test");
    ctx.add_message(Message::User(UserMessage::text("use tool")));
    // Assistant with non-composite tool call ID (no "|")
    let asst = AssistantMessage::builder()
        .api(Api::OpenAIResponses)
        .provider(Provider::OpenAIResponses)
        .model("gpt-4o")
        .content(vec![ContentBlock::ToolCall(ToolCall {
            id: "simple_id".to_string(),
            name: "fn_a".to_string(),
            arguments: json!({"x": 1}),
            thought_signature: None,
        })])
        .stop_reason(StopReason::ToolUse)
        .build()
        .unwrap();
    ctx.add_message(Message::Assistant(asst));
    // Tool result with non-composite ID
    ctx.add_message(Message::ToolResult(ToolResultMessage::text(
        "simple_id",
        "fn_a",
        "result",
        false,
    )));
    ctx.add_message(Message::User(UserMessage::text("continue")));

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let options = make_options("key");
    let stream = provider.stream(&model, &ctx, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
}

#[tokio::test]
async fn test_stream_function_call_arguments_done_and_tool_choice_payload() {
    let server = MockServer::start().await;
    let captured = Arc::new(Mutex::new(None));

    let sse_body = responses_sse(vec![
        ("response.output_item.added", &json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":""}}).to_string()),
        ("response.function_call_arguments.delta", &json!({"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"q\":","item_id":"fc_1","call_id":"call_1","name":"lookup"}).to_string()),
        ("response.function_call_arguments.done", &json!({"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\"q\":\"tokyo\"}","item_id":"fc_1","call_id":"call_1","name":"lookup"}).to_string()),
        ("response.output_item.done", &json!({"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"tokyo\"}"}}).to_string()),
        ("response.completed", &json!({"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":20,"output_tokens":1,"total_tokens":21},"output":[]}}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let mut context = make_context("test", "use tool");
    context.set_tools(vec![Tool::new(
        "lookup",
        "Lookup",
        json!({"type": "object", "properties": {"q": {"type": "string"}}}),
    )]);
    let mut options = make_options("key");
    options.tool_choice = Some(ToolChoice::Mode(ToolChoiceMode::Required));
    options.on_payload = Some(Arc::new({
        let captured = captured.clone();
        move |payload, _model| {
            let captured = captured.clone();
            Box::pin(async move {
                *captured.lock() = Some(payload.clone());
                Some(payload)
            })
        }
    }));

    let result = provider.stream(&model, &context, options).result().await;
    assert_eq!(result.stop_reason, StopReason::ToolUse);
    assert_eq!(result.tool_calls()[0].arguments["q"], "tokyo");

    let payload = captured.lock().clone().expect("payload captured");
    assert_eq!(payload["tool_choice"], json!("required"));
}

#[tokio::test]
async fn test_stream_function_call_arguments_done_prefer_local_accumulated_args() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        ("response.output_item.added", &json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":""}}).to_string()),
        ("response.function_call_arguments.delta", &json!({"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"q\":","item_id":"fc_1","call_id":"call_1","name":"lookup"}).to_string()),
        ("response.function_call_arguments.done", &json!({"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\"q\":\"tokyo\"}","item_id":"fc_1","call_id":"call_1","name":"lookup"}).to_string()),
        ("response.output_item.done", &json!({"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"osaka\"}"}}).to_string()),
        ("response.completed", &json!({"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":20,"output_tokens":1,"total_tokens":21},"output":[]}}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "use tool");
    let options = make_options("key");

    let result = provider.stream(&model, &context, options).result().await;
    assert_eq!(result.stop_reason, StopReason::ToolUse);
    assert_eq!(result.tool_calls()[0].arguments["q"], "tokyo");
}

#[tokio::test]
async fn test_tool_result_images_become_function_call_output_parts() {
    let server = MockServer::start().await;
    let captured = Arc::new(Mutex::new(None));

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(responses_sse(vec![
                    ("response.output_item.added", &json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","content":[]}}).to_string()),
                    ("response.output_text.delta", &json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"ok"}).to_string()),
                    ("response.output_item.done", &json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1"}}).to_string()),
                    ("response.completed", &json!({"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12},"output":[]}}).to_string()),
                ]))
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let mut context = Context::new();
    context.add_message(Message::ToolResult(ToolResultMessage {
        role: Role::ToolResult,
        tool_call_id: "call_1|fc_1".to_string(),
        tool_name: "vision_tool".to_string(),
        content: vec![
            ContentBlock::Text(TextContent::new("caption")),
            ContentBlock::Image(ImageContent::new("abcd", "image/png")),
        ],
        details: None,
        is_error: false,
        timestamp: 0,
    }));
    let options = make_options_with_capture("key", captured.clone());

    let _ = provider.stream(&model, &context, options).result().await;
    let payload = captured.lock().clone().expect("payload captured");
    let output_item = payload["input"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["type"] == json!("function_call_output"))
        .cloned()
        .expect("function_call_output should exist");
    assert!(output_item["output"].is_array());
    assert_eq!(output_item["output"][0]["type"], json!("input_text"));
    assert_eq!(output_item["output"][1]["type"], json!("input_image"));
}

#[tokio::test]
async fn test_stream_cancelled_status() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        ("response.completed", &json!({"type":"response.completed","response":{"id":"r","status":"cancelled","usage":{"input_tokens":5,"output_tokens":0,"total_tokens":5},"output":[]}}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Error);
}

#[tokio::test]
async fn test_stream_http_error_response() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal server error"))
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Error);
}

#[tokio::test]
async fn test_stream_usage_with_cached_tokens() {
    let server = MockServer::start().await;

    let sse_body = responses_sse(vec![
        ("response.output_item.added", &json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"item_c","role":"assistant","content":[]}}).to_string()),
        ("response.output_text.delta", &json!({"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"cached"}).to_string()),
        ("response.output_item.done", &json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"item_c"}}).to_string()),
        ("response.completed", &json!({"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":100,"output_tokens":5,"total_tokens":105,"input_tokens_details":{"cached_tokens":80}},"output":[]}}).to_string()),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_model(&server.uri());
    let context = make_context("test", "hello");
    let options = make_options("key");

    let stream = provider.stream(&model, &context, options);
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.usage.input, 20);
    assert_eq!(result.usage.output, 5);
    assert_eq!(result.usage.cache_read, 80);
    assert_eq!(result.usage.total_tokens, 105);
}
