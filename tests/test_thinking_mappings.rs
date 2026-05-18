//! Focused tests for provider-specific thinking/reasoning payload mappings.

use serde_json::json;
use tiycore::protocol::anthropic::AnthropicProtocol;
use tiycore::protocol::google::GoogleProtocol;
use tiycore::protocol::openai_responses::OpenAIResponsesProtocol;
use tiycore::protocol::LLMProtocol;
use tiycore::thinking::ThinkingLevel;
use tiycore::types::*;
use wiremock::matchers::{body_partial_json, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn make_context(system_prompt: &str, user_msg: &str) -> Context {
    let mut ctx = Context::with_system_prompt(system_prompt);
    ctx.add_message(Message::User(UserMessage::text(user_msg)));
    ctx
}

fn make_openai_model(base_url: &str) -> Model {
    make_openai_model_with_id(base_url, "gpt-5.4-mini")
}

fn make_openai_model_with_id(base_url: &str, id: &str) -> Model {
    Model::builder()
        .id(id)
        .name(id)
        .api(Api::OpenAIResponses)
        .provider(Provider::OpenAI)
        .base_url(base_url)
        .reasoning(true)
        .input(vec![InputType::Text])
        .context_window(128000)
        .max_tokens(16384)
        .build()
        .unwrap()
}

fn make_anthropic_model(base_url: &str, id: &str) -> Model {
    Model::builder()
        .id(id)
        .name(id)
        .api(Api::AnthropicMessages)
        .provider(Provider::Anthropic)
        .base_url(base_url)
        .reasoning(true)
        .input(vec![InputType::Text])
        .context_window(200000)
        .max_tokens(8192)
        .build()
        .unwrap()
}

fn make_google_model(base_url: &str, id: &str, api: Api) -> Model {
    Model::builder()
        .id(id)
        .name(id)
        .api(api)
        .provider(Provider::Google)
        .base_url(base_url)
        .reasoning(true)
        .input(vec![InputType::Text])
        .context_window(1048576)
        .max_tokens(8192)
        .build()
        .unwrap()
}

fn responses_sse(events: Vec<(&str, &str)>) -> String {
    events
        .iter()
        .map(|(event_type, data)| format!("event: {}\ndata: {}\n\n", event_type, data))
        .collect::<String>()
}

fn anthropic_sse(events: Vec<(&str, &str)>) -> String {
    events
        .iter()
        .map(|(event_type, data)| format!("event: {}\ndata: {}\n\n", event_type, data))
        .collect::<String>()
}

fn google_sse(chunks: Vec<&str>) -> String {
    chunks
        .into_iter()
        .map(|chunk| format!("data: {}\n\n", chunk))
        .collect::<String>()
}

#[tokio::test]
async fn test_openai_responses_stream_simple_maps_reasoning() {
    let server = MockServer::start().await;
    let sse_body = responses_sse(vec![
        (
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": { "type": "message", "id": "item_1", "role": "assistant", "content": [] }
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "done"
            })
            .to_string(),
        ),
        (
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": { "type": "message", "id": "item_1" }
            })
            .to_string(),
        ),
        (
            "response.completed",
            &json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "usage": { "input_tokens": 5, "output_tokens": 1 },
                    "output": []
                }
            })
            .to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({
            "reasoning": {
                "effort": "high",
                "summary": "auto"
            },
            "include": ["reasoning.encrypted_content"]
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_openai_model(&server.uri());
    let context = make_context("You are helpful.", "hello");
    let stream = provider.stream_simple(
        &model,
        &context,
        SimpleStreamOptions {
            base: StreamOptions {
                api_key: Some("key".into()),
                ..Default::default()
            },
            reasoning: Some(ThinkingLevel::High),
            thinking_budget_tokens: None,
            thinking_display: None,
        },
    );

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "done");
}

#[tokio::test]
async fn test_openai_responses_stream_simple_keeps_xhigh_for_gpt5_5_and_later() {
    let server = MockServer::start().await;
    let sse_body = responses_sse(vec![
        (
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": { "type": "message", "id": "item_1", "role": "assistant", "content": [] }
            })
            .to_string(),
        ),
        (
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "done"
            })
            .to_string(),
        ),
        (
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": { "type": "message", "id": "item_1" }
            })
            .to_string(),
        ),
        (
            "response.completed",
            &json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "usage": { "input_tokens": 5, "output_tokens": 1 },
                    "output": []
                }
            })
            .to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({
            "reasoning": {
                "effort": "xhigh",
                "summary": "auto"
            }
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAIResponsesProtocol::new();
    let model = make_openai_model_with_id(&server.uri(), "openai/gpt-5.5-mini");
    let context = make_context("You are helpful.", "hello");
    let stream = provider.stream_simple(
        &model,
        &context,
        SimpleStreamOptions {
            base: StreamOptions {
                api_key: Some("key".into()),
                ..Default::default()
            },
            reasoning: Some(ThinkingLevel::XHigh),
            thinking_budget_tokens: None,
            thinking_display: None,
        },
    );

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "done");
}

#[tokio::test]
async fn test_anthropic_stream_simple_maps_budget_thinking() {
    let server = MockServer::start().await;
    let sse_body = anthropic_sse(vec![
        (
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-5-sonnet",
                    "usage": { "input_tokens": 5, "output_tokens": 0 }
                }
            })
            .to_string(),
        ),
        (
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "" }
            })
            .to_string(),
        ),
        (
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "text_delta", "text": "ok" }
            })
            .to_string(),
        ),
        (
            "content_block_stop",
            &json!({
                "type": "content_block_stop",
                "index": 0
            })
            .to_string(),
        ),
        (
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn" },
                "usage": { "output_tokens": 1 }
            })
            .to_string(),
        ),
        (
            "message_stop",
            &json!({ "type": "message_stop" }).to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "thinking": {
                "type": "enabled",
                "budget_tokens": 2048
            }
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = AnthropicProtocol::new();
    let model = make_anthropic_model(&server.uri(), "claude-3-5-sonnet");
    let context = make_context("You are helpful.", "hello");
    let stream = provider.stream_simple(
        &model,
        &context,
        SimpleStreamOptions {
            base: StreamOptions {
                api_key: Some("key".into()),
                ..Default::default()
            },
            reasoning: Some(ThinkingLevel::Medium),
            thinking_budget_tokens: Some(2048),
            thinking_display: None,
        },
    );

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "ok");
}

#[tokio::test]
async fn test_anthropic_stream_simple_maps_adaptive_thinking() {
    let server = MockServer::start().await;
    let sse_body = anthropic_sse(vec![
        (
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": "msg_2",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-4-6",
                    "usage": { "input_tokens": 5, "output_tokens": 0 }
                }
            })
            .to_string(),
        ),
        (
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "" }
            })
            .to_string(),
        ),
        (
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "text_delta", "text": "ok" }
            })
            .to_string(),
        ),
        (
            "content_block_stop",
            &json!({
                "type": "content_block_stop",
                "index": 0
            })
            .to_string(),
        ),
        (
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn" },
                "usage": { "output_tokens": 1 }
            })
            .to_string(),
        ),
        (
            "message_stop",
            &json!({ "type": "message_stop" }).to_string(),
        ),
    ]);

    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "thinking": { "type": "adaptive" },
            "output_config": { "effort": "max" }
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = AnthropicProtocol::new();
    let model = make_anthropic_model(&server.uri(), "claude-opus-4-6");
    let context = make_context("You are helpful.", "hello");
    let stream = provider.stream_simple(
        &model,
        &context,
        SimpleStreamOptions {
            base: StreamOptions {
                api_key: Some("key".into()),
                ..Default::default()
            },
            reasoning: Some(ThinkingLevel::XHigh),
            thinking_budget_tokens: None,
            thinking_display: None,
        },
    );

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "ok");
}

#[tokio::test]
async fn test_google_stream_simple_maps_budget_thinking() {
    let server = MockServer::start().await;
    let sse_body = google_sse(vec![&json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{ "text": "ok" }]
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 1,
            "totalTokenCount": 6
        }
    })
    .to_string()]);

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.5-flash:streamGenerateContent"))
        .and(query_param("alt", "sse"))
        .and(body_partial_json(json!({
            "generationConfig": {
                "thinkingConfig": {
                    "includeThoughts": true,
                    "thinkingBudget": 2048
                }
            }
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = GoogleProtocol::new();
    let model = make_google_model(&server.uri(), "gemini-2.5-flash", Api::GoogleGenerativeAi);
    let context = make_context("You are helpful.", "hello");
    let stream = provider.stream_simple(
        &model,
        &context,
        SimpleStreamOptions {
            base: StreamOptions {
                api_key: Some("key".into()),
                ..Default::default()
            },
            reasoning: Some(ThinkingLevel::Low),
            thinking_budget_tokens: Some(2048),
            thinking_display: None,
        },
    );

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "ok");
}

#[tokio::test]
async fn test_anthropic_stream_simple_maps_opus_4_7_xhigh() {
    let server = MockServer::start().await;
    let sse_body = anthropic_sse(vec![
        (
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": "msg_3",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-4-7",
                    "usage": { "input_tokens": 5, "output_tokens": 0 }
                }
            })
            .to_string(),
        ),
        (
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "" }
            })
            .to_string(),
        ),
        (
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "text_delta", "text": "ok" }
            })
            .to_string(),
        ),
        (
            "content_block_stop",
            &json!({
                "type": "content_block_stop",
                "index": 0
            })
            .to_string(),
        ),
        (
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn" },
                "usage": { "output_tokens": 1 }
            })
            .to_string(),
        ),
        (
            "message_stop",
            &json!({ "type": "message_stop" }).to_string(),
        ),
    ]);

    // Opus 4.7 with XHigh → adaptive + display: summarized + effort: xhigh
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "thinking": { "type": "adaptive", "display": "summarized" },
            "output_config": { "effort": "xhigh" }
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = AnthropicProtocol::new();
    let model = make_anthropic_model(&server.uri(), "claude-opus-4-7");
    let context = make_context("You are helpful.", "hello");
    let stream = provider.stream_simple(
        &model,
        &context,
        SimpleStreamOptions {
            base: StreamOptions {
                api_key: Some("key".into()),
                ..Default::default()
            },
            reasoning: Some(ThinkingLevel::XHigh),
            thinking_budget_tokens: None,
            thinking_display: None, // defaults to Summarized for Opus 4.7
        },
    );

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "ok");
}

#[tokio::test]
async fn test_anthropic_stream_simple_maps_opus_4_7_high_omitted() {
    let server = MockServer::start().await;
    let sse_body = anthropic_sse(vec![
        (
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": "msg_4",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-4-7",
                    "usage": { "input_tokens": 5, "output_tokens": 0 }
                }
            })
            .to_string(),
        ),
        (
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "" }
            })
            .to_string(),
        ),
        (
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "text_delta", "text": "ok" }
            })
            .to_string(),
        ),
        (
            "content_block_stop",
            &json!({
                "type": "content_block_stop",
                "index": 0
            })
            .to_string(),
        ),
        (
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn" },
                "usage": { "output_tokens": 1 }
            })
            .to_string(),
        ),
        (
            "message_stop",
            &json!({ "type": "message_stop" }).to_string(),
        ),
    ]);

    // Opus 4.7 with High + explicit Omitted → adaptive + display: omitted + effort: high
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(body_partial_json(json!({
            "thinking": { "type": "adaptive", "display": "omitted" },
            "output_config": { "effort": "high" }
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = AnthropicProtocol::new();
    let model = make_anthropic_model(&server.uri(), "claude-opus-4-7");
    let context = make_context("You are helpful.", "hello");
    let stream = provider.stream_simple(
        &model,
        &context,
        SimpleStreamOptions {
            base: StreamOptions {
                api_key: Some("key".into()),
                ..Default::default()
            },
            reasoning: Some(ThinkingLevel::High),
            thinking_budget_tokens: None,
            thinking_display: Some(tiycore::thinking::ThinkingDisplay::Omitted),
        },
    );

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "ok");
}

#[tokio::test]
async fn test_google_vertex_stream_simple_maps_level_thinking() {
    let server = MockServer::start().await;
    let sse_body = google_sse(vec![&json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{ "text": "ok" }]
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 1,
            "totalTokenCount": 6
        }
    })
    .to_string()]);

    Mock::given(method("POST"))
        .and(path(
            "/v1/publishers/google/models/gemini-3-flash-preview:streamGenerateContent",
        ))
        .and(query_param("alt", "sse"))
        .and(body_partial_json(json!({
            "generationConfig": {
                "thinkingConfig": {
                    "includeThoughts": true,
                    "thinkingLevel": "MEDIUM"
                }
            }
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = GoogleProtocol::new();
    let model = make_google_model(&server.uri(), "gemini-3-flash-preview", Api::GoogleVertex);
    let context = make_context("You are helpful.", "hello");
    let stream = provider.stream_simple(
        &model,
        &context,
        SimpleStreamOptions {
            base: StreamOptions {
                api_key: Some("key".into()),
                ..Default::default()
            },
            reasoning: Some(ThinkingLevel::Medium),
            thinking_budget_tokens: None,
            thinking_display: None,
        },
    );

    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "ok");
}
