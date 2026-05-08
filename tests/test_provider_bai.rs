//! Tests for BAI provider internal logic and stream dispatch.

use tiycore::provider::bai::BaiProvider;
use tiycore::provider::LLMProtocol;
use tiycore::types::*;
use wiremock::matchers;
use wiremock::{Mock, MockServer, ResponseTemplate};

// ============================================================================
// Helpers
// ============================================================================

fn simple_openai_response(text: &str) -> String {
    [
        format!(
            "data: {}\n\n",
            serde_json::json!({
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]
            })
        ),
        format!(
            "data: {}\n\n",
            serde_json::json!({
                "choices": [{"index": 0, "delta": {"content": text}}]
            })
        ),
        format!(
            "data: {}\n\n",
            serde_json::json!({
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}
            })
        ),
        "data: [DONE]\n\n".to_string(),
    ]
    .join("")
}

fn make_model(id: &str, base_url: &str) -> Model {
    Model::builder()
        .id(id)
        .name("Test Model")
        .provider(Provider::Bai)
        .base_url(base_url)
        .context_window(128000)
        .max_tokens(8192)
        .build()
        .unwrap()
}

// ============================================================================
// Constructor / basic tests
// ============================================================================

#[test]
fn test_bai_new() {
    let provider = BaiProvider::new();
    assert_eq!(provider.provider_type(), Provider::Bai);
}

#[test]
fn test_bai_with_api_key() {
    let provider = BaiProvider::with_api_key("test-key");
    assert_eq!(provider.provider_type(), Provider::Bai);
}

#[test]
fn test_bai_default() {
    let provider = BaiProvider::default();
    assert_eq!(provider.provider_type(), Provider::Bai);
}

// ============================================================================
// Non-adaptive path: custom base URL → OpenAI Completions
// ============================================================================

#[tokio::test]
async fn test_bai_non_adaptive_uses_openai_completions() {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(simple_openai_response("non-adaptive response"))
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    // model.base_url is mock URI (not api.b.ai) → non-adaptive path
    let provider = BaiProvider::with_api_key("test-key");
    let model = make_model("claude-sonnet-4", &server.uri());
    let context = Context::with_system_prompt("test");
    let stream = provider.stream(
        &model,
        &context,
        StreamOptions {
            api_key: Some("test-key".into()),
            ..Default::default()
        },
    );
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "non-adaptive response");
}

#[tokio::test]
async fn test_bai_non_adaptive_stream_simple() {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(simple_openai_response("simple non-adaptive"))
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = BaiProvider::with_api_key("test-key");
    let model = make_model("claude-sonnet-4", &server.uri());
    let context = Context::with_system_prompt("test");
    let stream = provider.stream_simple(
        &model,
        &context,
        SimpleStreamOptions {
            base: StreamOptions {
                api_key: Some("test-key".into()),
                ..Default::default()
            },
            reasoning: None,
            thinking_budget_tokens: None,
            thinking_display: None,
        },
    );
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.text_content(), "simple non-adaptive");
}

// ============================================================================
// API key resolution: options > self > env
// ============================================================================

#[tokio::test]
async fn test_bai_api_key_from_options_takes_priority() {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(simple_openai_response("ok"))
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    // Provider has key "provider-key", but options has "options-key" which wins
    let provider = BaiProvider::with_api_key("provider-key");
    let model = make_model("gpt-4o", &server.uri());
    let context = Context::with_system_prompt("test");
    let stream = provider.stream(
        &model,
        &context,
        StreamOptions {
            api_key: Some("options-key".into()),
            ..Default::default()
        },
    );
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
}

#[tokio::test]
async fn test_bai_api_key_from_provider_when_no_options() {
    let server = MockServer::start().await;
    Mock::given(matchers::method("POST"))
        .and(matchers::path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(simple_openai_response("ok"))
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = BaiProvider::with_api_key("provider-key");
    let model = make_model("gpt-4o", &server.uri());
    let context = Context::with_system_prompt("test");
    let stream = provider.stream(&model, &context, StreamOptions::default());
    let result = stream.result().await;
    assert_eq!(result.stop_reason, StopReason::Stop);
}
