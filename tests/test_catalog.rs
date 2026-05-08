//! Tests for model catalog fetching and enrichment.

use serde_json::json;
use tiycore::catalog::{
    enrich_manual_model, list_models, list_models_with_enrichment, CatalogModelMetadata,
    FetchModelsRequest, InMemoryCatalogMetadataStore, ModelCatalogError,
};
use tiycore::types::Provider;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn test_list_models_with_openai_enrichment() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [
                {
                    "id": "gpt-4.1",
                    "created": 1710000000,
                    "owned_by": "openai"
                }
            ]
        })))
        .mount(&server)
        .await;

    let store = InMemoryCatalogMetadataStore::new(vec![CatalogModelMetadata {
        canonical_model_key: "openai:gpt-4.1".to_string(),
        aliases: vec!["openai/gpt-4.1".to_string()],
        display_name: Some("GPT-4.1".to_string()),
        description: Some("General-purpose flagship".to_string()),
        context_window: Some(1_000_000),
        max_output_tokens: Some(32_768),
        max_input_tokens: Some(1_000_000),
        modalities: Some(vec!["text".to_string(), "image".to_string()]),
        capabilities: Some(vec!["tools".to_string(), "reasoning".to_string()]),
        reasoning_content_constrained: false,
        pricing: Some(json!({"input": "2.0", "output": "8.0"})),
        source: "openrouter".to_string(),
        raw: json!({}),
    }]);

    let result = list_models_with_enrichment(
        FetchModelsRequest {
            provider: Provider::OpenAI,
            api_key: Some("test-key".to_string()),
            base_url: Some(format!("{}/v1", server.uri())),
            headers: None,
        },
        &store,
    )
    .await
    .expect("openai list should succeed");

    assert_eq!(result.models.len(), 1);
    let model = &result.models[0];
    assert_eq!(model.raw_id, "gpt-4.1");
    assert_eq!(model.canonical_model_key.as_deref(), Some("openai:gpt-4.1"));
    assert_eq!(model.display_name.as_deref(), Some("GPT-4.1"));
    assert_eq!(model.context_window, Some(1_000_000));
    assert_eq!(model.max_output_tokens, Some(32_768));
    assert_eq!(model.match_confidence, Some(1.0));
    assert_eq!(model.metadata_sources, vec!["openrouter".to_string()]);
    assert_eq!(result.raw_response["data"][0]["id"], "gpt-4.1");
}

#[tokio::test]
async fn test_list_models_for_anthropic_uses_models_endpoint() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("x-api-key", "anth-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                {
                    "id": "claude-sonnet-4-20250514",
                    "display_name": "Claude Sonnet 4",
                    "context_window": 200000,
                    "max_output_tokens": 16000,
                    "created_at": "2025-05-14T00:00:00Z"
                },
                {
                    "id": "claude-opus-4-6",
                    "display_name": "Claude Opus 4.6",
                    "context_window": 200000,
                    "max_output_tokens": 32000
                }
            ]
        })))
        .mount(&server)
        .await;

    let result = list_models(FetchModelsRequest {
        provider: Provider::Anthropic,
        api_key: Some("anth-key".to_string()),
        base_url: Some(format!("{}/v1", server.uri())),
        headers: None,
    })
    .await
    .expect("anthropic list should succeed");

    assert_eq!(result.models.len(), 2);
    assert_eq!(result.models[0].raw_id, "claude-sonnet-4-20250514");
    assert_eq!(result.models[0].max_output_tokens, Some(16000));
    assert_eq!(result.models[1].raw_id, "claude-opus-4-6");
    assert_eq!(result.models[1].context_window, Some(200000));
    assert!(result.raw_response["pages"].is_null());
}

#[tokio::test]
async fn test_list_models_for_google_uses_google_api_key_header() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1beta/models"))
        .and(header("x-goog-api-key", "google-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                {
                    "id": "gemini-2.5-pro",
                    "displayName": "Gemini 2.5 Pro",
                    "inputTokenLimit": 1048576,
                    "outputTokenLimit": 65536,
                    "inputModalities": ["text", "image"],
                    "outputModalities": ["text"]
                }
            ]
        })))
        .mount(&server)
        .await;

    let result = list_models(FetchModelsRequest {
        provider: Provider::Google,
        api_key: Some("google-key".to_string()),
        base_url: Some(format!("{}/v1beta", server.uri())),
        headers: None,
    })
    .await
    .expect("google list should succeed");

    assert_eq!(result.models.len(), 1);
    assert_eq!(result.models[0].raw_id, "gemini-2.5-pro");
    assert_eq!(
        result.models[0].display_name.as_deref(),
        Some("Gemini 2.5 Pro")
    );
    assert_eq!(result.models[0].context_window, Some(1_048_576));
    assert_eq!(result.models[0].max_input_tokens, Some(1_048_576));
    assert_eq!(result.models[0].max_output_tokens, Some(65_536));
}

#[tokio::test]
async fn test_list_models_for_openai_responses_uses_bearer_header() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("authorization", "Bearer openai-responses-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                {
                    "id": "gpt-5-mini",
                    "display_name": "GPT-5 Mini"
                }
            ]
        })))
        .mount(&server)
        .await;

    let result = list_models(FetchModelsRequest {
        provider: Provider::OpenAIResponses,
        api_key: Some("openai-responses-key".to_string()),
        base_url: Some(format!("{}/v1", server.uri())),
        headers: None,
    })
    .await
    .expect("openai responses list should succeed");

    assert_eq!(result.models.len(), 1);
    assert_eq!(result.models[0].raw_id, "gpt-5-mini");
    assert_eq!(result.models[0].display_name.as_deref(), Some("GPT-5 Mini"));
}

#[tokio::test]
async fn test_list_models_for_opencode_go_uses_predefined_list() {
    // OpenCode Go uses PredefinedModelsAdapter, not HTTP calls
    let result = list_models(FetchModelsRequest {
        provider: Provider::OpenCodeGo,
        api_key: Some("opencode-go-key".to_string()),
        base_url: None,
        headers: None,
    })
    .await
    .expect("opencode go list should succeed");

    // Should return 7 predefined models
    assert_eq!(result.models.len(), 7);

    // Verify all expected model IDs are present
    let model_ids: Vec<&str> = result.models.iter().map(|m| m.raw_id.as_str()).collect();
    assert!(model_ids.contains(&"glm-5.1"));
    assert!(model_ids.contains(&"kimi-k2.6"));
    assert!(model_ids.contains(&"mimo-v2.5-pro"));
    assert!(model_ids.contains(&"mimo-v2.5"));
    assert!(model_ids.contains(&"minimax-m2.7"));
    assert!(model_ids.contains(&"deepseek-v4-pro"));
    assert!(model_ids.contains(&"deepseek-v4-flash"));

    // Verify no metadata is pre-filled (enriched by upstream catalog)
    for model in &result.models {
        assert!(model.display_name.is_none());
        assert!(model.context_window.is_none());
        assert!(model.max_output_tokens.is_none());
    }
}

#[tokio::test]
async fn test_list_models_for_minimax_uses_predefined_list() {
    let result = list_models(FetchModelsRequest {
        provider: Provider::MiniMax,
        api_key: None,
        base_url: None,
        headers: None,
    })
    .await
    .expect("minimax list should succeed");

    assert_eq!(result.models.len(), 2);

    let model_ids: Vec<&str> = result.models.iter().map(|m| m.raw_id.as_str()).collect();
    assert!(model_ids.contains(&"MiniMax-M2.7"));
    assert!(model_ids.contains(&"MiniMax-M2.7-highspeed"));

    for model in &result.models {
        assert_eq!(model.provider, Provider::MiniMax);
        assert!(model.display_name.is_none());
        assert!(model.context_window.is_none());
        assert!(model.max_output_tokens.is_none());
    }
}

#[tokio::test]
async fn test_list_models_for_minimax_cn_uses_predefined_list() {
    let result = list_models(FetchModelsRequest {
        provider: Provider::MiniMaxCN,
        api_key: None,
        base_url: None,
        headers: None,
    })
    .await
    .expect("minimax-cn list should succeed");

    assert_eq!(result.models.len(), 2);

    let model_ids: Vec<&str> = result.models.iter().map(|m| m.raw_id.as_str()).collect();
    assert!(model_ids.contains(&"MiniMax-M2.7"));
    assert!(model_ids.contains(&"MiniMax-M2.7-highspeed"));

    for model in &result.models {
        assert_eq!(model.provider, Provider::MiniMaxCN);
        assert!(model.display_name.is_none());
        assert!(model.context_window.is_none());
        assert!(model.max_output_tokens.is_none());
    }
}

#[tokio::test]
async fn test_list_models_rejects_unimplemented_provider() {
    // Providers without a known default base_url fall through to
    // ModelsEndpointAdapter but fail with MissingBaseUrl when no
    // base_url is provided in the request.
    let error = list_models(FetchModelsRequest::new(Provider::GoogleVertex))
        .await
        .expect_err("google vertex without base_url should fail");

    match error {
        ModelCatalogError::MissingBaseUrl { provider } => {
            assert_eq!(provider, Provider::GoogleVertex);
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[tokio::test]
async fn test_list_models_for_openrouter_extracts_reasoning_from_supported_parameters() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("authorization", "Bearer openrouter-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                {
                    "id": "google/gemini-2.5-flash-image",
                    "name": "Google: Gemini 2.5 Flash Image",
                    "supported_parameters": ["max_tokens", "seed", "stop"]
                },
                {
                    "id": "minimax/minimax-m2.7",
                    "name": "MiniMax: M2.7",
                    "supported_parameters": [
                        "max_tokens",
                        "include_reasoning",
                        "reasoning",
                        "tools",
                        "tool_choice"
                    ]
                }
            ]
        })))
        .mount(&server)
        .await;

    Mock::given(method("GET"))
        .and(path("/v1/embeddings/models"))
        .and(header("authorization", "Bearer openrouter-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                {
                    "id": "openai/text-embedding-3-small",
                    "name": "OpenAI: Text Embedding 3 Small",
                    "context_length": 8192,
                    "architecture": {
                        "input_modalities": ["text"],
                        "output_modalities": ["embeddings"]
                    }
                }
            ]
        })))
        .mount(&server)
        .await;

    let result = list_models(FetchModelsRequest {
        provider: Provider::OpenRouter,
        api_key: Some("openrouter-key".to_string()),
        base_url: Some(format!("{}/v1", server.uri())),
        headers: None,
    })
    .await
    .expect("openrouter list should succeed");

    assert_eq!(result.models.len(), 3);
    assert_eq!(result.models[0].raw_id, "google/gemini-2.5-flash-image");
    assert_eq!(result.models[0].capabilities, None);
    assert_eq!(result.models[1].raw_id, "minimax/minimax-m2.7");
    assert_eq!(
        result.models[1].capabilities,
        Some(vec!["reasoning".to_string(), "tools".to_string()])
    );
    assert_eq!(result.models[2].raw_id, "openai/text-embedding-3-small");
    assert_eq!(
        result.models[2].modalities,
        Some(vec!["text".to_string(), "embeddings".to_string()])
    );
}

#[tokio::test]
async fn test_list_models_for_zenmux_merges_vertex_models() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(header("authorization", "Bearer zenmux-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                {
                    "id": "openai/gpt-5.4-mini",
                    "display_name": "GPT-5.4 Mini",
                    "context_length": 400000,
                    "max_output_tokens": 128000,
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                    "capabilities": {"reasoning": true}
                }
            ]
        })))
        .mount(&server)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/vertex-ai/v1beta/models"))
        .and(header("authorization", "Bearer zenmux-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [
                {
                    "name": "google/gemini-2.5-flash-image",
                    "displayName": "Gemini 2.5 Flash Image",
                    "description": "Image generation model",
                    "inputTokenLimit": 32768,
                    "outputTokenLimit": 8192,
                    "thinking": false,
                    "inputModalities": ["text", "image"],
                    "outputModalities": ["image"]
                }
            ]
        })))
        .mount(&server)
        .await;

    let result = list_models(FetchModelsRequest {
        provider: Provider::Zenmux,
        api_key: Some("zenmux-key".to_string()),
        base_url: Some(format!("{}/api/v1", server.uri())),
        headers: None,
    })
    .await
    .expect("zenmux list should succeed");

    assert_eq!(result.models.len(), 2);
    assert_eq!(result.models[0].raw_id, "openai/gpt-5.4-mini");
    assert_eq!(
        result.models[0].capabilities,
        Some(vec!["reasoning".to_string()])
    );
    assert_eq!(result.models[1].raw_id, "google/gemini-2.5-flash-image");
    assert_eq!(
        result.models[1].display_name.as_deref(),
        Some("Gemini 2.5 Flash Image")
    );
    assert_eq!(result.models[1].context_window, Some(32768));
    assert_eq!(result.models[1].max_input_tokens, Some(32768));
    assert_eq!(result.models[1].max_output_tokens, Some(8192));
    assert_eq!(
        result.models[1].modalities,
        Some(vec!["text".to_string(), "image".to_string()])
    );
    assert_eq!(result.models[1].capabilities, None);
}

#[test]
fn test_enrich_manual_model_uses_snapshot_metadata() {
    let store = InMemoryCatalogMetadataStore::new(vec![CatalogModelMetadata {
        canonical_model_key: "openai:gpt-4.1".to_string(),
        aliases: vec!["openai/gpt-4.1".to_string()],
        display_name: Some("GPT-4.1".to_string()),
        description: Some("General-purpose flagship".to_string()),
        context_window: Some(1_000_000),
        max_output_tokens: Some(32_768),
        max_input_tokens: Some(1_000_000),
        modalities: Some(vec!["text".to_string(), "image".to_string()]),
        capabilities: Some(vec!["tools".to_string(), "reasoning".to_string()]),
        reasoning_content_constrained: false,
        pricing: Some(json!({"input": "2.0", "output": "8.0"})),
        source: "openrouter".to_string(),
        raw: json!({}),
    }]);

    let model = enrich_manual_model(Provider::OpenAI, "openai/gpt-4.1", None, &store);

    assert_eq!(model.raw_id, "openai/gpt-4.1");
    assert_eq!(model.canonical_model_key.as_deref(), Some("openai:gpt-4.1"));
    assert_eq!(model.display_name.as_deref(), Some("GPT-4.1"));
    assert_eq!(model.context_window, Some(1_000_000));
    assert_eq!(model.max_output_tokens, Some(32_768));
    assert_eq!(model.match_confidence, Some(1.0));
    assert_eq!(model.metadata_sources, vec!["openrouter".to_string()]);
}

#[test]
fn test_enrich_manual_model_preserves_manual_display_name_without_snapshot_match() {
    let store = InMemoryCatalogMetadataStore::new(vec![]);

    let model = enrich_manual_model(
        Provider::OpenAI,
        "custom-model-id",
        Some("My Custom Model".to_string()),
        &store,
    );

    assert_eq!(model.raw_id, "custom-model-id");
    assert_eq!(model.display_name.as_deref(), Some("My Custom Model"));
    assert!(model.canonical_model_key.is_none());
    assert!(model.context_window.is_none());
    assert!(model.metadata_sources.is_empty());
}
