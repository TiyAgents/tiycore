//! Tests for types module: model, message, content, context, usage, events.

use serde_json::json;
use tiycore::types::*;

// ============================================================================
// Api enum tests
// ============================================================================

#[test]
fn test_api_as_str_known_variants() {
    assert_eq!(Api::OpenAICompletions.as_str(), "openai-completions");
    assert_eq!(Api::AnthropicMessages.as_str(), "anthropic-messages");
    assert_eq!(Api::GoogleGenerativeAi.as_str(), "google-generative-ai");
    assert_eq!(Api::Ollama.as_str(), "ollama");
    assert_eq!(Api::OpenAIResponses.as_str(), "openai-responses");
    assert_eq!(Api::MistralConversations.as_str(), "mistral-conversations");
    assert_eq!(Api::GoogleVertex.as_str(), "google-vertex");
}

#[test]
fn test_api_custom_variant() {
    let custom = Api::Custom("my-custom-api".to_string());
    assert_eq!(custom.as_str(), "my-custom-api");
}

#[test]
fn test_api_from_string_known() {
    assert_eq!(
        Api::from("openai-completions".to_string()),
        Api::OpenAICompletions
    );
    assert_eq!(
        Api::from("anthropic-messages".to_string()),
        Api::AnthropicMessages
    );
    assert_eq!(
        Api::from("google-generative-ai".to_string()),
        Api::GoogleGenerativeAi
    );
    assert_eq!(Api::from("ollama".to_string()), Api::Ollama);
}

#[test]
fn test_api_from_string_unknown_becomes_custom() {
    let api = Api::from("unknown-api".to_string());
    assert_eq!(api, Api::Custom("unknown-api".to_string()));
}

#[test]
fn test_api_is_openai_compatible() {
    assert!(Api::OpenAICompletions.is_openai_compatible());
    assert!(Api::Ollama.is_openai_compatible());
    assert!(Api::MistralConversations.is_openai_compatible());
    assert!(Api::OpenAIResponses.is_openai_compatible());

    assert!(!Api::AnthropicMessages.is_openai_compatible());
    assert!(!Api::GoogleGenerativeAi.is_openai_compatible());
    assert!(!Api::GoogleVertex.is_openai_compatible());
}

#[test]
fn test_api_display() {
    assert_eq!(format!("{}", Api::OpenAICompletions), "openai-completions");
    assert_eq!(format!("{}", Api::Custom("x".into())), "x");
}

#[test]
fn test_api_serde_roundtrip() {
    // Now serde rename matches as_str() consistently
    let api = Api::OpenAICompletions;
    let json = serde_json::to_string(&api).unwrap();
    assert_eq!(json, "\"openai-completions\"");
    let back: Api = serde_json::from_str(&json).unwrap();
    assert_eq!(back, api);
}

#[test]
fn test_api_serde_all_known_variants() {
    let variants = vec![
        (Api::OpenAICompletions, "\"openai-completions\""),
        (Api::AnthropicMessages, "\"anthropic-messages\""),
        (Api::GoogleGenerativeAi, "\"google-generative-ai\""),
        (Api::Ollama, "\"ollama\""),
    ];

    for (variant, expected_json) in variants {
        let json = serde_json::to_string(&variant).unwrap();
        assert_eq!(json, expected_json, "Failed for {:?}", variant);
        let back: Api = serde_json::from_str(&json).unwrap();
        assert_eq!(back, variant);
    }
}

// ============================================================================
// Provider enum tests
// ============================================================================

#[test]
fn test_provider_as_str() {
    assert_eq!(Provider::OpenAI.as_str(), "openai");
    assert_eq!(Provider::OpenAICompatible.as_str(), "openai-compatible");
    assert_eq!(Provider::Anthropic.as_str(), "anthropic");
    assert_eq!(Provider::Google.as_str(), "google");
    assert_eq!(Provider::Groq.as_str(), "groq");
    assert_eq!(Provider::Ollama.as_str(), "ollama");
    assert_eq!(Provider::XAI.as_str(), "xai");
    assert_eq!(Provider::KimiCoding.as_str(), "kimi-coding");
}

#[test]
fn test_provider_from_string() {
    assert_eq!(Provider::from("openai".to_string()), Provider::OpenAI);
    assert_eq!(
        Provider::from("openai-compatible".to_string()),
        Provider::OpenAICompatible
    );
    assert_eq!(Provider::from("anthropic".to_string()), Provider::Anthropic);
    assert_eq!(Provider::from("google".to_string()), Provider::Google);
    assert_eq!(Provider::from("groq".to_string()), Provider::Groq);
    assert_eq!(Provider::from("ollama".to_string()), Provider::Ollama);
    assert_eq!(
        Provider::from("unknown".to_string()),
        Provider::Custom("unknown".to_string())
    );
}

#[test]
fn test_provider_serde_roundtrip() {
    let provider = Provider::Anthropic;
    let json = serde_json::to_string(&provider).unwrap();
    let back: Provider = serde_json::from_str(&json).unwrap();
    assert_eq!(back, provider);
}

// ============================================================================
// Cost tests
// ============================================================================

#[test]
fn test_cost_new() {
    let cost = Cost::new(1.0, 2.0, 0.5, 0.25);
    assert_eq!(cost.input, 1.0);
    assert_eq!(cost.output, 2.0);
    assert_eq!(cost.cache_read, 0.5);
    assert_eq!(cost.cache_write, 0.25);
}

#[test]
fn test_cost_free() {
    let cost = Cost::free();
    assert_eq!(cost.input, 0.0);
    assert_eq!(cost.output, 0.0);
    assert_eq!(cost.cache_read, 0.0);
    assert_eq!(cost.cache_write, 0.0);
}

#[test]
fn test_cost_default_is_free() {
    assert_eq!(Cost::default(), Cost::free());
}

// ============================================================================
// Model tests
// ============================================================================

#[test]
fn test_model_builder_success() {
    let model = Model::builder()
        .id("gpt-4o")
        .name("GPT-4o")
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .base_url("https://api.openai.com/v1")
        .reasoning(false)
        .input(vec![InputType::Text, InputType::Image])
        .cost(Cost::new(2.5, 10.0, 1.25, 0.0))
        .context_window(128000)
        .max_tokens(16384)
        .build();

    assert!(model.is_ok());
    let model = model.unwrap();
    assert_eq!(model.id, "gpt-4o");
    assert_eq!(model.name, "GPT-4o");
    assert_eq!(model.api, Some(Api::OpenAICompletions));
    assert_eq!(model.provider, Provider::OpenAI);
    assert!(!model.reasoning);
    assert!(model.supports_text());
    assert!(model.supports_image());
}

#[test]
fn test_model_builder_missing_required_fields() {
    // Missing id
    let result = Model::builder()
        .name("test")
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .base_url("http://test")
        .context_window(1000)
        .max_tokens(100)
        .build();
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "id is required");

    // api is now optional (Option<Api>), so building without it should succeed
    let result = Model::builder()
        .id("test")
        .name("test")
        .provider(Provider::OpenAI)
        .base_url("http://test")
        .context_window(1000)
        .max_tokens(100)
        .build();
    assert!(result.is_ok());
    assert_eq!(result.unwrap().api, None);
}

#[test]
fn test_model_calculate_cost() {
    let model = Model::builder()
        .id("test")
        .name("test")
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .base_url("http://test")
        .cost(Cost::new(10.0, 30.0, 5.0, 0.0))
        .context_window(128000)
        .max_tokens(16384)
        .build()
        .unwrap();

    let usage = Usage {
        input: 1_000_000,
        output: 500_000,
        cache_read: 200_000,
        cache_write: 0,
        total_tokens: 1_700_000,
        cost: UsageCost::default(),
    };

    let cost = model.calculate_cost(&usage);
    // input: 1M * 10 / 1M = 10.0
    // output: 500K * 30 / 1M = 15.0
    // cache_read: 200K * 5 / 1M = 1.0
    assert!((cost - 26.0).abs() < 0.001);
}

#[test]
fn test_model_supports_input_types() {
    let model = Model::builder()
        .id("t")
        .name("t")
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .base_url("http://t")
        .input(vec![InputType::Text])
        .context_window(1000)
        .max_tokens(100)
        .build()
        .unwrap();

    assert!(model.supports_text());
    assert!(!model.supports_image());
}

#[test]
fn test_model_serde_roundtrip() {
    let model = Model::builder()
        .id("gpt-4o-mini")
        .name("GPT-4o Mini")
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .base_url("https://api.openai.com/v1")
        .input(vec![InputType::Text])
        .cost(Cost::new(0.15, 0.60, 0.075, 0.0))
        .context_window(128000)
        .max_tokens(16384)
        .build()
        .unwrap();

    let json = serde_json::to_string(&model).unwrap();
    let back: Model = serde_json::from_str(&json).unwrap();
    assert_eq!(back.id, model.id);
    assert_eq!(back.api, model.api);
    assert_eq!(back.provider, model.provider);
}

// ============================================================================
// OpenAICompletionsCompat tests
// ============================================================================

#[test]
fn test_openai_completions_compat_default() {
    let compat = OpenAICompletionsCompat::default();
    assert!(compat.supports_store);
    assert!(!compat.supports_developer_role);
    assert!(compat.supports_reasoning_effort);
    assert!(compat.supports_usage_in_streaming);
    assert!(compat.supports_strict_mode);
    assert!(!compat.requires_tool_result_name);
    assert!(!compat.requires_assistant_after_tool_result);
    assert!(!compat.requires_thinking_as_text);
    assert_eq!(compat.thinking_format, "openai");
}

// ============================================================================
// Content block tests
// ============================================================================

#[test]
fn test_text_content_new() {
    let tc = TextContent::new("Hello");
    assert_eq!(tc.text, "Hello");
    assert!(tc.text_signature.is_none());
}

#[test]
fn test_thinking_content_new() {
    let tc = ThinkingContent::new("Let me think...");
    assert_eq!(tc.thinking, "Let me think...");
    assert!(!tc.redacted);
    assert!(tc.thinking_signature.is_none());
}

#[test]
fn test_image_content_new() {
    let ic = ImageContent::new("base64data", "image/png");
    assert_eq!(ic.data, "base64data");
    assert_eq!(ic.mime_type, "image/png");
}

#[test]
fn test_tool_call_new() {
    let tc = ToolCall::new("call_1", "get_weather", json!({"city": "Tokyo"}));
    assert_eq!(tc.id, "call_1");
    assert_eq!(tc.name, "get_weather");
    assert_eq!(tc.arguments["city"], "Tokyo");
    assert!(tc.thought_signature.is_none());
}

#[test]
fn test_content_block_type_checking() {
    let text = ContentBlock::Text(TextContent::new("hello"));
    assert!(text.is_text());
    assert!(!text.is_thinking());
    assert!(!text.is_tool_call());
    assert!(!text.is_image());
    assert!(text.as_text().is_some());
    assert!(text.as_thinking().is_none());

    let thinking = ContentBlock::Thinking(ThinkingContent::new("hmm"));
    assert!(thinking.is_thinking());
    assert!(!thinking.is_text());
    assert!(thinking.as_thinking().is_some());

    let tc = ContentBlock::ToolCall(ToolCall::new("id", "name", json!({})));
    assert!(tc.is_tool_call());
    assert!(tc.as_tool_call().is_some());

    let img = ContentBlock::Image(ImageContent::new("data", "image/png"));
    assert!(img.is_image());
    assert!(img.as_image().is_some());
}

#[test]
fn test_content_block_from_impls() {
    let block: ContentBlock = TextContent::new("hi").into();
    assert!(block.is_text());

    let block: ContentBlock = ThinkingContent::new("hmm").into();
    assert!(block.is_thinking());

    let block: ContentBlock = ToolCall::new("id", "name", json!({})).into();
    assert!(block.is_tool_call());

    let block: ContentBlock = ImageContent::new("data", "image/png").into();
    assert!(block.is_image());
}

#[test]
fn test_user_content_text() {
    let uc = UserContent::text("Hello");
    assert!(uc.is_text());
    assert_eq!(uc.as_text(), Some("Hello"));
}

#[test]
fn test_user_content_from_string() {
    let uc: UserContent = "Hello".into();
    assert!(uc.is_text());
    let uc: UserContent = String::from("Hello").into();
    assert!(uc.is_text());
}

#[test]
fn test_user_content_blocks() {
    let uc = UserContent::Blocks(vec![
        ContentBlock::Text(TextContent::new("hello")),
        ContentBlock::Image(ImageContent::new("data", "image/png")),
    ]);
    assert!(!uc.is_text());
    assert!(uc.as_text().is_none());
}

// ============================================================================
// Message tests
// ============================================================================

#[test]
fn test_user_message_text() {
    let msg = UserMessage::text("Hello world");
    assert_eq!(msg.role, Role::User);
    assert!(matches!(msg.content, UserContent::Text(ref s) if s == "Hello world"));
    assert!(msg.timestamp > 0);
}

#[test]
fn test_user_message_blocks() {
    let msg = UserMessage::blocks(vec![ContentBlock::Text(TextContent::new("hello"))]);
    assert_eq!(msg.role, Role::User);
    assert!(matches!(msg.content, UserContent::Blocks(_)));
}

#[test]
fn test_assistant_message_builder() {
    let msg = AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("gpt-4o-mini")
        .content(vec![ContentBlock::Text(TextContent::new("Hello!"))])
        .stop_reason(StopReason::Stop)
        .build()
        .unwrap();

    assert_eq!(msg.role, Role::Assistant);
    assert_eq!(msg.model, "gpt-4o-mini");
    assert_eq!(msg.text_content(), "Hello!");
    assert_eq!(msg.stop_reason, StopReason::Stop);
    assert!(!msg.has_tool_calls());
    assert!(msg.tool_calls().is_empty());
}

#[test]
fn test_assistant_message_with_tool_calls() {
    let msg = AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("gpt-4o")
        .content(vec![
            ContentBlock::Text(TextContent::new("Let me check...")),
            ContentBlock::ToolCall(ToolCall::new(
                "call_1",
                "get_weather",
                json!({"city": "Tokyo"}),
            )),
            ContentBlock::ToolCall(ToolCall::new("call_2", "get_time", json!({"tz": "JST"}))),
        ])
        .stop_reason(StopReason::ToolUse)
        .build()
        .unwrap();

    assert!(msg.has_tool_calls());
    assert_eq!(msg.tool_calls().len(), 2);
    assert_eq!(msg.tool_calls()[0].name, "get_weather");
    assert_eq!(msg.tool_calls()[1].name, "get_time");
    assert_eq!(msg.text_content(), "Let me check...");
}

#[test]
fn test_assistant_message_thinking_content() {
    let msg = AssistantMessage::builder()
        .api(Api::AnthropicMessages)
        .provider(Provider::Anthropic)
        .model("claude-sonnet-4")
        .content(vec![
            ContentBlock::Thinking(ThinkingContent::new("First, let me consider...")),
            ContentBlock::Thinking(ThinkingContent::new("Also, I should check...")),
            ContentBlock::Text(TextContent::new("Here's my answer.")),
        ])
        .build()
        .unwrap();

    assert_eq!(
        msg.thinking_content(),
        "First, let me consider...\nAlso, I should check..."
    );
    assert_eq!(msg.text_content(), "Here's my answer.");
}

#[test]
fn test_assistant_message_builder_missing_fields() {
    let result = AssistantMessage::builder().model("test").build();
    assert!(result.is_err()); // missing api and provider
}

#[test]
fn test_tool_result_message_text() {
    let msg = ToolResultMessage::text("call_1", "get_weather", "Sunny, 25C", false);
    assert_eq!(msg.role, Role::ToolResult);
    assert_eq!(msg.tool_call_id, "call_1");
    assert_eq!(msg.tool_name, "get_weather");
    assert!(!msg.is_error);
    assert_eq!(msg.content.len(), 1);
}

#[test]
fn test_tool_result_message_error() {
    let msg = ToolResultMessage::error("call_1", "unknown_tool", "Tool not found");
    assert!(msg.is_error);
    assert_eq!(msg.tool_name, "unknown_tool");
}

#[test]
fn test_tool_result_with_details() {
    let msg = ToolResultMessage::text("call_1", "tool", "result", false);
    let with_details = msg.with_details(json!({"extra": "data"}));
    assert_eq!(with_details.details, Some(json!({"extra": "data"})));
    assert_eq!(with_details.tool_call_id, "call_1");
}

#[test]
fn test_message_enum_variants() {
    let user = Message::User(UserMessage::text("hi"));
    assert!(user.is_user());
    assert!(!user.is_assistant());
    assert!(!user.is_tool_result());
    assert_eq!(user.role(), Role::User);

    let assistant = Message::Assistant(
        AssistantMessage::builder()
            .api(Api::OpenAICompletions)
            .provider(Provider::OpenAI)
            .model("gpt-4o")
            .build()
            .unwrap(),
    );
    assert!(assistant.is_assistant());
    assert_eq!(assistant.role(), Role::Assistant);

    let tool = Message::ToolResult(ToolResultMessage::text("id", "name", "result", false));
    assert!(tool.is_tool_result());
    assert_eq!(tool.role(), Role::ToolResult);
}

#[test]
fn test_message_from_impls() {
    let msg: Message = UserMessage::text("hi").into();
    assert!(msg.is_user());

    let msg: Message = AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("m")
        .build()
        .unwrap()
        .into();
    assert!(msg.is_assistant());

    let msg: Message = ToolResultMessage::text("id", "name", "r", false).into();
    assert!(msg.is_tool_result());
}

#[test]
fn test_message_timestamp() {
    let msg = Message::User(UserMessage::text("hello"));
    assert!(msg.timestamp() > 0);
}

// ============================================================================
// StopReason tests
// ============================================================================

#[test]
fn test_stop_reason_display() {
    assert_eq!(format!("{}", StopReason::Stop), "stop");
    assert_eq!(format!("{}", StopReason::Length), "length");
    assert_eq!(format!("{}", StopReason::ToolUse), "toolUse");
    assert_eq!(format!("{}", StopReason::Error), "error");
    assert_eq!(format!("{}", StopReason::Aborted), "aborted");
}

#[test]
fn test_stop_reason_default() {
    assert_eq!(StopReason::default(), StopReason::Stop);
}

#[test]
fn test_role_display() {
    assert_eq!(format!("{}", Role::User), "user");
    assert_eq!(format!("{}", Role::Assistant), "assistant");
    assert_eq!(format!("{}", Role::ToolResult), "toolResult");
}

// ============================================================================
// Context tests
// ============================================================================

#[test]
fn test_context_new() {
    let ctx = Context::new();
    assert!(ctx.system_prompt.is_none());
    assert!(ctx.messages.is_empty());
    assert!(ctx.tools.is_none());
    assert!(ctx.is_empty());
    assert_eq!(ctx.len(), 0);
}

#[test]
fn test_context_with_system_prompt() {
    let ctx = Context::with_system_prompt("You are helpful.");
    assert_eq!(ctx.system_prompt, Some("You are helpful.".to_string()));
    assert!(ctx.is_empty());
}

#[test]
fn test_context_add_message() {
    let mut ctx = Context::new();
    ctx.add_message(Message::User(UserMessage::text("Hello")));
    assert_eq!(ctx.len(), 1);
    assert!(!ctx.is_empty());
    assert!(ctx.last_message().unwrap().is_user());
}

#[test]
fn test_context_user_shorthand() {
    let mut ctx = Context::new();
    ctx.user("Hello");
    assert_eq!(ctx.len(), 1);
    assert!(ctx.messages[0].is_user());
}

#[test]
fn test_context_clear() {
    let mut ctx = Context::new();
    ctx.user("Hello");
    ctx.user("World");
    assert_eq!(ctx.len(), 2);
    ctx.clear();
    assert!(ctx.is_empty());
}

#[test]
fn test_context_set_tools() {
    let mut ctx = Context::new();
    ctx.set_tools(vec![Tool::new("tool1", "desc1", json!({"type": "object"}))]);
    assert!(ctx.tools.is_some());
    assert_eq!(ctx.tools.as_ref().unwrap().len(), 1);
}

// ============================================================================
// Tool tests
// ============================================================================

#[test]
fn test_tool_new() {
    let tool = Tool::new("get_weather", "Get weather info", json!({"type": "object"}));
    assert_eq!(tool.name, "get_weather");
    assert_eq!(tool.description, "Get weather info");
}

#[test]
fn test_tool_builder() {
    let tool = Tool::builder()
        .name("calculator")
        .description("Perform calculations")
        .parameters(json!({
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"]
        }))
        .build()
        .unwrap();

    assert_eq!(tool.name, "calculator");
    assert_eq!(tool.description, "Perform calculations");
}

#[test]
fn test_tool_builder_missing_name() {
    let result = Tool::builder().description("test").build();
    assert!(result.is_err());
}

#[test]
fn test_tool_builder_defaults() {
    let tool = Tool::builder().name("test").build().unwrap();
    assert_eq!(tool.description, "");
    assert_eq!(tool.parameters, json!({"type": "object", "properties": {}}));
}

// ============================================================================
// StreamOptions tests
// ============================================================================

#[test]
fn test_stream_options_default() {
    let opts = StreamOptions::default();
    assert!(opts.temperature.is_none());
    assert!(opts.max_tokens.is_none());
    assert!(opts.api_key.is_none());
    assert!(opts.headers.is_none());
    assert!(opts.session_id.is_none());
}

#[test]
fn test_simple_stream_options_from_base() {
    let base = StreamOptions {
        temperature: Some(0.7),
        max_tokens: Some(1000),
        ..Default::default()
    };
    let simple = SimpleStreamOptions::from(base.clone());
    assert_eq!(simple.base.temperature, Some(0.7));
    assert_eq!(simple.base.max_tokens, Some(1000));
    assert!(simple.reasoning.is_none());
}

// ============================================================================
// Usage tests
// ============================================================================

#[test]
fn test_usage_default() {
    let usage = Usage::default();
    assert_eq!(usage.input, 0);
    assert_eq!(usage.output, 0);
    assert_eq!(usage.cache_read, 0);
    assert_eq!(usage.cache_write, 0);
    assert_eq!(usage.total_tokens, 0);
    assert_eq!(usage.total_cost(), 0.0);
}

#[test]
fn test_usage_from_tokens() {
    let usage = Usage::from_tokens(100, 200);
    assert_eq!(usage.input, 100);
    assert_eq!(usage.output, 200);
    assert_eq!(usage.total_tokens, 300);
    assert_eq!(usage.cache_read, 0);
}

#[test]
fn test_usage_add() {
    let mut u1 = Usage::from_tokens(100, 200); // total_tokens = 300
    let u2 = Usage::from_tokens(50, 100); // total_tokens = 150
    u1.add(&u2);
    assert_eq!(u1.input, 150);
    assert_eq!(u1.output, 300);
    // total_tokens is now recomputed as input + output + cache_read + cache_write
    assert_eq!(u1.total_tokens, 450);
}

#[test]
fn test_usage_cost_from_costs() {
    let cost = UsageCost::from_costs(1.0, 2.0);
    assert_eq!(cost.input, 1.0);
    assert_eq!(cost.output, 2.0);
    assert_eq!(cost.total, 3.0);
}

#[test]
fn test_usage_cost_total_method_vs_field() {
    let cost = UsageCost {
        input: 1.0,
        output: 2.0,
        cache_read: 0.5,
        cache_write: 0.25,
        total: 0.0, // field not updated
    };
    // Method computes dynamically
    assert_eq!(cost.total(), 3.75);
    // Field is stale
    assert_eq!(cost.total, 0.0);
}

// ============================================================================
// Events tests
// ============================================================================

#[test]
fn test_event_is_complete() {
    let msg = AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("gpt-4o")
        .build()
        .unwrap();

    let done = AssistantMessageEvent::Done {
        reason: StopReason::Stop,
        message: msg.clone(),
    };
    assert!(done.is_complete());

    let error = AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: msg.clone(),
    };
    assert!(error.is_complete());

    let start = AssistantMessageEvent::Start { partial: msg };
    assert!(!start.is_complete());
}

#[test]
fn test_event_type_checks() {
    let msg = AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("m")
        .build()
        .unwrap();

    let text_delta = AssistantMessageEvent::TextDelta {
        content_index: 0,
        delta: "hello".to_string(),
        partial: msg.clone(),
    };
    assert!(text_delta.is_text_event());
    assert!(!text_delta.is_thinking_event());
    assert!(!text_delta.is_tool_call_event());

    let thinking = AssistantMessageEvent::ThinkingDelta {
        content_index: 0,
        delta: "hmm".to_string(),
        partial: msg.clone(),
    };
    assert!(thinking.is_thinking_event());

    let tc = AssistantMessageEvent::ToolCallStart {
        content_index: 0,
        partial: msg,
    };
    assert!(tc.is_tool_call_event());
}

#[test]
fn test_event_partial_message() {
    let msg = AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("m")
        .build()
        .unwrap();

    let start = AssistantMessageEvent::Start {
        partial: msg.clone(),
    };
    assert!(start.partial_message().is_some());

    let done = AssistantMessageEvent::Done {
        reason: StopReason::Stop,
        message: msg,
    };
    assert!(done.partial_message().is_some());
}

#[test]
fn test_event_content_index() {
    let msg = AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("m")
        .build()
        .unwrap();

    let event = AssistantMessageEvent::TextDelta {
        content_index: 5,
        delta: "x".to_string(),
        partial: msg.clone(),
    };
    assert_eq!(event.content_index(), Some(5));

    let start = AssistantMessageEvent::Start { partial: msg };
    assert_eq!(start.content_index(), None);
}

#[test]
fn test_event_delta() {
    let msg = AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("m")
        .build()
        .unwrap();

    let event = AssistantMessageEvent::TextDelta {
        content_index: 0,
        delta: "hello".to_string(),
        partial: msg.clone(),
    };
    assert_eq!(event.delta(), Some("hello"));

    let start = AssistantMessageEvent::Start { partial: msg };
    assert_eq!(start.delta(), None);
}

#[test]
fn test_event_stop_reason() {
    let msg = AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("m")
        .build()
        .unwrap();

    let done = AssistantMessageEvent::Done {
        reason: StopReason::ToolUse,
        message: msg.clone(),
    };
    assert_eq!(done.stop_reason(), Some(StopReason::ToolUse));

    let start = AssistantMessageEvent::Start {
        partial: msg.clone(),
    };
    assert_eq!(start.stop_reason(), None);

    let error = AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: msg,
    };
    assert_eq!(error.stop_reason(), Some(StopReason::Error));
}

// ============================================================================
// Event type-check methods
// ============================================================================

fn make_partial() -> AssistantMessage {
    AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("m")
        .build()
        .unwrap()
}

#[test]
fn test_event_is_text_event() {
    let msg = make_partial();
    let text_start = AssistantMessageEvent::TextStart {
        content_index: 0,
        partial: msg.clone(),
    };
    let text_delta = AssistantMessageEvent::TextDelta {
        content_index: 0,
        delta: "hi".into(),
        partial: msg.clone(),
    };
    let text_end = AssistantMessageEvent::TextEnd {
        content_index: 0,
        content: "hi".into(),
        partial: msg.clone(),
    };
    let thinking_start = AssistantMessageEvent::ThinkingStart {
        content_index: 0,
        partial: msg.clone(),
    };
    let start = AssistantMessageEvent::Start {
        partial: msg.clone(),
    };

    assert!(text_start.is_text_event());
    assert!(text_delta.is_text_event());
    assert!(text_end.is_text_event());
    assert!(!thinking_start.is_text_event());
    assert!(!start.is_text_event());
}

#[test]
fn test_event_is_thinking_event() {
    let msg = make_partial();
    let thinking_start = AssistantMessageEvent::ThinkingStart {
        content_index: 0,
        partial: msg.clone(),
    };
    let thinking_delta = AssistantMessageEvent::ThinkingDelta {
        content_index: 0,
        delta: "hmm".into(),
        partial: msg.clone(),
    };
    let thinking_end = AssistantMessageEvent::ThinkingEnd {
        content_index: 0,
        content: "hmm".into(),
        partial: msg.clone(),
    };
    let text_start = AssistantMessageEvent::TextStart {
        content_index: 0,
        partial: msg.clone(),
    };

    assert!(thinking_start.is_thinking_event());
    assert!(thinking_delta.is_thinking_event());
    assert!(thinking_end.is_thinking_event());
    assert!(!text_start.is_thinking_event());
}

#[test]
fn test_event_is_tool_call_event() {
    let msg = make_partial();
    let tc = ToolCall::new("id1", "func", json!({}));
    let tc_start = AssistantMessageEvent::ToolCallStart {
        content_index: 0,
        partial: msg.clone(),
    };
    let tc_delta = AssistantMessageEvent::ToolCallDelta {
        content_index: 0,
        delta: "{".into(),
        partial: msg.clone(),
    };
    let tc_end = AssistantMessageEvent::ToolCallEnd {
        content_index: 0,
        tool_call: tc,
        partial: msg.clone(),
    };
    let text_start = AssistantMessageEvent::TextStart {
        content_index: 0,
        partial: msg.clone(),
    };

    assert!(tc_start.is_tool_call_event());
    assert!(tc_delta.is_tool_call_event());
    assert!(tc_end.is_tool_call_event());
    assert!(!text_start.is_tool_call_event());
}

#[test]
fn test_event_partial_message_all_variants() {
    let msg = make_partial();
    let tc = ToolCall::new("id1", "func", json!({}));

    // All variants must return Some(partial)
    let events: Vec<AssistantMessageEvent> = vec![
        AssistantMessageEvent::Start {
            partial: msg.clone(),
        },
        AssistantMessageEvent::TextStart {
            content_index: 0,
            partial: msg.clone(),
        },
        AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: "x".into(),
            partial: msg.clone(),
        },
        AssistantMessageEvent::TextEnd {
            content_index: 0,
            content: "x".into(),
            partial: msg.clone(),
        },
        AssistantMessageEvent::ThinkingStart {
            content_index: 0,
            partial: msg.clone(),
        },
        AssistantMessageEvent::ThinkingDelta {
            content_index: 0,
            delta: "t".into(),
            partial: msg.clone(),
        },
        AssistantMessageEvent::ThinkingEnd {
            content_index: 0,
            content: "t".into(),
            partial: msg.clone(),
        },
        AssistantMessageEvent::ToolCallStart {
            content_index: 0,
            partial: msg.clone(),
        },
        AssistantMessageEvent::ToolCallDelta {
            content_index: 0,
            delta: "{".into(),
            partial: msg.clone(),
        },
        AssistantMessageEvent::ToolCallEnd {
            content_index: 0,
            tool_call: tc,
            partial: msg.clone(),
        },
        AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: msg.clone(),
        },
        AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: msg.clone(),
        },
    ];
    for event in &events {
        assert!(
            event.partial_message().is_some(),
            "partial_message() should be Some for {:?}",
            event
        );
    }
}

#[test]
fn test_event_content_index_thinking_and_tool() {
    let msg = make_partial();
    let tc = ToolCall::new("id1", "func", json!({}));

    assert_eq!(
        AssistantMessageEvent::ThinkingStart {
            content_index: 3,
            partial: msg.clone()
        }
        .content_index(),
        Some(3)
    );
    assert_eq!(
        AssistantMessageEvent::ThinkingDelta {
            content_index: 4,
            delta: "".into(),
            partial: msg.clone()
        }
        .content_index(),
        Some(4)
    );
    assert_eq!(
        AssistantMessageEvent::ThinkingEnd {
            content_index: 5,
            content: "".into(),
            partial: msg.clone()
        }
        .content_index(),
        Some(5)
    );
    assert_eq!(
        AssistantMessageEvent::ToolCallStart {
            content_index: 6,
            partial: msg.clone()
        }
        .content_index(),
        Some(6)
    );
    assert_eq!(
        AssistantMessageEvent::ToolCallDelta {
            content_index: 7,
            delta: "".into(),
            partial: msg.clone()
        }
        .content_index(),
        Some(7)
    );
    assert_eq!(
        AssistantMessageEvent::ToolCallEnd {
            content_index: 8,
            tool_call: tc,
            partial: msg.clone()
        }
        .content_index(),
        Some(8)
    );

    // Done and Error have no content_index
    assert_eq!(
        AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: msg.clone()
        }
        .content_index(),
        None
    );
    assert_eq!(
        AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: msg
        }
        .content_index(),
        None
    );
}

#[test]
fn test_event_delta_thinking_and_tool_call() {
    let msg = make_partial();
    let thinking_delta = AssistantMessageEvent::ThinkingDelta {
        content_index: 0,
        delta: "reasoning...".to_string(),
        partial: msg.clone(),
    };
    assert_eq!(thinking_delta.delta(), Some("reasoning..."));

    let tc_delta = AssistantMessageEvent::ToolCallDelta {
        content_index: 0,
        delta: "{\"arg\":".to_string(),
        partial: msg.clone(),
    };
    assert_eq!(tc_delta.delta(), Some("{\"arg\":"));

    // Non-delta events return None
    let done = AssistantMessageEvent::Done {
        reason: StopReason::Stop,
        message: msg,
    };
    assert_eq!(done.delta(), None);
}

// ============================================================================
// Provider as_str coverage for less-common variants
// ============================================================================

#[test]
fn test_provider_as_str_all_variants() {
    // Cover the less-commonly-tested provider variants
    assert_eq!(Provider::AmazonBedrock.as_str(), "amazon-bedrock");
    assert_eq!(Provider::GoogleGeminiCli.as_str(), "google-gemini-cli");
    assert_eq!(Provider::GoogleAntigravity.as_str(), "google-antigravity");
    assert_eq!(Provider::GoogleVertex.as_str(), "google-vertex");
    assert_eq!(Provider::OpenAICompatible.as_str(), "openai-compatible");
    assert_eq!(Provider::OpenAIResponses.as_str(), "openai-responses");
    assert_eq!(
        Provider::AzureOpenAIResponses.as_str(),
        "azure-openai-responses"
    );
    assert_eq!(Provider::OpenAICodex.as_str(), "openai-codex");
    assert_eq!(Provider::GitHubCopilot.as_str(), "github-copilot");
    assert_eq!(Provider::Cerebras.as_str(), "cerebras");
    assert_eq!(Provider::VercelAiGateway.as_str(), "vercel-ai-gateway");
    assert_eq!(Provider::Mistral.as_str(), "mistral");
    assert_eq!(Provider::MiniMaxCN.as_str(), "minimax-cn");
    assert_eq!(Provider::HuggingFace.as_str(), "huggingface");
    assert_eq!(Provider::OpenCode.as_str(), "opencode");
    assert_eq!(Provider::OpenCodeGo.as_str(), "opencode-go");
    assert_eq!(Provider::Ollama.as_str(), "ollama");
    assert_eq!(
        Provider::Custom("my-provider".to_string()).as_str(),
        "my-provider"
    );
}

#[test]
fn test_provider_from_string_all_variants() {
    assert_eq!(
        Provider::from("amazon-bedrock".to_string()),
        Provider::AmazonBedrock
    );
    assert_eq!(
        Provider::from("google-gemini-cli".to_string()),
        Provider::GoogleGeminiCli
    );
    assert_eq!(
        Provider::from("google-antigravity".to_string()),
        Provider::GoogleAntigravity
    );
    assert_eq!(
        Provider::from("google-vertex".to_string()),
        Provider::GoogleVertex
    );
    assert_eq!(
        Provider::from("openai-compatible".to_string()),
        Provider::OpenAICompatible
    );
    assert_eq!(
        Provider::from("openai-responses".to_string()),
        Provider::OpenAIResponses
    );
    assert_eq!(
        Provider::from("azure-openai-responses".to_string()),
        Provider::AzureOpenAIResponses
    );
    assert_eq!(
        Provider::from("openai-codex".to_string()),
        Provider::OpenAICodex
    );
    assert_eq!(
        Provider::from("github-copilot".to_string()),
        Provider::GitHubCopilot
    );
    assert_eq!(Provider::from("cerebras".to_string()), Provider::Cerebras);
    assert_eq!(
        Provider::from("vercel-ai-gateway".to_string()),
        Provider::VercelAiGateway
    );
    assert_eq!(Provider::from("mistral".to_string()), Provider::Mistral);
    assert_eq!(
        Provider::from("minimax-cn".to_string()),
        Provider::MiniMaxCN
    );
    assert_eq!(
        Provider::from("huggingface".to_string()),
        Provider::HuggingFace
    );
    assert_eq!(Provider::from("opencode".to_string()), Provider::OpenCode);
    assert_eq!(
        Provider::from("opencode-go".to_string()),
        Provider::OpenCodeGo
    );
    assert_eq!(Provider::from("ollama".to_string()), Provider::Ollama);
    assert_eq!(
        Provider::from("unknown-provider".to_string()),
        Provider::Custom("unknown-provider".to_string())
    );
}

// ============================================================================
// Model builder: headers and compat setters
// ============================================================================

#[test]
fn test_model_builder_with_headers() {
    let mut headers = std::collections::HashMap::new();
    headers.insert("x-custom".to_string(), "value".to_string());

    let model = Model::builder()
        .id("test")
        .name("Test")
        .provider(Provider::OpenAI)
        .context_window(4096)
        .max_tokens(1024)
        .headers(headers.clone())
        .build()
        .unwrap();

    assert_eq!(model.headers, Some(headers));
}

#[test]
fn test_model_builder_with_compat() {
    let compat = OpenAICompletionsCompat {
        supports_store: false,
        supports_developer_role: true,
        ..Default::default()
    };

    let model = Model::builder()
        .id("test")
        .name("Test")
        .provider(Provider::OpenAI)
        .context_window(4096)
        .max_tokens(1024)
        .compat(compat.clone())
        .build()
        .unwrap();

    assert_eq!(model.compat.unwrap().supports_store, false);
}

// ============================================================================
// SimpleStreamOptions from StreamOptions
// ============================================================================

#[test]
fn test_simple_stream_options_with_reasoning() {
    let simple = SimpleStreamOptions {
        base: StreamOptions {
            api_key: Some("key".to_string()),
            ..Default::default()
        },
        reasoning: Some(tiycore::thinking::ThinkingLevel::Medium),
        thinking_budget_tokens: None,
        thinking_display: None,
    };
    assert!(simple.reasoning.is_some());
    assert_eq!(simple.base.api_key, Some("key".to_string()));
}

// ============================================================================
// StreamOptions Debug (custom impl covers redacted api_key)
// ============================================================================

#[test]
fn test_stream_options_debug_redacts_api_key() {
    let opts = StreamOptions {
        api_key: Some("secret-key-123".to_string()),
        base_url: Some("https://example.com".to_string()),
        ..Default::default()
    };
    let debug_str = format!("{:?}", opts);
    assert!(debug_str.contains("[REDACTED]"));
    assert!(!debug_str.contains("secret-key-123"));
}

// ============================================================================
// Context: ToolBuilder with parameters_from_schema
// ============================================================================

#[test]
fn test_context_tools_is_none_by_default() {
    let ctx = Context::new();
    assert!(ctx.tools.is_none());
}

// ============================================================================
// Message: various From impls
// ============================================================================

#[test]
fn test_message_from_user_message() {
    let user_msg = UserMessage::text("hello");
    let msg: Message = user_msg.into();
    assert!(msg.is_user());
}

#[test]
fn test_message_from_tool_result() {
    let tr = ToolResultMessage::text("call_1", "func", "result text", false);
    let msg: Message = tr.into();
    assert!(msg.is_tool_result());
}
