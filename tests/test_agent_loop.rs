//! Integration tests for the Agent loop with a mock provider.
//!
//! These tests verify the full agent loop: prompt → LLM call → tool execution → loop.

use async_trait::async_trait;
use serde_json::json;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tiycore::agent::*;
use tiycore::provider::{ArcProtocol, LLMProtocol};
use tiycore::stream::AssistantMessageEventStream;
use tiycore::types::*;

// ============================================================================
// Mock Provider
// ============================================================================

/// A mock LLM provider that returns predetermined responses.
struct MockProvider {
    /// Responses to return for each call (consumed in order).
    responses: parking_lot::Mutex<Vec<AssistantMessage>>,
    /// Track how many times stream() was called.
    call_count: AtomicUsize,
}

impl MockProvider {
    fn new(responses: Vec<AssistantMessage>) -> Self {
        Self {
            responses: parking_lot::Mutex::new(responses),
            call_count: AtomicUsize::new(0),
        }
    }

    fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LLMProtocol for MockProvider {
    fn provider_type(&self) -> Provider {
        Provider::OpenAI
    }

    fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: StreamOptions,
    ) -> AssistantMessageEventStream {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        let stream = AssistantMessageEventStream::new_assistant_stream();

        let mut responses = self.responses.lock();
        let response = if responses.is_empty() {
            // Default: return a simple text response
            let mut msg = make_assistant_message("Default response");
            msg.stop_reason = StopReason::Stop;
            msg
        } else {
            responses.remove(0)
        };

        let stop_reason = response.stop_reason;
        let response_clone = response.clone();

        // Push events asynchronously
        let stream_clone = stream.clone();
        tokio::spawn(async move {
            stream_clone.push(AssistantMessageEvent::Start {
                partial: response_clone.clone(),
            });
            stream_clone.push(AssistantMessageEvent::Done {
                reason: stop_reason,
                message: response_clone,
            });
            stream_clone.end(None);
        });

        stream
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        self.stream(model, context, options.base)
    }
}

/// A mock LLM provider that records the request contexts for each call.
struct RecordingProvider {
    responses: parking_lot::Mutex<Vec<AssistantMessage>>,
    contexts: parking_lot::Mutex<Vec<Vec<Message>>>,
    call_count: AtomicUsize,
}

impl RecordingProvider {
    fn new(responses: Vec<AssistantMessage>) -> Self {
        Self {
            responses: parking_lot::Mutex::new(responses),
            contexts: parking_lot::Mutex::new(Vec::new()),
            call_count: AtomicUsize::new(0),
        }
    }

    fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }

    fn contexts(&self) -> Vec<Vec<Message>> {
        self.contexts.lock().clone()
    }
}

#[async_trait]
impl LLMProtocol for RecordingProvider {
    fn provider_type(&self) -> Provider {
        Provider::OpenAI
    }

    fn stream(
        &self,
        _model: &Model,
        context: &Context,
        _options: StreamOptions,
    ) -> AssistantMessageEventStream {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        self.contexts.lock().push(context.messages.clone());

        let stream = AssistantMessageEventStream::new_assistant_stream();

        let mut responses = self.responses.lock();
        let response = if responses.is_empty() {
            let mut msg = make_assistant_message("Default response");
            msg.stop_reason = StopReason::Stop;
            msg
        } else {
            responses.remove(0)
        };

        let stop_reason = response.stop_reason;
        let response_clone = response.clone();
        let stream_clone = stream.clone();
        tokio::spawn(async move {
            stream_clone.push(AssistantMessageEvent::Start {
                partial: response_clone.clone(),
            });
            stream_clone.push(AssistantMessageEvent::Done {
                reason: stop_reason,
                message: response_clone,
            });
            stream_clone.end(None);
        });

        stream
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        self.stream(model, context, options.base)
    }
}
// ============================================================================
// Helper Functions
// ============================================================================

fn make_model() -> Model {
    Model::builder()
        .id("mock-model")
        .name("Mock Model")
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .base_url("http://localhost:0")
        .context_window(128000)
        .max_tokens(4096)
        .build()
        .unwrap()
}

fn make_assistant_message(text: &str) -> AssistantMessage {
    AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("mock-model")
        .content(vec![ContentBlock::Text(TextContent::new(text))])
        .stop_reason(StopReason::Stop)
        .build()
        .unwrap()
}

fn make_tool_call_message(
    tool_name: &str,
    tool_id: &str,
    args: serde_json::Value,
) -> AssistantMessage {
    AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("mock-model")
        .content(vec![ContentBlock::ToolCall(ToolCall::new(
            tool_id, tool_name, args,
        ))])
        .stop_reason(StopReason::ToolUse)
        .build()
        .unwrap()
}

fn messages_contain_user_text(messages: &[Message], expected: &str) -> bool {
    messages.iter().any(|message| {
        matches!(
            message,
            Message::User(user)
                if matches!(&user.content, UserContent::Text(text) if text == expected)
        )
    })
}

// ============================================================================
// Basic Agent Loop Tests
// ============================================================================

#[tokio::test]
async fn test_agent_prompt_with_provider() {
    let response = make_assistant_message("Hello! How can I help?");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    let result = agent.prompt(UserMessage::text("Hello")).await;
    assert!(result.is_ok());

    let messages = result.unwrap();
    assert!(!messages.is_empty());

    // Should have at least the assistant response
    let has_assistant = messages
        .iter()
        .any(|m| matches!(m, AgentMessage::Assistant(_)));
    assert!(has_assistant, "Expected an assistant message in results");

    // State should have user + assistant
    assert!(agent.state().message_count() >= 2);
    assert!(!agent.state().is_streaming());
}

#[tokio::test]
async fn test_agent_prompt_text_content() {
    let response = make_assistant_message("The answer is 42.");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    let result = agent
        .prompt(UserMessage::text("What is the meaning of life?"))
        .await;
    assert!(result.is_ok());

    let messages = result.unwrap();
    let assistant_msg = messages
        .iter()
        .find_map(|m| match m {
            AgentMessage::Assistant(a) => Some(a),
            _ => None,
        })
        .expect("Should have assistant message");

    assert_eq!(assistant_msg.text_content(), "The answer is 42.");
}

// ============================================================================
// Tool Execution Tests
// ============================================================================

#[tokio::test]
async fn test_agent_tool_execution_loop() {
    // First response: tool call, second response: final text
    let tool_response = make_tool_call_message("get_weather", "call_123", json!({"city": "Tokyo"}));
    let final_response = make_assistant_message("The weather in Tokyo is sunny.");

    let mock_provider = Arc::new(MockProvider::new(vec![tool_response, final_response]));
    let provider: ArcProtocol = mock_provider.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_tools(vec![AgentTool::new(
        "get_weather",
        "Get Weather",
        "Get weather for a city",
        json!({"type": "object", "properties": {"city": {"type": "string"}}}),
    )]);

    // Set up tool executor
    agent.set_tool_executor_simple(|name: &str, _id: &str, _args: &serde_json::Value| {
        let name = name.to_string();
        async move {
            if name == "get_weather" {
                AgentToolResult::text("Sunny, 25°C")
            } else {
                AgentToolResult::error(format!("Unknown tool: {}", name))
            }
        }
    });

    let result = agent
        .prompt(UserMessage::text("What's the weather in Tokyo?"))
        .await;
    assert!(result.is_ok());

    let messages = result.unwrap();

    // Should have: assistant(tool_call) + tool_result + assistant(text)
    let assistant_count = messages
        .iter()
        .filter(|m| matches!(m, AgentMessage::Assistant(_)))
        .count();
    assert_eq!(
        assistant_count, 2,
        "Expected 2 assistant messages (tool call + final)"
    );

    let tool_result_count = messages
        .iter()
        .filter(|m| matches!(m, AgentMessage::ToolResult(_)))
        .count();
    assert_eq!(tool_result_count, 1, "Expected 1 tool result");

    // Provider should have been called twice
    assert_eq!(mock_provider.call_count(), 2);
}

#[tokio::test]
async fn test_agent_tool_execution_no_executor() {
    // Tool call without setting an executor should result in error message
    let tool_response = make_tool_call_message("get_weather", "call_123", json!({"city": "Tokyo"}));
    let final_response = make_assistant_message("I couldn't get the weather.");

    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![tool_response, final_response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    let result = agent.prompt(UserMessage::text("What's the weather?")).await;
    assert!(result.is_ok());

    let messages = result.unwrap();
    let tool_results: Vec<_> = messages
        .iter()
        .filter_map(|m| match m {
            AgentMessage::ToolResult(tr) => Some(tr),
            _ => None,
        })
        .collect();

    assert_eq!(tool_results.len(), 1);
    // The tool result should contain an error about no executor
    let result_text: String = tool_results[0]
        .content
        .iter()
        .filter_map(|b| b.as_text())
        .map(|t| t.text.as_str())
        .collect::<Vec<_>>()
        .join("");
    assert!(
        result_text.contains("No tool executor"),
        "Expected error about missing executor, got: {}",
        result_text
    );
}

// ============================================================================
// Max Turns Tests
// ============================================================================

#[tokio::test]
async fn test_agent_max_turns_limit() {
    // Create a provider that always returns tool calls (infinite loop)
    let responses: Vec<AssistantMessage> = (0..30)
        .map(|i| make_tool_call_message("loop_tool", &format!("call_{}", i), json!({})))
        .collect();

    let provider: ArcProtocol = Arc::new(MockProvider::new(responses));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_max_turns(3);

    agent.set_tool_executor_simple(|_name: &str, _id: &str, _args: &serde_json::Value| {
        async move {
            AgentToolResult::text("ok")
        }
    });

    let result = agent.prompt(UserMessage::text("Start")).await;
    assert!(matches!(result, Err(AgentError::MaxTurnsReached(3))));

    // The loop should stop after max_turns without issuing a fourth LLM call.
    let messages = agent.state().messages.read().clone();
    let assistant_count = messages
        .iter()
        .filter(|m| matches!(m, AgentMessage::Assistant(_)))
        .count();
    assert!(
        assistant_count <= 3,
        "Should not exceed max_turns, got {}",
        assistant_count
    );
}

// ============================================================================
// Event Subscription Tests
// ============================================================================

#[tokio::test]
async fn test_agent_events_emitted() {
    let response = make_assistant_message("Hi there!");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    let event_count = Arc::new(AtomicUsize::new(0));
    let agent_start_count = Arc::new(AtomicUsize::new(0));
    let agent_end_count = Arc::new(AtomicUsize::new(0));
    let turn_start_count = Arc::new(AtomicUsize::new(0));

    let ec = event_count.clone();
    let asc = agent_start_count.clone();
    let aec = agent_end_count.clone();
    let tsc = turn_start_count.clone();

    let _unsub = agent.subscribe(move |event| {
        ec.fetch_add(1, Ordering::SeqCst);
        match event {
            AgentEvent::AgentStart => {
                asc.fetch_add(1, Ordering::SeqCst);
            }
            AgentEvent::AgentEnd { .. } => {
                aec.fetch_add(1, Ordering::SeqCst);
            }
            AgentEvent::TurnStart { .. } => {
                tsc.fetch_add(1, Ordering::SeqCst);
            }
            _ => {}
        }
    });

    let result = agent.prompt(UserMessage::text("Hello")).await;
    assert!(result.is_ok());

    assert_eq!(
        agent_start_count.load(Ordering::SeqCst),
        1,
        "Should emit exactly 1 AgentStart"
    );
    assert_eq!(
        agent_end_count.load(Ordering::SeqCst),
        1,
        "Should emit exactly 1 AgentEnd"
    );
    assert!(
        turn_start_count.load(Ordering::SeqCst) >= 1,
        "Should emit at least 1 TurnStart"
    );
    assert!(
        event_count.load(Ordering::SeqCst) >= 3,
        "Should emit multiple events"
    );
}

#[tokio::test]
async fn test_agent_tool_execution_events() {
    let tool_response = make_tool_call_message("my_tool", "call_1", json!({"x": 1}));
    let final_response = make_assistant_message("Done!");

    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![tool_response, final_response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    let tool_start_count = Arc::new(AtomicUsize::new(0));
    let tool_end_count = Arc::new(AtomicUsize::new(0));

    let tsc = tool_start_count.clone();
    let tec = tool_end_count.clone();

    let _unsub = agent.subscribe(move |event| match event {
        AgentEvent::ToolExecutionStart { .. } => {
            tsc.fetch_add(1, Ordering::SeqCst);
        }
        AgentEvent::ToolExecutionEnd { .. } => {
            tec.fetch_add(1, Ordering::SeqCst);
        }
        _ => {}
    });

    agent.set_tool_executor_simple(
        |_name: &str, _id: &str, _args: &serde_json::Value| async move {
            AgentToolResult::text("result")
        },
    );

    let result = agent.prompt(UserMessage::text("Do something")).await;
    assert!(result.is_ok());

    assert_eq!(
        tool_start_count.load(Ordering::SeqCst),
        1,
        "Should emit 1 ToolExecutionStart"
    );
    assert_eq!(
        tool_end_count.load(Ordering::SeqCst),
        1,
        "Should emit 1 ToolExecutionEnd"
    );
}

// ============================================================================
// Continue Tests
// ============================================================================

#[tokio::test]
async fn test_agent_continue_after_tool_result() {
    let response = make_assistant_message("Based on the tool result...");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    // Manually add a user message and tool result
    agent.append_message(AgentMessage::User(UserMessage::text("Do something")));
    agent.append_message(AgentMessage::ToolResult(ToolResultMessage::text(
        "call_1",
        "my_tool",
        "some result",
        false,
    )));

    let result = agent.continue_().await;
    assert!(result.is_ok());

    let messages = result.unwrap();
    let has_assistant = messages
        .iter()
        .any(|m| matches!(m, AgentMessage::Assistant(_)));
    assert!(
        has_assistant,
        "Continue should produce an assistant message"
    );
}

#[tokio::test]
async fn test_agent_continue_already_streaming() {
    let agent = Agent::with_model(make_model());
    agent.state().set_streaming(true);

    let result = agent.continue_().await;
    assert!(matches!(result, Err(AgentError::AlreadyStreaming)));

    agent.state().set_streaming(false);
}

// ============================================================================
// Abort Tests
// ============================================================================

#[tokio::test]
async fn test_agent_abort_resets_state() {
    let agent = Agent::with_model(make_model());
    agent.state().set_streaming(true);
    agent.steer(AgentMessage::User(UserMessage::text("interrupt")));
    agent.follow_up(AgentMessage::User(UserMessage::text("follow")));

    agent.abort();

    assert!(!agent.state().is_streaming());
    assert!(agent.has_queued_messages());
}

// ============================================================================
// Follow-up Tests
// ============================================================================

#[tokio::test]
async fn test_agent_follow_up_processed() {
    // First response is text (no tool call) but a follow-up is queued.
    let response1 = make_assistant_message("First response");
    let response2 = make_assistant_message("Second response");
    let mock_provider = Arc::new(MockProvider::new(vec![response1, response2]));
    let provider: ArcProtocol = mock_provider.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    // Queue a follow-up before prompting.
    agent.follow_up(AgentMessage::User(UserMessage::text("Follow-up question")));

    let result = agent.prompt(UserMessage::text("Hello")).await;
    assert!(result.is_ok());

    // A follow-up after a completed non-tool task should trigger another turn.
    assert!(mock_provider.call_count() >= 1);
}

#[tokio::test]
async fn test_agent_follow_up_waits_until_current_tool_task_completes() {
    let provider = Arc::new(RecordingProvider::new(vec![
        make_tool_call_message("my_tool", "call_1", json!({})),
        make_assistant_message("Current task complete"),
        make_assistant_message("Follow-up handled"),
    ]));
    let arc_provider: ArcProtocol = provider.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(arc_provider);
    agent.set_tool_execution(ToolExecutionMode::Sequential);
    agent.set_tool_executor_simple(
        |_name: &str, _id: &str, _args: &serde_json::Value| async move {
            AgentToolResult::text("tool result")
        },
    );

    agent.follow_up(AgentMessage::User(UserMessage::text("Queued follow-up")));

    let result = agent.prompt(UserMessage::text("Start task")).await;
    assert!(result.is_ok());

    assert_eq!(
        provider.call_count(),
        3,
        "initial task, current task completion, and follow-up should each call the provider"
    );

    let contexts = provider.contexts();
    assert_eq!(contexts.len(), 3);
    assert!(
        !messages_contain_user_text(&contexts[1], "Queued follow-up"),
        "follow-up must not be consumed immediately after the tool-call boundary"
    );
    assert!(
        messages_contain_user_text(&contexts[2], "Queued follow-up"),
        "follow-up should be consumed after the current task's final assistant response"
    );
}

// ============================================================================
// Multiple Tool Calls Tests
// ============================================================================

#[tokio::test]
async fn test_agent_multiple_tool_calls() {
    // Response with two tool calls
    let multi_tool = AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("mock-model")
        .content(vec![
            ContentBlock::ToolCall(ToolCall::new("call_1", "tool_a", json!({"x": 1}))),
            ContentBlock::ToolCall(ToolCall::new("call_2", "tool_b", json!({"y": 2}))),
        ])
        .stop_reason(StopReason::ToolUse)
        .build()
        .unwrap();

    let final_response = make_assistant_message("Both tools executed.");

    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![multi_tool, final_response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    let execution_count = Arc::new(AtomicUsize::new(0));
    let ec = execution_count.clone();

    agent.set_tool_executor_simple(move |_name: &str, _id: &str, _args: &serde_json::Value| {
        let ec = ec.clone();
        async move {
            ec.fetch_add(1, Ordering::SeqCst);
            AgentToolResult::text("ok")
        }
    });

    let result = agent.prompt(UserMessage::text("Do both things")).await;
    assert!(result.is_ok());

    // Both tools should have been executed (parallel by default)
    assert_eq!(
        execution_count.load(Ordering::SeqCst),
        2,
        "Expected 2 tool executions"
    );

    let messages = result.unwrap();
    let tool_result_count = messages
        .iter()
        .filter(|m| matches!(m, AgentMessage::ToolResult(_)))
        .count();
    assert_eq!(tool_result_count, 2, "Expected 2 tool results");
}

#[tokio::test]
async fn test_agent_sequential_tool_execution() {
    let tool_response = make_tool_call_message("my_tool", "call_1", json!({}));
    let final_response = make_assistant_message("Done");

    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![tool_response, final_response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_tool_execution(ToolExecutionMode::Sequential);

    agent.set_tool_executor_simple(
        |_name: &str, _id: &str, _args: &serde_json::Value| async move {
            AgentToolResult::text("sequential result")
        },
    );

    let result = agent.prompt(UserMessage::text("Do it")).await;
    assert!(result.is_ok());
}

// ============================================================================
// State Management Tests
// ============================================================================

#[tokio::test]
async fn test_agent_state_after_prompt() {
    let response = make_assistant_message("Hi!");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_system_prompt("You are helpful.");

    let result = agent.prompt(UserMessage::text("Hello")).await;
    assert!(result.is_ok());

    // State should contain user message + assistant message
    assert_eq!(agent.state().message_count(), 2);
    assert!(!agent.state().is_streaming());
}

#[tokio::test]
async fn test_agent_reset_clears_state() {
    let response = make_assistant_message("Hi!");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    let _ = agent.prompt(UserMessage::text("Hello")).await;
    assert!(agent.state().message_count() > 0);

    agent.reset();
    assert_eq!(agent.state().message_count(), 0);
}
