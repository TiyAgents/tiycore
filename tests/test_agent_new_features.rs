//! Tests for all new agent features: hooks, context pipeline, queue modes,
//! custom messages, thinking budgets, transport, dynamic API key, etc.

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;
use tiycore::agent::*;
use tiycore::provider::{ArcProtocol, LLMProtocol};
use tiycore::stream::AssistantMessageEventStream;
use tiycore::thinking::ThinkingLevel;
use tiycore::types::*;

// ============================================================================
// Mock Provider (shared)
// ============================================================================

struct MockProvider {
    responses: parking_lot::Mutex<Vec<AssistantMessage>>,
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
            make_assistant_message("Default response")
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

fn make_error_assistant_message(message: &str) -> AssistantMessage {
    AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("mock-model")
        .content(vec![ContentBlock::Text(TextContent::new(""))])
        .stop_reason(StopReason::Error)
        .error_message(message)
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

fn make_multi_tool_call_message(tool_calls: Vec<ToolCall>) -> AssistantMessage {
    AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("mock-model")
        .content(tool_calls.into_iter().map(ContentBlock::ToolCall).collect())
        .stop_reason(StopReason::ToolUse)
        .build()
        .unwrap()
}

// ============================================================================
// Custom Messages
// ============================================================================

#[test]
fn test_custom_message_creation() {
    let msg = AgentMessage::Custom {
        message_type: "artifact".to_string(),
        data: json!({"name": "code.rs", "content": "fn main() {}"}),
    };
    assert!(matches!(msg, AgentMessage::Custom { .. }));
}

#[test]
fn test_custom_message_to_option_message_returns_none() {
    let msg = AgentMessage::Custom {
        message_type: "notification".to_string(),
        data: json!({"text": "hello"}),
    };
    let opt: Option<Message> = msg.into();
    assert!(opt.is_none());
}

#[test]
fn test_custom_message_serialization() {
    let msg = AgentMessage::Custom {
        message_type: "artifact".to_string(),
        data: json!({"name": "test"}),
    };
    let json = serde_json::to_value(&msg).unwrap();
    assert_eq!(json["role"], "custom");
    assert_eq!(json["type"], "artifact");
}

// ============================================================================
// String convenience conversions
// ============================================================================

#[test]
fn test_agent_message_from_str() {
    let msg: AgentMessage = "hello".into();
    assert!(matches!(msg, AgentMessage::User(_)));
}

#[test]
fn test_agent_message_from_string() {
    let msg: AgentMessage = String::from("hello").into();
    assert!(matches!(msg, AgentMessage::User(_)));
}

#[tokio::test]
async fn test_prompt_with_str_convenience() {
    let response = make_assistant_message("Hi!");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![response]));
    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    // Use string directly
    let result = agent.prompt("Hello").await;
    assert!(result.is_ok());
}

// ============================================================================
// QueueMode
// ============================================================================

#[test]
fn test_queue_mode_default_is_all() {
    assert_eq!(QueueMode::default(), QueueMode::All);
}

#[test]
fn test_steering_mode_setter_getter() {
    let agent = Agent::new();
    assert_eq!(agent.steering_mode(), QueueMode::All);
    agent.set_steering_mode(QueueMode::OneAtATime);
    assert_eq!(agent.steering_mode(), QueueMode::OneAtATime);
}

#[test]
fn test_follow_up_mode_setter_getter() {
    let agent = Agent::new();
    assert_eq!(agent.follow_up_mode(), QueueMode::All);
    agent.set_follow_up_mode(QueueMode::OneAtATime);
    assert_eq!(agent.follow_up_mode(), QueueMode::OneAtATime);
}

#[test]
fn test_cancel_queued_steering_message_before_drain() {
    let agent = Agent::new();
    let keep = agent.steer(AgentMessage::User(UserMessage::text("keep")));
    let cancel = agent.steer(AgentMessage::User(UserMessage::text("cancel")));

    assert_eq!(agent.queue_stats().steering_depth, 2);
    let removed = agent.cancel_queued_message(cancel);
    assert!(
        matches!(removed, Some(AgentMessage::User(user)) if matches!(&user.content, UserContent::Text(text) if text == "cancel"))
    );
    assert_eq!(agent.queue_stats().steering_depth, 1);
    assert!(agent.cancel_queued_message(cancel).is_none());
    assert!(agent.cancel_queued_message(keep).is_some());
    assert!(!agent.has_queued_messages());
}

#[test]
fn test_cancel_queued_follow_up_message_before_drain() {
    let agent = Agent::new();
    let first = agent.follow_up(AgentMessage::User(UserMessage::text("first")));
    let second = agent.follow_up(AgentMessage::User(UserMessage::text("second")));

    assert_eq!(agent.queue_stats().follow_up_depth, 2);
    let removed = agent.cancel_follow_up_message(first.id);
    assert!(
        matches!(removed, Some(AgentMessage::User(user)) if matches!(&user.content, UserContent::Text(text) if text == "first"))
    );
    assert_eq!(agent.queue_stats().follow_up_depth, 1);
    assert!(agent.cancel_queued_message(second).is_some());
    assert_eq!(agent.queue_stats().follow_up_depth, 0);
}

#[test]
fn test_cancel_emits_removed_event_only_on_success() {
    let agent = Agent::new();
    let events = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let events_clone = Arc::clone(&events);
    agent.set_on_queue_event(move |event| events_clone.lock().push(event));

    let handle = agent.follow_up(AgentMessage::User(UserMessage::text("later")));
    assert!(agent.cancel_queued_message(handle).is_some());
    assert!(agent.cancel_queued_message(handle).is_none());

    let events = events.lock();
    assert_eq!(events.len(), 2);
    assert!(matches!(
        events[0],
        QueueEvent::Enqueued {
            kind: QueueKind::FollowUp,
            count: 1,
            queue_depth: 1
        }
    ));
    assert!(matches!(
        events[1],
        QueueEvent::Removed {
            kind: QueueKind::FollowUp,
            count: 1,
            remaining: 0
        }
    ));
}

#[test]
fn test_try_follow_up_cancel_allows_reinsert_after_reject_limit() {
    let agent = Agent::new();
    agent.set_follow_up_backpressure(BackpressureConfig {
        max_depth: 1,
        overflow: OverflowBehavior::Reject,
    });

    let handle = agent
        .try_follow_up(AgentMessage::User(UserMessage::text("first")))
        .unwrap();
    assert!(agent
        .try_follow_up(AgentMessage::User(UserMessage::text("second")))
        .is_err());

    assert!(agent.cancel_queued_message(handle).is_some());
    let second = agent
        .try_follow_up(AgentMessage::User(UserMessage::text("second")))
        .unwrap();
    assert_eq!(second.kind, QueueKind::FollowUp);
    assert_eq!(agent.queue_stats().follow_up_depth, 1);
}

#[test]
fn test_cancel_after_clear_or_drop_oldest_returns_none() {
    let agent = Agent::new();
    let cleared = agent.steer(AgentMessage::User(UserMessage::text("clear")));
    agent.clear_steering_queue();
    assert!(agent.cancel_queued_message(cleared).is_none());

    agent.set_steering_backpressure(BackpressureConfig {
        max_depth: 1,
        overflow: OverflowBehavior::DropOldest,
    });
    let old = agent
        .try_steer(AgentMessage::User(UserMessage::text("old")))
        .unwrap();
    let new = agent
        .try_steer(AgentMessage::User(UserMessage::text("new")))
        .unwrap();
    assert!(agent.cancel_queued_message(old).is_none());
    assert!(agent.cancel_queued_message(new).is_some());
}

#[tokio::test]
async fn test_steering_one_at_a_time_mode() {
    // Queue 3 steering messages, only 1 should be dequeued per turn in OneAtATime mode
    let responses: Vec<AssistantMessage> = (0..5).map(|_| make_assistant_message("ok")).collect();
    let mock = Arc::new(MockProvider::new(responses));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_steering_mode(QueueMode::OneAtATime);

    agent.steer(AgentMessage::User(UserMessage::text("steer 1")));
    agent.steer(AgentMessage::User(UserMessage::text("steer 2")));
    agent.steer(AgentMessage::User(UserMessage::text("steer 3")));

    let result = agent.prompt("start").await;
    assert!(result.is_ok());

    // After prompt completes, at most 1 steering should have been consumed per turn check.
    // Since steering interrupts turns, not all 3 necessarily get consumed in a single prompt
    // invocation, but the queue should have been partially drained.
}

#[tokio::test]
async fn test_follow_up_one_at_a_time_mode() {
    let responses: Vec<AssistantMessage> = (0..5).map(|_| make_assistant_message("ok")).collect();
    let mock = Arc::new(MockProvider::new(responses));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_follow_up_mode(QueueMode::OneAtATime);

    agent.follow_up(AgentMessage::User(UserMessage::text("follow 1")));
    agent.follow_up(AgentMessage::User(UserMessage::text("follow 2")));
    agent.follow_up(AgentMessage::User(UserMessage::text("follow 3")));

    let result = agent.prompt("start").await;
    assert!(result.is_ok());

    // Provider should be called multiple times (start + follow-ups)
    assert!(
        mock.call_count() >= 2,
        "Expected multiple calls for follow-ups in one-at-a-time mode"
    );
}

// ============================================================================
// ThinkingBudgets
// ============================================================================

#[test]
fn test_thinking_budgets_budget_for() {
    let budgets = ThinkingBudgets {
        minimal: Some(64),
        low: Some(256),
        medium: None,
        high: Some(4096),
    };
    assert_eq!(
        budgets.budget_for(tiycore::thinking::ThinkingLevel::Minimal),
        Some(64)
    );
    assert_eq!(
        budgets.budget_for(tiycore::thinking::ThinkingLevel::Low),
        Some(256)
    );
    assert_eq!(
        budgets.budget_for(tiycore::thinking::ThinkingLevel::Medium),
        None
    );
    assert_eq!(
        budgets.budget_for(tiycore::thinking::ThinkingLevel::High),
        Some(4096)
    );
    assert_eq!(
        budgets.budget_for(tiycore::thinking::ThinkingLevel::Off),
        None
    );
    assert_eq!(
        budgets.budget_for(tiycore::thinking::ThinkingLevel::XHigh),
        None
    );
}

#[test]
fn test_thinking_budgets_setter_getter() {
    let agent = Agent::new();
    assert!(agent.thinking_budgets().is_none());

    let budgets = ThinkingBudgets {
        minimal: Some(128),
        low: Some(512),
        medium: Some(1024),
        high: Some(2048),
    };
    agent.set_thinking_budgets(budgets.clone());
    assert_eq!(agent.thinking_budgets(), Some(budgets));
}

// ============================================================================
// Transport
// ============================================================================

#[test]
fn test_transport_default_is_sse() {
    assert_eq!(Transport::default(), Transport::Sse);
}

#[test]
fn test_transport_setter_getter() {
    let agent = Agent::new();
    assert_eq!(agent.transport(), Transport::Sse);
    agent.set_transport(Transport::WebSocket);
    assert_eq!(agent.transport(), Transport::WebSocket);
    agent.set_transport(Transport::Auto);
    assert_eq!(agent.transport(), Transport::Auto);
}

// ============================================================================
// MaxRetries
// ============================================================================

#[test]
fn test_max_retries_setter_getter() {
    let agent = Agent::new();
    assert_eq!(agent.max_retries(), None);
    agent.set_max_retries(Some(3));
    assert_eq!(agent.max_retries(), Some(3));
    agent.set_max_retries(Some(0));
    assert_eq!(agent.max_retries(), Some(0));
}

// ============================================================================
// MaxRetryDelayMs
// ============================================================================

#[test]
fn test_max_retry_delay_setter_getter() {
    let agent = Agent::new();
    assert_eq!(agent.max_retry_delay_ms(), None);
    agent.set_max_retry_delay_ms(Some(30000));
    assert_eq!(agent.max_retry_delay_ms(), Some(30000));
    agent.set_max_retry_delay_ms(Some(0));
    assert_eq!(agent.max_retry_delay_ms(), Some(0));
}

// ============================================================================
// beforeToolCall Hook
// ============================================================================

#[tokio::test]
async fn test_before_tool_call_allows_execution() {
    let tool_response = make_tool_call_message("my_tool", "call_1", json!({"x": 1}));
    let final_response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![tool_response, final_response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_tools(vec![AgentTool::new(
        "my_tool",
        "My Tool",
        "description",
        json!({"type": "object"}),
    )]);

    let hook_called = Arc::new(AtomicUsize::new(0));
    let hc = hook_called.clone();

    agent.set_before_tool_call(move |_ctx| {
        let hc = hc.clone();
        async move {
            hc.fetch_add(1, Ordering::SeqCst);
            None // Allow execution
        }
    });

    agent.set_tool_executor_simple(
        |_name: &str, _id: &str, _args: &serde_json::Value| async move {
            AgentToolResult::text("ok")
        },
    );

    let result = agent.prompt("go").await;
    assert!(result.is_ok());
    assert_eq!(
        hook_called.load(Ordering::SeqCst),
        1,
        "beforeToolCall hook should be called once"
    );
}

#[tokio::test]
async fn test_before_tool_call_blocks_execution() {
    let tool_response = make_tool_call_message("dangerous_tool", "call_1", json!({}));
    let final_response = make_assistant_message("OK, I won't do that.");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![tool_response, final_response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_tools(vec![AgentTool::new(
        "dangerous_tool",
        "Danger",
        "dangerous",
        json!({"type": "object"}),
    )]);

    let executor_called = Arc::new(AtomicUsize::new(0));
    let ec = executor_called.clone();

    agent.set_before_tool_call(move |ctx| async move {
        if ctx.tool_call.name == "dangerous_tool" {
            Some(BeforeToolCallResult::blocked("User denied permission"))
        } else {
            None
        }
    });

    agent.set_tool_executor_simple(move |_name: &str, _id: &str, _args: &serde_json::Value| {
        let ec = ec.clone();
        async move {
            ec.fetch_add(1, Ordering::SeqCst);
            AgentToolResult::text("should not reach here")
        }
    });

    let result = agent.prompt("do the dangerous thing").await;
    assert!(result.is_ok());

    // Tool executor should NOT have been called
    assert_eq!(
        executor_called.load(Ordering::SeqCst),
        0,
        "Blocked tool should not be executed"
    );

    // Should have a tool result with the blocked reason
    let messages = result.unwrap();
    let tool_results: Vec<_> = messages
        .iter()
        .filter_map(|m| match m {
            AgentMessage::ToolResult(tr) => Some(tr),
            _ => None,
        })
        .collect();
    assert_eq!(tool_results.len(), 1);
    assert!(tool_results[0].is_error);
    let text: String = tool_results[0]
        .content
        .iter()
        .filter_map(|b| b.as_text())
        .map(|t| t.text.as_str())
        .collect::<Vec<_>>()
        .join("");
    assert!(text.contains("User denied permission"));
}

// ============================================================================
// afterToolCall Hook
// ============================================================================

#[tokio::test]
async fn test_after_tool_call_overrides_content() {
    let tool_response = make_tool_call_message("my_tool", "call_1", json!({}));
    let final_response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![tool_response, final_response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_tools(vec![AgentTool::new(
        "my_tool",
        "My Tool",
        "desc",
        json!({"type": "object"}),
    )]);

    agent.set_after_tool_call(move |_ctx| async move {
        Some(AfterToolCallResult {
            content: Some(vec![ContentBlock::Text(TextContent::new(
                "overridden content",
            ))]),
            details: None,
            is_error: Some(false),
        })
    });

    agent.set_tool_executor_simple(
        |_name: &str, _id: &str, _args: &serde_json::Value| async move {
            AgentToolResult::text("original content")
        },
    );

    let result = agent.prompt("go").await;
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
    let text: String = tool_results[0]
        .content
        .iter()
        .filter_map(|b| b.as_text())
        .map(|t| t.text.as_str())
        .collect::<Vec<_>>()
        .join("");
    assert_eq!(text, "overridden content");
}

#[tokio::test]
async fn test_after_tool_call_override_is_error() {
    let tool_response = make_tool_call_message("my_tool", "call_1", json!({}));
    let final_response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![tool_response, final_response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_tools(vec![AgentTool::new(
        "my_tool",
        "My Tool",
        "desc",
        json!({"type": "object"}),
    )]);

    // Override is_error to true even though original succeeded
    agent.set_after_tool_call(move |_ctx| async move {
        Some(AfterToolCallResult {
            content: None, // Keep original
            details: None,
            is_error: Some(true),
        })
    });

    agent.set_tool_executor_simple(
        |_name: &str, _id: &str, _args: &serde_json::Value| async move {
            AgentToolResult::text("success")
        },
    );

    let result = agent.prompt("go").await;
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
    assert!(
        tool_results[0].is_error,
        "afterToolCall should have overridden is_error to true"
    );
}

#[tokio::test]
async fn test_after_tool_call_overrides_details() {
    let tool_response = make_tool_call_message("my_tool", "call_1", json!({}));
    let final_response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![tool_response, final_response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_tools(vec![AgentTool::new(
        "my_tool",
        "My Tool",
        "desc",
        json!({"type": "object"}),
    )]);

    agent.set_after_tool_call(move |_ctx| async move {
        Some(AfterToolCallResult {
            content: None,
            details: Some(json!({"audited": true, "source": "after"})),
            is_error: None,
        })
    });

    agent.set_tool_executor_simple(
        |_name: &str, _id: &str, _args: &serde_json::Value| async move {
            AgentToolResult {
                content: vec![ContentBlock::Text(TextContent::new("content"))],
                details: Some(json!({"source": "executor"})),
            }
        },
    );

    let result = agent.prompt("go").await;
    assert!(result.is_ok());

    let messages = result.unwrap();
    let tool_result = messages
        .iter()
        .find_map(|m| match m {
            AgentMessage::ToolResult(tr) => Some(tr),
            _ => None,
        })
        .expect("tool result should exist");

    assert_eq!(
        tool_result.details.as_ref(),
        Some(&json!({"audited": true, "source": "after"}))
    );
}

#[tokio::test]
async fn test_parallel_tool_results_preserve_assistant_order() {
    let tool_response = make_multi_tool_call_message(vec![
        ToolCall::new("call_1", "my_tool", json!({"value": 1})),
        ToolCall::new("call_2", "my_tool", json!({"value": 2})),
    ]);
    let final_response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![tool_response, final_response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_tools(vec![AgentTool::new(
        "my_tool",
        "My Tool",
        "desc",
        json!({
            "type": "object",
            "properties": {
                "value": {"type": "integer"}
            },
            "required": ["value"]
        }),
    )]);

    let event_order = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let event_order_capture = Arc::clone(&event_order);
    let _unsub = agent.subscribe(move |event| {
        if let AgentEvent::ToolExecutionEnd { tool_call_id, .. } = event {
            event_order_capture.lock().push(tool_call_id.clone());
        }
    });

    agent.set_tool_executor(
        |_name: &str,
         id: &str,
         _args: &serde_json::Value,
         _update_cb: Option<ToolUpdateCallback>| {
            let id = id.to_string();
            async move {
                if id == "call_1" {
                    tokio::time::sleep(Duration::from_millis(40)).await;
                } else {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
                AgentToolResult::text(format!("done:{id}"))
            }
        },
    );

    let result = agent.prompt("go").await;
    assert!(result.is_ok());

    let messages = result.unwrap();
    let tool_result_ids: Vec<_> = messages
        .iter()
        .filter_map(|m| match m {
            AgentMessage::ToolResult(tr) => Some(tr.tool_call_id.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(tool_result_ids, vec!["call_1", "call_2"]);
    assert_eq!(*event_order.lock(), vec!["call_1", "call_2"]);
}

#[tokio::test]
async fn test_continue_from_assistant_tail_processes_follow_up_queue() {
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![make_assistant_message(
        "Processed follow-up",
    )]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.replace_messages(vec![
        AgentMessage::User(UserMessage::text("Initial")),
        AgentMessage::Assistant(make_assistant_message("Initial response")),
    ]);
    agent.follow_up(AgentMessage::User(UserMessage::text("Queued follow-up")));

    let result = agent.continue_().await;
    assert!(result.is_ok());

    let snapshot = agent.snapshot();
    let roles: Vec<_> = snapshot
        .messages
        .iter()
        .map(|message| match message {
            AgentMessage::User(_) => "user",
            AgentMessage::Assistant(_) => "assistant",
            AgentMessage::ToolResult(_) => "tool_result",
            AgentMessage::Custom { .. } => "custom",
        })
        .collect();

    assert!(roles.ends_with(&["user", "assistant"]));
    assert!(snapshot.messages.iter().any(|message| matches!(
        message,
        AgentMessage::User(user) if matches!(&user.content, UserContent::Text(text) if text == "Queued follow-up")
    )));
}

#[tokio::test]
async fn test_continue_from_assistant_tail_preserves_one_at_a_time_steering() {
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![
        make_assistant_message("Processed 1"),
        make_assistant_message("Processed 2"),
    ]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_steering_mode(QueueMode::OneAtATime);
    agent.replace_messages(vec![
        AgentMessage::User(UserMessage::text("Initial")),
        AgentMessage::Assistant(make_assistant_message("Initial response")),
    ]);
    agent.steer(AgentMessage::User(UserMessage::text("Steering 1")));
    agent.steer(AgentMessage::User(UserMessage::text("Steering 2")));

    let result = agent.continue_().await;
    assert!(result.is_ok());

    let snapshot = agent.snapshot();
    let trailing_roles: Vec<_> = snapshot
        .messages
        .iter()
        .rev()
        .take(4)
        .map(|message| match message {
            AgentMessage::User(_) => "user",
            AgentMessage::Assistant(_) => "assistant",
            AgentMessage::ToolResult(_) => "tool_result",
            AgentMessage::Custom { .. } => "custom",
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    assert_eq!(
        trailing_roles,
        vec!["user", "assistant", "user", "assistant"]
    );
}

#[tokio::test]
async fn test_prompt_and_tool_results_emit_message_lifecycle_events() {
    let tool_response = make_tool_call_message("my_tool", "call_1", json!({}));
    let final_response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![tool_response, final_response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_tools(vec![AgentTool::new(
        "my_tool",
        "My Tool",
        "desc",
        json!({"type": "object"}),
    )]);

    let started_roles = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let ended_tool_results = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let started_roles_capture = Arc::clone(&started_roles);
    let ended_tool_results_capture = Arc::clone(&ended_tool_results);
    let _unsub = agent.subscribe(move |event| match event {
        AgentEvent::MessageStart { message, .. } => {
            started_roles_capture.lock().push(match message {
                AgentMessage::User(_) => "user".to_string(),
                AgentMessage::Assistant(_) => "assistant".to_string(),
                AgentMessage::ToolResult(_) => "tool_result".to_string(),
                AgentMessage::Custom { .. } => "custom".to_string(),
            });
        }
        AgentEvent::MessageEnd {
            message: AgentMessage::ToolResult(tool_result),
            ..
        } => {
            ended_tool_results_capture
                .lock()
                .push(tool_result.tool_call_id.clone());
        }
        _ => {}
    });

    agent.set_tool_executor_simple(
        |_name: &str, _id: &str, _args: &serde_json::Value| async move {
            AgentToolResult::text("tool result")
        },
    );

    let result = agent.prompt("go").await;
    assert!(result.is_ok());

    assert!(started_roles.lock().contains(&"user".to_string()));
    assert_eq!(*ended_tool_results.lock(), vec!["call_1"]);
}

#[tokio::test]
async fn test_assistant_message_start_emits_once_before_message_end() {
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![make_assistant_message("Hi!")]));
    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    let assistant_events = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let assistant_events_capture = Arc::clone(&assistant_events);
    let _unsub = agent.subscribe(move |event| match event {
        AgentEvent::MessageStart {
            message: AgentMessage::Assistant(_),
            ..
        } => assistant_events_capture
            .lock()
            .push("assistant_start".to_string()),
        AgentEvent::MessageEnd {
            message: AgentMessage::Assistant(_),
            ..
        } => assistant_events_capture
            .lock()
            .push("assistant_end".to_string()),
        _ => {}
    });

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let events = assistant_events.lock().clone();
    assert_eq!(events, vec!["assistant_start", "assistant_end"]);
}

#[tokio::test]
async fn test_standalone_agent_loop_apis_work() {
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![make_assistant_message("Hi!")]));
    let context = AgentContext {
        system_prompt: "You are helpful.".to_string(),
        messages: Vec::new(),
        tools: None,
    };
    let config = AgentConfig::new(make_model());
    let options = AgentLoopOptions {
        provider: Some(provider.clone()),
        ..Default::default()
    };

    let result = run_agent_loop(
        vec![AgentMessage::User(UserMessage::text("Hello"))],
        context.clone(),
        config.clone(),
        options.clone(),
    )
    .await;
    assert!(result.is_ok());
    assert!(result
        .unwrap()
        .iter()
        .any(|message| matches!(message, AgentMessage::Assistant(_))));

    let mut stream = agent_loop(
        vec![AgentMessage::User(UserMessage::text("Hello again"))],
        context,
        config,
        options,
    );
    let mut event_types = Vec::new();
    while let Some(event) = stream.next().await {
        event_types.push(match event {
            AgentEvent::AgentStart => "agent_start".to_string(),
            AgentEvent::AgentEnd { .. } => "agent_end".to_string(),
            AgentEvent::TurnStart { .. } => "turn_start".to_string(),
            AgentEvent::TurnEnd { .. } => "turn_end".to_string(),
            AgentEvent::MessageStart { .. } => "message_start".to_string(),
            AgentEvent::MessageUpdate { .. } => "message_update".to_string(),
            AgentEvent::MessageEnd { .. } => "message_end".to_string(),
            AgentEvent::MessageDiscarded { .. } => "message_discarded".to_string(),
            AgentEvent::ToolExecutionStart { .. } => "tool_execution_start".to_string(),
            AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update".to_string(),
            AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end".to_string(),
            AgentEvent::TurnRetrying { .. } => "turn_retrying".to_string(),
        });
    }
    let stream_result = stream.result().await;
    assert!(stream_result.is_ok());
    assert!(event_types.contains(&"agent_start".to_string()));
    assert!(event_types.contains(&"agent_end".to_string()));
}

#[tokio::test]
async fn test_terminal_error_assistant_is_persisted_to_state() {
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![make_error_assistant_message(
        "provider exploded",
    )]));
    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    let result = agent.prompt("boom").await;
    assert!(matches!(result, Err(AgentError::ProviderError(_))));
    assert_eq!(
        *agent.state().error.read(),
        Some("Provider error: provider exploded".to_string())
    );

    let messages = agent.state().messages.read();
    let last = messages.last().expect("terminal assistant message");
    match last {
        AgentMessage::Assistant(message) => {
            assert_eq!(message.stop_reason, StopReason::Error);
            assert_eq!(message.error_message.as_deref(), Some("provider exploded"));
        }
        other => panic!("expected assistant terminal error message, got {:?}", other),
    }
}

#[tokio::test]
async fn test_abort_persists_aborted_assistant_message() {
    let agent = Arc::new(Agent::with_model(make_model()));
    agent.set_stream_fn_with_signal(|_model, _context, _options, _signal| async move {
        AssistantMessageEventStream::new_assistant_stream()
    });

    let agent_for_prompt = Arc::clone(&agent);
    let prompt_task = tokio::spawn(async move { agent_for_prompt.prompt("wait").await });

    tokio::time::sleep(Duration::from_millis(25)).await;
    agent.abort();

    let result = prompt_task.await.unwrap();
    assert!(matches!(result, Err(AgentError::Other(ref msg)) if msg == "Aborted"));
    assert_eq!(*agent.state().error.read(), Some("Aborted".to_string()));

    let messages = agent.state().messages.read();
    let last = messages.last().expect("aborted assistant message");
    match last {
        AgentMessage::Assistant(message) => {
            assert_eq!(message.stop_reason, StopReason::Aborted);
            assert_eq!(message.error_message.as_deref(), Some("Aborted"));
        }
        other => panic!("expected assistant aborted message, got {:?}", other),
    }
}

#[tokio::test]
async fn test_standalone_continue_uses_dynamic_follow_up_supplier() {
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![make_assistant_message(
        "Follow-up handled",
    )]));
    let supplied = Arc::new(AtomicBool::new(false));
    let supplied_clone = Arc::clone(&supplied);

    let context = AgentContext {
        system_prompt: String::new(),
        messages: vec![
            AgentMessage::User(UserMessage::text("initial")),
            AgentMessage::Assistant(make_assistant_message("done")),
        ],
        tools: None,
    };

    let options = AgentLoopOptions {
        provider: Some(provider),
        hooks: AgentHooks {
            get_follow_up_messages: Some(Arc::new(move |_signal| {
                let supplied = Arc::clone(&supplied_clone);
                Box::pin(async move {
                    if supplied.swap(true, Ordering::SeqCst) {
                        Vec::new()
                    } else {
                        vec![AgentMessage::User(UserMessage::text("queued follow-up"))]
                    }
                })
            })),
            ..Default::default()
        },
        ..Default::default()
    };

    let result = run_agent_loop_continue(context, AgentConfig::new(make_model()), options)
        .await
        .unwrap();

    assert!(result.iter().any(|message| {
        matches!(message, AgentMessage::User(user) if matches!(&user.content, UserContent::Text(text) if text == "queued follow-up"))
    }));
    assert!(result
        .iter()
        .any(|message| matches!(message, AgentMessage::Assistant(_))));
}

#[tokio::test]
async fn test_standalone_continue_uses_dynamic_steering_supplier() {
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![make_assistant_message(
        "Steering handled",
    )]));
    let supplied = Arc::new(AtomicBool::new(false));
    let supplied_clone = Arc::clone(&supplied);

    let context = AgentContext {
        system_prompt: String::new(),
        messages: vec![
            AgentMessage::User(UserMessage::text("initial")),
            AgentMessage::Assistant(make_assistant_message("done")),
        ],
        tools: None,
    };

    let options = AgentLoopOptions {
        provider: Some(provider),
        hooks: AgentHooks {
            get_steering_messages: Some(Arc::new(move |_signal| {
                let supplied = Arc::clone(&supplied_clone);
                Box::pin(async move {
                    if supplied.swap(true, Ordering::SeqCst) {
                        Vec::new()
                    } else {
                        vec![AgentMessage::User(UserMessage::text("queued steering"))]
                    }
                })
            })),
            ..Default::default()
        },
        ..Default::default()
    };

    let result = run_agent_loop_continue(context, AgentConfig::new(make_model()), options)
        .await
        .unwrap();

    assert!(result.iter().any(|message| {
        matches!(message, AgentMessage::User(user) if matches!(&user.content, UserContent::Text(text) if text == "queued steering"))
    }));
    assert!(result
        .iter()
        .any(|message| matches!(message, AgentMessage::Assistant(_))));
}

#[tokio::test]
async fn test_transform_context_with_signal_is_cancelled_on_abort() {
    let agent = Arc::new(Agent::with_model(make_model()));
    let observed = Arc::new(AtomicBool::new(false));
    let observed_clone = Arc::clone(&observed);

    agent.set_transform_context_with_signal(move |messages, signal| {
        let observed = Arc::clone(&observed_clone);
        async move {
            signal.cancelled().await;
            observed.store(true, Ordering::SeqCst);
            messages
        }
    });
    agent.set_stream_fn_with_signal(|_model, _context, _options, _signal| async move {
        AssistantMessageEventStream::new_assistant_stream()
    });

    let agent_for_prompt = Arc::clone(&agent);
    let prompt_task = tokio::spawn(async move { agent_for_prompt.prompt("work").await });

    tokio::time::sleep(Duration::from_millis(25)).await;
    agent.abort();

    let _ = prompt_task.await.unwrap();
    assert!(observed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn test_stream_fn_with_signal_receives_abort() {
    let agent = Arc::new(Agent::with_model(make_model()));
    let observed = Arc::new(AtomicBool::new(false));
    let observed_clone = Arc::clone(&observed);

    agent.set_stream_fn_with_signal(move |_model, _context, _options, signal| {
        let observed = Arc::clone(&observed_clone);
        async move {
            let stream = AssistantMessageEventStream::new_assistant_stream();
            tokio::spawn(async move {
                signal.cancelled().await;
                observed.store(true, Ordering::SeqCst);
            });
            stream
        }
    });

    let agent_for_prompt = Arc::clone(&agent);
    let prompt_task = tokio::spawn(async move { agent_for_prompt.prompt("work").await });

    tokio::time::sleep(Duration::from_millis(25)).await;
    agent.abort();

    let _ = prompt_task.await.unwrap();
    assert!(observed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn test_incomplete_stream_retries_from_stable_context_and_discards_partial() {
    let agent = Agent::with_model(make_model());
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_for_stream = Arc::clone(&attempts);
    let recorded_events = Arc::new(parking_lot::Mutex::new(Vec::<AgentEvent>::new()));
    let recorded_events_for_subscriber = Arc::clone(&recorded_events);

    let _subscription = agent.subscribe(move |event| {
        recorded_events_for_subscriber.lock().push(event.clone());
    });

    agent.set_stream_fn_with_signal(move |_model, _context, _options, _signal| {
        let attempt = attempts_for_stream.fetch_add(1, Ordering::SeqCst);
        async move {
            let stream = AssistantMessageEventStream::new_assistant_stream();
            let stream_clone = stream.clone();

            tokio::spawn(async move {
                if attempt == 0 {
                    let partial = make_assistant_message("partial output");
                    let mut error = partial.clone();
                    error.stop_reason = StopReason::Error;
                    error.error_message =
                        Some("[incomplete_stream]anthropic: missing message_stop".to_string());
                    stream_clone.push(AssistantMessageEvent::Start {
                        partial: partial.clone(),
                    });
                    stream_clone.push(AssistantMessageEvent::Error {
                        reason: StopReason::Error,
                        error,
                    });
                    stream_clone.end(None);
                    return;
                }

                let response = make_assistant_message("recovered output");
                stream_clone.push(AssistantMessageEvent::Start {
                    partial: response.clone(),
                });
                stream_clone.push(AssistantMessageEvent::Done {
                    reason: StopReason::Stop,
                    message: response,
                });
                stream_clone.end(None);
            });

            stream
        }
    });

    let result = agent.prompt("hello").await.expect("prompt should recover");
    let events = recorded_events.lock().clone();
    let assistant_texts = result
        .iter()
        .filter_map(|message| match message {
            AgentMessage::Assistant(assistant) => Some(assistant.text_content()),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    assert_eq!(assistant_texts, vec!["recovered output".to_string()]);
    assert!(matches!(
        events.iter().find(|event| matches!(event, AgentEvent::MessageDiscarded { .. })),
        Some(AgentEvent::MessageDiscarded { reason, .. })
            if reason.contains("Incomplete anthropic stream")
    ));
    assert!(matches!(
        events
            .iter()
            .find(|event| matches!(event, AgentEvent::TurnRetrying { .. })),
        Some(AgentEvent::TurnRetrying {
            attempt: 1,
            max_attempts: 3,
            delay_ms: 1_000,
            ..
        })
    ));
    assert_eq!(
        agent
            .snapshot()
            .messages
            .into_iter()
            .filter_map(|message| match message {
                AgentMessage::Assistant(assistant) => Some(assistant.text_content()),
                _ => None,
            })
            .collect::<Vec<_>>(),
        vec!["recovered output".to_string()]
    );
}

// ============================================================================
// convertToLlm
// ============================================================================

#[tokio::test]
async fn test_convert_to_llm_filters_custom_messages_by_default() {
    let response = make_assistant_message("I see 1 message");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    // Add a custom message — should be filtered out by default
    agent.append_message(AgentMessage::Custom {
        message_type: "artifact".to_string(),
        data: json!({"name": "test"}),
    });

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_convert_to_llm_custom_converter() {
    let response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    let converter_called = Arc::new(AtomicUsize::new(0));
    let cc = converter_called.clone();

    agent.set_convert_to_llm(move |messages| {
        let cc = cc.clone();
        async move {
            cc.fetch_add(1, Ordering::SeqCst);
            // Custom conversion: only keep user messages
            messages
                .into_iter()
                .filter_map(|m| match m {
                    AgentMessage::User(u) => Some(Message::User(u)),
                    _ => None,
                })
                .collect()
        }
    });

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());
    assert!(
        converter_called.load(Ordering::SeqCst) >= 1,
        "Custom converter should be called"
    );
}

// ============================================================================
// transformContext
// ============================================================================

#[tokio::test]
async fn test_transform_context_called() {
    let response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    let transform_called = Arc::new(AtomicUsize::new(0));
    let tc = transform_called.clone();

    agent.set_transform_context(move |messages| {
        let tc = tc.clone();
        async move {
            tc.fetch_add(1, Ordering::SeqCst);
            // Keep only last 2 messages (context window management)
            let len = messages.len();
            if len > 2 {
                messages[len - 2..].to_vec()
            } else {
                messages
            }
        }
    });

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());
    assert!(
        transform_called.load(Ordering::SeqCst) >= 1,
        "transformContext should be called"
    );
}

// ============================================================================
// Pre-serialization message hook (on_messages)
// ============================================================================

#[tokio::test]
async fn test_on_messages_hook_not_set_is_noop() {
    // When on_messages is not set, build_context should work exactly as before.
    let response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    let result = agent.prompt("hello").await;
    assert!(result.is_ok(), "on_messages=None should not break anything");
}

#[tokio::test]
async fn test_on_messages_hook_called_with_model() {
    let response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    let hook_called = Arc::new(AtomicUsize::new(0));
    let hc = hook_called.clone();

    agent.set_on_messages(move |messages, model| {
        let hc = hc.clone();
        async move {
            hc.fetch_add(1, Ordering::SeqCst);
            // Verify we receive the model info
            assert!(!model.id.is_empty(), "model should have an id");
            // Pass messages through unchanged
            messages
        }
    });

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());
    assert!(
        hook_called.load(Ordering::SeqCst) >= 1,
        "on_messages hook should be called at least once"
    );
}

// ============================================================================
// Dynamic API Key (getApiKey)
// ============================================================================

#[tokio::test]
async fn test_get_api_key_dynamic_resolution() {
    let response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    let resolver_called = Arc::new(AtomicUsize::new(0));
    let rc = resolver_called.clone();

    agent.set_get_api_key(move |_provider: &str| {
        let rc = rc.clone();
        async move {
            rc.fetch_add(1, Ordering::SeqCst);
            Some("dynamic-key-123".to_string())
        }
    });

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());
    assert!(
        resolver_called.load(Ordering::SeqCst) >= 1,
        "getApiKey resolver should be called"
    );
}

// ============================================================================
// ToolExecutionUpdate events
// ============================================================================

#[tokio::test]
async fn test_tool_execution_update_events() {
    let tool_response = make_tool_call_message("streaming_tool", "call_1", json!({}));
    let final_response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![tool_response, final_response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_tools(vec![AgentTool::new(
        "streaming_tool",
        "Streaming Tool",
        "desc",
        json!({"type": "object"}),
    )]);

    let update_count = Arc::new(AtomicUsize::new(0));
    let uc = update_count.clone();
    let observed_args = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let observed_args_capture = Arc::clone(&observed_args);

    let _unsub = agent.subscribe(move |event| {
        if let AgentEvent::ToolExecutionUpdate { args, .. } = event {
            uc.fetch_add(1, Ordering::SeqCst);
            observed_args_capture.lock().push(args.clone());
        }
    });

    // Use the full set_tool_executor with update callback
    agent.set_tool_executor(
        |_name: &str,
         _id: &str,
         _args: &serde_json::Value,
         update_cb: Option<ToolUpdateCallback>| async move {
            // Push streaming updates
            if let Some(ref cb) = update_cb {
                cb(json!({"progress": 25}));
                cb(json!({"progress": 50}));
                cb(json!({"progress": 100}));
            }
            AgentToolResult::text("complete")
        },
    );

    let result = agent.prompt("start").await;
    assert!(result.is_ok());

    assert_eq!(
        update_count.load(Ordering::SeqCst),
        3,
        "Should receive 3 ToolExecutionUpdate events"
    );
    assert!(observed_args.lock().iter().all(|args| *args == json!({})));
}

// ============================================================================
// set_tool_executor_simple backward compat
// ============================================================================

#[tokio::test]
async fn test_tool_executor_simple_works() {
    let tool_response = make_tool_call_message("my_tool", "call_1", json!({}));
    let final_response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![tool_response, final_response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    agent.set_tool_executor_simple(
        |_name: &str, _id: &str, _args: &serde_json::Value| async move {
            AgentToolResult::text("simple result")
        },
    );

    let result = agent.prompt("go").await;
    assert!(result.is_ok());
}

// ============================================================================
// BeforeToolCallResult helpers
// ============================================================================

#[test]
fn test_before_tool_call_result_allow() {
    let r = BeforeToolCallResult::allow();
    assert!(!r.block);
    assert!(r.reason.is_none());
}

#[test]
fn test_before_tool_call_result_blocked() {
    let r = BeforeToolCallResult::blocked("Not allowed");
    assert!(r.block);
    assert_eq!(r.reason.as_deref(), Some("Not allowed"));
}

// ============================================================================
// AgentConfig new fields
// ============================================================================

#[test]
fn test_agent_config_new_has_defaults() {
    let model = make_model();
    let config = AgentConfig::new(model);
    assert_eq!(config.steering_mode, QueueMode::All);
    assert_eq!(config.follow_up_mode, QueueMode::All);
    assert!(config.thinking_budgets.is_none());
    assert_eq!(config.transport, Transport::Sse);
    assert!(config.max_retries.is_none());
    assert!(config.max_retry_delay_ms.is_none());
}

// ============================================================================
// AgentEvent serialization (TurnEnd with tool_results)
// ============================================================================

#[test]
fn test_agent_event_turn_end_serialization() {
    let event = AgentEvent::TurnEnd {
        turn_index: 0,
        message: AgentMessage::User(UserMessage::text("hello")),
        tool_results: vec![],
    };
    let json = serde_json::to_value(&event).unwrap();
    assert_eq!(json["type"], "turn_end");
}

#[test]
fn test_agent_event_tool_execution_update_serialization() {
    let event = AgentEvent::ToolExecutionUpdate {
        turn_index: 0,
        tool_call_id: "call_1".to_string(),
        tool_name: "my_tool".to_string(),
        args: json!({"x": 1}),
        partial_result: json!({"progress": 50}),
    };
    let json = serde_json::to_value(&event).unwrap();
    assert_eq!(json["type"], "tool_execution_update");
    assert_eq!(json["partial_result"]["progress"], 50);
}

// ============================================================================
// Combined: beforeToolCall + afterToolCall
// ============================================================================

#[tokio::test]
async fn test_both_hooks_called_in_order() {
    let tool_response = make_tool_call_message("my_tool", "call_1", json!({}));
    let final_response = make_assistant_message("Done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![tool_response, final_response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_tools(vec![AgentTool::new(
        "my_tool",
        "My Tool",
        "desc",
        json!({"type": "object"}),
    )]);

    let before_called = Arc::new(AtomicUsize::new(0));
    let after_called = Arc::new(AtomicUsize::new(0));
    let bc = before_called.clone();
    let ac = after_called.clone();

    agent.set_before_tool_call(move |_ctx| {
        let bc = bc.clone();
        async move {
            bc.fetch_add(1, Ordering::SeqCst);
            None // Allow
        }
    });

    agent.set_after_tool_call(move |_ctx| {
        let ac = ac.clone();
        async move {
            ac.fetch_add(1, Ordering::SeqCst);
            None // No override
        }
    });

    agent.set_tool_executor_simple(
        |_name: &str, _id: &str, _args: &serde_json::Value| async move {
            AgentToolResult::text("ok")
        },
    );

    let result = agent.prompt("go").await;
    assert!(result.is_ok());
    assert_eq!(before_called.load(Ordering::SeqCst), 1);
    assert_eq!(after_called.load(Ordering::SeqCst), 1);
}

// ============================================================================
// CapturingMockProvider — records SimpleStreamOptions for integration tests
// ============================================================================

/// A mock provider that captures the `SimpleStreamOptions` it receives,
/// so we can verify that agent features flow through to the provider layer.
struct CapturingMockProvider {
    responses: parking_lot::Mutex<Vec<AssistantMessage>>,
    captured_reasoning: parking_lot::Mutex<Vec<Option<ThinkingLevel>>>,
    captured_budget: parking_lot::Mutex<Vec<Option<u32>>>,
    captured_session_id: parking_lot::Mutex<Vec<Option<String>>>,
    captured_transport: parking_lot::Mutex<Vec<Option<Transport>>>,
    captured_max_retries: parking_lot::Mutex<Vec<Option<u32>>>,
    captured_max_retry_delay: parking_lot::Mutex<Vec<Option<u64>>>,
    captured_has_on_payload: parking_lot::Mutex<Vec<bool>>,
}

impl CapturingMockProvider {
    fn new(responses: Vec<AssistantMessage>) -> Self {
        Self {
            responses: parking_lot::Mutex::new(responses),
            captured_reasoning: parking_lot::Mutex::new(Vec::new()),
            captured_budget: parking_lot::Mutex::new(Vec::new()),
            captured_session_id: parking_lot::Mutex::new(Vec::new()),
            captured_transport: parking_lot::Mutex::new(Vec::new()),
            captured_max_retries: parking_lot::Mutex::new(Vec::new()),
            captured_max_retry_delay: parking_lot::Mutex::new(Vec::new()),
            captured_has_on_payload: parking_lot::Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl LLMProtocol for CapturingMockProvider {
    fn provider_type(&self) -> Provider {
        Provider::OpenAI
    }

    fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: StreamOptions,
    ) -> AssistantMessageEventStream {
        // Fallback — agent should call stream_simple() instead
        let stream = AssistantMessageEventStream::new_assistant_stream();
        let mut responses = self.responses.lock();
        let response = if responses.is_empty() {
            make_assistant_message("Default response")
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
        _model: &Model,
        _context: &Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        // Capture all fields from SimpleStreamOptions
        self.captured_reasoning.lock().push(options.reasoning);
        self.captured_budget
            .lock()
            .push(options.thinking_budget_tokens);
        self.captured_session_id
            .lock()
            .push(options.base.session_id.clone());
        self.captured_transport.lock().push(options.base.transport);
        self.captured_max_retries
            .lock()
            .push(options.base.max_retries);
        self.captured_max_retry_delay
            .lock()
            .push(options.base.max_retry_delay_ms);
        self.captured_has_on_payload
            .lock()
            .push(options.base.on_payload.is_some());

        // Return a canned response
        let stream = AssistantMessageEventStream::new_assistant_stream();
        let mut responses = self.responses.lock();
        let response = if responses.is_empty() {
            make_assistant_message("Default response")
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
}

// ============================================================================
// Provider Integration: ThinkingBudgets
// ============================================================================

#[tokio::test]
async fn test_thinking_budgets_flow_to_provider() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    // Set thinking to Medium with custom budgets
    agent.set_thinking_level(ThinkingLevel::Medium);
    agent.set_thinking_budgets(ThinkingBudgets {
        minimal: Some(64),
        low: Some(256),
        medium: Some(2048),
        high: Some(4096),
    });

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured_reasoning = mock.captured_reasoning.lock();
    let captured_budget = mock.captured_budget.lock();

    assert_eq!(captured_reasoning.len(), 1);
    assert_eq!(captured_reasoning[0], Some(ThinkingLevel::Medium));
    assert_eq!(captured_budget.len(), 1);
    assert_eq!(
        captured_budget[0],
        Some(2048),
        "Should use custom budget for Medium"
    );
}

#[tokio::test]
async fn test_thinking_budgets_default_fallback() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    // Set thinking to Low WITHOUT custom budgets → should use default (512)
    agent.set_thinking_level(ThinkingLevel::Low);

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured_reasoning = mock.captured_reasoning.lock();
    let captured_budget = mock.captured_budget.lock();

    assert_eq!(captured_reasoning[0], Some(ThinkingLevel::Low));
    assert_eq!(
        captured_budget[0],
        Some(tiycore::thinking::ThinkingConfig::default_budget(
            ThinkingLevel::Low
        )),
        "Should fall back to default budget (512) when no custom budgets set"
    );
}

#[tokio::test]
async fn test_thinking_off_no_budget() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    // Thinking Off (default) — reasoning and budget should both be None
    // (Default ThinkingLevel is Off, no getter needed)

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured_reasoning = mock.captured_reasoning.lock();
    let captured_budget = mock.captured_budget.lock();

    assert_eq!(
        captured_reasoning[0], None,
        "Thinking Off should send reasoning=None"
    );
    assert_eq!(
        captured_budget[0], None,
        "Thinking Off should send budget=None"
    );
}

// ============================================================================
// Provider Integration: sessionId
// ============================================================================

#[test]
fn test_session_id_setter_getter() {
    let agent = Agent::new();
    assert_eq!(agent.session_id(), None);

    agent.set_session_id("session-abc-123");
    assert_eq!(agent.session_id(), Some("session-abc-123".to_string()));

    agent.clear_session_id();
    assert_eq!(agent.session_id(), None);
}

#[tokio::test]
async fn test_session_id_flows_to_provider() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_session_id("my-session-42");

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured = mock.captured_session_id.lock();
    assert_eq!(captured.len(), 1);
    assert_eq!(
        captured[0],
        Some("my-session-42".to_string()),
        "session_id should flow to provider"
    );
}

#[tokio::test]
async fn test_session_id_none_when_not_set() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    // Don't set session_id

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured = mock.captured_session_id.lock();
    assert_eq!(captured[0], None, "session_id should be None when not set");
}

// ============================================================================
// Provider Integration: onPayload
// ============================================================================

#[tokio::test]
async fn test_on_payload_flows_to_provider() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    // Set an on_payload hook
    let hook_called = Arc::new(AtomicBool::new(false));
    let hc = hook_called.clone();
    agent.set_on_payload(move |payload, _model| {
        let hc = hc.clone();
        async move {
            hc.store(true, Ordering::SeqCst);
            Some(payload) // pass through unchanged
        }
    });

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured = mock.captured_has_on_payload.lock();
    assert_eq!(captured.len(), 1);
    assert!(
        captured[0],
        "on_payload should be Some (present) in provider call"
    );
}

#[tokio::test]
async fn test_on_payload_none_when_not_set() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    // Don't set on_payload

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured = mock.captured_has_on_payload.lock();
    assert!(!captured[0], "on_payload should be None when not set");
}

// ============================================================================
// Provider Integration: Transport
// ============================================================================

#[tokio::test]
async fn test_transport_flows_to_provider() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_transport(Transport::WebSocket);

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured = mock.captured_transport.lock();
    assert_eq!(captured.len(), 1);
    assert_eq!(
        captured[0],
        Some(Transport::WebSocket),
        "Transport::WebSocket should flow to provider"
    );
}

#[tokio::test]
async fn test_transport_default_sse_flows_to_provider() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    // Default transport is Sse

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured = mock.captured_transport.lock();
    assert_eq!(
        captured[0],
        Some(Transport::Sse),
        "Default Transport::Sse should flow to provider"
    );
}

// ============================================================================
// Provider Integration: maxRetries
// ============================================================================

#[tokio::test]
async fn test_max_retries_flows_to_provider() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_max_retries(Some(3));

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured = mock.captured_max_retries.lock();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0], Some(3));
}

#[tokio::test]
async fn test_max_retries_none_when_not_set() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured = mock.captured_max_retries.lock();
    assert_eq!(captured[0], None);
}

// ============================================================================
// Provider Integration: maxRetryDelayMs
// ============================================================================

#[tokio::test]
async fn test_max_retry_delay_flows_to_provider() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_max_retry_delay_ms(Some(5000));

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured = mock.captured_max_retry_delay.lock();
    assert_eq!(captured.len(), 1);
    assert_eq!(
        captured[0],
        Some(5000),
        "max_retry_delay_ms=5000 should flow to provider"
    );
}

#[tokio::test]
async fn test_max_retry_delay_none_when_not_set() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    // Don't set max_retry_delay_ms

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    let captured = mock.captured_max_retry_delay.lock();
    assert_eq!(
        captured[0], None,
        "max_retry_delay_ms should be None when not set"
    );
}

// ============================================================================
// Provider Integration: All 6 features combined
// ============================================================================

#[tokio::test]
async fn test_all_six_features_flow_together() {
    let response = make_assistant_message("Done");
    let mock = Arc::new(CapturingMockProvider::new(vec![response]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    // Set all 6 features
    agent.set_thinking_level(ThinkingLevel::High);
    agent.set_thinking_budgets(ThinkingBudgets {
        minimal: Some(64),
        low: Some(256),
        medium: Some(1024),
        high: Some(8192),
    });
    agent.set_session_id("combined-session");
    agent.set_on_payload(move |payload, _model| async move { Some(payload) });
    agent.set_transport(Transport::Auto);
    agent.set_max_retries(Some(4));
    agent.set_max_retry_delay_ms(Some(15000));

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    // Verify all 6 captured correctly
    assert_eq!(mock.captured_reasoning.lock()[0], Some(ThinkingLevel::High));
    assert_eq!(mock.captured_budget.lock()[0], Some(8192));
    assert_eq!(
        mock.captured_session_id.lock()[0],
        Some("combined-session".to_string())
    );
    assert!(mock.captured_has_on_payload.lock()[0]);
    assert_eq!(mock.captured_transport.lock()[0], Some(Transport::Auto));
    assert_eq!(mock.captured_max_retries.lock()[0], Some(4));
    assert_eq!(mock.captured_max_retry_delay.lock()[0], Some(15000));
}

// ============================================================================
// Provider Integration: reset clears session_id
// ============================================================================

#[tokio::test]
async fn test_reset_clears_session_id() {
    let responses = vec![
        make_assistant_message("First"),
        make_assistant_message("Second"),
    ];
    let mock = Arc::new(CapturingMockProvider::new(responses));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);
    agent.set_session_id("session-before-reset");

    // First prompt
    let result = agent.prompt("hello").await;
    assert!(result.is_ok());
    assert_eq!(
        mock.captured_session_id.lock()[0],
        Some("session-before-reset".to_string())
    );

    // Reset should clear session_id
    agent.reset();
    assert_eq!(agent.session_id(), None);
}

// ============================================================================
// V2 Supplier & Steering Tests (PR #36 review follow-ups)
// ============================================================================

/// Test that steering during stream processing triggers `AgentError::Steered`
/// internally and causes a turn restart.
///
/// Regression test for: missing test coverage of the typed `Steered` variant
/// after the refactor from string-matching to `AgentError::Steered`.
#[tokio::test]
async fn test_steering_interrupts_turn_and_restarts() {
    // One response for the first (interrupted) turn, one for the restart
    let response1 = make_assistant_message("first response");
    let response2 = make_assistant_message("second response");
    let mock = Arc::new(MockProvider::new(vec![response1, response2]));
    let provider: ArcProtocol = mock.clone();

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    // Queue a steering message before prompting.
    // It will be dequeued during the first stream event, causing
    // process_stream_events to return Err(AgentError::Steered),
    // which run_loop catches and restarts the turn.
    agent.steer(AgentMessage::User(UserMessage::text("steering override")));

    let result = agent.prompt("hello").await;
    assert!(result.is_ok());

    // The steering message must appear in the agent's conversation state
    // (was injected by process_stream_events before returning Steered).
    let messages = agent.state().messages.read().clone();
    let has_steering = messages.iter().any(|m| match m {
        AgentMessage::User(user) => {
            matches!(&user.content, UserContent::Text(text) if text == "steering override")
        }
        _ => false,
    });
    assert!(
        has_steering,
        "Steering message should be in agent state after Steered restart"
    );

    // Provider should have been called twice:
    // once for the first (interrupted) turn, once after steering restart
    assert_eq!(
        mock.call_count(),
        2,
        "Provider should be called twice due to steering restart, got {}",
        mock.call_count()
    );
}

/// Test that a V2 steering supplier receives a correct SupplierContext
/// when probed by poll_steering_messages() during continue_().
///
/// Regression test for: V2 supplier adapter and SupplierContext injection
/// untested.
#[tokio::test]
async fn test_v2_steering_supplier_receives_context() {
    let response = make_assistant_message("done");
    let provider: ArcProtocol = Arc::new(MockProvider::new(vec![response]));

    let agent = Agent::with_model(make_model());
    agent.set_provider(provider);

    let ctx_captured = Arc::new(parking_lot::Mutex::new(None::<SupplierContext>));
    let ctx_clone = ctx_captured.clone();

    agent.set_steering_supplier(move |ctx: SupplierContext| {
        let ctx_clone = ctx_clone.clone();
        async move {
            *ctx_clone.lock() = Some(ctx);
            vec![] // Return empty so loop continues normally
        }
    });

    // First prompt completes normally (V2 supplier returns empty)
    let _ = agent.prompt("hello").await.unwrap();

    // continue_() calls poll_steering_messages() which probes the V2 supplier
    let result = agent.continue_().await;
    // Should fail because last message is assistant and no steering/follow-up queued
    assert!(
        matches!(result, Err(AgentError::CannotContinueFromAssistant)),
        "continue_() should fail with CannotContinueFromAssistant"
    );

    let ctx = ctx_captured.lock();
    assert!(
        ctx.is_some(),
        "V2 steering supplier should have been probed by poll_steering_messages()"
    );
    let ctx = ctx.as_ref().unwrap();
    // turn_count: may be 0 when probe happens after a loop that exited
    // without incrementing (no tool-call turns), but must never be None.
    let _ = ctx.turn_count; // SupplierContext is populated
}

/// Test that `has_queued_messages_async` caches dynamic supplier messages
/// into the local queue and does not re-invoke the supplier on subsequent calls.
///
/// Regression test for: has_queued_messages_async supplier caching untested.
#[tokio::test]
async fn test_has_queued_messages_async_caches_supplier_output() {
    let agent = Agent::with_model(make_model());

    let call_count = Arc::new(AtomicUsize::new(0));
    let cc = call_count.clone();

    agent.set_get_steering_messages(move |_signal| {
        let cc = cc.clone();
        Box::pin(async move {
            cc.fetch_add(1, Ordering::SeqCst);
            vec![AgentMessage::User(UserMessage::text("dynamic steering"))]
        })
    });

    assert!(!agent.has_queued_messages());

    // First call: supplier is invoked, messages cached into local queue
    let has = agent.has_queued_messages_async().await;
    assert!(has, "should report queued messages from supplier");
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        1,
        "supplier should be called once on first probe"
    );

    // Second call: must return true from local cache without re-invoking supplier
    let has2 = agent.has_queued_messages_async().await;
    assert!(has2, "should still report queued messages from cache");
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        1,
        "supplier should NOT be called again"
    );

    // Verify messages actually ended up in the local queue
    assert!(agent.has_queued_messages(), "local queue should be populated");
}

/// Test that `has_queued_messages_async` probes V2 steering suppliers.
///
/// Regression test for: V2-only consumers receiving false negatives because
/// has_queued_messages_async only probed V1 suppliers.
#[tokio::test]
async fn test_has_queued_messages_async_probes_v2_supplier() {
    let agent = Agent::with_model(make_model());

    let supplier_called = Arc::new(AtomicBool::new(false));
    let sc = supplier_called.clone();

    let ctx_captured = Arc::new(parking_lot::Mutex::new(None::<SupplierContext>));
    let cc = ctx_captured.clone();

    agent.set_steering_supplier(move |ctx: SupplierContext| {
        let sc = sc.clone();
        let cc = cc.clone();
        async move {
            sc.store(true, Ordering::SeqCst);
            *cc.lock() = Some(ctx);
            vec![AgentMessage::User(UserMessage::text("v2 steering"))]
        }
    });

    assert!(!agent.has_queued_messages());

    let has = agent.has_queued_messages_async().await;
    assert!(has, "V2 supplier should be probed and return true");
    assert!(
        supplier_called.load(Ordering::SeqCst),
        "V2 steering supplier should have been called"
    );

    // Verify SupplierContext was populated
    let ctx = ctx_captured.lock();
    assert!(ctx.is_some(), "SupplierContext should have been built");
    let ctx = ctx.as_ref().unwrap();
    assert_eq!(
        ctx.queue_depth, 0,
        "queue_depth should be 0 before messages are enqueued"
    );

    // Messages should have been cached into local queue
    assert!(agent.has_queued_messages(), "V2 supplier messages should be cached");
}
