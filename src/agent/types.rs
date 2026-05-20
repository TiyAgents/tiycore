//! Agent types and configurations.

use crate::provider::ArcProtocol;
use crate::thinking::ThinkingLevel;
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

// ============================================================================
// AgentMessage
// ============================================================================

/// Shared cancellation signal used by agent hooks and provider requests.
pub type AbortSignal = CancellationToken;

/// Agent message - can include custom message types.
///
/// The `Custom` variant allows applications to inject arbitrary domain-specific
/// messages (e.g., artifacts, notifications) into the conversation. Custom messages
/// are filtered out by the default `convert_to_llm` implementation; provide your
/// own converter to handle them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum AgentMessage {
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
    /// Application-specific custom message.
    /// Ignored by the default LLM conversion; callers should provide a custom
    /// `convert_to_llm` to handle these.
    #[serde(rename = "custom")]
    Custom {
        /// Custom message type identifier (e.g., "artifact", "notification").
        #[serde(rename = "type")]
        message_type: String,
        /// Arbitrary payload.
        data: serde_json::Value,
    },
}

// --- From impls for AgentMessage ---

impl From<UserMessage> for AgentMessage {
    fn from(msg: UserMessage) -> Self {
        AgentMessage::User(msg)
    }
}

impl From<AssistantMessage> for AgentMessage {
    fn from(msg: AssistantMessage) -> Self {
        AgentMessage::Assistant(msg)
    }
}

impl From<ToolResultMessage> for AgentMessage {
    fn from(msg: ToolResultMessage) -> Self {
        AgentMessage::ToolResult(msg)
    }
}

impl From<Message> for AgentMessage {
    fn from(msg: Message) -> Self {
        match msg {
            Message::User(m) => AgentMessage::User(m),
            Message::Assistant(m) => AgentMessage::Assistant(m),
            Message::ToolResult(m) => AgentMessage::ToolResult(m),
        }
    }
}

impl From<AgentMessage> for Option<Message> {
    fn from(msg: AgentMessage) -> Self {
        match msg {
            AgentMessage::User(m) => Some(Message::User(m)),
            AgentMessage::Assistant(m) => Some(Message::Assistant(m)),
            AgentMessage::ToolResult(m) => Some(Message::ToolResult(m)),
            AgentMessage::Custom { .. } => None,
        }
    }
}

/// Convenience: create a user text message from a `String`.
impl From<String> for AgentMessage {
    fn from(s: String) -> Self {
        AgentMessage::User(UserMessage::text(s))
    }
}

/// Convenience: create a user text message from a `&str`.
impl From<&str> for AgentMessage {
    fn from(s: &str) -> Self {
        AgentMessage::User(UserMessage::text(s))
    }
}

// ============================================================================
// AgentTool
// ============================================================================

/// Agent tool with execution capability.
#[derive(Debug, Clone)]
pub struct AgentTool {
    /// Tool name.
    pub name: String,
    /// Human-readable label for UI.
    pub label: String,
    /// Tool description.
    pub description: String,
    /// Parameters schema.
    pub parameters: serde_json::Value,
}

impl AgentTool {
    /// Create a new agent tool.
    pub fn new(
        name: impl Into<String>,
        label: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            label: label.into(),
            description: description.into(),
            parameters,
        }
    }

    /// Convert to a basic Tool.
    pub fn as_tool(&self) -> Tool {
        Tool::new(&self.name, &self.description, self.parameters.clone())
    }
}

impl From<Tool> for AgentTool {
    fn from(tool: Tool) -> Self {
        let name = tool.name.clone();
        Self {
            name,
            label: tool.name,
            description: tool.description,
            parameters: tool.parameters,
        }
    }
}

// ============================================================================
// AgentContext
// ============================================================================

/// Agent context.
#[derive(Debug, Clone, Default)]
pub struct AgentContext {
    /// System prompt.
    pub system_prompt: String,
    /// Messages.
    pub messages: Vec<AgentMessage>,
    /// Tools.
    pub tools: Option<Vec<AgentTool>>,
}

// ============================================================================
// AgentEvent
// ============================================================================

/// Agent event types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// Agent started.
    AgentStart,
    /// Agent finished.
    AgentEnd { messages: Vec<AgentMessage> },
    /// Turn started.
    TurnStart {
        /// Zero-based turn index within the current run.
        turn_index: usize,
    },
    /// Turn finished.
    TurnEnd {
        /// Zero-based turn index within the current run.
        turn_index: usize,
        message: AgentMessage,
        tool_results: Vec<ToolResultMessage>,
    },
    /// Message started.
    MessageStart {
        /// Zero-based turn index within the current run.
        turn_index: usize,
        message: AgentMessage,
    },
    /// Message updated (streaming).
    MessageUpdate {
        /// Zero-based turn index within the current run.
        turn_index: usize,
        message: AgentMessage,
        assistant_event: Box<AssistantMessageEvent>,
    },
    /// Message finished.
    MessageEnd {
        /// Zero-based turn index within the current run.
        turn_index: usize,
        /// Provider-assigned response identifier, extracted from `AssistantMessage.response_id`.
        response_id: Option<String>,
        message: AgentMessage,
    },
    /// A previously streamed message was discarded and removed from model state.
    MessageDiscarded {
        /// Zero-based turn index within the current run.
        turn_index: usize,
        message: AgentMessage,
        reason: String,
    },
    /// Tool execution started.
    ToolExecutionStart {
        /// Zero-based turn index within the current run.
        turn_index: usize,
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
    },
    /// Tool execution progress (streaming partial results from tools).
    ToolExecutionUpdate {
        /// Zero-based turn index within the current run.
        turn_index: usize,
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
        partial_result: serde_json::Value,
    },
    /// Tool execution finished.
    ToolExecutionEnd {
        /// Zero-based turn index within the current run.
        turn_index: usize,
        tool_call_id: String,
        tool_name: String,
        result: serde_json::Value,
        is_error: bool,
    },
    /// The current assistant turn is being retried from the latest stable context.
    TurnRetrying {
        attempt: usize,
        max_attempts: usize,
        delay_ms: u64,
        reason: String,
    },
}

// ============================================================================
// Tool Execution Mode
// ============================================================================

/// Tool execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ToolExecutionMode {
    /// Execute tools sequentially.
    Sequential,
    /// Execute tools in parallel.
    #[default]
    Parallel,
}

// ============================================================================
// Queue Mode
// ============================================================================

/// Queue delivery mode for steering and follow-up messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum QueueMode {
    /// Deliver all queued messages in one batch.
    #[default]
    All,
    /// Deliver only the first message per turn.
    OneAtATime,
}

// ============================================================================
// Queue Stats
// ============================================================================

/// Non-consuming snapshot of queue state for observability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueStats {
    /// Number of messages in the steering queue local buffer.
    pub steering_depth: usize,
    /// Number of messages in the follow-up queue local buffer.
    pub follow_up_depth: usize,
    /// Whether steering deferral is currently active.
    pub is_deferring_steering: bool,
}

// ============================================================================
// Queue Events (Observability)
// ============================================================================

/// Queue identity for event reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueueKind {
    /// Steering message queue.
    Steering,
    /// Follow-up message queue.
    FollowUp,
}

/// Lifecycle events emitted by the queue system for observability.
///
/// These are fire-and-forget notifications; handlers must not block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueEvent {
    /// Messages were added to a queue.
    Enqueued {
        /// Which queue received the messages.
        kind: QueueKind,
        /// How many messages were added.
        count: usize,
        /// Total queue depth after this operation.
        queue_depth: usize,
    },
    /// Messages were consumed from a queue (drained into a turn).
    Consumed {
        /// Which queue was drained.
        kind: QueueKind,
        /// How many messages were consumed.
        count: usize,
        /// Remaining messages in the queue after consumption.
        remaining: usize,
    },
    /// Queue was cleared (all pending messages dropped).
    Cleared {
        /// Which queue was cleared.
        kind: QueueKind,
        /// How many messages were dropped.
        count_dropped: usize,
    },
    /// Messages were removed from a queue before consumption.
    Removed {
        /// Which queue had messages removed.
        kind: QueueKind,
        /// How many messages were removed.
        count: usize,
        /// Remaining messages in the queue after removal.
        remaining: usize,
    },
}

/// Callback type for queue lifecycle events.
///
/// Handlers should be non-blocking and fast; they run synchronously on the
/// agent's task. Use channels or other async primitives internally if you
/// need to defer work.
pub type OnQueueEventFn = Arc<dyn Fn(QueueEvent) + Send + Sync>;

// ============================================================================
// ThinkingBudgets
// ============================================================================

/// Custom token budgets for each thinking level.
///
/// Allows overriding the default budget_tokens for each ThinkingLevel.
/// Only relevant for providers that use token-based thinking budgets
/// (e.g., Anthropic). Omitted levels use the provider default.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinkingBudgets {
    /// Budget for Minimal level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimal: Option<u32>,
    /// Budget for Low level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub low: Option<u32>,
    /// Budget for Medium level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub medium: Option<u32>,
    /// Budget for High level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub high: Option<u32>,
}

impl ThinkingBudgets {
    /// Get the custom budget for a thinking level, if set.
    pub fn budget_for(&self, level: ThinkingLevel) -> Option<u32> {
        match level {
            ThinkingLevel::Minimal => self.minimal,
            ThinkingLevel::Low => self.low,
            ThinkingLevel::Medium => self.medium,
            ThinkingLevel::High => self.high,
            _ => None,
        }
    }
}

// ============================================================================
// Transport (re-exported from crate::types)
// ============================================================================

pub use crate::types::Transport;

// ============================================================================
// Tool Call Hooks
// ============================================================================

/// Context provided to `before_tool_call` hooks.
#[derive(Debug, Clone)]
pub struct BeforeToolCallContext {
    /// The assistant message that contained the tool call.
    pub assistant_message: AssistantMessage,
    /// The tool call being executed.
    pub tool_call: ToolCall,
    /// Parsed & validated arguments.
    pub args: serde_json::Value,
    /// The current conversation context.
    pub context: Context,
    /// Cancellation signal for the current agent run.
    pub abort_signal: AbortSignal,
}

/// Result returned by `before_tool_call` hooks.
#[derive(Debug, Clone, Default)]
pub struct BeforeToolCallResult {
    /// Set to `true` to block execution of this tool call.
    pub block: bool,
    /// Optional reason shown to the LLM when the call is blocked.
    pub reason: Option<String>,
}

impl BeforeToolCallResult {
    /// Create a result that allows execution.
    pub fn allow() -> Self {
        Self {
            block: false,
            reason: None,
        }
    }

    /// Create a result that blocks execution with an optional reason.
    pub fn blocked(reason: impl Into<String>) -> Self {
        Self {
            block: true,
            reason: Some(reason.into()),
        }
    }
}

/// Context provided to `after_tool_call` hooks.
#[derive(Debug, Clone)]
pub struct AfterToolCallContext {
    /// The assistant message that contained the tool call.
    pub assistant_message: AssistantMessage,
    /// The tool call that was executed.
    pub tool_call: ToolCall,
    /// Arguments that were passed to the tool.
    pub args: serde_json::Value,
    /// The result from tool execution.
    pub result: AgentToolResult,
    /// Whether the tool execution reported an error.
    pub is_error: bool,
    /// The current conversation context.
    pub context: Context,
    /// Cancellation signal for the current agent run.
    pub abort_signal: AbortSignal,
}

/// Result returned by `after_tool_call` hooks.
///
/// Omitted (None) fields keep original values — no deep merge.
#[derive(Debug, Clone, Default)]
pub struct AfterToolCallResult {
    /// Override the content blocks of the tool result.
    pub content: Option<Vec<ContentBlock>>,
    /// Override the details payload.
    pub details: Option<serde_json::Value>,
    /// Override the is_error flag.
    pub is_error: Option<bool>,
}

// ============================================================================
// Callback type aliases
// ============================================================================

/// Async callback for `before_tool_call`.
///
/// Receives the hook context and returns an optional result.
/// Returning `None` is equivalent to allowing the call.
pub type BeforeToolCallFn = Arc<
    dyn Fn(
            BeforeToolCallContext,
        )
            -> Pin<Box<dyn std::future::Future<Output = Option<BeforeToolCallResult>> + Send>>
        + Send
        + Sync,
>;

/// Async callback for `after_tool_call`.
///
/// Receives the hook context and returns an optional result.
/// Returning `None` keeps the original tool result unchanged.
pub type AfterToolCallFn = Arc<
    dyn Fn(
            AfterToolCallContext,
        )
            -> Pin<Box<dyn std::future::Future<Output = Option<AfterToolCallResult>> + Send>>
        + Send
        + Sync,
>;

/// Dynamic API key resolver.
///
/// Called before each LLM request. Receives the provider name string.
/// Return `Some(key)` to override the static API key, or `None` to fall back
/// to the configured key.
pub type GetApiKeyFn = Arc<
    dyn Fn(&str, AbortSignal) -> Pin<Box<dyn std::future::Future<Output = Option<String>> + Send>>
        + Send
        + Sync,
>;

/// Pre-serialization message hook: `Message[] + Model -> Message[]`.
///
/// Called after `convert_to_llm` and before the messages are assembled into
/// a `Context`. Receives the target `Model` so callers can apply
/// provider-specific structural normalisation at the typed-message level.
pub type OnMessagesFn = Arc<
    dyn Fn(Vec<Message>, Model) -> Pin<Box<dyn std::future::Future<Output = Vec<Message>> + Send>>
        + Send
        + Sync,
>;

/// Payload inspection / replacement hook (re-exported from crate::types).
pub use crate::types::OnPayloadFn;

/// Message conversion function: `AgentMessage[]` -> `Message[]`.
///
/// Called before each LLM request to convert agent-level messages (which may
/// include custom types) into LLM-compatible messages.
pub type ConvertToLlmFn = Arc<
    dyn Fn(Vec<AgentMessage>) -> Pin<Box<dyn std::future::Future<Output = Vec<Message>> + Send>>
        + Send
        + Sync,
>;

/// Context transformation function applied BEFORE `convert_to_llm`.
///
/// Use this for context window management (pruning old messages),
/// injecting context from external sources, etc.
pub type TransformContextFn = Arc<
    dyn Fn(
            Vec<AgentMessage>,
            AbortSignal,
        ) -> Pin<Box<dyn std::future::Future<Output = Vec<AgentMessage>> + Send>>
        + Send
        + Sync,
>;

/// Tool execution update callback.
///
/// Passed to the `ToolExecutor` so tools can push streaming partial results.
pub type ToolUpdateCallback = Arc<dyn Fn(serde_json::Value) + Send + Sync>;

/// Custom stream function type.
///
/// Allows replacing the default provider streaming with a custom implementation
/// (e.g., proxy streaming through a server).
pub type StreamFn = Arc<
    dyn Fn(
            &Model,
            &Context,
            SimpleStreamOptions,
            AbortSignal,
        ) -> Pin<
            Box<
                dyn std::future::Future<Output = crate::stream::AssistantMessageEventStream> + Send,
            >,
        > + Send
        + Sync,
>;

/// Dynamic queued-message supplier for steering/follow-up injection.
pub type GetQueuedMessagesFn = Arc<
    dyn Fn(AbortSignal) -> Pin<Box<dyn std::future::Future<Output = Vec<AgentMessage>> + Send>>
        + Send
        + Sync,
>;

/// Context provided to V2 dynamic message suppliers.
///
/// Enriched with agent state information so suppliers can make informed
/// decisions about what messages to inject.
#[derive(Debug, Clone)]
pub struct SupplierContext {
    /// Cancellation signal for the current agent run.
    pub abort: AbortSignal,
    /// Current turn count within this run.
    pub turn_count: usize,
    /// Current queue depth (local buffer, before this supplier call).
    pub queue_depth: usize,
    /// Whether the agent is currently streaming a response.
    pub is_streaming: bool,
}

/// V2 dynamic message supplier with enriched context.
///
/// Preferred over [`GetQueuedMessagesFn`] when the supplier needs to
/// make decisions based on agent state.
pub type GetQueuedMessagesFnV2 = Arc<
    dyn Fn(SupplierContext) -> Pin<Box<dyn std::future::Future<Output = Vec<AgentMessage>> + Send>>
        + Send
        + Sync,
>;

// ============================================================================
// ToolExecutor
// ============================================================================

/// Tool executor function type.
///
/// Receives `(tool_name, tool_call_id, arguments, update_callback)`.
/// The optional `update_callback` can be called during execution to push
/// streaming partial results (emitted as `ToolExecutionUpdate` events).
pub type ToolExecutor = Arc<
    dyn Fn(
            &str,
            &str,
            &serde_json::Value,
            Option<ToolUpdateCallback>,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = AgentToolResult> + Send>>
        + Send
        + Sync,
>;

// ============================================================================
// AgentHooks — Aggregated hook container
// ============================================================================

/// Aggregated hook container for Agent callbacks.
///
/// Groups all optional callback fields that customize Agent behaviour
/// into a single structure, reducing the number of top-level fields on
/// `Agent` (from 18 to 11).
#[derive(Clone, Default)]
pub struct AgentHooks {
    /// Tool executor callback.
    pub tool_executor: Option<ToolExecutor>,
    /// Before tool call hook.
    pub before_tool_call: Option<BeforeToolCallFn>,
    /// After tool call hook.
    pub after_tool_call: Option<AfterToolCallFn>,
    /// Custom message-to-LLM conversion function.
    pub convert_to_llm: Option<ConvertToLlmFn>,
    /// Context transformation applied before convert_to_llm.
    pub transform_context: Option<TransformContextFn>,
    /// Dynamic API key resolver.
    pub get_api_key: Option<GetApiKeyFn>,
    /// Payload inspection / replacement hook.
    pub on_payload: Option<OnPayloadFn>,
    /// Pre-serialization message hook (operates on typed `Message` + `Model`).
    pub on_messages: Option<OnMessagesFn>,
    /// Custom stream function (for proxy backends, etc.).
    pub stream_fn: Option<StreamFn>,
    /// Dynamic steering-message supplier.
    pub get_steering_messages: Option<GetQueuedMessagesFn>,
    /// Dynamic follow-up-message supplier.
    pub get_follow_up_messages: Option<GetQueuedMessagesFn>,
    /// V2 steering supplier with enriched context (takes precedence if set).
    pub get_steering_messages_v2: Option<GetQueuedMessagesFnV2>,
    /// V2 follow-up supplier with enriched context (takes precedence if set).
    pub get_follow_up_messages_v2: Option<GetQueuedMessagesFnV2>,
    /// Queue lifecycle event observer.
    pub on_queue_event: Option<OnQueueEventFn>,
}

/// Additional options for the standalone agent loop APIs.
#[derive(Clone, Default)]
pub struct AgentLoopOptions {
    /// Aggregated hooks applied to the loop.
    pub hooks: AgentHooks,
    /// Explicit provider override. Falls back to the registry when omitted.
    pub provider: Option<ArcProtocol>,
    /// Static API key forwarded to provider requests.
    pub api_key: Option<String>,
    /// Optional provider session identifier.
    pub session_id: Option<String>,
    /// Override for the default max-turn limit.
    pub max_turns: Option<usize>,
}

// ============================================================================
// AgentConfig
// ============================================================================

/// Agent configuration.
#[derive(Clone)]
pub struct AgentConfig {
    /// Model to use.
    pub model: Model,
    /// Thinking level.
    pub thinking_level: ThinkingLevel,
    /// Tool execution mode.
    pub tool_execution: ToolExecutionMode,
    /// Security and resource limits.
    pub security: crate::types::SecurityConfig,
    /// Steering queue delivery mode.
    pub steering_mode: QueueMode,
    /// Follow-up queue delivery mode.
    pub follow_up_mode: QueueMode,
    /// Custom thinking budgets per level.
    pub thinking_budgets: Option<ThinkingBudgets>,
    /// Preferred transport.
    pub transport: Transport,
    /// Maximum number of retries for transient HTTP or pre-stream transport failures.
    /// `None` = use provider default. Set to `Some(0)` to disable retries.
    pub max_retries: Option<u32>,
    /// Maximum retry delay in milliseconds. `None` = use default (60_000ms).
    /// Set to `Some(0)` to disable the cap entirely.
    pub max_retry_delay_ms: Option<u64>,
    /// Custom HTTP headers to include in every LLM API request.
    pub custom_headers: Option<std::collections::HashMap<String, String>>,
}

impl std::fmt::Debug for AgentConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentConfig")
            .field("model", &self.model.id)
            .field("thinking_level", &self.thinking_level)
            .field("tool_execution", &self.tool_execution)
            .field("security", &"SecurityConfig{...}")
            .field("steering_mode", &self.steering_mode)
            .field("follow_up_mode", &self.follow_up_mode)
            .field("thinking_budgets", &self.thinking_budgets)
            .field("transport", &self.transport)
            .field("max_retries", &self.max_retries)
            .field("max_retry_delay_ms", &self.max_retry_delay_ms)
            .field(
                "custom_headers",
                &self.custom_headers.as_ref().map(|h| h.len()),
            )
            .finish()
    }
}

impl AgentConfig {
    /// Create a new agent config with a model.
    pub fn new(model: Model) -> Self {
        Self {
            model,
            thinking_level: ThinkingLevel::default(),
            tool_execution: ToolExecutionMode::default(),
            security: crate::types::SecurityConfig::default(),
            steering_mode: QueueMode::default(),
            follow_up_mode: QueueMode::default(),
            thinking_budgets: None,
            transport: Transport::default(),
            max_retries: None,
            max_retry_delay_ms: None,
            custom_headers: None,
        }
    }
}

// ============================================================================
// AgentToolResult
// ============================================================================

/// Tool result from execution.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentToolResult<T = serde_json::Value> {
    /// Content blocks.
    pub content: Vec<ContentBlock>,
    /// Additional details.
    pub details: Option<T>,
}

impl AgentToolResult {
    /// Create a text result.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Text(TextContent::new(text))],
            details: None,
        }
    }

    /// Create an error result.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Text(TextContent::new(message))],
            details: None,
        }
    }
}
