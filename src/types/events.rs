//! Stream event types for LLM responses.

use crate::types::{AssistantMessage, StopReason, ToolCall};
use serde::{Deserialize, Serialize};

/// Events emitted during streaming assistant message generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantMessageEvent {
    /// Stream started with partial message.
    Start { partial: AssistantMessage },
    /// Text block started.
    TextStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    /// Text delta received.
    TextDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    /// Text block ended.
    TextEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    /// Thinking block started.
    ThinkingStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    /// Thinking delta received.
    ThinkingDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    /// Thinking block ended.
    ThinkingEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    /// Tool call started.
    ToolCallStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    /// Tool call arguments delta.
    ToolCallDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    /// Tool call completed.
    ToolCallEnd {
        content_index: usize,
        tool_call: ToolCall,
        partial: AssistantMessage,
    },
    /// Protocol-level retry is scheduled after a transient request or pre-stream error.
    Retrying {
        /// One-based retry attempt number.
        attempt: u32,
        /// Maximum number of retry attempts configured for this request.
        max_retries: u32,
        /// Delay before the next attempt, in milliseconds.
        delay_ms: u64,
        /// Human-readable retry reason, such as an HTTP status or transport error.
        reason: String,
        /// HTTP status code when retrying due to a response status.
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<u16>,
    },
    /// Stream completed successfully.
    Done {
        reason: StopReason,
        message: AssistantMessage,
    },
    /// Stream ended with error.
    Error {
        reason: StopReason,
        error: AssistantMessage,
    },
}

impl AssistantMessageEvent {
    /// Check if this is a completion event (done or error).
    pub fn is_complete(&self) -> bool {
        matches!(
            self,
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
        )
    }

    /// Check if this is a text event.
    pub fn is_text_event(&self) -> bool {
        matches!(
            self,
            AssistantMessageEvent::TextStart { .. }
                | AssistantMessageEvent::TextDelta { .. }
                | AssistantMessageEvent::TextEnd { .. }
        )
    }

    /// Check if this is a thinking event.
    pub fn is_thinking_event(&self) -> bool {
        matches!(
            self,
            AssistantMessageEvent::ThinkingStart { .. }
                | AssistantMessageEvent::ThinkingDelta { .. }
                | AssistantMessageEvent::ThinkingEnd { .. }
        )
    }

    /// Check if this is a tool call event.
    pub fn is_tool_call_event(&self) -> bool {
        matches!(
            self,
            AssistantMessageEvent::ToolCallStart { .. }
                | AssistantMessageEvent::ToolCallDelta { .. }
                | AssistantMessageEvent::ToolCallEnd { .. }
        )
    }

    /// Get the partial message from this event.
    pub fn partial_message(&self) -> Option<&AssistantMessage> {
        match self {
            AssistantMessageEvent::Start { partial } => Some(partial),
            AssistantMessageEvent::TextStart { partial, .. } => Some(partial),
            AssistantMessageEvent::TextDelta { partial, .. } => Some(partial),
            AssistantMessageEvent::TextEnd { partial, .. } => Some(partial),
            AssistantMessageEvent::ThinkingStart { partial, .. } => Some(partial),
            AssistantMessageEvent::ThinkingDelta { partial, .. } => Some(partial),
            AssistantMessageEvent::ThinkingEnd { partial, .. } => Some(partial),
            AssistantMessageEvent::ToolCallStart { partial, .. } => Some(partial),
            AssistantMessageEvent::ToolCallDelta { partial, .. } => Some(partial),
            AssistantMessageEvent::ToolCallEnd { partial, .. } => Some(partial),
            AssistantMessageEvent::Retrying { .. } => None,
            AssistantMessageEvent::Done { message, .. } => Some(message),
            AssistantMessageEvent::Error { error, .. } => Some(error),
        }
    }

    /// Get the content index if applicable.
    pub fn content_index(&self) -> Option<usize> {
        match self {
            AssistantMessageEvent::TextStart { content_index, .. } => Some(*content_index),
            AssistantMessageEvent::TextDelta { content_index, .. } => Some(*content_index),
            AssistantMessageEvent::TextEnd { content_index, .. } => Some(*content_index),
            AssistantMessageEvent::ThinkingStart { content_index, .. } => Some(*content_index),
            AssistantMessageEvent::ThinkingDelta { content_index, .. } => Some(*content_index),
            AssistantMessageEvent::ThinkingEnd { content_index, .. } => Some(*content_index),
            AssistantMessageEvent::ToolCallStart { content_index, .. } => Some(*content_index),
            AssistantMessageEvent::ToolCallDelta { content_index, .. } => Some(*content_index),
            AssistantMessageEvent::ToolCallEnd { content_index, .. } => Some(*content_index),
            _ => None,
        }
    }

    /// Get the delta text if this is a delta event.
    pub fn delta(&self) -> Option<&str> {
        match self {
            AssistantMessageEvent::TextDelta { delta, .. } => Some(delta),
            AssistantMessageEvent::ThinkingDelta { delta, .. } => Some(delta),
            AssistantMessageEvent::ToolCallDelta { delta, .. } => Some(delta),
            _ => None,
        }
    }

    /// Get the stop reason if this is a completion event.
    pub fn stop_reason(&self) -> Option<StopReason> {
        match self {
            AssistantMessageEvent::Done { reason, .. } => Some(*reason),
            AssistantMessageEvent::Error { reason, .. } => Some(*reason),
            _ => None,
        }
    }
}
