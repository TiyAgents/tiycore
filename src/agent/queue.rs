//! Unified message queue for steering and follow-up.
//!
//! Encapsulates local FIFO buffer + optional dynamic supplier into a single
//! consumption path, eliminating the inconsistency between `dequeue_*` (local
//! only) and `poll_*` (local + supplier) methods.

use crate::agent::{AbortSignal, AgentMessage, GetQueuedMessagesFn, QueueKind, QueueMode};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// Queued message identity
// ============================================================================

/// Stable identifier for a message while it remains in a local agent queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QueuedMessageId(u64);

impl QueuedMessageId {
    /// Return the raw numeric identifier.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// Handle returned when a message is enqueued.
///
/// It can be used to cancel the message before it is drained from the queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueuedMessageHandle {
    /// Which queue owns this handle.
    pub kind: QueueKind,
    /// Queue-local message identifier.
    pub id: QueuedMessageId,
}

#[derive(Debug, Clone)]
struct QueuedMessageItem {
    id: QueuedMessageId,
    message: AgentMessage,
}

// ============================================================================
// DrainStrategy trait
// ============================================================================

/// Defines how messages are selected from the queue buffer on consumption.
///
/// Implementors decide which messages to emit and which to retain.
/// The buffer is passed mutably — the strategy should drain the messages
/// it wants to emit and leave the rest.
pub trait DrainStrategy: Send + Sync {
    /// Select messages to emit from the buffer.
    ///
    /// The implementation should remove selected messages from `buffer`
    /// and return them. Messages left in `buffer` will be available
    /// for future drain calls.
    fn select(&self, buffer: &mut VecDeque<AgentMessage>) -> Vec<AgentMessage>;

    /// Display name for diagnostics.
    fn name(&self) -> &'static str;
}

/// Built-in strategy: emit all queued messages in one batch.
#[allow(dead_code)]
pub struct DrainAll;

impl DrainStrategy for DrainAll {
    fn select(&self, buffer: &mut VecDeque<AgentMessage>) -> Vec<AgentMessage> {
        buffer.drain(..).collect()
    }

    fn name(&self) -> &'static str {
        "all"
    }
}

/// Built-in strategy: emit only the first (oldest) message.
#[allow(dead_code)]
pub struct DrainOne;

impl DrainStrategy for DrainOne {
    fn select(&self, buffer: &mut VecDeque<AgentMessage>) -> Vec<AgentMessage> {
        buffer.pop_front().into_iter().collect()
    }

    fn name(&self) -> &'static str {
        "one_at_a_time"
    }
}

/// Built-in strategy: emit up to N messages per drain.
#[allow(dead_code)]
pub struct DrainBatch {
    /// Maximum messages to emit per drain call.
    pub max: usize,
}

impl DrainStrategy for DrainBatch {
    fn select(&self, buffer: &mut VecDeque<AgentMessage>) -> Vec<AgentMessage> {
        let take = self.max.min(buffer.len());
        buffer.drain(..take).collect()
    }

    fn name(&self) -> &'static str {
        "batch"
    }
}

impl QueueMode {
    /// Convert to the corresponding built-in DrainStrategy.
    #[allow(dead_code)]
    pub(crate) fn to_strategy(self) -> Box<dyn DrainStrategy> {
        match self {
            QueueMode::All => Box::new(DrainAll),
            QueueMode::OneAtATime => Box::new(DrainOne),
        }
    }
}

// ============================================================================
// BackpressureConfig
// ============================================================================

/// Backpressure configuration for message queues.
///
/// Controls behavior when the queue depth exceeds limits.
/// Default is `Unlimited` (no restriction), preserving backward compatibility.
#[derive(Debug, Clone)]
pub struct BackpressureConfig {
    /// Maximum queue depth. 0 = unlimited (default).
    pub max_depth: usize,
    /// What to do when the limit is exceeded.
    pub overflow: OverflowBehavior,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            max_depth: 0,
            overflow: OverflowBehavior::Unlimited,
        }
    }
}

/// Overflow behavior when backpressure limit is reached.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum OverflowBehavior {
    /// No restriction (default, backward-compatible).
    #[default]
    Unlimited,
    /// Drop oldest messages to make room for new ones.
    DropOldest,
    /// Reject new messages (caller receives error).
    Reject,
}

/// Error returned when a message is rejected due to backpressure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueFullError {
    /// Current queue depth at the time of rejection.
    pub current_depth: usize,
    /// Configured maximum depth.
    pub max_depth: usize,
}

impl std::fmt::Display for QueueFullError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "queue full: depth {} >= max {}",
            self.current_depth, self.max_depth
        )
    }
}

impl std::error::Error for QueueFullError {}

// ============================================================================
// MessageQueue
// ============================================================================

/// Unified message queue handle.
///
/// Each `MessageQueue` manages a local FIFO buffer. Consumption methods accept
/// an optional dynamic supplier and merge results according to [`QueueMode`].
///
/// Design invariants:
/// - `push()` is synchronous and lock-free beyond the Mutex.
/// - `drain_local()` is synchronous (for hot-path stream loop checks).
/// - `drain()` is async (calls supplier) and should be used at turn boundaries.
/// - The Mutex critical section is nanosecond-scale (VecDeque ops only).
pub(crate) struct MessageQueue {
    buffer: Mutex<VecDeque<QueuedMessageItem>>,
    kind: QueueKind,
    backpressure: Mutex<BackpressureConfig>,
    next_id: AtomicU64,
}

impl MessageQueue {
    /// Create a new empty queue.
    pub fn new(kind: QueueKind) -> Self {
        Self {
            buffer: Mutex::new(VecDeque::new()),
            kind,
            backpressure: Mutex::new(BackpressureConfig::default()),
            next_id: AtomicU64::new(1),
        }
    }

    fn next_message_id(&self) -> QueuedMessageId {
        QueuedMessageId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    fn make_item(&self, message: AgentMessage) -> QueuedMessageItem {
        QueuedMessageItem {
            id: self.next_message_id(),
            message,
        }
    }

    /// Returns the queue kind (Steering or FollowUp).
    #[allow(dead_code)]
    pub fn kind(&self) -> QueueKind {
        self.kind
    }

    /// Push a single message into the queue and return its queue-local id.
    pub fn push(&self, message: AgentMessage) -> QueuedMessageId {
        let item = self.make_item(message);
        let id = item.id;
        self.buffer.lock().push_back(item);
        id
    }

    /// Push multiple messages into the queue.
    ///
    /// This bulk path intentionally bypasses backpressure. It is used to cache
    /// messages that were already produced by dynamic suppliers; callers that
    /// need backpressure enforcement should use [`MessageQueue::try_push`].
    #[allow(dead_code)]
    pub fn push_many(&self, messages: impl IntoIterator<Item = AgentMessage>) {
        let mut buf = self.buffer.lock();
        buf.extend(messages.into_iter().map(|message| self.make_item(message)));
    }

    /// Try to push a message, respecting backpressure configuration.
    ///
    /// Returns the queue-local id if the message was accepted, or
    /// `Err(QueueFullError)` if the queue is full and overflow behavior is
    /// `Reject`.
    ///
    /// With `DropOldest`, oldest messages are evicted to make room.
    /// With `Unlimited`, always succeeds (equivalent to `push()`).
    pub fn try_push(&self, message: AgentMessage) -> Result<QueuedMessageId, QueueFullError> {
        let bp = self.backpressure.lock().clone();
        let item = self.make_item(message);
        let id = item.id;
        if bp.max_depth == 0 || bp.overflow == OverflowBehavior::Unlimited {
            // No limit
            self.buffer.lock().push_back(item);
            return Ok(id);
        }

        let mut buf = self.buffer.lock();
        if buf.len() >= bp.max_depth {
            match bp.overflow {
                OverflowBehavior::Reject => {
                    return Err(QueueFullError {
                        current_depth: buf.len(),
                        max_depth: bp.max_depth,
                    });
                }
                OverflowBehavior::DropOldest => {
                    // Evict oldest to make room
                    buf.pop_front();
                }
                OverflowBehavior::Unlimited => unreachable!(),
            }
        }
        buf.push_back(item);
        Ok(id)
    }

    /// Set the backpressure configuration.
    pub fn set_backpressure(&self, config: BackpressureConfig) {
        *self.backpressure.lock() = config;
    }

    /// Get the current backpressure configuration.
    #[allow(dead_code)]
    pub fn backpressure(&self) -> BackpressureConfig {
        self.backpressure.lock().clone()
    }

    /// Drain using a custom strategy (instead of QueueMode).
    ///
    /// This allows pluggable consumption logic beyond All/OneAtATime. Custom
    /// strategies operate on bare messages, so retained messages are requeued
    /// with fresh internal ids.
    #[allow(dead_code)]
    pub fn drain_with_strategy(&self, strategy: &dyn DrainStrategy) -> Vec<AgentMessage> {
        let mut buf = self.buffer.lock();
        let mut messages: VecDeque<AgentMessage> = buf.drain(..).map(|item| item.message).collect();
        let selected = strategy.select(&mut messages);
        buf.extend(messages.into_iter().map(|message| self.make_item(message)));
        selected
    }

    /// Synchronous drain from local buffer only (no supplier call).
    ///
    /// This is the fast path used in the stream event loop and sequential
    /// tool execution, where calling an async supplier would block stream
    /// processing.
    pub fn drain_local(&self, mode: QueueMode) -> Vec<AgentMessage> {
        let mut buf = self.buffer.lock();
        match mode {
            QueueMode::All => buf.drain(..).map(|item| item.message).collect(),
            QueueMode::OneAtATime => {
                if let Some(first) = buf.pop_front() {
                    vec![first.message]
                } else {
                    Vec::new()
                }
            }
        }
    }

    /// Async drain: local buffer + dynamic supplier, merged per mode.
    ///
    /// This is the full consumption path used at turn boundaries, in
    /// `continue_()`, and in deferred steering processing.
    ///
    /// Merge semantics:
    /// - `All`: local messages first, then all supplier messages appended.
    /// - `OneAtATime`: if local has a message, return it (supplier not called).
    ///   If local is empty, call supplier and take at most 1 message.
    pub async fn drain(
        &self,
        mode: QueueMode,
        supplier: &Option<GetQueuedMessagesFn>,
        abort: AbortSignal,
    ) -> Vec<AgentMessage> {
        let local = self.drain_local(mode);

        let dynamic = match supplier {
            Some(s) if mode == QueueMode::All || local.is_empty() => s(abort).await,
            _ => Vec::new(),
        };

        match mode {
            QueueMode::All => {
                let mut merged = local;
                merged.extend(dynamic);
                merged
            }
            QueueMode::OneAtATime => {
                if !local.is_empty() {
                    // Local already yielded one message; supplier wasn't called
                    // (or mode == All which is handled above)
                    local
                } else {
                    // Take at most 1 from supplier
                    dynamic.into_iter().take(1).collect()
                }
            }
        }
    }

    /// Returns true if the local buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.lock().is_empty()
    }

    /// Returns the number of messages in the local buffer.
    pub fn len(&self) -> usize {
        self.buffer.lock().len()
    }

    /// Remove a queued message by id.
    ///
    /// Returns `Some(message)` only if the message is still in the local buffer.
    pub fn remove(&self, id: QueuedMessageId) -> Option<AgentMessage> {
        let mut buf = self.buffer.lock();
        let index = buf.iter().position(|item| item.id == id)?;
        buf.remove(index).map(|item| item.message)
    }

    /// Clear all messages from the local buffer.
    pub fn clear(&self) {
        self.buffer.lock().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentMessage;
    use crate::types::UserContent;
    use std::sync::Arc;

    fn make_msg(text: &str) -> AgentMessage {
        AgentMessage::from(text)
    }

    #[test]
    fn test_push_and_drain_all() {
        let q = MessageQueue::new(QueueKind::Steering);
        q.push(make_msg("a"));
        q.push(make_msg("b"));
        q.push(make_msg("c"));

        let drained = q.drain_local(QueueMode::All);
        assert_eq!(drained.len(), 3);
        assert!(q.is_empty());
    }

    #[test]
    fn test_push_and_drain_one_at_a_time() {
        let q = MessageQueue::new(QueueKind::FollowUp);
        q.push(make_msg("a"));
        q.push(make_msg("b"));

        let drained = q.drain_local(QueueMode::OneAtATime);
        assert_eq!(drained.len(), 1);
        assert_eq!(q.len(), 1);

        let drained2 = q.drain_local(QueueMode::OneAtATime);
        assert_eq!(drained2.len(), 1);
        assert!(q.is_empty());
    }

    #[test]
    fn test_drain_local_empty() {
        let q = MessageQueue::new(QueueKind::Steering);
        let drained = q.drain_local(QueueMode::All);
        assert!(drained.is_empty());

        let drained2 = q.drain_local(QueueMode::OneAtATime);
        assert!(drained2.is_empty());
    }

    #[test]
    fn test_clear() {
        let q = MessageQueue::new(QueueKind::Steering);
        q.push(make_msg("a"));
        q.push(make_msg("b"));
        assert_eq!(q.len(), 2);

        q.clear();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn test_push_many() {
        let q = MessageQueue::new(QueueKind::FollowUp);
        q.push_many(vec![make_msg("a"), make_msg("b"), make_msg("c")]);
        assert_eq!(q.len(), 3);
    }

    #[test]
    fn test_push_many_bypasses_backpressure() {
        let q = MessageQueue::new(QueueKind::FollowUp);
        q.set_backpressure(BackpressureConfig {
            max_depth: 1,
            overflow: OverflowBehavior::Reject,
        });

        q.push_many(vec![make_msg("a"), make_msg("b"), make_msg("c")]);
        assert_eq!(q.len(), 3);

        let err = q.try_push(make_msg("d")).unwrap_err();
        assert_eq!(err.current_depth, 3);
        assert_eq!(err.max_depth, 1);
    }

    #[test]
    fn test_remove_by_id_before_drain() {
        let q = MessageQueue::new(QueueKind::Steering);
        let id_a = q.push(make_msg("a"));
        let id_b = q.push(make_msg("b"));
        let id_c = q.push(make_msg("c"));

        let removed = q.remove(id_b);
        assert_eq!(removed, Some(make_msg("b")));
        assert_eq!(q.len(), 2);

        let drained = q.drain_local(QueueMode::All);
        assert_eq!(drained, vec![make_msg("a"), make_msg("c")]);
        assert!(q.remove(id_a).is_none());
        assert!(q.remove(id_c).is_none());
    }

    #[test]
    fn test_remove_unknown_id_returns_none() {
        let q = MessageQueue::new(QueueKind::FollowUp);
        let id = q.push(make_msg("a"));
        assert_eq!(q.remove(QueuedMessageId(id.as_u64() + 1)), None);
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_remove_after_drain_returns_none() {
        let q = MessageQueue::new(QueueKind::Steering);
        let id = q.push(make_msg("a"));
        assert_eq!(q.drain_local(QueueMode::All), vec![make_msg("a")]);
        assert!(q.remove(id).is_none());
    }

    #[test]
    fn test_reject_then_remove_then_accept() {
        let q = MessageQueue::new(QueueKind::FollowUp);
        q.set_backpressure(BackpressureConfig {
            max_depth: 1,
            overflow: OverflowBehavior::Reject,
        });

        let id = q.try_push(make_msg("a")).unwrap();
        assert!(q.try_push(make_msg("b")).is_err());
        assert_eq!(q.remove(id), Some(make_msg("a")));
        assert!(q.try_push(make_msg("b")).is_ok());
        assert_eq!(q.drain_local(QueueMode::All), vec![make_msg("b")]);
    }

    #[test]
    fn test_drop_oldest_makes_old_handle_uncancellable() {
        let q = MessageQueue::new(QueueKind::Steering);
        q.set_backpressure(BackpressureConfig {
            max_depth: 1,
            overflow: OverflowBehavior::DropOldest,
        });

        let old_id = q.try_push(make_msg("old")).unwrap();
        let new_id = q.try_push(make_msg("new")).unwrap();
        assert!(q.remove(old_id).is_none());
        assert_eq!(q.remove(new_id), Some(make_msg("new")));
        assert!(q.is_empty());
    }

    #[tokio::test]
    async fn test_drain_with_supplier_all_mode() {
        let q = MessageQueue::new(QueueKind::Steering);
        q.push(make_msg("local"));

        let supplier: GetQueuedMessagesFn =
            Arc::new(|_signal| Box::pin(async { vec![AgentMessage::from("dynamic")] }));

        let abort = AbortSignal::new();
        let result = q.drain(QueueMode::All, &Some(supplier), abort).await;
        assert_eq!(result.len(), 2); // local + dynamic
    }

    #[tokio::test]
    async fn test_drain_one_at_a_time_local_first() {
        let q = MessageQueue::new(QueueKind::Steering);
        q.push(make_msg("local"));

        let supplier: GetQueuedMessagesFn =
            Arc::new(|_signal| Box::pin(async { vec![AgentMessage::from("dynamic")] }));

        let abort = AbortSignal::new();
        let result = q.drain(QueueMode::OneAtATime, &Some(supplier), abort).await;
        // Local has message, so supplier is NOT called; only 1 local msg returned
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn test_drain_one_at_a_time_falls_to_supplier() {
        let q = MessageQueue::new(QueueKind::Steering);
        // Local is empty

        let supplier: GetQueuedMessagesFn = Arc::new(|_signal| {
            Box::pin(async { vec![AgentMessage::from("d1"), AgentMessage::from("d2")] })
        });

        let abort = AbortSignal::new();
        let result = q.drain(QueueMode::OneAtATime, &Some(supplier), abort).await;
        // Supplier returns 2, but OneAtATime takes only 1
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn test_drain_all_with_empty_local_uses_supplier_only() {
        let q = MessageQueue::new(QueueKind::Steering);

        let supplier: GetQueuedMessagesFn =
            Arc::new(|_signal| Box::pin(async { vec![AgentMessage::from("dynamic")] }));

        let abort = AbortSignal::new();
        let result = q.drain(QueueMode::All, &Some(supplier), abort).await;
        assert_eq!(result, vec![AgentMessage::from("dynamic")]);
        assert!(q.is_empty());
    }

    #[tokio::test]
    async fn test_drain_one_at_a_time_empty_supplier_returns_empty() {
        let q = MessageQueue::new(QueueKind::Steering);

        let supplier: GetQueuedMessagesFn = Arc::new(|_signal| Box::pin(async { Vec::new() }));

        let abort = AbortSignal::new();
        let result = q.drain(QueueMode::OneAtATime, &Some(supplier), abort).await;
        assert!(result.is_empty());
        assert!(q.is_empty());
    }

    #[tokio::test]
    async fn test_drain_no_supplier() {
        let q = MessageQueue::new(QueueKind::FollowUp);
        q.push(make_msg("a"));

        let abort = AbortSignal::new();
        let result = q.drain(QueueMode::All, &None, abort).await;
        assert_eq!(result.len(), 1);
    }

    // ========== Phase 4 tests ==========

    #[test]
    fn test_drain_strategy_all() {
        let mut buf = VecDeque::from(vec![make_msg("a"), make_msg("b"), make_msg("c")]);
        let strategy = DrainAll;
        let result = strategy.select(&mut buf);
        assert_eq!(result.len(), 3);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_drain_strategy_one() {
        let mut buf = VecDeque::from(vec![make_msg("a"), make_msg("b")]);
        let strategy = DrainOne;
        let result = strategy.select(&mut buf);
        assert_eq!(result.len(), 1);
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_drain_strategy_batch() {
        let mut buf = VecDeque::from(vec![
            make_msg("a"),
            make_msg("b"),
            make_msg("c"),
            make_msg("d"),
        ]);
        let strategy = DrainBatch { max: 2 };
        let result = strategy.select(&mut buf);
        assert_eq!(result.len(), 2);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_drain_with_strategy() {
        let q = MessageQueue::new(QueueKind::Steering);
        q.push_many(vec![make_msg("a"), make_msg("b"), make_msg("c")]);

        let strategy = DrainBatch { max: 2 };
        let result = q.drain_with_strategy(&strategy);
        assert_eq!(result.len(), 2);
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_backpressure_unlimited() {
        let q = MessageQueue::new(QueueKind::Steering);
        // Default is unlimited
        for i in 0..1000 {
            assert!(q.try_push(make_msg(&format!("msg{}", i))).is_ok());
        }
        assert_eq!(q.len(), 1000);
    }

    #[test]
    fn test_backpressure_reject() {
        let q = MessageQueue::new(QueueKind::Steering);
        q.set_backpressure(BackpressureConfig {
            max_depth: 3,
            overflow: OverflowBehavior::Reject,
        });

        assert!(q.try_push(make_msg("a")).is_ok());
        assert!(q.try_push(make_msg("b")).is_ok());
        assert!(q.try_push(make_msg("c")).is_ok());
        // Queue is full
        let err = q.try_push(make_msg("d")).unwrap_err();
        assert_eq!(err.current_depth, 3);
        assert_eq!(err.max_depth, 3);
        assert_eq!(q.len(), 3);
    }

    #[test]
    fn test_backpressure_drop_oldest() {
        let q = MessageQueue::new(QueueKind::Steering);
        q.set_backpressure(BackpressureConfig {
            max_depth: 3,
            overflow: OverflowBehavior::DropOldest,
        });

        assert!(q.try_push(make_msg("a")).is_ok());
        assert!(q.try_push(make_msg("b")).is_ok());
        assert!(q.try_push(make_msg("c")).is_ok());
        // This should succeed, dropping "a"
        assert!(q.try_push(make_msg("d")).is_ok());
        assert_eq!(q.len(), 3);

        // Drain and verify oldest was dropped, remainder order preserved
        let msgs = q.drain_local(QueueMode::All);
        assert_eq!(msgs.len(), 3);
        let texts: Vec<&str> = msgs
            .iter()
            .filter_map(|m| match m {
                AgentMessage::User(u) => match &u.content {
                    UserContent::Text(t) => Some(t.as_str()),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["b", "c", "d"]);
    }

    #[test]
    fn test_queue_mode_to_strategy() {
        let mut buf = VecDeque::from(vec![make_msg("a"), make_msg("b")]);
        let strategy = QueueMode::All.to_strategy();
        let result = strategy.select(&mut buf);
        assert_eq!(result.len(), 2);

        let mut buf2 = VecDeque::from(vec![make_msg("x"), make_msg("y")]);
        let strategy2 = QueueMode::OneAtATime.to_strategy();
        let result2 = strategy2.select(&mut buf2);
        assert_eq!(result2.len(), 1);
        assert_eq!(buf2.len(), 1);
    }
}
