//! Agent implementation with full conversation loop.

use crate::agent::{
    AbortSignal, AfterToolCallContext, AfterToolCallFn, AgentConfig, AgentContext, AgentEvent,
    AgentHooks, AgentLoopOptions, AgentMessage, AgentState, AgentStateSnapshot, AgentTool,
    AgentToolResult, BeforeToolCallContext, BeforeToolCallFn, BeforeToolCallResult, QueueMode,
    ThinkingBudgets, ToolExecutionMode, ToolExecutor, ToolUpdateCallback, Transport,
};
use crate::provider::{get_provider, ArcProtocol};
use crate::stream::{AssistantMessageEventStream, EventStream};
use crate::thinking::ThinkingLevel;
use crate::types::*;
use futures::StreamExt;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Default maximum number of turns (LLM calls) per prompt.
pub const DEFAULT_MAX_TURNS: usize = 25;

const INCOMPLETE_TURN_MAX_RETRIES: usize = 3;
const INCOMPLETE_TURN_RETRY_DELAYS_MS: [u64; INCOMPLETE_TURN_MAX_RETRIES] = [1_000, 2_000, 4_000];
const INCOMPLETE_TURN_TOTAL_RETRY_BUDGET: Duration = Duration::from_secs(10);

/// Subscriber ID for unsubscription.
pub type SubscriberId = u64;

/// Stream of agent lifecycle events with a final loop result.
pub type AgentEventStream = EventStream<AgentEvent, Result<Vec<AgentMessage>, AgentError>>;

/// Callback type for event subscribers.
type SubscriberCallback = Arc<dyn Fn(&AgentEvent) + Send + Sync>;

/// Thread-safe subscriber storage using HashMap to avoid tombstone leaks.
struct Subscribers {
    callbacks: RwLock<HashMap<u64, SubscriberCallback>>,
    next_id: AtomicU64,
}

impl Subscribers {
    fn new() -> Self {
        Self {
            callbacks: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(0),
        }
    }

    fn subscribe(&self, callback: SubscriberCallback) -> SubscriberId {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        self.callbacks.write().insert(id, callback);
        id
    }

    fn unsubscribe(&self, id: SubscriberId) {
        self.callbacks.write().remove(&id);
    }

    /// Emit an event to all subscribers.
    /// Clones Arcs under read lock, then calls callbacks outside the lock
    /// to prevent blocking subscribe/unsubscribe operations.
    fn emit(&self, event: &AgentEvent) {
        let snapshot: Vec<SubscriberCallback> =
            { self.callbacks.read().values().cloned().collect() };
        for cb in &snapshot {
            cb(event);
        }
    }
}

/// Agent for managing stateful conversations with LLM providers.
pub struct Agent {
    /// Agent state.
    state: Arc<AgentState>,
    /// Configuration.
    config: RwLock<AgentConfig>,
    /// Provider (optional, resolved from registry if not set).
    provider: RwLock<Option<ArcProtocol>>,
    /// Aggregated hooks (tool executor, before/after hooks, converters, etc.).
    hooks: RwLock<AgentHooks>,
    /// Maximum turns per prompt.
    max_turns: RwLock<usize>,
    /// Steering message queue.
    steering_queue: Mutex<VecDeque<AgentMessage>>,
    /// Follow-up message queue.
    follow_up_queue: Mutex<VecDeque<AgentMessage>>,
    /// Event subscribers (HashMap-based, no tombstone leak).
    subscribers: Arc<Subscribers>,
    /// Abort flag.
    abort_flag: Arc<AtomicBool>,
    /// When true, queued steering messages are injected only after the current turn ends.
    defer_steering_until_turn_end: Arc<AtomicBool>,
    /// API key for the provider.
    api_key: RwLock<Option<String>>,
    /// Session ID for caching.
    session_id: RwLock<Option<String>>,
    /// Cancellation signal for the active run, if any.
    run_abort_signal: RwLock<Option<AbortSignal>>,
}

impl Agent {
    /// Create a new agent with default configuration.
    pub fn new() -> Self {
        Self {
            state: Arc::new(AgentState::new()),
            config: RwLock::new(AgentConfig::new(
                Model::builder()
                    .id("gpt-4o-mini")
                    .name("GPT-4o Mini")
                    .provider(Provider::OpenAI)
                    .base_url("https://api.openai.com/v1")
                    .context_window(128000)
                    .max_tokens(16384)
                    .build()
                    .unwrap(),
            )),
            provider: RwLock::new(None),
            hooks: RwLock::new(AgentHooks::default()),
            max_turns: RwLock::new(DEFAULT_MAX_TURNS),
            steering_queue: Mutex::new(VecDeque::new()),
            follow_up_queue: Mutex::new(VecDeque::new()),
            subscribers: Arc::new(Subscribers::new()),
            abort_flag: Arc::new(AtomicBool::new(false)),
            defer_steering_until_turn_end: Arc::new(AtomicBool::new(false)),
            api_key: RwLock::new(None),
            session_id: RwLock::new(None),
            run_abort_signal: RwLock::new(None),
        }
    }

    /// Create an agent with a model.
    pub fn with_model(model: Model) -> Self {
        let agent = Self::new();
        agent.set_model(model.clone());
        *agent.config.write() = AgentConfig::new(model);
        agent
    }

    /// Create an agent from explicit loop state, config, and runtime options.
    pub fn from_parts(
        context: AgentContext,
        config: AgentConfig,
        options: AgentLoopOptions,
    ) -> Self {
        let agent = Self {
            state: Arc::new(AgentState::new()),
            config: RwLock::new(config),
            provider: RwLock::new(options.provider),
            hooks: RwLock::new(options.hooks),
            max_turns: RwLock::new(options.max_turns.unwrap_or(DEFAULT_MAX_TURNS)),
            steering_queue: Mutex::new(VecDeque::new()),
            follow_up_queue: Mutex::new(VecDeque::new()),
            subscribers: Arc::new(Subscribers::new()),
            abort_flag: Arc::new(AtomicBool::new(false)),
            defer_steering_until_turn_end: Arc::new(AtomicBool::new(false)),
            api_key: RwLock::new(options.api_key),
            session_id: RwLock::new(options.session_id),
            run_abort_signal: RwLock::new(None),
        };

        agent.set_system_prompt(context.system_prompt);
        agent.replace_messages(context.messages);
        if let Some(tools) = context.tools {
            agent.set_tools(tools);
        }

        agent
    }

    // ============================================================================
    // Provider & API Key
    // ============================================================================

    /// Set the LLM provider explicitly.
    pub fn set_provider(&self, provider: ArcProtocol) {
        *self.provider.write() = Some(provider);
    }

    /// Set a static API key.
    pub fn set_api_key(&self, key: impl Into<String>) {
        *self.api_key.write() = Some(key.into());
    }

    /// Set a dynamic API key resolver.
    ///
    /// Called before each LLM request. Useful for short-lived OAuth tokens
    /// that may expire during long-running tool execution phases.
    pub fn set_get_api_key<F, Fut>(&self, resolver: F)
    where
        F: Fn(&str) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Option<String>> + Send + 'static,
    {
        let resolver = Arc::new(move |provider: &str, _signal: AbortSignal| {
            let fut = resolver(provider);
            Box::pin(fut)
                as std::pin::Pin<Box<dyn std::future::Future<Output = Option<String>> + Send>>
        });
        self.hooks.write().get_api_key = Some(resolver);
    }

    /// Set a dynamic API key resolver with cancellation awareness.
    pub fn set_get_api_key_with_signal<F, Fut>(&self, resolver: F)
    where
        F: Fn(&str, AbortSignal) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Option<String>> + Send + 'static,
    {
        let resolver = Arc::new(move |provider: &str, signal: AbortSignal| {
            let fut = resolver(provider, signal);
            Box::pin(fut)
                as std::pin::Pin<Box<dyn std::future::Future<Output = Option<String>> + Send>>
        });
        self.hooks.write().get_api_key = Some(resolver);
    }

    // ============================================================================
    // Tool Executor & Hooks
    // ============================================================================

    /// Set the tool executor callback.
    ///
    /// The executor receives `(tool_name, tool_call_id, arguments, update_callback)`.
    /// The `update_callback` can be called during execution to push streaming
    /// partial results (emitted as `ToolExecutionUpdate` events).
    pub fn set_tool_executor<F, Fut>(&self, executor: F)
    where
        F: Fn(&str, &str, &serde_json::Value, Option<ToolUpdateCallback>) -> Fut
            + Send
            + Sync
            + 'static,
        Fut: std::future::Future<Output = AgentToolResult> + Send + 'static,
    {
        let executor = Arc::new(
            move |name: &str,
                  id: &str,
                  args: &serde_json::Value,
                  update_cb: Option<ToolUpdateCallback>| {
                let fut = executor(name, id, args, update_cb);
                Box::pin(fut)
                    as std::pin::Pin<Box<dyn std::future::Future<Output = AgentToolResult> + Send>>
            },
        );
        self.hooks.write().tool_executor = Some(executor);
    }

    /// Set the tool executor callback (simple version without update callback).
    ///
    /// Convenience method for tools that don't need streaming updates.
    pub fn set_tool_executor_simple<F, Fut>(&self, executor: F)
    where
        F: Fn(&str, &str, &serde_json::Value) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = AgentToolResult> + Send + 'static,
    {
        let executor = Arc::new(
            move |name: &str,
                  id: &str,
                  args: &serde_json::Value,
                  _update_cb: Option<ToolUpdateCallback>| {
                let fut = executor(name, id, args);
                Box::pin(fut)
                    as std::pin::Pin<Box<dyn std::future::Future<Output = AgentToolResult> + Send>>
            },
        );
        self.hooks.write().tool_executor = Some(executor);
    }

    /// Set the `before_tool_call` hook.
    ///
    /// Called after arguments are validated but before tool execution.
    /// Return `BeforeToolCallResult { block: true, .. }` to prevent execution.
    pub fn set_before_tool_call<F, Fut>(&self, hook: F)
    where
        F: Fn(BeforeToolCallContext) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Option<BeforeToolCallResult>> + Send + 'static,
    {
        let hook = Arc::new(move |ctx: BeforeToolCallContext| {
            let fut = hook(ctx);
            Box::pin(fut)
                as std::pin::Pin<
                    Box<dyn std::future::Future<Output = Option<BeforeToolCallResult>> + Send>,
                >
        });
        self.hooks.write().before_tool_call = Some(hook);
    }

    /// Set the `after_tool_call` hook.
    ///
    /// Called after tool execution, before the result is committed.
    /// Return `AfterToolCallResult` to override content, details, or is_error.
    pub fn set_after_tool_call<F, Fut>(&self, hook: F)
    where
        F: Fn(AfterToolCallContext) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Option<crate::agent::AfterToolCallResult>>
            + Send
            + 'static,
    {
        let hook = Arc::new(move |ctx: AfterToolCallContext| {
            let fut = hook(ctx);
            Box::pin(fut)
                as std::pin::Pin<
                    Box<
                        dyn std::future::Future<Output = Option<crate::agent::AfterToolCallResult>>
                            + Send,
                    >,
                >
        });
        self.hooks.write().after_tool_call = Some(hook);
    }

    // ============================================================================
    // Context Pipeline
    // ============================================================================

    /// Set the custom `AgentMessage[]` → `Message[]` conversion function.
    ///
    /// Called before each LLM request. The default filters out `Custom` messages
    /// and maps User/Assistant/ToolResult directly.
    pub fn set_convert_to_llm<F, Fut>(&self, converter: F)
    where
        F: Fn(Vec<AgentMessage>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Vec<Message>> + Send + 'static,
    {
        let converter = Arc::new(move |msgs: Vec<AgentMessage>| {
            let fut = converter(msgs);
            Box::pin(fut)
                as std::pin::Pin<Box<dyn std::future::Future<Output = Vec<Message>> + Send>>
        });
        self.hooks.write().convert_to_llm = Some(converter);
    }

    /// Set the context transformation function (applied BEFORE `convert_to_llm`).
    ///
    /// Use this for context window management, message pruning, injecting
    /// external context, etc.
    pub fn set_transform_context<F, Fut>(&self, transform: F)
    where
        F: Fn(Vec<AgentMessage>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Vec<AgentMessage>> + Send + 'static,
    {
        let transform = Arc::new(move |msgs: Vec<AgentMessage>, _signal: AbortSignal| {
            let fut = transform(msgs);
            Box::pin(fut)
                as std::pin::Pin<Box<dyn std::future::Future<Output = Vec<AgentMessage>> + Send>>
        });
        self.hooks.write().transform_context = Some(transform);
    }

    /// Set the context transformation function with cancellation awareness.
    pub fn set_transform_context_with_signal<F, Fut>(&self, transform: F)
    where
        F: Fn(Vec<AgentMessage>, AbortSignal) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Vec<AgentMessage>> + Send + 'static,
    {
        let transform = Arc::new(move |msgs: Vec<AgentMessage>, signal: AbortSignal| {
            let fut = transform(msgs, signal);
            Box::pin(fut)
                as std::pin::Pin<Box<dyn std::future::Future<Output = Vec<AgentMessage>> + Send>>
        });
        self.hooks.write().transform_context = Some(transform);
    }

    // ============================================================================
    // Payload & Stream Hooks
    // ============================================================================

    /// Set the payload inspection / replacement hook.
    ///
    /// Called with the serialized request body before it is sent to the provider.
    pub fn set_on_payload<F, Fut>(&self, hook: F)
    where
        F: Fn(serde_json::Value, Model) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Option<serde_json::Value>> + Send + 'static,
    {
        let hook = Arc::new(move |payload: serde_json::Value, model: Model| {
            let fut = hook(payload, model);
            Box::pin(fut)
                as std::pin::Pin<
                    Box<dyn std::future::Future<Output = Option<serde_json::Value>> + Send>,
                >
        });
        self.hooks.write().on_payload = Some(hook);
    }

    /// Set the pre-serialization message hook.
    ///
    /// Called after `convert_to_llm` converts `AgentMessage[]` into `Message[]`
    /// and before the messages are assembled into a `Context`. The hook receives
    /// the target `Model`, allowing provider-specific structural normalisation
    /// (e.g., injecting `reasoning_content` for DeepSeek) at the typed-message
    /// level rather than on raw JSON.
    pub fn set_on_messages<F, Fut>(&self, handler: F)
    where
        F: Fn(Vec<Message>, Model) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Vec<Message>> + Send + 'static,
    {
        let handler = Arc::new(move |messages: Vec<Message>, model: Model| {
            let fut = handler(messages, model);
            Box::pin(fut)
                as std::pin::Pin<Box<dyn std::future::Future<Output = Vec<Message>> + Send>>
        });
        self.hooks.write().on_messages = Some(handler);
    }

    /// Set a custom stream function to replace the default provider streaming.
    ///
    /// Useful for proxy backends, custom routing, etc.
    pub fn set_stream_fn<F, Fut>(&self, stream_fn: F)
    where
        F: Fn(&Model, &Context, StreamOptions) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = AssistantMessageEventStream> + Send + 'static,
    {
        let stream_fn = Arc::new(
            move |model: &Model,
                  context: &Context,
                  options: SimpleStreamOptions,
                  _signal: AbortSignal| {
                let fut = stream_fn(model, context, options.base);
                Box::pin(fut)
                    as std::pin::Pin<
                        Box<dyn std::future::Future<Output = AssistantMessageEventStream> + Send>,
                    >
            },
        );
        self.hooks.write().stream_fn = Some(stream_fn);
    }

    /// Set a custom stream function with full simple-stream options and cancellation.
    pub fn set_stream_fn_with_signal<F, Fut>(&self, stream_fn: F)
    where
        F: Fn(&Model, &Context, SimpleStreamOptions, AbortSignal) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = AssistantMessageEventStream> + Send + 'static,
    {
        let stream_fn = Arc::new(
            move |model: &Model,
                  context: &Context,
                  options: SimpleStreamOptions,
                  signal: AbortSignal| {
                let fut = stream_fn(model, context, options, signal);
                Box::pin(fut)
                    as std::pin::Pin<
                        Box<dyn std::future::Future<Output = AssistantMessageEventStream> + Send>,
                    >
            },
        );
        self.hooks.write().stream_fn = Some(stream_fn);
    }

    /// Set a dynamic steering-message supplier.
    pub fn set_get_steering_messages<F, Fut>(&self, getter: F)
    where
        F: Fn(AbortSignal) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Vec<AgentMessage>> + Send + 'static,
    {
        let getter = Arc::new(move |signal: AbortSignal| {
            let fut = getter(signal);
            Box::pin(fut)
                as std::pin::Pin<Box<dyn std::future::Future<Output = Vec<AgentMessage>> + Send>>
        });
        self.hooks.write().get_steering_messages = Some(getter);
    }

    /// Set a dynamic follow-up-message supplier.
    pub fn set_get_follow_up_messages<F, Fut>(&self, getter: F)
    where
        F: Fn(AbortSignal) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Vec<AgentMessage>> + Send + 'static,
    {
        let getter = Arc::new(move |signal: AbortSignal| {
            let fut = getter(signal);
            Box::pin(fut)
                as std::pin::Pin<Box<dyn std::future::Future<Output = Vec<AgentMessage>> + Send>>
        });
        self.hooks.write().get_follow_up_messages = Some(getter);
    }

    // ============================================================================
    // Configuration Setters
    // ============================================================================

    /// Set maximum turns per prompt.
    pub fn set_max_turns(&self, max: usize) {
        *self.max_turns.write() = max;
    }

    /// Set the security configuration.
    pub fn set_security_config(&self, config: crate::types::SecurityConfig) {
        self.config.write().security = config;
    }

    /// Get the current security configuration.
    pub fn security_config(&self) -> crate::types::SecurityConfig {
        self.config.read().security.clone()
    }

    /// Set tool execution mode.
    pub fn set_tool_execution(&self, mode: ToolExecutionMode) {
        self.config.write().tool_execution = mode;
    }

    /// Set the steering queue mode.
    pub fn set_steering_mode(&self, mode: QueueMode) {
        self.config.write().steering_mode = mode;
    }

    /// Get the steering queue mode.
    pub fn steering_mode(&self) -> QueueMode {
        self.config.read().steering_mode
    }

    /// Set the follow-up queue mode.
    pub fn set_follow_up_mode(&self, mode: QueueMode) {
        self.config.write().follow_up_mode = mode;
    }

    /// Get the follow-up queue mode.
    pub fn follow_up_mode(&self) -> QueueMode {
        self.config.read().follow_up_mode
    }

    /// Set custom thinking budgets.
    pub fn set_thinking_budgets(&self, budgets: ThinkingBudgets) {
        self.config.write().thinking_budgets = Some(budgets);
    }

    /// Get the current thinking budgets.
    pub fn thinking_budgets(&self) -> Option<ThinkingBudgets> {
        self.config.read().thinking_budgets.clone()
    }

    /// Set the preferred transport.
    pub fn set_transport(&self, transport: Transport) {
        self.config.write().transport = transport;
    }

    /// Get the preferred transport.
    pub fn transport(&self) -> Transport {
        self.config.read().transport
    }

    /// Set the maximum number of retries for transient HTTP or pre-stream transport failures.
    pub fn set_max_retries(&self, retries: Option<u32>) {
        self.config.write().max_retries = retries;
    }

    /// Get the current max retry count.
    pub fn max_retries(&self) -> Option<u32> {
        self.config.read().max_retries
    }

    /// Set the maximum retry delay in milliseconds.
    ///
    /// If the server requests a retry delay exceeding this value, the request
    /// fails immediately so higher-level retry logic can handle it with user
    /// visibility. `None` = use default (60_000ms). `Some(0)` = disable cap.
    pub fn set_max_retry_delay_ms(&self, ms: Option<u64>) {
        self.config.write().max_retry_delay_ms = ms;
    }

    /// Get the current max retry delay.
    pub fn max_retry_delay_ms(&self) -> Option<u64> {
        self.config.read().max_retry_delay_ms
    }

    /// Set custom HTTP headers to include in every LLM API request.
    pub fn set_custom_headers(&self, headers: std::collections::HashMap<String, String>) {
        self.config.write().custom_headers = Some(headers);
    }

    /// Set the session ID for caching.
    pub fn set_session_id(&self, id: impl Into<String>) {
        *self.session_id.write() = Some(id.into());
    }

    /// Get the current session ID.
    pub fn session_id(&self) -> Option<String> {
        self.session_id.read().clone()
    }

    /// Clear the session ID.
    pub fn clear_session_id(&self) {
        *self.session_id.write() = None;
    }

    // ============================================================================
    // Event Subscription
    // ============================================================================

    /// Subscribe to agent events. Returns an unsubscribe closure.
    pub fn subscribe<F>(&self, callback: F) -> impl Fn()
    where
        F: Fn(&AgentEvent) + Send + Sync + 'static,
    {
        let id = self.subscribers.subscribe(Arc::new(callback));
        let subs = Arc::clone(&self.subscribers);
        move || {
            subs.unsubscribe(id);
        }
    }

    /// Emit an event to all subscribers.
    fn emit(&self, event: AgentEvent) {
        self.subscribers.emit(&event);
    }

    // ============================================================================
    // State Management
    // ============================================================================

    /// Set the system prompt.
    pub fn set_system_prompt(&self, prompt: impl Into<String>) {
        self.state.set_system_prompt(prompt);
    }

    /// Set the model.
    pub fn set_model(&self, model: Model) {
        self.config.write().model = model;
    }

    /// Set the thinking level.
    pub fn set_thinking_level(&self, level: ThinkingLevel) {
        self.config.write().thinking_level = level;
    }

    /// Set the tools.
    pub fn set_tools(&self, tools: Vec<AgentTool>) {
        self.state.set_tools(tools);
    }

    /// Replace all messages.
    pub fn replace_messages(&self, messages: Vec<AgentMessage>) {
        self.state.replace_messages(messages);
    }

    /// Append a message.
    pub fn append_message(&self, message: AgentMessage) {
        self.state.add_message(message);
    }

    /// Clear all messages.
    pub fn clear_messages(&self) {
        self.state.clear_messages();
    }

    /// Reset the agent.
    pub fn reset(&self) {
        self.state.reset();
        self.steering_queue.lock().clear();
        self.follow_up_queue.lock().clear();
        self.defer_steering_until_turn_end
            .store(false, Ordering::SeqCst);
        *self.session_id.write() = None;
        if let Some(signal) = self.run_abort_signal.write().take() {
            signal.cancel();
        }
    }

    // ============================================================================
    // Steering and Follow-up
    // ============================================================================

    /// Add a steering message (interrupts current work).
    pub fn steer(&self, message: AgentMessage) {
        self.steering_queue.lock().push_back(message);
    }

    /// Add a follow-up message (processed after current work completes).
    pub fn follow_up(&self, message: AgentMessage) {
        self.follow_up_queue.lock().push_back(message);
    }

    /// Clear steering queue.
    pub fn clear_steering_queue(&self) {
        self.steering_queue.lock().clear();
    }

    /// Clear follow-up queue.
    pub fn clear_follow_up_queue(&self) {
        self.follow_up_queue.lock().clear();
    }

    /// Clear all queues.
    pub fn clear_all_queues(&self) {
        self.clear_steering_queue();
        self.clear_follow_up_queue();
    }

    /// Check if there are queued messages.
    pub fn has_queued_messages(&self) -> bool {
        !self.steering_queue.lock().is_empty() || !self.follow_up_queue.lock().is_empty()
    }

    fn current_abort_signal(&self) -> AbortSignal {
        self.run_abort_signal.read().clone().unwrap_or_default()
    }

    fn dequeue_local_messages(
        queue: &Mutex<VecDeque<AgentMessage>>,
        mode: QueueMode,
    ) -> Vec<AgentMessage> {
        let mut queue = queue.lock();
        match mode {
            QueueMode::All => queue.drain(..).collect(),
            QueueMode::OneAtATime => {
                if let Some(first) = queue.pop_front() {
                    vec![first]
                } else {
                    Vec::new()
                }
            }
        }
    }

    /// Dequeue steering messages from the local queue only.
    fn dequeue_steering_messages(&self) -> Vec<AgentMessage> {
        let mode = self.config.read().steering_mode;
        Self::dequeue_local_messages(&self.steering_queue, mode)
    }

    async fn poll_queue_messages(
        &self,
        mode: QueueMode,
        local: &Mutex<VecDeque<AgentMessage>>,
        dynamic: &Option<
            Arc<
                dyn Fn(
                        AbortSignal,
                    ) -> std::pin::Pin<
                        Box<dyn std::future::Future<Output = Vec<AgentMessage>> + Send>,
                    > + Send
                    + Sync,
            >,
        >,
    ) -> Vec<AgentMessage> {
        let mut messages = Self::dequeue_local_messages(local, mode);
        let dynamic_messages = if let Some(callback) = dynamic {
            callback(self.current_abort_signal()).await
        } else {
            Vec::new()
        };

        match mode {
            QueueMode::All => {
                messages.extend(dynamic_messages);
                messages
            }
            QueueMode::OneAtATime => {
                if messages.is_empty() {
                    dynamic_messages.into_iter().take(1).collect()
                } else {
                    messages.truncate(1);
                    messages
                }
            }
        }
    }

    async fn poll_steering_messages(&self) -> Vec<AgentMessage> {
        let mode = self.config.read().steering_mode;
        let dynamic = self.hooks.read().get_steering_messages.clone();
        self.poll_queue_messages(mode, &self.steering_queue, &dynamic)
            .await
    }

    async fn poll_follow_up_messages(&self) -> Vec<AgentMessage> {
        let mode = self.config.read().follow_up_mode;
        let dynamic = self.hooks.read().get_follow_up_messages.clone();
        self.poll_queue_messages(mode, &self.follow_up_queue, &dynamic)
            .await
    }

    async fn dequeue_deferred_steering_messages(&self) -> Vec<AgentMessage> {
        if !self.defer_steering_until_turn_end.load(Ordering::SeqCst) {
            return Vec::new();
        }

        let messages = self.poll_steering_messages().await;
        let still_has_queued = !self.steering_queue.lock().is_empty();
        self.defer_steering_until_turn_end
            .store(still_has_queued, Ordering::SeqCst);
        messages
    }

    // ============================================================================
    // Core Agent Loop
    // ============================================================================

    /// Default `convert_to_llm`: filters out Custom messages and maps directly.
    fn default_convert_to_llm(messages: Vec<AgentMessage>) -> Vec<Message> {
        messages
            .into_iter()
            .filter_map(|m| {
                let opt: Option<Message> = m.into();
                opt
            })
            .collect()
    }

    /// Build the context from current agent state using the full pipeline:
    /// `messages → transform_context → convert_to_llm → Context`
    async fn build_context(&self) -> Context {
        let system_prompt = self.state.system_prompt.read().clone();
        let messages = self.state.messages.read().clone();
        let tools = self.state.tools.read().clone();

        // Step 1: transform_context (if set)
        let transform = self.hooks.read().transform_context.clone();
        let abort_signal = self.current_abort_signal();
        let messages = if let Some(ref transform) = transform {
            transform(messages, abort_signal).await
        } else {
            messages
        };

        // Step 2: convert_to_llm
        let converter = self.hooks.read().convert_to_llm.clone();
        let llm_messages = if let Some(ref converter) = converter {
            converter(messages).await
        } else {
            Self::default_convert_to_llm(messages)
        };

        // Step 2.5: on_messages hook (pre-serialization, with Model)
        let on_messages = self.hooks.read().on_messages.clone();
        let llm_messages = if let Some(ref handler) = on_messages {
            let model = self.config.read().model.clone();
            handler(llm_messages, model).await
        } else {
            llm_messages
        };

        // Step 3: Build Context
        let mut context = if system_prompt.is_empty() {
            Context::new()
        } else {
            Context::with_system_prompt(&system_prompt)
        };

        for msg in llm_messages {
            context.add_message(msg);
        }

        // Add tools
        if !tools.is_empty() {
            let tool_defs: Vec<Tool> = tools.iter().map(|t| t.as_tool()).collect();
            context.set_tools(tool_defs);
        }

        context
    }

    /// Resolve the provider to use.
    fn resolve_provider(&self) -> Result<ArcProtocol, AgentError> {
        // First check explicit provider
        if let Some(ref provider) = *self.provider.read() {
            return Ok(provider.clone());
        }

        // Then try registry by Provider type
        let model = self.config.read().model.clone();
        if let Some(provider) = get_provider(&model.provider) {
            return Ok(provider);
        }

        Err(AgentError::ProviderError(format!(
            "No provider registered for provider type: {}",
            model.provider.as_str()
        )))
    }

    /// Build stream options, resolving API key dynamically if configured.
    async fn build_stream_options(&self) -> StreamOptions {
        let security = self.config.read().security.clone();
        let model = self.config.read().model.clone();
        let on_payload = self.hooks.read().on_payload.clone();
        let transport = self.config.read().transport;
        let max_retries = self.config.read().max_retries;
        let max_retry_delay_ms = self.config.read().max_retry_delay_ms;
        let custom_headers = self.config.read().custom_headers.clone();
        let session_id = self.session_id.read().clone();
        let abort_signal = self.current_abort_signal();

        // Dynamic API key resolution: getApiKey > static api_key
        let get_api_key = self.hooks.read().get_api_key.clone();
        let api_key = if let Some(ref resolver) = get_api_key {
            let dynamic = resolver(model.provider.as_str(), abort_signal.clone()).await;
            dynamic.or_else(|| self.api_key.read().clone())
        } else {
            self.api_key.read().clone()
        };

        StreamOptions {
            api_key,
            security: Some(security),
            session_id,
            on_payload,
            transport: Some(transport),
            max_retries,
            max_retry_delay_ms,
            headers: custom_headers,
            cancel_token: Some(abort_signal),
            ..Default::default()
        }
    }

    /// Build SimpleStreamOptions with thinking level/budget resolution.
    async fn build_simple_stream_options(&self) -> SimpleStreamOptions {
        let base = self.build_stream_options().await;
        let thinking_level = self.config.read().thinking_level;

        let (reasoning, thinking_budget_tokens) = if thinking_level != ThinkingLevel::Off {
            let budget = self
                .config
                .read()
                .thinking_budgets
                .as_ref()
                .and_then(|b| b.budget_for(thinking_level))
                .or_else(|| {
                    Some(crate::thinking::ThinkingConfig::default_budget(
                        thinking_level,
                    ))
                });
            (Some(thinking_level), budget)
        } else {
            (None, None)
        };

        SimpleStreamOptions {
            base,
            reasoning,
            thinking_budget_tokens,
            thinking_display: None,
        }
    }

    fn append_run_message(
        &self,
        new_messages: &mut Vec<AgentMessage>,
        message: AgentMessage,
        emit_start: bool,
        emit_end: bool,
        turn_index: usize,
    ) {
        self.state.add_message(message.clone());
        new_messages.push(message.clone());
        if emit_start {
            self.emit(AgentEvent::MessageStart {
                turn_index,
                message: message.clone(),
            });
        }
        if emit_end {
            let response_id = match &message {
                AgentMessage::Assistant(a) => a.response_id.clone(),
                _ => None,
            };
            self.emit(AgentEvent::MessageEnd {
                turn_index,
                response_id,
                message,
            });
        }
    }

    fn append_terminal_error_message(
        &self,
        new_messages: &mut Vec<AgentMessage>,
        error: &AgentError,
        turn_index: usize,
    ) -> AgentMessage {
        let model = self.config.read().model.clone();
        let partial = self
            .state
            .stream_message
            .read()
            .clone()
            .and_then(|message| match message {
                AgentMessage::Assistant(assistant) => Some(assistant),
                _ => None,
            });
        let (stop_reason, error_message) = stop_reason_and_message_for_error(error);
        let terminal =
            self.build_terminal_assistant_message(&model, partial, stop_reason, error_message);
        let message = AgentMessage::Assistant(terminal);
        self.append_run_message(new_messages, message.clone(), true, true, turn_index);
        message
    }

    fn build_terminal_assistant_message(
        &self,
        model: &Model,
        partial: Option<AssistantMessage>,
        stop_reason: StopReason,
        error_message: impl Into<String>,
    ) -> AssistantMessage {
        let error_message = error_message.into();
        let mut message = partial.unwrap_or_else(|| {
            AssistantMessage::builder()
                .api(effective_api_for_model(model))
                .provider(model.provider.clone())
                .model(model.id.clone())
                .usage(Usage::default())
                .stop_reason(stop_reason)
                .build()
                .expect("terminal assistant message should be buildable")
        });
        message.api = effective_api_for_model(model);
        message.provider = model.provider.clone();
        message.model = model.id.clone();
        message.stop_reason = stop_reason;
        message.error_message = Some(error_message.clone());
        if message.content.is_empty() {
            message.content = vec![ContentBlock::Text(TextContent::new(""))];
        }
        message
    }

    /// Run a single LLM turn: call provider, consume stream, return AssistantMessage.
    async fn run_turn(
        &self,
        provider: &ArcProtocol,
        turn_index: usize,
    ) -> Result<AssistantMessage, AgentError> {
        let context = self.build_context().await;
        let model = self.config.read().model.clone();
        let options = self.build_simple_stream_options().await;
        let stream_timeout = self.config.read().security.stream.result_timeout();
        let abort_signal = self.current_abort_signal();

        // Create the stream (custom stream_fn or default provider via stream_simple)
        let stream_fn = self.hooks.read().stream_fn.clone();
        let mut stream: AssistantMessageEventStream = if let Some(ref custom_stream) = stream_fn {
            custom_stream(&model, &context, options.clone(), abort_signal.clone()).await
        } else {
            provider.stream_simple(&model, &context, options)
        };
        let mut emitted_message_start = false;

        // Process stream events
        loop {
            let next_event = tokio::select! {
                _ = abort_signal.cancelled() => {
                    let partial = self
                        .state
                        .stream_message
                        .read()
                        .clone()
                        .and_then(|message| match message {
                            AgentMessage::Assistant(assistant) => Some(assistant),
                            _ => None,
                        });
                    *self.state.stream_message.write() = None;
                    return Ok(self.build_terminal_assistant_message(
                        &model,
                        partial,
                        StopReason::Aborted,
                        "Aborted",
                    ));
                }
                event = stream.next() => event,
            };

            let Some(event) = next_event else {
                break;
            };

            // Check for steering messages
            if !self.defer_steering_until_turn_end.load(Ordering::SeqCst) {
                let steering = self.dequeue_steering_messages();
                if !steering.is_empty() {
                    // Apply steering: add steering messages to state
                    for steer_msg in steering {
                        self.state.add_message(steer_msg.clone());
                        self.emit(AgentEvent::MessageStart {
                            turn_index,
                            message: steer_msg.clone(),
                        });
                        self.emit(AgentEvent::MessageEnd {
                            turn_index,
                            response_id: None,
                            message: steer_msg,
                        });
                    }
                    // Abort current turn and restart
                    return Err(AgentError::Other("Steered".to_string()));
                }
            }

            // Forward stream event to subscribers
            match &event {
                AssistantMessageEvent::Start { partial } => {
                    *self.state.stream_message.write() =
                        Some(AgentMessage::Assistant(partial.clone()));
                    self.emit(AgentEvent::MessageStart {
                        turn_index,
                        message: AgentMessage::Assistant(partial.clone()),
                    });
                    emitted_message_start = true;
                    self.emit(AgentEvent::MessageUpdate {
                        turn_index,
                        message: AgentMessage::Assistant(partial.clone()),
                        assistant_event: Box::new(event.clone()),
                    });
                }
                AssistantMessageEvent::TextDelta { .. }
                | AssistantMessageEvent::ThinkingDelta { .. }
                | AssistantMessageEvent::ToolCallDelta { .. } => {
                    if let Some(partial) = event.partial_message() {
                        *self.state.stream_message.write() =
                            Some(AgentMessage::Assistant(partial.clone()));
                        self.emit(AgentEvent::MessageUpdate {
                            turn_index,
                            message: AgentMessage::Assistant(partial.clone()),
                            assistant_event: Box::new(event.clone()),
                        });
                    }
                }
                AssistantMessageEvent::Retrying { .. } => {
                    let message = self.state.stream_message.read().clone().unwrap_or_else(|| {
                        AgentMessage::Assistant(
                            AssistantMessage::builder()
                                .api(effective_api_for_model(&model))
                                .provider(model.provider.clone())
                                .model(model.id.clone())
                                .usage(Usage::default())
                                .stop_reason(StopReason::Stop)
                                .build()
                                .expect("retrying assistant message should be buildable"),
                        )
                    });
                    self.emit(AgentEvent::MessageUpdate {
                        turn_index,
                        message,
                        assistant_event: Box::new(event.clone()),
                    });
                }
                _ => {
                    if let Some(partial) = event.partial_message() {
                        self.emit(AgentEvent::MessageUpdate {
                            turn_index,
                            message: AgentMessage::Assistant(partial.clone()),
                            assistant_event: Box::new(event.clone()),
                        });
                    }
                }
            }
        }

        // Get the final result with timeout to prevent infinite blocking
        let result = match stream.try_result(stream_timeout).await {
            Some(r) => r,
            None => {
                return Err(AgentError::Other(format!(
                    "Stream result timed out after {:?}",
                    stream_timeout
                )));
            }
        };

        // Clear streaming message
        *self.state.stream_message.write() = None;
        if !emitted_message_start {
            self.emit(AgentEvent::MessageStart {
                turn_index,
                message: AgentMessage::Assistant(result.clone()),
            });
        }

        Ok(result)
    }

    /// Execute tool calls from an assistant message.
    ///
    /// Supports: beforeToolCall/afterToolCall hooks, tool validation,
    /// streaming ToolExecutionUpdate events, bounded parallel exec + timeout,
    /// and abort-aware execution.
    async fn execute_tool_calls(
        &self,
        assistant_msg: &AssistantMessage,
        context: &Context,
        turn_index: usize,
    ) -> Vec<ToolResultMessage> {
        let tool_calls = assistant_msg.tool_calls();
        if tool_calls.is_empty() {
            return Vec::new();
        }

        let executor = self.hooks.read().tool_executor.clone();
        let execution_mode = self.config.read().tool_execution;
        let security = self.config.read().security.clone();
        let tool_timeout = security.agent.tool_execution_timeout();
        let before_hook = self.hooks.read().before_tool_call.clone();
        let after_hook = self.hooks.read().after_tool_call.clone();

        // Build Tool list for validation
        let agent_tools = self.state.tools.read().clone();
        let tool_defs: Vec<Tool> = agent_tools.iter().map(|t| t.as_tool()).collect();

        let mut results = Vec::new();

        match execution_mode {
            ToolExecutionMode::Parallel => {
                let max_parallel = security.agent.max_parallel_tool_calls;
                let abort_flag = Arc::clone(&self.abort_flag);
                let abort_signal = self.current_abort_signal();

                let mut ordered_results: Vec<Option<ToolResultMessage>> =
                    vec![None; tool_calls.len()];
                let mut tool_futures = Vec::new();

                for (index, tc) in tool_calls.iter().enumerate() {
                    let tc_id = tc.id.clone();
                    let tc_name = tc.name.clone();
                    let tc_args = tc.arguments.clone();
                    let tc_clone = (*tc).clone();

                    self.emit(AgentEvent::ToolExecutionStart {
                        turn_index,
                        tool_call_id: tc_id.clone(),
                        tool_name: tc_name.clone(),
                        args: tc_args.clone(),
                    });

                    self.state.pending_tool_calls.write().insert(tc_id.clone());

                    // Validate tool call before execution
                    if let Some(result) =
                        validate_tool_call_or_error(&tc_name, &tc_args, &tool_defs, &security)
                    {
                        self.emit(AgentEvent::ToolExecutionEnd {
                            turn_index,
                            tool_call_id: tc_id.clone(),
                            tool_name: tc_name.clone(),
                            result: tool_result_payload(&result),
                            is_error: true,
                        });
                        self.state.pending_tool_calls.write().remove(&tc_id);
                        ordered_results[index] =
                            Some(build_tool_result_message(tc_id, tc_name, result, true));
                        continue;
                    }

                    // beforeToolCall hook
                    if let Some(result) = run_before_hook(
                        &before_hook,
                        assistant_msg,
                        &tc_clone,
                        &tc_args,
                        context,
                        abort_signal.clone(),
                    )
                    .await
                    {
                        self.emit(AgentEvent::ToolExecutionEnd {
                            turn_index,
                            tool_call_id: tc_id.clone(),
                            tool_name: tc_name.clone(),
                            result: tool_result_payload(&result),
                            is_error: true,
                        });
                        self.state.pending_tool_calls.write().remove(&tc_id);
                        ordered_results[index] =
                            Some(build_tool_result_message(tc_id, tc_name, result, true));
                        continue;
                    }

                    let executor = executor.clone();
                    let abort = abort_flag.clone();
                    let after_hook = after_hook.clone();
                    let assistant_msg_clone = assistant_msg.clone();
                    let context_clone = context.clone();
                    let subscribers = Arc::clone(&self.subscribers);
                    let abort_signal = abort_signal.clone();

                    tool_futures.push(async move {
                        let (final_result, final_is_error) =
                            execute_and_apply_after_hook(ToolExecCtx {
                                executor: &executor,
                                after_hook: &after_hook,
                                subscribers: &subscribers,
                                tc_id: &tc_id,
                                tc_name: &tc_name,
                                tc_args: &tc_args,
                                tc: &tc_clone,
                                assistant_msg: &assistant_msg_clone,
                                context: &context_clone,
                                tool_timeout,
                                abort_flag: abort,
                                abort_signal,
                                turn_index,
                            })
                            .await;

                        (index, tc_id, tc_name, final_result, final_is_error)
                    });
                }

                // Use buffer_unordered for bounded parallel execution
                let mut buffered =
                    futures::stream::iter(tool_futures).buffer_unordered(max_parallel);

                while let Some((index, tc_id, tc_name, result, is_error)) = buffered.next().await {
                    ordered_results[index] = Some(build_tool_result_message(
                        tc_id.clone(),
                        tc_name.clone(),
                        result,
                        is_error,
                    ));
                }

                for result in ordered_results.into_iter().flatten() {
                    self.emit(AgentEvent::ToolExecutionEnd {
                        turn_index,
                        tool_call_id: result.tool_call_id.clone(),
                        tool_name: result.tool_name.clone(),
                        result: tool_result_message_payload(&result),
                        is_error: result.is_error,
                    });

                    self.state
                        .pending_tool_calls
                        .write()
                        .remove(&result.tool_call_id);

                    results.push(result);
                }
            }
            ToolExecutionMode::Sequential => {
                let abort_signal = self.current_abort_signal();
                for tc in &tool_calls {
                    if self.abort_flag.load(Ordering::SeqCst) {
                        break;
                    }

                    let tc_id = tc.id.clone();
                    let tc_name = tc.name.clone();
                    let tc_args = tc.arguments.clone();
                    let tc_clone = (*tc).clone();

                    self.emit(AgentEvent::ToolExecutionStart {
                        turn_index,
                        tool_call_id: tc_id.clone(),
                        tool_name: tc_name.clone(),
                        args: tc_args.clone(),
                    });

                    self.state.pending_tool_calls.write().insert(tc_id.clone());

                    // Validate tool call before execution
                    if let Some(result) =
                        validate_tool_call_or_error(&tc_name, &tc_args, &tool_defs, &security)
                    {
                        let result_msg =
                            build_tool_result_message(tc_id.clone(), tc_name.clone(), result, true);
                        self.emit(AgentEvent::ToolExecutionEnd {
                            turn_index,
                            tool_call_id: tc_id.clone(),
                            tool_name: tc_name.clone(),
                            result: tool_result_message_payload(&result_msg),
                            is_error: true,
                        });
                        self.state.pending_tool_calls.write().remove(&tc_id);
                        results.push(result_msg);
                        continue;
                    }

                    // beforeToolCall hook
                    if let Some(result) = run_before_hook(
                        &before_hook,
                        assistant_msg,
                        &tc_clone,
                        &tc_args,
                        context,
                        abort_signal.clone(),
                    )
                    .await
                    {
                        let result_msg =
                            build_tool_result_message(tc_id.clone(), tc_name.clone(), result, true);
                        self.emit(AgentEvent::ToolExecutionEnd {
                            turn_index,
                            tool_call_id: tc_id.clone(),
                            tool_name: tc_name.clone(),
                            result: tool_result_message_payload(&result_msg),
                            is_error: true,
                        });
                        self.state.pending_tool_calls.write().remove(&tc_id);
                        results.push(result_msg);
                        continue;
                    }

                    let abort_flag = Arc::clone(&self.abort_flag);
                    let (final_result, final_is_error) =
                        execute_and_apply_after_hook(ToolExecCtx {
                            executor: &executor,
                            after_hook: &after_hook,
                            subscribers: &self.subscribers,
                            tc_id: &tc_id,
                            tc_name: &tc_name,
                            tc_args: &tc_args,
                            tc: &tc_clone,
                            assistant_msg,
                            context,
                            tool_timeout,
                            abort_flag,
                            abort_signal: abort_signal.clone(),
                            turn_index,
                        })
                        .await;

                    let result_msg = build_tool_result_message(
                        tc_id.clone(),
                        tc_name.clone(),
                        final_result,
                        final_is_error,
                    );
                    self.emit(AgentEvent::ToolExecutionEnd {
                        turn_index,
                        tool_call_id: tc_id.clone(),
                        tool_name: tc_name.clone(),
                        result: tool_result_message_payload(&result_msg),
                        is_error: result_msg.is_error,
                    });

                    self.state.pending_tool_calls.write().remove(&tc_id);

                    results.push(result_msg);

                    // Check for steering messages after each sequential tool
                    let steering = self.dequeue_steering_messages();
                    if !steering.is_empty() {
                        for steer_msg in steering {
                            self.state.add_message(steer_msg);
                        }
                        // Break out of remaining tool calls
                        break;
                    }
                }
            }
        }

        results
    }

    /// Run the agent loop: stream LLM → check tool calls → execute → loop.
    async fn run_loop(&self) -> AgentRunOutcome {
        let provider = if self.hooks.read().stream_fn.is_some() {
            // When a custom stream function is set, we don't need a provider.
            // Create a dummy Arc for the loop (won't be used).
            None
        } else {
            match self.resolve_provider() {
                Ok(provider) => Some(provider),
                Err(error) => {
                    let mut messages = Vec::new();
                    self.append_terminal_error_message(&mut messages, &error, 0);
                    *self.state.error.write() = Some(error.to_string());
                    return AgentRunOutcome::error(messages, error);
                }
            }
        };

        let max_turns = *self.max_turns.read();
        let mut new_messages = Vec::new();
        let mut turn_count = 0;
        let mut incomplete_turn_retries = 0usize;
        let mut incomplete_turn_retry_started_at: Option<Instant> = None;

        // Sync message limit from security config
        let max_messages = self.config.read().security.agent.max_messages;
        self.state.set_max_messages(max_messages);

        loop {
            // Check abort
            if self.abort_flag.load(Ordering::SeqCst) {
                let error = AgentError::Other("Aborted".to_string());
                if !matches!(
                    new_messages.last(),
                    Some(AgentMessage::Assistant(message))
                        if message.stop_reason == StopReason::Aborted
                ) {
                    self.append_terminal_error_message(&mut new_messages, &error, turn_count);
                }
                *self.state.error.write() = Some(error.to_string());
                return AgentRunOutcome::error(new_messages, error);
            }

            // Stop explicitly when the loop budget is exhausted so callers can
            // distinguish "hit safety limit" from a normal, model-authored stop.
            if turn_count >= max_turns {
                return AgentRunOutcome::error(
                    new_messages,
                    AgentError::MaxTurnsReached(max_turns),
                );
            }

            self.emit(AgentEvent::TurnStart {
                turn_index: turn_count,
            });
            let turn_snapshot = self.snapshot();
            let new_messages_len_before_turn = new_messages.len();

            // Run one LLM turn
            let dummy_provider: ArcProtocol = Arc::new(DummyProvider);
            let active_provider = provider.as_ref().unwrap_or(&dummy_provider);
            let assistant_result = self.run_turn(active_provider, turn_count).await;

            match assistant_result {
                Ok(assistant_msg) => {
                    // Add assistant message to state and new_messages
                    let agent_msg = AgentMessage::Assistant(assistant_msg.clone());
                    self.append_run_message(
                        &mut new_messages,
                        agent_msg.clone(),
                        false,
                        true,
                        turn_count,
                    );

                    // Check if there are tool calls
                    if assistant_msg.has_tool_calls()
                        && assistant_msg.stop_reason == StopReason::ToolUse
                    {
                        // Build context snapshot for tool hook use after the assistant message
                        // has been committed to state, matching the visible conversation.
                        let context = self.build_context().await;
                        let tool_results = self
                            .execute_tool_calls(&assistant_msg, &context, turn_count)
                            .await;

                        for result in &tool_results {
                            let result_msg = AgentMessage::ToolResult(result.clone());
                            self.state.add_message(result_msg.clone());
                            new_messages.push(result_msg.clone());
                            self.emit(AgentEvent::MessageStart {
                                turn_index: turn_count,
                                message: result_msg.clone(),
                            });
                            self.emit(AgentEvent::MessageEnd {
                                turn_index: turn_count,
                                response_id: None,
                                message: result_msg,
                            });
                        }

                        self.emit(AgentEvent::TurnEnd {
                            turn_index: turn_count,
                            message: agent_msg,
                            tool_results,
                        });

                        let deferred_steering = self.dequeue_deferred_steering_messages().await;
                        if !deferred_steering.is_empty() {
                            for msg in deferred_steering {
                                self.append_run_message(
                                    &mut new_messages,
                                    msg,
                                    true,
                                    true,
                                    turn_count,
                                );
                            }
                            incomplete_turn_retries = 0;
                            incomplete_turn_retry_started_at = None;
                            turn_count += 1;
                            continue;
                        }

                        // Check for follow-up messages
                        let follow_ups = self.poll_follow_up_messages().await;
                        for msg in follow_ups {
                            self.append_run_message(&mut new_messages, msg, true, true, turn_count);
                        }

                        incomplete_turn_retries = 0;
                        incomplete_turn_retry_started_at = None;
                        turn_count += 1;
                        continue;
                    } else {
                        // No tool calls — conversation turn is complete
                        self.emit(AgentEvent::TurnEnd {
                            turn_index: turn_count,
                            message: agent_msg.clone(),
                            tool_results: Vec::new(),
                        });

                        if matches!(
                            assistant_msg.stop_reason,
                            StopReason::Error | StopReason::Aborted
                        ) {
                            let agent_error = agent_error_from_assistant(&assistant_msg);
                            if matches!(agent_error, AgentError::IncompleteStream { .. }) {
                                let started_at = incomplete_turn_retry_started_at
                                    .get_or_insert_with(Instant::now);
                                let retry_delay_ms = INCOMPLETE_TURN_RETRY_DELAYS_MS
                                    .get(incomplete_turn_retries)
                                    .copied();
                                let can_retry = retry_delay_ms
                                    .map(Duration::from_millis)
                                    .is_some_and(|delay| {
                                        incomplete_turn_retries < INCOMPLETE_TURN_MAX_RETRIES
                                            && started_at.elapsed() + delay
                                                <= INCOMPLETE_TURN_TOTAL_RETRY_BUDGET
                                    });

                                self.emit(AgentEvent::MessageDiscarded {
                                    turn_index: turn_count,
                                    message: agent_msg.clone(),
                                    reason: agent_error.to_string(),
                                });
                                new_messages.truncate(new_messages_len_before_turn);
                                self.restore_snapshot(&turn_snapshot);

                                if can_retry {
                                    let delay_ms =
                                        retry_delay_ms.expect("retry delay should exist");
                                    incomplete_turn_retries += 1;
                                    self.emit(AgentEvent::TurnRetrying {
                                        attempt: incomplete_turn_retries,
                                        max_attempts: INCOMPLETE_TURN_MAX_RETRIES,
                                        delay_ms,
                                        reason: agent_error.to_string(),
                                    });
                                    if let Err(retry_error) = self
                                        .sleep_for_turn_retry(Duration::from_millis(delay_ms))
                                        .await
                                    {
                                        *self.state.error.write() = Some(retry_error.to_string());
                                        return AgentRunOutcome::error(new_messages, retry_error);
                                    }
                                    continue;
                                }

                                *self.state.error.write() = Some(agent_error.to_string());
                                return AgentRunOutcome::error(new_messages, agent_error);
                            }

                            *self.state.error.write() = Some(agent_error.to_string());
                            return AgentRunOutcome::error(new_messages, agent_error);
                        }

                        incomplete_turn_retries = 0;
                        incomplete_turn_retry_started_at = None;
                        let deferred_steering = self.dequeue_deferred_steering_messages().await;
                        if !deferred_steering.is_empty() {
                            for msg in deferred_steering {
                                self.append_run_message(
                                    &mut new_messages,
                                    msg,
                                    true,
                                    true,
                                    turn_count,
                                );
                            }
                            turn_count += 1;
                            continue;
                        }

                        // Check for follow-up messages
                        let follow_ups = self.poll_follow_up_messages().await;
                        if !follow_ups.is_empty() {
                            for msg in follow_ups {
                                self.append_run_message(
                                    &mut new_messages,
                                    msg,
                                    true,
                                    true,
                                    turn_count,
                                );
                            }
                            turn_count += 1;
                            continue;
                        }

                        break;
                    }
                }
                Err(AgentError::Other(ref msg)) if msg == "Steered" => {
                    incomplete_turn_retries = 0;
                    incomplete_turn_retry_started_at = None;
                    turn_count += 1;
                    continue;
                }
                Err(e) => {
                    *self.state.error.write() = Some(e.to_string());
                    let terminal_message =
                        self.append_terminal_error_message(&mut new_messages, &e, turn_count);
                    self.emit(AgentEvent::TurnEnd {
                        turn_index: turn_count,
                        message: terminal_message,
                        tool_results: Vec::new(),
                    });
                    return AgentRunOutcome::error(new_messages, e);
                }
            }
        }

        AgentRunOutcome::ok(new_messages)
    }

    // ============================================================================
    // Prompt methods
    // ============================================================================

    async fn finish_run(&self, outcome: AgentRunOutcome) -> Result<Vec<AgentMessage>, AgentError> {
        self.state.set_streaming(false);
        *self.state.stream_message.write() = None;
        self.state.pending_tool_calls.write().clear();
        self.run_abort_signal.write().take();

        match outcome.error {
            None => {
                let messages = outcome.messages;
                self.emit(AgentEvent::AgentEnd {
                    messages: messages.clone(),
                });
                Ok(messages)
            }
            Some(error) => {
                self.emit(AgentEvent::AgentEnd {
                    messages: outcome.messages,
                });
                Err(error)
            }
        }
    }

    async fn prompt_messages_locked(
        &self,
        messages: Vec<AgentMessage>,
    ) -> Result<Vec<AgentMessage>, AgentError> {
        self.abort_flag.store(false, Ordering::SeqCst);
        *self.run_abort_signal.write() = Some(AbortSignal::new());
        *self.state.error.write() = None;
        *self.state.stream_message.write() = None;

        for message in &messages {
            self.state.add_message(message.clone());
        }

        self.emit(AgentEvent::AgentStart);
        for message in &messages {
            self.emit(AgentEvent::MessageStart {
                turn_index: 0,
                message: message.clone(),
            });
            self.emit(AgentEvent::MessageEnd {
                turn_index: 0,
                response_id: None,
                message: message.clone(),
            });
        }

        let mut outcome = self.run_loop().await;
        let mut prefixed_messages = messages;
        prefixed_messages.extend(outcome.messages);
        outcome.messages = prefixed_messages;
        self.finish_run(outcome).await
    }

    async fn continue_locked(&self) -> Result<Vec<AgentMessage>, AgentError> {
        self.abort_flag.store(false, Ordering::SeqCst);
        *self.run_abort_signal.write() = Some(AbortSignal::new());
        *self.state.error.write() = None;
        *self.state.stream_message.write() = None;
        self.emit(AgentEvent::AgentStart);
        let outcome = self.run_loop().await;
        self.finish_run(outcome).await
    }

    /// Send a prompt to the agent.
    ///
    /// Uses atomic compare_exchange to prevent TOCTOU race condition.
    pub async fn prompt(
        &self,
        message: impl Into<AgentMessage>,
    ) -> Result<Vec<AgentMessage>, AgentError> {
        self.prompt_messages(vec![message.into()]).await
    }

    /// Send multiple prompt messages as a single loop start.
    ///
    /// Uses atomic compare_exchange to prevent TOCTOU race condition.
    pub async fn prompt_messages(
        &self,
        messages: Vec<AgentMessage>,
    ) -> Result<Vec<AgentMessage>, AgentError> {
        // Atomic CAS: only one caller wins the race
        if self
            .state
            .is_streaming
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(AgentError::AlreadyStreaming);
        }

        self.prompt_messages_locked(messages).await
    }

    /// Continue from current state (e.g., after adding tool results externally).
    ///
    /// Uses atomic compare_exchange to prevent TOCTOU race condition.
    pub async fn continue_(&self) -> Result<Vec<AgentMessage>, AgentError> {
        if self
            .state
            .is_streaming
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(AgentError::AlreadyStreaming);
        }

        let last_is_assistant = {
            let messages = self.state.messages.read();
            if messages.is_empty() {
                self.state.set_streaming(false);
                return Err(AgentError::NoMessages);
            }
            matches!(messages.last(), Some(AgentMessage::Assistant(_)))
        };

        if last_is_assistant {
            let steering = self.poll_steering_messages().await;
            if !steering.is_empty() {
                self.defer_steering_until_turn_end
                    .store(!self.steering_queue.lock().is_empty(), Ordering::SeqCst);
                return self.prompt_messages_locked(steering).await;
            }

            let follow_ups = self.poll_follow_up_messages().await;
            if !follow_ups.is_empty() {
                return self.prompt_messages_locked(follow_ups).await;
            }

            self.state.set_streaming(false);
            return Err(AgentError::CannotContinueFromAssistant);
        }

        self.continue_locked().await
    }

    /// Abort current operation.
    pub fn abort(&self) {
        self.abort_flag.store(true, Ordering::SeqCst);
        self.defer_steering_until_turn_end
            .store(false, Ordering::SeqCst);
        if let Some(signal) = self.run_abort_signal.read().clone() {
            signal.cancel();
        }
        self.state.set_streaming(false);
    }

    /// Wait for the agent to become idle.
    pub async fn wait_for_idle(&self) {
        while self.state.is_streaming() {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }

    /// Get the current state.
    pub fn state(&self) -> &Arc<AgentState> {
        &self.state
    }

    /// Take a consistent point-in-time snapshot of the agent's full state.
    ///
    /// Combines runtime state from [`AgentState`] with configuration
    /// (model, thinking_level) from [`AgentConfig`].
    pub fn snapshot(&self) -> AgentStateSnapshot {
        let config = self.config.read();
        let system_prompt = self.state.system_prompt.read().clone();
        let messages = self.state.messages.read().clone();
        let is_streaming = self.state.is_streaming();
        let stream_message = self.state.stream_message.read().clone();
        let pending_tool_calls = self.state.pending_tool_calls.read().clone();
        let error = self.state.error.read().clone();
        let max_messages = self.state.get_max_messages();
        let message_count = messages.len();

        AgentStateSnapshot {
            system_prompt,
            model: config.model.clone(),
            thinking_level: config.thinking_level,
            messages,
            is_streaming,
            stream_message,
            pending_tool_calls,
            error,
            message_count,
            max_messages,
        }
    }

    fn restore_snapshot(&self, snapshot: &AgentStateSnapshot) {
        self.state.set_system_prompt(snapshot.system_prompt.clone());
        self.state.replace_messages(snapshot.messages.clone());
        self.state.set_streaming(snapshot.is_streaming);
        *self.state.stream_message.write() = snapshot.stream_message.clone();
        *self.state.pending_tool_calls.write() = snapshot.pending_tool_calls.clone();
        *self.state.error.write() = snapshot.error.clone();
        self.state.set_max_messages(snapshot.max_messages);

        let mut config = self.config.write();
        config.model = snapshot.model.clone();
        config.thinking_level = snapshot.thinking_level;
    }

    async fn sleep_for_turn_retry(&self, delay: Duration) -> Result<(), AgentError> {
        let abort_signal = self.current_abort_signal();
        tokio::select! {
            _ = abort_signal.cancelled() => Err(AgentError::Other("Aborted".to_string())),
            _ = tokio::time::sleep(delay) => Ok(()),
        }
    }
}

/// Helper: wait until the abort flag is set.
async fn wait_for_abort(flag: Arc<AtomicBool>) {
    loop {
        if flag.load(Ordering::SeqCst) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

// ============================================================================
// Extracted helpers for execute_tool_calls deduplication
// ============================================================================

/// Validate a tool call against the tool definitions and security config.
///
/// Returns `Some(AgentToolResult)` (an error result) if validation fails,
/// or `None` if the tool call is valid and execution should proceed.
/// Skips validation when disabled in config or when no tools are registered.
fn validate_tool_call_or_error(
    tc_name: &str,
    tc_args: &serde_json::Value,
    tool_defs: &[Tool],
    security: &SecurityConfig,
) -> Option<AgentToolResult> {
    if !security.agent.validate_tool_calls || tool_defs.is_empty() {
        return None;
    }

    let tc = ToolCall::new("validation", tc_name, tc_args.clone());
    match crate::validation::validate_tool_call(tool_defs, &tc) {
        Ok(_) => None,
        Err(e) => Some(AgentToolResult::error(e.to_string())),
    }
}

/// Run the `before_tool_call` hook if set.
///
/// Returns `Some(AgentToolResult)` (a blocked/error result) if the hook
/// blocked execution, or `None` if execution should proceed.
async fn run_before_hook(
    before_hook: &Option<BeforeToolCallFn>,
    assistant_msg: &AssistantMessage,
    tc: &ToolCall,
    tc_args: &serde_json::Value,
    context: &Context,
    abort_signal: AbortSignal,
) -> Option<AgentToolResult> {
    let hook = before_hook.as_ref()?;
    let ctx = BeforeToolCallContext {
        assistant_message: assistant_msg.clone(),
        tool_call: tc.clone(),
        args: tc_args.clone(),
        context: context.clone(),
        abort_signal,
    };
    match hook(ctx).await {
        Some(result) if result.block => {
            let reason = result
                .reason
                .unwrap_or_else(|| "Tool call blocked by before_tool_call hook".to_string());
            Some(AgentToolResult::error(reason))
        }
        _ => None,
    }
}

fn tool_result_payload(result: &AgentToolResult) -> serde_json::Value {
    serde_json::json!({
        "content": result.content,
        "details": result.details,
    })
}

fn build_tool_result_message(
    tool_call_id: impl Into<String>,
    tool_name: impl Into<String>,
    result: AgentToolResult,
    is_error: bool,
) -> ToolResultMessage {
    let tool_call_id = tool_call_id.into();
    let tool_name = tool_name.into();
    let message = ToolResultMessage::new(tool_call_id, tool_name, result.content, is_error);

    if let Some(details) = result.details {
        message.with_details(details)
    } else {
        message
    }
}

fn tool_result_message_payload(result: &ToolResultMessage) -> serde_json::Value {
    serde_json::json!({
        "content": result.content,
        "details": result.details,
    })
}

/// Context for a single tool call execution.
///
/// Groups the parameters needed by `execute_and_apply_after_hook` to avoid
/// exceeding clippy's `too_many_arguments` limit.
struct ToolExecCtx<'a> {
    executor: &'a Option<ToolExecutor>,
    after_hook: &'a Option<AfterToolCallFn>,
    subscribers: &'a Arc<Subscribers>,
    tc_id: &'a str,
    tc_name: &'a str,
    tc_args: &'a serde_json::Value,
    tc: &'a ToolCall,
    assistant_msg: &'a AssistantMessage,
    context: &'a Context,
    tool_timeout: std::time::Duration,
    abort_flag: Arc<AtomicBool>,
    abort_signal: AbortSignal,
    turn_index: usize,
}

/// Execute a tool call and apply the `after_tool_call` hook if set.
///
/// Handles: executor invocation with timeout, abort-awareness,
/// streaming `ToolExecutionUpdate` events, error detection,
/// and after-hook overrides.
///
/// Returns `(final_result, final_is_error)`.
async fn execute_and_apply_after_hook(ctx: ToolExecCtx<'_>) -> (AgentToolResult, bool) {
    let ToolExecCtx {
        executor,
        after_hook,
        subscribers,
        tc_id,
        tc_name,
        tc_args,
        tc,
        assistant_msg,
        context,
        tool_timeout,
        abort_flag,
        abort_signal,
        turn_index,
    } = ctx;
    // Execute the tool
    let tool_result = if let Some(ref exec) = executor {
        // Build update callback for streaming partial results
        let subs = Arc::clone(subscribers);
        let update_tc_id = tc_id.to_string();
        let update_tc_name = tc_name.to_string();
        let update_tc_args = tc_args.clone();
        let update_cb: ToolUpdateCallback = Arc::new(move |partial: serde_json::Value| {
            subs.emit(&AgentEvent::ToolExecutionUpdate {
                turn_index,
                tool_call_id: update_tc_id.clone(),
                tool_name: update_tc_name.clone(),
                args: update_tc_args.clone(),
                partial_result: partial,
            });
        });

        let exec_future = exec(tc_name, tc_id, tc_args, Some(update_cb));

        // Race: tool execution vs timeout vs abort
        tokio::select! {
            result = exec_future => result,
            _ = tokio::time::sleep(tool_timeout) => {
                AgentToolResult::error(format!(
                    "Tool '{}' timed out after {:?}",
                    tc_name, tool_timeout
                ))
            }
            _ = wait_for_abort(abort_flag) => {
                AgentToolResult::error(format!("Tool '{}' aborted", tc_name))
            }
        }
    } else {
        AgentToolResult::error(format!(
            "No tool executor configured for tool '{}'",
            tc_name
        ))
    };

    // Detect is_error from content
    let mut is_error = tool_result.content.iter().any(|block| {
        if let Some(text) = block.as_text() {
            text.text.starts_with("Error:") || text.text.starts_with("error:")
        } else {
            false
        }
    });

    let mut final_result = tool_result.clone();

    // Apply after_tool_call hook
    if let Some(ref hook) = after_hook {
        let after_ctx = AfterToolCallContext {
            assistant_message: assistant_msg.clone(),
            tool_call: tc.clone(),
            args: tc_args.clone(),
            result: tool_result,
            is_error,
            context: context.clone(),
            abort_signal,
        };
        if let Some(overrides) = hook(after_ctx).await {
            if let Some(content_override) = overrides.content {
                final_result.content = content_override;
            }
            if let Some(details_override) = overrides.details {
                final_result.details = Some(details_override);
            }
            if let Some(error_override) = overrides.is_error {
                is_error = error_override;
            }
        }
    }

    (final_result, is_error)
}

impl Default for Agent {
    fn default() -> Self {
        Self::new()
    }
}

struct AgentRunOutcome {
    messages: Vec<AgentMessage>,
    error: Option<AgentError>,
}

impl AgentRunOutcome {
    fn ok(messages: Vec<AgentMessage>) -> Self {
        Self {
            messages,
            error: None,
        }
    }

    fn error(messages: Vec<AgentMessage>, error: AgentError) -> Self {
        Self {
            messages,
            error: Some(error),
        }
    }
}

fn stop_reason_and_message_for_error(error: &AgentError) -> (StopReason, String) {
    match error {
        AgentError::Other(message) if message == "Aborted" => {
            (StopReason::Aborted, message.clone())
        }
        AgentError::ProviderError(message) => (StopReason::Error, message.clone()),
        other => (StopReason::Error, other.to_string()),
    }
}

fn agent_error_from_assistant(message: &AssistantMessage) -> AgentError {
    let error_message = message
        .error_message
        .clone()
        .unwrap_or_else(|| message.stop_reason.to_string());
    match message.stop_reason {
        StopReason::Aborted => AgentError::Other("Aborted".to_string()),
        StopReason::Error => {
            if let Some((provider, detail)) =
                crate::protocol::common::parse_incomplete_stream_error(&error_message)
            {
                AgentError::IncompleteStream { provider, detail }
            } else {
                AgentError::ProviderError(error_message)
            }
        }
        _ => AgentError::Other(error_message),
    }
}

fn effective_api_for_model(model: &Model) -> Api {
    if let Some(api) = model.api.clone() {
        return api;
    }

    match &model.provider {
        Provider::OpenAI | Provider::OpenAIResponses | Provider::AzureOpenAIResponses => {
            Api::OpenAIResponses
        }
        Provider::Anthropic | Provider::MiniMax | Provider::MiniMaxCN | Provider::KimiCoding => {
            Api::AnthropicMessages
        }
        Provider::Google | Provider::GoogleGeminiCli | Provider::GoogleAntigravity => {
            Api::GoogleGenerativeAi
        }
        Provider::GoogleVertex => Api::GoogleVertex,
        Provider::Ollama => Api::Ollama,
        Provider::Zenmux => {
            let base = model.base_url.as_deref().unwrap_or("");
            if base.is_empty() || base.starts_with(crate::provider::zenmux::ZENMUX_HOST_PREFIX) {
                crate::provider::zenmux::zenmux_detect_api(&model.id)
            } else {
                Api::OpenAICompletions
            }
        }
        Provider::Bai => {
            let base = model.base_url.as_deref().unwrap_or("");
            if base.is_empty() || base.starts_with(crate::provider::bai::BAI_HOST_PREFIX) {
                crate::provider::bai::bai_detect_api(&model.id)
            } else {
                Api::OpenAICompletions
            }
        }
        Provider::XAI
        | Provider::Groq
        | Provider::OpenRouter
        | Provider::OpenAICompatible
        | Provider::OpenAICodex
        | Provider::GitHubCopilot
        | Provider::Cerebras
        | Provider::VercelAiGateway
        | Provider::ZAI
        | Provider::Mistral
        | Provider::HuggingFace
        | Provider::OpenCode
        | Provider::OpenCodeGo
        | Provider::DeepSeek
        | Provider::XiaomiMIMO => Api::OpenAICompletions,
        Provider::AmazonBedrock => Api::BedrockConverseStream,
        Provider::Custom(name) => Api::Custom(name.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::effective_api_for_model;
    use crate::types::{Api, Model, Provider};

    fn zenmux_model(id: &str, base_url: Option<&str>) -> Model {
        let mut builder = Model::builder()
            .id(id)
            .name(id)
            .provider(Provider::Zenmux)
            .context_window(128000)
            .max_tokens(8192);
        if let Some(base_url) = base_url {
            builder = builder.base_url(base_url);
        }
        builder.build().unwrap()
    }

    #[test]
    fn test_effective_api_for_zenmux_kimi_models_uses_openai_completions() {
        assert_eq!(
            effective_api_for_model(&zenmux_model("kimi-k2.5", None)),
            Api::OpenAICompletions
        );
        assert_eq!(
            effective_api_for_model(&zenmux_model("moonshotai/kimi-k2.5", None)),
            Api::OpenAICompletions
        );
        assert_eq!(
            effective_api_for_model(&zenmux_model(
                "moonshotai/kimi-k2.5:anthropic",
                Some("https://zenmux.ai/api/v1")
            )),
            Api::OpenAICompletions
        );
    }

    fn bai_model(id: &str, base_url: Option<&str>) -> Model {
        let mut builder = Model::builder()
            .id(id)
            .name(id)
            .provider(Provider::Bai)
            .context_window(128000)
            .max_tokens(8192);
        if let Some(base_url) = base_url {
            builder = builder.base_url(base_url);
        }
        builder.build().unwrap()
    }

    #[test]
    fn test_effective_api_for_bai_claude_models_uses_anthropic() {
        assert_eq!(
            effective_api_for_model(&bai_model("claude-sonnet-4", None)),
            Api::AnthropicMessages
        );
        assert_eq!(
            effective_api_for_model(&bai_model("claude-opus-4.6", None)),
            Api::AnthropicMessages
        );
        assert_eq!(
            effective_api_for_model(&bai_model("Claude-3.5-Sonnet", None)),
            Api::AnthropicMessages
        );
    }

    #[test]
    fn test_effective_api_for_bai_non_claude_models_uses_openai_completions() {
        assert_eq!(
            effective_api_for_model(&bai_model("gpt-4o", None)),
            Api::OpenAIResponses
        );
        assert_eq!(
            effective_api_for_model(&bai_model("deepseek-r1", None)),
            Api::OpenAICompletions
        );
    }

    #[test]
    fn test_effective_api_for_bai_custom_base_url_uses_openai_completions() {
        assert_eq!(
            effective_api_for_model(&bai_model(
                "claude-sonnet-4",
                Some("https://custom.example.com/v1")
            )),
            Api::OpenAICompletions
        );
    }
}

fn agent_event_never_completes(_: &AgentEvent) -> bool {
    false
}

fn unreachable_agent_event_result(_: AgentEvent) -> Result<Vec<AgentMessage>, AgentError> {
    unreachable!("agent event streams complete via EventStream::end")
}

fn new_agent_event_stream() -> AgentEventStream {
    EventStream::new(agent_event_never_completes, unreachable_agent_event_result)
}

/// Run the standalone agent loop starting with prompt messages.
pub async fn run_agent_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: AgentConfig,
    options: AgentLoopOptions,
) -> Result<Vec<AgentMessage>, AgentError> {
    let agent = Agent::from_parts(context, config, options);
    agent.prompt_messages(prompts).await
}

/// Continue the standalone agent loop from an existing context.
pub async fn run_agent_loop_continue(
    context: AgentContext,
    config: AgentConfig,
    options: AgentLoopOptions,
) -> Result<Vec<AgentMessage>, AgentError> {
    let agent = Agent::from_parts(context, config, options);
    agent.continue_().await
}

/// Stream standalone agent-loop events while processing prompt messages.
pub fn agent_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: AgentConfig,
    options: AgentLoopOptions,
) -> AgentEventStream {
    let stream = new_agent_event_stream();
    let stream_for_task = stream.clone();

    tokio::spawn(async move {
        let agent = Agent::from_parts(context, config, options);
        let stream_for_events = stream_for_task.clone();
        let _unsubscribe = agent.subscribe(move |event| {
            stream_for_events.push(event.clone());
        });

        let result = agent.prompt_messages(prompts).await;
        stream_for_task.end(Some(result));
    });

    stream
}

/// Stream standalone agent-loop events while continuing an existing context.
pub fn agent_loop_continue(
    context: AgentContext,
    config: AgentConfig,
    options: AgentLoopOptions,
) -> AgentEventStream {
    let stream = new_agent_event_stream();
    let stream_for_task = stream.clone();

    tokio::spawn(async move {
        let agent = Agent::from_parts(context, config, options);
        let stream_for_events = stream_for_task.clone();
        let _unsubscribe = agent.subscribe(move |event| {
            stream_for_events.push(event.clone());
        });

        let result = agent.continue_().await;
        stream_for_task.end(Some(result));
    });

    stream
}

/// Minimal dummy provider used when a custom `stream_fn` is set.
/// This should never actually be called.
struct DummyProvider;

#[async_trait::async_trait]
impl crate::provider::LLMProtocol for DummyProvider {
    fn provider_type(&self) -> Provider {
        Provider::Custom("dummy".to_string())
    }

    fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: StreamOptions,
    ) -> AssistantMessageEventStream {
        let stream = AssistantMessageEventStream::new_assistant_stream();
        let error_msg = AssistantMessage::builder()
            .api(Api::Custom("dummy".to_string()))
            .provider(Provider::Custom("dummy".to_string()))
            .model("dummy")
            .stop_reason(StopReason::Error)
            .error_message("DummyProvider should not be called when stream_fn is set")
            .build()
            .unwrap();
        stream.push(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: error_msg,
        });
        stream.end(None);
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

/// Agent error type.
#[derive(Debug, Clone, thiserror::Error)]
pub enum AgentError {
    #[error("Agent is already streaming")]
    AlreadyStreaming,

    #[error("No messages in context")]
    NoMessages,

    #[error("Cannot continue from assistant message")]
    CannotContinueFromAssistant,

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Provider error: {0}")]
    ProviderError(String),

    #[error("Incomplete {provider} stream: {detail}")]
    IncompleteStream { provider: String, detail: String },

    #[error("Agent reached the maximum turn limit ({0}) before producing a final response")]
    MaxTurnsReached(usize),

    #[error("{0}")]
    Other(String),
}
