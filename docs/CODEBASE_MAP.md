# 📦 TiYCore Codebase - Complete Comprehensive Map

## Quick Facts
- **Project**: Unified LLM API and stateful Agent runtime (Rust)
- **Version**: 0.1.0 | **Edition**: Rust 2021 | **License**: MIT
- **Size**: ~21,000 lines of code across 54 Rust files
- **Providers**: 18+ LLM providers supported
- **Main Purpose**: Unified interface for multiple LLM APIs + stateful agent runtime

---

## 🏗️ Architecture Overview

```
                        ┌─────────────────────┐
                        │   Application Code  │
                        └──────────┬──────────┘
                                   │
        ┌──────────────────────────┴──────────────────────┐
        │                                                  │
        ▼                                                  ▼
   ┌─────────────────┐                         ┌──────────────────┐
   │   Agent Loop    │                         │ Direct Protocol  │
   │  (high-level)   │                         │   Usage          │
   └────────┬────────┘                         └────────┬─────────┘
            │                                           │
            └──────────────┬────────────────────────────┘
                           ▼
                  ┌──────────────────────┐
                  │ Provider Registry &  │
                  │ Facades (18+ options)│
                  └────────────┬─────────┘
                               ▼
                  ┌──────────────────────┐
                  │ Protocol Layer       │
                  │ (OpenAI, Anthropic,  │
                  │  Google, etc.)       │
                  └────────────┬─────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
    [OpenAI]          [Anthropic]             [Google]
    API Servers       API Servers             API Servers

Supporting Modules:
├── transform/ ───── Message/Context conversion
├── stream/ ─────── Streaming & event parsing
├── validation/ ── Parameter & schema validation
├── thinking/ ───── Extended reasoning support
└── types/ ──────── Unified data structures
```

---

## 📂 Directory Structure

### Root Level Files
- **Cargo.toml** — Project manifest with ~25 dependencies
- **Cargo.lock** — Reproducible build lock file
- **README.md** — Main documentation
- **README-ZH.md** — Chinese documentation
- **AGENTS.md** — Agent framework documentation
- **LICENSE** — MIT License
- **CHANGELOG.md** — Version history

### Main Source Directory: src/

```
src/
├── lib.rs ····················· Main crate entry (47 lines)
│
├── types/ ···················· Core type definitions (8 files)
│   ├── context.rs ·········· Context, Tool types
│   ├── message.rs ·········· Message hierarchy (User, Assistant, ToolResult)
│   ├── content.rs ·········· ContentBlock (Text, Image, Thinking, ToolCall)
│   ├── model.rs ············ Model, Provider, Api (via define_string_enum! macro), Cost, OpenAICompletionsCompat (CompatCapabilities, CompatThinking, CompatMessageFormat)
│   ├── usage.rs ············ Token usage metrics
│   ├── events.rs ··········· Event stream types
│   └── limits.rs ··········· Rate limiting configs
│
├── protocol/ ·················· Wire protocol implementations (7 files)
│   ├── traits.rs ············ LLMProtocol trait
│   ├── common.rs ··········· Shared utilities
│   ├── openai_completions.rs
│   ├── openai_responses.rs ·· OpenAI Responses API (beta)
│   ├── anthropic.rs ········· Anthropic Messages API
│   └── google.rs ··········· Google Generative AI
│
├── provider/ ················· Provider facades & registry (20 files)
│   ├── registry.rs ········· Global provider registry
│   ├── delegation.rs ······· Delegation macros
│   ├── openai.rs ··········· OpenAI facade
│   ├── openai_compatible.rs · Ollama, Mistral, etc.
│   ├── anthropic.rs ········· Anthropic facade
│   ├── google.rs ··········· Google facade
│   ├── ollama.rs ··········· Ollama facade
│   ├── xai.rs ·············· X.AI / Grok
│   ├── groq.rs ·············· Groq
│   ├── openrouter.rs ········ OpenRouter
│   ├── minimax.rs ·········· MiniMax (Chinese)
│   ├── kimi_coding.rs ······· Kimi Coding
│   ├── zai.rs ·············· ZAI (Chinese)
│   ├── deepseek.rs ········· DeepSeek (Chinese)
│   ├── xiaomi_mimo.rs ······· Xiaomi MIMO
│   ├── zenmux.rs ··········· Zenmux (image gen)
│   ├── bai.rs ·············· BAI provider
│   └── opencode_go.rs ······· OpenCodeGo
│
├── agent/ ····················· Stateful agent runtime (5 files)
│   ├── agent.rs ············ Main Agent impl & agent_loop()
│   ├── queue.rs ············ Unified message queue (MessageQueue, DrainStrategy, BackpressureConfig)
│   ├── state.rs ············ AgentState & snapshots
│   └── types.rs ············ Agent-specific types (QueueMode, QueueEvent, SupplierContext, etc.)
│
├── stream/ ····················· Streaming utilities (3 files)
│   ├── event_stream.rs ····· EventStream type
│   └── json_parser.rs ····· JSON parsing for streams
│
├── transform/ ················· Message transformation (3 files)
│   ├── messages.rs ········· Message transformations
│   └── tool_calls.rs ······· Tool call extraction
│
├── validation/ ················ Input validation (2 files)
│   └── tool_validation.rs ··· JSON Schema validation
│
├── thinking/ ················· Extended thinking (2 files)
│   └── config.rs ··········· Thinking config
│
├── models/ ··················· Predefined models (2 files)
│   └── predefined.rs ······· 100+ model definitions
│
├── catalog/ ··················· Model catalog system (2 files)
│   └── README.md ··········· Catalog documentation
│
└── bin/ ······················ CLI tools (1 file)
    └── tiy-catalog-sync.rs · Catalog sync tool
```

---

## 📊 Module Breakdown

### types/ (8 files) — Core Data Structures
**Responsibility**: Unified types for all LLM communication

- `context.rs` — Context struct with system prompt, messages, tools
- `message.rs` — Message types (User, Assistant, ToolResult) and roles
- `content.rs` — ContentBlock enum (Text, Thinking, Image, ToolCall)
- `model.rs` — Model metadata, Provider and Api enums (generated by `define_string_enum!` macro), Cost types, OpenAICompletionsCompat (split into CompatCapabilities, CompatThinking, CompatMessageFormat sub-structs)
- `usage.rs` — Token usage tracking
- `events.rs` — Streaming event types
- `limits.rs` — Rate limiting and stream limiting configs

### protocol/ (7 files) — Wire Protocol Implementations
**Responsibility**: Low-level API protocol adapters

- `traits.rs` — LLMProtocol trait (abstract interface for all protocols)
- `common.rs` — Shared utilities across protocols
- `openai_completions.rs` — OpenAI Chat Completions API
- `openai_responses.rs` — OpenAI Responses API (new beta)
- `anthropic.rs` — Anthropic Messages API
- `google.rs` — Google Generative AI & Vertex AI

### provider/ (20 files) — Provider Facades & Registry
**Responsibility**: High-level provider interfaces + auto-registry

**Core Infrastructure:**
- `registry.rs` — ProtocolRegistry with auto-registration
- `delegation.rs` — Delegation macros

**Provider Facades:**
- `openai.rs`, `openai_compatible.rs`, `openai_responses.rs`
- `anthropic.rs`, `google.rs`, `ollama.rs`, `xai.rs`, `groq.rs`
- `openrouter.rs`, `minimax.rs`, `kimi_coding.rs`, `zai.rs`
- `deepseek.rs`, `xiaomi_mimo.rs`, `zenmux.rs`, `bai.rs`, `opencode_go.rs`

### agent/ (5 files) — Stateful Agent Runtime
**Responsibility**: Multi-turn conversations with tool execution, steering/follow-up queues

- `agent.rs` — Agent struct, agent_loop(), event streaming, steering/follow-up consumption
- `queue.rs` — MessageQueue (unified FIFO buffer + supplier), DrainStrategy trait, BackpressureConfig
- `state.rs` — AgentState, AgentStateSnapshot (for persistence)
- `types.rs` — Agent-specific event types, QueueMode, QueueEvent, QueueKind, QueueStats, SupplierContext

### stream/ (3 files) — Streaming Utilities
**Responsibility**: SSE parsing and event buffering

- `event_stream.rs` — EventStream type for streaming responses (backed by `parking_lot::Mutex<VecDeque>` + `tokio::sync::Notify`)
- `json_parser.rs` — Incremental JSON parsing

### transform/ (3 files) — Message Transformation
**Responsibility**: Convert between tiycore types and provider formats

- `messages.rs` — Context → request, response → Message
- `tool_calls.rs` — Tool call extraction and formatting

### validation/ (2 files) — Input Validation
**Responsibility**: JSON Schema validation for tools

- `tool_validation.rs` — Tool definition and parameter validation

### thinking/ (2 files) — Extended Thinking Support
**Responsibility**: Configuration for reasoning models

- `config.rs` — Thinking behavior config (Claude Opus 4, OpenAI o1, etc.)

### models/ (2 files) — Predefined Models
**Responsibility**: Model metadata catalog

- `predefined.rs` — 100+ hardcoded model definitions with costs and limits

### catalog/ (2 files) — Model Catalog System
**Responsibility**: Dynamic catalog management

- `mod.rs` — Catalog snapshots, manifests, remote updates, enrichment

---

## 🔌 Key APIs

### Core Types
```rust
// Context & Tools
pub struct Context {
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Option<Vec<Tool>>,
}

pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,  // JSON Schema
}

// Messages
pub enum Message {
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
}

// Content Blocks
pub enum ContentBlock {
    Text(TextContent),
    Thinking(ThinkingContent),
    Image(ImageContent),
    ToolCall(ToolCall),
}

// Model Configuration
pub struct Model {
    pub id: String,
    pub name: String,
    pub provider: Provider,
    pub context_window: usize,
    pub max_tokens: usize,
    // ... more fields
}
```

### Protocol Trait
```rust
pub trait LLMProtocol: Send + Sync {
    fn build_request(&self, context: &Context, model: &Model) 
        -> Result<Request>;
    fn parse_response(&self, body: &str) 
        -> Result<Response>;
    fn parse_streaming(&self, chunk: &str) 
        -> Result<Vec<Event>>;
    fn get_name(&self) -> &str;
}
```

### Provider Registry
```rust
pub fn register_provider(provider: Arc<dyn LLMProtocol>);
pub fn get_provider(model: &Model) -> Result<ArcProtocol>;
pub fn get_registered_providers() -> Vec<String>;
pub fn register_all_providers();  // Auto-register all built-in providers
```

### Agent API
```rust
pub async fn agent_loop(
    model: Model,
    context: Context,
) -> Result<AgentEventStream>;

pub async fn agent_loop_continue(
    agent_state: AgentStateSnapshot,
) -> Result<AgentEventStream>;

pub struct AgentEventStream {
    // Streaming iterator of agent events
}

// Steering & Follow-up (runtime message injection)
impl Agent {
    pub fn steer(&self, msg: AgentMessage);              // Mid-stream interrupt
    pub fn follow_up(&self, msg: AgentMessage);          // Post-turn append
    pub fn try_steer(&self, msg: AgentMessage) -> Result<(), QueueFullError>;
    pub fn has_queued_messages(&self) -> bool;            // Sync local check
    pub async fn has_queued_messages_async(&self) -> bool; // Async with supplier probe
    pub fn queue_stats(&self) -> QueueStats;
    pub fn set_steering_mode(&self, mode: QueueMode);
    pub fn set_on_queue_event<F>(&self, handler: F);
    pub fn set_steering_supplier<F, Fut>(&self, supplier: F); // V2 with SupplierContext
    pub fn set_steering_backpressure(&self, config: BackpressureConfig);
}
```

---

## 📈 Statistics

| Category | Count | Details |
|----------|-------|---------|
| **Total .rs Files** | 55 | Including lib.rs, bin, src/types/protocol/provider/agent/etc |
| **Total Lines of Code** | ~21,000 | Across all Rust source files |
| **Main Modules** | 11 | types, protocol, provider, agent, stream, transform, validation, thinking, models, catalog, bin |
| **Supported Providers** | 18+ | OpenAI, Anthropic, Google, Ollama, Groq, etc. |
| **Protocol Implementations** | 4 | OpenAI Completions, OpenAI Responses, Anthropic, Google |
| **Dependencies** | ~25 | tokio, reqwest, serde, futures, jsonschema, etc. |
| **Predefined Models** | 100+ | Model metadata with costs, limits, capabilities |

---

## 🚀 Usage Examples

### High-Level: Agent Loop
```rust
use tiycore::agent::agent_loop;
use tiycore::types::{Model, Context, Provider};

let model = Model {
    id: "gpt-4o-mini".to_string(),
    name: "GPT-4o Mini".to_string(),
    provider: Provider::OpenAI,
    // ... more config
};

let mut context = Context::with_system_prompt("You are helpful");
context.user("What is 2+2?");

let mut stream = agent_loop(model, context).await?;
while let Some(event) = stream.next().await {
    println!("{:?}", event);
}
```

### Mid-Level: Provider Facade
```rust
use tiycore::provider::get_provider;

let provider = get_provider(&model)?;
let request = provider.build_request(&context)?;
// Make HTTP request with your own client
```

### Low-Level: Protocol
```rust
use tiycore::protocol::{openai_completions::*, LLMProtocol};

let protocol = OpenAICompletionsProtocol::new();
let request = protocol.build_request(&context, &model)?;
let response = protocol.parse_response(&response_body)?;
```

---

## 🎯 Reading Recommendations

### To Understand Architecture
1. src/lib.rs — Public API surface
2. src/types/context.rs — Core Context type
3. src/protocol/traits.rs — Abstract LLMProtocol trait
4. src/provider/registry.rs — Provider registration system
5. src/agent/agent.rs — Agent event loop

### To Add a New Provider
1. src/provider/openai.rs — Example implementation
2. src/protocol/traits.rs — Protocol trait
3. src/protocol/openai_completions.rs — Example protocol
4. src/provider/mod.rs — How to register

### For Tool Integration
1. src/types/content.rs — ToolCall definition
2. src/validation/tool_validation.rs — Validation
3. src/agent/agent.rs — Tool call handling
4. src/transform/tool_calls.rs — Tool extraction

### For Streaming
1. src/stream/mod.rs — Stream utilities
2. src/stream/event_stream.rs — EventStream type
3. src/protocol/* — Protocol streaming implementations
4. src/agent/agent.rs — Agent streaming integration

---

## 🏆 Key Strengths

1. ✅ **Unified API** — Single codebase supports 18+ providers
2. ✅ **Protocol Abstraction** — Add new APIs without modifying core logic
3. ✅ **Type Safety** — Strong typing prevents runtime errors
4. ✅ **Async Native** — All I/O is non-blocking with tokio
5. ✅ **Streaming First** — Built-in streaming event support
6. ✅ **Tool Support** — JSON Schema validation + execution
7. ✅ **Extended Thinking** — Support for reasoning models
8. ✅ **Persistence Ready** — Agent state snapshots for recovery

---

## 💻 Dependencies

**Key Crates:**
- tokio (async runtime)
- reqwest (HTTP client)
- serde/serde_json (serialization)
- jsonschema (validation)
- futures (async utilities)
- parking_lot (concurrency)
- chrono (date/time)
- uuid (ID generation)
- tracing (logging)
- anyhow/thiserror (error handling)

---

## 📄 File Count Summary

- **src/lib.rs** — 1 file
- **src/types/** — 8 files
- **src/protocol/** — 7 files
- **src/provider/** — 20 files
- **src/agent/** — 5 files
- **src/stream/** — 3 files
- **src/transform/** — 3 files
- **src/validation/** — 2 files
- **src/thinking/** — 2 files
- **src/models/** — 2 files
- **src/catalog/** — 2 files
- **src/bin/** — 1 file

**Total: 54 Rust source files, ~21,000 lines of code**

---

*This map was generated on 2026-05-12 for TiYCore v0.1.0*
*A unified LLM API and stateful Agent runtime in Rust*
