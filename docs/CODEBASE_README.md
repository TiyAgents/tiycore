# Tiycore Codebase Comprehensive Map

> A complete, detailed map of the Tiycore project structure, architecture, and key components.

## 📖 Documentation Files

This directory contains three comprehensive reference documents:

### 1. **CODEBASE_MAP.md** (18KB) - Deep Dive Reference
The most detailed and comprehensive guide. Includes:
- Complete module descriptions with line counts
- Detailed type specifications for all key structs/enums
- Full file organization by concern
- Architecture patterns explanation
- Code statistics table
- Quick navigation guide

**Use this when:** You need to understand how types work, find specific functionality, or understand design patterns.

### 2. **CODEBASE_SUMMARY.txt** (20KB) - Quick Reference
A structured text format with ASCII formatting. Includes:
- Directory tree visualization
- All 51 Rust files listed with descriptions
- Key types organized by module
- Architecture patterns summary
- Data flow diagram
- Quick navigation table

**Use this when:** You want a printable reference or quick visual scan of the codebase.

### 3. **CODEBASE_MODULES.txt** (21KB) - Module Relationships
Visual diagram showing how modules relate and depend on each other. Includes:
- Library interface overview
- Module dependency graph
- Type hierarchies within each module
- Key patterns and design insights
- Dependency flow explanation

**Use this when:** You're learning the overall architecture or tracing data flow.

---

## 🎯 Quick Facts

| Metric | Value |
|--------|-------|
| **Total Lines of Code** | ~21,000 Rust LOC |
| **Source Files** | 55 .rs files |
| **Main Modules** | 10 (types, agent, protocol, provider, stream, transform, catalog, models, thinking, validation) |
| **Supported Providers** | 17+ LLM providers |
| **Language** | Rust 2021 Edition |
| **Dependencies** | 20+ production crates |
| **Test Files** | 20+ integration tests |

---

## 📁 Module Quick Reference

```
src/
├── lib.rs (82 LOC)              ← Library root
├── types/ (7 files, 1,800 LOC)  ← Core data model: Context, Message, ContentBlock, Tool
├── agent/ (5 files, 3,200 LOC)  ← Stateful agent runtime + queue system
├── protocol/ (7 files, 2,000 LOC) ← Wire format implementations (OpenAI, Anthropic, Google)
├── provider/ (18 files, 3,000 LOC) ← Provider facades + registry
├── stream/ (3 files, 500 LOC)   ← Event streaming (SSE/WebSocket)
├── transform/ (3 files, 400 LOC) ← Message transformation
├── catalog/ (1 file, 1,000 LOC) ← Model catalog management
├── thinking/ (2 files, 300 LOC) ← Extended thinking support
├── validation/ (2 files, 200 LOC) ← JSON Schema validation
└── models/ (2 files, 500 LOC)   ← Predefined model configurations
```

---

## 🔑 Core Type System

### Foundation: `types/context.rs`
- **Context** - Conversation state (system prompt, messages, tools)
- **Tool** - Tool definition with JSON Schema parameters
- **StreamOptions** - Configuration (temperature, max_tokens, api_key, etc.)

### Messages: `types/message.rs`
- **UserMessage** - User input with timestamp
- **AssistantMessage** - LLM response with metadata
- **ToolResultMessage** - Tool execution result
- **Message** - Enum union of all message types

### Content: `types/content.rs`
- **TextContent** - Text response
- **ThinkingContent** - Extended thinking/reasoning
- **ImageContent** - Image data (base64)
- **ToolCall** - Tool invocation
- **ContentBlock** - Enum union of content types

---

## 🏗️ Architecture Patterns

### 1. Protocol-Provider Separation
```
protocol/openai_completions.rs ────┐
protocol/anthropic.rs              ├─→ Implements LLMProtocol trait
protocol/google.rs                 │
protocol/openai_responses.rs ──────┘

provider/openai.rs ────────┐
provider/anthropic.rs      ├─→ Wraps protocol, provides facade + registry
provider/google.rs ────────┘
```

### 2. Type-Driven Design
All modules build around the core type system:
- Types define the contract
- Protocols transform wire formats to types
- Providers expose type-based APIs
- Agent orchestrates with types

### 3. Streaming Pipeline
```
HTTP/WebSocket ──→ EventStream ──→ JSON Parser ──→ Transform ──→ Types ──→ Agent
```

### 4. Auto-Registration Registry
```
get_provider("openai") ──→ Provider Registry ──→ [cached or created on demand]
```

---

## 📍 Navigation Guide

### If you need to...

**Understand conversation state**
→ See `src/types/context.rs` for Context and Tool types

**Handle messages**
→ See `src/types/message.rs` for all message types

**Work with content**
→ See `src/types/content.rs` for ContentBlock and variants

**Add a new LLM provider**
→ Implement LLMProtocol trait in `src/protocol/`
→ Create facade in `src/provider/`

**Build an agent**
→ See `src/agent/agent.rs` for Agent type and functions

**Stream events**
→ See `src/stream/event_stream.rs` for EventStream<T>

**Transform data formats**
→ See `src/transform/messages.rs` and `tool_calls.rs`

**Validate tool calls**
→ See `src/validation/tool_validation.rs` for JSON Schema validation

**Query model catalog**
→ See `src/catalog/mod.rs` for catalog management

**Configure extended thinking**
→ See `src/thinking/config.rs` for ThinkingLevel and ThinkingDisplay

---

## 🔄 Data Flow Example

```
┌─────────────────────────────────────────────────────────────┐
│                      User Code                              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Create StreamOptions (config)                              │
│  provider::openai::OpenAIProvider::new()                    │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Provider facade delegates to:                              │
│  protocol::openai_completions::OpenAICompletionsProtocol   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  HTTP Request via reqwest                                   │
│  OpenAI API /chat/completions endpoint                      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  EventStream (SSE or WebSocket)                             │
│  stream::event_stream::EventStream<T>                       │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  JSON Parser & Transform                                    │
│  stream::json_parser + transform::messages                  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Unified Types                                              │
│  Message, AssistantMessage, ContentBlock, ToolCall         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Agent Loop                                                 │
│  agent::agent::Agent + tool execution                       │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Tool Validation & Execution                                │
│  validation::tool_validation + custom tool handlers         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Response/Callback to User                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔌 Dependencies Overview

**Async Runtime:**
- `tokio` - Full async runtime with I/O, timers, sync primitives

**HTTP:**
- `reqwest` - HTTP client with streaming support
- `reqwest-eventsource` - SSE support

**Serialization:**
- `serde` + `serde_json` - Serialization framework

**Schema & Validation:**
- `jsonschema` - JSON Schema validation
- `schemars` - Schema generation from types

**Streaming:**
- `futures` - Async utilities
- `async-stream` - Async stream macros
- `pin-project-lite` - Pin utilities

**Concurrency:**
- `parking_lot` - Efficient synchronization
- `arc-swap` - Atomic reference swap

**Error Handling:**
- `thiserror` - Error type derivation
- `anyhow` - Error context

**Utilities:**
- `chrono` - DateTime handling
- `uuid` - UUID generation
- `url` - URL parsing
- `sha2` - Hashing

---

## 💡 Key Design Insights

1. **Unified API** - Single interface abstracts 17+ LLM providers
2. **Streaming-First** - Built for real-time, event-based responses
3. **Type-Safe** - Strong typing with comprehensive enums prevents errors
4. **Extensible** - Easy to add new providers via `LLMProtocol` trait
5. **Agent-Ready** - Built-in support for stateful agent loops
6. **Tool-Aware** - First-class support for function calling with validation
7. **Extended Thinking** - Support for reasoning/thinking models
8. **Multi-Transport** - Both SSE and WebSocket support
9. **Error-Rich** - Detailed error types with context
10. **Performance** - Efficient async/await and streaming processing

---

## 🧪 Testing & Examples

**Test Coverage:**
- 20+ integration tests in `tests/` directory
- Provider-specific tests for each LLM service
- Stream, transform, type, validation, thinking tests

**Examples:**
- `examples/basic_usage.rs` - Basic API usage
- `examples/agent_example.rs` - Agent loop example

---

## 🚀 Getting Started

### To understand the basics:
1. Read **CODEBASE_SUMMARY.txt** for directory overview
2. Look at `src/types/context.rs`, `message.rs`, `content.rs`
3. Read `src/agent/agent.rs` for agent patterns

### To add a new provider:
1. Implement `LLMProtocol` trait in `src/protocol/{name}.rs`
2. Create facade in `src/provider/{name}.rs`
3. Add to provider registration in `src/provider/mod.rs`

### To extend message handling:
1. Add new `ContentBlock` variant in `src/types/content.rs`
2. Update transform logic in `src/transform/`
3. Update protocol implementations

---

## 📚 Document Map

| File | Purpose | Best For |
|------|---------|----------|
| CODEBASE_MAP.md | Detailed reference | Understanding types, finding code |
| CODEBASE_SUMMARY.txt | Quick reference | Printing, quick lookup |
| CODEBASE_MODULES.txt | Architecture | Understanding relationships |
| This file (CODEBASE_README.md) | Navigation guide | Getting started |

---

## 📊 File Statistics

- **Main Library:** ~21,000 Rust LOC across 51 files
- **Tests:** ~20 integration test files
- **Examples:** 2 example files
- **Documentation:** This codebase map + README.md + AGENTS.md + CHANGELOG.md

---

## ✨ Key Takeaways

Tiycore is a **well-architected, type-safe Rust library** for building LLM applications with:
- A unified API for 17+ providers
- Streaming-first event-based responses
- Strong type system for compile-time safety
- Built-in agent runtime for stateful loops
- First-class tool/function calling support
- Extensible protocol system for adding providers

The codebase demonstrates **excellent Rust patterns**:
- Trait-based abstraction (LLMProtocol)
- Registry pattern for extensibility
- Builder pattern for complex types
- Strong error handling with thiserror
- Async/await throughout

---

Generated: May 12, 2026
Total Map Size: ~60KB of comprehensive documentation
