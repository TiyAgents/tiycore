# tiycore `reasoning_content_constrained` 泛化改进计划

## Summary

将当前 tiycode 中硬编码的 DeepSeek `reasoning_content` payload 规范化逻辑下沉到 tiycore 内部，由 tiycore 通过 `OpenAICompletionsCompat` 能力标记自行消化哪些 Provider/Model 需要该约束，从而使通过任意第三方 Provider（OpenRouter、ZenMux、硅基流动、自定义 `openai-compatible` 等）访问 DeepSeek 模型的场景也能自动生效。

## Context

### 现状

tiycode 中当前有 9 处与 DeepSeek 相关的硬编码逻辑（详见 `tiycode-deepseek-hardcoded-analysis.md`），其中 **最关键的运行时逻辑** 是 `agent_session.rs` 中的两段代码：

1. **`is_deepseek_provider()`**（L709-710）：通过比较 `provider_type == "deepseek"` 或 `base_url` 包含 `"api.deepseek.com"` 来判断是否需要 payload 规范化。这是硬编码的品牌检测。

2. **`normalize_deepseek_thinking_payload()`**（L727-798）：在 thinking 启用时回填缺失的 `reasoning_content` 并确保 `content` 不为 null；在 thinking 禁用时剥离所有 `reasoning_content`。仅当 `is_deepseek=true` 时执行。

这两段逻辑通过 `agent.set_on_payload()` hook 注入到 tiycore 的请求链路中，在 `protocol/common.rs::apply_on_payload()` 中作为 JSON 后处理步骤执行。Subagent（`subagent/orchestrator.rs` L174-200）**完整复制**了同一套逻辑。

**当前生效范围分析**：

| 场景 | 是否生效 | 原因 |
|------|---------|------|
| 内置 `deepseek` provider | ✅ | `provider_type == "deepseek"` |
| Custom `openai-compatible` 直连 `api.deepseek.com` | ✅ | baseUrl 包含 `api.deepseek.com` |
| 通过 OpenRouter 访问 DeepSeek | ❌ | 两者都不满足 |
| 通过 ZenMux 访问 DeepSeek | ❌ | 两者都不满足 |
| 通过硅基流动 `deepseek-ai/DeepSeek-R1` | ❌ | 两者都不满足 |

### tiycore 现有架构

tiycore 已有 `OpenAICompletionsCompat` 兼容性标记体系（`src/types/model.rs:317-361`），当前包含 13 个字段用于控制不同 Provider 的 API 差异行为：

- `supports_store` / `supports_developer_role` / `supports_reasoning_effort`
- `requires_thinking_as_text` / `thinking_format`
- `requires_tool_result_name` / `requires_assistant_after_tool_result`
- `supports_usage_in_streaming` / `supports_strict_mode`
- `max_tokens_field` / `reasoning_effort_map` / `open_router_routing`

这些 compat 标记由各 Provider 的 `default_compat()` 闭包声明（如 `src/provider/deepseek.rs`），在 `delegation.rs` 的 `stream()` 方法中被注入到 `Model.compat` 字段，协议层在序列化时读取。

**`reasoning_content` 的处理现状**（`src/protocol/openai_completions.rs`）：

- L714-719：`convert_assistant_message()` 中，当 `reasoning_content` 存在但 `content` 为空时，已做 `content: ""` 的防御处理（"for provider compatibility, some providers reject null content with reasoning"）——但这是**无条件对所有 provider 生效的**，不依赖 compat 标记。
- L727-746：thinking 文本通过 `extra_fields` 序列化为 `reasoning_content` 字段（当 `!compat.requires_thinking_as_text` 时）。

**现有 `reasoning_content` 约束中 tiycore 已覆盖的部分**：单个 message 的 content-null 防御已由 L714-719 处理。但**跨消息的 reasoning_content 回填和剥离**——即 `normalize_deepseek_thinking_payload` 的核心逻辑——仍在 tiycode 中。

### 设计约束

1. tiycore 内部消化哪些模型需要 `reasoning_content_constrained`，不由 tiycode 传入
2. 通过中间商访问 DeepSeek 模型的场景也需生效
3. 不能破坏现有 Provider 的行为
4. 减少 tiycode 中的硬编码和代码重复（主 Agent 和 Subagent 中的两处复制）

## Design

### 总体思路

在 `OpenAICompletionsCompat` 中新增 `reasoning_content_constrained: bool` 字段。各 Provider 通过 `default_compat()` 自行声明；对于自定义 `openai-compatible` Provider（无法通过 Provider type 识别），在协议层通过 base_url 启发式检测自动推断该标记。

将 tiycode 中 `normalize_deepseek_thinking_payload` 的逻辑重构为 tiycore 内部的 **消息级（typed message）规范化器**，在 `convert_to_llm` 之后、消息序列化之前执行，而非在 JSON 后处理阶段。这样可以：
- 在类型安全层面操作，避免 JSON 字符串操作
- 自动对所有标记了 `reasoning_content_constrained` 的 Provider 生效
- 不需要 tiycode 调用方关心内部实现

### 架构改动

```
Before:
  tiycode::agent_session::set_on_payload()
    → is_deepseek_provider() // hardcoded brand check
    → normalize_deepseek_thinking_payload() // raw JSON manipulation
    → tiycore::protocol::common::apply_on_payload() // hook invoked

After:
  tiycore::protocol::openai_completions::stream()
    → convert_to_llm() // AgentMessage → Message
    → normalize_reasoning_content() // NEW: typed message normalization
    → build_request() // Message → OpenAIMessage
    → serialize → send
```

### 关键决策

**决策 1：规范化在哪一层执行？**

选择在 `convert_to_llm()` 之后、`build_request()` 之前的**消息级（typed message）层**执行，而非 JSON 后处理层。

理由：
- 类型安全：操作 `Vec<Message>` 比解析/修改 `serde_json::Value` 更安全、更易测试
- 统一入口：所有走 OpenAI Completions 协议的消息都经过同一路径，无需依赖 `set_on_payload` hook
- 可组合：消息级规范化可以与现有 `set_on_messages` hook（`agent/agent.rs:389`）结合

**决策 2：如何让自定义 Provider 也能自动获得标记？**

对于未设置 `reasoning_content_constrained` 的 Provider，在协议层做 base_url 启发式检测：

```rust
fn infer_reasoning_content_constrained(compat: &OpenAICompletionsCompat, base_url: &str) -> bool {
    if compat.reasoning_content_constrained {
        return true;
    }
    // Heuristic: known DeepSeek API endpoints
    base_url.contains("api.deepseek.com")
}
```

这替代了 tiycode 中 `is_deepseek_provider()` 的 base_url 分支。未来如果有其他 Provider 暴露相同约束，可以通过在其 `default_compat()` 中设置标记来覆盖，无需修改代码。

**决策 3：消息规范化的具体行为**

规范化器 `normalize_reasoning_content(messages: Vec<Message>, thinking_enabled: bool) -> Vec<Message>` 的行为：

- **thinking 启用时**：遍历助手消息，追踪最近的非空 `reasoning_content`，对缺失它的助手消息回填；确保每条助手消息的 `content` 不为 null（即使为空字符串）
- **thinking 禁用时**：从所有助手消息中移除 `reasoning_content`

这与当前 `normalize_deepseek_thinking_payload` 的行为一致，但操作对象是 `Vec<Message>` 而非 `serde_json::Value`。

### 边缘情况

1. **多 Provider 混合链路**：如 Primary 模型是 DeepSeek、Auxiliary 是 OpenAI——规范化仅在 Primary 的协议调用中执行，因为 compat 标记随 Model 传递，不影响其他 Provider。

2. **thinking 启用但模型不支持 reasoning**：此时 `normalize_reasoning_content` 不应执行回填逻辑。这由调用方的 `thinking_enabled` 参数控制（`thinking_level != Off && model.reasoning`）。

3. **Base URL 为空或未知**：`infer_reasoning_content_constrained` 的启发式分支仅在 base_url 存在时生效；无 base_url 时依赖 Provider 自行声明的标记。

4. **Streaming 中断导致的空消息**：tiycode 的 `agent_session_history.rs` 中对空 reasoning/plain_message 的跳过逻辑保留在 tiycode 中——这些是历史记录持久化的防御性编码，不属于 tiycore 协议层的职责。

## Key Implementation

### 1. tiycore 改动

#### 文件：`src/types/model.rs`

在 `OpenAICompletionsCompat` 中新增字段：

```rust
pub struct OpenAICompletionsCompat {
    // ... existing fields ...

    /// 该 Provider 要求在 thinking 模式启用时，每条 assistant 消息都必须
    /// 携带 reasoning_content 字段，且 content 不能为 null。
    /// 目前 DeepSeek API 有此严格约束。
    #[serde(default)]
    pub reasoning_content_constrained: bool,
}
```

并在 `Default` 实现中设为 `false`。

#### 文件：`src/provider/deepseek.rs`

在 `default_compat` 闭包中设置标记：

```rust
default_compat: || OpenAICompletionsCompat {
    supports_store: false,
    supports_developer_role: false,
    thinking_format: "openai".to_string(),
    reasoning_content_constrained: true,  // NEW
    ..Default::default()
},
```

#### 文件：`src/protocol/openai_completions.rs`（新增函数）

实现 `normalize_reasoning_content` 函数，在 `convert_to_llm()` 与 `build_request()` 之间调用：

```rust
/// 对 reasoning_content_constrained 的 Provider 执行消息级规范化。
///
/// * thinking_enabled=true：回填缺失的 reasoning_content，确保 content 不为 null
/// * thinking_enabled=false：剥离所有 reasoning_content
fn normalize_reasoning_content(
    messages: Vec<Message>,
    compat: &OpenAICompletionsCompat,
    thinking_enabled: bool,
    base_url: &str,
) -> Vec<Message> {
    let constrained = compat.reasoning_content_constrained
        || base_url.contains("api.deepseek.com");

    if !constrained {
        return messages;
    }

    // ... 规范化逻辑，操作 Vec<Message> ...
}
```

在 `stream()` 和 `stream_simple()` 方法中，于 `convert_to_llm()` 之后、构建 `Context` 之前调用此函数。

#### 文件：`src/protocol/openai_completions.rs`（L714-719 增强）

现有的 content-null 防御（`if thinking_text.is_some() { Some(OpenAIContent::Text(String::new())) }`）已经是无条件执行的，保留不变。这为消息级规范化提供兜底。

#### 文件：`src/protocol/openai_completions.rs`（convert_assistant_message 增强）

在单条消息序列化时，保留现有的 `reasoning_content` extra_fields 逻辑。消息级规范化在序列化之前确保数据一致性，序列化层继续做字段映射。

### 2. tiycode 改动

#### 文件：`src-tauri/src/core/agent_session.rs`

- **移除** `is_deepseek_provider()` 函数（L709-710）
- **移除** `normalize_deepseek_thinking_payload()` 函数（L727-798）
- **简化** `set_on_payload` hook（L483-511）：仅保留 `merge_payload(p, opts)` 逻辑，移除 `is_deepseek_provider` 和 `normalize_deepseek_thinking_payload` 调用

#### 文件：`src-tauri/src/core/subagent/orchestrator.rs`

- **简化** `set_on_payload` hook（L174-200）：同上，移除 DeepSeek 相关逻辑

#### 文件：`src-tauri/src/core/agent_session_tests.rs`

- 移除 `is_deepseek_provider` 测试（L3067-3096）
- 移除 `normalize_deepseek_thinking_payload` 测试（L3099-3260）
- 保留端到端验证测试（`deepseek_payload_uses_non_thinking_mode`），但改为验证 tiycore 内部规范化的行为

#### 文件：`src-tauri/src/core/agent_session_history.rs`

- 保留现有逻辑（跳过空 reasoning、跳过空 plain_message、工具调用合并）——这些是防御性通用编码，对所有 Provider 都有益，且操作在持久化层而非协议层

### 数据流总览

```
tiycode::agent_session::run()
  └─ agent.set_on_payload(merge_payload)         // 仅合并 provider_options
  └─ agent.run(model, context, options)
       └─ tiycore::openai_completions::stream()
            └─ convert_to_llm(messages)          // AgentMessage → Message
            └─ normalize_reasoning_content()     // NEW: compat-driven sanitize
            └─ build_request(messages)            // Message → OpenAIMessage
            └─ apply_on_payload()                 // on_payload hook (merge only)
            └─ serialize → HTTP send
```

## Steps

1. **tiycore：在 `OpenAICompletionsCompat` 中新增 `reasoning_content_constrained` 字段**
   - 文件：`src/types/model.rs`
   - 在结构体末尾添加 `pub reasoning_content_constrained: bool`（带 `#[serde(default)]`）
   - 在 `Default` 实现中添加 `reasoning_content_constrained: false`

2. **tiycore：在 `DeepSeekProvider::default_compat()` 中设置标记为 true**
   - 文件：`src/provider/deepseek.rs`
   - 在 `default_compat` 闭包中添加 `reasoning_content_constrained: true`

3. **tiycore：实现 `normalize_reasoning_content()` 消息级规范化函数**
   - 文件：`src/protocol/openai_completions.rs`
   - 新增函数，操作 `Vec<Message>` 类型而非 `serde_json::Value`
   - thinking 启用时：回填缺失的 reasoning_content，确保 content 不为 null
   - thinking 禁用时：剥离 reasoning_content
   - 包含 base_url 启发式检测作为 fallback

4. **tiycore：在协议流中集成规范化调用**
   - 文件：`src/protocol/openai_completions.rs`
   - 在 `stream()` 和 `stream_simple()` 中，于 `convert_to_llm()` 之后调用 `normalize_reasoning_content()`
   - 将 `thinking_enabled` 参数下传到调用点（从 `StreamOptions.thinking` 或 `Context` 中获取）

5. **tiycore：添加单元测试**
   - 文件：`src/protocol/openai_completions.rs` 测试模块
   - 测试 thinking 启用回填、保留已有 reasoning、content not null、thinking 禁用剥离、非 constrained provider 直通、base_url 启发式检测等场景

6. **tiycode：移除硬编码 DeepSeek 逻辑**
   - 文件：`src-tauri/src/core/agent_session.rs`
   - 移除 `is_deepseek_provider()` 和 `normalize_deepseek_thinking_payload()`
   - 简化 `set_on_payload` hook

7. **tiycode：简化 Subagent payload hook**
   - 文件：`src-tauri/src/core/subagent/orchestrator.rs`
   - 移除 DeepSeek 相关 import 和逻辑

8. **tiycode：更新测试**
   - 文件：`src-tauri/src/core/agent_session_tests.rs`
   - 移除旧的 `normalize_deepseek_thinking_payload` 单元测试
   - 更新端到端测试以验证新的 tiycore 内部规范化行为

## Verification

1. **tiycore 单元测试**：运行 `cargo test --manifest-path /Users/jorbenzhu/Documents/Workplace/TiyAgents/tiycore/Cargo.toml` ——验证 `normalize_reasoning_content` 的所有场景（回填、剥离、直通、base_url 启发式）
2. **tiycore 集成测试**：已有 `tests/test_provider_openai.rs` 中 reasoning_content 相关测试，确认通过
3. **tiycode Rust 测试**：运行 `cargo test --manifest-path src-tauri/Cargo.toml` ——验证 `agent_session_tests` 中修改后的测试通过
4. **tiycode TypeScript 类型检查**：运行 `npm run typecheck` ——验证前端代码无类型错误
5. **手动验证场景**：
   - 使用内置 `deepseek` provider + R1 模型，thinking=medium，发起一次含工具调用的对话 → 确认无 400 错误
   - 使用 custom `openai-compatible` provider 直连 `api.deepseek.com` → 同上
   - 使用 OpenRouter provider + DeepSeek R1 模型 → 确认规范化生效
   - 使用 OpenAI provider（非 constrained） → 确认 reasoning_content 行为不变

## Risks

1. **其他 Provider 的意外影响**：`reasoning_content_constrained` 默认为 `false`，对所有现有 Provider 无影响。风险低。
2. **Base URL 启发式误判**：如果第三方 API 恰好包含 `api.deepseek.com` 子串但并非 DeepSeek API，会被错误标记。概率极低——该域名是 DeepSeek 专有域名。
3. **消息类型重构风险**：从 `serde_json::Value` 操作改为 `Vec<Message>` 类型操作，需要确保 `Message` 类型能完整表达 `reasoning_content` 字段。当前 tiycore 的 `Message`（来自 `crate::types`，转化为 LLM 消息格式的中间表示）已支持 thinking 内容，评估为低风险。
4. **回归风险**：tiycode 中 `agent_session_history.rs` 的防御性逻辑（跳过空消息、工具调用合并）依赖 `normalize_deepseek_thinking_payload` 作为最后一层安全网。如果 tiycore 的消息级规范化不能完全覆盖历史重建中的边缘情况，可能仍有 400 错误。需要保留历史重建层的防御逻辑不变，并确保测试覆盖。
5. **thinking_enabled 参数传递**：当前 `stream()` 方法从 `StreamOptions` 获取 thinking 配置。需要确认 `thinking_enabled` 信息在协议层可用。

## Assumptions

1. tiycore 的 `Message` 类型（转化后的 LLM 中间表示）有足够的字段表达 `reasoning_content` 的完整信息，包括跨消息的回填场景。
2. 自定义 Provider 的 base_url 启发式检测（`contains("api.deepseek.com")`）覆盖了绝大多数第三方直连 DeepSeek API 的场景。对于通过中间商的场景，依赖中间商 Provider 自身声明 compat 标记（如 OpenRouter 已有 `open_router_routing` 机制）。
3. tiycode 的 `normalize_deepseek_thinking_payload` 可以在 tiycore 更新后同步移除，不涉及 API 兼容性破坏（这些函数都是 `pub(crate)` 的）。
