//! Model and Provider type definitions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Known API types.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Api {
    /// OpenAI Chat Completions API.
    #[serde(rename = "openai-completions")]
    OpenAICompletions,
    /// Mistral Conversations API.
    #[serde(rename = "mistral-conversations")]
    MistralConversations,
    /// OpenAI Responses API (new).
    #[serde(rename = "openai-responses")]
    OpenAIResponses,
    /// Azure OpenAI Responses API.
    #[serde(rename = "azure-openai-responses")]
    AzureOpenAIResponses,
    /// OpenAI Codex Responses API.
    #[serde(rename = "openai-codex-responses")]
    OpenAICodexResponses,
    /// Anthropic Messages API.
    #[serde(rename = "anthropic-messages")]
    AnthropicMessages,
    /// AWS Bedrock Converse Stream API.
    #[serde(rename = "bedrock-converse-stream")]
    BedrockConverseStream,
    /// Google Generative AI API.
    #[serde(rename = "google-generative-ai")]
    GoogleGenerativeAi,
    /// Google Gemini CLI API.
    #[serde(rename = "google-gemini-cli")]
    GoogleGeminiCli,
    /// Google Vertex AI API.
    #[serde(rename = "google-vertex")]
    GoogleVertex,
    /// Ollama API (OpenAI compatible).
    #[serde(rename = "ollama")]
    Ollama,
    /// Custom API type.
    Custom(String),
}

impl Api {
    /// Get the string representation of this API type.
    pub fn as_str(&self) -> &str {
        match self {
            Api::OpenAICompletions => "openai-completions",
            Api::MistralConversations => "mistral-conversations",
            Api::OpenAIResponses => "openai-responses",
            Api::AzureOpenAIResponses => "azure-openai-responses",
            Api::OpenAICodexResponses => "openai-codex-responses",
            Api::AnthropicMessages => "anthropic-messages",
            Api::BedrockConverseStream => "bedrock-converse-stream",
            Api::GoogleGenerativeAi => "google-generative-ai",
            Api::GoogleGeminiCli => "google-gemini-cli",
            Api::GoogleVertex => "google-vertex",
            Api::Ollama => "ollama",
            Api::Custom(s) => s.as_str(),
        }
    }

    /// Check if this is an OpenAI-compatible API.
    pub fn is_openai_compatible(&self) -> bool {
        matches!(
            self,
            Api::OpenAICompletions | Api::Ollama | Api::MistralConversations | Api::OpenAIResponses
        )
    }
}

impl std::fmt::Display for Api {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<String> for Api {
    fn from(s: String) -> Self {
        match s.as_str() {
            "openai-completions" => Api::OpenAICompletions,
            "mistral-conversations" => Api::MistralConversations,
            "openai-responses" => Api::OpenAIResponses,
            "azure-openai-responses" => Api::AzureOpenAIResponses,
            "openai-codex-responses" => Api::OpenAICodexResponses,
            "anthropic-messages" => Api::AnthropicMessages,
            "bedrock-converse-stream" => Api::BedrockConverseStream,
            "google-generative-ai" => Api::GoogleGenerativeAi,
            "google-gemini-cli" => Api::GoogleGeminiCli,
            "google-vertex" => Api::GoogleVertex,
            "ollama" => Api::Ollama,
            _ => Api::Custom(s),
        }
    }
}

/// Known provider types.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Provider {
    /// Amazon Bedrock.
    #[serde(rename = "amazon-bedrock")]
    AmazonBedrock,
    /// Anthropic.
    #[serde(rename = "anthropic")]
    Anthropic,
    /// Google.
    #[serde(rename = "google")]
    Google,
    /// Google Gemini CLI.
    #[serde(rename = "google-gemini-cli")]
    GoogleGeminiCli,
    /// Google Antigravity.
    #[serde(rename = "google-antigravity")]
    GoogleAntigravity,
    /// Google Vertex AI.
    #[serde(rename = "google-vertex")]
    GoogleVertex,
    /// OpenAI.
    #[serde(rename = "openai")]
    OpenAI,
    /// Generic OpenAI-compatible provider facade.
    #[serde(rename = "openai-compatible")]
    OpenAICompatible,
    /// OpenAI Responses API.
    #[serde(rename = "openai-responses")]
    OpenAIResponses,
    /// Azure OpenAI Responses.
    #[serde(rename = "azure-openai-responses")]
    AzureOpenAIResponses,
    /// OpenAI Codex.
    #[serde(rename = "openai-codex")]
    OpenAICodex,
    /// GitHub Copilot.
    #[serde(rename = "github-copilot")]
    GitHubCopilot,
    /// xAI.
    #[serde(rename = "xai")]
    XAI,
    /// Groq.
    #[serde(rename = "groq")]
    Groq,
    /// Cerebras.
    #[serde(rename = "cerebras")]
    Cerebras,
    /// OpenRouter.
    #[serde(rename = "openrouter")]
    OpenRouter,
    /// Vercel AI Gateway.
    #[serde(rename = "vercel-ai-gateway")]
    VercelAiGateway,
    /// ZAI.
    #[serde(rename = "zai")]
    ZAI,
    /// Mistral.
    #[serde(rename = "mistral")]
    Mistral,
    /// MiniMax.
    #[serde(rename = "minimax")]
    MiniMax,
    /// MiniMax CN.
    #[serde(rename = "minimax-cn")]
    MiniMaxCN,
    /// HuggingFace.
    #[serde(rename = "huggingface")]
    HuggingFace,
    /// OpenCode.
    #[serde(rename = "opencode")]
    OpenCode,
    /// OpenCode Go.
    #[serde(rename = "opencode-go")]
    OpenCodeGo,
    /// Kimi Coding.
    #[serde(rename = "kimi-coding")]
    KimiCoding,
    /// DeepSeek.
    #[serde(rename = "deepseek")]
    DeepSeek,
    /// Xiaomi MiMo.
    #[serde(rename = "xiaomi-mimo")]
    XiaomiMIMO,
    /// Zenmux.
    #[serde(rename = "zenmux")]
    Zenmux,
    /// BAI.
    #[serde(rename = "bai")]
    Bai,
    /// Ollama.
    #[serde(rename = "ollama")]
    Ollama,
    /// Custom provider.
    Custom(String),
}

impl Provider {
    /// Get the string representation of this provider.
    pub fn as_str(&self) -> &str {
        match self {
            Provider::AmazonBedrock => "amazon-bedrock",
            Provider::Anthropic => "anthropic",
            Provider::Google => "google",
            Provider::GoogleGeminiCli => "google-gemini-cli",
            Provider::GoogleAntigravity => "google-antigravity",
            Provider::GoogleVertex => "google-vertex",
            Provider::OpenAI => "openai",
            Provider::OpenAICompatible => "openai-compatible",
            Provider::OpenAIResponses => "openai-responses",
            Provider::AzureOpenAIResponses => "azure-openai-responses",
            Provider::OpenAICodex => "openai-codex",
            Provider::GitHubCopilot => "github-copilot",
            Provider::XAI => "xai",
            Provider::Groq => "groq",
            Provider::Cerebras => "cerebras",
            Provider::OpenRouter => "openrouter",
            Provider::VercelAiGateway => "vercel-ai-gateway",
            Provider::ZAI => "zai",
            Provider::Mistral => "mistral",
            Provider::MiniMax => "minimax",
            Provider::MiniMaxCN => "minimax-cn",
            Provider::HuggingFace => "huggingface",
            Provider::OpenCode => "opencode",
            Provider::OpenCodeGo => "opencode-go",
            Provider::KimiCoding => "kimi-coding",
            Provider::DeepSeek => "deepseek",
            Provider::XiaomiMIMO => "xiaomi-mimo",
            Provider::Zenmux => "zenmux",
            Provider::Bai => "bai",
            Provider::Ollama => "ollama",
            Provider::Custom(s) => s.as_str(),
        }
    }
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<String> for Provider {
    fn from(s: String) -> Self {
        match s.as_str() {
            "amazon-bedrock" => Provider::AmazonBedrock,
            "anthropic" => Provider::Anthropic,
            "google" => Provider::Google,
            "google-gemini-cli" => Provider::GoogleGeminiCli,
            "google-antigravity" => Provider::GoogleAntigravity,
            "google-vertex" => Provider::GoogleVertex,
            "openai" => Provider::OpenAI,
            "openai-compatible" => Provider::OpenAICompatible,
            "openai-responses" => Provider::OpenAIResponses,
            "azure-openai-responses" => Provider::AzureOpenAIResponses,
            "openai-codex" => Provider::OpenAICodex,
            "github-copilot" => Provider::GitHubCopilot,
            "xai" => Provider::XAI,
            "groq" => Provider::Groq,
            "cerebras" => Provider::Cerebras,
            "openrouter" => Provider::OpenRouter,
            "vercel-ai-gateway" => Provider::VercelAiGateway,
            "zai" => Provider::ZAI,
            "mistral" => Provider::Mistral,
            "minimax" => Provider::MiniMax,
            "minimax-cn" => Provider::MiniMaxCN,
            "huggingface" => Provider::HuggingFace,
            "opencode" => Provider::OpenCode,
            "opencode-go" => Provider::OpenCodeGo,
            "kimi-coding" => Provider::KimiCoding,
            "deepseek" => Provider::DeepSeek,
            "xiaomi-mimo" => Provider::XiaomiMIMO,
            "zenmux" => Provider::Zenmux,
            "bai" => Provider::Bai,
            "ollama" => Provider::Ollama,
            _ => Provider::Custom(s),
        }
    }
}

/// Input type supported by a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputType {
    /// Text input.
    Text,
    /// Image input.
    Image,
}

/// Cost configuration for a model (in USD per million tokens).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Cost {
    /// Cost per million input tokens.
    pub input: f64,
    /// Cost per million output tokens.
    pub output: f64,
    /// Cost per million cached tokens read.
    pub cache_read: f64,
    /// Cost per million tokens written to cache.
    pub cache_write: f64,
}

impl Default for Cost {
    fn default() -> Self {
        Self {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
        }
    }
}

impl Cost {
    /// Create a new cost configuration.
    pub fn new(input: f64, output: f64, cache_read: f64, cache_write: f64) -> Self {
        Self {
            input,
            output,
            cache_read,
            cache_write,
        }
    }

    /// Create a free tier cost (all zeros).
    pub fn free() -> Self {
        Self::default()
    }
}

/// OpenAI Completions compatibility settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAICompletionsCompat {
    /// Supports the 'store' parameter.
    #[serde(default)]
    pub supports_store: bool,
    /// Supports the 'developer' role.
    #[serde(default)]
    pub supports_developer_role: bool,
    /// Supports 'reasoning_effort' parameter.
    #[serde(default)]
    pub supports_reasoning_effort: bool,
    /// Mapping of thinking levels to reasoning effort values.
    #[serde(default)]
    pub reasoning_effort_map: HashMap<String, String>,
    /// Supports usage in streaming responses.
    #[serde(default = "default_true")]
    pub supports_usage_in_streaming: bool,
    /// Which field to use for max tokens.
    #[serde(default)]
    pub max_tokens_field: Option<String>,
    /// Requires 'name' field in tool results.
    #[serde(default)]
    pub requires_tool_result_name: bool,
    /// Requires assistant message after tool results.
    #[serde(default)]
    pub requires_assistant_after_tool_result: bool,
    /// Requires thinking blocks to be sent as text.
    #[serde(default)]
    pub requires_thinking_as_text: bool,
    /// Thinking format ("openai", "zai", "qwen", "qwen-chat-template").
    #[serde(default = "default_openai")]
    pub thinking_format: String,
    /// Supports strict mode for tool calls.
    #[serde(default = "default_true")]
    pub supports_strict_mode: bool,
    /// OpenRouter routing preferences (e.g., `{"only": ["anthropic"]}`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_router_routing: Option<serde_json::Value>,

    /// When true, this provider requires every assistant message to carry
    /// reasoning/thinking content when thinking is enabled, and content must
    /// not be null.  Currently DeepSeek API enforces this constraint.
    #[serde(default)]
    pub reasoning_content_constrained: bool,
}

fn default_true() -> bool {
    true
}

fn default_openai() -> String {
    "openai".to_string()
}

impl Default for OpenAICompletionsCompat {
    fn default() -> Self {
        Self {
            supports_store: true,
            supports_developer_role: false,
            supports_reasoning_effort: true,
            reasoning_effort_map: HashMap::new(),
            supports_usage_in_streaming: true,
            max_tokens_field: None,
            requires_tool_result_name: false,
            requires_assistant_after_tool_result: false,
            requires_thinking_as_text: false,
            thinking_format: "openai".to_string(),
            supports_strict_mode: true,
            open_router_routing: None,
            reasoning_content_constrained: false,
        }
    }
}

/// Model configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    /// Model identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// API type (optional — determined by Provider implementation if not set).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<Api>,
    /// Provider name.
    pub provider: Provider,
    /// Base URL for API calls. When None, the provider uses its own default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Whether this model supports reasoning/thinking.
    pub reasoning: bool,
    /// Supported input types.
    pub input: Vec<InputType>,
    /// Cost configuration.
    pub cost: Cost,
    /// Context window size in tokens.
    pub context_window: u32,
    /// Maximum output tokens.
    pub max_tokens: u32,
    /// Custom headers for API calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Compatibility settings for OpenAI Completions API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<OpenAICompletionsCompat>,
}

impl Model {
    /// Create a new model builder.
    pub fn builder() -> ModelBuilder {
        ModelBuilder::default()
    }

    /// Check if this model supports text input.
    pub fn supports_text(&self) -> bool {
        self.input.contains(&InputType::Text)
    }

    /// Check if this model supports image input.
    pub fn supports_image(&self) -> bool {
        self.input.contains(&InputType::Image)
    }

    /// Calculate the cost for given usage.
    pub fn calculate_cost(&self, usage: &crate::types::Usage) -> f64 {
        let input_cost = (usage.input as f64 / 1_000_000.0) * self.cost.input;
        let output_cost = (usage.output as f64 / 1_000_000.0) * self.cost.output;
        let cache_read_cost = (usage.cache_read as f64 / 1_000_000.0) * self.cost.cache_read;
        let cache_write_cost = (usage.cache_write as f64 / 1_000_000.0) * self.cost.cache_write;
        input_cost + output_cost + cache_read_cost + cache_write_cost
    }
}

/// Builder for Model.
#[derive(Debug, Default)]
pub struct ModelBuilder {
    id: Option<String>,
    name: Option<String>,
    api: Option<Api>,
    provider: Option<Provider>,
    base_url: Option<String>,
    reasoning: bool,
    input: Vec<InputType>,
    cost: Cost,
    context_window: Option<u32>,
    max_tokens: Option<u32>,
    headers: Option<HashMap<String, String>>,
    compat: Option<OpenAICompletionsCompat>,
}

impl ModelBuilder {
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn api(mut self, api: Api) -> Self {
        self.api = Some(api);
        self
    }

    pub fn provider(mut self, provider: Provider) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    pub fn reasoning(mut self, reasoning: bool) -> Self {
        self.reasoning = reasoning;
        self
    }

    pub fn input(mut self, input: Vec<InputType>) -> Self {
        self.input = input;
        self
    }

    pub fn cost(mut self, cost: Cost) -> Self {
        self.cost = cost;
        self
    }

    pub fn context_window(mut self, tokens: u32) -> Self {
        self.context_window = Some(tokens);
        self
    }

    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    pub fn headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers = Some(headers);
        self
    }

    pub fn compat(mut self, compat: OpenAICompletionsCompat) -> Self {
        self.compat = Some(compat);
        self
    }

    pub fn build(self) -> Result<Model, String> {
        let id = self.id.ok_or("id is required")?;
        let name = self.name.ok_or("name is required")?;
        let provider = self.provider.ok_or("provider is required")?;
        let context_window = self.context_window.ok_or("context_window is required")?;
        let max_tokens = self.max_tokens.ok_or("max_tokens is required")?;

        Ok(Model {
            id,
            name,
            api: self.api,
            provider,
            base_url: self.base_url,
            reasoning: self.reasoning,
            input: self.input,
            cost: self.cost,
            context_window,
            max_tokens,
            headers: self.headers,
            compat: self.compat,
        })
    }
}
