//! LLM service provider facades and registry.
//!
//! Each provider represents a specific LLM service vendor.
//! Providers internally delegate to protocol implementations in `crate::protocol`.
//!
//! **Auto-registration:** Built-in providers are automatically created on first
//! access via [`get_provider`]. You do *not* need to call [`register_provider`]
//! unless you want to inject a custom instance (e.g., with a pre-set API key).

// Infrastructure — registry & delegation macros
mod registry;

#[macro_use]
pub(crate) mod delegation;

// Provider facades
pub mod anthropic;
pub mod bai;
pub mod deepseek;
pub mod google;
pub mod groq;
pub mod kimi_coding;
pub mod minimax;
pub mod ollama;
pub mod openai;
pub mod openai_compatible;
pub mod openai_responses;
pub mod opencode_go;
pub mod openrouter;
pub mod xai;
pub mod xiaomi_mimo;
pub mod zai;
pub mod zenmux;

// Re-export protocol trait & type aliases (these stay in protocol/)
pub use crate::protocol::{ArcProtocol, BoxedProtocol, LLMProtocol};

// Re-export registry API (now lives here)
pub use registry::{
    clear_providers, get_provider, get_registered_providers, global_registry, register_provider,
    ProtocolRegistry,
};

/// Register all built-in providers into the global registry.
///
/// This is a convenience function that explicitly registers every built-in
/// provider. In most cases you do **not** need to call this — providers are
/// auto-registered on first access via [`get_provider`]. Use this only if you
/// need `get_registered_providers()` to list all providers up front.
pub fn register_all_providers() {
    use std::sync::Arc;
    register_provider(Arc::new(openai::OpenAIProvider::new()));
    register_provider(Arc::new(openai_compatible::OpenAICompatibleProvider::new()));
    register_provider(Arc::new(openai_responses::OpenAIResponsesProvider::new()));
    register_provider(Arc::new(anthropic::AnthropicProvider::new()));
    register_provider(Arc::new(google::GoogleProvider::new()));
    register_provider(Arc::new(ollama::OllamaProvider::new()));
    register_provider(Arc::new(xai::XAIProvider::new()));
    register_provider(Arc::new(groq::GroqProvider::new()));
    register_provider(Arc::new(openrouter::OpenRouterProvider::new()));
    register_provider(Arc::new(minimax::MiniMaxProvider::new()));
    register_provider(Arc::new(kimi_coding::KimiCodingProvider::new()));
    register_provider(Arc::new(zai::ZAIProvider::new()));
    register_provider(Arc::new(deepseek::DeepSeekProvider::new()));
    register_provider(Arc::new(xiaomi_mimo::XiaomiMIMOProvider::new()));
    register_provider(Arc::new(zenmux::ZenmuxProvider::new()));
    register_provider(Arc::new(bai::BaiProvider::new()));
    register_provider(Arc::new(opencode_go::OpenCodeGoProvider::new()));
}
