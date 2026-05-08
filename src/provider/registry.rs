//! Provider registry for managing LLM providers.
//!
//! Built-in providers are auto-registered on first access via [`get_provider`].
//! Manual [`register_provider`] calls are still supported for overriding defaults
//! (e.g., injecting a custom API key via `with_api_key()`).

use crate::protocol::ArcProtocol;
use crate::types::Provider;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Provider registry for managing LLM providers.
pub struct ProtocolRegistry {
    providers: RwLock<HashMap<String, ArcProtocol>>,
}

impl ProtocolRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            providers: RwLock::new(HashMap::new()),
        }
    }

    /// Register a provider.
    pub fn register(&self, provider: ArcProtocol) {
        let provider_type = provider.provider_type();
        let mut providers = self.providers.write();
        providers.insert(provider_type.as_str().to_string(), provider);
    }

    /// Get a provider by provider type.
    pub fn get(&self, provider: &Provider) -> Option<ArcProtocol> {
        let providers = self.providers.read();
        providers.get(provider.as_str()).cloned()
    }

    /// Get a provider by provider type string.
    pub fn get_by_name(&self, provider_name: &str) -> Option<ArcProtocol> {
        let providers = self.providers.read();
        providers.get(provider_name).cloned()
    }

    /// Unregister a provider by provider type.
    pub fn unregister(&self, provider: &Provider) {
        let mut providers = self.providers.write();
        providers.remove(provider.as_str());
    }

    /// Clear all providers.
    pub fn clear(&self) {
        let mut providers = self.providers.write();
        providers.clear();
    }

    /// Get all registered provider types.
    pub fn provider_types(&self) -> Vec<String> {
        let providers = self.providers.read();
        providers.keys().cloned().collect()
    }

    /// Check if a provider is registered.
    pub fn contains(&self, provider: &Provider) -> bool {
        let providers = self.providers.read();
        providers.contains_key(provider.as_str())
    }
}

impl Default for ProtocolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global provider registry.
static GLOBAL_REGISTRY: once_cell::sync::Lazy<ProtocolRegistry> =
    once_cell::sync::Lazy::new(ProtocolRegistry::new);

/// Get the global provider registry.
pub fn global_registry() -> &'static ProtocolRegistry {
    &GLOBAL_REGISTRY
}

/// Register a provider globally.
pub fn register_provider(provider: ArcProtocol) {
    GLOBAL_REGISTRY.register(provider);
}

/// Get a provider from the global registry.
///
/// If the requested provider is not yet registered, a default instance is
/// created automatically for all built-in providers. This means most users
/// never need to call [`register_provider`] manually.
///
/// To override the default (e.g., to inject an API key via `with_api_key()`),
/// call [`register_provider`] before the first `get_provider` call.
pub fn get_provider(provider: &Provider) -> Option<ArcProtocol> {
    // Fast path: already registered
    if let Some(p) = GLOBAL_REGISTRY.get(provider) {
        return Some(p);
    }
    // Slow path: auto-register built-in provider on first access
    if let Some(p) = create_default_provider(provider) {
        GLOBAL_REGISTRY.register(p.clone());
        Some(p)
    } else {
        None
    }
}

/// Create a default provider instance for a built-in [`Provider`] variant.
///
/// Returns `None` for [`Provider::Custom`] and provider variants that exist
/// only as type data (e.g., `AmazonBedrock`, `Cerebras`) without a
/// corresponding provider module.
fn create_default_provider(provider: &Provider) -> Option<ArcProtocol> {
    match provider {
        Provider::OpenAI => Some(Arc::new(super::openai::OpenAIProvider::new())),
        Provider::OpenAICompatible => Some(Arc::new(
            super::openai_compatible::OpenAICompatibleProvider::new(),
        )),
        Provider::OpenAIResponses => Some(Arc::new(
            super::openai_responses::OpenAIResponsesProvider::new(),
        )),
        Provider::Anthropic => Some(Arc::new(super::anthropic::AnthropicProvider::new())),
        Provider::Google => Some(Arc::new(super::google::GoogleProvider::new())),
        Provider::Ollama => Some(Arc::new(super::ollama::OllamaProvider::new())),
        Provider::XAI => Some(Arc::new(super::xai::XAIProvider::new())),
        Provider::Groq => Some(Arc::new(super::groq::GroqProvider::new())),
        Provider::OpenRouter => Some(Arc::new(super::openrouter::OpenRouterProvider::new())),
        Provider::MiniMax | Provider::MiniMaxCN => {
            Some(Arc::new(super::minimax::MiniMaxProvider::new()))
        }
        Provider::KimiCoding => Some(Arc::new(super::kimi_coding::KimiCodingProvider::new())),
        Provider::ZAI => Some(Arc::new(super::zai::ZAIProvider::new())),
        Provider::DeepSeek => Some(Arc::new(super::deepseek::DeepSeekProvider::new())),
        Provider::XiaomiMIMO => Some(Arc::new(super::xiaomi_mimo::XiaomiMIMOProvider::new())),
        Provider::Zenmux => Some(Arc::new(super::zenmux::ZenmuxProvider::new())),
        Provider::Bai => Some(Arc::new(super::bai::BaiProvider::new())),
        Provider::OpenCodeGo => Some(Arc::new(super::opencode_go::OpenCodeGoProvider::new())),
        _ => None,
    }
}

/// Get all registered provider type names from the global registry.
pub fn get_registered_providers() -> Vec<String> {
    GLOBAL_REGISTRY.provider_types()
}

/// Clear all providers from the global registry.
pub fn clear_providers() {
    GLOBAL_REGISTRY.clear();
}
