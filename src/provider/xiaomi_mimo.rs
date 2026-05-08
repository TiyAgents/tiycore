//! Xiaomi MiMo provider (reuses OpenAI Completions protocol).
//!
//! Xiaomi MiMo uses the OpenAI Chat Completions API at:
//! - `https://api.xiaomimimo.com/v1`
//!
//! This provider delegates to `OpenAICompletionsProtocol` with a default base URL.

use crate::protocol::LLMProtocol;
use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use async_trait::async_trait;

/// Default base URL for Xiaomi MiMo API.
const DEFAULT_BASE_URL: &str = "https://api.xiaomimimo.com/v1";

/// Xiaomi MiMo provider (OpenAI-compatible).
pub struct XiaomiMIMOProvider {
    default_api_key: Option<String>,
}

impl XiaomiMIMOProvider {
    /// Create a new Xiaomi MiMo provider.
    pub fn new() -> Self {
        Self {
            default_api_key: None,
        }
    }

    /// Create a provider with a default API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        Self {
            default_api_key: Some(api_key.into()),
        }
    }

    /// Resolve API key from options, self, or environment.
    fn resolve_api_key(&self, options: &StreamOptions) -> Option<String> {
        if let Some(ref key) = options.api_key {
            return Some(key.clone());
        }
        if let Some(ref key) = self.default_api_key {
            return Some(key.clone());
        }
        std::env::var("XIAOMI_MIMO_API_KEY").ok()
    }
}

impl Default for XiaomiMIMOProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProtocol for XiaomiMIMOProvider {
    fn provider_type(&self) -> Provider {
        Provider::XiaomiMIMO
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        let mut opts = options;
        if opts.api_key.is_none() {
            opts.api_key = self.resolve_api_key(&opts);
        }

        let mut m = model.clone();

        // Set default base_url if not provided
        if m.base_url.is_none() && opts.base_url.is_none() {
            m.base_url = Some(DEFAULT_BASE_URL.to_string());
        }

        let provider = crate::protocol::openai_completions::OpenAICompletionsProtocol::new();
        provider.stream(&m, context, opts)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let mut opts = options;
        if opts.base.api_key.is_none() {
            opts.base.api_key = self.resolve_api_key(&opts.base);
        }

        let mut m = model.clone();

        // Set default base_url if not provided
        if m.base_url.is_none() && opts.base.base_url.is_none() {
            m.base_url = Some(DEFAULT_BASE_URL.to_string());
        }

        let provider = crate::protocol::openai_completions::OpenAICompletionsProtocol::new();
        provider.stream_simple(&m, context, opts)
    }
}
