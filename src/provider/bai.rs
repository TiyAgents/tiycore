//! BAI provider (adaptive multi-protocol proxy based on model ID).
//!
//! BAI is a multi-protocol proxy that supports:
//! - OpenAI Responses protocol at `https://api.b.ai/v1/responses`
//! - OpenAI-compatible protocol at `https://api.b.ai/v1/chat/completions`
//! - Anthropic Messages protocol at `https://api.b.ai/v1/messages`
//!
//! Adaptive routing logic (when base_url is empty or starts with `https://api.b.ai`):
//! - If the model ID contains "claude" (case-insensitive),
//!   routes to Anthropic Messages protocol
//! - If the model ID contains "gpt" or "openai" (case-insensitive),
//!   routes to OpenAI Responses protocol
//! - Otherwise, routes to OpenAI Completions protocol (default)
//!
//! When a custom base_url is provided (not empty and not starting with
//! `https://api.b.ai`), the provider uses OpenAI Completions protocol
//! with the given base_url as-is.
//!
//! API key environment variable: `BAI_API_KEY`

use crate::protocol::LLMProtocol;
use crate::stream::AssistantMessageEventStream;
use crate::types::*;
use async_trait::async_trait;

/// BAI base URL prefix used to detect adaptive routing mode.
pub(crate) const BAI_HOST_PREFIX: &str = "https://api.b.ai";

/// Default base URL for BAI (shared by both protocols).
const BAI_BASE_URL: &str = "https://api.b.ai/v1";

/// Protocol routing decision for a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BaiProtocolRoute {
    OpenAICompatible,
    OpenAIResponses,
    Anthropic,
}

/// Determine the adaptive BAI protocol route from a model ID.
pub(crate) fn bai_detect_route(model_id: &str) -> BaiProtocolRoute {
    let lower = model_id.to_ascii_lowercase();
    if lower.contains("claude") {
        BaiProtocolRoute::Anthropic
    } else if lower.contains("gpt") || lower.contains("openai") {
        BaiProtocolRoute::OpenAIResponses
    } else {
        BaiProtocolRoute::OpenAICompatible
    }
}

/// Map a BAI adaptive route to the corresponding API type.
pub(crate) fn bai_api_for_route(route: BaiProtocolRoute) -> Api {
    match route {
        BaiProtocolRoute::OpenAICompatible => Api::OpenAICompletions,
        BaiProtocolRoute::OpenAIResponses => Api::OpenAIResponses,
        BaiProtocolRoute::Anthropic => Api::AnthropicMessages,
    }
}

/// Determine the adaptive BAI API type from a model ID.
pub(crate) fn bai_detect_api(model_id: &str) -> Api {
    bai_api_for_route(bai_detect_route(model_id))
}

/// BAI provider (multi-protocol proxy).
pub struct BaiProvider {
    default_api_key: Option<String>,
}

impl BaiProvider {
    /// Create a new BAI provider.
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
        std::env::var("BAI_API_KEY").ok()
    }

    /// Check if adaptive routing should be enabled.
    ///
    /// Resolves the effective base_url (options.base_url > model.base_url),
    /// then returns true when it is None, empty, or starts with `https://api.b.ai`.
    fn should_adapt(options_base_url: &Option<String>, model_base_url: &Option<String>) -> bool {
        let effective = options_base_url.as_deref().or(model_base_url.as_deref());
        match effective {
            None => true,
            Some(url) => url.is_empty() || url.starts_with(BAI_HOST_PREFIX),
        }
    }

    /// Determine protocol route based on model ID.
    fn detect_route(model_id: &str) -> BaiProtocolRoute {
        bai_detect_route(model_id)
    }
}

impl Default for BaiProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLMProtocol for BaiProvider {
    fn provider_type(&self) -> Provider {
        Provider::Bai
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

        if Self::should_adapt(&opts.base_url, &m.base_url) {
            // Adaptive mode: choose protocol based on model ID.
            opts.base_url = None;
            let route = Self::detect_route(&m.id);
            m.api = Some(bai_api_for_route(route));
            m.base_url = Some(BAI_BASE_URL.to_string());
            match route {
                BaiProtocolRoute::OpenAICompatible => {
                    let provider =
                        crate::protocol::openai_completions::OpenAICompletionsProtocol::new();
                    provider.stream(&m, context, opts)
                }
                BaiProtocolRoute::OpenAIResponses => {
                    let provider =
                        crate::protocol::openai_responses::OpenAIResponsesProtocol::new();
                    provider.stream(&m, context, opts)
                }
                BaiProtocolRoute::Anthropic => {
                    let provider = crate::protocol::anthropic::AnthropicProtocol::new();
                    provider.stream(&m, context, opts)
                }
            }
        } else {
            // Custom base_url: use OpenAI Completions protocol as-is
            m.api = Some(Api::OpenAICompletions);
            let provider = crate::protocol::openai_completions::OpenAICompletionsProtocol::new();
            provider.stream(&m, context, opts)
        }
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

        if Self::should_adapt(&opts.base.base_url, &m.base_url) {
            opts.base.base_url = None;
            let route = Self::detect_route(&m.id);
            m.api = Some(bai_api_for_route(route));
            m.base_url = Some(BAI_BASE_URL.to_string());
            match route {
                BaiProtocolRoute::OpenAICompatible => {
                    let provider =
                        crate::protocol::openai_completions::OpenAICompletionsProtocol::new();
                    provider.stream_simple(&m, context, opts)
                }
                BaiProtocolRoute::OpenAIResponses => {
                    let provider =
                        crate::protocol::openai_responses::OpenAIResponsesProtocol::new();
                    provider.stream_simple(&m, context, opts)
                }
                BaiProtocolRoute::Anthropic => {
                    let provider = crate::protocol::anthropic::AnthropicProtocol::new();
                    provider.stream_simple(&m, context, opts)
                }
            }
        } else {
            // Custom base_url: use OpenAI Completions protocol as-is
            m.api = Some(Api::OpenAICompletions);
            let provider = crate::protocol::openai_completions::OpenAICompletionsProtocol::new();
            provider.stream_simple(&m, context, opts)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{bai_api_for_route, bai_detect_api, bai_detect_route, BaiProtocolRoute};
    use crate::types::Api;

    #[test]
    fn test_bai_route_detection() {
        // Claude models → Anthropic
        assert_eq!(
            bai_detect_route("claude-sonnet-4"),
            BaiProtocolRoute::Anthropic
        );
        assert_eq!(
            bai_detect_route("claude-opus-4.6"),
            BaiProtocolRoute::Anthropic
        );
        assert_eq!(
            bai_detect_route("Claude-3.5-Sonnet"),
            BaiProtocolRoute::Anthropic
        );
        assert_eq!(
            bai_detect_route("CLAUDE-HAIKU"),
            BaiProtocolRoute::Anthropic
        );

        // GPT / OpenAI models → OpenAIResponses
        assert_eq!(
            bai_detect_route("gpt-4o"),
            BaiProtocolRoute::OpenAIResponses
        );
        assert_eq!(
            bai_detect_route("gpt-4o-mini"),
            BaiProtocolRoute::OpenAIResponses
        );
        assert_eq!(
            bai_detect_route("GPT-4-turbo"),
            BaiProtocolRoute::OpenAIResponses
        );
        assert_eq!(
            bai_detect_route("openai-o3"),
            BaiProtocolRoute::OpenAIResponses
        );

        // Other models → OpenAICompatible
        assert_eq!(
            bai_detect_route("deepseek-r1"),
            BaiProtocolRoute::OpenAICompatible
        );
        assert_eq!(
            bai_detect_route("gemini-2.5-pro"),
            BaiProtocolRoute::OpenAICompatible
        );
        assert_eq!(
            bai_detect_route(""),
            BaiProtocolRoute::OpenAICompatible
        );
    }

    #[test]
    fn test_bai_route_to_api_mapping() {
        assert_eq!(
            bai_api_for_route(BaiProtocolRoute::OpenAICompatible),
            Api::OpenAICompletions
        );
        assert_eq!(
            bai_api_for_route(BaiProtocolRoute::OpenAIResponses),
            Api::OpenAIResponses
        );
        assert_eq!(
            bai_api_for_route(BaiProtocolRoute::Anthropic),
            Api::AnthropicMessages
        );

        assert_eq!(bai_detect_api("deepseek-r1"), Api::OpenAICompletions);
        assert_eq!(bai_detect_api("gpt-4o"), Api::OpenAIResponses);
        assert_eq!(bai_detect_api("claude-sonnet-4"), Api::AnthropicMessages);
    }
}
