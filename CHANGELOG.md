# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] - 2026-05-01
### :recycle: Refactors
- [`74e54da`](https://github.com/TiyAgents/tiycore/commit/74e54daefb54c330e8cc4fd21666d09c4b1115ca) - ♻️ simplify code with Default derives and idiomatic Rust patterns *(commit by [@jorben](https://github.com/jorben))*


## [0.2.2] - 2026-04-28
### :sparkles: New Features
- [`bd0565d`](https://github.com/TiyAgents/tiycore/commit/bd0565d62bef8b66a0cd6efd2296eae506af80a8) - **catalog**: ✨ add reasoning_content_constrained field to UnifiedModelInfo *(commit by [@jorben](https://github.com/jorben))*

### :recycle: Refactors
- [`a8ccc84`](https://github.com/TiyAgents/tiycore/commit/a8ccc847f896d6b9002ae9b115c0b549901fa73d) - **protocol**: ♻️ remove model-id heuristic from reasoning content normalization *(commit by [@jorben](https://github.com/jorben))*


## [0.2.1] - 2026-04-27
### :bug: Bug Fixes
- [`7ca9761`](https://github.com/TiyAgents/tiycore/commit/7ca9761cda88c6739943fc61e6d43ba1ad482f18) - **protocol**: 🐛 always use reasoning_content for thinking text *(commit by [@jorben](https://github.com/jorben))*


## [0.2.0] - 2026-04-27
### :sparkles: New Features
- [`d42c906`](https://github.com/TiyAgents/tiycore/commit/d42c906dfe05986a2db92bf8c681e5cd9a2d539d) - **protocol**: ✨ add reasoning content normalization for DeepSeek and constrained providers *(commit by [@jorben](https://github.com/jorben))*

## [0.1.21] - 2026-04-26
### :sparkles: New Features
- [`3acbefc`](https://github.com/TiyAgents/tiycore/commit/3acbefc4f43748f833bee15e1833a201ca8f0703) - **agent**: ✨ add turn_index to events and pre-serialization message hook *(commit by [@jorben](https://github.com/jorben))*


## [0.1.20] - 2026-04-25
### :sparkles: New Features
- [`5ee1b71`](https://github.com/TiyAgents/tiycore/commit/5ee1b71d6e6abce72e09122e7b20a36d80a4023f) - **provider/zenmux**: ✨ add deepseek model routing to OpenAI-compatible protocol *(commit by [@jorben](https://github.com/jorben))*


## [0.1.19] - 2026-04-25
### :sparkles: New Features
- [`6b71c79`](https://github.com/TiyAgents/tiycore/commit/6b71c79d1680a3ace5ad927d3834b8cb1983d3d8) - **protocol**: ✨ dump request body on client errors for debugging *(commit by [@jorben](https://github.com/jorben))*


## [0.1.18] - 2026-04-25
### :bug: Bug Fixes
- [`67826dc`](https://github.com/TiyAgents/tiycore/commit/67826dc46f896cca53e4d7a0842b6d1ddd6923bd) - **protocol**: 🐛 handle empty content with reasoning for provider compatibility *(commit by [@jorben](https://github.com/jorben))*


## [0.1.17] - 2026-04-24
### :sparkles: New Features
- [`735e94f`](https://github.com/TiyAgents/tiycore/commit/735e94f11bbb7c8b2390dc709c1e6c7674fe3a95) - **catalog**: ✨ add tools capability to DeepSeek and MiniMax injections *(commit by [@jorben](https://github.com/jorben))*


## [0.1.16] - 2026-04-24
### :sparkles: New Features
- [`da763ba`](https://github.com/TiyAgents/tiycore/commit/da763baf53f751a3ae0e28fb9d60ad0b7f3ab30c) - **catalog**: ✨ add MiniMax & MiniMaxCN predefined model support *(commit by [@jorben](https://github.com/jorben))*

### :bug: Bug Fixes
- [`8fcd40e`](https://github.com/TiyAgents/tiycore/commit/8fcd40ecd23a0e8e3daf53d6c5c727cc74a6bed7) - **protocol**: 🐛 tolerate missing message_delta when message_stop present *(commit by [@jorben](https://github.com/jorben))*
- [`e9e36af`](https://github.com/TiyAgents/tiycore/commit/e9e36af09957b39c6dff7979caa773c336da5f6b) - **protocol**: 🐛 tolerate missing terminal events when usage metadata is received *(commit by [@jorben](https://github.com/jorben))*
- [`5801c72`](https://github.com/TiyAgents/tiycore/commit/5801c72c5f55621505be7c2593a96343634a91a0) - **provider**: 🐛 correct MiniMax base URLs to include /v1 suffix *(commit by [@jorben](https://github.com/jorben))*
- [`91ae5eb`](https://github.com/TiyAgents/tiycore/commit/91ae5ebc77461bffd2de73952e79d5bde2929672) - **provider**: 🐛 correct MiniMax model base URL assignment and borrowing *(commit by [@jorben](https://github.com/jorben))*


## [0.1.15] - 2026-04-23
### :sparkles: New Features
- [`274c04c`](https://github.com/TiyAgents/tiycore/commit/274c04c08d492086400980d172e6167017d4d573) - **catalog**: ✨ add predefined models adapter for OpenCode Go *(PR [#17](https://github.com/TiyAgents/tiycore/pull/17) by [@HayWolf](https://github.com/HayWolf))*


## [0.1.14] - 2026-04-23
### :sparkles: New Features
- [`8671ab8`](https://github.com/TiyAgents/tiycore/commit/8671ab8691930906ad8f00d0cc4843d22c90caeb) - **protocol**: ✨ improve OpenAI Responses ID normalization with hashing *(commit by [@jorben](https://github.com/jorben))*


## [0.1.13] - 2026-04-22
### :sparkles: New Features
- [`464e0c0`](https://github.com/TiyAgents/tiycore/commit/464e0c01681ef3c9b0865459bf742d3b4e889614) - **catalog**: ✨ add model injections to catalog patches *(PR [#14](https://github.com/TiyAgents/tiycore/pull/14) by [@jorben](https://github.com/jorben))*
- [`d8d68fe`](https://github.com/TiyAgents/tiycore/commit/d8d68fe810df4351c6d43bf6feb9462e66d4d7c4) - **protocol/anthropic**: ✨ support thinking disabled, display option, redacted data, and tool_use input *(commit by [@jorben](https://github.com/jorben))*
- [`364d17e`](https://github.com/TiyAgents/tiycore/commit/364d17ed655c8fe5ac285d23cedf5dddd5063c09) - **protocol/openai**: ✨ add tool choice, reasoning, image placeholder, function_call_arguments.done, and composite ID clamping *(commit by [@jorben](https://github.com/jorben))*
- [`c82e29d`](https://github.com/TiyAgents/tiycore/commit/c82e29d342875ed1297b1b121318b4cf1b929cb5) - **provider/zenmux**: ✨ route kimi/moonshotai models to OpenAI Completions protocol *(commit by [@jorben](https://github.com/jorben))*

### :bug: Bug Fixes
- [`74cf103`](https://github.com/TiyAgents/tiycore/commit/74cf1038c114939eef0f627c69c6a113f3dd0366) - **protocol**: 🐛 clamp debug preview at char boundary *(commit by [@jorben](https://github.com/jorben))*


## [0.1.12] - 2026-04-21
### :sparkles: New Features
- [`7aa18af`](https://github.com/TiyAgents/tiycore/commit/7aa18af6bfd9fc90896c956981cdba7602efe7bc) - **catalog**: ✨ expand list-model provider support *(commit by [@jorben](https://github.com/jorben))*


## [0.1.11] - 2026-04-20
### :sparkles: New Features
- [`0f6428f`](https://github.com/TiyAgents/tiycore/commit/0f6428f7bf6f3b8ced7868f3c9137bd6178d1bdc) - **zenmux**: ✨ ignore source suffix in route detection *(commit by [@jorben](https://github.com/jorben))*


## [0.1.10] - 2026-04-20
### :sparkles: New Features
- [`879999e`](https://github.com/TiyAgents/tiycore/commit/879999e8e4ca8864b98c49b4f081a6b85830d725) - **provider**: ✨ add standalone OpenAI Responses provider facade *(commit by [@jorben](https://github.com/jorben))*


## [0.1.9] - 2026-04-19
### :bug: Bug Fixes
- [`7d33f76`](https://github.com/TiyAgents/tiycore/commit/7d33f76fe2701df44399f009a0c5ba66c860c106) - **openai_completions**: 🐛 infer ToolUse when finish_reason is omitted *(commit by [@jorben](https://github.com/jorben))*
- [`0d0998a`](https://github.com/TiyAgents/tiycore/commit/0d0998a184fb9c45e4027ab59adae8048a423604) - **protocol**: 🐛 handle null tool_calls and add stream tracing *(commit by [@jorben](https://github.com/jorben))*


## [0.1.8] - 2026-04-19
### :sparkles: New Features
- [`d3fc3a4`](https://github.com/TiyAgents/tiycore/commit/d3fc3a4e8281ab6755b32cc061c857134ef7f8ca) - **provider**: ✨ Add OpenCode Go provider with adaptive routing *(PR [#8](https://github.com/TiyAgents/tiycore/pull/8) by [@HayWolf](https://github.com/HayWolf))*

### :bug: Bug Fixes
- [`967f0c7`](https://github.com/TiyAgents/tiycore/commit/967f0c7b025ccb2eb9f4916988946837976679b9) - **openai**: 🐛 tolerate missing finish_reason when [DONE] received *(PR [#7](https://github.com/TiyAgents/tiycore/pull/7) by [@HayWolf](https://github.com/HayWolf))*


## [0.1.7] - 2026-04-17
### :sparkles: New Features
- [`8c323dc`](https://github.com/TiyAgents/tiycore/commit/8c323dca3de707d74b44d265300f78a5c5de6c58) - **anthropic**: ✨ add Claude Opus 4.7 support with adaptive thinking and xhigh effort *(commit by [@jorben](https://github.com/jorben))*


## [0.1.6] - 2026-04-14
### :sparkles: New Features
- [`d2d77c1`](https://github.com/TiyAgents/tiycore/commit/d2d77c13d8b5edee4315cb4baa2509efee286597) - **url-policy**: ✨ allow HTTP for configured HTTPS-exempt hosts *(commit by [@jorben](https://github.com/jorben))*


## [0.1.4] - 2026-04-12
### :bug: Bug Fixes
- [`5c60ce0`](https://github.com/TiyAgents/tiycore/commit/5c60ce0358fdea6e9a231347f7af3f7e1bafacfd) - 🐛 Use TIY_CACHE_RETENTION for cache retention *(commit by [@jorben](https://github.com/jorben))*


## [0.1.3] - 2026-04-07
### :sparkles: New Features
- [`303f1a2`](https://github.com/TiyAgents/tiycore/commit/303f1a208106734d425d896d51d4b2aec57f9e6b) - **catalog**: ✨ add scheduled snapshot patching via catalog/patches.json *(commit by [@jorben](https://github.com/jorben))*
- [`1f975ea`](https://github.com/TiyAgents/tiycore/commit/1f975ea3e3e82071420dfcd8d14ff68f58e32652) - **agent**: ✨ add custom HTTP headers support for LLM requests *(commit by [@jorben](https://github.com/jorben))*


## [0.1.2] - 2026-04-05
### :sparkles: New Features
- [`b862e72`](https://github.com/TiyAgents/tiycore/commit/b862e723059a39d4871786eb9ebdc135a651e644) - **provider**: ✨ implement Anthropic, Google, OpenAI Responses and add 7 new providers
- [`74d4970`](https://github.com/TiyAgents/tiycore/commit/74d49702c459ec583ef7b94b65a37347d7160995) - **agent**: ✨ implement full agent conversation loop with tool execution
- [`2fc925a`](https://github.com/TiyAgents/tiycore/commit/2fc925ac8df0f4cd1a0cd23280e3485b18db8ec8) - **provider**: ✨ add base_url override, tracing, and Google auth header *(commit by [@jorben](https://github.com/jorben))*
- [`13d64e1`](https://github.com/TiyAgents/tiycore/commit/13d64e17a9a84a718b69667d652ac003e65e6098) - **provider**: add Provider::OpenAIResponses variant and use registry pattern in example *(commit by [@jorben](https://github.com/jorben))*
- [`1841c36`](https://github.com/TiyAgents/tiycore/commit/1841c3639a760eb8cf9fb6735f66b7ac6e534f16) - **provider**: ✨ add get_registered_providers() and clarify model registry docs *(commit by [@jorben](https://github.com/jorben))*
- [`e7cd4ea`](https://github.com/TiyAgents/tiycore/commit/e7cd4ea26490db630075cd33469f2ae7e88023dd) - **google**: ✨ add Vertex AI URL format and auth header support *(commit by [@jorben](https://github.com/jorben))*
- [`96b1097`](https://github.com/TiyAgents/tiycore/commit/96b1097658b80f93563b1b3492df0510eadd42e4) - **security**: ✨ add SecurityConfig and comprehensive hardening across providers and agent *(commit by [@jorben](https://github.com/jorben))*
- [`7665729`](https://github.com/TiyAgents/tiycore/commit/7665729b4f0ee4fec939ca3c452294eb0b31e27a) - **agent**: ✨ implement full Agent capability set with provider integration *(commit by [@jorben](https://github.com/jorben))*
- [`c1f1203`](https://github.com/TiyAgents/tiycore/commit/c1f1203a09b4e0c755cff8c354a36bc69bcf99c4) - **provider**: ✨ wire onPayload hook into all protocol providers *(commit by [@jorben](https://github.com/jorben))*
- [`4293551`](https://github.com/TiyAgents/tiycore/commit/4293551d5717c509e9120503acf367bb2b344ac8) - **provider**: ✨ add DeepSeek provider (OpenAI-compatible delegation) *(commit by [@jorben](https://github.com/jorben))*
- [`cb7c8ce`](https://github.com/TiyAgents/tiycore/commit/cb7c8ce979f0da30181563ed32dc85615efc7997) - **protocol**: ✨ implement full compat-aware request building for OpenAI Completions *(commit by [@jorben](https://github.com/jorben))*
- [`04c032b`](https://github.com/TiyAgents/tiycore/commit/04c032b01826ccff692edf10fb386e81af88f21d) - **provider**: ✨ add OpenAI-compatible facade *(commit by [@jorben](https://github.com/jorben))*
- [`2b37988`](https://github.com/TiyAgents/tiycore/commit/2b37988ee02e73f559dd2a05b5d3b7795ee9eb61) - **catalog**: ✨ add model catalog snapshots and sync workflow *(commit by [@jorben](https://github.com/jorben))*
- [`f7b83de`](https://github.com/TiyAgents/tiycore/commit/f7b83dec5a8677f191a232613624b64190d91545) - **catalog**: ✨ add manual model enrichment API *(commit by [@jorben](https://github.com/jorben))*
- [`e229331`](https://github.com/TiyAgents/tiycore/commit/e2293318b7defad74de1ba4122c406dd9cd97201) - **catalog**: ✨ merge embedding and vertex model lists *(commit by [@jorben](https://github.com/jorben))*
- [`5cb8aa3`](https://github.com/TiyAgents/tiycore/commit/5cb8aa3da504918c9ec64592fcbcbe21eb47d7c7) - **agent**: ✨ align loop semantics *(commit by [@jorben](https://github.com/jorben))*
- [`873e309`](https://github.com/TiyAgents/tiycore/commit/873e309ed7fb60be9df45ce93c2ce2768a701524) - **agent**: ✨ add pi-mono runtime parity *(commit by [@jorben](https://github.com/jorben))*
- [`6f0e5e5`](https://github.com/TiyAgents/tiycore/commit/6f0e5e5505c1d0eea8f6afff644078be0da186c4) - **retry**: ✨ add transparent protocol-layer retries with Retry-After support *(commit by [@jorben](https://github.com/jorben))*
- [`2c028da`](https://github.com/TiyAgents/tiycore/commit/2c028dadba90ea62473dea6859a685bd7f1ed7a4) - **agent**: ✨ retry incomplete LLM streams from stable context *(commit by [@jorben](https://github.com/jorben))*
- [`87b6666`](https://github.com/TiyAgents/tiycore/commit/87b66669c8b40688a0d8c6d0e8324ded9612a038) - **catalog**: ✨ add vendor prefixes for additional AI providers *(commit by [@jorben](https://github.com/jorben))*

### :bug: Bug Fixes
- [`a7ed89c`](https://github.com/TiyAgents/tiycore/commit/a7ed89cd94f91d4f509f6a0a5b509df47486f120) - **openai-responses**: 🐛 extract event type from JSON data instead of SSE event line *(commit by [@jorben](https://github.com/jorben))*
- [`ad9e192`](https://github.com/TiyAgents/tiycore/commit/ad9e1920a4c58c76156a8757451c306c6f1611c6) - **catalog**: 🐛 detect reasoning from OpenRouter parameters *(commit by [@jorben](https://github.com/jorben))*
- [`4dcc8b7`](https://github.com/TiyAgents/tiycore/commit/4dcc8b78771900d6dba774dcd05f1ac0ca12a1a3) - **protocol**: 🐛 harden provider edge cases *(commit by [@jorben](https://github.com/jorben))*
- [`5c88c3a`](https://github.com/TiyAgents/tiycore/commit/5c88c3a92a42b59f8c1e20f84fa1f18606fd3ace) - **protocol**: 🐛 map simple-stream reasoning *(commit by [@jorben](https://github.com/jorben))*
- [`cf03c20`](https://github.com/TiyAgents/tiycore/commit/cf03c20157a98ac1a2f4ef9d28f99018d1ba3357) - **protocol**: 🐛 align replay parity with pi-mono *(commit by [@jorben](https://github.com/jorben))*
- [`79942b3`](https://github.com/TiyAgents/tiycore/commit/79942b37259f50978d3f6b5a5f1c2631ec67e2ba) - **protocol**: 🐛 align provider protocol options *(commit by [@jorben](https://github.com/jorben))*
- [`055708d`](https://github.com/TiyAgents/tiycore/commit/055708d4d5070bf45029c815e54f8b6adcde294e) - **protocol**: 🐛 close remaining pi-mono parity gaps *(commit by [@jorben](https://github.com/jorben))*
- [`3923e35`](https://github.com/TiyAgents/tiycore/commit/3923e3553f30a91b5762ace0b9c0df55fe2d1d8b) - **openai**: 🐛 Strip unstored response item IDs *(commit by [@jorben](https://github.com/jorben))*
- [`1433615`](https://github.com/TiyAgents/tiycore/commit/1433615eea340663a39779d0a9b5303ccb2347b9) - **agent**: 🐛 error on max turn limit exhaustion *(commit by [@jorben](https://github.com/jorben))*

### :recycle: Refactors
- [`2c850f8`](https://github.com/TiyAgents/tiycore/commit/2c850f81edf844e7e874aac894b8b2110343dde2) - ♻️ improve core types, stream, and model foundations
- [`d5eb78d`](https://github.com/TiyAgents/tiycore/commit/d5eb78dc2e04b63dce42eb2554355956cc565c61) - **provider**: ♻️ key ProviderRegistry by Provider instead of Api *(commit by [@jorben](https://github.com/jorben))*
- [`c3c27ee`](https://github.com/TiyAgents/tiycore/commit/c3c27ee95614114bdd0e88d37735290c31bb1111) - **model**: ♻️ make base_url optional, let providers own defaults *(commit by [@jorben](https://github.com/jorben))*
- [`024f1c6`](https://github.com/TiyAgents/tiycore/commit/024f1c6586321539c5b26e66b5f55240fd537f63) - **zenmux**: ♻️ adaptive 3-way protocol routing with Vertex AI support *(commit by [@jorben](https://github.com/jorben))*
- [`0d63c68`](https://github.com/TiyAgents/tiycore/commit/0d63c68da80fbc22eb122811c87834998d771ee0) - ♻️ modular architecture optimization (P0/P1) *(commit by [@jorben](https://github.com/jorben))*
- [`439a09b`](https://github.com/TiyAgents/tiycore/commit/439a09bbf0a4567c78f2ba6afea2329af6b35ad0) - ♻️ separate protocol and provider naming to eliminate ambiguity *(commit by [@jorben](https://github.com/jorben))*
- [`4736540`](https://github.com/TiyAgents/tiycore/commit/47365404350e7376b14ce6a030328c7e1c69a86d) - ♻️ move registry and delegation from protocol/ to provider/ *(commit by [@jorben](https://github.com/jorben))*

### :white_check_mark: Tests
- [`88199ef`](https://github.com/TiyAgents/tiycore/commit/88199effbfccfbae2d9476451ff92f5f7b435812) - ✅ add provider tests for Ollama, Zenmux, and delegation providers *(commit by [@jorben](https://github.com/jorben))*
- [`647f02d`](https://github.com/TiyAgents/tiycore/commit/647f02d3eef43774f83d261092983fae9cab289b) - ✅ improve test coverage from 75.59% to 85.80% *(commit by [@jorben](https://github.com/jorben))*

### :wrench: Chores
- [`5180071`](https://github.com/TiyAgents/tiycore/commit/518007167af688b1265b7823d6c0bb200ce5e71e) - 🔧 clean up basic_usage example
- [`cfa9803`](https://github.com/TiyAgents/tiycore/commit/cfa9803aff7522192d0dfea9d0dbbb9eafa381f4) - ✨ rename crate ti y-core to tiycore *(commit by [@jorben](https://github.com/jorben))*


## [0.1.1] - 2026-04-05
### :sparkles: New Features
- [`b862e72`](https://github.com/TiyAgents/tiycore/commit/b862e723059a39d4871786eb9ebdc135a651e644) - **provider**: ✨ implement Anthropic, Google, OpenAI Responses and add 7 new providers
- [`74d4970`](https://github.com/TiyAgents/tiycore/commit/74d49702c459ec583ef7b94b65a37347d7160995) - **agent**: ✨ implement full agent conversation loop with tool execution
- [`2fc925a`](https://github.com/TiyAgents/tiycore/commit/2fc925ac8df0f4cd1a0cd23280e3485b18db8ec8) - **provider**: ✨ add base_url override, tracing, and Google auth header *(commit by [@jorben](https://github.com/jorben))*
- [`13d64e1`](https://github.com/TiyAgents/tiycore/commit/13d64e17a9a84a718b69667d652ac003e65e6098) - **provider**: add Provider::OpenAIResponses variant and use registry pattern in example *(commit by [@jorben](https://github.com/jorben))*
- [`1841c36`](https://github.com/TiyAgents/tiycore/commit/1841c3639a760eb8cf9fb6735f66b7ac6e534f16) - **provider**: ✨ add get_registered_providers() and clarify model registry docs *(commit by [@jorben](https://github.com/jorben))*
- [`e7cd4ea`](https://github.com/TiyAgents/tiycore/commit/e7cd4ea26490db630075cd33469f2ae7e88023dd) - **google**: ✨ add Vertex AI URL format and auth header support *(commit by [@jorben](https://github.com/jorben))*
- [`96b1097`](https://github.com/TiyAgents/tiycore/commit/96b1097658b80f93563b1b3492df0510eadd42e4) - **security**: ✨ add SecurityConfig and comprehensive hardening across providers and agent *(commit by [@jorben](https://github.com/jorben))*
- [`7665729`](https://github.com/TiyAgents/tiycore/commit/7665729b4f0ee4fec939ca3c452294eb0b31e27a) - **agent**: ✨ implement full Agent capability set with provider integration *(commit by [@jorben](https://github.com/jorben))*
- [`c1f1203`](https://github.com/TiyAgents/tiycore/commit/c1f1203a09b4e0c755cff8c354a36bc69bcf99c4) - **provider**: ✨ wire onPayload hook into all protocol providers *(commit by [@jorben](https://github.com/jorben))*
- [`4293551`](https://github.com/TiyAgents/tiycore/commit/4293551d5717c509e9120503acf367bb2b344ac8) - **provider**: ✨ add DeepSeek provider (OpenAI-compatible delegation) *(commit by [@jorben](https://github.com/jorben))*
- [`cb7c8ce`](https://github.com/TiyAgents/tiycore/commit/cb7c8ce979f0da30181563ed32dc85615efc7997) - **protocol**: ✨ implement full compat-aware request building for OpenAI Completions *(commit by [@jorben](https://github.com/jorben))*
- [`04c032b`](https://github.com/TiyAgents/tiycore/commit/04c032b01826ccff692edf10fb386e81af88f21d) - **provider**: ✨ add OpenAI-compatible facade *(commit by [@jorben](https://github.com/jorben))*
- [`2b37988`](https://github.com/TiyAgents/tiycore/commit/2b37988ee02e73f559dd2a05b5d3b7795ee9eb61) - **catalog**: ✨ add model catalog snapshots and sync workflow *(commit by [@jorben](https://github.com/jorben))*
- [`f7b83de`](https://github.com/TiyAgents/tiycore/commit/f7b83dec5a8677f191a232613624b64190d91545) - **catalog**: ✨ add manual model enrichment API *(commit by [@jorben](https://github.com/jorben))*
- [`e229331`](https://github.com/TiyAgents/tiycore/commit/e2293318b7defad74de1ba4122c406dd9cd97201) - **catalog**: ✨ merge embedding and vertex model lists *(commit by [@jorben](https://github.com/jorben))*
- [`5cb8aa3`](https://github.com/TiyAgents/tiycore/commit/5cb8aa3da504918c9ec64592fcbcbe21eb47d7c7) - **agent**: ✨ align loop semantics *(commit by [@jorben](https://github.com/jorben))*
- [`873e309`](https://github.com/TiyAgents/tiycore/commit/873e309ed7fb60be9df45ce93c2ce2768a701524) - **agent**: ✨ add pi-mono runtime parity *(commit by [@jorben](https://github.com/jorben))*
- [`6f0e5e5`](https://github.com/TiyAgents/tiycore/commit/6f0e5e5505c1d0eea8f6afff644078be0da186c4) - **retry**: ✨ add transparent protocol-layer retries with Retry-After support *(commit by [@jorben](https://github.com/jorben))*
- [`2c028da`](https://github.com/TiyAgents/tiycore/commit/2c028dadba90ea62473dea6859a685bd7f1ed7a4) - **agent**: ✨ retry incomplete LLM streams from stable context *(commit by [@jorben](https://github.com/jorben))*
- [`87b6666`](https://github.com/TiyAgents/tiycore/commit/87b66669c8b40688a0d8c6d0e8324ded9612a038) - **catalog**: ✨ add vendor prefixes for additional AI providers *(commit by [@jorben](https://github.com/jorben))*

### :bug: Bug Fixes
- [`a7ed89c`](https://github.com/TiyAgents/tiycore/commit/a7ed89cd94f91d4f509f6a0a5b509df47486f120) - **openai-responses**: 🐛 extract event type from JSON data instead of SSE event line *(commit by [@jorben](https://github.com/jorben))*
- [`ad9e192`](https://github.com/TiyAgents/tiycore/commit/ad9e1920a4c58c76156a8757451c306c6f1611c6) - **catalog**: 🐛 detect reasoning from OpenRouter parameters *(commit by [@jorben](https://github.com/jorben))*
- [`4dcc8b7`](https://github.com/TiyAgents/tiycore/commit/4dcc8b78771900d6dba774dcd05f1ac0ca12a1a3) - **protocol**: 🐛 harden provider edge cases *(commit by [@jorben](https://github.com/jorben))*
- [`5c88c3a`](https://github.com/TiyAgents/tiycore/commit/5c88c3a92a42b59f8c1e20f84fa1f18606fd3ace) - **protocol**: 🐛 map simple-stream reasoning *(commit by [@jorben](https://github.com/jorben))*
- [`cf03c20`](https://github.com/TiyAgents/tiycore/commit/cf03c20157a98ac1a2f4ef9d28f99018d1ba3357) - **protocol**: 🐛 align replay parity with pi-mono *(commit by [@jorben](https://github.com/jorben))*
- [`79942b3`](https://github.com/TiyAgents/tiycore/commit/79942b37259f50978d3f6b5a5f1c2631ec67e2ba) - **protocol**: 🐛 align provider protocol options *(commit by [@jorben](https://github.com/jorben))*
- [`055708d`](https://github.com/TiyAgents/tiycore/commit/055708d4d5070bf45029c815e54f8b6adcde294e) - **protocol**: 🐛 close remaining pi-mono parity gaps *(commit by [@jorben](https://github.com/jorben))*
- [`3923e35`](https://github.com/TiyAgents/tiycore/commit/3923e3553f30a91b5762ace0b9c0df55fe2d1d8b) - **openai**: 🐛 Strip unstored response item IDs *(commit by [@jorben](https://github.com/jorben))*
- [`1433615`](https://github.com/TiyAgents/tiycore/commit/1433615eea340663a39779d0a9b5303ccb2347b9) - **agent**: 🐛 error on max turn limit exhaustion *(commit by [@jorben](https://github.com/jorben))*

### :recycle: Refactors
- [`2c850f8`](https://github.com/TiyAgents/tiycore/commit/2c850f81edf844e7e874aac894b8b2110343dde2) - ♻️ improve core types, stream, and model foundations
- [`d5eb78d`](https://github.com/TiyAgents/tiycore/commit/d5eb78dc2e04b63dce42eb2554355956cc565c61) - **provider**: ♻️ key ProviderRegistry by Provider instead of Api *(commit by [@jorben](https://github.com/jorben))*
- [`c3c27ee`](https://github.com/TiyAgents/tiycore/commit/c3c27ee95614114bdd0e88d37735290c31bb1111) - **model**: ♻️ make base_url optional, let providers own defaults *(commit by [@jorben](https://github.com/jorben))*
- [`024f1c6`](https://github.com/TiyAgents/tiycore/commit/024f1c6586321539c5b26e66b5f55240fd537f63) - **zenmux**: ♻️ adaptive 3-way protocol routing with Vertex AI support *(commit by [@jorben](https://github.com/jorben))*
- [`0d63c68`](https://github.com/TiyAgents/tiycore/commit/0d63c68da80fbc22eb122811c87834998d771ee0) - ♻️ modular architecture optimization (P0/P1) *(commit by [@jorben](https://github.com/jorben))*
- [`439a09b`](https://github.com/TiyAgents/tiycore/commit/439a09bbf0a4567c78f2ba6afea2329af6b35ad0) - ♻️ separate protocol and provider naming to eliminate ambiguity *(commit by [@jorben](https://github.com/jorben))*
- [`4736540`](https://github.com/TiyAgents/tiycore/commit/47365404350e7376b14ce6a030328c7e1c69a86d) - ♻️ move registry and delegation from protocol/ to provider/ *(commit by [@jorben](https://github.com/jorben))*

### :white_check_mark: Tests
- [`88199ef`](https://github.com/TiyAgents/tiycore/commit/88199effbfccfbae2d9476451ff92f5f7b435812) - ✅ add provider tests for Ollama, Zenmux, and delegation providers *(commit by [@jorben](https://github.com/jorben))*
- [`647f02d`](https://github.com/TiyAgents/tiycore/commit/647f02d3eef43774f83d261092983fae9cab289b) - ✅ improve test coverage from 75.59% to 85.80% *(commit by [@jorben](https://github.com/jorben))*

### :wrench: Chores
- [`5180071`](https://github.com/TiyAgents/tiycore/commit/518007167af688b1265b7823d6c0bb200ce5e71e) - 🔧 clean up basic_usage example
- [`cfa9803`](https://github.com/TiyAgents/tiycore/commit/cfa9803aff7522192d0dfea9d0dbbb9eafa381f4) - ✨ rename crate ti y-core to tiycore *(commit by [@jorben](https://github.com/jorben))*

[0.1.1]: https://github.com/TiyAgents/tiycore/compare/0.0.1...0.1.1
[0.1.2]: https://github.com/TiyAgents/tiycore/compare/0.0.1...0.1.2
[0.1.3]: https://github.com/TiyAgents/tiycore/compare/0.1.2...0.1.3
[0.1.4]: https://github.com/TiyAgents/tiycore/compare/0.1.3...0.1.4
[0.1.6]: https://github.com/TiyAgents/tiycore/compare/0.1.5...0.1.6
[0.1.7]: https://github.com/TiyAgents/tiycore/compare/0.1.6...0.1.7
[0.1.8]: https://github.com/TiyAgents/tiycore/compare/0.1.7...0.1.8
[0.1.9]: https://github.com/TiyAgents/tiycore/compare/0.1.8...0.1.9
[0.1.10]: https://github.com/TiyAgents/tiycore/compare/0.1.9...0.1.10
[0.1.11]: https://github.com/TiyAgents/tiycore/compare/0.1.10...0.1.11
[0.1.12]: https://github.com/TiyAgents/tiycore/compare/0.1.11...0.1.12
[0.1.13]: https://github.com/TiyAgents/tiycore/compare/0.1.12...0.1.13
[0.1.14]: https://github.com/TiyAgents/tiycore/compare/0.1.13...0.1.14
[0.1.15]: https://github.com/TiyAgents/tiycore/compare/0.1.14...0.1.15
[0.1.16]: https://github.com/TiyAgents/tiycore/compare/0.1.15...0.1.16
[0.1.17]: https://github.com/TiyAgents/tiycore/compare/0.1.16...0.1.17
[0.1.18]: https://github.com/TiyAgents/tiycore/compare/0.1.17...0.1.18
[0.1.19]: https://github.com/TiyAgents/tiycore/compare/0.1.18...0.1.19
[0.1.20]: https://github.com/TiyAgents/tiycore/compare/0.1.19...0.1.20
[0.1.21]: https://github.com/TiyAgents/tiycore/compare/0.1.20...0.1.21
[0.2.0]: https://github.com/TiyAgents/tiycore/compare/0.1.21...0.2.0
[0.2.1]: https://github.com/TiyAgents/tiycore/compare/0.2.0...0.2.1
[0.2.2]: https://github.com/TiyAgents/tiycore/compare/0.2.1...0.2.2

[0.2.3]: https://github.com/TiyAgents/tiycore/compare/0.2.2...0.2.3
