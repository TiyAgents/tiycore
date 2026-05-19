//! Tests for stream module: event_stream and json_parser.

use tiycore::stream::{parse_streaming_json, AssistantMessageEventStream, EventStream};
use tiycore::types::*;

// ============================================================================
// parse_streaming_json tests (extending existing)
// ============================================================================

#[test]
fn test_parse_empty_string() {
    let result = parse_streaming_json("");
    assert!(result.is_object());
    assert!(result.as_object().unwrap().is_empty());
}

#[test]
fn test_parse_whitespace_only() {
    let result = parse_streaming_json("   ");
    assert!(result.is_object());
}

#[test]
fn test_parse_complete_object() {
    let result = parse_streaming_json(r#"{"name": "test", "value": 42}"#);
    assert_eq!(result["name"], "test");
    assert_eq!(result["value"], 42);
}

#[test]
fn test_parse_complete_array() {
    let result = parse_streaming_json(r#"[1, 2, 3]"#);
    assert_eq!(result.as_array().unwrap().len(), 3);
}

#[test]
fn test_parse_incomplete_object_missing_brace() {
    let result = parse_streaming_json(r#"{"name": "test""#);
    assert_eq!(result["name"], "test");
}

#[test]
fn test_parse_incomplete_nested_object() {
    let result = parse_streaming_json(r#"{"outer": {"inner": "value""#);
    assert_eq!(result["outer"]["inner"], "value");
}

#[test]
fn test_parse_incomplete_array() {
    let result = parse_streaming_json(r#"{"items": [1, 2, 3"#);
    let items = result["items"].as_array().unwrap();
    assert_eq!(items.len(), 3);
}

#[test]
fn test_parse_incomplete_string_value() {
    let result = parse_streaming_json(r#"{"text": "hello"#);
    assert_eq!(result["text"], "hello");
}

#[test]
fn test_parse_escaped_quotes() {
    let result = parse_streaming_json(r#"{"text": "hello \"world\""}"#);
    assert_eq!(result["text"], "hello \"world\"");
}

#[test]
fn test_parse_deeply_nested_incomplete() {
    let result = parse_streaming_json(r#"{"a": {"b": {"c": "deep""#);
    assert_eq!(result["a"]["b"]["c"], "deep");
}

#[test]
fn test_parse_incomplete_with_number() {
    // Trailing comma after a number - may fail to parse, should fallback to empty
    let result = parse_streaming_json(r#"{"count": 42,"#);
    // The fix_incomplete_json closes the brace, giving {"count": 42,}
    // which is invalid JSON, so it falls back to empty
    assert!(result.is_object());
}

#[test]
fn test_parse_mixed_nesting() {
    let result = parse_streaming_json(r#"{"arr": [{"nested": true}, {"also": "here""#);
    assert!(result.is_object());
}

#[test]
fn test_parse_boolean_and_null() {
    let result = parse_streaming_json(r#"{"a": true, "b": false, "c": null}"#);
    assert_eq!(result["a"], true);
    assert_eq!(result["b"], false);
    assert!(result["c"].is_null());
}

// ============================================================================
// EventStream basic tests
// ============================================================================

#[test]
fn test_event_stream_creation() {
    let stream: EventStream<String, String> = EventStream::new(|s| s == "done", |s| s.clone());
    assert!(!stream.is_done());
}

#[test]
fn test_event_stream_push_and_done() {
    let stream: EventStream<String, String> = EventStream::new(|s| s == "done", |s| s.clone());

    stream.push("event1".to_string());
    stream.push("event2".to_string());
    assert!(!stream.is_done());

    stream.push("done".to_string());
    assert!(stream.is_done());
}

#[test]
fn test_event_stream_end() {
    let stream: EventStream<String, String> = EventStream::new(|s| s == "done", |s| s.clone());

    stream.push("event1".to_string());
    stream.end(Some("final".to_string()));
    assert!(stream.is_done());
}

#[test]
fn test_event_stream_clone_shares_state() {
    let stream: EventStream<String, String> = EventStream::new(|s| s == "done", |s| s.clone());
    let clone = stream.clone();

    stream.push("event".to_string());
    assert!(!clone.is_done());

    stream.end(None);
    assert!(clone.is_done());
}

// ============================================================================
// AssistantMessageEventStream tests
// ============================================================================

#[test]
fn test_assistant_stream_creation() {
    let stream = AssistantMessageEventStream::new_assistant_stream();
    assert!(!stream.is_done());
}

#[test]
fn test_assistant_stream_done_event_completes() {
    let stream = AssistantMessageEventStream::new_assistant_stream();

    let msg = AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("gpt-4o")
        .content(vec![ContentBlock::Text(TextContent::new("Hello"))])
        .build()
        .unwrap();

    stream.push(AssistantMessageEvent::Start {
        partial: msg.clone(),
    });
    assert!(!stream.is_done());

    stream.push(AssistantMessageEvent::Done {
        reason: StopReason::Stop,
        message: msg,
    });
    assert!(stream.is_done());
}

#[test]
fn test_assistant_stream_error_event_completes() {
    let stream = AssistantMessageEventStream::new_assistant_stream();

    let msg = AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("gpt-4o")
        .error_message("Something went wrong")
        .stop_reason(StopReason::Error)
        .build()
        .unwrap();

    stream.push(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: msg,
    });
    assert!(stream.is_done());
}

#[tokio::test]
async fn test_assistant_stream_result() {
    let stream = AssistantMessageEventStream::new_assistant_stream();
    let stream2 = stream.clone();

    let msg = AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("gpt-4o")
        .content(vec![ContentBlock::Text(TextContent::new("Hello"))])
        .build()
        .unwrap();

    // Push done event from a task
    let msg_clone = msg.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        stream2.push(AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: msg_clone,
        });
    });

    let result = stream.result().await;
    assert_eq!(result.model, "gpt-4o");
    assert_eq!(result.text_content(), "Hello");
}

#[tokio::test]
async fn test_event_stream_result_available_after_completion() {
    let stream: EventStream<String, String> = EventStream::new(|s| s == "done", |s| s.clone());

    stream.push("done".to_string());

    let result = tokio::time::timeout(std::time::Duration::from_millis(100), stream.result())
        .await
        .unwrap();
    assert_eq!(result, "done");

    let result = tokio::time::timeout(std::time::Duration::from_millis(100), stream.result())
        .await
        .unwrap();
    assert_eq!(result, "done");
}

#[tokio::test]
async fn test_event_stream_result_multiple_waiters_complete() {
    let stream: EventStream<String, String> = EventStream::new(|s| s == "done", |s| s.clone());

    let mut handles = Vec::new();
    for _ in 0..16 {
        let waiter = stream.clone();
        handles.push(tokio::spawn(async move { waiter.result().await }));
    }

    tokio::task::yield_now().await;
    stream.push("done".to_string());

    for handle in handles {
        let result = tokio::time::timeout(std::time::Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result, "done");
    }
}

#[test]
fn test_assistant_stream_retrying_event_does_not_complete() {
    let stream = AssistantMessageEventStream::new_assistant_stream();

    stream.push(AssistantMessageEvent::Retrying {
        attempt: 1,
        max_retries: 2,
        delay_ms: 500,
        reason: "HTTP 503 Service Unavailable".to_string(),
        status: Some(503),
    });

    assert!(!stream.is_done());
}

#[test]
fn test_assistant_stream_events_not_yielded_after_done() {
    let stream = AssistantMessageEventStream::new_assistant_stream();

    let msg = AssistantMessage::builder()
        .api(Api::OpenAICompletions)
        .provider(Provider::OpenAI)
        .model("gpt-4o")
        .build()
        .unwrap();

    // Push events after done — they should be silently ignored
    stream.push(AssistantMessageEvent::Done {
        reason: StopReason::Stop,
        message: msg.clone(),
    });

    // This should be a no-op since stream is done
    stream.push(AssistantMessageEvent::Start { partial: msg });
    assert!(stream.is_done());
}
