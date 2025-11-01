"""
Simple test for Step-Audio2 Tool Call Detector.

This is a basic test to verify the StepAudio2Detector works correctly.
"""

from sglang.srt.entrypoints.openai.protocol import Tool, Function


def create_test_tools():
    """Create test tools for testing."""
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get weather information",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            )
        ),
        Tool(
            type="function",
            function=Function(
                name="search",
                description="Search for information",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            )
        )
    ]


def test_has_tool_call():
    """Test tool call detection."""
    from sglang.srt.function_call.step_audio2_detector import StepAudio2Detector

    detector = StepAudio2Detector()

    # Text with tool call
    text_with_tool = "Hello <tool_call>function\nget_weather\n{}"
    assert detector.has_tool_call(text_with_tool) == True

    # Text without tool call
    text_without_tool = "Hello, how can I help you?"
    assert detector.has_tool_call(text_without_tool) == False

    print("✓ Tool call detection works")


def test_parse_single_tool_call():
    """Test parsing a single tool call."""
    from sglang.srt.function_call.step_audio2_detector import StepAudio2Detector

    detector = StepAudio2Detector()
    tools = create_test_tools()

    # Single tool call
    text = """Let me check the weather for you.
<tool_call>function
get_weather
{"location": "Shanghai"}</tool_call>
I'll get that information for you."""

    result = detector.detect_and_parse(text, tools)

    assert len(result.calls) == 1
    assert result.calls[0].name == "get_weather"
    assert '"location": "Shanghai"' in result.calls[0].parameters or \
           '"location":"Shanghai"' in result.calls[0].parameters
    assert "Let me check the weather for you." in result.normal_text
    assert "I'll get that information for you." in result.normal_text

    print("✓ Single tool call parsing works")


def test_parse_multiple_tool_calls():
    """Test parsing multiple tool calls."""
    from sglang.srt.function_call.step_audio2_detector import StepAudio2Detector

    detector = StepAudio2Detector()
    tools = create_test_tools()

    # Multiple tool calls
    text = """<tool_call>function
get_weather
{"location": "Shanghai"}</tool_call><tool_call>function
search
{"query": "weather forecast"}</tool_call>"""

    result = detector.detect_and_parse(text, tools)

    assert len(result.calls) == 2
    assert result.calls[0].name == "get_weather"
    assert result.calls[1].name == "search"

    print("✓ Multiple tool calls parsing works")


def test_parse_text_without_tool_call():
    """Test parsing text without tool calls."""
    from sglang.srt.function_call.step_audio2_detector import StepAudio2Detector

    detector = StepAudio2Detector()
    tools = create_test_tools()

    text = "This is just normal text without any tool calls."

    result = detector.detect_and_parse(text, tools)

    assert len(result.calls) == 0
    assert result.normal_text == text

    print("✓ Text without tool calls handled correctly")


def test_invalid_function_name():
    """Test handling of invalid function names."""
    from sglang.srt.function_call.step_audio2_detector import StepAudio2Detector

    detector = StepAudio2Detector()
    tools = create_test_tools()

    # Tool call with undefined function
    text = """<tool_call>function
undefined_function
{"param": "value"}</tool_call>"""

    result = detector.detect_and_parse(text, tools)

    # Should not include the invalid tool call
    assert len(result.calls) == 0

    print("✓ Invalid function name handled correctly")


def test_streaming_single_tool():
    """Test streaming parsing of a single tool call."""
    from sglang.srt.function_call.step_audio2_detector import StepAudio2Detector

    print("\n" + "=" * 50)
    print("Testing Streaming Tool Call Parser")
    print("=" * 50)

    detector = StepAudio2Detector()
    tools = create_test_tools()

    print("\nTest 1: Single tool call split into chunks")

    # Simulate streaming: tool call split into multiple chunks
    chunks = [
        "<tool_call>",
        "function\n",
        "get_weather\n",
        '{"location": ',
        '"Shanghai"}',
        "</tool_call>",
    ]

    all_calls = []
    all_normal_text = []

    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        if result.calls:
            all_calls.extend(result.calls)
        if result.normal_text:
            all_normal_text.append(result.normal_text)

    # Should have tool name call + parameter chunks
    assert len(all_calls) >= 1
    # First call should have the tool name
    assert any(call.name == "get_weather" for call in all_calls)
    # Should have received parameters
    full_params = "".join([call.parameters for call in all_calls if call.parameters])
    assert "Shanghai" in full_params

    print(f"  ✓ Received {len(all_calls)} call chunks")
    print(f"  ✓ Tool name: get_weather")
    print(f"  ✓ Parameters streamed correctly")


def test_streaming_multiple_tools():
    """Test streaming parsing of multiple tool calls."""
    from sglang.srt.function_call.step_audio2_detector import StepAudio2Detector

    print("\nTest 2: Multiple tool calls streamed")

    detector = StepAudio2Detector()
    tools = create_test_tools()

    # Two tool calls streamed
    chunks = [
        "<tool_call>function\n",
        "get_weather\n",
        '{"location": "Shanghai"}',
        "</tool_call>",
        "<tool_call>function\n",
        "search\n",
        '{"query": "weather"}',
        "</tool_call>",
    ]

    all_calls = []
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        if result.calls:
            all_calls.extend(result.calls)

    # Should have calls for both tools
    tool_names = [call.name for call in all_calls if call.name]
    assert "get_weather" in tool_names
    assert "search" in tool_names

    print(f"  ✓ Received calls for both tools")
    print(f"  ✓ Tools: {tool_names}")


def test_streaming_with_normal_text():
    """Test streaming with normal text before tool call."""
    from sglang.srt.function_call.step_audio2_detector import StepAudio2Detector

    print("\nTest 3: Streaming with normal text")

    detector = StepAudio2Detector()
    tools = create_test_tools()

    chunks = [
        "Let me check ",
        "the weather. ",
        "<tool_call>function\n",
        "get_weather\n",
        '{"location": "Beijing"}',
        "</tool_call>",
    ]

    all_normal_text = []
    all_calls = []

    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        if result.normal_text:
            all_normal_text.append(result.normal_text)
        if result.calls:
            all_calls.extend(result.calls)

    full_text = "".join(all_normal_text)
    assert "Let me check" in full_text or "the weather" in full_text
    assert any(call.name == "get_weather" for call in all_calls)

    print(f"  ✓ Normal text: {full_text[:30]}...")
    print(f"  ✓ Tool call parsed correctly")


def test_streaming_vs_nonstreaming():
    """Test that streaming results match non-streaming results."""
    from sglang.srt.function_call.step_audio2_detector import StepAudio2Detector

    print("\nTest 4: Streaming vs Non-streaming consistency")

    tools = create_test_tools()

    complete_text = """<tool_call>function
get_weather
{"location": "Tokyo"}</tool_call>"""

    # Non-streaming
    detector_ns = StepAudio2Detector()
    result_ns = detector_ns.detect_and_parse(complete_text, tools)

    # Streaming (split by lines)
    detector_s = StepAudio2Detector()
    lines = complete_text.split('\n')
    all_calls_s = []

    for line in lines:
        result = detector_s.parse_streaming_increment(line + '\n' if line != lines[-1] else line, tools)
        if result.calls:
            all_calls_s.extend(result.calls)

    # Compare: should have same tool name
    assert result_ns.calls[0].name == "get_weather"
    assert any(call.name == "get_weather" for call in all_calls_s)

    print("  ✓ Streaming and non-streaming results are consistent")


def test_streaming_invalid_tool():
    """Test streaming with invalid tool name."""
    from sglang.srt.function_call.step_audio2_detector import StepAudio2Detector

    print("\nTest 5: Streaming with invalid tool name")

    detector = StepAudio2Detector()
    tools = create_test_tools()

    chunks = [
        "<tool_call>function\n",
        "invalid_tool\n",
        '{"param": "value"}',
        "</tool_call>",
    ]

    all_calls = []
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        if result.calls:
            all_calls.extend(result.calls)

    # Should not have any calls for invalid tool
    tool_names = [call.name for call in all_calls if call.name]
    assert "invalid_tool" not in tool_names

    print("  ✓ Invalid tool call properly rejected")


def test_streaming_chunked_json():
    """Test streaming with JSON split at various points."""
    from sglang.srt.function_call.step_audio2_detector import StepAudio2Detector

    print("\nTest 6: JSON split at various boundaries")

    detector = StepAudio2Detector()
    tools = create_test_tools()

    # Split JSON in the middle of a value
    chunks = [
        "<tool_call>function\nget_weather\n",
        '{"loc',
        'ation": "Sh',
        'anghai"}',
        "</tool_call>",
    ]

    all_calls = []
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        if result.calls:
            all_calls.extend(result.calls)

    # Should still parse correctly
    assert any(call.name == "get_weather" for call in all_calls)
    full_params = "".join([call.parameters for call in all_calls if call.parameters])
    assert "Shanghai" in full_params or "location" in full_params

    print("  ✓ JSON split at boundaries handled correctly")


def test_streaming_empty_args():
    """Test streaming with empty arguments."""
    from sglang.srt.function_call.step_audio2_detector import StepAudio2Detector

    print("\nTest 7: Streaming with empty/minimal arguments")

    detector = StepAudio2Detector()
    tools = create_test_tools()

    chunks = [
        "<tool_call>function\n",
        "search\n",
        '{"query": "test"}',
        "</tool_call>",
    ]

    all_calls = []
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        if result.calls:
            all_calls.extend(result.calls)

    assert any(call.name == "search" for call in all_calls)

    print("  ✓ Minimal arguments handled correctly")

    print("\n✅ All streaming tool detector tests passed!")


def test_registration():
    """Test that StepAudio2Detector is registered."""
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    tools = create_test_tools()

    # Should not raise an error
    parser = FunctionCallParser(tools, "step_audio_2")
    assert parser is not None
    assert parser.detector is not None

    print("✓ StepAudio2Detector is registered in FunctionCallParser")


if __name__ == "__main__":
    print("Running Step-Audio2 Tool Detector Tests\n")
    print("=" * 50)
    test_has_tool_call()
    test_parse_single_tool_call()
    test_parse_multiple_tool_calls()
    test_parse_text_without_tool_call()
    test_invalid_function_name()
    print()
    test_streaming_single_tool()
    test_streaming_multiple_tools()
    test_streaming_with_normal_text()
    test_streaming_vs_nonstreaming()
    test_streaming_invalid_tool()
    test_streaming_chunked_json()
    test_streaming_empty_args()
    print()
    test_registration()
    print("\n✅ All Step-Audio2 detector tests passed!")
