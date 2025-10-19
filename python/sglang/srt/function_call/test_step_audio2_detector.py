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
    test_registration()
    print("\n✅ All Step-Audio2 detector tests passed!")
