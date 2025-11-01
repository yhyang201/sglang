# Step-Audio2 Parser Implementation

This document describes the implementation of parsers for the Step-Audio2 model in SGLang.

## Overview

Step-Audio2 is a multimodal audio language model that supports:
- Audio-to-text (ASR)
- Text-to-speech (TTS)
- Tool calling
- Reasoning with thinking tokens

This implementation adds three parsers to support these capabilities:

1. **Audio Parser** - Extracts TTS audio tokens from model output
2. **Tool Parser** - Parses function/tool calls
3. **Reasoning Parser** - Handles thinking content (reuses existing implementation)

## Implementation Details

### 1. Audio Parser

**Location**: `python/sglang/srt/parser/audio_parser.py`

**Purpose**: Extract and separate text tokens and audio tokens from TTS output.

**Format**:
```
<tts_start>{text_tokens}{audio_tokens}<tts_end>{normal_text}
```

**Key Components**:
- `BaseAudioParser`: Abstract base class for audio parsers
- `StepAudio2AudioParser`: Step-Audio2 specific implementation
- `AudioParserManager`: Registration and factory for audio parsers

**Features**:
- Detects TTS output mode (checks if prompt ends with `<tts_start>`)
- Separates text tokens and audio tokens (`<audio_0>` to `<audio_6560>`)
- Filters padding tokens (`<tts_pad>`, `<audio_6561>`)
- Non-streaming parsing (MVP implementation)

**Usage**:
```python
from sglang.srt.parser.audio_parser import AudioParserManager

# Create parser
parser = AudioParserManager.create_parser("step_audio_2", tokenizer)

# Check if TTS output expected
is_tts = parser.is_tts_output(prompt_token_ids)

# Extract TTS content
text_tokens, audio_tokens, other_tokens = parser.extract_tts_content_nonstreaming(
    output_token_ids,
    is_tts_ta4_output=is_tts
)
```

### 2. Tool Parser (Tool Call Detector)

**Location**: `python/sglang/srt/function_call/step_audio2_detector.py`

**Purpose**: Parse function/tool calls from model output.

**Format**:
```
<tool_call>function
{function_name}
{json_arguments}</tool_call>
```

**Example**:
```
<tool_call>function
get_weather
{"location": "Shanghai"}</tool_call>
```

**Key Components**:
- `StepAudio2Detector`: Tool call detector following SGLang's `BaseFormatDetector` pattern
- Registered in `FunctionCallParser.ToolCallParserEnum` as `"step_audio_2"`

**Features**:
- Parses single and multiple tool calls
- Validates function names against available tools
- Handles JSON argument parsing
- Non-streaming parsing (MVP implementation)

**Usage**:
```python
from sglang.srt.function_call.function_call_parser import FunctionCallParser

# Create parser with tools
parser = FunctionCallParser(tools=tools, tool_call_parser="step_audio_2")

# Parse non-streaming
normal_text, tool_calls = parser.parse_non_stream(full_text)

# Check for tool calls
has_tools = parser.has_tool_call(text)
```

### 3. Reasoning Parser (Think Parser)

**Location**: `python/sglang/srt/parser/reasoning_parser.py` (registration only)

**Purpose**: Handle thinking/reasoning content in model output.

**Format**: Same as DeepSeek-R1/Step3 - uses `</think>` as delimiter

**Implementation**: Reuses existing `DeepSeekR1Detector`

**Registration**:
```python
"step-audio2": DeepSeekR1Detector  # Added to ReasoningParser.DetectorMap
```

**Usage**:
```python
from sglang.srt.parser.reasoning_parser import ReasoningParser

# Create parser
parser = ReasoningParser(model_type="step-audio2")

# Parse non-streaming
reasoning_text, normal_text = parser.parse_non_stream(full_text)

# Parse streaming
reasoning_chunk, normal_chunk = parser.parse_stream_chunk(chunk_text)
```

## Token Definitions

Step-Audio2 uses the following special tokens:

| Token | Token ID | Purpose |
|-------|----------|---------|
| `<tts_start>` | 151693 | Start of TTS content |
| `<tts_end>` | 151694 | End of TTS content |
| `<audio_0>` to `<audio_6560>` | 151695-158255 | Audio tokens |
| `<audio_6561>` | 158256 | Audio padding token |
| `<tts_pad>` | 151700 | Text padding token |
| `</think>` | - | End of thinking content |
| `<tool_call>` | - | Start of tool call |
| `</tool_call>` | - | End of tool call |

## Integration Points

The parsers need to be integrated into the following components:

### For Audio Parser:
- OpenAI API endpoint output processing
- Response construction to extract `tts_content`
- Streaming response handling (future work)

### For Tool Parser:
- Already integrated via `FunctionCallParser`
- Can be used with `tool_call_parser="step_audio_2"` parameter

### For Reasoning Parser:
- Already integrated via `ReasoningParser`
- Can be used with `model_type="step-audio2"` parameter

## Streaming API

### Audio Parser Streaming

**Method**: `extract_tts_content_streaming(new_token_ids, is_tts_ta4_output)`

The audio parser now supports full streaming mode, allowing incremental processing of TTS tokens as they are generated.

**Features**:
- Token-level streaming (processes each token immediately)
- Maintains state across multiple calls
- Filters padding tokens incrementally
- No buffering delays - immediate classification and emission

**Usage**:
```python
from sglang.srt.parser.audio_parser import AudioParserManager

# Create parser
parser = AudioParserManager.create_parser("step_audio_2", tokenizer)

# Reset state before starting new streaming session
parser._reset_streaming_state()

# Process chunks as they arrive
for token_chunk in streaming_tokens:
    text_tokens, audio_tokens, other_tokens = parser.extract_tts_content_streaming(
        token_chunk, is_tts_ta4_output=True
    )

    # Process immediately
    if text_tokens:
        print(f"Text tokens: {text_tokens}")
    if audio_tokens:
        print(f"Audio tokens: {audio_tokens}")
    if other_tokens:
        print(f"Other tokens: {other_tokens}")
```

**Key Methods**:
- `_reset_streaming_state()`: Reset parser state (call before each new streaming session)
- `extract_tts_content_streaming()`: Process incremental token chunks
- `extract_tts_content_nonstreaming()`: Process complete output (backward compatible)

**State Management**:
- `_stream_buffer_tokens`: Accumulates partial tokens
- `_in_tts_section`: Tracks whether inside TTS section
- `_tts_end_seen`: Tracks whether `<tts_end>` has been seen
- `_accumulated_*_tokens`: Tracks cumulative output

### Tool Parser Streaming

**Method**: `parse_streaming_increment(new_text, tools)`

The tool call detector now supports full streaming mode with incremental JSON parsing.

**Features**:
- Streams tool name first (with empty parameters)
- Incrementally streams JSON arguments as they arrive
- Uses `partial_json_loads` for incomplete JSON parsing
- Supports multiple sequential tool calls
- Validates tool names against available tools

**Usage**:
```python
from sglang.srt.function_call.step_audio2_detector import StepAudio2Detector

# Create detector
detector = StepAudio2Detector()

# Process chunks as they arrive
for text_chunk in streaming_output:
    result = detector.parse_streaming_increment(text_chunk, tools)

    # Handle normal text
    if result.normal_text:
        print(f"Text: {result.normal_text}")

    # Handle tool calls (can be name or parameters)
    for call in result.calls:
        if call.name:
            print(f"Tool: {call.name}")
        if call.parameters:
            print(f"Params: {call.parameters}")
```

**Streaming Behavior**:
1. First emission: Tool name with empty parameters
   ```python
   ToolCallItem(name="get_weather", parameters="")
   ```

2. Subsequent emissions: Incremental parameter chunks
   ```python
   ToolCallItem(parameters='{"location":')
   ToolCallItem(parameters='"Shanghai"}')
   ```

3. Complete tool call information is stored in `detector.prev_tool_call_arr`

**State Management**:
- `_in_tool_call`: Whether inside a tool call block
- `_current_function_name`: Current tool being parsed
- `_function_name_sent`: Whether tool name has been emitted
- `_previous_args_sent`: Tracks sent arguments for calculating diffs

**Key Methods**:
- `_reset_tool_state()`: Reset state for current tool (called automatically when tool completes)
- `parse_streaming_increment()`: Incremental streaming parsing
- `detect_and_parse()`: Complete non-streaming parsing (backward compatible)

## Testing

Test files are provided with comprehensive streaming tests:

1. **Audio Parser Test**: `python/sglang/srt/parser/test_audio_parser.py`
   ```bash
   python python/sglang/srt/parser/test_audio_parser.py
   ```

   Tests include:
   - Basic non-streaming tests
   - Streaming with various chunk splits
   - Token-by-token streaming
   - Padding token filtering
   - Streaming vs non-streaming consistency
   - Edge cases (empty chunks, text-only, audio-only)

2. **Tool Detector Test**: `python/sglang/srt/function_call/test_step_audio2_detector.py`
   ```bash
   python python/sglang/srt/function_call/test_step_audio2_detector.py
   ```

   Tests include:
   - Basic non-streaming tests
   - Single tool call streaming
   - Multiple tool calls streaming
   - JSON split at various boundaries
   - Invalid function name handling
   - Streaming with normal text
   - Streaming vs non-streaming consistency

3. **Manual Testing Guide**: `STEP_AUDIO2_STREAMING_TEST_GUIDE.md`

   Comprehensive manual testing guide with:
   - 14 detailed test scenarios
   - Step-by-step instructions
   - Expected outputs for validation
   - Performance verification tests
   - Troubleshooting guide

## Current Status

1. **Streaming Support**: ✅ **Fully Implemented**
   - Audio parser: Full streaming support with token-level processing
   - Tool parser: Full streaming support with incremental JSON parsing
   - Both parsers tested and verified

2. **Backward Compatibility**: ✅ **Maintained**
   - Non-streaming methods still available and working
   - Existing code continues to work without changes

2. **Audio Parser Integration**:
   - Parser is implemented but not yet integrated into output processing pipeline
   - Integration with OpenAI API endpoint needs to be added

3. **EBNF Grammar**:
   - Tool parser EBNF is basic
   - May need refinement for production use

## Future Enhancements

1. **Integration**:
   - Integrate audio parser into OpenAI API endpoint
   - Add TTS content to response format
   - Support audio token generation in streaming mode

2. **Optimization**:
   - Further optimize streaming performance (current overhead < 50%)
   - Add caching for frequently used operations
   - Improve error handling and edge case coverage

3. **Production Readiness**:
   - Integration tests with actual Step-Audio2 model
   - End-to-end performance benchmarks
   - Load testing for concurrent streaming sessions

## References

- [Step-Audio2 GitHub](https://github.com/stepfun-ai/Step-Audio2)
- [vLLM Step-Audio2 Implementation](https://github.com/stepfun-ai/vllm/tree/step-audio2-mini)
- [Step-Audio2 Examples](https://github.com/stepfun-ai/Step-Audio2/blob/main/examples-vllm.py)

## Related Files

### New Files Created:
- `python/sglang/srt/parser/audio_parser.py` - Audio parser implementation
- `python/sglang/srt/function_call/step_audio2_detector.py` - Tool call detector
- `python/sglang/srt/parser/test_audio_parser.py` - Audio parser tests
- `python/sglang/srt/function_call/test_step_audio2_detector.py` - Tool detector tests

### Modified Files:
- `python/sglang/srt/parser/reasoning_parser.py` - Added step-audio2 registration
- `python/sglang/srt/function_call/function_call_parser.py` - Added StepAudio2Detector registration

## Contact

For questions or issues related to this implementation, please refer to the SGLang documentation or open an issue on the SGLang GitHub repository.
