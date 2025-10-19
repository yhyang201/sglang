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

## Testing

Test files are provided:

1. **Audio Parser Test**: `python/sglang/srt/parser/test_audio_parser.py`
   ```bash
   python python/sglang/srt/parser/test_audio_parser.py
   ```

2. **Tool Detector Test**: `python/sglang/srt/function_call/test_step_audio2_detector.py`
   ```bash
   python python/sglang/srt/function_call/test_step_audio2_detector.py
   ```

## Current Limitations (MVP)

1. **No Streaming Support**:
   - Audio parser only supports non-streaming parsing
   - Tool parser has simplified streaming implementation
   - Full streaming support to be added in future iterations

2. **Audio Parser Integration**:
   - Parser is implemented but not yet integrated into output processing pipeline
   - Integration with OpenAI API endpoint needs to be added

3. **EBNF Grammar**:
   - Tool parser EBNF is basic
   - May need refinement for production use

## Future Enhancements

1. **Streaming Support**:
   - Implement `parse_stream_chunk()` for audio parser
   - Enhance streaming support in tool parser
   - Support incremental audio token generation

2. **Integration**:
   - Integrate audio parser into OpenAI API endpoint
   - Add TTS content to response format
   - Support audio token generation in streaming mode

3. **Optimization**:
   - Optimize token parsing performance
   - Add caching for frequently used operations
   - Improve error handling and logging

4. **Testing**:
   - Add integration tests
   - Add performance benchmarks
   - Test with actual Step-Audio2 model

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
