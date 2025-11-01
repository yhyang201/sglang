# Step-Audio2 Streaming Parsers - Manual Testing Guide

This comprehensive guide provides detailed instructions for manually testing the streaming parsers for Step-Audio2 models, including both Audio Parser and Tool Call Detector.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Audio Parser Streaming Tests](#audio-parser-streaming-tests)
- [Tool Call Detector Streaming Tests](#tool-call-detector-streaming-tests)
- [Integration Tests](#integration-tests)
- [Performance Verification](#performance-verification)
- [Troubleshooting Guide](#troubleshooting-guide)

---

## Environment Setup

### Prerequisites
```bash
# Ensure you're in the sglang directory
cd /path/to/sglang

# Activate your Python environment
# source venv/bin/activate  # if using venv

# Install dependencies if needed
pip install partial-json-parser
```

### Quick Verification
```bash
# Run automated tests first to verify basic functionality
python python/sglang/srt/parser/test_audio_parser.py
python python/sglang/srt/function_call/test_step_audio2_detector.py
```

---

## Audio Parser Streaming Tests

### Test 1: Basic Streaming with Simple Chunks

**Objective**: Verify that the audio parser can handle token streams split at natural boundaries.

**Test Steps**:
```python
from sglang.srt.parser.audio_parser import StepAudio2AudioParser

# Create mock tokenizer
class MockTokenizer:
    def get_vocab(self):
        return {
            "<tts_start>": 151693,
            "<tts_end>": 151694,
            "<audio_0>": 151695,
            "<audio_1>": 151696,
            "<audio_2>": 151697,
            "<tts_pad>": 151700,
            "<audio_6561>": 158256,
        }

# Initialize parser
tokenizer = MockTokenizer()
parser = StepAudio2AudioParser(tokenizer)
parser._reset_streaming_state()

# Simulate streaming: 3 chunks
# Chunk 1: Text tokens
text_tokens, audio_tokens, other_tokens = parser.extract_tts_content_streaming(
    [100, 101, 102],  # Text token IDs
    is_tts_ta4_output=True
)
print(f"Chunk 1: text={text_tokens}, audio={audio_tokens}, other={other_tokens}")

# Chunk 2: Audio tokens
text_tokens, audio_tokens, other_tokens = parser.extract_tts_content_streaming(
    [151695, 151696, 151697],  # Audio token IDs
    is_tts_ta4_output=True
)
print(f"Chunk 2: text={text_tokens}, audio={audio_tokens}, other={other_tokens}")

# Chunk 3: End marker + other tokens
text_tokens, audio_tokens, other_tokens = parser.extract_tts_content_streaming(
    [151694, 200, 201],  # <tts_end> + other tokens
    is_tts_ta4_output=True
)
print(f"Chunk 3: text={text_tokens}, audio={audio_tokens}, other={other_tokens}")
```

**Expected Output**:
```
Chunk 1: text=[100, 101, 102], audio=[], other=[]
Chunk 2: text=[], audio=[151695, 151696, 151697], other=[]
Chunk 3: text=[], audio=[], other=[200, 201]
```

**Validation**:
- ✅ Text tokens are emitted immediately in first chunk
- ✅ Audio tokens are emitted immediately in second chunk
- ✅ After `<tts_end>`, remaining tokens are classified as "other"
- ✅ No tokens are lost or duplicated

---

### Test 2: Token-by-Token Streaming

**Objective**: Verify the parser can handle extremely fine-grained streaming (one token at a time).

**Test Steps**:
```python
parser._reset_streaming_state()

tokens = [100, 101, 151695, 151696, 151694, 200]
all_text = []
all_audio = []
all_other = []

for token in tokens:
    text, audio, other = parser.extract_tts_content_streaming(
        [token], is_tts_ta4_output=True
    )
    all_text.extend(text)
    all_audio.extend(audio)
    all_other.extend(other)
    print(f"Token {token}: text={text}, audio={audio}, other={other}")

print(f"\nFinal: text={all_text}, audio={all_audio}, other={all_other}")
```

**Expected Output**:
```
Token 100: text=[100], audio=[], other=[]
Token 101: text=[101], audio=[], other=[]
Token 151695: text=[], audio=[151695], other=[]
Token 151696: text=[], audio=[151696], other=[]
Token 151694: text=[], audio=[], other=[]  # <tts_end> is consumed
Token 200: text=[], audio=[], other=[200]

Final: text=[100, 101], audio=[151695, 151696], other=[200]
```

**Validation**:
- ✅ Each token is processed independently
- ✅ Classification remains correct across single-token boundaries
- ✅ State is maintained across multiple calls

---

### Test 3: Padding Token Filtering

**Objective**: Verify that padding tokens are filtered out during streaming.

**Test Steps**:
```python
parser._reset_streaming_state()

# Chunk with padding tokens interleaved
text, audio, other = parser.extract_tts_content_streaming(
    [100, 151700, 151695, 158256, 151696, 151694],
    is_tts_ta4_output=True
)

print(f"Result: text={text}, audio={audio}, other={other}")
print(f"151700 (<tts_pad>) filtered: {151700 not in text + audio + other}")
print(f"158256 (<audio_6561>) filtered: {158256 not in text + audio + other}")
```

**Expected Output**:
```
Result: text=[100], audio=[151695, 151696], other=[]
151700 (<tts_pad>) filtered: True
158256 (<audio_6561>) filtered: True
```

**Validation**:
- ✅ `<tts_pad>` (151700) is not in any output list
- ✅ `<audio_6561>` (158256) is not in any output list
- ✅ Non-padding tokens are preserved

---

### Test 4: Streaming vs Non-Streaming Consistency

**Objective**: Verify that streaming produces the same results as non-streaming when given the same input.

**Test Steps**:
```python
complete_output = [100, 101, 102, 151695, 151696, 151697, 151694, 200, 201]

# Non-streaming
text_ns, audio_ns, other_ns = parser.extract_tts_content_nonstreaming(
    complete_output, is_tts_ta4_output=True
)

# Streaming (3 chunks)
parser._reset_streaming_state()
text_s1, audio_s1, other_s1 = parser.extract_tts_content_streaming([100, 101, 102], True)
text_s2, audio_s2, other_s2 = parser.extract_tts_content_streaming([151695, 151696, 151697], True)
text_s3, audio_s3, other_s3 = parser.extract_tts_content_streaming([151694, 200, 201], True)

text_s = text_s1 + text_s2 + text_s3
audio_s = audio_s1 + audio_s2 + audio_s3
other_s = other_s1 + other_s2 + other_s3

print(f"Non-streaming: text={text_ns}, audio={audio_ns}, other={other_ns}")
print(f"Streaming:     text={text_s}, audio={audio_s}, other={other_s}")
print(f"Match: {text_s == text_ns and audio_s == audio_ns and other_s == other_ns}")
```

**Expected Output**:
```
Non-streaming: text=[100, 101, 102], audio=[151695, 151696, 151697], other=[200, 201]
Streaming:     text=[100, 101, 102], audio=[151695, 151696, 151697], other=[200, 201]
Match: True
```

**Validation**:
- ✅ All three lists match exactly
- ✅ Order is preserved
- ✅ No tokens are lost or duplicated

---

### Test 5: Edge Cases

#### 5a. Empty Chunks
```python
parser._reset_streaming_state()
text, audio, other = parser.extract_tts_content_streaming([], True)
print(f"Empty chunk: text={text}, audio={audio}, other={other}")
# Expected: text=[], audio=[], other=[]
```

#### 5b. Only Audio Tokens (No Text)
```python
parser._reset_streaming_state()
text, audio, other = parser.extract_tts_content_streaming(
    [151695, 151696, 151697, 151694], True
)
print(f"Audio only: text={text}, audio={audio}, other={other}")
# Expected: text=[], audio=[151695, 151696, 151697], other=[]
```

#### 5c. Only Text Tokens (No Audio)
```python
parser._reset_streaming_state()
text, audio, other = parser.extract_tts_content_streaming(
    [100, 101, 102, 151694, 200], True
)
print(f"Text only: text={text}, audio={audio}, other={other}")
# Expected: text=[100, 101, 102], audio=[], other=[200]
```

#### 5d. Non-TTS Output
```python
parser._reset_streaming_state()
text, audio, other = parser.extract_tts_content_streaming(
    [100, 101, 102], is_tts_ta4_output=False
)
print(f"Non-TTS: text={text}, audio={audio}, other={other}")
# Expected: text=[], audio=[], other=[100, 101, 102]
```

**Validation**:
- ✅ All edge cases handled without errors
- ✅ Output is semantically correct

---

## Tool Call Detector Streaming Tests

### Test 6: Single Tool Call Streaming

**Objective**: Verify that tool calls can be streamed incrementally (name first, then arguments).

**Test Steps**:
```python
from sglang.srt.function_call.step_audio2_detector import StepAudio2Detector
from sglang.srt.entrypoints.openai.protocol import Tool, Function

# Create test tools
tools = [
    Tool(type="function", function=Function(
        name="get_weather",
        description="Get weather",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}}
    ))
]

detector = StepAudio2Detector()

# Simulate streaming: tool call split into chunks
chunks = [
    "<tool_call>",
    "function\n",
    "get_weather\n",
    '{"location": ',
    '"Shanghai"}',
    "</tool_call>",
]

all_calls = []
for i, chunk in enumerate(chunks):
    result = detector.parse_streaming_increment(chunk, tools)
    print(f"Chunk {i+1} ({repr(chunk)}): calls={len(result.calls)}, normal_text={repr(result.normal_text)}")
    if result.calls:
        for call in result.calls:
            print(f"  → name={call.name}, params={repr(call.parameters)}")
            all_calls.append(call)

# Combine all parameters
full_params = "".join([call.parameters for call in all_calls if call.parameters])
print(f"\nFull parameters: {full_params}")
```

**Expected Output**:
```
Chunk 1 ('<tool_call>'): calls=0, normal_text=''
Chunk 2 ('function\n'): calls=0, normal_text=''
Chunk 3 ('get_weather\n'): calls=1, normal_text=''
  → name=get_weather, params=''
Chunk 4 ('{"location": '): calls=0, normal_text=''
Chunk 5 ('"Shanghai"}'): calls=1, normal_text=''
  → name=None, params='{"location":"Shanghai"}'
Chunk 6 ('</tool_call>'): calls=0, normal_text=''

Full parameters: {"location":"Shanghai"}
```

**Validation**:
- ✅ Tool name is sent first with empty parameters
- ✅ Arguments are streamed incrementally as they become available
- ✅ Complete JSON is reconstructed correctly
- ✅ `<tool_call>` and `</tool_call>` markers don't appear in output

---

### Test 7: Multiple Tool Calls Streaming

**Objective**: Verify that multiple sequential tool calls can be streamed correctly.

**Test Steps**:
```python
tools = [
    Tool(type="function", function=Function(name="get_weather", description="Get weather")),
    Tool(type="function", function=Function(name="search", description="Search")),
]

detector = StepAudio2Detector()

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

tool_names = [call.name for call in all_calls if call.name]
print(f"Tools called: {tool_names}")
print(f"Total call events: {len(all_calls)}")
```

**Expected Output**:
```
Tools called: ['get_weather', 'search']
Total call events: 4  # 2 tool names + 2 parameter chunks
```

**Validation**:
- ✅ Both tools are detected
- ✅ Each tool is processed independently
- ✅ State is properly reset between tools

---

### Test 8: Streaming with Normal Text

**Objective**: Verify that normal text before/after tool calls is handled correctly during streaming.

**Test Steps**:
```python
detector = StepAudio2Detector()
tools = [
    Tool(type="function", function=Function(name="get_weather", description="Get weather")),
]

chunks = [
    "Let me check ",
    "the weather. ",
    "<tool_call>function\n",
    "get_weather\n",
    '{"location": "Beijing"}',
    "</tool_call>",
    " Done!",
]

all_normal_text = []
all_calls = []

for chunk in chunks:
    result = detector.parse_streaming_increment(chunk, tools)
    if result.normal_text:
        all_normal_text.append(result.normal_text)
        print(f"Normal text: {repr(result.normal_text)}")
    if result.calls:
        all_calls.extend(result.calls)
        print(f"Tool call: {[c.name for c in result.calls if c.name]}")

full_text = "".join(all_normal_text)
print(f"\nFull normal text: {repr(full_text)}")
```

**Expected Output**:
```
Normal text: 'Let me check '
Normal text: 'the weather. '
Tool call: ['get_weather']
Normal text: ' Done!'

Full normal text: 'Let me check the weather.  Done!'
```

**Validation**:
- ✅ Normal text before tool call is emitted
- ✅ Normal text after tool call is emitted
- ✅ Tool call is correctly separated

---

### Test 9: JSON Split at Various Boundaries

**Objective**: Verify that JSON can be split at arbitrary points and still be parsed correctly.

**Test Steps**:
```python
detector = StepAudio2Detector()
tools = [
    Tool(type="function", function=Function(name="get_weather", description="Get weather")),
]

# Split JSON in the middle of a key, value, etc.
test_cases = [
    # Case 1: Split in middle of value
    ['<tool_call>function\nget_weather\n', '{"loc', 'ation": "Sh', 'anghai"}', '</tool_call>'],

    # Case 2: Split after colon
    ['<tool_call>function\nget_weather\n', '{"location":', ' "Tokyo"}', '</tool_call>'],

    # Case 3: Complete JSON in one chunk
    ['<tool_call>function\nget_weather\n', '{"location": "Paris"}</tool_call>'],
]

for i, chunks in enumerate(test_cases):
    print(f"\nCase {i+1}: {len(chunks)} chunks")
    detector = StepAudio2Detector()  # Reset detector

    all_calls = []
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        if result.calls:
            all_calls.extend(result.calls)

    tool_names = [call.name for call in all_calls if call.name]
    full_params = "".join([call.parameters for call in all_calls if call.parameters])

    print(f"  Tools: {tool_names}")
    print(f"  Parameters: {full_params}")
```

**Expected Output**:
```
Case 1: 4 chunks
  Tools: ['get_weather']
  Parameters: {"location":"Shanghai"}

Case 2: 4 chunks
  Tools: ['get_weather']
  Parameters: {"location":"Tokyo"}

Case 3: 3 chunks
  Tools: ['get_weather']
  Parameters: {"location":"Paris"}
```

**Validation**:
- ✅ All cases produce valid JSON
- ✅ Split points don't affect correctness
- ✅ No corruption or truncation

---

### Test 10: Invalid Function Name Handling

**Objective**: Verify that invalid/undefined function names are rejected during streaming.

**Test Steps**:
```python
detector = StepAudio2Detector()
tools = [
    Tool(type="function", function=Function(name="get_weather", description="Get weather")),
]

chunks = [
    "<tool_call>function\n",
    "undefined_function\n",
    '{"param": "value"}',
    "</tool_call>",
]

all_calls = []
for chunk in chunks:
    result = detector.parse_streaming_increment(chunk, tools)
    if result.calls:
        all_calls.extend(result.calls)
        print(f"Call: {[c.name for c in result.calls if c.name]}")

tool_names = [call.name for call in all_calls if call.name]
print(f"\nTools called: {tool_names}")
print(f"Undefined function rejected: {'undefined_function' not in tool_names}")
```

**Expected Output**:
```
Tools called: []
Undefined function rejected: True
```

**Validation**:
- ✅ Invalid function names are not included in output
- ✅ Warning is logged (check console)
- ✅ Parser recovers and can handle subsequent valid calls

---

### Test 11: Streaming vs Non-Streaming Consistency

**Objective**: Verify that streaming produces the same tool calls as non-streaming.

**Test Steps**:
```python
tools = [
    Tool(type="function", function=Function(name="get_weather", description="Get weather")),
]

complete_text = """<tool_call>function
get_weather
{"location": "Seoul"}</tool_call>"""

# Non-streaming
detector_ns = StepAudio2Detector()
result_ns = detector_ns.detect_and_parse(complete_text, tools)

# Streaming (split by lines)
detector_s = StepAudio2Detector()
lines = complete_text.split('\n')
all_calls_s = []

for line in lines:
    suffix = '\n' if line != lines[-1] else ''
    result = detector_s.parse_streaming_increment(line + suffix, tools)
    if result.calls:
        all_calls_s.extend(result.calls)

print(f"Non-streaming: {[c.name for c in result_ns.calls]}")
print(f"Streaming:     {[c.name for c in all_calls_s if c.name]}")

ns_params = result_ns.calls[0].parameters if result_ns.calls else ""
s_params = "".join([c.parameters for c in all_calls_s if c.parameters])

print(f"\nNon-streaming params: {ns_params}")
print(f"Streaming params:     {s_params}")
print(f"Match: {ns_params == s_params}")
```

**Expected Output**:
```
Non-streaming: ['get_weather']
Streaming:     ['get_weather']

Non-streaming params: {"location":"Seoul"}
Streaming params:     {"location":"Seoul"}
Match: True
```

**Validation**:
- ✅ Tool names match
- ✅ Parameters match
- ✅ Order is consistent

---

## Integration Tests

### Test 12: Combined Audio + Tool Call Streaming

**Objective**: Simulate a realistic scenario where both TTS output and tool calls are streamed together.

**Test Scenario**: Model generates TTS audio response, then makes a tool call.

**Test Steps**:
```python
# This would require integration with the serving layer
# For manual testing, you can simulate this with a Python script:

from sglang.srt.parser.audio_parser import StepAudio2AudioParser
from sglang.srt.function_call.step_audio2_detector import StepAudio2Detector

# Mock tokenizer and tools
# ... (use previous setup)

# Simulate mixed output: TTS first, then tool call
output_chunks = [
    # TTS section
    [100, 101, 151695, 151696, 151694],  # text + audio + end
    # Tool call section (as text)
    "<tool_call>function\nget_weather\n",
    '{"location": "NYC"}',
    "</tool_call>",
]

# Process audio tokens
audio_parser = StepAudio2AudioParser(tokenizer)
audio_parser._reset_streaming_state()

text_tokens, audio_tokens, other_tokens = audio_parser.extract_tts_content_streaming(
    output_chunks[0], is_tts_ta4_output=True
)

print(f"Audio section: text={text_tokens}, audio={audio_tokens}, other={other_tokens}")

# Process tool call text
tool_detector = StepAudio2Detector()
for chunk in output_chunks[1:]:
    result = tool_detector.parse_streaming_increment(chunk, tools)
    if result.calls:
        print(f"Tool call: {[c.name for c in result.calls if c.name]}")
```

**Expected Behavior**:
- ✅ Audio section is parsed completely
- ✅ Tool call section is parsed after audio
- ✅ No interference between parsers

---

## Performance Verification

### Test 13: Latency Measurement

**Objective**: Measure the overhead of streaming parsing compared to batch processing.

**Test Steps**:
```python
import time

# Test Audio Parser
parser = StepAudio2AudioParser(tokenizer)

# Batch processing
start = time.time()
for _ in range(1000):
    parser.extract_tts_content_nonstreaming([100, 101, 151695, 151696, 151694], True)
batch_time = time.time() - start

# Streaming processing
start = time.time()
for _ in range(1000):
    parser._reset_streaming_state()
    parser.extract_tts_content_streaming([100, 101], True)
    parser.extract_tts_content_streaming([151695, 151696], True)
    parser.extract_tts_content_streaming([151694], True)
streaming_time = time.time() - start

print(f"Batch processing:     {batch_time:.4f}s (1000 iterations)")
print(f"Streaming processing: {streaming_time:.4f}s (1000 iterations)")
print(f"Overhead: {((streaming_time - batch_time) / batch_time * 100):.1f}%")
```

**Expected Performance**:
- Streaming overhead should be < 50% compared to batch processing
- Absolute time per operation should be < 1ms

**Validation**:
- ✅ Performance is acceptable for production use
- ✅ No memory leaks during extended runs

---

### Test 14: Memory Usage

**Objective**: Verify that streaming doesn't accumulate unbounded memory.

**Test Steps**:
```python
import tracemalloc

tracemalloc.start()

parser = StepAudio2AudioParser(tokenizer)

# Simulate long streaming session
for i in range(10000):
    if i % 1000 == 0:
        parser._reset_streaming_state()

    parser.extract_tts_content_streaming([100 + (i % 100)], True)

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Current memory: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory:    {peak / 1024 / 1024:.2f} MB")
```

**Expected Result**:
- Memory usage should remain relatively constant
- Peak memory should be < 10 MB for this test

**Validation**:
- ✅ No memory leaks
- ✅ Buffers are properly cleared

---

## Troubleshooting Guide

### Issue 1: Streaming Results Don't Match Non-Streaming

**Symptoms**:
- Different tokens in output
- Missing tokens
- Extra tokens

**Debug Steps**:
1. Check that `_reset_streaming_state()` is called before each test
2. Print intermediate results for each chunk
3. Compare token-by-token

**Example Debug Code**:
```python
# Add detailed logging
parser._reset_streaming_state()

for i, chunk in enumerate(chunks):
    result = parser.extract_tts_content_streaming(chunk, True)
    print(f"Chunk {i}: input={chunk}")
    print(f"  Output: text={result[0]}, audio={result[1]}, other={result[2]}")
    print(f"  Buffer state: {parser._stream_buffer_tokens}")
```

---

### Issue 2: Tool Call Parameters Are Incomplete

**Symptoms**:
- JSON is truncated
- Missing key-value pairs
- Parameters only partially streamed

**Debug Steps**:
1. Check if partial JSON parsing is working:
```python
from sglang.srt.function_call.utils import _partial_json_loads
from partial_json_parser.core.options import Allow

# Test partial parsing directly
test_json = '{"location": "Sha'
try:
    obj, consumed = _partial_json_loads(test_json, Allow.ALL)
    print(f"Parsed: {obj}, consumed: {consumed}")
except Exception as e:
    print(f"Error: {e}")
```

2. Verify that arguments are being accumulated:
```python
# Add logging in parse_streaming_increment
print(f"Args buffer: {self._args_buffer}")
print(f"Previous sent: {self._previous_args_sent}")
```

---

### Issue 3: State Not Resetting Between Calls

**Symptoms**:
- Second streaming session has leftover data from first
- Tool IDs are incorrect
- Buffer contains old data

**Solution**:
Always call reset methods before starting new session:

```python
# For Audio Parser
parser._reset_streaming_state()

# For Tool Detector
detector._reset_tool_state()
detector._buffer = ""
```

---

### Issue 4: Invalid Function Names Not Rejected

**Symptoms**:
- Undefined tools appear in output
- No warning logged

**Debug Steps**:
1. Verify tool indices are built:
```python
print(f"Tool indices: {detector._tool_indices}")
```

2. Check logging level:
```python
import logging
logging.basicConfig(level=logging.WARNING)
```

---

## Test Checklist

Use this checklist to verify all tests have been run:

### Audio Parser
- [ ] Test 1: Basic streaming with simple chunks
- [ ] Test 2: Token-by-token streaming
- [ ] Test 3: Padding token filtering
- [ ] Test 4: Streaming vs non-streaming consistency
- [ ] Test 5a: Empty chunks
- [ ] Test 5b: Only audio tokens
- [ ] Test 5c: Only text tokens
- [ ] Test 5d: Non-TTS output

### Tool Call Detector
- [ ] Test 6: Single tool call streaming
- [ ] Test 7: Multiple tool calls streaming
- [ ] Test 8: Streaming with normal text
- [ ] Test 9: JSON split at various boundaries
- [ ] Test 10: Invalid function name handling
- [ ] Test 11: Streaming vs non-streaming consistency

### Integration & Performance
- [ ] Test 12: Combined audio + tool call streaming
- [ ] Test 13: Latency measurement
- [ ] Test 14: Memory usage

---

## Success Criteria

All tests pass if:

1. **Correctness**:
   - ✅ Streaming and non-streaming produce identical results
   - ✅ No tokens are lost or duplicated
   - ✅ Classification (text/audio/other) is always correct

2. **Robustness**:
   - ✅ All edge cases handled without errors
   - ✅ Invalid input is rejected gracefully
   - ✅ State resets work correctly

3. **Performance**:
   - ✅ Streaming overhead < 50%
   - ✅ No memory leaks
   - ✅ Suitable for production use

4. **Functionality**:
   - ✅ Tool names streamed before parameters
   - ✅ JSON parsed incrementally
   - ✅ Multiple tool calls supported

---

## Additional Resources

- **Automated Tests**: See `python/sglang/srt/parser/test_audio_parser.py` and `python/sglang/srt/function_call/test_step_audio2_detector.py`
- **API Documentation**: See `STEP_AUDIO2_PARSERS.md`
- **Integration Guide**: See `STEP_AUDIO2_INTEGRATION_GUIDE.md`

---

## Reporting Issues

If you find issues during manual testing:

1. Note the specific test case that failed
2. Capture the full input and output
3. Check console for warnings/errors
4. Include Python version and dependency versions
5. Report in project issues with "streaming" label

---

**Last Updated**: 2025-01-XX
**Version**: 1.0
**Author**: Streaming Parser Implementation Team
