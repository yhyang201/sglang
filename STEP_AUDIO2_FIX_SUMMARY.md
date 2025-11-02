# Step-Audio2 Parser Fix Summary

## Problem Statement

The Step-Audio2 model integration had critical parser conflicts that caused:
1. **TTS content appearing in `reasoning_content`**: Audio tokens were incorrectly parsed as reasoning
2. **Tool calls appearing in `reasoning_content`**: Tool call JSON was treated as reasoning
3. **User confusion**: Required manual `<tts_start>` token insertion in messages

## Root Cause

The `DeepSeekR1Detector` reasoning parser used `force_reasoning=True`, which caused it to:
- Consume ALL content when no `</think>` tag was present
- Run BEFORE audio and tool parsers
- Treat audio tokens and tool calls as reasoning content

## Solution Implemented

### 1. Created Custom `StepAudio2Detector`
**File**: `/sgl-workspace/sglang/python/sglang/srt/parser/reasoning_parser.py:195-279`

```python
class StepAudio2Detector(BaseReasoningFormatDetector):
    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=False,  # Key fix: don't be greedy
            stream_reasoning=stream_reasoning,
        )
```

**Key Features**:
- `force_reasoning=False`: Only treats content between `<think></think>` as reasoning
- `_remove_protected_content()`: Skips tool call and TTS markers
- Proper null checking to avoid crashes

### 2. Fixed Parser Execution Order
**File**: `/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:957-1009`

**Old Order** (Broken):
```
Reasoning Parser (consumes everything) →
Tool Parser (gets nothing) →
Audio Parser (gets nothing)
```

**New Order** (Fixed):
```
Audio Parser (extracts TTS, returns remaining) →
Tool Parser (extracts tools, returns remaining) →
Reasoning Parser (processes remaining only)
```

### 3. Added `tts_output` Parameter
**File**: `/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/protocol.py:498-501`

```python
class ChatCompletionRequest(BaseModel):
    ...
    tts_output: bool = Field(
        default=False,
        description="Enable text-to-speech output"
    )
```

**Auto-append Logic** (`serving_chat.py:555-563`):
```python
if request.tts_output:
    tts_start_token = "<tts_start>"
    tts_start_id = self.tokenizer_manager.tokenizer.vocab.get(tts_start_token)
    if tts_start_id is not None:
        prompt_ids.append(tts_start_id)
```

### 4. Improved TTS Detection
**File**: `/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:110-135`

**Priority**:
1. Check `request.tts_output` parameter (recommended)
2. Fallback to checking if prompt ends with `<tts_start>` token (legacy)

## API Changes

### Before (Manual Token - Still Works)
```python
response = requests.post(url, json={
    "messages": [
        {"role": "user", "content": "Say hello"},
        {"role": "assistant", "content": "<tts_start>", "eot": False}
    ],
    ...
})
```

### After (Recommended - New Parameter)
```python
response = requests.post(url, json={
    "messages": [
        {"role": "user", "content": "Say hello"}
    ],
    "tts_output": True,  # Automatically adds <tts_start>
    ...
})
```

## Files Modified

1. **`audio_parser.py`**
   - Added boundary detection logic in `extract_tts_content` method (lines 276-364)
   - Detects and extracts protected regions (`<tool_call>`, `<think>`)
   - Re-tokenizes TTS segments and protected segments separately
   - Returns protected content in `other_tokens` for downstream parsers

2. **`reasoning_parser.py`**
   - Added `StepAudio2Detector` class (lines 195-279)
   - Registered detector for "step-audio2" (line 360)
   - Added null check in `_remove_protected_content` (line 232)

3. **`serving_chat.py`**
   - Reordered parser execution (lines 957-1009)
   - Added auto-append logic for TTS token (lines 555-563)
   - Improved TTS detection (lines 110-135)
   - Updated text extraction logic (line 1016)
   - Fixed null check for `tts_content.get("text")` (line 1016)

4. **`step_audio2_detector.py`**
   - Added null checks in `has_tool_call` (line 80)
   - Added null checks in `detect_and_parse` (line 95)

5. **`protocol.py`**
   - Added `tts_output` parameter (lines 498-501)

6. **`STEP_AUDIO2_INTEGRATION_GUIDE.md`**
   - Updated TTS examples with new parameter
   - Added API reference for `tts_output`
   - Updated troubleshooting section
   - Added "Recent Improvements" section
   - Added "Boundary Detection Implementation" section
   - Added "Mixed Content (TTS + Tool Calling)" example

## Testing

### TTS Output Test
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "step-audio-2-mini",
    "messages": [{"role": "user", "content": "Hello"}],
    "tts_output": true,
    "max_tokens": 100
  }'
```

**Expected Result**:
```json
{
  "choices": [{
    "message": {
      "content": "remaining text",
      "tts_content": {
        "tts_text": "Hello! How can I help you?",
        "tts_audio": "<audio_0><audio_1>..."
      },
      "reasoning_content": null
    }
  }]
}
```

### Tool Call Test
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "step-audio-2-mini",
    "messages": [{"role": "user", "content": "What is the weather?"}],
    "tools": [...],
    "max_tokens": 100
  }'
```

**Expected Result**:
```json
{
  "choices": [{
    "message": {
      "content": "Let me check",
      "tool_calls": [{
        "function": {
          "name": "get_weather",
          "arguments": "{...}"
        }
      }],
      "reasoning_content": null
    }
  }]
}
```

### Mixed Content Test (TTS + Tool Call)

**Test Script** (`/sgl-workspace/test.py`):
```python
import requests

url = "http://localhost:8001/v1/chat/completions"

tools = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "搜索工具",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}]

messages = [
    {"role": "system", "content": "你的名字叫做小跃，是由阶跃星辰公司训练出来的语音大模型。"},
    {"role": "user", "content": "帮我查一下今天沪深300的开盘价是多少"}
]

response = requests.post(
    url,
    json={
        "model": "step-audio-2-mini",
        "messages": messages,
        "tools": tools,
        "max_tokens": 4096,
        "tts_output": True,
        "temperature": 0.7,
    }
)

result = response.json()
message = result["choices"][0]["message"]
print("TTS Content:", message.get("tts_content"))
print("Tool Calls:", message.get("tool_calls"))
```

**Actual Result** (✅ Success):
```json
{
  "choices": [{
    "message": {
      "tts_content": {
        "tts_text": "好的,我查查沪深300的开盘价",
        "tts_audio": "<audio_1499><audio_2187>...<audio_6524>"
      },
      "tool_calls": [{
        "function": {
          "name": "search",
          "arguments": "{\"query\": \"2025年8月28日 沪深300 开盘价\"}"
        }
      }]
    }
  }]
}
```

**Key Observations**:
- ✅ `tts_text` is clean (no `<tool_call>` markers)
- ✅ `tool_calls` is properly populated
- ✅ Both fields coexist in the same response
- ✅ 81 audio tokens extracted successfully

## Boundary Detection Implementation

### Technical Details

The boundary detection logic in `audio_parser.py:276-364` solves the mixed content problem by:

1. **Decoding Tokens to Text**:
   ```python
   full_text = self.tokenizer.decode(input_token_ids, skip_special_tokens=False)
   ```

2. **Detecting Protected Regions** using regex patterns:
   ```python
   tool_call_pattern = r'<tool_call>.*?</tool_call>'
   think_pattern = r'<think>.*?</think>'

   has_protected = (
       re.search(tool_call_pattern, full_text, re.DOTALL) or
       re.search(think_pattern, full_text, re.DOTALL)
   )
   ```

3. **Extracting and Sorting Protected Regions**:
   ```python
   protected_regions = []
   for match in re.finditer(tool_call_pattern, full_text, re.DOTALL):
       protected_regions.append((match.start(), match.end()))
   for match in re.finditer(think_pattern, full_text, re.DOTALL):
       protected_regions.append((match.start(), match.end()))

   protected_regions.sort()
   ```

4. **Segmenting Content**:
   ```python
   tts_segments = []
   protected_segments = []
   last_end = 0

   for start, end in protected_regions:
       if start > last_end:
           tts_segments.append(full_text[last_end:start])  # TTS content
       protected_segments.append(full_text[start:end])     # Protected content
       last_end = end

   if last_end < len(full_text):
       tts_segments.append(full_text[last_end:])
   ```

5. **Re-tokenizing Separately**:
   ```python
   # Re-tokenize TTS segments only
   tts_text_only = ''.join(tts_segments)
   tts_tokens = self.tokenizer.encode(tts_text_only, add_special_tokens=False)

   # Re-tokenize protected content for other_tokens
   protected_text = ''.join(protected_segments)
   other_token_ids = self.tokenizer.encode(protected_text, add_special_tokens=False)

   return tts_text_token_ids, tts_audio_token_ids, other_token_ids
   ```

### Why This Approach Works

- **Not Simple Parser Ordering**: The solution is based on **boundary detection**, not sequential parser execution
- **Nested Content Support**: Can handle TTS content mixed with tool calls and reasoning in any order
- **Model Training Context**: Step-Audio2 was trained to output mixed content, so the model naturally produces these patterns
- **Minimal Performance Impact**: Boundary detection only activates when protected markers are detected

### Key Insight from User

> "也不是说 parser 顺序，而是说 TTS token 并不妨碍做 tool call，就是如果出现了 \<tool_call\> 那就是立刻进 tool call parser，然后从原本的 tts content 摘出来。都是一样的。只要统计到 \<think\> \<tool_call\>之类的，立马根据最近的这个 special token 进去相应的 parser。然后根据相应的结束 parser 结束状态。不是简单的谁先谁后的问题。要看嵌套关系"

Translation: "It's not about parser order, but about detecting special markers (\<tool_call\>, \<think\>) and extracting them from TTS content. It's about understanding the nesting relationship, not simple sequential execution."

## Benefits

1. **✅ Correct Content Separation**: Audio, tool calls, and reasoning are properly separated
2. **✅ Simpler API**: No need to manually add `<tts_start>` token
3. **✅ Backward Compatible**: Legacy method still works
4. **✅ Mixed Content Support**: TTS, tool calls, and reasoning can coexist in a single response
   - Boundary detection automatically separates different content types
   - TTS text is cleaned of tool call and reasoning markers
   - All parsers receive their relevant content
5. **✅ Better Error Messages**: Clear warnings when TTS token not found
6. **✅ Robust Null Handling**: Fixed all null pointer errors in parser chain

## Limitations

1. **No Streaming for TTS**: Audio parsing only works in non-streaming mode
2. **Re-tokenization Overhead**: Output text is re-encoded for audio parsing
3. **Fixed Parser Order**: Cannot customize parser execution order per request

## Future Improvements

1. Add streaming support for audio parser
2. Access output token IDs directly (avoid re-tokenization)
3. Add integration tests for all parser combinations
4. Optimize boundary detection performance (consider caching regex patterns)
5. Support nested reasoning within TTS content (e.g., TTS with embedded `<think>` blocks)

## Migration Guide

### For Users Currently Using Manual `<tts_start>`

**Option 1: Switch to `tts_output` (Recommended)**
```python
# Old
messages.append({"role": "assistant", "content": "<tts_start>", "eot": False})

# New
# Remove the assistant message, add tts_output=True
request["tts_output"] = True
```

**Option 2: Keep Using Manual Token**
- No changes needed
- Your existing code will continue to work

### For Server Operators

**Required**:
- Update SGLang to the version with these fixes
- Restart server (no config changes needed)

**Optional**:
- Update documentation to recommend `tts_output` parameter
- Add examples using the new parameter

## Verification

To verify the fix is working:

1. Check server logs for:
   ```
   TTS output enabled: appended <tts_start> token
   [TTS] is_tts_output: True, from_request: True
   ```

2. Check response has correct structure:
   - `tts_content` is populated (not null)
   - `reasoning_content` is null (unless using `separate_reasoning`)
   - Audio tokens are in `tts_content.tts_audio`, not in `content`

3. Tool calls are in `tool_calls` array, not in `reasoning_content`

## Support

For issues or questions:
- Check logs for parser initialization errors
- Verify server arguments include: `--audio-parser step_audio_2 --reasoning-parser step-audio2`
- Review `/sgl-workspace/sglang/STEP_AUDIO2_INTEGRATION_GUIDE.md`
