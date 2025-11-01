"""
Simple test for Audio Parser functionality.

This is a basic test to verify the StepAudio2AudioParser works correctly.
"""

from typing import Dict


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.vocab_dict = {
            "<tts_start>": 151693,
            "<tts_end>": 151694,
            "<audio_0>": 151695,
            "<audio_1>": 151696,
            "<audio_2>": 151697,
            "<tts_pad>": 151700,
            "<audio_6561>": 158256,
        }

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab_dict


def test_audio_parser_basic():
    """Test basic audio parser functionality."""
    from sglang.srt.parser.audio_parser import StepAudio2AudioParser

    tokenizer = MockTokenizer()
    parser = StepAudio2AudioParser(tokenizer)

    # Test 1: Check if TTS output detection works
    prompt_with_tts = [123, 456, 151693]  # ends with <tts_start>
    prompt_without_tts = [123, 456, 789]

    assert parser.is_tts_output(prompt_with_tts) == True
    assert parser.is_tts_output(prompt_without_tts) == False
    print("✓ TTS output detection works")

    # Test 2: Extract TTS content (non-streaming)
    # Simulate output: text_tokens + audio_tokens + <tts_end> + other_tokens
    output_tokens = [
        100, 101, 102,  # text tokens
        151695, 151696, 151697,  # audio tokens <audio_0>, <audio_1>, <audio_2>
        151694,  # <tts_end>
        200, 201  # other tokens after TTS
    ]

    text_tokens, audio_tokens, other_tokens = parser.extract_tts_content_nonstreaming(
        output_tokens, is_tts_ta4_output=True
    )

    assert text_tokens == [100, 101, 102]
    assert audio_tokens == [151695, 151696, 151697]
    assert other_tokens == [200, 201]
    print("✓ TTS content extraction works")

    # Test 3: Non-TTS output
    normal_output = [100, 101, 102, 103]
    text_tokens, audio_tokens, other_tokens = parser.extract_tts_content_nonstreaming(
        normal_output, is_tts_ta4_output=False
    )

    assert text_tokens == []
    assert audio_tokens == []
    assert other_tokens == normal_output
    print("✓ Non-TTS output handled correctly")

    # Test 4: Filter padding tokens
    output_with_padding = [
        100, 101,  # text tokens
        151700,  # <tts_pad> - should be filtered
        151695, 151696,  # audio tokens
        158256,  # <audio_6561> padding - should be filtered
        151694,  # <tts_end>
        200  # other token
    ]

    text_tokens, audio_tokens, other_tokens = parser.extract_tts_content_nonstreaming(
        output_with_padding, is_tts_ta4_output=True
    )

    assert text_tokens == [100, 101]
    assert audio_tokens == [151695, 151696]
    assert other_tokens == [200]
    print("✓ Padding token filtering works")

    print("\n✅ All audio parser tests passed!")


def test_audio_parser_streaming():
    """Test streaming audio parser functionality."""
    from sglang.srt.parser.audio_parser import StepAudio2AudioParser

    tokenizer = MockTokenizer()
    parser = StepAudio2AudioParser(tokenizer)

    print("\n" + "=" * 50)
    print("Testing Streaming Audio Parser")
    print("=" * 50)

    # Test 1: Simple streaming - complete TTS section split into chunks
    print("\nTest 1: Simple streaming with chunk splits")
    parser._reset_streaming_state()

    # Chunk 1: text tokens
    text, audio, other = parser.extract_tts_content_streaming(
        [100, 101, 102], is_tts_ta4_output=True
    )
    assert text == [100, 101, 102]
    assert audio == []
    assert other == []
    print("  ✓ Chunk 1 (text tokens): text=3, audio=0, other=0")

    # Chunk 2: audio tokens
    text, audio, other = parser.extract_tts_content_streaming(
        [151695, 151696, 151697], is_tts_ta4_output=True
    )
    assert text == []
    assert audio == [151695, 151696, 151697]
    assert other == []
    print("  ✓ Chunk 2 (audio tokens): text=0, audio=3, other=0")

    # Chunk 3: end token + other tokens
    text, audio, other = parser.extract_tts_content_streaming(
        [151694, 200, 201], is_tts_ta4_output=True
    )
    assert text == []
    assert audio == []
    assert other == [200, 201]
    print("  ✓ Chunk 3 (end + other): text=0, audio=0, other=2")

    # Test 2: All in one chunk (streaming with complete content)
    print("\nTest 2: Streaming with complete content in one chunk")
    parser._reset_streaming_state()

    text, audio, other = parser.extract_tts_content_streaming(
        [100, 101, 151695, 151696, 151694, 200],
        is_tts_ta4_output=True
    )
    assert text == [100, 101]
    assert audio == [151695, 151696]
    assert other == [200]
    print("  ✓ Single chunk: text=2, audio=2, other=1")

    # Test 3: Streaming vs Non-streaming consistency
    print("\nTest 3: Streaming vs Non-streaming consistency check")
    complete_output = [100, 101, 102, 151695, 151696, 151697, 151694, 200, 201]

    # Non-streaming
    text_ns, audio_ns, other_ns = parser.extract_tts_content_nonstreaming(
        complete_output, is_tts_ta4_output=True
    )

    # Streaming (split into 3 chunks)
    parser._reset_streaming_state()
    text_s1, audio_s1, other_s1 = parser.extract_tts_content_streaming(
        [100, 101, 102], is_tts_ta4_output=True
    )
    text_s2, audio_s2, other_s2 = parser.extract_tts_content_streaming(
        [151695, 151696, 151697], is_tts_ta4_output=True
    )
    text_s3, audio_s3, other_s3 = parser.extract_tts_content_streaming(
        [151694, 200, 201], is_tts_ta4_output=True
    )

    # Combine streaming results
    text_s = text_s1 + text_s2 + text_s3
    audio_s = audio_s1 + audio_s2 + audio_s3
    other_s = other_s1 + other_s2 + other_s3

    assert text_s == text_ns
    assert audio_s == audio_ns
    assert other_s == other_ns
    print("  ✓ Streaming results match non-streaming results")

    # Test 4: Padding token filtering in streaming
    print("\nTest 4: Padding token filtering in streaming mode")
    parser._reset_streaming_state()

    # Chunk with padding tokens
    text, audio, other = parser.extract_tts_content_streaming(
        [100, 151700, 151695, 158256, 151696],  # includes <tts_pad> and <audio_6561>
        is_tts_ta4_output=True
    )
    assert text == [100]
    assert audio == [151695, 151696]  # padding filtered
    assert other == []
    print("  ✓ Padding tokens filtered: text=1, audio=2 (padding removed)")

    # Test 5: Non-TTS output in streaming
    print("\nTest 5: Non-TTS output in streaming mode")
    parser._reset_streaming_state()

    text, audio, other = parser.extract_tts_content_streaming(
        [100, 101, 102], is_tts_ta4_output=False
    )
    assert text == []
    assert audio == []
    assert other == [100, 101, 102]
    print("  ✓ Non-TTS output handled: all tokens are 'other'")

    # Test 6: Multiple small chunks
    print("\nTest 6: Multiple small chunks (token-by-token)")
    parser._reset_streaming_state()

    all_text = []
    all_audio = []
    all_other = []

    # Send tokens one by one
    test_tokens = [100, 101, 151695, 151696, 151694, 200]
    for token in test_tokens:
        text, audio, other = parser.extract_tts_content_streaming(
            [token], is_tts_ta4_output=True
        )
        all_text.extend(text)
        all_audio.extend(audio)
        all_other.extend(other)

    assert all_text == [100, 101]
    assert all_audio == [151695, 151696]
    assert all_other == [200]
    print("  ✓ Token-by-token streaming works correctly")

    # Test 7: Empty chunks (edge case)
    print("\nTest 7: Empty chunks handling")
    parser._reset_streaming_state()

    text, audio, other = parser.extract_tts_content_streaming(
        [], is_tts_ta4_output=True
    )
    assert text == []
    assert audio == []
    assert other == []
    print("  ✓ Empty chunk handled correctly")

    # Test 8: Only audio tokens (no text)
    print("\nTest 8: Only audio tokens (no text)")
    parser._reset_streaming_state()

    text, audio, other = parser.extract_tts_content_streaming(
        [151695, 151696, 151697, 151694],
        is_tts_ta4_output=True
    )
    assert text == []
    assert audio == [151695, 151696, 151697]
    assert other == []
    print("  ✓ Audio-only content: text=0, audio=3, other=0")

    # Test 9: Only text tokens (no audio)
    print("\nTest 9: Only text tokens (no audio)")
    parser._reset_streaming_state()

    text, audio, other = parser.extract_tts_content_streaming(
        [100, 101, 102, 151694, 200],
        is_tts_ta4_output=True
    )
    assert text == [100, 101, 102]
    assert audio == []
    assert other == [200]
    print("  ✓ Text-only content: text=3, audio=0, other=1")

    # Test 10: Reset state between streaming sessions
    print("\nTest 10: State reset between sessions")
    parser._reset_streaming_state()

    # First session
    parser.extract_tts_content_streaming([100, 151695, 151694], True)

    # Reset and start new session
    parser._reset_streaming_state()

    text, audio, other = parser.extract_tts_content_streaming(
        [200, 201, 151696, 151694],
        is_tts_ta4_output=True
    )
    # Should work correctly without interference from previous session
    assert text == [200, 201]
    assert audio == [151696]
    assert other == []
    print("  ✓ State reset works correctly")

    print("\n✅ All streaming audio parser tests passed!")


def test_audio_parser_manager():
    """Test AudioParserManager registration."""
    from sglang.srt.parser.audio_parser import AudioParserManager

    # Test that step_audio_2 is registered
    parser_class = AudioParserManager.get_parser("step_audio_2")
    assert parser_class is not None
    print("✓ StepAudio2AudioParser is registered")

    # Test creating a parser instance
    tokenizer = MockTokenizer()
    parser = AudioParserManager.create_parser("step_audio_2", tokenizer)
    assert parser is not None
    assert parser.audio_start_token == "<tts_start>"
    print("✓ Parser instance creation works")

    print("\n✅ AudioParserManager tests passed!")


if __name__ == "__main__":
    print("Running Audio Parser Tests\n")
    print("=" * 50)
    test_audio_parser_basic()
    print()
    test_audio_parser_streaming()
    print()
    test_audio_parser_manager()
