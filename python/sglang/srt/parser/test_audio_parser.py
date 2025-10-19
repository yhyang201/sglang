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
    test_audio_parser_manager()
