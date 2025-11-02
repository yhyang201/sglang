"""Audio Parser for extracting audio tokens from model output.

This module provides functionality to parse and extract audio tokens and text tokens
from TTS-enabled model outputs, specifically for Step-Audio2 style models.
"""

import logging
from collections.abc import Sequence
from typing import Dict, Optional, Tuple, Type

logger = logging.getLogger(__name__)


class BaseAudioParser:
    """
    Base class for audio parsers that extract audio content from model output.

    Audio parsers are used to separate text tokens and audio tokens from
    model outputs in TTS (Text-to-Speech) scenarios.
    """

    def __init__(self, tokenizer):
        """
        Initialize the audio parser with a tokenizer.

        Args:
            tokenizer: The model tokenizer for vocabulary access
        """
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else {}

    def extract_tts_content_nonstreaming(
        self,
        output_token_ids: Sequence[int],
        is_tts_ta4_output: bool = False
    ) -> Tuple[list[int], list[int], list[int]]:
        """
        Extract TTS content from output tokens (non-streaming).

        Args:
            output_token_ids: The complete output token sequence
            is_tts_ta4_output: Whether the output is in TA4 format

        Returns:
            A tuple of (text_token_ids, audio_token_ids, other_token_ids)
        """
        raise NotImplementedError(
            "extract_tts_content_nonstreaming must be implemented by subclass"
        )

    def is_tts_output(self, input_token_ids: Sequence[int]) -> bool:
        """
        Check if the input indicates TTS output format.

        Args:
            input_token_ids: The input token sequence (prompt)

        Returns:
            True if TTS output is expected, False otherwise
        """
        raise NotImplementedError(
            "is_tts_output must be implemented by subclass"
        )


class StepAudio2AudioParser(BaseAudioParser):
    """
    Audio parser for Step-Audio2 model TTS output.

    The Step-Audio2 model uses special tokens to delimit TTS content:
    - <tts_start>: Marks the beginning of TTS content
    - <tts_end>: Marks the end of TTS content
    - <audio_0> to <audio_6560>: Audio tokens
    - <tts_pad>: Text padding token
    - <audio_6561>: Audio padding token

    Format:
        <tts_start>{text_tokens}{audio_tokens}<tts_end>{normal_text}
    """

    def __init__(self, tokenizer):
        """
        Initialize Step-Audio2 audio parser.

        Args:
            tokenizer: The Step-Audio2 tokenizer
        """
        super().__init__(tokenizer)

        # Define special tokens
        self.audio_start_token = "<tts_start>"
        self.audio_end_token = "<tts_end>"
        self.first_audio_token = "<audio_0>"
        self.tts_pad_token = "<tts_pad>"
        self.audio_pad_token = "<audio_6561>"

        # Get token IDs from vocabulary
        self.audio_start_token_id = self.vocab.get(self.audio_start_token)
        self.audio_end_token_id = self.vocab.get(self.audio_end_token)
        self.first_audio_token_id = self.vocab.get(self.first_audio_token)
        self.tts_pad_token_id = self.vocab.get(self.tts_pad_token)
        self.audio_pad_token_id = self.vocab.get(self.audio_pad_token)

        # Validate that special tokens exist in vocabulary
        if self.audio_start_token_id is None or self.audio_end_token_id is None:
            raise RuntimeError(
                "Step-Audio2 audio parser could not locate tts_start/tts_end "
                "tokens in the tokenizer vocabulary!"
            )

        if self.first_audio_token_id is None:
            raise RuntimeError(
                "Step-Audio2 audio parser could not locate first audio token "
                "in the tokenizer vocabulary!"
            )

        logger.info(
            f"StepAudio2AudioParser initialized: "
            f"tts_start={self.audio_start_token_id}, "
            f"tts_end={self.audio_end_token_id}, "
            f"first_audio={self.first_audio_token_id}"
        )

        # Initialize streaming state
        self._reset_streaming_state()

    def _reset_streaming_state(self):
        """
        Reset streaming state for a new request.

        This should be called at the start of each streaming session.
        """
        self._stream_buffer_tokens = []  # Accumulate partial token chunks
        self._in_tts_section = False  # Whether we're inside a TTS section
        self._tts_start_seen = False  # Whether we've seen <tts_start>
        self._tts_end_seen = False  # Whether we've seen <tts_end>

        # Accumulators for tokens by type
        self._accumulated_text_tokens = []
        self._accumulated_audio_tokens = []
        self._accumulated_other_tokens = []

    def is_step_audio_token(self, token_id: int) -> bool:
        """
        Check if a token ID is an audio token.

        Audio tokens are in the range [<audio_0>, <audio_6560>].

        Args:
            token_id: The token ID to check

        Returns:
            True if the token is an audio token, False otherwise
        """
        return token_id >= self.first_audio_token_id

    def is_tts_output(self, input_token_ids: Sequence[int]) -> bool:
        """
        Check if the last prompt token is <tts_start>.

        If the last token is <tts_start>, the model should generate TTS output.

        Args:
            input_token_ids: The input token sequence

        Returns:
            True if TTS output is expected, False otherwise
        """
        if not input_token_ids:
            return False
        return input_token_ids[-1] == self.audio_start_token_id

    def extract_tts_content(
        self,
        input_token_ids: Sequence[int],
        has_tts_start: bool,
        has_tts_end: bool,
        is_text_audio_section: bool = True,
        include_pad_token: bool = False
    ) -> Tuple[list[int], list[int], list[int]]:
        """
        Extract TTS content from token sequence.

        This method handles various scenarios of tts_start/tts_end presence.

        Args:
            input_token_ids: The token sequence to parse
            has_tts_start: Whether <tts_start> is in the sequence
            has_tts_end: Whether <tts_end> is in the sequence
            is_text_audio_section: Whether to treat tokens as TTS content when no markers
            include_pad_token: Whether to include padding tokens in output

        Returns:
            A tuple of (text_token_ids, audio_token_ids, other_token_ids)
        """
        tts_text_token_ids = []
        tts_audio_token_ids = []
        other_token_ids = []
        in_tts_content_section = False

        if has_tts_start and has_tts_end:
            # Both markers present: <tts_start>text{audio}<tts_end>other
            for token_id in input_token_ids:
                if token_id == self.audio_start_token_id:
                    in_tts_content_section = True
                    continue

                if token_id == self.audio_end_token_id:
                    in_tts_content_section = False
                    continue

                if not in_tts_content_section:
                    other_token_ids.append(token_id)
                else:
                    # Skip padding tokens unless explicitly requested
                    if not include_pad_token and (
                        token_id == self.audio_pad_token_id or
                        token_id == self.tts_pad_token_id
                    ):
                        continue

                    if self.is_step_audio_token(token_id):
                        tts_audio_token_ids.append(token_id)
                    else:
                        tts_text_token_ids.append(token_id)

            return tts_text_token_ids, tts_audio_token_ids, other_token_ids

        elif has_tts_start and not has_tts_end:
            # Only start marker: other<tts_start>text{audio}
            for token_id in input_token_ids:
                if token_id == self.audio_start_token_id:
                    in_tts_content_section = True
                    continue

                if not in_tts_content_section:
                    other_token_ids.append(token_id)
                else:
                    if not include_pad_token and (
                        token_id == self.audio_pad_token_id or
                        token_id == self.tts_pad_token_id
                    ):
                        continue

                    if self.is_step_audio_token(token_id):
                        tts_audio_token_ids.append(token_id)
                    else:
                        tts_text_token_ids.append(token_id)

            return tts_text_token_ids, tts_audio_token_ids, other_token_ids

        elif not has_tts_start and has_tts_end:
            # Only end marker: text{audio}<tts_end>other
            in_tts_content_section = True
            for token_id in input_token_ids:
                if token_id == self.audio_end_token_id:
                    in_tts_content_section = False
                    continue

                if not in_tts_content_section:
                    other_token_ids.append(token_id)
                else:
                    if not include_pad_token and (
                        token_id == self.audio_pad_token_id or
                        token_id == self.tts_pad_token_id
                    ):
                        continue

                    if self.is_step_audio_token(token_id):
                        tts_audio_token_ids.append(token_id)
                    else:
                        tts_text_token_ids.append(token_id)

            return tts_text_token_ids, tts_audio_token_ids, other_token_ids

        else:
            # No markers present
            if is_text_audio_section:
                # Check for protected content markers (<tool_call>, <think>) that should not be in TTS
                # Decode to text first to detect boundaries
                import re
                full_text = self.tokenizer.decode(input_token_ids, skip_special_tokens=False)

                # Detect protected regions (tool calls and reasoning)
                tool_call_pattern = r'<tool_call>.*?</tool_call>'
                think_pattern = r'<think>.*?</think>'

                has_protected = (
                    re.search(tool_call_pattern, full_text, re.DOTALL) or
                    re.search(think_pattern, full_text, re.DOTALL)
                )

                if has_protected:
                    # Extract protected regions and process TTS content separately
                    protected_regions = []

                    # Find all tool call regions
                    for match in re.finditer(tool_call_pattern, full_text, re.DOTALL):
                        protected_regions.append((match.start(), match.end()))

                    # Find all reasoning regions
                    for match in re.finditer(think_pattern, full_text, re.DOTALL):
                        protected_regions.append((match.start(), match.end()))

                    # Sort by start position
                    protected_regions.sort()

                    # Build TTS text (exclude protected regions)
                    tts_segments = []
                    protected_segments = []
                    last_end = 0

                    for start, end in protected_regions:
                        if start > last_end:
                            # This segment is TTS content
                            tts_segments.append(full_text[last_end:start])
                        # Save protected content
                        protected_segments.append(full_text[start:end])
                        last_end = end

                    # Add remaining text after last protected region
                    if last_end < len(full_text):
                        tts_segments.append(full_text[last_end:])

                    # Re-tokenize TTS segments only
                    tts_text_only = ''.join(tts_segments)
                    tts_tokens = self.tokenizer.encode(tts_text_only, add_special_tokens=False) if tts_text_only.strip() else []

                    # Separate text and audio tokens in TTS content
                    for token_id in tts_tokens:
                        if not include_pad_token and (
                            token_id == self.audio_pad_token_id or
                            token_id == self.tts_pad_token_id
                        ):
                            continue

                        if self.is_step_audio_token(token_id):
                            tts_audio_token_ids.append(token_id)
                        else:
                            tts_text_token_ids.append(token_id)

                    # Re-tokenize protected content for other_tokens
                    protected_text = ''.join(protected_segments)
                    other_token_ids = self.tokenizer.encode(protected_text, add_special_tokens=False) if protected_text.strip() else []

                    return tts_text_token_ids, tts_audio_token_ids, other_token_ids
                else:
                    # No protected content, treat all as TTS content (original logic)
                    for token_id in input_token_ids:
                        if not include_pad_token and (
                            token_id == self.audio_pad_token_id or
                            token_id == self.tts_pad_token_id
                        ):
                            continue

                        if self.is_step_audio_token(token_id):
                            tts_audio_token_ids.append(token_id)
                        else:
                            tts_text_token_ids.append(token_id)

                    return tts_text_token_ids, tts_audio_token_ids, other_token_ids
            else:
                # Treat all as other tokens
                return [], [], list(input_token_ids)

    def extract_tts_content_streaming(
        self,
        new_token_ids: Sequence[int],
        is_tts_ta4_output: bool = False
    ) -> Tuple[list[int], list[int], list[int]]:
        """
        Extract TTS content from new token chunks (streaming).

        This method processes tokens incrementally and returns only the
        newly available tokens that can be safely emitted.

        Args:
            new_token_ids: New token chunk to process
            is_tts_ta4_output: Whether the output is in TA4 format

        Returns:
            A tuple of (text_token_ids, audio_token_ids, other_token_ids)
            containing only the incremental tokens that can be emitted now.

        Example:
            # First chunk
            >>> parser.extract_tts_content_streaming([123, 456], True)
            ([123, 456], [], [])

            # Second chunk with audio tokens
            >>> parser.extract_tts_content_streaming([<audio_0>, <audio_1>], True)
            ([], [<audio_0>, <audio_1>], [])

            # Final chunk with end marker
            >>> parser.extract_tts_content_streaming([<tts_end>, 789], True)
            ([], [], [789])

        Note:
            - Call _reset_streaming_state() before starting a new streaming session
            - The method maintains internal state across calls
            - Tokens may be buffered if we can't determine their type yet
        """
        if not is_tts_ta4_output:
            # Not TTS output, treat all tokens as other tokens
            return [], [], list(new_token_ids)

        # Add new tokens to buffer
        self._stream_buffer_tokens.extend(new_token_ids)

        # Result accumulators for this chunk
        text_tokens_to_emit = []
        audio_tokens_to_emit = []
        other_tokens_to_emit = []

        # Process tokens in buffer
        tokens_to_keep = []  # Tokens we need to keep buffering

        for i, token_id in enumerate(self._stream_buffer_tokens):
            # Check for <tts_end> token
            if token_id == self.audio_end_token_id:
                self._tts_end_seen = True
                self._in_tts_section = False
                # Don't emit the end token itself
                continue

            # After seeing <tts_end>, everything is "other" tokens
            if self._tts_end_seen:
                other_tokens_to_emit.append(token_id)
                continue

            # We're in TTS section (or assumed to be, since <tts_start> is in prompt)
            # The format is: {text_tokens}{audio_tokens}<tts_end>
            # We need to determine if this is text or audio

            # Skip padding tokens
            if token_id == self.audio_pad_token_id or token_id == self.tts_pad_token_id:
                continue

            # Check if it's an audio token
            if self.is_step_audio_token(token_id):
                # It's an audio token, safe to emit
                audio_tokens_to_emit.append(token_id)
                self._accumulated_audio_tokens.append(token_id)
            else:
                # It's a text token
                # We need to be careful here:
                # - If we haven't seen any audio tokens yet, it's likely text
                # - If we've already seen audio tokens, this might be an error,
                #   but we'll treat it as text for safety

                # Check if we're at the end of the buffer
                # If this is the last token and we haven't seen <tts_end> yet,
                # we might want to buffer it in case it's part of a special token
                # But for simplicity and immediate responsiveness, we emit text tokens
                # immediately since text should come before audio in the format

                if len(self._accumulated_audio_tokens) > 0:
                    # We've already seen audio tokens, so text after audio is unusual
                    # but we'll emit it as text
                    logger.warning(
                        f"Unexpected text token {token_id} after audio tokens in TTS section"
                    )

                text_tokens_to_emit.append(token_id)
                self._accumulated_text_tokens.append(token_id)

        # Clear the buffer since we've processed all tokens
        # (or keep last few if we want to handle partial special tokens, but for MVP we don't)
        self._stream_buffer_tokens = []

        return text_tokens_to_emit, audio_tokens_to_emit, other_tokens_to_emit

    def extract_tts_content_nonstreaming(
        self,
        output_token_ids: Sequence[int],
        is_tts_ta4_output: bool = False
    ) -> Tuple[list[int], list[int], list[int]]:
        """
        Extract TTS content from complete output (non-streaming).

        Args:
            output_token_ids: The complete output token sequence
            is_tts_ta4_output: Whether the output is in TA4 format
                               (expected to end with <tts_end>)

        Returns:
            A tuple of (text_token_ids, audio_token_ids, other_token_ids)

        Example:
            Input: [<tts_start>, 123, 456, <audio_0>, <audio_1>, <tts_end>, 789]
            Output: ([123, 456], [<audio_0>, <audio_1>], [789])
        """
        if not is_tts_ta4_output:
            # Not TTS output, return empty TTS content
            return [], [], list(output_token_ids)

        # For TA4 output, we expect the format to end with <tts_end>
        # The output starts after <tts_start> from the prompt
        tts_text_token_ids, tts_audio_token_ids, other_token_ids = (
            self.extract_tts_content(
                output_token_ids,
                has_tts_start=False,  # <tts_start> is in prompt, not output
                has_tts_end=True      # Output should end with <tts_end>
            )
        )

        return tts_text_token_ids, tts_audio_token_ids, other_token_ids


class AudioParserManager:
    """
    Manager for audio parser registration and retrieval.

    This follows the same pattern as ReasoningParserManager and ToolParserManager.
    """

    _parsers: Dict[str, Type[BaseAudioParser]] = {}

    @classmethod
    def register(cls, name: str, parser_class: Type[BaseAudioParser]):
        """
        Register an audio parser class.

        Args:
            name: The name to register the parser under
            parser_class: The parser class to register
        """
        if not issubclass(parser_class, BaseAudioParser):
            raise TypeError(
                f"Parser class must be a subclass of BaseAudioParser, "
                f"got {type(parser_class)}"
            )

        cls._parsers[name] = parser_class
        logger.info(f"Registered audio parser: {name} -> {parser_class.__name__}")

    @classmethod
    def get_parser(cls, name: str) -> Type[BaseAudioParser]:
        """
        Get a registered audio parser class.

        Args:
            name: The name of the parser to retrieve

        Returns:
            The parser class

        Raises:
            KeyError: If the parser name is not registered
        """
        if name not in cls._parsers:
            raise KeyError(
                f"Audio parser '{name}' not found. "
                f"Available parsers: {list(cls._parsers.keys())}"
            )
        return cls._parsers[name]

    @classmethod
    def create_parser(cls, name: str, tokenizer) -> BaseAudioParser:
        """
        Create an audio parser instance.

        Args:
            name: The name of the parser to create
            tokenizer: The tokenizer to pass to the parser

        Returns:
            An instance of the audio parser
        """
        parser_class = cls.get_parser(name)
        return parser_class(tokenizer)


# Register Step-Audio2 parser
AudioParserManager.register("step_audio_2", StepAudio2AudioParser)
