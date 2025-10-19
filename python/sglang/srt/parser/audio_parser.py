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
                # Assume all content is TTS content
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
