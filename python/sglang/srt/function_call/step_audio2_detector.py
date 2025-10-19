"""
Step-Audio2 Tool Call Detector

This detector handles tool calls in the Step-Audio2 format:
<tool_call>function
{function_name}
{json_arguments}</tool_call>
"""

import json
import logging
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.ebnf_composer import EBNFComposer

logger = logging.getLogger(__name__)


class StepAudio2Detector(BaseFormatDetector):
    """
    Detector for Step-Audio2 model tool call format.

    Format Structure:
    ```
    <tool_call>function
    {function_name}
    {json_arguments}</tool_call>
    ```

    Example:
    ```
    <tool_call>function
    search
    {"query": "weather in Shanghai"}</tool_call>
    ```

    Multiple tool calls:
    ```
    <tool_call>function
    get_weather
    {"location": "Shanghai"}</tool_call><tool_call>function
    get_temperature
    {"location": "Beijing"}</tool_call>
    ```
    """

    def __init__(self):
        super().__init__()
        self.tool_call_start = "<tool_call>"
        self.tool_call_end = "</tool_call>"

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Step-Audio2 format tool call."""
        return self.tool_call_start in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Parse complete text to extract all tool calls (non-streaming).

        Args:
            text: The complete model output text
            tools: List of available tools

        Returns:
            StreamingParseResult containing parsed tool calls and remaining content
        """
        if self.tool_call_start not in text:
            return StreamingParseResult(normal_text=text, calls=[])

        # Split text into before tool calls, tool calls section, and after
        parts = text.split(self.tool_call_start, 1)
        content_before = parts[0]
        remaining = parts[1] if len(parts) > 1 else ""

        calls = []
        content_after = ""

        # Parse all tool calls
        while remaining:
            if self.tool_call_end not in remaining:
                # Incomplete tool call, treat as normal text
                content_after += remaining
                break

            # Extract one tool call
            tool_content, rest = remaining.split(self.tool_call_end, 1)

            # Parse the tool call content
            # Expected format: function\n{name}\n{json_args}
            lines = tool_content.split('\n', 2)

            if len(lines) >= 3:
                function_type = lines[0].strip()
                function_name = lines[1].strip()
                arguments_str = lines[2].strip()

                if function_type == "function":
                    try:
                        # Parse JSON arguments
                        arguments = json.loads(arguments_str)

                        # Validate function name is in available tools
                        tool_indices = self._get_tool_indices(tools)
                        if function_name in tool_indices:
                            calls.append(
                                ToolCallItem(
                                    tool_index=tool_indices[function_name],
                                    name=function_name,
                                    parameters=json.dumps(arguments, ensure_ascii=False),
                                )
                            )
                        else:
                            logger.warning(
                                f"Tool call to undefined function: {function_name}"
                            )
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse tool call arguments: {e}")

            # Check for next tool call
            if self.tool_call_start in rest:
                remaining = rest.split(self.tool_call_start, 1)[1]
                content_after += rest.split(self.tool_call_start, 1)[0]
            else:
                content_after += rest
                break

        # Combine content before and after tool calls
        normal_text = content_before + content_after

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for Step-Audio2 format.

        Note: This is a simplified MVP implementation that does not support streaming.
        For MVP, we buffer everything and parse when tool calls are complete.

        Args:
            new_text: New text chunk to parse
            tools: List of available tools

        Returns:
            StreamingParseResult with parsed content
        """
        # Accumulate text in buffer
        self._buffer += new_text

        # Build tool indices if not already built
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        # Check if we have any complete tool calls
        if self.tool_call_start not in self._buffer:
            # No tool call started yet
            # Check if we might have a partial start token
            partial_len = self._ends_with_partial_token(
                self._buffer, self.tool_call_start
            )
            if partial_len:
                # Might be start of tool call, keep buffering
                return StreamingParseResult()
            else:
                # No tool call, return as normal text
                normal_text = self._buffer
                self._buffer = ""
                return StreamingParseResult(normal_text=normal_text)

        # We have at least the start of a tool call
        # Check if we have a complete tool call
        if self.tool_call_end not in self._buffer:
            # Tool call not complete yet, keep buffering
            return StreamingParseResult()

        # We have at least one complete tool call
        # For MVP, we use the non-streaming parser
        # Extract everything up to and including the complete tool calls
        last_end_idx = self._buffer.rfind(self.tool_call_end)
        if last_end_idx == -1:
            return StreamingParseResult()

        # Parse the portion with complete tool calls
        complete_text = self._buffer[:last_end_idx + len(self.tool_call_end)]
        self._buffer = self._buffer[last_end_idx + len(self.tool_call_end):]

        # Use the non-streaming parser for complete tool calls
        result = self.detect_and_parse(complete_text, tools)

        return result

    def supports_structural_tag(self) -> bool:
        """
        Step-Audio2 format does not support structural tag constrained generation.

        The format is simple text-based and doesn't follow JSON structure strictly.
        """
        return False

    def structure_info(self) -> _GetInfoFunc:
        """Not implemented as structural tag is not supported."""
        raise NotImplementedError(
            "StepAudio2Detector does not support structural tag format"
        )

    def build_ebnf(self, tools: List[Tool]) -> str:
        """
        Build EBNF grammar for Step-Audio2 tool call format.

        The format is:
        <tool_call>function
        {function_name}
        {json_arguments}</tool_call>

        Args:
            tools: List of available tools

        Returns:
            EBNF grammar string for constrained generation
        """
        if not tools:
            return ""

        # Extract tool names
        tool_names = [tool.function.name for tool in tools if tool.function.name]

        if not tool_names:
            return ""

        # Build EBNF for Step-Audio2 format
        # The format is simpler than other formats, so we build it manually

        # Start with the opening tag and "function" keyword
        ebnf = 'root ::= "<tool_call>" "function" "\\n" tool_name "\\n" arguments "</tool_call>"\n'

        # Add tool name options
        ebnf += "tool_name ::= "
        ebnf += " | ".join(f'"{name}"' for name in tool_names)
        ebnf += "\n"

        # Add arguments as JSON object
        # For MVP, we allow any JSON object
        ebnf += 'arguments ::= "{" ws members ws "}"\n'
        ebnf += 'members ::= (string ws ":" ws value (ws "," ws string ws ":" ws value)*)?\n'
        ebnf += 'value ::= string | number | "true" | "false" | "null" | object | array\n'
        ebnf += 'string ::= "\\"" ([^"\\\\] | "\\\\" ["\\\\/bfnrt] | "\\\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])* "\\""\n'
        ebnf += 'number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?\n'
        ebnf += 'object ::= "{" ws members ws "}"\n'
        ebnf += 'array ::= "[" ws (value (ws "," ws value)*)? ws "]"\n'
        ebnf += 'ws ::= [ \\t\\n\\r]*\n'

        return ebnf
