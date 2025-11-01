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

from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.ebnf_composer import EBNFComposer
from sglang.srt.function_call.utils import _is_complete_json, _partial_json_loads

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

        # Enhanced streaming state management
        self._in_tool_call = False  # Whether we're inside a tool call block
        self._current_function_name = None  # Name of current tool being parsed
        self._args_buffer = ""  # Accumulate JSON arguments as string
        self._function_name_sent = False  # Whether we've sent the current function name
        self._previous_args_sent = ""  # Track what arguments we've already sent

    def _reset_tool_state(self):
        """Reset state for current tool call (called when tool completes or fails)."""
        self._in_tool_call = False
        self._current_function_name = None
        self._args_buffer = ""
        self._function_name_sent = False
        self._previous_args_sent = ""

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
        Streaming incremental parsing for Step-Audio2 format with full streaming support.

        Supports incremental parsing of:
        - Tool names (sent first with empty parameters)
        - JSON arguments (streamed incrementally as they arrive)
        - Multiple sequential tool calls

        Args:
            new_text: New text chunk to parse
            tools: List of available tools

        Returns:
            StreamingParseResult with parsed content and/or tool calls
        """
        # Accumulate text in buffer
        self._buffer += new_text

        # Build tool indices if not already built
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        # Handle content before any tool call
        if not self._in_tool_call and self.tool_call_start not in self._buffer:
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

        # Handle start of tool call
        if not self._in_tool_call and self.tool_call_start in self._buffer:
            # Extract normal text before tool call
            idx = self._buffer.find(self.tool_call_start)
            normal_text = self._buffer[:idx]

            # Move buffer to after <tool_call>
            self._buffer = self._buffer[idx + len(self.tool_call_start):]
            self._in_tool_call = True

            # Return normal text if any, otherwise continue processing
            if normal_text:
                return StreamingParseResult(normal_text=normal_text)

        # Now we're inside a tool call: <tool_call>function\n{name}\n{json_args}</tool_call>
        # Parse line by line
        if self._in_tool_call:
            # Split buffer by newlines
            lines = self._buffer.split('\n')

            # Line 0: Should be "function"
            if len(lines) >= 1 and lines[0].strip() == "function":
                # Good, we have the "function" keyword

                # Line 1: Function name
                if len(lines) >= 2 and not self._current_function_name:
                    function_name = lines[1].strip()

                    # Check if this is a valid, complete function name
                    # (not cut off mid-name)
                    if function_name and not self._buffer.endswith(function_name):
                        # We have a complete function name
                        self._current_function_name = function_name

                        # Validate function name
                        if function_name not in self._tool_indices:
                            logger.warning(
                                f"Tool call to undefined function: {function_name}"
                            )
                            # Reset and skip this tool call
                            self._reset_tool_state()
                            # Try to skip to end of this tool call
                            if self.tool_call_end in self._buffer:
                                end_idx = self._buffer.find(self.tool_call_end)
                                self._buffer = self._buffer[
                                    end_idx + len(self.tool_call_end):
                                ]
                            else:
                                self._buffer = ""
                            return StreamingParseResult()

                        # Send tool name with empty parameters if not sent yet
                        if not self._function_name_sent:
                            self._function_name_sent = True

                            # Increment tool index for tracking
                            if self.current_tool_id == -1:
                                self.current_tool_id = 0
                            else:
                                self.current_tool_id += 1

                            return StreamingParseResult(
                                calls=[
                                    ToolCallItem(
                                        tool_index=self._tool_indices[function_name],
                                        name=function_name,
                                        parameters="",
                                    )
                                ]
                            )

                # Line 2+: JSON arguments (streaming)
                if len(lines) >= 3 and self._current_function_name:
                    # Extract JSON arguments (everything from line 2 onwards)
                    # Join all lines after line 1
                    args_text = '\n'.join(lines[2:])

                    # Check if we have the end token
                    if self.tool_call_end in args_text:
                        # Tool call is complete
                        end_idx = args_text.find(self.tool_call_end)
                        args_text = args_text[:end_idx]

                        # Parse complete JSON
                        try:
                            args_obj = json.loads(args_text.strip())
                            complete_args_json = json.dumps(args_obj, ensure_ascii=False)

                            # Calculate what we haven't sent yet
                            args_diff = complete_args_json[len(self._previous_args_sent):]

                            # Save to prev_tool_call_arr for serving layer
                            tool_call_info = {
                                "name": self._current_function_name,
                                "arguments": args_obj,
                            }
                            if self.current_tool_id < len(self.prev_tool_call_arr):
                                self.prev_tool_call_arr[self.current_tool_id] = tool_call_info
                            else:
                                self.prev_tool_call_arr.append(tool_call_info)

                            # Update streamed_args_for_tool
                            if self.current_tool_id < len(self.streamed_args_for_tool):
                                self.streamed_args_for_tool[self.current_tool_id] = complete_args_json
                            else:
                                self.streamed_args_for_tool.append(complete_args_json)

                            # Reset state for next tool call
                            self._buffer = self._buffer[
                                self._buffer.find(self.tool_call_end) + len(self.tool_call_end):
                            ]
                            self._reset_tool_state()

                            # Return final arguments diff
                            if args_diff:
                                return StreamingParseResult(
                                    calls=[
                                        ToolCallItem(
                                            tool_index=self._tool_indices[tool_call_info["name"]],
                                            parameters=args_diff,
                                        )
                                    ]
                                )
                            else:
                                # No new content, continue processing buffer
                                return StreamingParseResult()

                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse complete tool call arguments: {e}")
                            self._reset_tool_state()
                            return StreamingParseResult()

                    else:
                        # Tool call not complete yet, try to parse partial JSON
                        try:
                            (args_obj, consumed_len) = _partial_json_loads(
                                args_text, Allow.ALL
                            )

                            # Convert to JSON string
                            if args_obj:
                                # For partial JSON, only send stable prefix
                                # Check if JSON is stable enough to send
                                partial_args_json = json.dumps(args_obj, ensure_ascii=False)

                                # Only send if we have more than what we've sent before
                                if len(partial_args_json) > len(self._previous_args_sent):
                                    # Check if the new part is stable
                                    # For now, we'll be conservative and only send complete key-value pairs
                                    # We can detect this by checking if partial_args_json is longer
                                    # and doesn't end with a partial value

                                    # Simple heuristic: if it ends with : or , or is incomplete string,
                                    # don't send yet
                                    if not (partial_args_json.rstrip().endswith((':',',')) or
                                            partial_args_json.count('"') % 2 != 0):
                                        # Looks stable, send the diff
                                        args_diff = partial_args_json[len(self._previous_args_sent):]
                                        self._previous_args_sent = partial_args_json

                                        return StreamingParseResult(
                                            calls=[
                                                ToolCallItem(
                                                    tool_index=self._tool_indices[self._current_function_name],
                                                    parameters=args_diff,
                                                )
                                            ]
                                        )

                        except MalformedJSON:
                            # Not enough data yet, keep buffering
                            pass

        # No new content to emit, keep buffering
        return StreamingParseResult()

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
