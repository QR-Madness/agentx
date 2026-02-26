"""
Output Parser for AgentX.

Parses model output to extract thinking tags, format content,
and handle structured output.
"""

import re
from typing import Optional
from pydantic import BaseModel


class ParsedOutput(BaseModel):
    """Parsed model output with separated components."""
    content: str
    thinking: Optional[str] = None
    has_thinking: bool = False
    raw: str  # Original unmodified output


# Patterns for thinking tags (case-insensitive)
THINKING_PATTERNS = [
    # Standard thinking tags (closed)
    re.compile(r'<thinking>(.*?)</thinking>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE),
    # Alternative formats (closed)
    re.compile(r'\[thinking\](.*?)\[/thinking\]', re.DOTALL | re.IGNORECASE),
    re.compile(r'\[think\](.*?)\[/think\]', re.DOTALL | re.IGNORECASE),
    # Anthropic-style
    re.compile(r'<internal_monologue>(.*?)</internal_monologue>', re.DOTALL | re.IGNORECASE),
    # Reasoning/reflection tags
    re.compile(r'<reasoning>(.*?)</reasoning>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<reflection>(.*?)</reflection>', re.DOTALL | re.IGNORECASE),
]

# Pattern for unclosed thinking tags (streaming or malformed) - handled separately
UNCLOSED_THINKING_PATTERN = re.compile(r'^<think(?:ing)?>\s*', re.IGNORECASE)

# Patterns for extracting final answer from reasoning models
# These capture text AFTER common answer markers
FINAL_ANSWER_PATTERNS = [
    re.compile(r'(?:^|\n)\s*(?:Answer|Final Answer|Response|Result|Conclusion):\s*(.+?)$', re.MULTILINE | re.IGNORECASE),
    re.compile(r'(?:^|\n)\s*(?:Therefore|Thus|So|Hence),?\s*(?:the answer is|I conclude)?\s*(.+?)$', re.MULTILINE | re.IGNORECASE),
]


def parse_output(raw_output: str) -> ParsedOutput:
    """
    Parse model output to extract thinking and content.

    Handles various thinking tag formats:
    - <thinking>...</thinking>
    - <think>...</think>
    - [thinking]...[/thinking]
    - <internal_monologue>...</internal_monologue>
    - Unclosed <think> tags (streaming or malformed)

    Args:
        raw_output: The raw model output string

    Returns:
        ParsedOutput with separated content and thinking
    """
    if not raw_output:
        return ParsedOutput(content="", raw="")

    thinking_parts = []
    content = raw_output

    # Extract all closed thinking blocks first
    for pattern in THINKING_PATTERNS:
        matches = pattern.findall(content)
        for match in matches:
            thinking_parts.append(match.strip())
        # Remove thinking blocks from content
        content = pattern.sub('', content)

    # Handle unclosed thinking tags (e.g., "<think>reasoning..." without "</think>")
    # This happens during streaming or with some models that don't close tags
    if UNCLOSED_THINKING_PATTERN.match(content):
        # Strip the opening tag
        content = UNCLOSED_THINKING_PATTERN.sub('', content)
        # Everything is thinking content - but keep looking for answer at the end
        # Split on the last newline to preserve potential final answer
        lines = content.strip().split('\n')
        if len(lines) > 1:
            # Last line might be the answer, rest is thinking
            last_line = lines[-1].strip()
            # Check if last line looks like an answer (short, possibly YES/NO or JSON)
            if len(last_line) < 100 and not last_line.startswith(('I ', 'The ', 'This ', 'We ')):
                thinking_parts.append('\n'.join(lines[:-1]).strip())
                content = last_line
            else:
                # All thinking, no clear answer
                thinking_parts.append(content.strip())
                content = ""
        else:
            # Single line - treat as content (might be answer)
            pass

    # Clean up content
    content = content.strip()
    # Remove multiple consecutive newlines
    content = re.sub(r'\n{3,}', '\n\n', content)

    # Combine thinking parts
    thinking = None
    if thinking_parts:
        thinking = "\n\n".join(thinking_parts)

    return ParsedOutput(
        content=content,
        thinking=thinking,
        has_thinking=bool(thinking_parts),
        raw=raw_output,
    )


def extract_code_blocks(content: str) -> list[dict]:
    """
    Extract code blocks from markdown content.
    
    Returns a list of dicts with 'language' and 'code' keys.
    """
    pattern = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
    blocks = []
    
    for match in pattern.finditer(content):
        blocks.append({
            "language": match.group(1) or "text",
            "code": match.group(2).strip(),
        })
    
    return blocks


def validate_json_output(content: str) -> tuple[bool, Optional[dict], Optional[str]]:
    """
    Validate that content is valid JSON.
    
    Returns:
        Tuple of (is_valid, parsed_json, error_message)
    """
    import json
    
    # Try to find JSON in the content (might be wrapped in markdown)
    json_pattern = re.compile(r'```json\n(.*?)```', re.DOTALL)
    match = json_pattern.search(content)
    
    json_str = match.group(1) if match else content.strip()
    
    try:
        parsed = json.loads(json_str)
        return True, parsed, None
    except json.JSONDecodeError as e:
        return False, None, str(e)


def clean_for_display(content: str) -> str:
    """
    Clean content for display, removing any remaining artifacts.
    """
    # Remove any remaining empty thinking tags
    for pattern in THINKING_PATTERNS:
        content = pattern.sub('', content)

    # Clean up whitespace
    content = content.strip()
    content = re.sub(r'\n{3,}', '\n\n', content)

    return content


def extract_yes_no_answer(content: str) -> Optional[bool]:
    """
    Extract a YES/NO answer from model output.

    Handles reasoning models that output thinking before the answer.
    Looks for YES/NO in various positions and formats.

    Args:
        content: Raw model output

    Returns:
        True for YES, False for NO, None if neither found
    """
    if not content:
        return None

    # First, strip any thinking tags
    parsed = parse_output(content)
    text = parsed.content.upper()

    # Check for explicit answer markers first
    answer_patterns = [
        r'(?:ANSWER|RESPONSE|RESULT|CONCLUSION):\s*(YES|NO)\b',
        r'(?:THEREFORE|THUS|SO|HENCE),?\s*(?:THE ANSWER IS\s*)?(YES|NO)\b',
        r'\b(YES|NO)\s*[.!]?\s*$',  # YES/NO at end of response
        r'^\s*(YES|NO)\b',  # YES/NO at start
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            answer = match.group(1) if match.lastindex else match.group(0)
            return answer.strip() == "YES"

    # Fallback: count occurrences (last one wins for reasoning models)
    yes_matches = list(re.finditer(r'\bYES\b', text))
    no_matches = list(re.finditer(r'\bNO\b', text))

    if yes_matches and no_matches:
        # If both present, take the last occurring one (likely the conclusion)
        last_yes = yes_matches[-1].start() if yes_matches else -1
        last_no = no_matches[-1].start() if no_matches else -1
        return last_yes > last_no
    elif yes_matches:
        return True
    elif no_matches:
        return False

    return None


def extract_json_from_reasoning(content: str) -> Optional[str]:
    """
    Extract JSON from a response that may contain reasoning.

    Handles models that output thinking/reasoning before the JSON.

    Args:
        content: Raw model output

    Returns:
        Extracted JSON string or None
    """
    if not content:
        return None

    # First strip thinking tags
    parsed = parse_output(content)
    text = parsed.content

    # Look for JSON patterns
    json_patterns = [
        re.compile(r'```json\s*([\s\S]*?)```', re.IGNORECASE),
        re.compile(r'```\s*([\s\S]*?)```'),
        re.compile(r'(\{[\s\S]*\})'),  # Raw JSON object
    ]

    for pattern in json_patterns:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()

    return None
