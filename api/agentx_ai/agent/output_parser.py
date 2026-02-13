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
    # Standard thinking tags
    re.compile(r'<thinking>(.*?)</thinking>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE),
    # Alternative formats
    re.compile(r'\[thinking\](.*?)\[/thinking\]', re.DOTALL | re.IGNORECASE),
    re.compile(r'\[think\](.*?)\[/think\]', re.DOTALL | re.IGNORECASE),
    # Anthropic-style
    re.compile(r'<internal_monologue>(.*?)</internal_monologue>', re.DOTALL | re.IGNORECASE),
]


def parse_output(raw_output: str) -> ParsedOutput:
    """
    Parse model output to extract thinking and content.
    
    Handles various thinking tag formats:
    - <thinking>...</thinking>
    - <think>...</think>
    - [thinking]...[/thinking]
    - <internal_monologue>...</internal_monologue>
    
    Args:
        raw_output: The raw model output string
        
    Returns:
        ParsedOutput with separated content and thinking
    """
    if not raw_output:
        return ParsedOutput(content="", raw="")
    
    thinking_parts = []
    content = raw_output
    
    # Extract all thinking blocks
    for pattern in THINKING_PATTERNS:
        matches = pattern.findall(content)
        for match in matches:
            thinking_parts.append(match.strip())
        # Remove thinking blocks from content
        content = pattern.sub('', content)
    
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
