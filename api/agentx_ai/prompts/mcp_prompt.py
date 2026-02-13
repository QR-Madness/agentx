"""
MCP Tools Prompt Generator.

Generates a prompt section that teaches the model how to use
available MCP tools effectively.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


def generate_mcp_tools_prompt(
    tools: list[dict],
    include_examples: bool = True,
    max_tools: Optional[int] = None,
) -> str:
    """
    Generate a prompt section describing available MCP tools.
    
    Args:
        tools: List of tool dictionaries with name, description, inputSchema
        include_examples: Whether to include usage examples
        max_tools: Maximum number of tools to include (None for all)
        
    Returns:
        Formatted prompt string describing the tools
    """
    if not tools:
        return ""
    
    # Limit tools if specified
    if max_tools and len(tools) > max_tools:
        tools = tools[:max_tools]
        truncated = True
    else:
        truncated = False
    
    parts = [
        "## Available Tools",
        "",
        "You have access to the following tools to help accomplish tasks:",
        "",
    ]
    
    for tool in tools:
        name = tool.get("name", "unknown")
        description = tool.get("description", "No description available")
        schema = tool.get("inputSchema", {})
        server = tool.get("server", "unknown")
        
        parts.append(f"### {name}")
        parts.append(f"**Server:** {server}")
        parts.append(f"**Description:** {description}")
        
        # Format input parameters if available
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        if properties:
            parts.append("**Parameters:**")
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                is_required = param_name in required
                req_marker = " (required)" if is_required else " (optional)"
                
                parts.append(f"- `{param_name}` ({param_type}){req_marker}: {param_desc}")
        
        parts.append("")
    
    if truncated:
        parts.append(f"*Note: Showing {max_tools} of {len(tools)} available tools.*")
        parts.append("")
    
    # Add usage instructions
    parts.extend([
        "## Tool Usage Guidelines",
        "",
        "When using tools:",
        "1. **Choose the right tool** - Select the most appropriate tool for the task",
        "2. **Provide required parameters** - Ensure all required parameters are included",
        "3. **Handle errors gracefully** - If a tool fails, explain the issue and try alternatives",
        "4. **Explain your actions** - Tell the user what tool you're using and why",
        "",
    ])
    
    if include_examples:
        parts.extend([
            "## Example Tool Usage",
            "",
            "When you need to use a tool, indicate it clearly in your response.",
            "The system will execute the tool and provide the results.",
            "",
        ])
    
    return "\n".join(parts)


def format_tool_for_prompt(tool: dict) -> str:
    """
    Format a single tool for inclusion in a prompt.
    
    Args:
        tool: Tool dictionary with name, description, inputSchema
        
    Returns:
        Formatted string for the tool
    """
    name = tool.get("name", "unknown")
    description = tool.get("description", "No description")
    
    return f"- **{name}**: {description}"
