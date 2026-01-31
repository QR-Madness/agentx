"""
Server Registry: Configuration and tracking of MCP servers.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TransportType(str, Enum):
    """Supported MCP transport types."""
    STDIO = "stdio"
    SSE = "sse"
    WEBSOCKET = "websocket"


@dataclass
class ServerConfig:
    """Configuration for an MCP server."""
    
    name: str
    transport: TransportType
    
    # For stdio transport
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    
    # For SSE/WebSocket transport
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    
    # Connection settings
    timeout: float = 30.0
    auto_reconnect: bool = True
    
    def __post_init__(self):
        if isinstance(self.transport, str):
            self.transport = TransportType(self.transport)
    
    def validate(self) -> bool:
        """Validate the server configuration."""
        if self.transport == TransportType.STDIO:
            if not self.command:
                raise ValueError(f"Server '{self.name}': stdio transport requires 'command'")
        elif self.transport in (TransportType.SSE, TransportType.WEBSOCKET):
            if not self.url:
                raise ValueError(f"Server '{self.name}': {self.transport.value} transport requires 'url'")
        return True
    
    def resolve_env(self) -> dict[str, str]:
        """Resolve environment variables (e.g., ${GITHUB_TOKEN} -> actual value)."""
        resolved = {}
        for key, value in self.env.items():
            if value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                env_value = os.environ.get(env_var)
                if env_value is None:
                    logger.warning(f"Environment variable '{env_var}' not set for server '{self.name}'")
                    resolved[key] = ""
                else:
                    resolved[key] = env_value
            else:
                resolved[key] = value
        return resolved
    
    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "ServerConfig":
        """Create ServerConfig from dictionary."""
        return cls(
            name=name,
            transport=TransportType(data.get("transport", "stdio")),
            command=data.get("command"),
            args=data.get("args", []),
            env=data.get("env", {}),
            url=data.get("url"),
            headers=data.get("headers", {}),
            timeout=data.get("timeout", 30.0),
            auto_reconnect=data.get("auto_reconnect", True),
        )


class ServerRegistry:
    """Registry for managing MCP server configurations."""
    
    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize the server registry.
        
        Args:
            config_path: Path to mcp_servers.json config file.
                        Defaults to 'mcp_servers.json' in project root.
        """
        self._servers: dict[str, ServerConfig] = {}
        self._config_path = Path(config_path) if config_path else None
        
        if self._config_path and self._config_path.exists():
            self.load_from_file(self._config_path)
    
    def load_from_file(self, path: Path) -> None:
        """Load server configurations from a JSON file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            servers = data.get("servers", {})
            for name, config in servers.items():
                try:
                    server = ServerConfig.from_dict(name, config)
                    server.validate()
                    self._servers[name] = server
                    logger.info(f"Loaded MCP server config: {name}")
                except Exception as e:
                    logger.warning(f"Failed to load server '{name}': {e}")
        except Exception as e:
            logger.error(f"Failed to load MCP config from {path}: {e}")
    
    def register(self, config: ServerConfig) -> None:
        """Register a server configuration."""
        config.validate()
        self._servers[config.name] = config
        logger.info(f"Registered MCP server: {config.name}")
    
    def unregister(self, name: str) -> bool:
        """Unregister a server configuration."""
        if name in self._servers:
            del self._servers[name]
            logger.info(f"Unregistered MCP server: {name}")
            return True
        return False
    
    def get(self, name: str) -> ServerConfig | None:
        """Get a server configuration by name."""
        return self._servers.get(name)
    
    def list(self) -> list[ServerConfig]:
        """List all registered server configurations."""
        return list(self._servers.values())
    
    def list_names(self) -> list[str]:
        """List all registered server names."""
        return list(self._servers.keys())
    
    def to_dict(self) -> dict[str, Any]:
        """Export registry as dictionary."""
        return {
            "servers": {
                name: {
                    "transport": config.transport.value,
                    "command": config.command,
                    "args": config.args,
                    "env": config.env,
                    "url": config.url,
                    "headers": config.headers,
                    "timeout": config.timeout,
                    "auto_reconnect": config.auto_reconnect,
                }
                for name, config in self._servers.items()
            }
        }
    
    def save_to_file(self, path: Path | None = None) -> None:
        """Save registry to a JSON file."""
        save_path = path or self._config_path
        if not save_path:
            raise ValueError("No config path specified")
        
        with open(save_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved MCP config to {save_path}")
