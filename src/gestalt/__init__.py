"""gestalt: a sandbox-world MCP server for voice-driven agents.

Pipeline: speech --(Qwen3-ASR)--> text --(Claude)--> tool calls --> gestalt world.
"""

from .world import World, WorldError, WorldObject

__all__ = ["World", "WorldError", "WorldObject"]
__version__ = "0.1.0"
