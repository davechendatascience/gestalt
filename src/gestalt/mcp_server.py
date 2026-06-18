"""gestalt MCP server: exposes the sandbox :class:`~gestalt.world.World` as tools.

Run over stdio (the default MCP transport for a local subprocess)::

    gestalt-mcp                      # console entry point
    python -m gestalt.mcp_server     # equivalent

Set ``GESTALT_WORLD_FILE=/path/world.json`` to persist the scene to disk so
another process can read it. Each tool returns a short human-readable string —
the format an LLM consumes most reliably — and surfaces errors as
``Error: ...`` text rather than crashing the tool call.
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from .world import World, WorldError


def build_server(world: World | None = None) -> FastMCP:
    """Build a FastMCP server bound to ``world`` (or a fresh, file-backed one)."""
    world = world or World(path=os.environ.get("GESTALT_WORLD_FILE"))
    mcp = FastMCP("gestalt-world")

    def _fmt(obj: dict) -> str:
        color = f"{obj['color']} " if obj.get("color") else ""
        return (
            f"{obj['name']} ({color}{obj['kind']}) at "
            f"({obj['x']:g}, {obj['y']:g}, {obj['z']:g})"
        )

    @mcp.tool()
    def add_object(
        name: str,
        kind: str = "thing",
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        color: str | None = None,
        notes: str | None = None,
    ) -> str:
        """Add a new object to the world at position (x, y, z).

        kind is a free-form label (e.g. cube, sphere, lamp). Names are unique.
        """
        try:
            return "Added " + _fmt(world.add(name, kind, x, y, z, color, notes)) + "."
        except WorldError as e:
            return f"Error: {e}"

    @mcp.tool()
    def place_object(name: str, x: float, y: float, z: float) -> str:
        """Move an existing object to an absolute position (x, y, z)."""
        try:
            return "Placed " + _fmt(world.place(name, x, y, z)) + "."
        except WorldError as e:
            return f"Error: {e}"

    @mcp.tool()
    def move_object(name: str, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> str:
        """Translate an object by a relative offset (dx, dy, dz)."""
        try:
            return "Moved " + _fmt(world.move(name, dx, dy, dz)) + "."
        except WorldError as e:
            return f"Error: {e}"

    @mcp.tool()
    def set_color(name: str, color: str) -> str:
        """Set the color of an existing object."""
        try:
            return "Recolored " + _fmt(world.set_color(name, color)) + "."
        except WorldError as e:
            return f"Error: {e}"

    @mcp.tool()
    def remove_object(name: str) -> str:
        """Remove an object from the world."""
        try:
            world.remove(name)
            return f"Removed {name}."
        except WorldError as e:
            return f"Error: {e}"

    @mcp.tool()
    def describe_object(name: str) -> str:
        """Describe a single object (kind, color, position, notes)."""
        try:
            o = world.get(name)
            note = f" Notes: {o['notes']}." if o.get("notes") else ""
            return _fmt(o) + "." + note
        except WorldError as e:
            return f"Error: {e}"

    @mcp.tool()
    def describe_scene() -> str:
        """List every object in the world with its kind, color and position."""
        return world.scene_text()

    @mcp.tool()
    def find_objects(
        kind: str | None = None,
        color: str | None = None,
        near: str | None = None,
        radius: float | None = None,
    ) -> str:
        """Find objects by kind and/or color, optionally within radius of `near`."""
        try:
            hits = world.find(kind=kind, color=color, near=near, radius=radius)
        except WorldError as e:
            return f"Error: {e}"
        if not hits:
            return "No matching objects."
        return "Found:\n" + "\n".join("  - " + _fmt(o) for o in hits)

    @mcp.tool()
    def distance(a: str, b: str) -> str:
        """Euclidean distance between two objects."""
        try:
            return f"Distance from {a} to {b} is {world.distance(a, b):.3g}."
        except WorldError as e:
            return f"Error: {e}"

    @mcp.tool()
    def clear_scene() -> str:
        """Remove all objects from the world."""
        return f"Cleared {world.clear()} object(s)."

    return mcp


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
