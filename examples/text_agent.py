"""Text-driven Hermes loop — the same agent as the notebook, minus the ASR.

Type commands instead of speaking them. Runs anywhere (no GPU needed), so use
it to verify the Claude -> gestalt-MCP wiring on your own machine.

    pip install -e ".[agent]"
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/text_agent.py

This is the integration seam a "Hermes" agent would wrap: `run_command(text)`
is transport-agnostic about where `text` came from (typed, transcribed, queued).
"""

from __future__ import annotations

import asyncio
import os
import sys

from anthropic import AsyncAnthropic
from anthropic.lib.tools.mcp import async_mcp_tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MODEL = "claude-opus-4-8"
SYSTEM = (
    "You are Hermes, an assistant that manipulates a 3D sandbox world through "
    "tools. Commands may be transcribed from speech, so tolerate minor "
    "mis-hearings and pick the most plausible intent. Use the tools to carry "
    "out each command, then confirm what you did in one short sentence. If a "
    "command is ambiguous, make a reasonable choice and say what you assumed."
)


async def run_command(client: AsyncAnthropic, mcp_client: ClientSession, text: str) -> str:
    """Run one user turn through Claude with the gestalt tools attached."""
    tools = (await mcp_client.list_tools()).tools
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=SYSTEM,
        messages=[{"role": "user", "content": text}],
        tools=[async_mcp_tool(t, mcp_client) for t in tools],
    )
    final = None
    async for message in runner:
        final = message
    reply = "".join(b.text for b in final.content if b.type == "text") if final else ""
    return reply.strip()


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY first.")
    os.environ.setdefault("GESTALT_WORLD_FILE", "world.json")

    client = AsyncAnthropic()
    # Launch the gestalt MCP server as a subprocess over stdio. Using the current
    # interpreter + `-m` is portable (no dependence on the console script's PATH).
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "gestalt.mcp_server"],
        env=dict(os.environ),
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as mcp_client:
            await mcp_client.initialize()
            print("Hermes ready. Type a command (or 'quit').")
            print('Try: "add a red cube called box at 1 2 0, then a blue sphere above it"')
            while True:
                try:
                    text = input("\nyou> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if text.lower() in {"quit", "exit", ""}:
                    break
                reply = await run_command(client, mcp_client, text)
                print(f"hermes> {reply}")


if __name__ == "__main__":
    asyncio.run(main())
