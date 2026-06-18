"""Bridge the gestalt MCP tools to a local Ollama (Llama) model.

The gestalt MCP server is LLM-agnostic; this module is the only piece that knows
about Ollama. It converts MCP tool schemas into Ollama's ``tools`` format and runs
the manual tool-calling loop: ask the model -> if it requests tools, execute them
via the MCP client -> feed results back -> repeat until the model answers.

``run_turn`` is the agent-turn seam: it doesn't care whether ``text`` was typed or
transcribed from speech, which is what a Hermes agent will wrap.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

DEFAULT_MODEL = "llama3.1:8b"

SYSTEM = (
    "You are Hermes, an assistant that manipulates a 3D sandbox world through "
    "tools. Commands may be transcribed from speech, so tolerate minor "
    "mis-hearings and pick the most plausible intent (e.g. 'q' likely means "
    "'cube'). Use the tools to carry out the command. Call each tool at most "
    "once per distinct action; once the tools have done the work, reply with one "
    "short sentence confirming what you did and stop. If a command refers to "
    "something implicitly, call describe_scene to re-ground. If no tool is "
    "needed, just answer briefly."
)


def mcp_to_ollama_tools(mcp_tools: list) -> list[dict]:
    """Convert MCP tool definitions into Ollama's ``tools`` schema list."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in mcp_tools
    ]


def tool_result_text(result: Any) -> str:
    """Extract plain text from an MCP CallToolResult."""
    parts = [getattr(c, "text", None) for c in getattr(result, "content", [])]
    parts = [p for p in parts if p]
    return "\n".join(parts) if parts else str(getattr(result, "content", result))


def server_params() -> StdioServerParameters:
    """Launch the gestalt MCP server with the current interpreter (portable)."""
    return StdioServerParameters(
        command=sys.executable, args=["-m", "gestalt.mcp_server"], env=dict(os.environ)
    )


async def run_turn(
    ollama_client,
    model: str,
    history: list[dict],
    text: str,
    system: str = SYSTEM,
    on_tool: Callable[[str, dict], None] | None = None,
    max_steps: int = 8,
) -> str:
    """Run one user turn: open the gestalt MCP server, loop the model over tools.

    ``history`` is a list of plain ``{"role", "content"}`` dicts; it is appended
    to in place with this turn's user text and the final assistant reply so the
    next turn has context for references like "move it up".
    """
    # In notebooks, sys.stderr (ipykernel) has no real file descriptor, which breaks
    # the MCP subprocess launch — route the server's stderr to os.devnull.
    errlog = open(os.devnull, "w")
    async with stdio_client(server_params(), errlog=errlog) as (read, write):
        async with ClientSession(read, write) as mcp:
            await mcp.initialize()
            tools = mcp_to_ollama_tools((await mcp.list_tools()).tools)
            messages: list = [{"role": "system", "content": system}, *history,
                              {"role": "user", "content": text}]
            reply = ""
            for _ in range(max_steps):
                resp = await ollama_client.chat(model=model, messages=messages, tools=tools)
                msg = resp.message
                messages.append(msg)
                if not msg.tool_calls:
                    reply = (msg.content or "").strip()
                    break
                for call in msg.tool_calls:
                    name = call.function.name
                    args = dict(call.function.arguments or {})
                    if on_tool:
                        on_tool(name, args)
                    result = await mcp.call_tool(name, args)
                    messages.append(
                        {"role": "tool", "tool_name": name, "content": tool_result_text(result)}
                    )
            else:
                # Hit max_steps without a final answer; surface what we have.
                reply = (messages[-1].get("content", "") if isinstance(messages[-1], dict)
                         else (getattr(messages[-1], "content", "") or "")).strip()
                reply = reply or "(stopped after too many tool calls)"
    errlog.close()

    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": reply})
    return reply
