"""Text-driven Hermes loop — the same agent as the notebook, minus the ASR.

Type commands instead of speaking them. Drives a local Llama model (via Ollama)
over the gestalt MCP tools. Use it to verify the wiring on your own machine.

Prereqs:
    1. Install + run Ollama (https://ollama.com) and pull a tool-capable model:
         ollama pull llama3.1:8b
       (Ollama must be serving — `ollama serve` or the desktop app.)
    2. pip install -e ".[agent]"
    3. python examples/text_agent.py        # OLLAMA_MODEL overrides the model

`run_turn(...)` is the integration seam a "Hermes" agent would wrap: it is
agnostic about where `text` came from (typed, transcribed, queued).
"""

from __future__ import annotations

import asyncio
import os

from ollama import AsyncClient

from gestalt.ollama_agent import DEFAULT_MODEL, run_turn


async def main() -> None:
    os.environ.setdefault("GESTALT_WORLD_FILE", "world.json")
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    client = AsyncClient()  # honors OLLAMA_HOST, defaults to http://localhost:11434
    history: list[dict] = []

    print(f"Hermes ready (model={model}). Type a command (or 'quit').")
    print('Try: "add a red cube called box at 1 2 0, then a blue sphere above it"')
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit", ""}:
            break
        reply = await run_turn(
            client, model, history, text, on_tool=lambda n, a: print(f"  · {n}({a})")
        )
        print(f"hermes> {reply}")


if __name__ == "__main__":
    asyncio.run(main())
