# gestalt

A voice-driven agent demo: **speak a command → [Qwen3-ASR] transcribes → [Llama, via Ollama] decides → [gestalt MCP] tools act on a sandbox 3D world.** Fully local — no cloud LLM, no API keys.

```
  mic ──► Qwen3-ASR (GPU) ──► text ──► Llama 3.1 8B via Ollama (GPU)
                                              │  tool calls (MCP, stdio)
                                              ▼
                                     gestalt-mcp server
                                     (sandbox world tools)
                                              │
                                              ▼  world.json ──► 3D view
```

`gestalt` itself is a small, installable package: an **MCP server** that exposes a
sandbox "world" of named 3D objects as tools (`add_object`, `move_object`,
`set_color`, `describe_scene`, `find_objects`, `distance`, ...). The MCP server is
LLM-agnostic — a thin bridge (`gestalt.ollama_agent`) drives those tools with a
local Llama model, and Qwen3-ASR turns speech into the commands.

## Layout

| Path | What |
|---|---|
| `src/gestalt/world.py` | `World` — the in-memory scene (pure logic, file-backed, testable) |
| `src/gestalt/mcp_server.py` | `gestalt-mcp` — FastMCP server exposing the world as tools (stdio) |
| `src/gestalt/ollama_agent.py` | MCP↔Ollama bridge + `run_turn` (the agent-turn seam) |
| `notebooks/hermes_voice_agent_colab.ipynb` | **Colab**: Qwen3-ASR + Llama (Ollama) on the GPU → gestalt world, with a 3D visualizer |
| `examples/text_agent.py` | The same Llama→MCP loop driven by typed text |
| `tests/` | Unit tests for the world logic and the MCP→Ollama bridge |

## Run the voice agent (Colab)

Both Qwen3-ASR and the Llama model run locally on a GPU, so use Colab:

1. Open `notebooks/hermes_voice_agent_colab.ipynb` in Colab and set **Runtime → GPU**.
2. Run the cells: it installs deps, clones+installs this repo, installs Ollama and
   pulls `llama3.1:8b`, loads Qwen3-ASR, records a command, and runs it through the
   model + the gestalt tools. On a free T4 (16 GB) the two models fit together
   (Qwen3-ASR-0.6B ~1.5 GB + Llama 3.1 8B Q4 ~5 GB).

## Run the agent locally (typed commands)

Verifies the Llama → gestalt-MCP wiring without speech. Requires Ollama running
locally (CPU works but is slow without a GPU):

```bash
# 1. Install Ollama (https://ollama.com), then pull a tool-capable model:
ollama pull llama3.1:8b          # or llama3.2:3b for something lighter
# 2. Install this package + the agent deps:
pip install -e ".[agent]"
# 3. Run it (OLLAMA_MODEL overrides the default):
python examples/text_agent.py
# you> add a red cube called box at 1 2 0, then a blue sphere above it
```

## Develop / test

```bash
pip install -e ".[dev]"
pytest                                        # world logic + MCP->Ollama bridge
python -m gestalt.mcp_server                  # run the MCP server on stdio (Ctrl-C to stop)
```

Set `GESTALT_WORLD_FILE=/path/world.json` to persist the scene to disk (the
notebook uses this to read and visualize the world after each command).

## How it fits together (and where Hermes plugs in)

- The agent turn is `run_turn(...)` — transport-agnostic about whether the command
  was typed, transcribed, or queued. That's the seam a **Hermes** agent wraps.
- `gestalt-mcp` is a standard MCP server. Swapping the LLM (this repo went from
  Claude to a local Llama) only touched `gestalt.ollama_agent`, not the server.
  It runs over **stdio** here but can be exposed over HTTP/SSE for any MCP client.
- The world is the ground truth (`world.json`), not the chat history; the model can
  call `describe_scene` to re-ground at any time.

## Requirements

- Python ≥ 3.10
- `mcp >= 1.20` (older FastMCP mishandles `X | None` tool annotations)
- For the agent: `ollama` (Python client) + an Ollama server with a tool-capable
  Llama pulled (`llama3.1:8b`, `llama3.2:3b`)
- For speech: `qwen-asr` + a GPU (Colab)
