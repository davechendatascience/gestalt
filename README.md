# gestalt

A voice-driven agent demo: **speak a command → [Qwen3-ASR] transcribes → [Claude] decides → [gestalt MCP] tools act on a sandbox 3D world.**

```
  mic ──► Qwen3-ASR (GPU) ──► text ──► Claude (claude-opus-4-8)
                                              │  tool calls (MCP, stdio)
                                              ▼
                                     gestalt-mcp server
                                     (sandbox world tools)
                                              │
                                              ▼  world.json ──► 3D view
```

`gestalt` itself is a small, installable package: an **MCP server** that exposes a
sandbox "world" of named 3D objects as tools (`add_object`, `move_object`,
`set_color`, `describe_scene`, `find_objects`, `distance`, ...). Claude drives those
tools; Qwen3-ASR turns speech into the commands.

## Layout

| Path | What |
|---|---|
| `src/gestalt/world.py` | `World` — the in-memory scene (pure logic, file-backed, testable) |
| `src/gestalt/mcp_server.py` | `gestalt-mcp` — FastMCP server exposing the world as tools (stdio) |
| `notebooks/hermes_voice_agent_colab.ipynb` | **Colab**: Qwen3-ASR (GPU) → Claude → gestalt world, with a 3D visualizer |
| `examples/text_agent.py` | The same Claude→MCP loop driven by typed text — runs anywhere, no GPU |
| `tests/test_world.py` | Unit tests for the world logic |

## Run the voice agent (Colab)

Qwen3-ASR runs locally on a GPU, so use Colab:

1. Open `notebooks/hermes_voice_agent_colab.ipynb` in Colab and set **Runtime → GPU**.
2. Add `ANTHROPIC_API_KEY` to Colab **Secrets** (🔑).
3. Run the cells: it installs deps, clones+installs this repo, loads Qwen3-ASR,
   records a command, and runs it through Claude + the gestalt tools.

## Run the agent locally (no GPU, typed commands)

Verifies the Claude → gestalt-MCP wiring without speech:

```bash
pip install -e ".[agent]"
export ANTHROPIC_API_KEY=sk-ant-...          # Windows (PowerShell): $env:ANTHROPIC_API_KEY="sk-ant-..."
python examples/text_agent.py
# you> add a red cube called box at 1 2 0, then a blue sphere above it
```

## Develop / test

```bash
pip install -e ".[dev]"
pytest                                        # world logic
python -m gestalt.mcp_server                  # run the MCP server on stdio (Ctrl-C to stop)
```

Set `GESTALT_WORLD_FILE=/path/world.json` to persist the scene to disk (the
notebook uses this to read and visualize the world after each command).

## How it fits together (and where Hermes plugs in)

- The agent turn is `run_command(text)` — transport-agnostic about whether `text`
  was typed, transcribed, or queued. That's the seam a **Hermes** agent wraps.
- `gestalt-mcp` is a standard MCP server. It's launched over **stdio** here, but the
  same server can be exposed over HTTP/SSE and shared by any MCP client.
- The world is the ground truth (`world.json`), not the chat history; the agent can
  call `describe_scene` to re-ground at any time.

## Requirements

- Python ≥ 3.10
- `mcp >= 1.20` (older FastMCP mishandles `X | None` tool annotations)
- For the agent: `anthropic[mcp]` and an `ANTHROPIC_API_KEY`
- For speech: `qwen-asr` + a GPU (Colab)
