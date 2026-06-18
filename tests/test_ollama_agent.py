"""Tests for the MCP -> Ollama bridge helpers (no network / no ollama needed)."""

from types import SimpleNamespace

from gestalt.ollama_agent import mcp_to_ollama_tools, tool_result_text


def test_mcp_to_ollama_tools_shape():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    mcp_tools = [SimpleNamespace(name="add_object", description="Add a thing.", inputSchema=schema)]
    out = mcp_to_ollama_tools(mcp_tools)
    assert out == [
        {
            "type": "function",
            "function": {
                "name": "add_object",
                "description": "Add a thing.",
                "parameters": schema,
            },
        }
    ]


def test_mcp_to_ollama_tools_handles_missing_description():
    t = SimpleNamespace(name="clear_scene", description=None, inputSchema={"type": "object"})
    out = mcp_to_ollama_tools([t])
    assert out[0]["function"]["description"] == ""


def test_tool_result_text_joins_text_blocks():
    result = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Added box."),
            SimpleNamespace(type="text", text="At (1, 2, 0)."),
        ]
    )
    assert tool_result_text(result) == "Added box.\nAt (1, 2, 0)."


def test_tool_result_text_falls_back_when_no_text():
    result = SimpleNamespace(content=[SimpleNamespace(type="image")])
    # No .text on the block -> stringified fallback, not a crash.
    assert isinstance(tool_result_text(result), str)
