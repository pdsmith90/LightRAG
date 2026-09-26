"""End-to-end: launch the proxy over stdio exactly as an MCP client does, list its
tools and call search_library against the fake LightRAG server."""
from __future__ import annotations

import sys
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PY = Path(__file__).resolve().parents[2] / "mcp" / "lightrag_mcp_server.py"

REFERENCES = [{"reference_id": "1", "file_path": "AAAA0001__Attention_Is_All_You_Need.md"}]


def _run(fake_lightrag, metadata_file, api_key, calls):
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PY)],
        env={"LIGHTRAG_BASE_URL": fake_lightrag.url,
             "LIGHTRAG_API_KEY": api_key,
             "ZOTERO_METADATA": str(metadata_file)},
    )

    # stdio_client + ClientSession: the client API that exists across all of mcp 2.x.
    async def main():
        with anyio.fail_after(60):
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    results = [await session.call_tool(name, args) for name, args in calls]
                    return tools, results

    return anyio.run(main)


def test_stdio_lists_tools_and_compiles_citations(fake_lightrag, metadata_file, api_key):
    fake_lightrag.routes[("POST", "/query")] = lambda body: (
        (500, {"detail": "server failure"}) if body["query"] == "boom"
        else (200, {"response": "RAW CONTEXT", "references": REFERENCES}))
    tools, results = _run(fake_lightrag, metadata_file, api_key, [
        ("search_library", {"query": "attention", "top_k": 500}),
        ("health", {}),
        ("search_library", {"query": "boom"}),
    ])
    assert {t.name for t in tools.tools} == {"search_library", "get_sources",
                                            "add_note", "health"}
    search, health, failed = results
    assert not search.is_error
    text = search.content[0].text
    assert text.startswith("### Retrieved documents (compiled citations)\n- [1] "
                           "Ashish Vaswani, Noam Shazeer, Niki Parmar. (2017). "
                           "Attention Is All You Need.")
    assert "[zotero://select/library/items/AAAA0001]" in text
    assert text.endswith("\n\nRAW CONTEXT")
    assert health.content[0].text.startswith("200:")
    # An HTTP error from the server surfaces as a tool error, not an empty result.
    assert failed.is_error
    assert "500" in failed.content[0].text
    assert "server failure" in failed.content[0].text
    sent = fake_lightrag.calls("/query")[0]
    assert sent["json"]["top_k"] == 40
    assert sent["headers"]["x-api-key"] == api_key
