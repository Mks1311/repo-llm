"""
MCP client: spawns the server process and opens a session to it.

Same nested `async with` the MCP docs use, wrapped in a context manager so
the session can stay open for a whole chat instead of one call.
"""

import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@asynccontextmanager
async def connect(repo_name):
    """Start the MCP server and yield an initialized session."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "repo_llm.mcp.server"],
        # Tells the server which indexed repo its tool should search.
        env={"REPO_LLM_REPO": repo_name},
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session: 
            await session.initialize()
            yield session
