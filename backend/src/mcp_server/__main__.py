"""Entry point for the MCP server.

Usage:
    python -m src.mcp_server

Runs the Geopolitical Market Tracker MCP server over stdio transport,
allowing Claude Desktop and Claude Code to call the registered tools.
"""

from src.mcp_server.server import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
