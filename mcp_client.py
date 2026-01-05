#!/usr/bin/env python3
"""
MCP Client for Google Workspace integration
Connects to the Google Workspace MCP server and provides tool access
"""

import json
from typing import List, Dict, Any, Optional
import asyncio
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client


class MCPClient:
    """Client for connecting to MCP servers via streamable HTTP"""

    def __init__(self, server_url: str = "http://localhost:8000/mcp"):
        """
        Initialize MCP client

        Args:
            server_url: URL of the MCP server
        """
        self.server_url = server_url.rstrip('/')
        self.session = None
        self.tools_cache = None

    async def initialize(self) -> bool:
        """
        Initialize connection to MCP server

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create streamable HTTP client and session
            async with streamable_http_client(self.server_url) as streams:
                # Extract only read and write streams (ignore extra values)
                read_stream, write_stream = streams[0], streams[1]
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize the session
                    await session.initialize()

                    # Store session for later use
                    self.session = session
                    print("✅ Connected to Google Workspace MCP server")
                    return True

        except Exception as e:
            print(f"⚠️  Could not connect to MCP server: {e}")
            print("   Make sure the Google Workspace MCP server is running")
            import traceback
            traceback.print_exc()
            return False

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools from MCP server

        Returns:
            List of tool definitions
        """
        if self.tools_cache:
            return self.tools_cache

        try:
            # Use context manager for session
            async with streamable_http_client(self.server_url) as streams:
                # Extract only read and write streams (ignore extra values)
                read_stream, write_stream = streams[0], streams[1]
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    # List tools
                    result = await session.list_tools()
                    tools = result.tools if hasattr(result, 'tools') else []

                    # Convert to dict format
                    tools_list = []
                    for tool in tools:
                        tool_dict = {
                            "name": tool.name,
                            "description": tool.description if hasattr(tool, 'description') else "",
                            "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                        }
                        tools_list.append(tool_dict)

                    self.tools_cache = tools_list
                    return tools_list

        except Exception as e:
            print(f"⚠️  Error listing MCP tools: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the MCP server

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result
        """
        try:
            # Use context manager for session
            async with streamable_http_client(self.server_url) as streams:
                # Extract only read and write streams (ignore extra values)
                read_stream, write_stream = streams[0], streams[1]
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    # Call tool
                    result = await session.call_tool(tool_name, arguments)

                    # Convert result to dict
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": item.text if hasattr(item, 'text') else str(item)
                            }
                            for item in (result.content if hasattr(result, 'content') else [])
                        ]
                    }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Error calling MCP tool: {str(e)}"}

    async def close(self):
        """Close the MCP session"""
        # Sessions are managed by context managers
        pass


def convert_mcp_tool_to_openrouter(mcp_tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert MCP tool definition to OpenRouter/OpenAI function format

    Args:
        mcp_tool: MCP tool definition

    Returns:
        OpenRouter-compatible tool definition
    """
    return {
        "type": "function",
        "function": {
            "name": mcp_tool["name"],
            "description": mcp_tool.get("description", ""),
            "parameters": mcp_tool.get("inputSchema", {
                "type": "object",
                "properties": {},
                "required": []
            })
        }
    }
