#!/bin/bash
# Start Google Workspace MCP Server
# Loads credentials from .env file and starts the server

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load environment variables from .env file
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(cat "$SCRIPT_DIR/.env" | grep -E '^(GOOGLE_OAUTH_CLIENT_ID|GOOGLE_OAUTH_CLIENT_SECRET|OAUTHLIB_INSECURE_TRANSPORT|USER_GOOGLE_EMAIL)=' | xargs)
    echo "✅ Loaded credentials from .env"
else
    echo "❌ .env file not found!"
    echo "   Please create a .env file with your Google OAuth credentials"
    exit 1
fi

# Check if required credentials are set
if [ -z "$GOOGLE_OAUTH_CLIENT_ID" ] || [ -z "$GOOGLE_OAUTH_CLIENT_SECRET" ]; then
    echo "❌ Missing Google OAuth credentials in .env file!"
    echo "   Please add GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET"
    exit 1
fi

echo "🚀 Starting Google Workspace MCP Server..."
echo ""

# Start the MCP server with core tools
# You can change 'core' to 'extended' or 'complete' for more tools
uvx workspace-mcp --transport streamable-http --tool-tier core
