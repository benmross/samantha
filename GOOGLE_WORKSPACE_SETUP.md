# Google Workspace Integration Setup

This guide will help you integrate Google Workspace (Gmail, Calendar, Drive, Docs, Sheets, etc.) with Samantha.

## Prerequisites

1. **Python 3.10+** and **uvx** installed
2. **Google Cloud Project** with OAuth 2.0 credentials
3. **Google Workspace APIs enabled** (see below)

## Step 1: Enable Google APIs

Visit the [Google Cloud Console](https://console.cloud.google.com/) and enable these APIs:

Quick links:
- [Google Calendar API](https://console.cloud.google.com/flows/enableapi?apiid=calendar-json.googleapis.com)
- [Google Drive API](https://console.cloud.google.com/flows/enableapi?apiid=drive.googleapis.com)
- [Gmail API](https://console.cloud.google.com/flows/enableapi?apiid=gmail.googleapis.com)
- [Google Docs API](https://console.cloud.google.com/flows/enableapi?apiid=docs.googleapis.com)
- [Google Sheets API](https://console.cloud.google.com/flows/enableapi?apiid=sheets.googleapis.com)
- [Google Slides API](https://console.cloud.google.com/flows/enableapi?apiid=slides.googleapis.com)

## Step 2: Create OAuth Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **APIs & Services → Credentials**
3. Click **Create Credentials → OAuth Client ID**
4. Choose **Desktop Application** as the application type
5. Download the credentials and note your Client ID and Client Secret

## Step 3: Configure Environment

Your `.env` file should already have these variables (you've added them):

```bash
GOOGLE_OAUTH_CLIENT_ID="your-client-id"
GOOGLE_OAUTH_CLIENT_SECRET="your-secret"
OAUTHLIB_INSECURE_TRANSPORT=1
```

## Step 4: Start the Google Workspace MCP Server

In a **separate terminal**, start the MCP server:

### Option 1: Use the start script (easiest)
```bash
./start_mcp.sh
```

The script automatically loads credentials from your `.env` file!

### Option 2: Manual start
```bash
# Export your Google OAuth credentials
export GOOGLE_OAUTH_CLIENT_ID="your-client-id"
export GOOGLE_OAUTH_CLIENT_SECRET="your-secret"
export OAUTHLIB_INSECURE_TRANSPORT=1

# Start with core tools (recommended)
uvx workspace-mcp --transport streamable-http --tool-tier core
```

### Other tool tier options:
- `--tool-tier core` - Essential tools (recommended)
- `--tool-tier extended` - Core + additional features
- `--tool-tier complete` - All available tools (more API quota usage)

The server will start on `http://localhost:8000/mcp`.

**Keep this terminal running!** The MCP server must be running for Samantha to access Google Workspace.

## Step 5: Start Samantha

In another terminal, start Samantha normally:

```bash
./venv/bin/python samantha.py
```

## Step 6: Test the Integration

Once Samantha starts, try asking:

- *"Check my Gmail inbox"*
- *"What's on my calendar today?"*
- *"Search my Drive for files about 'project'"*
- *"Create a Google Doc titled 'Meeting Notes'"*
- *"What emails did I get this week?"*

## Authentication Flow

The first time you use a Google Workspace tool:

1. The MCP server will provide an authorization URL
2. Open it in your browser
3. Authorize the app with your Google account
4. The token will be saved for future use

## Available Tools

Samantha now has access to:

### Gmail
- Search emails
- Read email content
- Send emails
- Manage labels
- Create drafts

### Google Calendar
- List calendars
- Get events
- Create events
- Modify events
- Delete events

### Google Drive
- Search files
- Read file content
- Create files
- Share files
- Update files

### Google Docs
- Get document content
- Create documents
- Modify text
- Insert elements
- Export to PDF

### Google Sheets
- Read spreadsheet values
- Modify cell values
- Create spreadsheets
- Manage sheets

### Google Slides
- Create presentations
- Get presentation details
- Update presentations

And more!

## Troubleshooting

### "Could not connect to MCP server"
- Make sure the MCP server is running in a separate terminal
- Check that it's running on `http://localhost:8000/mcp`
- Restart the MCP server and try again

### "Authentication failed"
- Verify your OAuth credentials in `.env`
- Delete `~/.google_workspace_mcp/credentials` to re-authenticate
- Make sure all required APIs are enabled

### "No tools are currently enabled"
- Restart Samantha after starting the MCP server
- Check that the MCP server started successfully

## Tool Tiers

The MCP server organizes tools into tiers to manage API quota:

- **Core** (`--tool-tier core`): Essential tools for basic functionality
- **Extended** (`--tool-tier extended`): Core + additional features
- **Complete** (`--tool-tier complete`): All available tools

Start with **core** and upgrade as needed!

## Notes

- Both servers must be running for Google Workspace integration to work
- The MCP server handles all Google API authentication
- Samantha connects to the MCP server via HTTP
- Your Google credentials are stored securely by the MCP server
