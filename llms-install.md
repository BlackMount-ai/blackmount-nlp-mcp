# Install blackmount-nlp-mcp

## One-line install

```bash
pip install blackmount-nlp-mcp
```

## Claude Desktop

File location:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

Add to `"mcpServers"`:

```json
{
  "mcpServers": {
    "nlp": {
      "command": "blackmount-nlp-mcp"
    }
  }
}
```

## Claude Code

```bash
claude mcp add nlp blackmount-nlp-mcp
```

## Cursor

File: `.cursor/mcp.json` in your project root (or `~/.cursor/mcp.json` for global):

```json
{
  "mcpServers": {
    "nlp": {
      "command": "blackmount-nlp-mcp"
    }
  }
}
```

## Cline / VS Code

In your Cline MCP settings, add:

```json
{
  "blackmount-nlp": {
    "command": "blackmount-nlp-mcp",
    "args": []
  }
}
```

## Windsurf

Add to MCP config:

```json
{
  "mcpServers": {
    "nlp": {
      "command": "blackmount-nlp-mcp"
    }
  }
}
```

## Verify

After restarting your client, ask: "What NLP tools do you have available?"

You should see 45 tools covering sentiment, readability, keywords, similarity, cleaning, detection, and summarization.

## Requirements

- Python 3.10+
- No other dependencies needed
