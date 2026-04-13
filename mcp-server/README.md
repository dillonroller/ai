# MCP Server

A minimal [Model Context Protocol](https://modelcontextprotocol.io) server exposing a small SQLite-backed library as tools, a resource, and a prompt. Any MCP-compatible client (Claude Desktop, Claude Code, Cursor, etc.) can connect and use it.

## Install

```bash
pip install "mcp[cli]" pydantic
```

## Run standalone

```bash
python main.py
```

The server speaks MCP over stdio and seeds `library.db` on first run.

## Connect from Claude Desktop

Add this to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "library": {
      "command": "python",
      "args": ["/absolute/path/to/python/mcp-server/main.py"]
    }
  }
}
```

Restart Claude Desktop. The `library` server's tools, resource, and prompt will appear.

## What it exposes

| Kind | Name | Purpose |
|------|------|---------|
| tool | `search_books` | Search by title/author with a rating floor |
| tool | `add_book` | Insert a new book |
| tool | `top_books` | Highest-rated books |
| resource | `library://schema` | Live SQL schema |
| prompt | `recommend` | Mood-based recommendation prompt |

## Why MCP

One protocol, many clients. Write the integration once; every MCP-aware LLM app can use it without bespoke glue.
