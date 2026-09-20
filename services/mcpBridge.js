/**
 * mcpBridge.js — Connects to a remote MCP (Model Context Protocol) server as
 * a client, and adapts its tools into the shape Gemini's Live API expects.
 *
 * Gemini's Live API does NOT support MCP directly — `mcpToTool()` from
 * `@google/genai` only works with the non-streaming
 * `ai.models.generateContent` path (confirmed against
 * ai.google.dev/gemini-api/docs/live-api/tools, which documents live tools
 * exclusively as plain `functionDeclarations`). So this bridges the two
 * protocols by hand:
 *
 *   MCP tools/list  → Gemini functionDeclarations (JSON Schema sanitized —
 *                      Gemini's `parameters` schema rejects $ref/$defs/
 *                      additionalProperties/$schema that MCP servers
 *                      commonly emit, e.g. from zod-to-json-schema)
 *   Gemini toolCall → MCP tools/call → Gemini sendToolResponse
 *
 * One bridge per phone call — opened alongside the Gemini Live session and
 * closed with it. A single shared MCP connection reused across concurrent
 * calls risks tool calls from different callers interleaving on a server
 * session not designed for that; a fresh connection per call sidesteps it
 * at the cost of one extra handshake per call.
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js';

const CLIENT_INFO = { name: 'ztwilio-voice-agent', version: '1.0.0' };

/** Strips JSON Schema constructs Gemini's function `parameters` schema
 *  doesn't accept, so an MCP server's inputSchema can be used as-is. */
function sanitizeSchema(schema) {
  if (!schema || typeof schema !== 'object') return schema;
  if (Array.isArray(schema)) return schema.map(sanitizeSchema);

  const { $ref, $defs, additionalProperties, $schema, ...rest } = schema;
  const out = {};
  for (const [key, value] of Object.entries(rest)) {
    out[key] = sanitizeSchema(value);
  }
  return out;
}

/**
 * @param {string} mcpUrl
 * @returns {Promise<{ functionDeclarations: object[], callTool: (name: string, args: object) => Promise<any>, close: () => Promise<void> }>}
 */
export async function connectMcpBridge(mcpUrl) {
  let client = new Client(CLIENT_INFO);
  try {
    await client.connect(new StreamableHTTPClientTransport(new URL(mcpUrl)));
  } catch (err) {
    console.warn('[mcp] Streamable HTTP transport failed, falling back to SSE:', err.message ?? err);
    client = new Client(CLIENT_INFO);
    await client.connect(new SSEClientTransport(new URL(mcpUrl)));
  }

  const { tools } = await client.listTools();
  const functionDeclarations = tools.map((tool) => ({
    name: tool.name,
    description: tool.description || '',
    parameters: sanitizeSchema(tool.inputSchema) || { type: 'object', properties: {} },
  }));

  console.log('[mcp] connected —', tools.length, 'tool(s):', tools.map((t) => t.name).join(', ') || '(none)');

  return {
    functionDeclarations,
    async callTool(name, args) {
      return client.callTool({ name, arguments: args || {} });
    },
    async close() {
      try { await client.close(); } catch (_) { /* already closed / never fully opened */ }
    },
  };
}
