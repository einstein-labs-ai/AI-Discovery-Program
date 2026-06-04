import {
  MCPServerStdio,
  getAllMcpTools,
  type MCPServer,
  type Tool,
} from "@openai/agents";

/**
 * Session-scoped manager for user-started stdio MCP ("science MCP") servers.
 *
 * Design constraints (safety):
 * - Connections live only for the current chat session. There is NO persisted
 *   MCP config file and no autoload — every server must be reconnected each run.
 * - Environment variable VALUES are never printed, logged, or saved. Only the
 *   env KEY names are retained for display so the user can confirm what was
 *   wired without leaking secrets.
 * - MCP tools are exposed to the agent with server-name prefixes (via
 *   `getAllMcpTools(..., { includeServerInToolNames: true })`) so two servers
 *   that both publish e.g. `search` cannot collide.
 */

export interface McpServerRecord {
  name: string;
  command: string;
  args: string[];
  cwd?: string;
  /** Env key names only — values are intentionally not retained for display. */
  envKeys: string[];
  server: MCPServerStdio;
}

export interface McpConnectSpec {
  name: string;
  command: string;
  args: string[];
  cwd?: string;
  /** Resolved env passed to the child process (never displayed). */
  env: Record<string, string>;
  /** Key names only, for status display. */
  envKeys: string[];
}

export class SessionMcpManager {
  private readonly servers = new Map<string, McpServerRecord>();

  has(name: string): boolean {
    return this.servers.has(name);
  }

  list(): McpServerRecord[] {
    return [...this.servers.values()];
  }

  get size(): number {
    return this.servers.size;
  }

  /** Connect a new stdio MCP server and register it for the session. */
  async connect(spec: McpConnectSpec): Promise<McpServerRecord> {
    if (this.servers.has(spec.name)) {
      throw new Error(
        `An MCP server named "${spec.name}" is already connected. Disconnect it first.`,
      );
    }
    const server = new MCPServerStdio({
      name: spec.name,
      command: spec.command,
      args: spec.args,
      cwd: spec.cwd,
      env: Object.keys(spec.env).length > 0 ? spec.env : undefined,
      // Tools rarely change within a session; cache to avoid re-listing on
      // every turn. The cache is dropped when the server disconnects.
      cacheToolsList: true,
    });
    await server.connect();
    const record: McpServerRecord = {
      name: spec.name,
      command: spec.command,
      args: spec.args,
      cwd: spec.cwd,
      envKeys: spec.envKeys,
      server,
    };
    this.servers.set(spec.name, record);
    return record;
  }

  /** Disconnect and forget a single server. */
  async disconnect(name: string): Promise<void> {
    const record = this.servers.get(name);
    if (!record) {
      throw new Error(`No MCP server named "${name}" is connected.`);
    }
    this.servers.delete(name);
    await record.server.close().catch(() => undefined);
  }

  /** List the tools published by one server (or throw if unknown). */
  async toolsFor(name: string): Promise<{ name: string; description?: string }[]> {
    const record = this.servers.get(name);
    if (!record) {
      throw new Error(`No MCP server named "${name}" is connected.`);
    }
    const tools = await record.server.listTools();
    return tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
    }));
  }

  /**
   * Build the agent-facing tool list across all connected servers, with
   * server-name prefixes to prevent collisions. Returns an empty array when no
   * servers are connected.
   */
  async agentTools(): Promise<Tool[]> {
    const mcpServers: MCPServer[] = this.list().map((record) => record.server);
    if (mcpServers.length === 0) {
      return [];
    }
    return getAllMcpTools({
      mcpServers,
      includeServerInToolNames: true,
    });
  }

  /** Close every server (best-effort) and clear the registry. */
  async closeAll(): Promise<void> {
    const records = this.list();
    this.servers.clear();
    await Promise.all(
      records.map((record) => record.server.close().catch(() => undefined)),
    );
  }
}

/** Split a command line into tokens, honoring simple single/double quoting. */
export function tokenizeMcpArgs(input: string): string[] {
  const tokens: string[] = [];
  const re = /"([^"]*)"|'([^']*)'|(\S+)/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(input)) !== null) {
    tokens.push(match[1] ?? match[2] ?? match[3] ?? "");
  }
  return tokens;
}

export interface ParsedMcpConnect {
  name: string;
  command: string;
  args: string[];
  cwd?: string;
  env: Record<string, string>;
  envKeys: string[];
}

/**
 * Parse the tokens that follow `/mcp connect`:
 *   <name> [--cwd <path>] [--env KEY=value | --env KEY]... -- <command> [args...]
 *
 * For `--env KEY` (no `=`), the value is read from the parent process env at
 * parse time. Values are placed into `env` for the child process but only the
 * key names are recorded in `envKeys` for later display.
 */
export function parseMcpConnect(tokens: string[]): ParsedMcpConnect {
  if (tokens.length === 0) {
    throw new Error(
      "Usage: /mcp connect <name> [--cwd <path>] [--env KEY=value | --env KEY]... -- <command> [args...]",
    );
  }
  const [name, ...rest] = tokens;
  if (name.startsWith("--")) {
    throw new Error("MCP server name must come first, before any flags.");
  }

  const env: Record<string, string> = {};
  const envKeys: string[] = [];
  let cwd: string | undefined;
  let index = 0;

  for (; index < rest.length; index += 1) {
    const token = rest[index];
    if (token === "--") {
      index += 1;
      break;
    }
    if (token === "--cwd") {
      const value = rest[index + 1];
      if (value === undefined || value === "--") {
        throw new Error("--cwd requires a path value.");
      }
      cwd = value;
      index += 1;
      continue;
    }
    if (token === "--env") {
      const value = rest[index + 1];
      if (value === undefined || value === "--") {
        throw new Error("--env requires KEY=value or KEY.");
      }
      const eq = value.indexOf("=");
      if (eq >= 0) {
        const key = value.slice(0, eq);
        if (!key) {
          throw new Error(`Invalid --env entry "${value}".`);
        }
        env[key] = value.slice(eq + 1);
        if (!envKeys.includes(key)) envKeys.push(key);
      } else {
        const key = value;
        const fromParent = process.env[key];
        if (fromParent !== undefined) {
          env[key] = fromParent;
        }
        if (!envKeys.includes(key)) envKeys.push(key);
      }
      index += 1;
      continue;
    }
    throw new Error(
      `Unknown /mcp connect flag "${token}". Put the command after a "--" separator.`,
    );
  }

  const command = rest[index];
  if (!command) {
    throw new Error(
      'No MCP command provided. Place it after "--", e.g. /mcp connect docs -- npx -y my-mcp-server.',
    );
  }
  const args = rest.slice(index + 1);
  return { name, command, args, cwd, env, envKeys };
}

export const MCP_HELP = [
  "MCP (session-only stdio servers):",
  "  /mcp connect <name> [--cwd <path>] [--env KEY=value | --env KEY]... -- <command> [args...]",
  "                     Start and attach a stdio MCP server for this session.",
  "                     --env KEY (no value) forwards KEY from the current env.",
  "                     Env values are never printed or saved; only key names are shown.",
  "  /mcp status        List connected servers (command, args, cwd, env key names).",
  "  /mcp tools [name]  List tools for one server, or all connected servers.",
  "  /mcp disconnect <name>",
  "                     Stop and detach a server.",
  "  /mcp help          Show this help.",
  "",
  "MCP tools are exposed to the assistant prefixed by server name to avoid collisions.",
  "Configs are session-only: nothing is persisted to disk.",
].join("\n");

/** Render `/mcp status` text for the current set of connected servers. */
export function formatMcpStatus(records: McpServerRecord[]): string {
  if (records.length === 0) {
    return "No MCP servers connected. Use `/mcp connect ...` (see `/mcp help`).";
  }
  const lines = [`Connected MCP servers (${records.length}):`];
  for (const record of records) {
    const cmd = [record.command, ...record.args].join(" ");
    lines.push(`  - ${record.name}: ${cmd}`);
    if (record.cwd) {
      lines.push(`      cwd: ${record.cwd}`);
    }
    if (record.envKeys.length > 0) {
      lines.push(`      env keys: ${record.envKeys.join(", ")} (values hidden)`);
    }
  }
  return lines.join("\n");
}
