import { tool, type Tool } from "@openai/agents";
import { readdir, readFile, stat, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { z } from "zod";

const MAX_READ_BYTES = 256 * 1024;
const MAX_WRITE_BYTES = 1 * 1024 * 1024;
const MAX_LIST_ENTRIES = 500;

export interface WorkspaceToolOptions {
  workspaceRoot: string;
  allowWrites: boolean;
}

function resolveInside(workspaceRoot: string, relPath: string): string {
  const cleaned = (relPath ?? "").replace(/^[/\\]+/, "");
  const resolved = path.resolve(workspaceRoot, cleaned);
  const rel = path.relative(workspaceRoot, resolved);
  if (rel === "") {
    return resolved;
  }
  if (rel.startsWith("..") || path.isAbsolute(rel)) {
    throw new Error(
      `Path escapes workspace: ${relPath}. Use paths relative to the workspace root.`,
    );
  }
  return resolved;
}

function toPosix(p: string): string {
  return p.split(path.sep).join("/");
}

function looksBinary(buffer: Buffer): boolean {
  const limit = Math.min(buffer.length, 8192);
  for (let i = 0; i < limit; i += 1) {
    if (buffer[i] === 0x00) {
      return true;
    }
  }
  return false;
}

export interface WorkspaceFileRead {
  path: string;
  bytes: number;
  truncated: boolean;
  content: string;
}

/**
 * Read a UTF-8 text file from inside `workspaceRoot`, enforcing the same sandbox
 * (`resolveInside`), size cap (`MAX_READ_BYTES`), and binary guard (`looksBinary`)
 * as the `read_workspace_file` tool. Shared so the interactive `/read` command and
 * the agent tool stay consistent.
 */
export async function readWorkspaceTextFile(
  workspaceRoot: string,
  relPath: string,
): Promise<WorkspaceFileRead> {
  const target = resolveInside(workspaceRoot, relPath);
  const info = await stat(target);
  if (!info.isFile()) {
    throw new Error(`Not a file: ${relPath}`);
  }
  const buffer = await readFile(target);
  if (looksBinary(buffer)) {
    throw new Error(
      `Refusing to read likely-binary file: ${relPath}. Tools only handle UTF-8 text.`,
    );
  }
  const truncated = buffer.length > MAX_READ_BYTES;
  const slice = truncated ? buffer.subarray(0, MAX_READ_BYTES) : buffer;
  return {
    path: toPosix(path.relative(workspaceRoot, target)),
    bytes: info.size,
    truncated,
    content: slice.toString("utf8"),
  };
}

export function createWorkspaceTools(options: WorkspaceToolOptions): Tool[] {
  const { workspaceRoot, allowWrites } = options;

  const listTool = tool({
    name: "list_workspace",
    description:
      "List files and subdirectories in the research workspace. Pass a relative path (default: workspace root). Returns entries with name, type (file|dir), and size in bytes for files.",
    parameters: z.object({
      path: z
        .string()
        .describe(
          "Relative path inside the workspace. Use '' or '.' for the workspace root.",
        )
        .default("."),
    }),
    async execute({ path: relPath }) {
      const target = resolveInside(workspaceRoot, relPath || ".");
      const info = await stat(target);
      if (!info.isDirectory()) {
        throw new Error(`Not a directory: ${relPath}`);
      }
      const entries = await readdir(target, { withFileTypes: true });
      const limited = entries.slice(0, MAX_LIST_ENTRIES);
      const rows = await Promise.all(
        limited.map(async (entry) => {
          const child = path.join(target, entry.name);
          const childInfo = await stat(child).catch(() => undefined);
          const type = entry.isDirectory()
            ? "dir"
            : entry.isFile()
              ? "file"
              : "other";
          return {
            name: entry.name,
            type,
            size: type === "file" && childInfo ? childInfo.size : undefined,
          };
        }),
      );
      return JSON.stringify(
        {
          path: toPosix(path.relative(workspaceRoot, target) || "."),
          truncated: entries.length > MAX_LIST_ENTRIES,
          entries: rows,
        },
        null,
        2,
      );
    },
  });

  const readToolDef = tool({
    name: "read_workspace_file",
    description:
      "Read a UTF-8 text file from the workspace. Returns up to ~256 KiB. Use list_workspace first to discover paths.",
    parameters: z.object({
      path: z.string().describe("Path to the file, relative to the workspace root."),
    }),
    async execute({ path: relPath }) {
      const result = await readWorkspaceTextFile(workspaceRoot, relPath);
      return JSON.stringify(result, null, 2);
    },
  });

  const tools: Tool[] = [listTool, readToolDef];

  if (allowWrites) {
    const writeToolDef = tool({
      name: "write_workspace_file",
      description:
        "Write a UTF-8 text file inside the workspace. Creates parent directories as needed. Overwrites existing files. Max 1 MiB.",
      parameters: z.object({
        path: z
          .string()
          .describe("Destination path, relative to the workspace root."),
        content: z.string().describe("UTF-8 text contents to write."),
      }),
      async execute({ path: relPath, content }) {
        const target = resolveInside(workspaceRoot, relPath);
        const byteLength = Buffer.byteLength(content, "utf8");
        if (byteLength > MAX_WRITE_BYTES) {
          throw new Error(
            `Refusing to write ${byteLength} bytes; limit is ${MAX_WRITE_BYTES}.`,
          );
        }
        await mkdir(path.dirname(target), { recursive: true });
        await writeFile(target, content, "utf8");
        return JSON.stringify({
          path: toPosix(path.relative(workspaceRoot, target)),
          bytesWritten: byteLength,
        });
      },
    });
    tools.push(writeToolDef);
  }

  return tools;
}
