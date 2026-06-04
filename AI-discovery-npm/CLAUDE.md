# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Node >= 22 is required. The repo is Windows-first; examples use `npm.cmd` and PowerShell, but the same scripts work under bash.

- `npm.cmd install` — install deps (`@openai/agents`, `zod`, `tsx`, `typescript`).
- `npm.cmd run build` — `tsc -p tsconfig.json` → emits `dist/` from `src/`.
- `npm.cmd run dev` — run the TS source directly via `tsx src/cli.ts` (no build step).
- `npm.cmd run dry-run` — `tsx src/cli.ts run --dry-run --topic "..."`. Prints the resolved workflow JSON without calling the OpenAI API; the only command that runs without `OPENAI_API_KEY`.
- `node dist/cli.js <command> --topic "..."` — run a built CLI. `command` ∈ `run | thesis | literature-review | abstract | discussion | experiment | conclusion`. See `src/cli.ts` `usage()` for the full flag list.
- `node dist/cli.js chat --workspace <path>` — interactive REPL (no `--topic` needed). Read workspace text files with `/read <path>` and chat about them; other slash commands: `/list`, `/reset`, `/help`, `/exit`. Implemented in `src/chat.ts`.
- `node --check ai-discovery.js` — syntax-check the standalone JS entry point (see "Two entry points" below).

There are no tests, linter, or formatter configured. Validation = `npm.cmd run build` (type check) + `npm.cmd run dry-run`.

## Environment

- `OPENAI_API_KEY` — required for any non-`--dry-run` invocation.
- `OPENAI_MODEL` — overrides the default model (`gpt-5.5`). `--model` / `--manager-model` / `--specialist-model` override per call.
- `OPENAI_VECTOR_STORE_IDS` — comma-separated default for `--vector-store-id`. File Search is only enabled when at least one ID is configured.

## Architecture

This is a single-process CLI that orchestrates OpenAI Agents using the **manager pattern**: one trusted host Agent owns the final answer and calls bounded specialist Agents exposed as tools via `agent.asTool(...)`. The host process is the only thing that writes to the local filesystem (the final Markdown artifact).

### Two entry points (intentional)

- **`src/cli.ts` → `dist/cli.js`** is the maintained CLI (declared as the `ai-discovery` bin in `package.json`). All new work goes here.
- **`ai-discovery.js`** at the repo root is a self-contained legacy JS implementation that predates the TS split. It duplicates the workspace-tools logic now in `src/workspaceTools.ts`. Don't edit it as a side-effect of changing `src/`; treat it as a separate artifact unless the user explicitly asks to update or remove it.

### Specialist contracts (`src/cli.ts`)

Specialists are declared as a single `specialistContracts` array in `src/specialistContracts.ts`. Each entry binds together: the agent display name, the tool name the manager calls it by, the human description, the instruction block, and the set of hosted tools it gets (`web`, `file`, `code`). The manager CLI and chat slash commands both import this shared contract list, so update it there instead of copying prompt text into `src/cli.ts` or `src/chat.ts`.

The same `rewriteContextOnlyLine()` helper mutates the boilerplate "workspace path is context only" line into a "use the workspace tools" line when `--no-workspace-fs` is not set. This is the only path-aware string transform on instructions; preserve it if you reword that line.

### Hosted tool gating

`createHostedTools()` is the single chokepoint that decides which OpenAI hosted tools attach to a specialist:

- `webSearchTool()` only if the contract requests `web` **and** `--no-web-search` is not set.
- `fileSearchTool(vectorStoreIds, …)` only if the contract requests `file` **and** at least one vector store ID is configured. Specialists are explicitly instructed to use File Search "when configured" — keep that conditional language so the prompt doesn't lie when no store is attached.
- `codeInterpreterTool()` whenever requested (currently the experiment specialist in manager workflows; chat also exposes it so `/experiment` can follow that same contract).

### Workspace tools (`src/workspaceTools.ts`)

`createWorkspaceTools()` returns local `list_workspace` / `read_workspace_file` / (optionally) `write_workspace_file` tools that the agent can call. Critical invariants:

- All paths go through `resolveInside()`, which rejects anything resolving outside `workspaceRoot` (escape attempts via `..` or absolute paths throw).
- Reads cap at 256 KiB and refuse files that look binary (`looksBinary` checks for NUL bytes in the first 8 KiB). Writes cap at 1 MiB and only exist when `allowWrites` is true (CLI flag `--workspace-write`, off by default).
- `MAX_LIST_ENTRIES = 500` — directory listings truncate with a `truncated: true` flag.

If you add a new workspace tool, route paths through `resolveInside` and surface limits in the tool description so the model knows the cap.

### Interactive chat (`src/chat.ts`)

`/save <path.text|path.txt|path.pdf>` and `/flash-save` save only assistant output history from the current chat session, excluding user inputs. `/literature-review`, `/hypothesis`, `/abstract`, `/discussion`, `/experiment`, and `/conclusion` are handled inline and prompt the chat agent with the shared specialist contract instructions.

Substantive slash-command results must remain available to the LLM for follow-up chat. Generated specialist output persists through `result.history`, `/read` inserts loaded file contents as a user context item, and any future substantive slash command should preserve its result in the active thread before returning the next prompt.

`runChat()` is a separate, manager-free path: `cli.ts` dispatches to it in `main()` when `command === "chat"` (which is why `chat` is added to `CLI_COMMANDS` and exempted from the topic requirement in `parseArgs`). It builds one chat Agent with web/file-search hosted tools plus the same `createWorkspaceTools()` and loops over stdin using **readline's async iterator** (`for await (const line of rl)`) — not sequential `rl.question()`, which silently drops buffered lines on piped input. Slash commands (`/read`, `/list`, `/reset`, `/help`, `/exit`) are handled inline; everything else is a chat turn. Conversation state is carried across turns via `result.history`. `/read` reads through the shared `readWorkspaceTextFile()` helper in `workspaceTools.ts`, so it honors the same `resolveInside` sandbox, 256 KiB cap, and binary guard as the agent's `read_workspace_file` tool — keep that helper as the single read path if you touch either.

### Streaming model

When `--stream` is on (the default), the run is split across two FDs:

- **stdout** = the manager's final Markdown, streamed token-by-token via `result.toTextStream(...)`. This is also what gets written to `artifacts/<command>-<slug>.md`.
- **stderr** = a `[stream]` / `[specialist:<name>]` log driven by `createStreamReporter()`. It taps each specialist's `onStream` callback and prints `output_text_delta`s plus `tool_called` / `handoff_*` / `tool_approval_requested` events.

`runManagerWithStreaming()` reads the manager's text stream itself (already printed live), then sets `stdoutAlreadyPrinted` so `main()` doesn't double-write the final output. If you change streaming, preserve this flag — otherwise the artifact will be printed twice on stdout.

### Safety posture (do not silently weaken)

- `Runner` and every specialist's `runConfig` set `traceIncludeSensitiveData: false`. Don't flip this to true without an explicit user request.
- The manager and specialist instructions encode a **hard citation policy** (real URLs/DOIs from search results, no fabricated references, unverifiable claims must be dropped or labeled). When editing instructions, keep the "never fabricate" phrasing — the workflow's value depends on it.
- The workspace is sandboxed *and* writes are off by default. If you broaden access (e.g. removing `resolveInside`, raising size caps, defaulting `--workspace-write` to on), call it out to the user.

## Project conventions

- ESM only (`"type": "module"` in `package.json`, `module: NodeNext` in `tsconfig.json`). TS imports must use the `.js` extension for relative paths (`./workspaceTools.js`), even though the source file is `.ts`.
- `artifacts/`, `dist/`, `node_modules/`, and `.env*` (except `.env.example`) are gitignored. Don't commit anything written into `artifacts/` from a CLI run.
- `AGENTS.override.md` describes the autonomous-delivery workflow this repo follows when invoked through Codex/Cursor (debug loop, validation expectations, citation discipline). It is policy for *how to work on this repo*, not runtime code — useful context when picking how much SDLC ceremony to apply to a change.
- `.codex/skills/` and `.agents/skills/` map slash-style workflows (`/auto-debug`, `/auto-SDLC`, `/automatic`) for Codex CLI surfaces; they're documented in `.agents/slash-commands.md`. They don't affect Claude Code directly but explain the names you'll see referenced in user messages.
