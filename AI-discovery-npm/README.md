# AI Discovery Manager CLI

Codex-style research CLI built on the OpenAI Agents SDK **manager pattern**: one trusted host agent owns the final answer and calls bounded specialist agents exposed as tools. The host process owns local filesystem writes for generated artifacts, chat exports, and optional workspace writes.

## Installation
How to install the AI Discovery Manager CLI
```text
npm i ai-discovery-manager-cli
```


## Features

### Workflow commands

A manager agent orchestrates specialists and produces a Markdown research artifact. Each command targets a different section, except `run`, which drives the full pipeline:

| Command | What it produces |
| --- | --- |
| `run` | Full manager-orchestrated PhD thesis workflow (calls every relevant specialist). |
| `thesis` | A complete PhD thesis draft (title, abstract, intro, lit review, methods, results, discussion, conclusion, references). |
| `literature-review` | A cited PhD-level literature review grouped by theme, method, evidence strength, and open questions. |
| `hypothesis` | A structured YAML research hypothesis covering evidence, mechanism, predictions, test plan, confounders, feasibility, evaluation, uncertainty, and status. |
| `abstract` | A concise thesis abstract covering problem, gap, method, evidence, contribution, and implications. |
| `discussion` | A discussion section with implications, limitations, counterarguments, threats to validity, and future work. |
| `experiment` | A designed-and-run experiment analyzed with Code Interpreter (stats, simulations, generated tables). |
| `conclusion` | A conclusion synthesizing the question, contribution, evidence, limitations, and next research. |
| `chat` | Interactive REPL for workspace Q&A, specialist slash commands, and assistant-output exports. |
| `doctor` | Local readiness checks for API key, workspace, models, vector stores, MCP support, and built dist files. |

### Specialists (manager tools)

The manager calls these bounded specialist agents as tools. Each gets only the hosted tools its contract requests:

| Specialist | Tool name | Hosted tools |
| --- | --- | --- |
| Literature Review | `generate_literature_review` | web search, File Search |
| Hypothesis | `generate_hypothesis` | web search, File Search |
| Abstract | `generate_abstract` | web search, File Search |
| Discussion | `generate_discussion` | web search, File Search |
| Experiment | `run_experiment_and_analysis` | Code Interpreter, File Search, web search |
| Conclusion | `generate_conclusion` | web search, File Search |
| Thesis Writer | `generate_phd_thesis` | web search, File Search |

### Hosted-tool gating

- **Web search** attaches to specialists that request it unless `--no-web-search` is set.
- **OpenAI File Search** attaches only when at least one vector store ID is configured (`--vector-store-id`, `--vector-store-ids`, or `OPENAI_VECTOR_STORE_IDS`).
- **Code Interpreter** attaches to the experiment specialist for quantitative analysis, simulations, statistics, and generated tables. Chat also exposes Code Interpreter so `/experiment` can use the same experiment contract.

### Sandboxed workspace tools

When workspace filesystem access is enabled (default; disable with `--no-workspace-fs`), specialists and the chat agent get local tools:

- `list_workspace` - list files/subdirectories (entries truncate at 500).
- `read_workspace_file` - read UTF-8 text files (capped at ~256 KiB; binary files refused).
- `write_workspace_file` - write UTF-8 text files (max 1 MiB) - **only** when `--workspace-write` is set (off by default).

All tool paths are resolved inside the workspace root; `..` and absolute-path escapes are rejected. Chat `/save` and `/flash-save` are host-side export commands, not model tools, and also constrain output paths inside the workspace.

Model-facing workspace reads also refuse default ignored paths such as `.env*`, `.git`, `node_modules`, lockfiles, and secret/key-like filenames before file content is sent to the model.

### Interactive chat

`ai-discovery chat --workspace <path>` opens a manager-free REPL. Conversation state carries across turns, and the agent can read/list the workspace itself. Specialist slash commands reuse the same shared specialist contracts as the manager CLI, so chat and workflow output stay aligned.

Slash commands:

| Command | Action |
| --- | --- |
| `/read <path>` | Load a workspace text file into the conversation, then ask about it. |
| `/list [<path>]` | List workspace files (default: workspace root). |
| `/save <path.text\|path.txt\|path.pdf>` | Save assistant output history only, excluding user inputs. `/flash-save` is an alias. |
| `/literature-review <topic>` | Generate a literature review using the same specialist contract as the CLI workflow. |
| `/hypothesis <question>` | Generate a structured YAML research hypothesis using the hypothesis schema. |
| `/abstract <topic>` | Generate an abstract using the same specialist contract as the CLI workflow. |
| `/discussion <topic>` | Generate a discussion using the same specialist contract as the CLI workflow. |
| `/experiment <topic/spec>` | Design, run, and analyze an experiment using the same specialist contract as the CLI workflow. |
| `/conclusion <topic>` | Generate a conclusion using the same specialist contract as the CLI workflow. |
| `/model [name\|number]` | Show or switch the chat model (text-only allowlist). |
| `/models` | List the allowed text models. |
| `/safety [1-5]` | Show or set the local safety preflight level for the session. |
| `/mcp <subcommand>` | Manage session-only stdio MCP servers (`connect` / `status` / `tools` / `disconnect` / `help`). |
| `/recursive [on\|off\|status\|<iterations>]` | Toggle bounded self-review/revision for subsequent assistant replies. |
| `/reset` | Clear conversation history, loaded files, and assistant output history used by `/save`. |
| `/help` | Show chat help. |
| `/exit`, `/quit` | Leave the chat. |

`/read` shares the same sandbox, 256 KiB cap, and binary guard as the agent's read tool.

Best-effort keyboard shortcuts on interactive TTYs: **Ctrl+S** saves assistant output history to a default `.ai-discovery/chat-output-<timestamp>.text` path inside the workspace, and **Ctrl+M** shows MCP status/help. Many terminals deliver Ctrl+M as Enter, so `/mcp` remains the reliable command.

### Text-only model allowlist

Every model input (`--model`, `--manager-model`, `--specialist-model`, `OPENAI_MODEL`, and chat `/model`) is validated against a curated text-only allowlist. Display aliases such as `GPT-5.5 Pro` and `GPT 5.4 mini` are normalized; unknown models are rejected with the allowed list. The allowed IDs are:

`gpt-5.5`, `gpt-5.5-pro`, `gpt-5.4`, `gpt-5.4-pro`, `gpt-5.4-mini`, `gpt-5.4-nano`.

### Safety levels and local preflight

A local, on-device safety preflight runs **before** any OpenAI API call (and before `--dry-run` prints), so disallowed prompts fail without leaving the machine. Levels run 1-5 (default `3`, configurable via `--safety-level` or `AI_DISCOVERY_SAFETY_LEVEL`, and per-session via chat `/safety`):

- **Levels 1-5** all block biological/chemical mass-hazard prompts (weaponization and dangerous-pathogen / chemical-agent synthesis).
- **Level 5** additionally blocks jailbreak, prompt-injection, secret-exfiltration, and policy-evasion ("ignore your system/tool policy") attempts.

The preflight is a coarse first gate that complements — not replaces — the agents' built-in "never fabricate / no procedural physical-world harm" instructions.

### Session MCP servers

Chat can attach user-started **stdio MCP servers** ("science MCPs") for the current session only — there is no persisted MCP config file and nothing is autoloaded. Manage them with `/mcp`:

```text
/mcp connect <name> [--cwd <path>] [--env KEY=value | --env KEY]... -- <command> [args...]
/mcp status
/mcp tools [name]
/mcp disconnect <name>
/mcp help
```

`--env KEY` (no value) forwards `KEY` from the current environment. **Env values are never printed or saved — only key names are shown.** MCP tools are exposed to the assistant prefixed by server name so two servers cannot collide.

### Chat Output Saving

Use `/save <path>` or `/flash-save <path>` inside `ai-discovery chat` to export only assistant output from the current chat session. User prompts, slash commands, and loaded file contents are not written to the export. Text exports support `.text` and `.txt`; PDF exports support `.pdf` and are converted from the plain text history.

Examples:

```text
/save notes/session-output.text
/save notes/session-output.pdf
/flash-save latest-output.txt
```

Saved text groups replies as `--- Assistant output N ---`. Export paths must stay inside the configured workspace.

### Streaming

Streaming is on by default (`--stream`):

- **stdout** - the manager's final Markdown, streamed token-by-token. This is also what gets saved to the artifact.
- **stderr** - `[stream]` / `[specialist:<name>]` progress: output deltas plus `tool_called` / `handoff_*` / `tool_approval_requested` events.

Use `--no-stream` to wait for the complete result before printing.

### Safety posture

- Sensitive trace payloads are disabled everywhere (`traceIncludeSensitiveData: false`).
- Hard citation policy: every external claim needs a real inline citation (author, year, venue, working URL/DOI) from actual search results - never fabricated. Unverifiable claims are dropped or labeled.
- Workspace is sandboxed and writes are off by default.
- A local safety preflight (levels 1-5) blocks disallowed prompts before any API call. See "Safety levels and local preflight".
- MCP servers are session-only; env values are never printed or persisted (only key names are shown).
- The CLI never asks for secrets in prompts.

## Setup

Node >= 22 is required.

```powershell
npm.cmd install
npm.cmd run build
```

### Environment variables

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Required for any non-`--dry-run` invocation. |
| `OPENAI_MODEL` | Overrides the default model (`gpt-5.5`); must be in the text-only allowlist. |
| `OPENAI_VECTOR_STORE_IDS` | Comma-separated default vector store IDs for File Search. |
| `AI_DISCOVERY_SAFETY_LEVEL` | Default safety preflight level `1-5` (overridden by `--safety-level`; default `3`). |

## Usage

```powershell
ai-discovery run --topic "Robust AI discovery workflows for scientific research" --workspace . --out artifacts
ai-discovery literature-review --topic "AI agents for laboratory planning" --vector-store-id vs_...
ai-discovery hypothesis --topic "Can retrieval-grounded agent debates improve hypothesis novelty screening?"
ai-discovery experiment --topic "Simulation-based hypothesis screening" --experiment-spec "Compare two synthetic baselines and analyze uncertainty"
ai-discovery run --topic "..." --manager-model "GPT-5.5 Pro" --specialist-model gpt-5.4-mini --safety-level 5
ai-discovery chat --workspace ./papers --safety-level 4
ai-discovery doctor --workspace . --json
ai-discovery run --resume <run-id>
```

Example chat commands:

```text
/literature-review AI agents for laboratory planning
/hypothesis Can retrieval-grounded agent debates improve novelty screening?
/abstract Robust AI discovery workflows for scientific research
/discussion Limitations and counterarguments for AI co-scientist systems
/experiment Compare two synthetic baselines and analyze uncertainty
/conclusion AI agents for scientific discovery
/models
/model gpt-5.5-pro
/safety 5
/mcp connect arxiv --env ARXIV_TOKEN -- npx -y @example/arxiv-mcp-server
/mcp tools arxiv
/recursive on 3
/save outputs/session-output.pdf
```

Run from TypeScript source without building via `npm.cmd run dev -- <command> --topic "..."`.

### Options

| Flag | Description |
| --- | --- |
| `--topic, -t <text>` | Research topic or user request (required except for `chat`). |
| `--workspace, -w <path>` | Workspace root for the sandboxed file tools (default: cwd). |
| `--out, -o <path>` | Output directory for the final Markdown (default: `artifacts`). |
| `--model <model>` | Model for both manager and specialists (default: `OPENAI_MODEL` or `gpt-5.5`). Validated against the text-only allowlist. |
| `--manager-model <model>` | Override the manager model only (same allowlist). |
| `--specialist-model <model>` | Override the specialist models only (same allowlist). |
| `--vector-store-id <id>` | Add an OpenAI vector store for File Search; repeatable. |
| `--vector-store-ids <ids>` | Comma-separated OpenAI vector store IDs. |
| `--experiment-spec <text>` | Extra experiment design/analysis requirements. |
| `--max-turns <number>` | Max manager turns (default: 24). |
| `--safety-level <1-5>` | Local safety preflight level (default: 3, env `AI_DISCOVERY_SAFETY_LEVEL`). |
| `--no-web-search` | Disable web search tools. |
| `--no-workspace-fs` | Disable workspace filesystem tools (read/list/write). |
| `--workspace-write` | Allow specialists/chat to write files into the workspace (off by default). |
| `--stream` / `--no-stream` | Stream live output (default) or wait for the final result. |
| `--dry-run` | Print the resolved workflow as JSON without calling the API. |
| `--json` | Emit machine-readable JSON. Live run/chat output is newline-delimited JSON events. |
| `--resume <id>` | Resume a saved workflow checkpoint from `.ai-discovery/runs/<id>`. |
| `--help, -h` | Show usage. |

Artifacts are written to `<out>/<command>-<topic-slug>.md`.

### Dry run

For a no-network / no-API configuration check (the only command that runs without `OPENAI_API_KEY`):

```powershell
npm.cmd run dry-run
```

This prints the resolved workflow JSON — command, models, the text-only `availableModels` allowlist, `safetyLevel`/`safetyPolicy`, workspace access, web-search state, vector stores, and the hosted/workspace tools each specialist would receive. The `chat` dry-run additionally lists the slash commands, keyboard shortcuts, and `mcpServers` (session-only).

### JSON, doctor, and resumable runs

Use `--json` when another agent or script needs stable machine-readable output:

```powershell
ai-discovery doctor --workspace . --json
ai-discovery run --topic "..." --json
ai-discovery chat --workspace . --json
```

Live workflow and chat commands emit newline-delimited JSON events such as `run_started`, `manager_output_delta`, `artifact_written`, `run_completed`, `chat_started`, `assistant_output_delta`, `assistant_output`, and `error`. Completion events include artifact paths, checkpoint paths, citations found in the final text, SDK token usage when reported, and a cost object with `amount: null` because model-specific pricing is not bundled.

Every non-chat workflow creates a checkpoint under `.ai-discovery/runs/<id>/` with `options.json`, `prompt.md`, `partial.md`, `final.md`, `metadata.json`, and `result.json`. Pressing Ctrl+C during a streamed run saves the current partial before exiting. Resume with:

```powershell
ai-discovery run --resume <id>
```

`doctor` does not call the OpenAI API. It checks local Node version, API-key presence, workspace existence/writability, model allowlist values, vector-store ID shape, MCP stdio availability, and whether the `ai-discovery` command target exists.
