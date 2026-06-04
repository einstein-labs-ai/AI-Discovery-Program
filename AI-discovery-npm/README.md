# AI Discovery Manager CLI

Codex-style research CLI built on the OpenAI Agents SDK **manager pattern**: one trusted host agent owns the final answer and calls bounded specialist agents exposed as tools. The host process stays the only thing that writes to the local filesystem — it streams the final Markdown to stdout and saves it as an artifact.

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
| `chat` | Interactive REPL — read workspace files and chat about them (see below). |

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
- **Code Interpreter** attaches to the experiment specialist for quantitative analysis, simulations, statistics, and generated tables.

### Sandboxed workspace tools

When workspace filesystem access is enabled (default; disable with `--no-workspace-fs`), specialists and the chat agent get local tools:

- `list_workspace` — list files/subdirectories (entries truncate at 500).
- `read_workspace_file` — read UTF-8 text files (capped at ~256 KiB; binary files refused).
- `write_workspace_file` — write UTF-8 text files (max 1 MiB) — **only** when `--workspace-write` is set (off by default).

All paths are resolved inside the workspace root; `..` and absolute-path escapes are rejected.

### Interactive chat

`ai-discovery chat --workspace <path>` opens a manager-free REPL. Conversation state carries across turns, and the agent can read/list the workspace itself. Slash commands:

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
| `/reset` | Clear conversation history (including loaded files). |
| `/help` | Show chat help. |
| `/exit`, `/quit` | Leave the chat. |

`/read` shares the same sandbox, 256 KiB cap, and binary guard as the agent's read tool.

### Streaming

Streaming is on by default (`--stream`):

- **stdout** — the manager's final Markdown, streamed token-by-token. This is also what gets saved to the artifact.
- **stderr** — `[stream]` / `[specialist:<name>]` progress: output deltas plus `tool_called` / `handoff_*` / `tool_approval_requested` events.

Use `--no-stream` to wait for the complete result before printing.

### Safety posture

- Sensitive trace payloads are disabled everywhere (`traceIncludeSensitiveData: false`).
- Hard citation policy: every external claim needs a real inline citation (author, year, venue, working URL/DOI) from actual search results — never fabricated. Unverifiable claims are dropped or labeled.
- Workspace is sandboxed and writes are off by default.
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
| `OPENAI_MODEL` | Overrides the default model (`gpt-5.5`). |
| `OPENAI_VECTOR_STORE_IDS` | Comma-separated default vector store IDs for File Search. |

## Usage

```powershell
node dist/cli.js run --topic "Robust AI discovery workflows for scientific research" --workspace . --out artifacts
node dist/cli.js literature-review --topic "AI agents for laboratory planning" --vector-store-id vs_...
node dist/cli.js hypothesis --topic "Can retrieval-grounded agent debates improve hypothesis novelty screening?"
node dist/cli.js experiment --topic "Simulation-based hypothesis screening" --experiment-spec "Compare two synthetic baselines and analyze uncertainty"
node dist/cli.js chat --workspace ./papers
```

Run from TypeScript source without building via `npm.cmd run dev -- <command> --topic "..."`.

### Options

| Flag | Description |
| --- | --- |
| `--topic, -t <text>` | Research topic or user request (required except for `chat`). |
| `--workspace, -w <path>` | Workspace root for the sandboxed file tools (default: cwd). |
| `--out, -o <path>` | Output directory for the final Markdown (default: `artifacts`). |
| `--model <model>` | Model for both manager and specialists (default: `OPENAI_MODEL` or `gpt-5.5`). |
| `--manager-model <model>` | Override the manager model only. |
| `--specialist-model <model>` | Override the specialist models only. |
| `--vector-store-id <id>` | Add an OpenAI vector store for File Search; repeatable. |
| `--vector-store-ids <ids>` | Comma-separated OpenAI vector store IDs. |
| `--experiment-spec <text>` | Extra experiment design/analysis requirements. |
| `--max-turns <number>` | Max manager turns (default: 24). |
| `--no-web-search` | Disable web search tools. |
| `--no-workspace-fs` | Disable workspace filesystem tools (read/list/write). |
| `--workspace-write` | Allow specialists/chat to write files into the workspace (off by default). |
| `--stream` / `--no-stream` | Stream live output (default) or wait for the final result. |
| `--dry-run` | Print the resolved workflow as JSON without calling the API. |
| `--help, -h` | Show usage. |

Artifacts are written to `<out>/<command>-<topic-slug>.md`.

### Dry run

For a no-network / no-API configuration check (the only command that runs without `OPENAI_API_KEY`):

```powershell
npm.cmd run dry-run
```

This prints the resolved workflow JSON — command, models, workspace access, web-search state, vector stores, and the hosted/workspace tools each specialist would receive.
