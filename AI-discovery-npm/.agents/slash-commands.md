# Slash Command Workflows

Codex CLI currently documents slash commands as built-in session controls. This repository maps the requested slash-style workflows to local Codex skills under `.codex/skills/` so they are reusable and discoverable in Codex surfaces that list repository skills.

## Commands

| Slash command | Backing skill | Purpose |
| --- | --- | --- |
| `/automatic` | `$automatic` at `.codex/skills/automatic` | Inspect the repo and suggest safe, testable automation opportunities. |
| `/auto-debug` | `$auto-debug` at `.codex/skills/auto-debug` | Reproduce failures, debug to root cause, fix code, validate, and open a GitHub PR when the worktree is Git-backed and access is available. |
| `/auto-SDLC` | `$auto-sdlc` at `.codex/skills/auto-sdlc` | Use `AGENTS.override.md` to gather architecture and requirements, then produce an SDLC delivery plan with validation, release, rollback, and verification steps. |
| `/read` | Workspace tools (`list_workspace` / `read_workspace_file` in `src/workspaceTools.ts`) | Read files from the sandboxed workspace and chat about their contents. Paths are resolved inside the workspace root, reads cap at 256 KiB, and binary files are refused. |

## Chat runtime commands (`src/chat.ts`)

These run inside `ai-discovery chat` and are distinct from the Codex repo-workflow skills above:

| Slash command | Backing module | Purpose |
| --- | --- | --- |
| `/model [name\|number]`, `/models` | `src/models.ts` | Show/switch the chat model against the text-only allowlist (`gpt-5.5`, `gpt-5.5-pro`, `gpt-5.4`, `gpt-5.4-pro`, `gpt-5.4-mini`, `gpt-5.4-nano`); display aliases are normalized and unknown models rejected. |
| `/safety [1-5]` | `src/safety.ts` | Show/set the local safety preflight level. All levels block bio/chemical mass-hazard prompts; level 5 also blocks jailbreak/prompt-injection/secret-exfiltration/policy-evasion. The preflight runs before any API call. |
| `/mcp connect\|status\|tools\|disconnect\|help` | `src/mcpManager.ts` | Manage session-only stdio MCP servers. Configs are never persisted; env values are never printed or saved (only key names). Tools are exposed prefixed by server name. |

Best-effort TTY shortcuts: **Ctrl+S** saves assistant output history to a default workspace path; **Ctrl+M** shows MCP status/help (terminal-dependent — `/mcp` is the reliable equivalent).

## Continuation Contract

Substantive backing skill or workspace-tool outputs must be returned into the active LLM conversation context so the user can ask follow-up questions about the result. Control, configuration, or export commands such as `/exit`, `/quit`, `/reset`, `/save`, `/flash-save`, `/help`, `/model`, `/models`, `/safety`, and `/mcp` are host-side exceptions and do not need to be added as conversational content.

## Usage

If the Codex surface does not support custom slash commands directly, invoke the backing skill explicitly:

- `Use $automatic for this repo.`
- `Use $auto-debug to debug and fix this failure.`
- `Use $auto-sdlc for this software idea.`

These workflows deliberately avoid automatic privileged actions unless access is available and the user request authorizes the action.
