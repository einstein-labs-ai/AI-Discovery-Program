# Slash Command Workflows

Codex CLI currently documents slash commands as built-in session controls. This repository maps the requested slash-style workflows to local Codex skills under `.codex/skills/` so they are reusable and discoverable in Codex surfaces that list repository skills.

## Commands

| Slash command | Backing skill | Purpose |
| --- | --- | --- |
| `/automatic` | `$automatic` at `.codex/skills/automatic` | Inspect the repo and suggest safe, testable automation opportunities. |
| `/auto-debug` | `$auto-debug` at `.codex/skills/auto-debug` | Reproduce failures, debug to root cause, fix code, validate, and open a GitHub PR when the worktree is Git-backed and access is available. |
| `/auto-SDLC` | `$auto-sdlc` at `.codex/skills/auto-sdlc` | Use `AGENTS.override.md` to gather architecture and requirements, then produce an SDLC delivery plan with validation, release, rollback, and verification steps. |
| `/read` | Workspace tools (`list_workspace` / `read_workspace_file` in `src/workspaceTools.ts`) | Read files from the sandboxed workspace and chat about their contents. Paths are resolved inside the workspace root, reads cap at 256 KiB, and binary files are refused. |

## Continuation Contract

Substantive backing skill or workspace-tool outputs must be returned into the active LLM conversation context so the user can ask follow-up questions about the result. Control or export commands such as `/exit`, `/quit`, `/reset`, `/save`, `/flash-save`, and `/help` are host-side exceptions and do not need to be added as conversational content.

## Usage

If the Codex surface does not support custom slash commands directly, invoke the backing skill explicitly:

- `Use $automatic for this repo.`
- `Use $auto-debug to debug and fix this failure.`
- `Use $auto-sdlc for this software idea.`

These workflows deliberately avoid automatic privileged actions unless access is available and the user request authorizes the action.
