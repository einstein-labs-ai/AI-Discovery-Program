# AI Discovery CLI

`AI Discovery CLI` is an npm/Node.js terminal interface for science workflows inspired by Codex-style slash commands and the local `EinsteinResearch.py` pipeline.

## Run

```powershell
npm start
```

One-shot examples:

```powershell
npm start -- "Can a new catalyst improve CO2 reduction yield?"
npm start -- /research "Can a new catalyst improve CO2 reduction yield?"
npm start -- /hypothesis "Can a new catalyst improve CO2 reduction yield?"
npm start -- /experiment "Design a controlled catalyst screening experiment"
npm start -- /writer "Draft a final-format paper from the current experiment"
npm start -- --safety-level 4 /research "Assess safeguards for a lab-adjacent study"
```

Use `OPENAI_API_KEY` for live OpenAI Responses API generation. Without `OPENAI_API_KEY`, the CLI automatically uses deterministic dry-run output so the workflow still runs locally.

In interactive mode, typing any research question starts the end-to-end workflow:

```text
ai-discovery> Can a new catalyst improve CO2 reduction yield?
```

Focused commands keep the terminal in that mode. After `/hypothesis`, `/experiment`, or `/writer`, type normal follow-up text to expand, change, or rewrite that artifact and save the updated files. Use `/exit` to leave the active mode; use `/quit` to close the CLI.

```text
ai-discovery> /hypothesis Can a new catalyst improve CO2 reduction yield?
ai-discovery:hypothesis> expand the null hypothesis and add measurement risks
ai-discovery:hypothesis> /exit
ai-discovery> /writer Draft the paper
ai-discovery:writer> rewrite the conclusion more carefully
```

The live literature-review stage uses the hosted Responses API web-search tool. Disable that stage's web-search tool with `--no-web-search` or `AI_DISCOVERY_DISABLE_WEB_SEARCH=1` when the prompt contains sensitive material or when hosted search is unavailable.

Use `--safety-level <1-5>`, `AI_DISCOVERY_SAFETY_LEVEL`, or `/safety <1-5>` to set the bio/chemical risk warning profile used by the science agents. Level 1 is lowest risk, and level 5 is highest risk with the strongest warning. The setting does not lower safety boundaries or permit hazardous operational guidance.

## Slash Commands

- `/research <question>` runs the full workflow: plan, literature review, hypothesis, experiment, experiment runner, data analysis, draft paper, technical review, final Markdown paper, and final LaTeX paper.
- Bare text also runs `/research`.
- `/hypothesis [question]` generates a falsifiable hypothesis package, enters hypothesis mode, and keeps follow-up text in that mode until `/exit`. In interactive mode, `/hypothesis` without a question prompts for one.
- `/experiment [brief]` creates an experiment design, saves runnable Node.js experiment code, runs the built-in deterministic synthetic experiment runner, saves the run output, enters experiment mode, and keeps follow-up text in that mode until `/exit`.
- `/writer [instructions]` writes a final-format paper from the current session artifacts, saves the generated Markdown and LaTeX output, enters writer mode, and keeps follow-up text in that mode until `/exit`.
- `/model [name]` shows or changes the model used by all science agents.
- `/safety [1-5]` shows or changes the bio/chemical risk warning level.
- `/status` shows model, mode, output path, artifacts, and token usage.
- `/new` starts a fresh science session.
- `/clear` clears the terminal view and resets visible session artifacts.
- `/history` lists completed commands.
- `/output [directory]` shows or changes the artifact output directory.
- `/exit` leaves the active focused mode, or exits the CLI when no focused mode is active.
- `/quit` exits the CLI.

## Artifacts

Each session writes files under `ai_discovery_runs/run_<timestamp>/`:

Full `/research` workflow:

- `01_research_plan.md`
- `02_literature_review.md`
- `03_hypothesis.md`
- `04_experiment_design.md`
- `05_experiment_code.js`
- `06_experiment_run.json`
- `06_experiment_run.md`
- `07_data_analysis.md`
- `08_draft_paper.md`
- `09_technical_review.md`
- `10_final_paper.md`
- `10_final_paper.tex`
- `session.json`

Single-stage commands also write their focused artifacts:

- `01_hypothesis.md`
- `02_experiment_design.md`
- `02_experiment_code.js`
- `02_experiment_run.json`
- `02_experiment_run.md`
- `03_research_paper.md`
- `03_research_paper.tex`
- `session.json`

Research, hypothesis, and writer outputs are instructed to expose uncertainty, alternatives, provenance, counterarguments, and concise debate. Hypothesis mode also asks for claim-level evidence audits, novelty checks against prior literature, and prospective tests for selected hypotheses.

Generated experiment code is not executed automatically. The `/experiment` command and the full workflow's experiment-runner stage use deterministic synthetic data so the pipeline can proceed safely. Review generated code first, then run it in a controlled environment:

```powershell
node .\ai_discovery_runs\<run>\05_experiment_code.js
```

## Validation

```powershell
npm run check
```

Dry-run smoke check:

```powershell
npm start -- --dry-run "Can a new catalyst improve CO2 reduction yield?"
npm start -- --dry-run --safety-level 5 /hypothesis "Can a catalyst improve CO2 reduction yield?"
npm start -- --dry-run /experiment "Design a controlled catalyst screening experiment"
npm start -- --dry-run /writer "Draft a final-format paper from the current experiment"
```
