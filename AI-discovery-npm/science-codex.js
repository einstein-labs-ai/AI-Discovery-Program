#!/usr/bin/env node
"use strict";

const fs = require("node:fs/promises");
const path = require("node:path");
const process = require("node:process");
const readline = require("node:readline/promises");
const { stdin: input, stdout: output } = require("node:process");

const APP_NAME = "AI Discovery CLI";
const APP_VERSION = "0.1.0";
const DEFAULT_MODEL = "gpt-5.5";
const DEFAULT_OUTPUT_DIR = "ai_discovery_runs";
const DEFAULT_TIMEOUT_MS = 120000;
const DEFAULT_MAX_OUTPUT_TOKENS = 6000;
const DEFAULT_SAFETY_LEVEL = 3;
const WEB_SEARCH_TOOL = { type: "web_search" };
const FOCUSED_MODE_COMMANDS = new Set(["/hypothesis", "/experiment", "/writer"]);

const SAFETY_LEVELS = new Map([
  [
    1,
    {
      label: "lowest bio/chemical risk",
      warning: "Standard scientific caution. Keep provenance clear and avoid unsupported operational claims.",
      prompt:
        "Treat the topic as low biological or chemical risk. Use normal scientific quality controls, cite provenance, and avoid speculative procedural detail.",
    },
  ],
  [
    2,
    {
      label: "low bio/chemical risk",
      warning: "Add basic misuse checks and source-quality cautions for biological or chemical claims.",
      prompt:
        "Treat the topic as low biological or chemical risk. Include basic misuse checks, source-quality notes, and safe alternatives when details could become operational.",
    },
  ],
  [
    3,
    {
      label: "moderate bio/chemical risk",
      warning: "Use non-operational framing for lab-relevant details and call out review requirements.",
      prompt:
        "Treat the topic as moderate biological or chemical risk. Keep lab-relevant details non-operational unless they are benign, add review requirements, and separate safe conceptual discussion from executable protocols.",
    },
  ],
  [
    4,
    {
      label: "high bio/chemical risk",
      warning: "Strong warning. Avoid actionable synthesis, culturing, optimization, dosing, scale-up, or release guidance.",
      prompt:
        "Treat the topic as high biological or chemical risk. Provide a strong warning, avoid actionable synthesis, culturing, optimization, dosing, scale-up, release, evasion, or harm-enabling instructions, and redirect to safety, ethics, compliance, and high-level risk analysis.",
    },
  ],
  [
    5,
    {
      label: "highest bio/chemical risk",
      warning: "Highest warning. Do not lower safeguards; keep content non-operational and compliance-focused.",
      prompt:
        "Treat the topic as highest biological or chemical risk. Give a prominent warning. Do not lower safeguards or provide hazardous operational steps. Keep the response non-operational, compliance-focused, and oriented around risk assessment, oversight, safer substitutes, incident prevention, and legitimate review channels.",
    },
  ],
]);

const SCIENTIFIC_TRANSPARENCY_REQUIREMENTS = [
  "Scientific transparency requirements for /research, /hypothesis, and /writer:",
  "- Expose uncertainty explicitly by labeling confidence, missing information, assumptions, and unknowns.",
  "- Present plausible alternatives, counterarguments, and concise debate between competing interpretations when useful.",
  "- State provenance for evidence, data, assumptions, retrieved sources, generated code, synthetic data, and model-produced content.",
  "- Distinguish direct evidence, inference, speculation, and planned validation.",
  "- Identify what observations, sources, or tests would resolve the debate.",
  "- Do not reveal private chain-of-thought; provide concise public reasoning and audit notes only.",
].join("\n");

const HYPOTHESIS_EVIDENCE_AUDIT_REQUIREMENTS = [
  "Hypothesis-mode evidence requirements:",
  "- Include claim-level evidence audits for selected hypotheses: claim, evidence/provenance, support strength, uncertainty, and needed validation.",
  "- Include novelty checks against prior literature when sources are available; if search is unavailable, label novelty as unverified.",
  "- Include prospective tests for selected hypotheses, with falsifying observations, measurements, controls, and next evidence to collect.",
].join("\n");

const COMMANDS = new Map([
  [
    "/help",
    {
      purpose: "Show available slash commands.",
      usage: "/help",
    },
  ],
  [
    "/status",
    {
      purpose: "Show session configuration, active model, output directory, and usage.",
      usage: "/status",
    },
  ],
  [
    "/verbose",
    {
      purpose: "Toggle live AI response output while API prompts are streaming.",
      usage: "/verbose [on|off]",
    },
  ],
  [
    "/model",
    {
      purpose: "Show or change the active model for all science agents.",
      usage: "/model [model-name]",
    },
  ],
  [
    "/safety",
    {
      purpose: "Show or change the bio/chemical risk safety level used by science agents.",
      usage: "/safety [1-5]",
    },
  ],
  [
    "/new",
    {
      purpose: "Start a fresh science session in the same CLI.",
      usage: "/new",
    },
  ],
  [
    "/clear",
    {
      purpose: "Clear the terminal view and start a fresh visible context.",
      usage: "/clear",
    },
  ],
  [
    "/history",
    {
      purpose: "List completed commands in this session.",
      usage: "/history",
    },
  ],
  [
    "/output",
    {
      purpose: "Show or change the artifact output directory.",
      usage: "/output [directory]",
    },
  ],
  [
    "/research",
    {
      purpose: "Run the full research workflow from a research question.",
      usage: "/research <research question>",
    },
  ],
  [
    "/hypothesis",
    {
      purpose: "Generate falsifiable scientific hypotheses from a research question and enter hypothesis mode.",
      usage: "/hypothesis [research question]",
    },
  ],
  [
    "/experiment",
    {
      purpose: "Create an experiment design, write runnable Node.js experiment code, and enter experiment mode.",
      usage: "/experiment [research question or experiment brief]",
    },
  ],
  [
    "/writer",
    {
      purpose: "Write a research paper from the current artifacts and enter writer mode.",
      usage: "/writer [paper title or extra instructions]",
    },
  ],
  [
    "/quit",
    {
      purpose: "Exit the CLI.",
      usage: "/quit",
    },
  ],
  [
    "/exit",
    {
      purpose: "Leave the active focused mode, or exit the CLI when no mode is active.",
      usage: "/exit",
    },
  ],
]);

const COMMAND_ALIASES = new Map([
  ["help", "/help"],
  ["status", "/status"],
  ["verbose", "/verbose"],
  ["model", "/model"],
  ["safety", "/safety"],
  ["new", "/new"],
  ["clear", "/clear"],
  ["history", "/history"],
  ["output", "/output"],
  ["research", "/research"],
  ["run", "/research"],
  ["start", "/research"],
  ["hypothesis", "/hypothesis"],
  ["experiment", "/experiment"],
  ["writer", "/writer"],
  ["quit", "/quit"],
  ["exit", "/exit"],
]);

const SCIENCE_SYSTEM_PROMPT = [
"You are an AI discovery agent for science using an EinsteinResearch.py-style pipeline.",
"Operate as a rigorous scientific collaborator: define research plans, search-informed literature reviews, testable hypotheses, experiment designs, reproducible analysis scaffolds, technical reviews, and final research papers.",
"Never invent citations, measurements, datasets, or executed results. Label assumptions, synthetic data, and limitations.",
"Treat generated code as untrusted until reviewed and run by the user. Do not claim it was executed unless execution evidence is provided.",
SCIENTIFIC_TRANSPARENCY_REQUIREMENTS,
"Use concise Markdown with the exact requested headings.",
].join("\n");
const PLAN_PROMPT = [
"Create an end-to-end research execution plan for the research question below.",
"",
"Return exactly these Markdown top-level headings:",
"## Problem Statement",
"## Research Objective",
"## Scope and Non-Goals",
"## Success Criteria",
"## Research Workflow",
"## Risks and Constraints",
"",
"Requirements:",
"- Make the plan specific enough to drive literature review, hypothesis generation, experiment design, experiment running, analysis, paper drafting, technical review, and final LaTeX paper output.",
"- Identify actors, data assumptions, operational constraints, and validation points.",
"- In the risks and constraints section, expose uncertainty, alternatives, provenance gaps, counterarguments, and debate points that the workflow should resolve.",
"- Do not claim any experiment or literature search has already happened.",
].join("\n");
const LITERATURE_PROMPT = [
"Act as the search agent for a literature review.",
"",
"Return exactly these Markdown top-level headings:",
"## Search Strategy",
"## Literature Review",
"## Evidence Table",
"## Gaps and Assumptions",
"## References",
"",
"Requirements:",
"- Search for reliable literature when a web-search tool is available.",
"- Prefer peer-reviewed papers, authoritative technical reports, datasets, and official documentation.",
"- Include citations or URLs only for sources actually retrieved or supplied.",
"- If search is unavailable or insufficient, clearly mark literature claims as assumption-based and do not fabricate references.",
"- Keep the review focused on what the later hypothesis and experiment can test.",
].join("\n");

const HYPOTHESIS_PROMPT = [
  "Given a user's research question, interpret and restate the question to ensure accurate understanding.",
  "Then formulate clear, specific, and testable null and alternative hypotheses related to the question.",
  "",
  "If the research question is missing or too ambiguous to make testable hypotheses, request clarification and do not invent details.",
  "",
  "Output format:",
  "- Begin with one paragraph of 2 to 5 sentences that restates the interpreted question and provides a concise but rigorous theoretical and empirical rationale for the hypotheses.",
  "- Ground the rationale in 1 to 2 relevant citations using APA or [Author, Year] style; use placeholders such as [Author, Year] only when sources are not supplied or retrieved.",
  "- Discuss key frameworks, prior findings, or major debates as appropriate, and explain why each hypothesis is warranted while addressing plausible alternative views.",
  "- After the rationale paragraph, use bullet points only.",
  "- Bullet format must be exactly:",
  "  - Null Hypothesis (H0): [precise, testable statement]",
  "  - Alternative Hypothesis (H1): [directional, testable statement when warranted]",
  "  - Alternative Hypothesis (H2): [additional plausible directional, testable statement when warranted]",
  "- If multiple plausible alternatives exist, state each separately and tie it to the rationale.",
  "- Do not add Markdown headings, summaries, recommendations, evidence tables, or commentary outside the rationale paragraph and hypothesis bullets.",
  "- Write at a postdoctoral level: concise, information-rich, critically engaged, rigorous, and free from filler or repetition.",
  "",
  "Quality requirements:",
  "- Each hypothesis must be precise, testable, and explicitly justified by the prior literature and reasoning in the rationale.",
  "- Use measurable constructs, a population/system, comparator, outcome, expected direction when warranted, and an estimable test wherever the question allows.",
  "- Preserve uncertainty, provenance, counterarguments, and concise debate inside the rationale rather than as separate sections.",
  "- Do not fabricate sources. If no retrieved sources are supplied, label literature claims as assumption-based or lower-confidence.",
  "- Evidence audit, novelty, and prospective-test requirements should inform the rationale and hypothesis wording without adding extra sections.",
  "",
  HYPOTHESIS_EVIDENCE_AUDIT_REQUIREMENTS,
].join("\n");

const EXPERIMENT_PROMPT = [
  "Create a concrete experiment design from the current research question and hypothesis.",
  "",
  "Return Markdown with these exact headings in order:",
  "## Experimental Design",
  "## Procedure",
  "## Controls",
  "## Materials",
  "## Sample Size and Power",
  "## Randomization and Blinding",
  "## Metrics",
  "## Data Collection",
  "## Ethical and Practical Considerations",
  "## Generated Experiment Code",
  "",
  "Requirements:",
  "- Ground the design in the provided question and hypothesis.",
  "- Label assumptions and missing information.",
  "- Include bounded failure handling and reproducibility notes.",
  "- Include runnable Node.js code in one fenced javascript block.",
  "- The code may simulate data if real data is unavailable, but must clearly label synthetic output.",
].join("\n");

const ANALYSIS_PROMPT = [
  "Analyze the experiment run output for the specified research question.",
  "",
  "Return Markdown with these exact headings in order:",
  "## Data",
  "## Results",
  "## Statistical Interpretation",
  "## Hypothesis Assessment",
  "## Limitations",
  "## Reproducibility Notes",
  "",
  "Requirements:",
  "- Use only the provided experiment run output and clearly label synthetic or simulated data.",
  "- Do not claim empirical findings when the run output is synthetic, planned, or unexecuted.",
  "- Connect the results back to the hypothesis, metrics, and experiment design.",
].join("\n");

const DRAFT_PAPER_PROMPT = [
  "Generate a draft research paper from the provided workflow artifacts.",
  "",
  "Return Markdown with these exact headings in order:",
  "# Title",
  "## Abstract",
  "## Introduction",
  "## Hypothesis",
  "## Experiment",
  "## Data",
  "## Results",
  "## Conclusion",
  "## References",
  "",
  "Requirements:",
  "- Use the requested final paper format exactly: abstract, introduction, hypothesis, experiment, data, results, conclusion, references.",
  "- Distinguish executed, synthetic, planned, and assumption-based material.",
  "- Expose uncertainty, alternatives, provenance, and counterarguments in the existing sections without inventing evidence.",
  "- Use only supplied or retrieved references. Do not invent citations.",
].join("\n");

const TECHNICAL_REVIEW_PROMPT = [
  "Perform a technical review of the draft research paper.",
  "",
  "Return Markdown with these exact headings in order:",
  "## Summary",
  "## Correctness Review",
  "## Methodology Review",
  "## Citation and Evidence Review",
  "## Security and Reproducibility Review",
  "## Required Revisions",
  "",
  "Requirements:",
  "- Check whether the draft supports the hypothesis, handles synthetic data honestly, and keeps references traceable.",
  "- Flag unsupported claims, missing controls, weak analysis, and reproducibility gaps.",
  "- Do not rewrite the whole paper in this review stage.",
].join("\n");

const FINAL_PAPER_PROMPT = [
  "Revise the draft into the final research paper using the technical review.",
  "",
  "Return Markdown with these exact headings in order:",
  "# Title",
  "## Abstract",
  "## Introduction",
  "## Hypothesis",
  "## Experiment",
  "## Data",
  "## Results",
  "## Conclusion",
  "## References",
  "",
  "Requirements:",
  "- Preserve the requested final paper format exactly.",
  "- Apply the technical review's required revisions when they are supported by the supplied artifacts.",
  "- Do not invent citations, data, experiments, or measurements.",
  "- Clearly label synthetic experiment-run output and assumption-based conclusions.",
  "- Preserve uncertainty, alternatives, provenance notes, and counterarguments in the existing paper sections.",
].join("\n");

const WRITER_PROMPT = [
  "Write a final-format research paper from the provided science workflow artifacts.",
  "",
  "Return Markdown with these exact headings in order:",
  "# Title",
  "## Abstract",
  "## Introduction",
  "## Hypothesis",
  "## Experiment",
  "## Data",
  "## Results",
  "## Conclusion",
  "## References",
  "",
  "Requirements:",
  "- Integrate the research question, hypothesis, experiment design, experiment run output, and analysis when available.",
  "- Do not claim generated experiment code was executed unless execution output is supplied.",
  "- Do not invent citations. Use a References section that includes only supplied or retrieved sources.",
  "- Clearly label synthetic data, planned analysis, and non-empirical material.",
  "- Expose uncertainty, alternatives, provenance, counterarguments, and concise debate in the existing final-paper sections.",
].join("\n");

function normalizeCommand(command) {
  if (!command) {
    return "";
  }
  const trimmed = String(command).trim();
  if (!trimmed) {
    return "";
  }
  if (trimmed.startsWith("/")) {
    return trimmed.toLowerCase();
  }
  return COMMAND_ALIASES.get(trimmed.toLowerCase()) || `/${trimmed.toLowerCase()}`;
}

function parseCommandLine(line) {
  const raw = String(line || "").trim();
  if (!raw) {
    return { command: "", args: "" };
  }
  const firstSpace = raw.search(/\s/);
  if (firstSpace === -1) {
    if (raw.startsWith("/")) {
      return { command: normalizeCommand(raw), args: "" };
    }
    const aliasCommand = COMMAND_ALIASES.get(raw.toLowerCase());
    if (aliasCommand) {
      return { command: aliasCommand, args: "" };
    }
    return { command: "/research", args: raw };
  }
  const head = raw.slice(0, firstSpace);
  const tail = raw.slice(firstSpace + 1).trim();
  if (!head.startsWith("/")) {
    const aliasCommand = COMMAND_ALIASES.get(head.toLowerCase());
    if (aliasCommand) {
      return { command: aliasCommand, args: tail };
    }
    return { command: "/research", args: raw };
  }
  return {
    command: normalizeCommand(head),
    args: tail,
  };
}

function normalizeSafetyLevel(value) {
  const parsed = Number.parseInt(String(value ?? "").trim(), 10);
  if (!Number.isFinite(parsed) || parsed < 1 || parsed > 5) {
    return DEFAULT_SAFETY_LEVEL;
  }
  return parsed;
}

function parseSafetyLevelInput(value) {
  const raw = String(value ?? "").trim().toLowerCase();
  const match = /^(?:level\s*)?([1-5])$/.exec(raw);
  if (!match) {
    return null;
  }
  return Number(match[1]);
}

function safetyProfile(level) {
  return SAFETY_LEVELS.get(normalizeSafetyLevel(level)) || SAFETY_LEVELS.get(DEFAULT_SAFETY_LEVEL);
}

function formatSafetyProfile(level) {
  const normalized = normalizeSafetyLevel(level);
  const profile = safetyProfile(normalized);
  return [
    `Bio/chemical safety level: ${normalized} (${profile.label}).`,
    `Warning: ${profile.warning}`,
    "This setting calibrates warning and review posture only. It does not bypass scientific, legal, or safety boundaries.",
  ].join("\n");
}

function buildSystemPrompt(safetyLevel) {
  const profile = safetyProfile(safetyLevel);
  return [
    SCIENCE_SYSTEM_PROMPT,
    "",
    "Bio/chemical safety profile:",
    profile.prompt,
    "Never reduce safeguards for hazardous biological or chemical content. Prefer safe, high-level, non-operational risk analysis when details could enable harm.",
  ].join("\n");
}

function appendRuntimeGovernance(userPrompt, safetyLevel) {
  return [
    String(userPrompt || "").trim(),
    "",
    SCIENTIFIC_TRANSPARENCY_REQUIREMENTS,
    "",
    formatSafetyProfile(safetyLevel),
  ].join("\n");
}

function parseGlobalArgs(argv) {
  const options = {
    model: process.env.AI_DISCOVERY_MODEL || DEFAULT_MODEL,
    outputDir: process.env.AI_DISCOVERY_OUTPUT_DIR || DEFAULT_OUTPUT_DIR,
    dryRun: process.env.SCIENCE_CODEX_DRY_RUN === "1",
    webSearch: process.env.AI_DISCOVERY_DISABLE_WEB_SEARCH !== "1",
    verbose: process.env.SCIENCE_CODEX_VERBOSE === "1" || process.env.AI_DISCOVERY_VERBOSE === "1",
    safetyLevel: normalizeSafetyLevel(process.env.AI_DISCOVERY_SAFETY_LEVEL || DEFAULT_SAFETY_LEVEL),
    maxOutputTokens: Number(process.env.AI_DISCOVERY_MAX_OUTPUT_TOKENS || DEFAULT_MAX_OUTPUT_TOKENS),
    commandLine: "",
    help: false,
  };
  const commandParts = [];

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") {
      options.help = true;
      continue;
    }
    if (arg === "--dry-run") {
      options.dryRun = true;
      continue;
    }
    if (arg === "--no-web-search") {
      options.webSearch = false;
      continue;
    }
    if (arg === "--verbose") {
      options.verbose = true;
      continue;
    }
    if (arg === "--no-verbose") {
      options.verbose = false;
      continue;
    }
    if (arg === "--model") {
      i += 1;
      options.model = argv[i] || options.model;
      continue;
    }
    if (arg.startsWith("--model=")) {
      options.model = arg.slice("--model=".length) || options.model;
      continue;
    }
    if (arg === "--safety-level" || arg === "--safety") {
      i += 1;
      options.safetyLevel = normalizeSafetyLevel(argv[i] || options.safetyLevel);
      continue;
    }
    if (arg.startsWith("--safety-level=")) {
      options.safetyLevel = normalizeSafetyLevel(arg.slice("--safety-level=".length) || options.safetyLevel);
      continue;
    }
    if (arg.startsWith("--safety=")) {
      options.safetyLevel = normalizeSafetyLevel(arg.slice("--safety=".length) || options.safetyLevel);
      continue;
    }
    if (arg === "--output") {
      i += 1;
      options.outputDir = argv[i] || options.outputDir;
      continue;
    }
    if (arg.startsWith("--output=")) {
      options.outputDir = arg.slice("--output=".length) || options.outputDir;
      continue;
    }
    if (arg === "--max-output-tokens") {
      i += 1;
      options.maxOutputTokens = Number(argv[i] || options.maxOutputTokens);
      continue;
    }
    if (arg.startsWith("--max-output-tokens=")) {
      options.maxOutputTokens = Number(arg.slice("--max-output-tokens=".length) || options.maxOutputTokens);
      continue;
    }
    commandParts.push(arg);
  }

  options.commandLine = commandParts.join(" ").trim();
  if (!Number.isFinite(options.maxOutputTokens) || options.maxOutputTokens < 256) {
    options.maxOutputTokens = DEFAULT_MAX_OUTPUT_TOKENS;
  }
  return options;
}

function formatHelp() {
  const rows = [...COMMANDS.entries()].map(([command, meta]) => {
    return `${command.padEnd(13)} ${meta.purpose}\n${" ".repeat(13)} Usage: ${meta.usage}`;
  });
  return [
    `${APP_NAME} v${APP_VERSION}`,
    "",
    "Codex-style slash commands for AI discovery science workflows.",
    "",
    rows.join("\n"),
    "",
    "Run examples:",
    "  npm start",
    "  npm start -- \"Can intervention X improve outcome Y?\"",
    "  npm start -- /research \"Can intervention X improve outcome Y?\"",
    "  npm start -- /hypothesis \"Can intervention X improve outcome Y?\"",
    "  npm start -- --safety-level 4 /research \"Assess risks and safeguards for a lab-adjacent study\"",
    "  npm start -- --verbose /research \"Can intervention X improve outcome Y?\"",
    "  npm start -- --dry-run /experiment \"Test a new catalyst for CO2 reduction\"",
    "",
    "Interactive shortcut: type any research question to run the full pipeline.",
    "Focused modes: /hypothesis, /experiment, and /writer keep follow-up text in that mode for expand/change/rewrite work.",
    "Use /exit to leave a focused mode. Use /quit to close the CLI.",
    "Use /verbose to toggle live streamed AI output in API mode.",
    "Use /safety 1-5 to set the bio/chemical risk warning level. Level 5 is highest risk and does not lower safeguards.",
    "",
    "API mode uses OPENAI_API_KEY. Without it, the CLI automatically uses deterministic dry-run output.",
    "The literature stage uses hosted web search in API mode unless --no-web-search or AI_DISCOVERY_DISABLE_WEB_SEARCH=1 is set.",
  ].join("\n");
}

function safeTimestamp(date = new Date()) {
  return date.toISOString().replace(/[:.]/g, "-");
}

function truncate(text, maxChars) {
  const value = String(text || "").trim();
  if (value.length <= maxChars) {
    return value;
  }
  return `${value.slice(0, maxChars - 20).trim()}\n[truncated]`;
}

function jsonUsageTotals(usage) {
  return {
    input_tokens: Number(usage.input_tokens || usage.prompt_tokens || 0),
    output_tokens: Number(usage.output_tokens || usage.completion_tokens || 0),
    total_tokens: Number(usage.total_tokens || 0),
  };
}

function addUsageTotals(target, usage) {
  const totals = jsonUsageTotals(usage || {});
  target.input_tokens += totals.input_tokens;
  target.output_tokens += totals.output_tokens;
  target.total_tokens += totals.total_tokens || totals.input_tokens + totals.output_tokens;
}

function extractResponseText(responseJson) {
  if (typeof responseJson.output_text === "string" && responseJson.output_text.trim()) {
    return responseJson.output_text.trim();
  }
  const chunks = [];
  for (const item of responseJson.output || []) {
    for (const content of item.content || []) {
      if (typeof content.text === "string") {
        chunks.push(content.text);
      }
    }
  }
  return chunks.join("\n").trim();
}

function extractUrlCitations(responseJson) {
  const seen = new Set();
  const citations = [];
  for (const item of responseJson.output || []) {
    for (const content of item.content || []) {
      for (const annotation of content.annotations || []) {
        if (annotation && annotation.type === "url_citation" && annotation.url && !seen.has(annotation.url)) {
          seen.add(annotation.url);
          citations.push({
            title: annotation.title || annotation.url,
            url: annotation.url,
          });
        }
      }
    }
  }
  return citations;
}

function appendCitationBlock(markdown, citations) {
  if (!Array.isArray(citations) || citations.length === 0) {
    return markdown;
  }
  const existingText = String(markdown || "");
  const missing = citations.filter((citation) => !existingText.includes(citation.url));
  if (!missing.length) {
    return existingText;
  }
  return [
    existingText.trim(),
    "",
    "## Retrieved Source Links",
    ...missing.map((citation) => `- ${citation.title}: ${citation.url}`),
  ].join("\n");
}

function findSseBoundary(buffer) {
  const crlf = buffer.indexOf("\r\n\r\n");
  const lf = buffer.indexOf("\n\n");
  if (crlf === -1 && lf === -1) {
    return null;
  }
  if (crlf !== -1 && (lf === -1 || crlf < lf)) {
    return { index: crlf, length: 4 };
  }
  return { index: lf, length: 2 };
}

function parseSseEventBlock(block) {
  let eventType = "";
  const dataLines = [];
  for (const line of String(block || "").split(/\r?\n/)) {
    if (!line || line.startsWith(":")) {
      continue;
    }
    const separatorIndex = line.indexOf(":");
    const field = separatorIndex === -1 ? line : line.slice(0, separatorIndex);
    const rawValue = separatorIndex === -1 ? "" : line.slice(separatorIndex + 1);
    const value = rawValue.startsWith(" ") ? rawValue.slice(1) : rawValue;
    if (field === "event") {
      eventType = value.trim();
    } else if (field === "data") {
      dataLines.push(value);
    }
  }
  const data = dataLines.join("\n").trim();
  if (!data || data === "[DONE]") {
    return null;
  }
  let payload = {};
  try {
    payload = JSON.parse(data);
  } catch (error) {
    throw new Error(`OpenAI API returned invalid stream event: ${truncate(data, 500)}`);
  }
  return {
    type: payload.type || eventType,
    payload,
  };
}

function streamErrorMessage(event) {
  const payload = event && event.payload ? event.payload : {};
  const response = payload.response || {};
  const error = payload.error || response.error || {};
  return error.message || error.code || response.status || payload.message || "unknown streaming error";
}

async function readResponsesStream(response, options = {}) {
  if (!response.body || typeof response.body[Symbol.asyncIterator] !== "function") {
    throw new Error("OpenAI API streaming response did not include a readable body.");
  }

  const decoder = new TextDecoder();
  const textDeltas = [];
  const completedTexts = [];
  let buffer = "";
  let completedResponse = null;

  const consumeBlock = (block) => {
    const event = parseSseEventBlock(block);
    if (!event) {
      return;
    }
    if (event.type === "response.output_text.delta" && typeof event.payload.delta === "string") {
      textDeltas.push(event.payload.delta);
      if (typeof options.onTextDelta === "function") {
        options.onTextDelta(event.payload.delta);
      }
      return;
    }
    if (event.type === "response.output_text.done" && typeof event.payload.text === "string") {
      completedTexts.push(event.payload.text);
      return;
    }
    if (event.type === "response.completed" && event.payload.response) {
      completedResponse = event.payload.response;
      return;
    }
    if (event.type === "response.failed" || event.type === "error") {
      throw new Error(`OpenAI API stream failed: ${streamErrorMessage(event)}`);
    }
  };

  for await (const chunk of response.body) {
    buffer += decoder.decode(chunk, { stream: true });
    let boundary = findSseBoundary(buffer);
    while (boundary) {
      const block = buffer.slice(0, boundary.index);
      buffer = buffer.slice(boundary.index + boundary.length);
      consumeBlock(block);
      boundary = findSseBoundary(buffer);
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) {
    consumeBlock(buffer);
  }

  const raw = completedResponse || {};
  const responseText = extractResponseText(raw);
  const streamedText = textDeltas.join("").trim() || completedTexts.join("\n").trim();
  return {
    text: responseText || streamedText,
    citations: extractUrlCitations(raw),
    usage: raw.usage || {},
    raw,
  };
}

function formatGeneratedResponse(primaryText, footerLines = [], options = {}) {
  const sections = [];
  if (!options.suppressPrimary && String(primaryText || "").trim()) {
    sections.push(String(primaryText).trim());
  }
  const footer = footerLines.filter(Boolean).join("\n");
  if (footer) {
    sections.push(footer);
  }
  return sections.join("\n\n");
}

function makePromptPayload(systemPrompt, userPrompt) {
  return [
    {
      role: "system",
      content: [{ type: "input_text", text: systemPrompt }],
    },
    {
      role: "user",
      content: [{ type: "input_text", text: userPrompt }],
    },
  ];
}

async function callResponsesApi({
  apiKey,
  model,
  systemPrompt,
  userPrompt,
  maxOutputTokens,
  timeoutMs,
  tools,
  toolChoice,
  stream,
  onTextDelta,
}) {
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is not set.");
  }
  if (typeof fetch !== "function") {
    throw new Error("This CLI requires Node.js 18+ for fetch support.");
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs || DEFAULT_TIMEOUT_MS);
  const payload = {
    model,
    input: makePromptPayload(systemPrompt, userPrompt),
    max_output_tokens: maxOutputTokens || DEFAULT_MAX_OUTPUT_TOKENS,
  };
  if (stream) {
    payload.stream = true;
  }
  if (Array.isArray(tools) && tools.length) {
    payload.tools = tools;
  }
  if (toolChoice) {
    payload.tool_choice = toolChoice;
  }

  try {
    const response = await fetch("https://api.openai.com/v1/responses", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
        Accept: stream ? "text/event-stream" : "application/json",
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (stream && response.ok) {
      return readResponsesStream(response, { onTextDelta });
    }

    const text = await response.text();
    let responseJson = {};
    try {
      responseJson = text ? JSON.parse(text) : {};
    } catch (error) {
      throw new Error(`OpenAI API returned non-JSON response (${response.status}): ${truncate(text, 500)}`);
    }

    if (!response.ok) {
      const message = responseJson.error && responseJson.error.message ? responseJson.error.message : truncate(text, 500);
      throw new Error(`OpenAI API request failed (${response.status}): ${message}`);
    }

    return {
      text: extractResponseText(responseJson),
      citations: extractUrlCitations(responseJson),
      usage: responseJson.usage || {},
      raw: responseJson,
    };
  } finally {
    clearTimeout(timer);
  }
}

function buildHypothesisUserPrompt(question) {
  return [
    HYPOTHESIS_PROMPT,
    "",
    `Research Question: ${question}`,
    "Research Plan: Generate the most defensible first-pass plan from the question. State assumptions.",
  ].join("\n");
}

function buildPlanUserPrompt(question) {
  return [
    PLAN_PROMPT,
    "",
    `Research Question:\n${question}`,
  ].join("\n");
}

function buildLiteratureReviewUserPrompt(state) {
  return [
    LITERATURE_PROMPT,
    "",
    `Research Question:\n${state.activeQuestion || "[Missing]"}`,
    "",
    `Research Plan:\n${state.researchPlan || "[Missing]"}`,
  ].join("\n");
}

function buildPipelineHypothesisUserPrompt(state) {
  return [
    HYPOTHESIS_PROMPT,
    "",
    `Research Question:\n${state.activeQuestion || "[Missing]"}`,
    "",
    `Research Plan:\n${state.researchPlan || "[Missing]"}`,
    "",
    `Literature Review:\n${state.literatureReview || "[Missing]"}`,
  ].join("\n");
}

function buildExperimentUserPrompt(state, brief) {
  const question = brief || state.activeQuestion || "No explicit research question supplied.";
  return [
    EXPERIMENT_PROMPT,
    "",
    `Research Question:\n${question}`,
    "",
    `Research Plan:\n${state.researchPlan || "[No research plan generated yet.]"}`,
    "",
    `Literature Review:\n${state.literatureReview || "[No literature review generated yet.]"}`,
    "",
    `Hypotheses:\n${state.hypothesis || "[No hypothesis generated yet. Provide a partial design and label this gap.]"}`,
  ].join("\n");
}

function buildAnalysisUserPrompt(state) {
  return [
    ANALYSIS_PROMPT,
    "",
    `Research Question:\n${state.activeQuestion || "[Missing]"}`,
    "",
    `Hypothesis:\n${state.hypothesis || "[Missing]"}`,
    "",
    `Experiment Design:\n${state.experimentDesign || "[Missing]"}`,
    "",
    `Experiment Run Output:\n${state.experimentRun || "[Missing]"}`,
  ].join("\n");
}

function buildDraftPaperUserPrompt(state) {
  return [
    DRAFT_PAPER_PROMPT,
    "",
    `Research Question:\n${state.activeQuestion || "[Missing]"}`,
    "",
    `Research Plan:\n${state.researchPlan || "[Missing]"}`,
    "",
    `Literature Review:\n${state.literatureReview || "[Missing]"}`,
    "",
    `Hypothesis:\n${state.hypothesis || "[Missing]"}`,
    "",
    `Experiment Design:\n${state.experimentDesign || "[Missing]"}`,
    "",
    `Experiment Run Output:\n${state.experimentRun || "[Missing]"}`,
    "",
    `Data Analysis:\n${state.dataAnalysis || "[Missing]"}`,
  ].join("\n");
}

function buildTechnicalReviewUserPrompt(state) {
  return [
    TECHNICAL_REVIEW_PROMPT,
    "",
    `Research Question:\n${state.activeQuestion || "[Missing]"}`,
    "",
    `Literature Review:\n${state.literatureReview || "[Missing]"}`,
    "",
    `Draft Paper:\n${state.draftPaper || "[Missing]"}`,
  ].join("\n");
}

function buildFinalPaperUserPrompt(state) {
  return [
    FINAL_PAPER_PROMPT,
    "",
    `Research Question:\n${state.activeQuestion || "[Missing]"}`,
    "",
    `Research Plan:\n${state.researchPlan || "[Missing]"}`,
    "",
    `Literature Review:\n${state.literatureReview || "[Missing]"}`,
    "",
    `Hypothesis:\n${state.hypothesis || "[Missing]"}`,
    "",
    `Experiment Design:\n${state.experimentDesign || "[Missing]"}`,
    "",
    `Experiment Run Output:\n${state.experimentRun || "[Missing]"}`,
    "",
    `Data Analysis:\n${state.dataAnalysis || "[Missing]"}`,
    "",
    `Draft Paper:\n${state.draftPaper || "[Missing]"}`,
    "",
    `Technical Review:\n${state.technicalReview || "[Missing]"}`,
  ].join("\n");
}

function buildWriterUserPrompt(state, instructions) {
  return [
    WRITER_PROMPT,
    "",
    `Extra Instructions:\n${instructions || "[None]"}`,
    "",
    `Research Question:\n${state.activeQuestion || "[Missing]"}`,
    "",
    `Hypothesis:\n${state.hypothesis || "[Missing]"}`,
    "",
    `Experiment Design:\n${state.experimentDesign || "[Missing]"}`,
    "",
    `Experiment Run Output:\n${state.experimentRun || "[Missing]"}`,
    "",
    `Data Analysis:\n${state.dataAnalysis || "[Missing]"}`,
    "",
    `Experiment Code:\n${state.experimentCode || "[Missing]"}`,
  ].join("\n");
}

function buildHypothesisFollowupUserPrompt(state, instruction) {
  return [
    HYPOTHESIS_PROMPT,
    "",
    "You are continuing an interactive /hypothesis mode session.",
    "Apply the user follow-up by expanding, changing, or rewriting the hypothesis artifact.",
    "Return the complete replacement hypothesis artifact, not a partial patch.",
    "",
    `Follow-Up Instruction:\n${instruction}`,
    "",
    `Research Question:\n${state.activeQuestion || "[Missing]"}`,
    "",
    `Current Hypothesis Artifact:\n${state.hypothesis || "[Missing]"}`,
    "",
    `Research Plan:\n${state.researchPlan || "[No research plan generated yet.]"}`,
    "",
    `Literature Review:\n${state.literatureReview || "[No literature review generated yet.]"}`,
  ].join("\n");
}

function buildExperimentFollowupUserPrompt(state, instruction) {
  return [
    EXPERIMENT_PROMPT,
    "",
    "You are continuing an interactive /experiment mode session.",
    "Apply the user follow-up by expanding, changing, or rewriting the experiment artifact.",
    "Return the complete replacement experiment design and include one fenced javascript block.",
    "Do not claim generated code was executed.",
    "",
    `Follow-Up Instruction:\n${instruction}`,
    "",
    `Research Question:\n${state.activeQuestion || "[Missing]"}`,
    "",
    `Current Hypothesis:\n${state.hypothesis || "[Missing]"}`,
    "",
    `Current Experiment Design:\n${state.experimentDesign || "[Missing]"}`,
    "",
    `Current Experiment Run Output:\n${state.experimentRun || "[No run output saved yet.]"}`,
  ].join("\n");
}

function buildWriterFollowupUserPrompt(state, instruction) {
  return [
    WRITER_PROMPT,
    "",
    "You are continuing an interactive /writer mode session.",
    "Apply the user follow-up by expanding, changing, or rewriting the paper.",
    "Return the complete replacement paper in the required final format, not a partial patch.",
    "",
    `Follow-Up Instruction:\n${instruction}`,
    "",
    `Current Paper:\n${state.paper || "[Missing]"}`,
    "",
    `Research Question:\n${state.activeQuestion || "[Missing]"}`,
    "",
    `Hypothesis:\n${state.hypothesis || "[Missing]"}`,
    "",
    `Experiment Design:\n${state.experimentDesign || "[Missing]"}`,
    "",
    `Experiment Run Output:\n${state.experimentRun || "[Missing]"}`,
    "",
    `Data Analysis:\n${state.dataAnalysis || "[Missing]"}`,
    "",
    `Experiment Code:\n${state.experimentCode || "[Missing]"}`,
  ].join("\n");
}

function buildDryRunResearchPlan(question, safetyLevel = DEFAULT_SAFETY_LEVEL) {
  return [
    "## Problem Statement",
    `The research question is: ${question}. The immediate problem is to convert this broad prompt into a testable research workflow without overclaiming evidence that has not been retrieved or measured.`,
    "",
    "## Research Objective",
    "Produce a falsifiable hypothesis, a controlled experiment design, a review-first analysis scaffold, a synthetic pilot run, an analysis, a technically reviewed paper draft, and final LaTeX-ready paper artifacts.",
    "",
    "## Scope and Non-Goals",
    "- Scope: plan, literature-review scaffold, hypothesis, experiment, synthetic runner output, analysis, draft, review, final paper, and LaTeX.",
    "- Non-goal: claiming real empirical results or real literature retrieval in dry-run mode.",
    "",
    "## Success Criteria",
    "- Every stage writes a local artifact.",
    "- The hypothesis is falsifiable and paired with a null hypothesis.",
    "- The experiment defines variables, controls, metrics, sample-size assumptions, and failure handling.",
    "- The paper uses the requested sections: abstract, introduction, hypothesis, experiment, data, results, conclusion, references.",
    "- Research, hypothesis, and writer artifacts expose uncertainty, alternatives, provenance, counterarguments, and debate points.",
    "",
    "## Research Workflow",
    "1. Build the research plan.",
    "2. Perform literature review or mark it assumption-based when search is unavailable.",
    "3. Generate hypotheses.",
    "4. Design the experiment and save review-first Node.js code.",
    "5. Run the built-in synthetic pilot runner without executing generated code.",
    "6. Analyze synthetic output.",
    "7. Generate a draft paper.",
    "8. Perform a technical review.",
    "9. Generate final Markdown and LaTeX paper artifacts.",
    "",
    "## Risks and Constraints",
    "- Dry-run mode does not call the model or retrieve literature.",
    "- Synthetic pilot output is not empirical evidence.",
    "- Generated code and LaTeX are saved for review and are not executed automatically.",
    "- User prompts and model output are treated as untrusted content.",
    `- ${formatSafetyProfile(safetyLevel).replace(/\n/g, " ")}`,
  ].join("\n");
}

function buildDryRunLiteratureReview(question, safetyLevel = DEFAULT_SAFETY_LEVEL) {
  return [
    "## Search Strategy",
    "Dry-run mode did not call hosted web search. In live API mode, this stage asks the Responses API to use hosted web search for peer-reviewed papers, official datasets, and authoritative technical sources relevant to the research question.",
    "",
    "## Literature Review",
    `For the question "${question}", the dry-run review can only identify likely evidence needs: prior empirical studies, measurement validity work, baseline or comparator literature, statistical power assumptions, and known threats to inference.`,
    "",
    "## Evidence Table",
    "| Evidence need | Dry-run status | How to validate |",
    "| --- | --- | --- |",
    "| Domain mechanism | Assumption-based | Retrieve review articles and mechanism papers |",
    "| Measurement method | Assumption-based | Retrieve validated measurement protocols |",
    "| Expected effect size | Unknown | Use pilot data or meta-analysis |",
    "| Confounders and bias | Assumption-based | Review study-design literature |",
    "| Provenance and counterarguments | Not retrieved in dry-run mode | Compare primary sources and explicitly record disagreements |",
    "",
    "## Gaps and Assumptions",
    "- No external sources were retrieved in dry-run mode.",
    "- Claims must be treated as low-confidence until live search or user-supplied sources are reviewed.",
    "- The later hypothesis and experiment should avoid source-specific claims unless references are added.",
    `- ${formatSafetyProfile(safetyLevel).replace(/\n/g, " ")}`,
    "",
    "## References",
    "- No retrieved references in dry-run mode.",
  ].join("\n");
}

function buildDryRunHypothesis(question, safetyLevel = DEFAULT_SAFETY_LEVEL) {
  const framedQuestion = `For the research question "${question}"`;
  return [
    `Interpreted question: the user is asking whether the effect or relationship described in "${question}" can be tested as a measurable contrast involving a focal intervention, exposure, or condition, a comparator, and a prespecified primary outcome in a defined population or system. A causal-inference and experimental-design framing requires a comparator, outcome, target population, measurement protocol, and decision rule before an observed association can be treated as evidence; because dry-run mode retrieves no literature, citations are placeholders and all literature claims are lower-confidence ([Author, Year]; [Author, Year]). The main empirical rationale is that the question implies a measurable contrast, while counterarguments such as selection bias, confounding, measurement reactivity, and weak source provenance remain plausible rival explanations. The claim-level evidence audit is therefore provisional: novelty against prior literature is unverified, and prospective tests must preregister the endpoint, model, controls, and falsifying observations before treating an alternative hypothesis as supported.`,
    `- Null Hypothesis (H0): ${framedQuestion}, the focal intervention, exposure, or condition has no statistically detectable effect on the prespecified primary outcome relative to a defined comparator in the target population or system under the planned measurement protocol.`,
    `- Alternative Hypothesis (H1): ${framedQuestion}, the focal intervention, exposure, or condition produces a statistically significant effect on the prespecified primary outcome in the direction implied by the question relative to the defined comparator under the planned measurement protocol.`,
    `- Alternative Hypothesis (H2): ${framedQuestion}, the H1 effect remains directionally consistent after adjustment for prespecified confounders and is not reproduced on a negative-control or placebo outcome, supporting a specific rather than artifactual association.`,
  ].join("\n");
}

function buildExperimentCode(question) {
  const escapedQuestion = JSON.stringify(question || "AI discovery experiment");
  return `#!/usr/bin/env node
"use strict";

// Synthetic pilot analysis generated by AI Discovery CLI.
// Review and adapt this script before using it for real scientific claims.

const researchQuestion = ${escapedQuestion};
const seed = 42;
const nPerGroup = 48;
const assumedEffect = 0.35;

function lcg(initialSeed) {
  let state = initialSeed >>> 0;
  return function next() {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function normal(random) {
  const u1 = Math.max(random(), Number.EPSILON);
  const u2 = Math.max(random(), Number.EPSILON);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function variance(values) {
  const avg = mean(values);
  return values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / (values.length - 1);
}

function summarize(values) {
  return {
    n: values.length,
    mean: mean(values),
    variance: variance(values),
  };
}

const random = lcg(seed);
const control = Array.from({ length: nPerGroup }, () => normal(random));
const treatment = Array.from({ length: nPerGroup }, () => normal(random) + assumedEffect);

const controlSummary = summarize(control);
const treatmentSummary = summarize(treatment);
const meanDifference = treatmentSummary.mean - controlSummary.mean;
const standardError = Math.sqrt(controlSummary.variance / controlSummary.n + treatmentSummary.variance / treatmentSummary.n);
const tStatistic = meanDifference / standardError;

console.log(JSON.stringify({
  researchQuestion,
  syntheticData: true,
  seed,
  nPerGroup,
  assumedEffect,
  control: controlSummary,
  treatment: treatmentSummary,
  meanDifference,
  standardError,
  tStatistic,
  interpretation: "Synthetic pilot only. Replace simulated values with validated empirical data before drawing scientific conclusions."
}, null, 2));
`;
}

function buildDryRunExperiment(question, hypothesis, safetyLevel = DEFAULT_SAFETY_LEVEL) {
  const code = buildExperimentCode(question);
  return {
    design: [
      "## Experimental Design",
      `A controlled pilot study will test the active hypothesis for: ${question || "the supplied research question"}. The design uses one focal condition and one comparator condition, with predefined primary outcome measurement and a negative-control check.`,
      "",
      "## Procedure",
      "1. Define inclusion/exclusion criteria and preregister the primary outcome.",
      "2. Randomly assign units to treatment and comparator conditions.",
      "3. Collect baseline covariates, deliver the intervention/exposure, and measure outcomes at the predefined endpoint.",
      "4. Run the provided Node.js pilot analysis template on validated data or synthetic data for dry-run verification.",
      "",
      "## Controls",
      "- Comparator group receiving standard condition or placebo/control exposure.",
      "- Negative-control outcome to detect broad measurement artifacts.",
      "- Baseline covariates for adjustment and balance checks.",
      "",
      "## Materials",
      "- Measurement protocol for primary and secondary outcomes.",
      "- Randomization record.",
      "- Data dictionary and CSV/JSON dataset.",
      "- Generated Node.js analysis script.",
      "",
      "## Sample Size and Power",
      "Initial planning target: N approximately 260 total for 80% power at alpha = 0.05 if the standardized effect is d = 0.35. Replace this with a domain-specific calculation after pilot variance is observed.",
      "",
      "## Randomization and Blinding",
      "Use block randomization when sample size is small or strata matter. Blind outcome assessors where feasible. Record any unblinding events as protocol deviations.",
      "",
      "## Metrics",
      "- Primary outcome: protocol-defined quantitative endpoint.",
      "- Secondary outcomes: robustness and mechanism checks.",
      "- Diagnostic metrics: missingness, attrition, baseline balance, and negative-control effect.",
      "",
      "## Data Collection",
      "Collect minimally necessary data for the research objective. Avoid credentials, PII, and sensitive payloads unless a reviewed protocol requires them.",
      "",
      "## Ethical and Practical Considerations",
      "Document consent, risk controls, data minimization, retention, and review requirements. Treat all generated code and synthetic findings as untrusted until reviewed.",
      `Safety profile: ${formatSafetyProfile(safetyLevel).replace(/\n/g, " ")}`,
      "",
      "## Generated Experiment Code",
      "```javascript",
      code.trimEnd(),
      "```",
      "",
      "Hypothesis context used:",
      hypothesis ? truncate(hypothesis, 1200) : "[No prior hypothesis in session; design is assumption-bounded.]",
    ].join("\n"),
    code,
  };
}

function makeSyntheticExperimentRun(question) {
  const seed = 42;
  const nPerGroup = 48;
  const assumedEffect = 0.35;
  const random = (() => {
    let state = seed >>> 0;
    return () => {
      state = (1664525 * state + 1013904223) >>> 0;
      return state / 0x100000000;
    };
  })();
  const normal = () => {
    const u1 = Math.max(random(), Number.EPSILON);
    const u2 = Math.max(random(), Number.EPSILON);
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
  const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = (values) => {
    const avg = mean(values);
    return values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / (values.length - 1);
  };
  const summarize = (values) => ({
    n: values.length,
    mean: mean(values),
    variance: variance(values),
  });

  const control = Array.from({ length: nPerGroup }, () => normal());
  const treatment = Array.from({ length: nPerGroup }, () => normal() + assumedEffect);
  const controlSummary = summarize(control);
  const treatmentSummary = summarize(treatment);
  const meanDifference = treatmentSummary.mean - controlSummary.mean;
  const standardError = Math.sqrt(controlSummary.variance / controlSummary.n + treatmentSummary.variance / treatmentSummary.n);

  return {
    researchQuestion: question || "AI discovery experiment",
    syntheticData: true,
    generatedBy: "AI Discovery CLI built-in synthetic runner",
    generatedCodeExecuted: false,
    seed,
    nPerGroup,
    assumedEffect,
    control: controlSummary,
    treatment: treatmentSummary,
    meanDifference,
    standardError,
    tStatistic: meanDifference / standardError,
    interpretation: "Synthetic pilot only. Replace simulated values with validated empirical data before drawing scientific conclusions.",
  };
}

function buildExperimentRunMarkdown(runOutput) {
  const runJson = typeof runOutput === "string" ? runOutput : JSON.stringify(runOutput, null, 2);
  return [
    "## Experiment Runner Output",
    "The CLI used its built-in deterministic synthetic runner. It did not execute model-generated code.",
    "",
    "```json",
    runJson,
    "```",
  ].join("\n");
}

function buildDryRunAnalysis(question, hypothesis, experimentRun, safetyLevel = DEFAULT_SAFETY_LEVEL) {
  const parsedRun = typeof experimentRun === "string" ? experimentRun : JSON.stringify(experimentRun, null, 2);
  return [
    "## Data",
    "The available data are synthetic pilot values generated by the CLI's built-in deterministic runner. They are suitable for validating workflow plumbing, not for scientific claims.",
    "",
    "## Results",
    "The synthetic output includes group summaries, a mean difference, a standard error, and a t-statistic. These values are generated from simulated normal data with an assumed treatment effect.",
    "",
    "## Statistical Interpretation",
    "Because the data are synthetic, the statistic should be interpreted only as a demonstration of the planned analysis path.",
    "",
    "## Hypothesis Assessment",
    hypothesis ? "The synthetic result can be used to check whether the planned test maps to the hypothesis, but it cannot support or refute the hypothesis empirically." : "No hypothesis was available, so no hypothesis assessment can be made.",
    "",
    "## Limitations",
    "- No real data were collected.",
    "- No live model or web search was used in dry-run mode.",
    "- The effect size and data distribution are assumptions.",
    "- Alternative explanations and counterarguments remain unresolved until real data, provenance checks, and preregistered tests are available.",
    `- ${formatSafetyProfile(safetyLevel).replace(/\n/g, " ")}`,
    "",
    "## Reproducibility Notes",
    `Research question: ${question || "[missing]"}`,
    "",
    "Experiment run excerpt:",
    "```json",
    truncate(parsedRun, 1600),
    "```",
  ].join("\n");
}

function buildDryRunFinalPaper(
  question,
  hypothesis,
  experimentDesign,
  experimentRun,
  analysis,
  literatureReview,
  safetyLevel = DEFAULT_SAFETY_LEVEL,
) {
  return [
    "# Title",
    `AI Discovery Research Paper: ${question || "Untitled Research Question"}`,
    "",
    "## Abstract",
    "This paper presents a dry-run AI discovery workflow that converts a research question into a research plan, literature-review scaffold, falsifiable hypothesis, experiment design, synthetic pilot run, analysis, technical review, and final LaTeX-ready paper. The experiment output is synthetic and should not be interpreted as empirical evidence.",
    "",
    "## Introduction",
    `The research question is: ${question || "[missing]"}. The workflow is designed to make the question testable while separating retrieved evidence, assumptions, planned methods, and synthetic pilot output.`,
    "Uncertainty, provenance, alternatives, and counterarguments are treated as first-class review items rather than hidden caveats.",
    "",
    "## Hypothesis",
    hypothesis ? truncate(hypothesis, 1400) : "No hypothesis artifact was available.",
    "",
    "## Experiment",
    experimentDesign ? truncate(experimentDesign, 1600) : "No experiment design artifact was available.",
    "",
    "## Data",
    "The data in this run are synthetic pilot data generated by the CLI's built-in deterministic runner. The runner did not execute model-generated code.",
    "",
    "```json",
    truncate(typeof experimentRun === "string" ? experimentRun : JSON.stringify(experimentRun, null, 2), 1400),
    "```",
    "",
    "## Results",
    analysis ? truncate(analysis, 1600) : "No analysis artifact was available.",
    "",
    "## Conclusion",
    "The workflow produced a reviewable first-pass research package and final paper artifact. Scientific conclusions remain deferred until literature sources are retrieved, empirical data are collected, and the generated analysis code is reviewed and run in a controlled environment.",
    `Safety and debate note: ${formatSafetyProfile(safetyLevel).replace(/\n/g, " ")}`,
    "",
    "## References",
    literatureReview && /https?:\/\//.test(literatureReview)
      ? "See the literature review artifact for retrieved source links."
      : "No retrieved references are available in dry-run mode.",
  ].join("\n");
}

function buildDryRunTechnicalReview(draftPaper, safetyLevel = DEFAULT_SAFETY_LEVEL) {
  return [
    "## Summary",
    "The draft is structurally complete for the requested final paper format, but dry-run evidence remains assumption-bound.",
    "",
    "## Correctness Review",
    "- The draft must not state or imply that synthetic output is empirical evidence.",
    "- The hypothesis should remain falsifiable and directly connected to the planned experiment.",
    "",
    "## Methodology Review",
    "- Controls, randomization, metrics, and power assumptions need domain-specific review before real data collection.",
    "- Generated code should be reviewed before execution.",
    "",
    "## Citation and Evidence Review",
    "- Dry-run mode has no retrieved sources; all literature claims must be labeled as assumptions.",
    "- Claim-level evidence audits, novelty checks, provenance notes, and counterarguments must remain visible in the final artifact.",
    "",
    "## Security and Reproducibility Review",
    "- The CLI writes local artifacts only and does not execute generated code.",
    "- The synthetic runner is deterministic and records the seed.",
    `- ${formatSafetyProfile(safetyLevel).replace(/\n/g, " ")}`,
    "",
    "## Required Revisions",
    draftPaper && draftPaper.includes("## References")
      ? "- Keep the references section explicit about retrieved versus unavailable sources."
      : "- Add the required References section before finalization.",
  ].join("\n");
}

function buildDryRunPaper(
  question,
  hypothesis,
  experimentDesign,
  experimentCode,
  experimentRun = "",
  analysis = "",
  literatureReview = "",
  safetyLevel = DEFAULT_SAFETY_LEVEL,
) {
  return [
    buildDryRunFinalPaper(question, hypothesis, experimentDesign, experimentRun, analysis, literatureReview, safetyLevel),
    "",
    "<!-- Generated experiment code excerpt -->",
    "```javascript",
    truncate(experimentCode || "[missing]", 1200),
    "```",
  ].join("\n");
}

function buildDryRunHypothesisFollowup(state, instruction) {
  const base = state.hypothesis || buildDryRunHypothesis(state.activeQuestion || instruction, state.safetyLevel);
  return [
    base.trim(),
    "",
    "Dry-run follow-up applied:",
    instruction,
    "",
    "Dry-run note: in live API mode, this follow-up rewrites the full hypothesis artifact while preserving the required rationale-first H0/H1 format.",
  ].join("\n");
}

function buildDryRunExperimentFollowup(state, instruction) {
  const question = state.activeQuestion || instruction;
  const base = buildDryRunExperiment(question, state.hypothesis, state.safetyLevel).design;
  return [
    base.trim(),
    "",
    "Dry-run follow-up applied:",
    instruction,
    "",
    "Dry-run note: the built-in synthetic runner is rerun after each experiment-mode follow-up. Model-generated code is still saved for review and not executed.",
  ].join("\n");
}

function buildDryRunWriterFollowup(state, instruction) {
  return [
    buildDryRunPaper(
      state.activeQuestion,
      state.hypothesis,
      state.experimentDesign,
      state.experimentCode,
      state.experimentRun,
      state.dataAnalysis,
      state.literatureReview,
      state.safetyLevel,
    ).trim(),
    "",
    "<!-- Writer mode follow-up applied:",
    instruction,
    "-->",
  ].join("\n");
}

function escapeLatex(text) {
  return String(text || "")
    .replace(/\\/g, "\\textbackslash{}")
    .replace(/&/g, "\\&")
    .replace(/%/g, "\\%")
    .replace(/\$/g, "\\$")
    .replace(/#/g, "\\#")
    .replace(/_/g, "\\_")
    .replace(/{/g, "\\{")
    .replace(/}/g, "\\}")
    .replace(/~/g, "\\textasciitilde{}")
    .replace(/\^/g, "\\textasciicircum{}");
}

function markdownToSimpleLatex(title, markdown) {
  const lines = String(markdown || "").split(/\r?\n/);
  const body = [];
  for (const line of lines) {
    if (line.startsWith("# ")) {
      body.push(`\\section*{${escapeLatex(line.slice(2).trim())}}`);
    } else if (line.startsWith("## ")) {
      body.push(`\\section{${escapeLatex(line.slice(3).trim())}}`);
    } else if (line.startsWith("- ")) {
      body.push(`\\noindent\\textbullet{} ${escapeLatex(line.slice(2).trim())}\\\\`);
    } else if (/^\d+\.\s/.test(line)) {
      body.push(`\\noindent ${escapeLatex(line.trim())}\\\\`);
    } else if (line.trim() === "```javascript" || line.trim() === "```") {
      body.push(`\\noindent\\texttt{${escapeLatex(line.trim())}}\\\\`);
    } else if (line.trim()) {
      body.push(`${escapeLatex(line.trim())}\n`);
    } else {
      body.push("");
    }
  }

  return [
    "\\documentclass[12pt,letterpaper]{article}",
    "\\usepackage[letterpaper, margin=1in]{geometry}",
    "\\usepackage{hyperref}",
    "\\usepackage{url}",
    "\\title{" + escapeLatex(title || "AI Discovery Research Paper") + "}",
    "\\author{AI Discovery CLI}",
    "\\date{\\today}",
    "\\begin{document}",
    "\\maketitle",
    body.join("\n"),
    "\\end{document}",
    "",
  ].join("\n");
}

class ScienceCodexCli {
  constructor(options = {}) {
    this.state = {
      model: options.model || DEFAULT_MODEL,
      dryRun: Boolean(options.dryRun || !process.env.OPENAI_API_KEY),
      outputDir: path.resolve(options.outputDir || DEFAULT_OUTPUT_DIR),
      sessionDir: "",
      activeMode: "",
      activeQuestion: "",
      webSearch: options.webSearch !== false,
      verbose: Boolean(options.verbose),
      safetyLevel: normalizeSafetyLevel(options.safetyLevel ?? process.env.AI_DISCOVERY_SAFETY_LEVEL ?? DEFAULT_SAFETY_LEVEL),
      researchPlan: "",
      literatureReview: "",
      hypothesis: "",
      experimentDesign: "",
      experimentCode: "",
      experimentRun: "",
      dataAnalysis: "",
      draftPaper: "",
      technicalReview: "",
      finalPaper: "",
      finalLatex: "",
      paper: "",
      history: [],
      usage: {
        input_tokens: 0,
        output_tokens: 0,
        total_tokens: 0,
      },
    };
    this.apiKey = options.apiKey || process.env.OPENAI_API_KEY || "";
    this.maxOutputTokens = options.maxOutputTokens || DEFAULT_MAX_OUTPUT_TOKENS;
    this.timeoutMs = options.timeoutMs || DEFAULT_TIMEOUT_MS;
    this.output = options.output || console.log;
    this.write = options.write || ((chunk) => output.write(chunk));
    this.error = options.error || console.error;
  }

  async ensureSessionDir() {
    if (!this.state.sessionDir) {
      this.state.sessionDir = path.join(this.state.outputDir, `run_${safeTimestamp()}`);
      await fs.mkdir(this.state.sessionDir, { recursive: true });
    }
    return this.state.sessionDir;
  }

  async writeArtifact(filename, content) {
    const sessionDir = await this.ensureSessionDir();
    const target = path.join(sessionDir, filename);
    await fs.writeFile(target, content, "utf8");
    return target;
  }

  async writeSessionManifest() {
    if (!this.state.sessionDir) {
      return "";
    }
    const manifest = {
      app: APP_NAME,
      version: APP_VERSION,
      model: this.state.model,
      dryRun: this.state.dryRun,
      safetyLevel: this.state.safetyLevel,
      safetyProfile: safetyProfile(this.state.safetyLevel).label,
      activeMode: this.state.activeMode,
      activeQuestion: this.state.activeQuestion,
      artifacts: {
        researchPlan: Boolean(this.state.researchPlan),
        literatureReview: Boolean(this.state.literatureReview),
        hypothesis: Boolean(this.state.hypothesis),
        experimentDesign: Boolean(this.state.experimentDesign),
        experimentCode: Boolean(this.state.experimentCode),
        experimentRun: Boolean(this.state.experimentRun),
        dataAnalysis: Boolean(this.state.dataAnalysis),
        draftPaper: Boolean(this.state.draftPaper),
        technicalReview: Boolean(this.state.technicalReview),
        finalPaper: Boolean(this.state.finalPaper),
        finalLatex: Boolean(this.state.finalLatex),
        paper: Boolean(this.state.paper),
      },
      usage: this.state.usage,
      history: this.state.history,
    };
    const target = path.join(this.state.sessionDir, "session.json");
    await fs.writeFile(target, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
    return target;
  }

  resetScienceSession() {
    this.state.sessionDir = "";
    this.state.activeMode = "";
    this.state.activeQuestion = "";
    this.state.researchPlan = "";
    this.state.literatureReview = "";
    this.state.hypothesis = "";
    this.state.experimentDesign = "";
    this.state.experimentCode = "";
    this.state.experimentRun = "";
    this.state.dataAnalysis = "";
    this.state.draftPaper = "";
    this.state.technicalReview = "";
    this.state.finalPaper = "";
    this.state.finalLatex = "";
    this.state.paper = "";
  }

  statusText() {
    return [
      `${APP_NAME} v${APP_VERSION}`,
      `Model: ${this.state.model}`,
      `Mode: ${this.state.dryRun ? "dry-run" : "OpenAI Responses API"}`,
      `Web search: ${this.state.webSearch ? "enabled for literature stage" : "disabled"}`,
      `Verbose streaming output: ${this.state.verbose ? "on" : "off"}`,
      `Bio/chemical safety level: ${this.state.safetyLevel} - ${safetyProfile(this.state.safetyLevel).label}`,
      `Output directory: ${this.state.outputDir}`,
      `Session directory: ${this.state.sessionDir || "[not created yet]"}`,
      `Active focused mode: ${this.state.activeMode || "[none]"}`,
      `Active question: ${this.state.activeQuestion || "[none]"}`,
      `Artifacts: plan=${Boolean(this.state.researchPlan)}, literature=${Boolean(this.state.literatureReview)}, hypothesis=${Boolean(this.state.hypothesis)}, experiment=${Boolean(this.state.experimentDesign)}, run=${Boolean(this.state.experimentRun)}, analysis=${Boolean(this.state.dataAnalysis)}, final=${Boolean(this.state.finalPaper || this.state.paper)}`,
      `Usage: input=${this.state.usage.input_tokens}, output=${this.state.usage.output_tokens}, total=${this.state.usage.total_tokens}`,
    ].join("\n");
  }

  async generateWithModel(kind, userPrompt, dryRunFactory, options = {}) {
    if (this.state.dryRun) {
      return {
        text: dryRunFactory(),
        usage: {},
        source: "dry-run",
        visibleStreamed: false,
      };
    }

    let visibleStreamed = false;
    let streamLineOpen = false;
    const onTextDelta = this.state.verbose
      ? (delta) => {
          if (!streamLineOpen) {
            this.output(`[${kind}] AI response:`);
            streamLineOpen = true;
            visibleStreamed = true;
          }
          this.write(delta);
        }
      : undefined;
    const closeVisibleStream = () => {
      if (streamLineOpen) {
        this.write("\n");
        streamLineOpen = false;
      }
    };
    const buildResult = (result, source) => ({
      text: options.appendCitations ? appendCitationBlock(result.text, result.citations) : result.text,
      usage: result.usage,
      source,
      visibleStreamed,
    });
    const systemPrompt = buildSystemPrompt(this.state.safetyLevel);
    const governedUserPrompt = appendRuntimeGovernance(userPrompt, this.state.safetyLevel);
    const invokeResponsesApi = async (requestOptions = {}) => {
      try {
        return await callResponsesApi({
          apiKey: this.apiKey,
          model: this.state.model,
          systemPrompt,
          userPrompt: governedUserPrompt,
          maxOutputTokens: this.maxOutputTokens,
          timeoutMs: this.timeoutMs,
          stream: true,
          onTextDelta,
          ...requestOptions,
        });
      } finally {
        closeVisibleStream();
      }
    };

    try {
      const result = await invokeResponsesApi({
        tools: options.tools,
        toolChoice: options.toolChoice,
      });
      addUsageTotals(this.state.usage, result.usage);
      return buildResult(result, "api");
    } catch (error) {
      if (options.fallbackWithoutTools && Array.isArray(options.tools) && options.tools.length) {
        const result = await invokeResponsesApi();
        addUsageTotals(this.state.usage, result.usage);
        return buildResult(result, "api-no-web-search-fallback");
      }
      throw new Error(`${kind} generation failed: ${error.message}`);
    }
  }

  async resolveResearchQuestion(args, ask, usageText) {
    let question = String(args || "").trim();
    if (!question && typeof ask === "function") {
      question = String(await ask("Research question> ") || "").trim();
    }
    if (!question) {
      return { error: usageText };
    }
    return { question };
  }

  handleVerbose(args) {
    const value = String(args || "").trim().toLowerCase();
    if (!value) {
      this.state.verbose = !this.state.verbose;
      return `Verbose streaming output: ${this.state.verbose ? "on" : "off"}.`;
    }
    if (["on", "true", "1", "yes", "show"].includes(value)) {
      this.state.verbose = true;
      return "Verbose streaming output: on.";
    }
    if (["off", "false", "0", "no", "hide", "quiet"].includes(value)) {
      this.state.verbose = false;
      return "Verbose streaming output: off.";
    }
    return "Usage: /verbose [on|off]";
  }

  handleSafety(args) {
    const value = String(args || "").trim();
    if (!value) {
      return formatSafetyProfile(this.state.safetyLevel);
    }
    const parsed = parseSafetyLevelInput(value);
    if (!parsed) {
      return "Usage: /safety [1-5]";
    }
    this.state.safetyLevel = parsed;
    return `Bio/chemical safety level set to ${parsed} - ${safetyProfile(parsed).label}.\n${safetyProfile(parsed).warning}`;
  }

  async runWorkflowStage(label, action) {
    this.output(`[${label}] Running...`);
    const result = await action();
    this.output(`[${label}] Done.`);
    return result;
  }

  enterFocusedMode(command) {
    if (FOCUSED_MODE_COMMANDS.has(command)) {
      this.state.activeMode = command;
    }
  }

  modeInstructionText() {
    if (!this.state.activeMode) {
      return "";
    }
    return `Active mode: ${this.state.activeMode}. Type follow-up instructions to expand/change/rewrite. Use /exit to leave this mode or /quit to close the CLI.`;
  }

  shouldRouteBareTextToActiveMode(raw) {
    if (!this.state.activeMode || !raw || raw.startsWith("/")) {
      return false;
    }
    const firstSpace = raw.search(/\s/);
    const head = firstSpace === -1 ? raw : raw.slice(0, firstSpace);
    return !COMMAND_ALIASES.has(head.toLowerCase());
  }

  routeCommandLine(line) {
    const raw = String(line || "").trim();
    if (this.shouldRouteBareTextToActiveMode(raw)) {
      return {
        command: this.state.activeMode,
        args: raw,
        modeFollowup: true,
      };
    }
    return {
      ...parseCommandLine(raw),
      modeFollowup: false,
    };
  }

  async handleExit() {
    if (!this.state.activeMode) {
      return "__EXIT__";
    }
    const exitedMode = this.state.activeMode;
    this.state.activeMode = "";
    await this.writeSessionManifest();
    return `Exited ${exitedMode} mode. Bare text will now start the full /research workflow. Use /quit to close the CLI.`;
  }

  async handleHypothesis(args, context = {}) {
    const resolved = await this.resolveResearchQuestion(args, context.ask, "Usage: /hypothesis [research question]");
    if (resolved.error) {
      return resolved.error;
    }
    const question = resolved.question;
    this.state.activeQuestion = question;
    const result = await this.generateWithModel(
      "Hypothesis",
      buildHypothesisUserPrompt(question),
      () => buildDryRunHypothesis(question, this.state.safetyLevel),
    );
    this.state.hypothesis = result.text;
    this.enterFocusedMode("/hypothesis");
    const artifactPath = await this.writeArtifact("01_hypothesis.md", result.text);
    await this.writeSessionManifest();
    return formatGeneratedResponse(result.text, [
      `Saved hypothesis artifact: ${artifactPath}`,
      this.modeInstructionText(),
      `Source: ${result.source}`,
    ], { suppressPrimary: result.visibleStreamed });
  }

  async handleHypothesisFollowup(instruction) {
    const result = await this.generateWithModel(
      "Hypothesis",
      buildHypothesisFollowupUserPrompt(this.state, instruction),
      () => buildDryRunHypothesisFollowup(this.state, instruction),
    );
    this.state.hypothesis = result.text;
    this.enterFocusedMode("/hypothesis");
    const artifactPath = await this.writeArtifact("01_hypothesis.md", result.text);
    await this.writeSessionManifest();
    return formatGeneratedResponse(result.text, [
      `Saved hypothesis artifact: ${artifactPath}`,
      this.modeInstructionText(),
      `Source: ${result.source}`,
    ], { suppressPrimary: result.visibleStreamed });
  }

  async handleResearch(args, context = {}) {
    const resolved = await this.resolveResearchQuestion(args, context.ask, "Usage: /research <research question>");
    if (resolved.error) {
      return resolved.error;
    }
    const question = resolved.question;
    this.resetScienceSession();
    this.state.activeQuestion = question;
    await this.ensureSessionDir();

    const outputFiles = {};
    this.output(`Starting end-to-end research workflow for: ${question}`);

    const plan = await this.runWorkflowStage("Plan", () => this.generateWithModel(
      "Plan",
      buildPlanUserPrompt(question),
      () => buildDryRunResearchPlan(question, this.state.safetyLevel),
    ));
    this.state.researchPlan = plan.text;
    outputFiles.plan = await this.writeArtifact("01_research_plan.md", plan.text);

    const literatureOptions = this.state.webSearch
      ? {
          tools: [WEB_SEARCH_TOOL],
          toolChoice: "required",
          fallbackWithoutTools: true,
          appendCitations: true,
        }
      : {};
    const literature = await this.runWorkflowStage("Literature Review", () => this.generateWithModel(
      "Literature Review",
      buildLiteratureReviewUserPrompt(this.state),
      () => buildDryRunLiteratureReview(question, this.state.safetyLevel),
      literatureOptions,
    ));
    this.state.literatureReview = literature.text;
    outputFiles.literatureReview = await this.writeArtifact("02_literature_review.md", literature.text);

    const hypothesis = await this.runWorkflowStage("Hypothesis", () => this.generateWithModel(
      "Hypothesis",
      buildPipelineHypothesisUserPrompt(this.state),
      () => buildDryRunHypothesis(question, this.state.safetyLevel),
    ));
    this.state.hypothesis = hypothesis.text;
    outputFiles.hypothesis = await this.writeArtifact("03_hypothesis.md", hypothesis.text);

    const experiment = await this.runWorkflowStage("Experiment", () => this.generateWithModel(
      "Experiment",
      buildExperimentUserPrompt(this.state, question),
      () => buildDryRunExperiment(question, this.state.hypothesis, this.state.safetyLevel).design,
    ));
    const generatedCode = buildExperimentCode(question);
    this.state.experimentDesign = experiment.text.includes("## Generated Experiment Code")
      ? experiment.text
      : `${experiment.text.trim()}\n\n## Generated Experiment Code\n\n\`\`\`javascript\n${generatedCode.trimEnd()}\n\`\`\`\n`;
    this.state.experimentCode = generatedCode;
    outputFiles.experimentDesign = await this.writeArtifact("04_experiment_design.md", this.state.experimentDesign);
    outputFiles.experimentCode = await this.writeArtifact("05_experiment_code.js", generatedCode);

    const experimentRunObject = await this.runWorkflowStage("Experiment Runner", async () => makeSyntheticExperimentRun(question));
    this.state.experimentRun = JSON.stringify(experimentRunObject, null, 2);
    outputFiles.experimentRun = await this.writeArtifact("06_experiment_run.json", `${this.state.experimentRun}\n`);
    outputFiles.experimentRunMarkdown = await this.writeArtifact(
      "06_experiment_run.md",
      buildExperimentRunMarkdown(this.state.experimentRun),
    );

    const analysis = await this.runWorkflowStage("Data Analysis", () => this.generateWithModel(
      "Data Analysis",
      buildAnalysisUserPrompt(this.state),
      () => buildDryRunAnalysis(question, this.state.hypothesis, this.state.experimentRun, this.state.safetyLevel),
    ));
    this.state.dataAnalysis = analysis.text;
    outputFiles.dataAnalysis = await this.writeArtifact("07_data_analysis.md", analysis.text);

    const draft = await this.runWorkflowStage("Draft Paper", () => this.generateWithModel(
      "Draft Paper",
      buildDraftPaperUserPrompt(this.state),
      () => buildDryRunFinalPaper(
        question,
        this.state.hypothesis,
        this.state.experimentDesign,
        this.state.experimentRun,
        this.state.dataAnalysis,
        this.state.literatureReview,
        this.state.safetyLevel,
      ),
    ));
    this.state.draftPaper = draft.text;
    outputFiles.draftPaper = await this.writeArtifact("08_draft_paper.md", draft.text);

    const technicalReview = await this.runWorkflowStage("Technical Review", () => this.generateWithModel(
      "Technical Review",
      buildTechnicalReviewUserPrompt(this.state),
      () => buildDryRunTechnicalReview(this.state.draftPaper, this.state.safetyLevel),
    ));
    this.state.technicalReview = technicalReview.text;
    outputFiles.technicalReview = await this.writeArtifact("09_technical_review.md", technicalReview.text);

    const finalPaper = await this.runWorkflowStage("Final Paper", () => this.generateWithModel(
      "Final Paper",
      buildFinalPaperUserPrompt(this.state),
      () => buildDryRunFinalPaper(
        question,
        this.state.hypothesis,
        this.state.experimentDesign,
        this.state.experimentRun,
        this.state.dataAnalysis,
        this.state.literatureReview,
        this.state.safetyLevel,
      ),
    ));
    this.state.finalPaper = finalPaper.text;
    this.state.paper = finalPaper.text;
    outputFiles.finalPaper = await this.writeArtifact("10_final_paper.md", finalPaper.text);
    this.state.finalLatex = markdownToSimpleLatex(this.state.activeQuestion || "AI Discovery Research Paper", finalPaper.text);
    outputFiles.finalLatex = await this.writeArtifact("10_final_paper.tex", this.state.finalLatex);
    await this.writeSessionManifest();

    return formatGeneratedResponse(finalPaper.text, [
      "Saved end-to-end research artifacts:",
      ...Object.values(outputFiles).map((filePath) => `- ${filePath}`),
      "Execution note: generated experiment code was saved but not executed. The experiment runner used deterministic synthetic data.",
      `Sources: plan=${plan.source}, literature=${literature.source}, hypothesis=${hypothesis.source}, experiment=${experiment.source}, analysis=${analysis.source}, draft=${draft.source}, review=${technicalReview.source}, final=${finalPaper.source}`,
    ], { suppressPrimary: finalPaper.visibleStreamed });
  }

  async handleExperiment(args) {
    const brief = args || this.state.activeQuestion;
    if (!brief && !this.state.hypothesis) {
      return "Usage: /experiment <research question or experiment brief>. Run /hypothesis first when possible.";
    }
    if (brief) {
      this.state.activeQuestion = brief;
    }

    const dryRunFactory = () => buildDryRunExperiment(
      this.state.activeQuestion,
      this.state.hypothesis,
      this.state.safetyLevel,
    ).design;
    const result = await this.generateWithModel(
      "Experiment",
      buildExperimentUserPrompt(this.state, brief),
      dryRunFactory,
    );
    const generatedCode = buildExperimentCode(this.state.activeQuestion || brief);
    const outputText = result.text.includes("## Generated Experiment Code")
      ? result.text
      : `${result.text.trim()}\n\n## Generated Experiment Code\n\n\`\`\`javascript\n${generatedCode.trimEnd()}\n\`\`\`\n`;

    this.state.experimentDesign = outputText;
    this.state.experimentCode = generatedCode;
    this.enterFocusedMode("/experiment");
    const designPath = await this.writeArtifact("02_experiment_design.md", outputText);
    const codePath = await this.writeArtifact("02_experiment_code.js", generatedCode);
    const experimentRunObject = await this.runWorkflowStage("Experiment Runner", async () => (
      makeSyntheticExperimentRun(this.state.activeQuestion || brief)
    ));
    this.state.experimentRun = JSON.stringify(experimentRunObject, null, 2);
    const runJsonPath = await this.writeArtifact("02_experiment_run.json", `${this.state.experimentRun}\n`);
    const runMarkdownPath = await this.writeArtifact(
      "02_experiment_run.md",
      buildExperimentRunMarkdown(this.state.experimentRun),
    );
    await this.writeSessionManifest();
    return formatGeneratedResponse(outputText, [
      `Saved experiment design artifact: ${designPath}`,
      `Saved experiment code artifact: ${codePath}`,
      `Saved experiment run artifact: ${runJsonPath}`,
      `Saved experiment run summary: ${runMarkdownPath}`,
      "Execution note: generated experiment code was saved but not executed. The built-in deterministic synthetic runner was run and saved.",
      this.modeInstructionText(),
      `Source: ${result.source}`,
    ], { suppressPrimary: result.visibleStreamed });
  }

  async handleExperimentFollowup(instruction) {
    const result = await this.generateWithModel(
      "Experiment",
      buildExperimentFollowupUserPrompt(this.state, instruction),
      () => buildDryRunExperimentFollowup(this.state, instruction),
    );
    const generatedCode = buildExperimentCode(this.state.activeQuestion || instruction);
    const outputText = result.text.includes("## Generated Experiment Code")
      ? result.text
      : `${result.text.trim()}\n\n## Generated Experiment Code\n\n\`\`\`javascript\n${generatedCode.trimEnd()}\n\`\`\`\n`;

    this.state.experimentDesign = outputText;
    this.state.experimentCode = generatedCode;
    this.enterFocusedMode("/experiment");
    const designPath = await this.writeArtifact("02_experiment_design.md", outputText);
    const codePath = await this.writeArtifact("02_experiment_code.js", generatedCode);
    const experimentRunObject = await this.runWorkflowStage("Experiment Runner", async () => (
      makeSyntheticExperimentRun(this.state.activeQuestion || instruction)
    ));
    this.state.experimentRun = JSON.stringify(experimentRunObject, null, 2);
    const runJsonPath = await this.writeArtifact("02_experiment_run.json", `${this.state.experimentRun}\n`);
    const runMarkdownPath = await this.writeArtifact(
      "02_experiment_run.md",
      buildExperimentRunMarkdown(this.state.experimentRun),
    );
    await this.writeSessionManifest();
    return formatGeneratedResponse(outputText, [
      `Saved experiment design artifact: ${designPath}`,
      `Saved experiment code artifact: ${codePath}`,
      `Saved experiment run artifact: ${runJsonPath}`,
      `Saved experiment run summary: ${runMarkdownPath}`,
      "Execution note: generated experiment code was saved but not executed. The built-in deterministic synthetic runner was run and saved.",
      this.modeInstructionText(),
      `Source: ${result.source}`,
    ], { suppressPrimary: result.visibleStreamed });
  }

  async handleWriter(args) {
    if (!this.state.experimentDesign && !args) {
      return "Usage: /writer [paper title or instructions]. Run /experiment first when possible.";
    }
    const result = await this.generateWithModel(
      "Writer",
      buildWriterUserPrompt(this.state, args),
      () => buildDryRunPaper(
        this.state.activeQuestion,
        this.state.hypothesis,
        this.state.experimentDesign,
        this.state.experimentCode,
        this.state.experimentRun,
        this.state.dataAnalysis,
        this.state.literatureReview,
        this.state.safetyLevel,
      ),
    );
    this.state.paper = result.text;
    this.enterFocusedMode("/writer");
    const paperPath = await this.writeArtifact("03_research_paper.md", result.text);
    const latex = markdownToSimpleLatex(this.state.activeQuestion || "AI Discovery Research Paper", result.text);
    const texPath = await this.writeArtifact("03_research_paper.tex", latex);
    await this.writeSessionManifest();
    return formatGeneratedResponse(result.text, [
      `Saved paper artifact: ${paperPath}`,
      `Saved LaTeX artifact: ${texPath}`,
      this.modeInstructionText(),
      `Source: ${result.source}`,
    ], { suppressPrimary: result.visibleStreamed });
  }

  async handleWriterFollowup(instruction) {
    const result = await this.generateWithModel(
      "Writer",
      buildWriterFollowupUserPrompt(this.state, instruction),
      () => buildDryRunWriterFollowup(this.state, instruction),
    );
    this.state.paper = result.text;
    this.enterFocusedMode("/writer");
    const paperPath = await this.writeArtifact("03_research_paper.md", result.text);
    const latex = markdownToSimpleLatex(this.state.activeQuestion || "AI Discovery Research Paper", result.text);
    const texPath = await this.writeArtifact("03_research_paper.tex", latex);
    await this.writeSessionManifest();
    return formatGeneratedResponse(result.text, [
      `Saved paper artifact: ${paperPath}`,
      `Saved LaTeX artifact: ${texPath}`,
      this.modeInstructionText(),
      `Source: ${result.source}`,
    ], { suppressPrimary: result.visibleStreamed });
  }

  async runCommand(line, context = {}) {
    const { command, args, modeFollowup } = this.routeCommandLine(line);
    if (!command) {
      return "";
    }
    if (!COMMANDS.has(command)) {
      return `Unknown command: ${command}\nUse /help to list commands.`;
    }

    if (!["/status", "/history"].includes(command)) {
      this.state.history.push({
        command,
        args,
        at: new Date().toISOString(),
      });
    }

    switch (command) {
      case "/help":
        return formatHelp();
      case "/status":
        return this.statusText();
      case "/verbose":
        return this.handleVerbose(args);
      case "/model":
        if (!args) {
          return `Current model: ${this.state.model}`;
        }
        this.state.model = args.trim();
        return `Model set to: ${this.state.model}`;
      case "/safety":
        return this.handleSafety(args);
      case "/output":
        if (!args) {
          return `Output directory: ${this.state.outputDir}`;
        }
        this.state.outputDir = path.resolve(args);
        this.state.sessionDir = "";
        return `Output directory set to: ${this.state.outputDir}`;
      case "/research":
        return this.handleResearch(args, context);
      case "/new":
        this.resetScienceSession();
        return "Started a fresh science session.";
      case "/clear":
        this.output("\x1Bc");
        this.resetScienceSession();
        return "Cleared terminal view and reset visible session artifacts.";
      case "/history":
        if (!this.state.history.length) {
          return "No completed commands yet.";
        }
        return this.state.history.map((item, index) => {
          const suffix = item.args ? ` ${item.args}` : "";
          return `${index + 1}. ${item.command}${suffix} (${item.at})`;
        }).join("\n");
      case "/hypothesis":
        return modeFollowup ? this.handleHypothesisFollowup(args) : this.handleHypothesis(args, context);
      case "/experiment":
        return modeFollowup ? this.handleExperimentFollowup(args) : this.handleExperiment(args);
      case "/writer":
        return modeFollowup ? this.handleWriterFollowup(args) : this.handleWriter(args);
      case "/quit":
        return "__EXIT__";
      case "/exit":
        return this.handleExit();
      default:
        return `Unsupported command: ${command}`;
    }
  }

  completer(line) {
    const hits = [...COMMANDS.keys()].filter((command) => command.startsWith(line));
    return [hits.length ? hits : [...COMMANDS.keys()], line];
  }

  async startInteractive() {
    this.output(formatBanner(this.state.dryRun, this.state.safetyLevel));
    const rl = readline.createInterface({
      input,
      output,
      completer: this.completer.bind(this),
    });

    try {
      while (true) {
        const prompt = this.state.activeMode
          ? `ai-discovery:${this.state.activeMode.slice(1)}> `
          : "ai-discovery> ";
        const line = await rl.question(prompt);
        const response = await this.runCommand(line, {
          ask: (question) => rl.question(question),
        });
        if (response === "__EXIT__") {
          this.output("Exiting AI Discovery CLI.");
          break;
        }
        if (response) {
          this.output(response);
        }
      }
    } finally {
      rl.close();
    }
  }
}

function formatBanner(dryRun, safetyLevel = DEFAULT_SAFETY_LEVEL) {
  return [
    `${APP_NAME} v${APP_VERSION}`,
    "Science workflow: plan -> literature -> hypothesis -> experiment -> runner -> analysis -> draft -> technical review -> final paper",
    dryRun
      ? "Mode: dry-run because OPENAI_API_KEY is not set or --dry-run was passed."
      : "Mode: OpenAI Responses API.",
    `Bio/chemical safety level: ${normalizeSafetyLevel(safetyLevel)} - ${safetyProfile(safetyLevel).label}.`,
    "Type any research question to run the full workflow, /verbose to show or hide live AI output, or /help for commands.",
  ].join("\n");
}

async function main(argv = process.argv.slice(2)) {
  const options = parseGlobalArgs(argv);
  const cli = new ScienceCodexCli(options);

  if (options.help) {
    console.log(formatHelp());
    return 0;
  }

  if (options.commandLine) {
    const response = await cli.runCommand(options.commandLine);
    if (response && response !== "__EXIT__") {
      console.log(response);
    }
    return 0;
  }

  await cli.startInteractive();
  return 0;
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
}

module.exports = {
  ScienceCodexCli,
  buildDryRunExperiment,
  buildDryRunHypothesis,
  buildDryRunLiteratureReview,
  buildDryRunPaper,
  buildDryRunResearchPlan,
  buildDryRunTechnicalReview,
  buildDryRunFinalPaper,
  buildExperimentCode,
  buildExperimentRunMarkdown,
  callResponsesApi,
  extractResponseText,
  extractUrlCitations,
  formatHelp,
  formatSafetyProfile,
  makePromptPayload,
  makeSyntheticExperimentRun,
  markdownToSimpleLatex,
  normalizeCommand,
  normalizeSafetyLevel,
  parseCommandLine,
  parseGlobalArgs,
};
