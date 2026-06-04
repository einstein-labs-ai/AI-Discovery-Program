/**
 * Text-only model allowlist for AI Discovery.
 *
 * The CLI and chat deliberately accept only this curated set of OpenAI text
 * models. Image, audio, realtime, and embedding models are intentionally
 * excluded — every workflow here produces or reasons over text artifacts, so a
 * narrow allowlist keeps `--model`, `--manager-model`, `--specialist-model`, and
 * the chat `/model` command from silently routing to an unsupported endpoint.
 */

export interface ModelInfo {
  /** Canonical OpenAI model ID passed to the Agents SDK. */
  id: string;
  /** Human-friendly display label, e.g. "GPT-5.5 Pro". */
  label: string;
  /** One-line description shown in `/models` and `--help`. */
  description: string;
}

/**
 * The complete, ordered catalog of allowed text models. Order is used for the
 * numbered `/model <n>` chat shortcut and the `/models` listing.
 */
export const MODEL_CATALOG: readonly ModelInfo[] = [
  {
    id: "gpt-5.5",
    label: "GPT-5.5",
    description: "Flagship text reasoning model. Default.",
  },
  {
    id: "gpt-5.5-pro",
    label: "GPT-5.5 Pro",
    description: "Highest-effort GPT-5.5 variant for the hardest reasoning.",
  },
  {
    id: "gpt-5.4",
    label: "GPT-5.4",
    description: "Previous-generation flagship text model.",
  },
  {
    id: "gpt-5.4-pro",
    label: "GPT-5.4 Pro",
    description: "Highest-effort GPT-5.4 variant.",
  },
  {
    id: "gpt-5.4-mini",
    label: "GPT-5.4 mini",
    description: "Faster, cheaper GPT-5.4 for lighter tasks.",
  },
  {
    id: "gpt-5.4-nano",
    label: "GPT-5.4 nano",
    description: "Smallest, fastest GPT-5.4 for simple tasks.",
  },
] as const;

/** Canonical IDs in catalog order. */
export const MODEL_IDS: readonly string[] = MODEL_CATALOG.map((m) => m.id);

const CANONICAL_BY_ID = new Map(MODEL_CATALOG.map((m) => [m.id, m]));

/**
 * Reduce a user-supplied model string to a comparable canonical key.
 *
 * Handles common display aliases such as `GPT-5.5 Pro`, `GPT 5.4 mini`, and
 * `gpt_5.4_nano` by lowercasing, collapsing whitespace/underscores to single
 * hyphens, and inserting the hyphen in a bare `gpt5.5` form. The result is only
 * a lookup key; it is not assumed to be a valid model on its own.
 */
function canonicalizeKey(input: string): string {
  return input
    .trim()
    .toLowerCase()
    .replace(/[\s_]+/g, "-")
    .replace(/^gpt(?=\d)/, "gpt-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}

/**
 * Resolve an arbitrary model string to a canonical catalog ID, or `undefined`
 * if it is not an allowed text model.
 */
export function normalizeModel(input: string): string | undefined {
  const key = canonicalizeKey(input);
  return CANONICAL_BY_ID.has(key) ? key : undefined;
}

/**
 * Resolve a model string to a canonical catalog ID or throw a descriptive
 * error listing the allowed models. Used by every model-accepting input
 * (`--model`, `--manager-model`, `--specialist-model`, `OPENAI_MODEL`, and the
 * chat `/model` command).
 */
export function resolveModel(input: string, source = "model"): string {
  const resolved = normalizeModel(input);
  if (!resolved) {
    throw new Error(
      `Unknown ${source} "${input}". Allowed text models: ${MODEL_IDS.join(", ")}.`,
    );
  }
  return resolved;
}

/** Look up display metadata for a canonical model ID. */
export function modelInfo(id: string): ModelInfo | undefined {
  return CANONICAL_BY_ID.get(id);
}

/** Pretty label for a model ID, falling back to the raw ID. */
export function modelLabel(id: string): string {
  return CANONICAL_BY_ID.get(id)?.label ?? id;
}

/** Multi-line, numbered catalog listing for `/models` and help output. */
export function formatModelCatalog(activeId?: string): string {
  return MODEL_CATALOG.map((m, index) => {
    const marker = m.id === activeId ? "*" : " ";
    return `${marker} ${index + 1}. ${m.label} (${m.id}) — ${m.description}`;
  }).join("\n");
}

/**
 * Resolve a `/model` argument that may be either a model name/alias or a
 * 1-based catalog index. Throws on anything unrecognized.
 */
export function resolveModelSelector(selector: string): string {
  const trimmed = selector.trim();
  if (/^\d+$/.test(trimmed)) {
    const index = Number.parseInt(trimmed, 10) - 1;
    const picked = MODEL_CATALOG[index];
    if (!picked) {
      throw new Error(
        `Model number ${trimmed} is out of range (1-${MODEL_CATALOG.length}).`,
      );
    }
    return picked.id;
  }
  return resolveModel(trimmed, "model");
}
