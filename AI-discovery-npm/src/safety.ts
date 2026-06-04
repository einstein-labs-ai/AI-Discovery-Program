/**
 * Local, defense-only safety preflight for AI Discovery.
 *
 * This is a deterministic, on-device check that runs BEFORE any OpenAI API call
 * so that disallowed prompts fail locally without sending data to the network.
 * It is intentionally conservative and pattern-based — it is a coarse first
 * gate, not a replacement for the model's own safety training or the citation /
 * "no procedural wet-lab harm" instructions baked into the agent prompts.
 *
 * Safety levels (1 = most permissive that we offer, 5 = strictest):
 *   - Levels 1-5 all block biological / chemical mass-hazard prompts
 *     (weaponization, dangerous-pathogen or chemical-agent synthesis).
 *   - Level 5 additionally blocks jailbreak, prompt-injection,
 *     secret-exfiltration, and "ignore your system/tool policy" attempts.
 */

export const MIN_SAFETY_LEVEL = 1;
export const MAX_SAFETY_LEVEL = 5;
export const DEFAULT_SAFETY_LEVEL = 3;

export type SafetyLevel = 1 | 2 | 3 | 4 | 5;

export interface SafetyRule {
  /** Stable category label surfaced to the user when blocked. */
  category: string;
  /** Minimum safety level at which this rule is enforced. */
  minLevel: SafetyLevel;
  /** Patterns that, if any match, trigger the block. */
  patterns: RegExp[];
}

export interface SafetyVerdict {
  allowed: boolean;
  level: SafetyLevel;
  category?: string;
  reason?: string;
}

/**
 * Parse and validate a safety level from a CLI flag or env var. Returns a
 * value clamped to the legal range or throws on non-integer / out-of-range
 * input depending on `strict`.
 */
export function parseSafetyLevel(value: string | undefined): SafetyLevel {
  if (value === undefined || value === "") {
    return DEFAULT_SAFETY_LEVEL;
  }
  const parsed = Number.parseInt(value, 10);
  if (
    !Number.isInteger(parsed) ||
    parsed < MIN_SAFETY_LEVEL ||
    parsed > MAX_SAFETY_LEVEL
  ) {
    throw new Error(
      `Safety level must be an integer ${MIN_SAFETY_LEVEL}-${MAX_SAFETY_LEVEL} (got "${value}").`,
    );
  }
  return parsed as SafetyLevel;
}

/**
 * Bio/chemical mass-hazard rules — enforced at every safety level. These target
 * weaponization and the synthesis/production of dangerous pathogens or chemical
 * agents, not legitimate discussion of biology, chemistry, or defensive work.
 */
const BIO_CHEM_RULES: SafetyRule[] = [
  {
    category: "biological-weapon",
    minLevel: 1,
    patterns: [
      /\b(bio[-\s]?weapon|biological\s+weapon|bioterror|germ\s+warfare)\b/i,
      /\b(weaponi[sz]e|aerosoli[sz]e)\b[^.\n]{0,40}\b(pathogen|virus|bacteri\w*|spore|toxin|anthrax)\b/i,
      /\b(enhance|increase|engineer)\b[^.\n]{0,40}\b(transmissibility|lethality|virulence)\b[^.\n]{0,40}\b(pathogen|virus|strain)\b/i,
    ],
  },
  {
    category: "dangerous-pathogen-synthesis",
    minLevel: 1,
    patterns: [
      /\b(synthesi[sz]e|culture|grow|cultivate|produce|manufacture|isolate|propagate|amplify)\b[^.\n]{0,50}\b(anthrax|bacillus\s+anthracis|ricin|botulinum|smallpox|variola|ebola|marburg|nipah|yersinia\s+pestis|plague|select\s+agent|nerve\s+pathogen)\b/i,
      /\b(make|build|create)\b[^.\n]{0,30}\b(a\s+)?(deadly|lethal|dangerous)\s+(virus|pathogen|bioagent)\b/i,
    ],
  },
  {
    category: "chemical-weapon",
    minLevel: 1,
    patterns: [
      /\b(chemical\s+weapon|nerve\s+agent|nerve\s+gas|chemical\s+warfare)\b/i,
      /\b(synthesi[sz]e|produce|manufacture|make|prepare|cook)\b[^.\n]{0,50}\b(sarin|soman|tabun|vx\b|novichok|mustard\s+(gas|agent)|chlorine\s+gas|phosgene|hydrogen\s+cyanide\s+gas)\b/i,
    ],
  },
];

/**
 * Stricter-level rules — enforced only at level 5. These target attempts to
 * subvert the assistant's own instructions or extract its secrets, rather than
 * real-world physical hazards.
 */
const STRICT_RULES: SafetyRule[] = [
  {
    category: "jailbreak",
    minLevel: 5,
    patterns: [
      /\b(jailbreak|do\s+anything\s+now|\bDAN\b\s+mode|developer\s+mode\s+(enabled|on)|unfiltered\s+mode)\b/i,
      /\b(act|behave|roleplay)\b[^.\n]{0,40}\b(without|no)\b[^.\n]{0,20}\b(restrictions|guardrails|rules|safety|filters?)\b/i,
      /\bpretend\b[^.\n]{0,40}\byou\s+(have\s+no|are\s+not\s+bound\s+by)\b[^.\n]{0,30}\b(rules|policy|guidelines|restrictions)\b/i,
    ],
  },
  {
    category: "prompt-injection",
    minLevel: 5,
    patterns: [
      /\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(all\s+|the\s+|your\s+|any\s+|previous\s+|prior\s+|above\s+)*(instructions|prompt|context|rules|guidelines)\b/i,
      /\bsystem\s+prompt\s+(override|injection)\b/i,
      /(^|\n)\s*(new|updated)\s+(system\s+)?instructions?\s*:/i,
      /\byou\s+are\s+now\b[^.\n]{0,40}\b(unrestricted|jailbroken|a\s+different)\b/i,
    ],
  },
  {
    category: "secret-exfiltration",
    minLevel: 5,
    patterns: [
      /\b(reveal|show|print|dump|expose|leak|exfiltrate|repeat|output)\b[^.\n]{0,40}\b(your\s+)?(system\s+prompt|hidden\s+instructions|api\s+key|secret\s+key|credentials?|passwords?|env(ironment)?\s+variables?|access\s+token)\b/i,
      /\bwhat\s+(is|are)\s+your\s+(system\s+prompt|hidden\s+instructions|api\s+key|secrets?)\b/i,
    ],
  },
  {
    category: "policy-evasion",
    minLevel: 5,
    patterns: [
      /\b(ignore|bypass|disable|turn\s+off|circumvent|evade)\b[^.\n]{0,40}\b(the\s+|your\s+|any\s+)*(system|tool|safety|content)\b[^.\n]{0,20}\b(policy|policies|filter|filters|guardrails?|restrictions?)\b/i,
    ],
  },
];

const ALL_RULES: readonly SafetyRule[] = [...BIO_CHEM_RULES, ...STRICT_RULES];

const CATEGORY_GUIDANCE: Record<string, string> = {
  "biological-weapon":
    "requests to create, weaponize, or enhance biological hazards are refused at every safety level",
  "dangerous-pathogen-synthesis":
    "requests to synthesize, culture, or produce dangerous pathogens are refused at every safety level",
  "chemical-weapon":
    "requests to make or weaponize chemical agents are refused at every safety level",
  jailbreak: "jailbreak attempts are blocked at safety level 5",
  "prompt-injection": "prompt-injection attempts are blocked at safety level 5",
  "secret-exfiltration":
    "attempts to extract system prompts or secrets are blocked at safety level 5",
  "policy-evasion":
    "attempts to disable system/tool policy are blocked at safety level 5",
};

/**
 * Run the local preflight on a piece of user text at a given safety level.
 * Returns the first matching rule (if any) so the caller can fail fast.
 */
export function runSafetyPreflight(
  level: SafetyLevel,
  text: string,
): SafetyVerdict {
  const haystack = text ?? "";
  for (const rule of ALL_RULES) {
    if (level < rule.minLevel) {
      continue;
    }
    if (rule.patterns.some((pattern) => pattern.test(haystack))) {
      return {
        allowed: false,
        level,
        category: rule.category,
        reason:
          CATEGORY_GUIDANCE[rule.category] ??
          `blocked by safety category ${rule.category}`,
      };
    }
  }
  return { allowed: true, level };
}

/** Short, user-facing description of what a given level enforces. */
export function describeSafetyLevel(level: SafetyLevel): string {
  const base =
    "blocks biological/chemical mass-hazard prompts (weaponization and dangerous-agent synthesis)";
  if (level >= 5) {
    return `level ${level}: ${base}, plus jailbreak, prompt-injection, secret-exfiltration, and policy-evasion attempts`;
  }
  return `level ${level}: ${base}`;
}

/** One-line block message suitable for stderr / chat output. */
export function formatBlockMessage(verdict: SafetyVerdict): string {
  return `Blocked by local safety preflight (level ${verdict.level}, category: ${verdict.category}). This request was not sent to the model: ${verdict.reason}.`;
}
