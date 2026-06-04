export const HYPOTHESIS_OUTPUT_SCHEMA = `title: Short descriptive name

research_question: >
  What question is this hypothesis trying to answer?

hypothesis_statement: >
  If X is true or if we intervene on X, then Y should happen because of mechanism Z.

domain:
  field: biology / chemistry / medicine / physics / AI / etc.
  system: organism, model, dataset, material, algorithm, etc.

background_evidence:
  supporting_evidence:
    - claim:
      source:
      strength: weak / moderate / strong
  conflicting_evidence:
    - claim:
      source:
      strength: weak / moderate / strong

mechanism:
  proposed_causal_chain:
    - Step 1
    - Step 2
    - Step 3
  key_assumptions:
    - Assumption 1
    - Assumption 2

predictions:
  primary_prediction: >
    What should be observed if the hypothesis is correct?
  secondary_predictions:
    - Additional expected outcome
    - Another expected outcome
  falsifying_observation: >
    What result would make the hypothesis unlikely?

test_plan:
  experiment_or_analysis: >
    How would this be tested?
  required_data:
    - Dataset, measurement, simulation, or experiment needed
  controls:
    positive_control:
    negative_control:
  comparison_baseline:
  success_criteria:

confounders_and_alternatives:
  possible_confounders:
    - Confounder 1
    - Confounder 2
  alternative_explanations:
    - Alternative mechanism 1
    - Alternative mechanism 2

feasibility:
  required_resources:
    - Data
    - Tools
    - Expertise
  estimated_cost: low / medium / high
  estimated_time: short / medium / long
  risk_level: low / medium / high

evaluation:
  novelty: 1-5
  plausibility: 1-5
  testability: 1-5
  impact: 1-5
  feasibility: 1-5
  overall_priority: 1-5

uncertainty:
  confidence_score: 0.0-1.0
  main_uncertainty: >
    What is the biggest thing we do not know?

status:
  state: proposed / under_review / testing / supported / weakened / rejected
  last_updated:`;

export const HYPOTHESIS_SCHEMA_INSTRUCTIONS = [
  "The final answer must be only a YAML document matching this schema exactly, with the keys in this order. Do not wrap it in Markdown fences, and do not add commentary, summaries, citations sections, or extra fields outside the schema.",
  "",
  HYPOTHESIS_OUTPUT_SCHEMA,
  "",
  "Field requirements:",
  "- Fill every top-level key in the schema, preserving nested key names and order.",
  "- Use the enum values shown in the schema for evidence strength, cost, time, risk, and status state.",
  "- Score novelty, plausibility, testability, impact, feasibility, and overall_priority as integers from 1 to 5.",
  "- Score confidence_score as a decimal from 0.0 to 1.0.",
  "- For last_updated, use an ISO date (YYYY-MM-DD) when the run date is known; otherwise leave the value empty.",
  "- Keep the hypothesis causal and falsifiable: make the X, Y, and mechanism Z relationship explicit.",
  "- Include supporting and conflicting evidence at the claim level. Each evidence item must include a source when verified; do not invent citations, URLs, DOIs, authors, or papers.",
  "- If evidence cannot be verified with available tools, mark the source as unverified or unavailable and lower the evidence strength accordingly.",
  "- Include at least one falsifying observation, at least one plausible confounder, and at least one alternative explanation.",
  "- Keep experiments and analyses safe, non-harmful, and appropriate for computational, observational, or review settings unless the user has supplied an approved real-world protocol context.",
  "",
  "Scientific safety and validity:",
  "- Treat biological, chemical, medical, and physical-world domains as safety-sensitive. Do not provide procedural wet-lab, clinical, chemical synthesis, or harmful operational instructions.",
  "- Preserve units, assumptions, provenance, uncertainty, and limitations where they affect the hypothesis or test plan.",
  "- Prefer bounded, reproducible, source-grounded analysis over speculative claims.",
].join("\n");
