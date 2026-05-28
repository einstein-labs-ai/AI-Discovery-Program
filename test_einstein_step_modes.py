import os
import tempfile
import unittest

import EinsteinResearch as research


class EinsteinStepModeTests(unittest.TestCase):
    def test_sanitize_suggestion_prompt_preserves_long_complete_prompt(self):
        long_prompt = "Research objective: " + ("validate evidence carefully. " * 180)

        sanitized = research._sanitize_suggestion_prompt(f"Suggested prompt:\n{long_prompt}")

        self.assertEqual(sanitized, long_prompt.strip())
        self.assertGreater(len(sanitized), 2500)

    def test_suggest_research_prompt_uses_configured_full_token_budget(self):
        original_runner = research._run_agent_with_fallback
        original_budget = os.environ.get("VIBE_SUGGEST_MAX_TOKENS")
        expected_prompt = "Research objective: build a complete validated study plan."

        def fake_runner(agent, prompt, **kwargs):
            self.assertIn("Return one complete prompt", prompt)
            self.assertEqual(agent.model_settings.max_tokens, 12345)
            return research.CLIInputSuggestion(prompt=f"Prompt:\n{expected_prompt}")

        os.environ["VIBE_SUGGEST_MAX_TOKENS"] = "12345"
        research._run_agent_with_fallback = fake_runner
        try:
            suggested = research._suggest_research_prompt(
                "rough quantum battery experiment",
            )

            self.assertEqual(suggested, expected_prompt)
        finally:
            research._run_agent_with_fallback = original_runner
            if original_budget is None:
                os.environ.pop("VIBE_SUGGEST_MAX_TOKENS", None)
            else:
                os.environ["VIBE_SUGGEST_MAX_TOKENS"] = original_budget

    def test_step_action_parses_ask_mode_exit_and_quit(self):
        self.assertEqual(research._parse_step_action("/ask"), ("ask_mode", ""))
        self.assertEqual(
            research._parse_step_action("/ask explain this"),
            ("ask", "explain this"),
        )
        self.assertEqual(research._parse_step_action("/exit"), ("exit", ""))
        self.assertEqual(research._parse_step_action("/quit"), ("quit", ""))

    def test_rewrite_step_artifact_updates_state_and_file(self):
        original_runner = research._run_agent_with_fallback

        def fake_runner(agent, prompt, **kwargs):
            self.assertIn("Active artifact:\nPlan", prompt)
            self.assertIn("add success criteria", prompt)
            return "## Revised Plan\n\nAdd measurable success criteria."

        research._run_agent_with_fallback = fake_runner
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                step_outputs = {"Plan": "## Plan\n\nOriginal content."}
                step_feedback = {}
                output_files = {}

                revised, saved_paths = research._rewrite_step_artifact(
                    step_title="Plan",
                    message="add success criteria",
                    question="Can X improve Y?",
                    data_note="No data provided.",
                    step_outputs=step_outputs,
                    step_feedback=step_feedback,
                    agents={"step_artifact_rewriter": object()},
                    output_dir=output_dir,
                    output_files=output_files,
                )

                self.assertEqual(step_outputs["Plan"], revised)
                self.assertIn("/ask revision: add success criteria", step_feedback["Plan"])
                self.assertEqual(len(saved_paths), 1)
                self.assertEqual(os.path.basename(saved_paths[0]), "01_plan.md")
                self.assertEqual(output_files["01_plan.md"], saved_paths[0])
                with open(saved_paths[0], encoding="utf-8") as handle:
                    self.assertEqual(handle.read(), revised)
        finally:
            research._run_agent_with_fallback = original_runner

    def test_academic_latex_normalizer_ragged_wraps_table_columns(self):
        latex = r"""\documentclass{article}
\usepackage{tabularx}
\begin{document}
\begin{table}[H]
\centering
\begin{tabularx}{\textwidth}{p{0.12\textwidth} X p{0.18\textwidth}}
\toprule
Long heading & Narrative cell & Notes \\
\midrule
Alpha & Long text with enough words to trigger poor narrow-column justification & Beta \\
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[H]
\centering
\begin{tabular}{llp{0.20\textwidth}}
\toprule
A & B & Long note \\
\bottomrule
\end{tabular}
\end{table}
\end{document}"""

        normalized = research._ensure_academic_paper_latex(latex)

        self.assertIn(r"\usepackage{adjustbox}", normalized)
        self.assertIn(r"\usepackage{ragged2e}", normalized)
        self.assertIn(r"\setlength{\tabcolsep}{4pt}", normalized)
        self.assertIn(r"\renewcommand{\arraystretch}{1.12}", normalized)
        self.assertIn(
            r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{0.12\textwidth} >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}p{0.18\textwidth}}",
            normalized,
        )
        self.assertIn(r"\begin{adjustbox}{max width=\textwidth}", normalized)
        self.assertIn(
            r"\begin{tabular}{ll>{\raggedright\arraybackslash}p{0.20\textwidth}}",
            normalized,
        )

        valid, message = research._validate_latex_table_layout(normalized)
        self.assertTrue(valid, message)

    def test_latex_table_layout_normalizer_is_idempotent(self):
        latex = r"""\documentclass{article}
\begin{document}
\begin{table}[H]
\centering
\begin{tabular}{lp{0.25\textwidth}}
\toprule
Name & Explanation \\
\bottomrule
\end{tabular}
\end{table}
\end{document}"""

        normalized_once = research._ensure_academic_paper_latex(latex)
        normalized_twice = research._ensure_academic_paper_latex(normalized_once)

        self.assertEqual(normalized_once, normalized_twice)
        self.assertEqual(normalized_once.count(r"\begin{adjustbox}{max width=\textwidth}"), 1)
        self.assertEqual(normalized_once.count(r"\usepackage{ragged2e}"), 1)


if __name__ == "__main__":
    unittest.main()
