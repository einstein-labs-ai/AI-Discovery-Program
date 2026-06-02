# AI Discovery Program
Einsteinian Labs helps researchers turn ideas into discovery by using end-to-end AI automation—from literature review to report generation—while keeping the process rigorous, measurable, and deeply human at its core

# NPM Implementation
Go to [AI Discovery NPM](https://github.com/einstein-labs-ai/AI-Discovery-Program/tree/main/AI-discovery-npm)

# Einstein Research Console — How to Use Guide

Einsteinian Labs helps researchers turn ideas into discovery using end-to-end AI automation, from literature review to hypothesis generation, experiment design, data analysis, and report generation.

This guide explains how to install, run, and use the program step by step.

---

## 1. Program Filename

The main program file must be named:

```text
EinsteinResearch.py
```

Use this exact filename in all terminal commands.

---

## 2. What the Program Does

`EinsteinResearch.py` is an AI research console that helps users move from a research question to a structured report.

The program can:

1. Help plan a research project.
2. Search for background information and supporting sources.
3. Generate research hypotheses.
4. Design experiments or simulations.
5. Run a prototype experiment or analysis.
6. Analyze experiment results.
7. Write a conclusion.
8. Generate a LaTeX research report.
9. Optionally compile the LaTeX report into a PDF.
10. Run a local browser-based chat interface.
11. Launch a separate lab research workflow, if configured.

---

## 3. Recommended Use Cases

Use `EinsteinResearch.py` when you want to:

- Start a new scientific research project.
- Turn a broad idea into a testable hypothesis.
- Create an experiment plan.
- Analyze research data.
- Generate an academic-style LaTeX report.
- Produce a PDF research report.
- Run repeatable AI-assisted research workflows.
- Use a local web chat interface for research interaction.

---

## 4. Before You Start

Make sure you have:

1. Python installed.
2. The file named `EinsteinResearch.py` in your project folder.
3. An OpenAI API key, if the program requires OpenAI model access.
4. The required Python packages installed.
5. A terminal or command prompt open in the folder that contains `EinsteinResearch.py`.

---

## 5. Install Required Packages

Open your terminal in the project folder and run:

```bash
pip install openai openai-agents pydantic
```

For optional session memory support, run:

```bash
pip install sqlalchemy aiosqlite
```

For full installation with both required and optional packages, run:

```bash
pip install openai openai-agents pydantic sqlalchemy aiosqlite
```

---

## 5a. Export an API key
For Windows:
```
setx OPENAI_API_KEY "your_api_key_here"
```

For Linux/MacOS: 
```
export OPENAI_API_KEY="your_api_key_here"
```

### For Lab Research & Search
For Windows:
```
setx PERPLEXITY_API_KEY "your_api_key_here"
```

For Linux/MacOS: 
```
export PERPLEXITY_API_KEY="your_api_key_here"
```

## 6. Optional: Install a LaTeX Compiler

To generate a PDF from the LaTeX report, install at least one of the following tools:

- `tectonic`
- `latexmk`
- `pdflatex`

If you do not install a LaTeX compiler, the program can still generate the `.tex` file, but PDF compilation may fail.

---

## 7. Recommended Folder Setup

A simple project folder should look like this:

```text
project-folder/
├── EinsteinResearch.py
├── index.html                  # optional, only needed for web chat mode
├── data/                       # optional, for input datasets
└── outputs/                    # optional, for saved research outputs
```

The only required file for the main program is:

```text
EinsteinResearch.py
```

---

## 8. Start the Program

To open the interactive startup menu, run:

```bash
python EinsteinResearch.py interactive
```

You should see a menu similar to this:

```text
+--------------------------------------------------------------------------+
|  Active model: gpt-5.5                                                   |
|                                                                          |
|  [1] Core Research Pipeline                                              |
|  [2] Lab Research                                                        |
|  [3] Web Chat Server                                                     |
|  [0] Exit                                                                |
+--------------------------------------------------------------------------+
> Choose an option [0-3] (or /model <name>):
```

---

## 9. Use the Core Research Pipeline

The Core Research Pipeline is the main workflow for AI-assisted scientific discovery.

### Step 1: Start Interactive Mode

Run:

```bash
python EinsteinResearch.py interactive
```

### Step 2: Select the Core Research Pipeline

At the menu, type:

```text
1
```

Then press Enter.

### Step 3: Enter Your Research Question

When prompted, enter a research question, such as:

```text
Does retrieval augmentation improve factual accuracy in biotech question-answering systems?
```

A strong research question should be:

- Clear
- Specific
- Testable
- Connected to a measurable outcome

### Step 4: Add Experiment Data or a Data File

When prompted, you can provide experiment data directly or enter a file path.

Example data-file path:

```text
./data/results.csv
```

Example text input:

```text
sample_id,condition,score
1,A,0.82
2,B,0.76
3,A,0.88
4,B,0.73
```

If you do not have data, press Enter. The program can still generate a prototype experiment or synthetic example, depending on its configuration.

### Step 5: Review Each Research Stage

The program walks through the research workflow one stage at a time:

1. Plan
2. Background Research
3. Hypothesis
4. Experiment Design
5. Experiment Run Output
6. Data Analysis
7. Conclusion
8. Search Plan
9. Search Sources
10. LaTeX Report

At each stage, you can review the output before continuing.

### Step 6: Use Stage Commands

During the step-by-step workflow, you may see a prompt like this:

```text
> [Plan] Enter=next | text=/note | /ask | a=auto | q=quit:
```

Use these controls:

| Command | What It Does |
|---|---|
| Press Enter | Move to the next stage. |
| Type a note | Add your own note to the current stage. |
| `/ask` | Ask the AI a question about the current stage. |
| `a` | Automatically run the remaining stages. |
| `q` | Quit the workflow. |

### Step 7: Review the Final Report

At the end of the pipeline, the program generates a LaTeX report. If PDF generation is enabled and a LaTeX compiler is installed, it may also generate a PDF.

---

## 10. Save Research Outputs

To save all outputs into a folder, use the `--save` option.

Example:

```bash
python EinsteinResearch.py --save outputs interactive
```

This creates a timestamped run directory inside the `outputs` folder.

Example output structure:

```text
outputs/
└── run_YYYYMMDD_HHMMSS/
    ├── 01_plan.md
    ├── 01b_background_research.md
    ├── 01b_background_sources.txt
    ├── 02_hypothesis.md
    ├── 03_experiment_design.md
    ├── 04_experiment_run.md
    ├── 05_data_analysis.md
    ├── 06_conclusion.md
    ├── 00_search_plan.md
    ├── 00_search_summaries.md
    ├── 00_sources.txt
    └── 07_report.tex
```

---

## 11. Choose a Model

To choose a model when starting the program, use:

```bash
python EinsteinResearch.py --model gpt-5.5 interactive
```

To save outputs and choose a model at the same time, use:

```bash
python EinsteinResearch.py --save outputs --model gpt-5.5 interactive
```

Inside the interactive menu, you can switch models by typing:

```text
/model gpt-5.5-pro
```

To view the current model and recommended models, type:

```text
/model
```

---

## 12. Ask the AI to Suggest a Research Topic

Inside interactive mode, you can ask the AI to complete or improve a partial research idea.

Example:

```text
/suggest Research on nanotechnology to heal humans
```

The program will suggest a stronger research prompt. After the suggestion appears, you can:

1. Press Enter to use the suggested prompt.
2. Type your own replacement question.
3. Type `/cancel` to cancel the suggestion.

---

## 13. Run the Automated Pipeline

Automated mode runs the research pipeline without pausing after every stage. Use this mode for repeatable runs, scripts, or batch workflows.

### Run with a Research Question Only

```bash
python EinsteinResearch.py auto --question "Does retrieval augmentation improve factual accuracy in biotech QA systems?"
```

### Run with Inline Data

```bash
python EinsteinResearch.py auto \
  --question "Does retrieval augmentation improve factual accuracy in biotech QA systems?" \
  --data "sample_id,condition,score\n1,A,0.82\n2,B,0.76"
```

### Run with a Data File

```bash
python EinsteinResearch.py auto \
  --question "Does retrieval augmentation improve factual accuracy in biotech QA systems?" \
  --data-file ./data/results.csv
```

### Run with a Data File and Save Outputs

```bash
python EinsteinResearch.py --save outputs auto \
  --question "Does retrieval augmentation improve factual accuracy in biotech QA systems?" \
  --data-file ./data/results.csv
```

### Pause After Each Stage in Automated Mode

```bash
python EinsteinResearch.py auto \
  --question "Your research question here" \
  --pause
```

### Skip PDF Generation

```bash
python EinsteinResearch.py --no-pdf auto \
  --question "Your research question here"
```

---

## 14. Convert an Existing LaTeX File to PDF

Use `latex2pdf` mode when you already have a `.tex` file and only want to compile it into a PDF.

### Basic Conversion

```bash
python EinsteinResearch.py latex2pdf --tex-file ./report.tex
```

### Save the Compiled Files to a Specific Folder

```bash
python EinsteinResearch.py latex2pdf \
  --tex-file ./report.tex \
  --output-dir ./compiled
```

If PDF conversion succeeds, the PDF is saved next to the `.tex` file or inside the folder specified by `--output-dir`.

---

## 15. Run the Local Web Chat Server

Use serve mode when you want to interact with the research assistant through a browser interface.

### Step 1: Make Sure `index.html` Exists

Your project folder should contain:

```text
index.html
```

### Step 2: Start the Server

```bash
python EinsteinResearch.py serve
```

### Step 3: Open the Browser Interface

Open your browser and go to the local address shown by the program.

Common default address:

```text
http://localhost:8000
```

### Step 4: Use the Chat API

The server exposes these endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Opens the browser interface. |
| `/health` | GET | Checks whether the server is running. |
| `/api/chat` | POST | Sends a chat message to the backend. |

Example JSON request for `/api/chat`:

```json
{
  "message": "Summarize the tradeoffs of wet-lab automation for biotech startups.",
  "history": [
    {"role": "user", "content": "I am researching biotech automation."},
    {"role": "assistant", "content": "Got it. What aspect are you exploring?"}
  ],
  "model": "gpt-5.5",
  "session_id": "demo_session_1"
}
```

### Use a Custom Host, Port, or HTML File

```bash
python EinsteinResearch.py serve \
  --host 127.0.0.1 \
  --port 8000 \
  --index index.html
```

---

## 16. Run Lab Research Mode

Lab mode launches the separate Lab Research workflow from the main `EinsteinResearch.py` console. This mode is useful when you want a focused research utility for literature review, data analysis, reference discovery, and saving results to LaTeX or PDF.

### Step 1: Confirm the Lab Workflow File Exists

Lab mode expects the lab workflow script to be available in the same project folder. The expected supporting file is:

```text
Perplexity-search.py
```

Your folder should look similar to this:

```text
project-folder/
├── EinsteinResearch.py
├── Perplexity-search.py        # required for Lab Research mode
├── index.html                  # optional, only needed for web chat mode
├── data/                       # optional, for input datasets
└── outputs/                    # optional, for saved research outputs
```

### Step 2: Start Lab Research Mode

Run this command from the project folder:

```bash
python EinsteinResearch.py lab
```

If the lab workflow file is missing, the program will show an error message and stop.

### Step 3: Use the Lab Research CLI Interface

After Lab Research mode starts, you should see a command-line menu similar to this:

```text
Einstein Search CLI
Model: sonar-pro
Type /quit at any prompt to exit.

+---------------------------------------+
| 1) Literature Review                  |
| 2) Data Analysis (file)               |
| 3) Find References                    |
| 4) Save Last Result to LaTeX + PDF    |
| /quit) Exit                           |
+---------------------------------------+
> Choose 1, 2, 3, 4, or /quit:
```

Use the menu numbers to select the workflow you want.

| Option | Purpose | When to Use It |
|---|---|---|
| `1` | Literature Review | Use this to summarize a research topic, identify themes, and create a literature-style overview. |
| `2` | Data Analysis (file) | Use this to load and analyze a local CSV, TSV, JSON, or text file. |
| `3` | Find References | Use this to search for relevant references, papers, guides, or supporting sources. |
| `4` | Save Last Result to LaTeX + PDF | Use this after generating a result that you want to save as a formal report. |
| `/quit` | Exit | Use this to close the Lab Research CLI. |

---

## 17. Lab Research CLI Interface and Examples

This section shows exactly how to use each Lab Research CLI option.

### Example 1: Run a Literature Review

Choose option `1` when you want the lab workflow to generate a literature review for a topic.

```text
> Choose 1, 2, 3, 4, or /quit: 1
> Literature review topic/query: AI Discovery Engine for R&D
> Specific focus (optional, press Enter to skip): R&D
> Output style [concise bullet summary]:
```

What happens next:

1. The program searches or synthesizes information about the topic.
2. It focuses the review around your optional focus area, if provided.
3. It returns a literature-style summary using the selected output style.

Use this option when you need a fast research overview before building hypotheses or designing experiments.

### Example 2: Analyze a Data File

Choose option `2` when you want to analyze a local data file.

```text
> Choose 1, 2, 3, 4, or /quit: 2
> Path to data file (csv/tsv/json/txt): research_ai.csv
```

The program loads the file and shows a preview similar to this:

```text
Loaded file preview:

File: C:\Users\Thomas_Yiu\Downloads\V-agent\webpage\research_ai.csv
Detected format: CSV
Columns: Table 5. Overview of research methods
Rows excluding header: 11
Preview rows:
1. Research method | Approach | Percent
2. Qualitative | Survey and interviews | 14.20%
3. Quantitative | Experiment | 49.40%
4. Conceptual | No methodology | 24.70%
5. Other | Other | 11.70%
```

Then the program asks for your analysis objective:

```text
> Analysis objective [Identify trends, anomalies, and recommendations]: Identify trends
```

The output may include:

1. Key findings.
2. Trends in the data.
3. Anomalies or unusual patterns.
4. Recommendations for interpretation.
5. Possible next research steps.

Use this option when you already have a dataset and want the lab workflow to summarize or interpret it.

### Example 3: Find References

Choose option `3` when you need references for a research topic.

```text
> Choose 1, 2, 3, 4, or /quit: 3
> Reference query/topic: Find references for AI research for R&D for science
> Max references [8]:
```

If you press Enter at `Max references [8]`, the program uses the default value of 8 references.

The output may look similar to this:

```text
References (8):

1. Artificial Intelligence Resources: AI Tools for Research
   URL: https://guides.library.georgetown.edu/ai/tools
   Snippet: AI tools for research can help discover sources for literature review, synthesize scholarly information, and save researchers time.

2. Additional reference title
   URL: https://example.com/reference
   Snippet: Short summary of the reference.
```

Use this option when you need source material for a report, literature review, hypothesis rationale, or research proposal.

### Example 4: Save the Last Result to LaTeX and PDF

Choose option `4` after you have generated a literature review, data analysis, or reference result that you want to save.

```text
> Choose 1, 2, 3, 4, or /quit: 4
```

The program attempts to save the latest generated result as a LaTeX file and, if a LaTeX compiler is available, compile it into a PDF.

Use this option after you are satisfied with the latest result in the Lab Research CLI.

### Example 5: Exit Lab Research Mode

At any menu prompt, type:

```text
/quit
```

This exits the Lab Research CLI and returns control to your terminal.

---

## 18. Example Quick Start

Use this sequence for a first successful run.

### Step 1: Install Packages

```bash
pip install openai openai-agents pydantic sqlalchemy aiosqlite
```

### Step 2: Start the Program

```bash
python EinsteinResearch.py --save outputs interactive
```

### Step 3: Select the Core Research Pipeline

At the menu, type:

```text
1
```

### Step 4: Enter a Research Question

Example:

```text
Can AI-assisted literature review improve the speed and quality of early-stage scientific hypothesis generation?
```

### Step 5: Provide Data or Press Enter

If you have data, enter a file path:

```text
./data/research_results.csv
```

If you do not have data, press Enter.

### Step 6: Review Each Stage

Press Enter to move through each stage, or type:

```text
a
```

to automatically run the remaining stages.

### Step 7: Open the Output Folder

After the run finishes, open the `outputs` folder and review the generated Markdown, source, LaTeX, and PDF files.

---

## 19. Command Reference

| Goal | Command |
|---|---|
| Open interactive mode | `python EinsteinResearch.py interactive` |
| Save outputs | `python EinsteinResearch.py --save outputs interactive` |
| Choose a model | `python EinsteinResearch.py --model gpt-5.5 interactive` |
| Run automated mode | `python EinsteinResearch.py auto --question "Your question"` |
| Use a data file | `python EinsteinResearch.py auto --question "Your question" --data-file ./data/results.csv` |
| Skip PDF generation | `python EinsteinResearch.py --no-pdf auto --question "Your question"` |
| Convert LaTeX to PDF | `python EinsteinResearch.py latex2pdf --tex-file ./report.tex` |
| Start web server | `python EinsteinResearch.py serve` |
| Start lab mode | `python EinsteinResearch.py lab` |

---

## 20. Output Files Explained

When `--save` is used, the program may generate these files:

| File | Purpose |
|---|---|
| `01_plan.md` | Research plan and workflow structure. |
| `01b_background_research.md` | Background research summary. |
| `01b_background_sources.txt` | Sources used during background research. |
| `02_hypothesis.md` | Primary hypothesis, null hypothesis, and predictions. |
| `03_experiment_design.md` | Experiment design, variables, controls, and method. |
| `04_experiment_run.md` | Experiment or simulation output. |
| `05_data_analysis.md` | Analysis of results. |
| `06_conclusion.md` | Final conclusion and interpretation. |
| `00_search_plan.md` | Search strategy. |
| `00_search_summaries.md` | Search result summaries. |
| `00_sources.txt` | Final source list. |
| `07_report.tex` | Final LaTeX research report. |

The program may also create a timestamped LaTeX file, such as:

```text
research_report_YYYYMMDD.HHMMSS.tex
```

If PDF generation succeeds, a PDF version of the report is also created.

---

## 21. Troubleshooting

### Problem: `python` Command Is Not Found

Try:

```bash
python3 EinsteinResearch.py interactive
```

or reinstall Python and make sure it is added to your system path.

### Problem: Required Package Is Missing

Install the required packages again:

```bash
pip install openai openai-agents pydantic sqlalchemy aiosqlite
```

### Problem: PDF Generation Fails

Possible causes:

1. No LaTeX compiler is installed.
2. The generated `.tex` file contains a syntax issue.
3. Required LaTeX packages are missing.

Recommended fix:

1. Install `tectonic`, `latexmk`, or `pdflatex`.
2. Re-run the program.
3. If PDF generation still fails, use the `.tex` file and compile it manually.

### Problem: Web Server Does Not Start

Check that:

1. `index.html` exists in the project folder.
2. The selected port is not already in use.
3. You are running the command from the correct folder.

Try a different port:

```bash
python EinsteinResearch.py serve --port 8080
```

### Problem: Lab Mode Fails

Lab mode may require extra local files or configuration. If a required lab workflow file is missing, add the file to the project folder or run the Core Research Pipeline instead.

### Problem: The Model Does Not Respond

Check that:

1. Your API key is configured correctly.
2. Your internet connection is working.
3. The selected model name is valid.
4. You have access to the selected model.

---

## 22. Best Practices

For stronger research results:

1. Use a specific research question.
2. Provide real data when possible.
3. Review each stage before accepting the output.
4. Verify citations and sources manually.
5. Check all generated hypotheses for scientific plausibility.
6. Treat generated experiment designs as drafts, not final protocols.
7. Use the LaTeX report as a starting point for a polished research paper.
8. Keep human review and domain expertise in the loop.

---

## Safety
Now we added safety level: 
/safety [1-5]` shows or changes the bio/chemical risk warning level.

## 23. Discussion

`EinsteinResearch.py` is designed as a full AI-assisted research workflow rather than a simple chatbot. Its main value is that it connects the major stages of scientific work into a repeatable process: planning, literature review, hypothesis generation, experiment design, data analysis, conclusion writing, and report generation.

The program is especially useful during early-stage research exploration. A researcher can start with a broad topic, refine it into a testable question, generate measurable predictions, design a prototype experiment, analyze preliminary data, and produce an academic-style report. This can reduce the manual overhead involved in organizing research, but it does not remove the need for human expertise.

The best way to use the program is as a research partner. The AI can accelerate background synthesis, structure ideas, and create drafts, but the researcher should still verify source quality, inspect assumptions, validate methods, and confirm whether the conclusions are scientifically justified.

Interactive mode is best for careful research review because the user can inspect each stage before continuing. Automated mode is best for repeatable workflows, batch experiments, and scripted research runs. Web chat mode is best for browser-based interaction, while LaTeX/PDF mode is useful for producing formal research artifacts.

---

## 24. Contact

For questions, demos, or collaboration opportunities, contact Einsteinian Labs.

## 25. Note
This respository is for YC batch 2026 demo. Please do not clone without consent

## 26. Links
For fully automated science coding agent, please use [Autocodex-sci](https://github.com/einstein-labs-ai/autocodex-sci)

### 27. Version
current version: 0.91

## 28. Appreciations 
- Many Thanks to [OpenAI](https://openai.com/)
- Thanks for [anthropic](https://www.anthropic.com/)
