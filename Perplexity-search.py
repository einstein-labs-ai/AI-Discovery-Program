from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from perplexity import Perplexity

DEFAULT_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar-deep-research")
DEFAULT_REF_RESULTS = int(os.getenv("PERPLEXITY_MAX_RESULTS", "8"))
MAX_FILE_CHARS = int(os.getenv("PERPLEXITY_MAX_FILE_CHARS", "30000"))
PROMPT_PREFIX = ">"

LATEX_ESCAPE_MAP: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

ASCII_EINSTEIN_LOGO = r'''
                 .-""""-.
               .'  _  _  '.
              /   (o)(o)   \
             |   .-.___.-.  |
             |  /  \_/  \ \ |
              \ \   ^   / /
               '._'-=-'_.-'
                  '---'
  _____ _           _       _         ____                      _
 | ____(_)_ __  ___| |_ ___(_)_ __   / ___|  ___  __ _ _ __ ___| |__
 |  _| | | '_ \/ __| __/ _ \ | '_ \  \___ \ / _ \/ _` | '__/ __| '_ \
 | |___| | | | \__ \ ||  __/ | | | |  ___) |  __/ (_| | | | (__| | | |
 |_____|_|_| |_|___/\__\___|_|_| |_| |____/ \___|\__,_|_|  \___|_| |_|
'''


class QuitRequested(Exception):
    """Raised when the user types /quit."""


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str
    date: str | None = None


@dataclass
class SessionOutput:
    title: str
    body: str
    citations: list[str]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        cleaned = item.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        output.append(cleaned)
    return output


def print_logo() -> None:
    print(ASCII_EINSTEIN_LOGO)
    print("Einstein Deep Search CLI")
    print(f"Model: {DEFAULT_MODEL}")
    print("Type /quit at any prompt to exit.")


def print_menu() -> None:
    print("\n+---------------------------------------+")
    print("| 1) Literature Review                  |")
    print("| 2) Data Analysis (file)               |")
    print("| 3) Find References                    |")
    print("| 4) Save Last Result to LaTeX + PDF    |")
    print("| /quit) Exit                           |")
    print("+---------------------------------------+")


def read_input(prompt: str, *, default: str | None = None, allow_empty: bool = False) -> str:
    while True:
        raw = input(f"{PROMPT_PREFIX} {prompt} ").strip()
        if raw.lower() == "/quit":
            raise QuitRequested()
        if raw:
            return raw
        if default is not None:
            return default
        if allow_empty:
            return ""
        print("Input required. Type /quit to exit.")


def _extract_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for chunk in content:
        if hasattr(chunk, "model_dump"):
            chunk = chunk.model_dump()
        if isinstance(chunk, dict):
            text = chunk.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts).strip()


def extract_message_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        return ""

    output: list[str] = []
    for choice in choices:
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")
        if message is None:
            message = getattr(choice, "delta", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("delta")
        if message is None:
            continue

        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        text = _extract_content_text(content)
        if text:
            output.append(text)
    return "\n".join(output).strip()


def extract_citations(response: Any) -> list[str]:
    citations = getattr(response, "citations", None)
    if citations is None and isinstance(response, dict):
        citations = response.get("citations")
    if not isinstance(citations, list):
        return []
    return _dedupe([str(item) for item in citations if item])


def _is_search_mode_unsupported_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "search_mode" in text and "not supported" in text


def ask_model(
    client: Perplexity,
    system_prompt: str,
    user_prompt: str,
    *,
    search_mode: str = "academic",
    num_search_results: int = 8,
    disable_search: bool = False,
) -> tuple[str, list[str]]:
    request_kwargs: dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "disable_search": disable_search,
    }
    if not disable_search:
        # Perplexity enforces search-result bounds when search is enabled.
        # Clamp to the documented range so callers do not trigger validation errors.
        request_kwargs["num_search_results"] = max(3, min(100, num_search_results))
    if search_mode and not disable_search:
        request_kwargs["search_mode"] = search_mode

    try:
        response = client.chat.completions.create(**request_kwargs)
    except Exception as exc:
        if not _is_search_mode_unsupported_error(exc):
            raise
        request_kwargs.pop("search_mode", None)
        response = client.chat.completions.create(**request_kwargs)

    return extract_message_text(response), extract_citations(response)


def _normalize_search_hit(item: Any) -> SearchHit:
    if hasattr(item, "model_dump"):
        item = item.model_dump()

    if isinstance(item, dict):
        return SearchHit(
            title=str(item.get("title", "<no title>")),
            url=str(item.get("url", "<no url>")),
            snippet=str(item.get("snippet", "")),
            date=item.get("date"),
        )

    if isinstance(item, (tuple, list)):
        title = str(item[0]) if len(item) > 0 else "<no title>"
        url = str(item[1]) if len(item) > 1 else "<no url>"
        snippet = str(item[2]) if len(item) > 2 else ""
        return SearchHit(title=title, url=url, snippet=snippet)

    return SearchHit(title=str(item), url="<unknown>", snippet="")


def search_references(client: Perplexity, query: str, max_results: int) -> list[SearchHit]:
    request_kwargs: dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "search_mode": "academic",
    }
    try:
        response = client.search.create(**request_kwargs)
    except Exception as exc:
        if not _is_search_mode_unsupported_error(exc):
            raise
        request_kwargs.pop("search_mode", None)
        response = client.search.create(**request_kwargs)

    results = getattr(response, "results", None)
    if results is None and isinstance(response, dict):
        results = response.get("results")
    if not isinstance(results, list):
        return []

    return [_normalize_search_hit(item) for item in results]


def print_citations(citations: list[str]) -> None:
    if not citations:
        return
    print("\nCitations:")
    for index, citation in enumerate(citations, start=1):
        print(f"[{index}] {citation}")


def latex_escape(text: str) -> str:
    return "".join(LATEX_ESCAPE_MAP.get(char, char) for char in text)


def _safe_url_for_latex(url: str) -> str:
    return url.replace("\\", "/").replace("{", "%7B").replace("}", "%7D")


def _format_latex_paragraphs(text: str) -> str:
    paragraphs = [part.strip() for part in text.strip().split("\n\n") if part.strip()]
    formatted: list[str] = []
    for paragraph in paragraphs:
        lines = [latex_escape(line.strip()) for line in paragraph.splitlines() if line.strip()]
        if lines:
            formatted.append(r"\\ ".join(lines))
    return "\n\n".join(formatted)


def build_latex_document(output: SessionOutput) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[hidelinks]{hyperref}",
        rf"\title{{{latex_escape(output.title)}}}",
        rf"\date{{Generated on {latex_escape(generated_at)}}}",
        r"\begin{document}",
        r"\maketitle",
        r"\section*{Result}",
        _format_latex_paragraphs(output.body or "<no output>"),
    ]

    if output.citations:
        lines.append(r"\section*{Citations}")
        lines.append(r"\begin{enumerate}")
        for citation in output.citations:
            cleaned = citation.strip()
            if cleaned.startswith(("http://", "https://")):
                lines.append(rf"\item \url{{{_safe_url_for_latex(cleaned)}}}")
            else:
                lines.append(rf"\item {latex_escape(cleaned)}")
        lines.append(r"\end{enumerate}")

    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def resolve_export_path(raw_path: str, extension: str) -> Path:
    normalized = raw_path.strip() or f"research_session_export{extension}"
    path = resolve_file_path(normalized)
    if path.suffix.lower() != extension:
        path = path.with_suffix(extension)
    return path


def save_output_to_latex(output: SessionOutput, raw_path: str) -> Path:
    tex_path = resolve_export_path(raw_path, ".tex")
    tex_path.write_text(build_latex_document(output), encoding="utf-8")
    return tex_path


def convert_latex_to_pdf(tex_path: Path) -> Path:
    compiler: str | None = None
    if shutil.which("pdflatex"):
        compiler = "pdflatex"
        command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    elif shutil.which("tectonic"):
        compiler = "tectonic"
        command = ["tectonic", tex_path.name]
    else:
        raise RuntimeError("No LaTeX compiler found. Install pdflatex or tectonic.")

    result = subprocess.run(
        command,
        cwd=tex_path.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        error_text = result.stderr.strip() or result.stdout.strip() or "Unknown compilation error."
        short_error = error_text.splitlines()[-1] if error_text else "Unknown compilation error."
        raise RuntimeError(f"{compiler} failed: {short_error}")

    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
        raise RuntimeError("Compiler reported success, but no PDF file was created.")
    return pdf_path


def run_export_last_result(last_output: SessionOutput | None) -> None:
    if last_output is None:
        print("Nothing to export yet. Run options 1, 2, or 3 first.")
        return

    output_path = read_input("Output LaTeX file [research_session_export.tex]:", default="research_session_export.tex")
    try:
        tex_path = save_output_to_latex(last_output, output_path)
    except OSError as exc:
        print(f"Could not save LaTeX file: {exc}")
        return

    print(f"LaTeX file saved to: {tex_path}")
    try:
        pdf_path = convert_latex_to_pdf(tex_path)
        print(f"PDF generated: {pdf_path}")
    except Exception as exc:
        print(f"LaTeX saved, but PDF conversion failed: {exc}")


def resolve_file_path(raw_path: str) -> Path:
    path = Path(raw_path.strip()).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def summarize_delimited_file(path: Path, delimiter: str, preview_rows: int = 10) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
        reader = csv.reader(file, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            return f"File: {path}\nThe file is empty."

        rows: list[list[str]] = []
        total_rows = 0
        for row in reader:
            total_rows += 1
            if len(rows) < preview_rows:
                rows.append(row)

    fmt = "CSV" if delimiter == "," else "TSV"
    lines = [
        f"File: {path}",
        f"Detected format: {fmt}",
        f"Columns: {', '.join(header) if header else '<none>'}",
        f"Rows (excluding header): {total_rows}",
        "Preview rows:",
    ]
    if rows:
        for i, row in enumerate(rows, start=1):
            row_text = " | ".join(cell.strip() for cell in row)
            if len(row_text) > 400:
                row_text = row_text[:400] + "..."
            lines.append(f"{i}. {row_text}")
    else:
        lines.append("<no rows>")
    return "\n".join(lines)


def summarize_json_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        parsed = json.loads(text)
        rendered = json.dumps(parsed, indent=2)
    except json.JSONDecodeError:
        rendered = text

    if len(rendered) > MAX_FILE_CHARS:
        rendered = rendered[:MAX_FILE_CHARS] + "\n...<truncated>"

    return f"File: {path}\nDetected format: JSON\nContent preview:\n{rendered}"


def load_file_context(raw_path: str) -> str:
    path = resolve_file_path(raw_path)
    if not path.exists():
        raise OSError(f"File not found: {path}")
    if not path.is_file():
        raise OSError(f"Not a file: {path}")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return summarize_delimited_file(path, delimiter="," if suffix == ".csv" else "\t")
    if suffix == ".json":
        return summarize_json_file(path)

    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_FILE_CHARS:
        text = text[:MAX_FILE_CHARS] + "\n...<truncated>"
    return f"File: {path}\nDetected format: {suffix.lstrip('.') or 'text'}\nContent preview:\n{text}"


def run_literature_review(client: Perplexity) -> SessionOutput:
    query = read_input("Literature review topic/query:")
    focus = read_input("Specific focus (optional, press Enter to skip):", allow_empty=True)
    style = read_input("Output style [concise bullet summary]:", default="concise bullet summary")

    user_prompt = (
        f"Topic/query: {query}\n"
        f"Specific focus: {focus or 'general overview'}\n"
        "Please produce a literature review with sections:\n"
        "1) Key findings\n"
        "2) Methods used in the field\n"
        "3) Contradictions or open questions\n"
        f"Style requirement: {style}\n"
        "Keep claims evidence-based and include concrete references."
    )

    text, citations = ask_model(
        client,
        system_prompt=(
            "You are an academic research assistant. Generate accurate, source-grounded "
            "literature reviews for technical and scientific topics."
        ),
        user_prompt=user_prompt,
        search_mode="academic",
        num_search_results=10,
        disable_search=False,
    )

    print("\nLiterature Review:\n")
    rendered_text = text or "<no output>"
    print(rendered_text)
    print_citations(citations)
    return SessionOutput(
        title=f"Literature Review: {query}",
        body=rendered_text,
        citations=citations,
    )


def run_data_analysis(client: Perplexity) -> SessionOutput:
    file_path = read_input("Path to data file (csv/tsv/json/txt):")
    context = load_file_context(file_path)

    preview = context[:1200]
    if len(context) > 1200:
        preview += "\n...<preview truncated>"
    print("\nLoaded file preview:\n")
    print(preview)

    objective = read_input(
        "Analysis objective [Identify trends, anomalies, and recommendations]:",
        default="Identify trends, anomalies, and recommendations.",
    )

    text, citations = ask_model(
        client,
        system_prompt=(
            "You are a scientific data analyst. Analyze provided dataset context and report "
            "findings clearly with caveats."
        ),
        user_prompt=(
            f"Objective: {objective}\n\n"
            f"Dataset context:\n{context}\n\n"
            "Return:\n"
            "1) Key findings\n"
            "2) Potential anomalies/outliers\n"
            "3) Caveats and data quality limits\n"
            "4) Next best analyses to run"
        ),
        disable_search=True,
        num_search_results=1,
    )

    print("\nData Analysis:\n")
    rendered_text = text or "<no output>"
    print(rendered_text)
    print_citations(citations)
    return SessionOutput(
        title=f"Data Analysis: {Path(file_path).name or file_path}",
        body=rendered_text,
        citations=citations,
    )


def run_find_references(client: Perplexity) -> SessionOutput:
    query = read_input("Reference query/topic:")
    raw_max = read_input(f"Max references [{DEFAULT_REF_RESULTS}]:", default=str(DEFAULT_REF_RESULTS))
    try:
        max_results = int(raw_max)
    except ValueError:
        max_results = DEFAULT_REF_RESULTS
    max_results = max(1, min(25, max_results))

    hits = search_references(client, query, max_results=max_results)
    if not hits:
        print("No references found.")
        return SessionOutput(
            title=f"References: {query}",
            body="No references found.",
            citations=[],
        )

    print(f"\nReferences ({len(hits)}):\n")
    lines: list[str] = []
    citations: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        date_text = f" ({hit.date})" if hit.date else ""
        print(f"{idx}. {hit.title}{date_text}")
        print(f"   URL: {hit.url}")
        if hit.snippet:
            print(f"   Snippet: {hit.snippet}")
        print("")
        entry = f"{idx}. {hit.title}{date_text}\nURL: {hit.url}"
        if hit.snippet:
            entry += f"\nSnippet: {hit.snippet}"
        lines.append(entry)
        if hit.url:
            citations.append(hit.url)

    return SessionOutput(
        title=f"References: {query}",
        body="\n\n".join(lines),
        citations=_dedupe(citations),
    )


def main() -> None:
    print_logo()
    client = Perplexity()
    last_output: SessionOutput | None = None

    while True:
        print_menu()
        choice = read_input("Choose 1, 2, 3, 4, or /quit:")
        try:
            if choice == "1":
                last_output = run_literature_review(client)
            elif choice == "2":
                last_output = run_data_analysis(client)
            elif choice == "3":
                last_output = run_find_references(client)
            elif choice == "4":
                run_export_last_result(last_output)
            else:
                print("Invalid choice. Please choose 1, 2, 3, 4, or /quit.")
        except QuitRequested:
            raise
        except Exception as exc:
            print(f"Request failed: {exc}")


if __name__ == "__main__":
    try:
        main()
    except QuitRequested:
        print("Goodbye.")

 
