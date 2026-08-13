#!/usr/bin/env python3
"""CLI entry point for the Code Risk Agent."""

from __future__ import annotations

import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click
import sqlite3
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import settings
from src.llm.client import LLMClient, EmbeddingClient
from src.rag.retriever import RAGRetriever
from src.rag.indexer import SecurityKnowledgeLoader, CodebaseIndexer
from src.memory.long_term import LongTermMemory
from src.parsers.diff_parser import GitDiffParser
from src.rules import registry
from src.graph import build_graph

console = Console()


def _init_components():
    """Create and initialise all infrastructure components.

    Returns a tuple of (llm, rag, ltm, registry).
    """
    db_path = settings.db_url

    llm = LLMClient(
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
        model=settings.model_name,
    )
    embedding_client = EmbeddingClient(
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
        model=settings.embedding_model,
    )

    rag = RAGRetriever(db_path, embedding_client)

    ltm = LongTermMemory(db_path)
    ltm.init_tables()

    security_loader = SecurityKnowledgeLoader(db_path)
    security_loader.init_tables()

    # Seed security knowledge from JSON if the table is empty
    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM security_knowledge"
        ).fetchone()[0]
    finally:
        conn.close()

    if count == 0:
        json_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "security_knowledge.json",
        )
        if os.path.exists(json_path):
            security_loader.load_from_json(json_path, embedding_client)
            console.print(
                "[dim]Loaded security knowledge from data/security_knowledge.json[/dim]"
            )

    return llm, rag, ltm, registry


def _render_report(report, console: Console) -> None:
    """Pretty-print a RiskReport using Rich panels and tables."""
    if report is None:
        console.print("[red]No report was generated.[/red]")
        return

    # -- Summary panel --
    summary_text = (
        f"[bold]Repository:[/bold] {report.repo}\n"
        f"[bold]Commit:[/bold] {report.commit_sha or 'N/A'}\n"
        f"[bold]Files scanned:[/bold] {len(report.files_scanned)}\n"
        f"[bold]Total hunks:[/bold] {report.total_hunks}\n"
        f"[bold]Overall risk score:[/bold] {report.overall_risk_score:.2f}\n"
        f"[bold]Reflection rounds:[/bold] {report.reflection_rounds}\n"
        f"\n{report.summary}"
    )
    console.print(Panel(summary_text, title="Code Change Risk Report", border_style="cyan"))

    if not report.risks:
        console.print("[green]No significant risks detected.[/green]")
        return

    # -- Risk table --
    table = Table(title=f"Detected {len(report.risks)} Risk(s)", border_style="red")
    table.add_column("#", style="dim", width=4)
    table.add_column("Severity", style="bold")
    table.add_column("Title")
    table.add_column("Category")
    table.add_column("Score", justify="right")
    table.add_column("Location")

    severity_colors = {
        "critical": "bright_red",
        "high": "red",
        "medium": "yellow",
        "low": "green",
        "info": "dim",
    }

    for i, risk in enumerate(report.risks, 1):
        sev = risk.severity.value
        color = severity_colors.get(sev, "white")
        loc = risk.file_path or ""
        if risk.line_range:
            loc += f":{risk.line_range[0]}-{risk.line_range[1]}"
        table.add_row(
            str(i),
            f"[{color}]{sev.upper()}[/{color}]",
            risk.title,
            risk.category.value,
            f"{risk.risk_score:.2f}",
            loc,
        )
    console.print(table)

    # -- Detail panels --
    for i, risk in enumerate(report.risks, 1):
        detail_lines = [
            f"[bold]Description:[/bold] {risk.description}",
            f"[bold]Evidence chain ({len(risk.evidence_chain)} item(s)):[/bold]",
        ]
        for ev in risk.evidence_chain:
            detail_lines.append(
                f"  - [{ev.source_type}] {ev.source} "
                f"({ev.severity.value}, conf={ev.confidence:.2f}): {ev.message}"
            )
        if risk.suggestion:
            detail_lines.append(f"[bold]Suggestion:[/bold] {risk.suggestion}")
        console.print(
            Panel(
                "\n".join(detail_lines),
                title=f"Risk #{i}: {risk.title}",
                border_style="yellow",
            )
        )

    if report.long_term_feedback_applied:
        console.print(
            f"\n[dim]Long-term feedback applied: "
            f"{', '.join(report.long_term_feedback_applied)}[/dim]"
        )


@click.group()
def cli():
    """Code Risk Agent — analyse git diffs for potential risks."""


@cli.command()
@click.option("--diff-file", default=None, help="Path to a file containing a git diff.")
@click.option("--diff-text", default=None, help="Inline git diff text to analyse.")
@click.option(
    "--thread-id",
    default=None,
    help="Thread ID for RAG history. Defaults to a random UUID."
)
@click.option("--repo", default=".", help="Repository name or path.")
@click.option("--commit", default=None, help="Commit SHA (for record-keeping).")
def analyze(diff_file, diff_text, thread_id, repo, commit):
    """Analyse a git diff and produce a risk report."""
    # -- Read diff --
    if diff_file:
        if not os.path.exists(diff_file):
            console.print(f"[red]Diff file not found: {diff_file}[/red]")
            sys.exit(1)
        with open(diff_file, encoding="utf-8") as f:
            raw_diff = f.read()
    elif diff_text:
        raw_diff = diff_text
    else:
        console.print("[red]Provide --diff-file or --diff-text.[/red]")
        sys.exit(1)

    if not raw_diff.strip():
        console.print("[yellow]Empty diff — nothing to analyse.[/yellow]")
        sys.exit(0)

    thread_id = thread_id or str(uuid.uuid4())

    # -- Parse diff --
    hunks = GitDiffParser().parse(raw_diff)
    if not hunks:
        console.print("[yellow]No hunks found in diff.[/yellow]")
        sys.exit(0)

    console.print(f"[cyan]Parsed {len(hunks)} hunk(s) from diff.[/cyan]")

    # -- Initialise components --
    console.print("[cyan]Initialising components...[/cyan]")
    llm, rag, ltm, reg = _init_components()

    # -- Build graph --
    console.print("[cyan]Building analysis graph...[/cyan]")
    graph = build_graph(llm, rag, ltm, reg, max_rounds=settings.max_reflection_rounds)

    # -- Run analysis --
    initial_state = {
        "repo": repo,
        "commit_sha": commit,
        "raw_diff": raw_diff,
        "hunks": [h.model_dump() for h in hunks],
    }

    console.print("[cyan]Running analysis...[/cyan]")
    result = graph.invoke(
        initial_state,
        {"configurable": {"thread_id": thread_id}},
    )

    # -- Write to RAG history --
    report = result.get("report")
    if report is not None:
        for hunk in hunks:
            try:
                rag.add_history(
                    thread_id=thread_id,
                    file_path=hunk.file_path,
                    diff_summary=hunk.added_code[:200],
                    risk_titles=[r.title for r in report.risks],
                    risk_categories=[r.category.value for r in report.risks],
                    overall_score=report.overall_risk_score,
                )
            except Exception as exc:
                console.print(f"[dim]Warning: could not save RAG history: {exc}[/dim]")

    # -- Render report --
    console.print()
    _render_report(report, console)


@cli.command()
@click.option("--thread-id", required=True, help="Thread ID for this feedback.")
@click.option("--file-pattern", required=True, help="File pattern (supports * wildcards).")
@click.option("--rule-id", default=None, help="Related rule ID.")
@click.option(
    "--type", "feedback_type", required=True,
    help="Feedback type (e.g. false_positive, missing, severity_adjust)."
)
@click.option("--content", required=True, help="Feedback content.")
def feedback(thread_id, file_pattern, rule_id, feedback_type, content):
    """Add human feedback to long-term memory."""
    db_path = settings.db_url
    ltm = LongTermMemory(db_path)
    ltm.init_tables()

    ltm.add_feedback(thread_id, file_pattern, rule_id, feedback_type, content)
    console.print(
        Panel(
            f"[green]Feedback added successfully.[/green]\n\n"
            f"[bold]Thread ID:[/bold] {thread_id}\n"
            f"[bold]File pattern:[/bold] {file_pattern}\n"
            f"[bold]Rule ID:[/bold] {rule_id or 'N/A'}\n"
            f"[bold]Type:[/bold] {feedback_type}\n"
            f"[bold]Content:[/bold] {content}",
            title="Feedback Recorded",
            border_style="green",
        )
    )


@cli.command()
@click.option("--repo-path", required=True, help="Path to the repository to index.")
@click.option("--clear", is_flag=True, default=False, help="Clear entire index before re-indexing.")
def index(repo_path, clear):
    """Index a codebase for RAG codebase search."""
    if not os.path.isdir(repo_path):
        console.print(f"[red]Directory not found: {repo_path}[/red]")
        sys.exit(1)

    db_path = settings.db_url
    embedding_client = EmbeddingClient(
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
        model=settings.embedding_model,
    )
    indexer = CodebaseIndexer(db_path, embedding_client)
    indexer.init_tables()

    if clear:
        indexer.clear_index()
        console.print("[dim]Cleared entire codebase index.[/dim]")

    total_symbols = 0
    total_files = 0

    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories and __pycache__
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d != "__pycache__"
        ]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    content = f.read()
                rel_path = os.path.relpath(fpath, repo_path)

                # Delete old records for this file to avoid duplicates
                deleted = indexer.delete_by_file(rel_path)
                if deleted:
                    console.print(f"  [dim]Removed {deleted} old record(s) for {rel_path}[/dim]")

                count = indexer.index_file_full(rel_path, content)
                total_symbols += count
                total_files += 1
                console.print(f"  [green]Indexed[/green] {rel_path} ({count} symbol(s))")
            except Exception as exc:
                console.print(f"  [red]Failed[/red] {fpath}: {exc}")

    # Resolve cross-file import paths for search_codebase
    cross_refs = indexer.resolve_imports()
    console.print(f"[dim]Resolved {cross_refs} cross-file import reference(s).[/dim]")

    console.print(
        Panel(
            f"[green]Indexing complete.[/green]\n\n"
            f"[bold]Files indexed:[/bold] {total_files}\n"
            f"[bold]Symbols indexed:[/bold] {total_symbols}",
            title="Codebase Index",
            border_style="green",
        )
    )


if __name__ == "__main__":
    cli()
