"""Deterministic rule-routing node."""

from __future__ import annotations

import logging
from copy import deepcopy

from src.models import AgentPhase, AgentState, DiffLine, Evidence, HunkInfo, RiskCategory, Severity
from src.rag.retriever import RAGRetriever
from src.rules.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolRouterNode:
    """Run selected rules against every hunk and retrieve code context."""

    def __init__(self, registry: ToolRegistry, rag: RAGRetriever) -> None:
        self.registry = registry
        self.rag = rag

    def __call__(self, state: AgentState) -> dict:
        # Consume the unified queue. Anti-idle validation in Reflection
        # guarantees a non-empty valid queue on loop-back; on the first
        # round it holds Planner's initial plan.
        selected_tools = state.pending_tools
        evidence_pool: list[Evidence] = []
        rules_executed: list[str] = []

        # Fetch codebase context (only on first round; reuse cached
        # results from state.rag_context on subsequent rounds to avoid
        # redundant RAG queries — the codebase index doesn't change
        # between reflection rounds)
        codebase: dict[str, list[dict]] = {}
        existing_codebase = state.rag_context.get("codebase", {})
        if existing_codebase and state.reflection_round > 0:
            # Round 2+: reuse cached results
            codebase = dict(existing_codebase)
        else:
            # First round: fetch fresh from RAG
            for hunk in state.hunks:
                fp = hunk.file_path
                if fp in codebase:
                    continue
                try:
                    codebase[fp] = self.rag.search_codebase(fp)
                except Exception:
                    codebase[fp] = []

        for rule_name in selected_tools:
            if rule_name not in self.registry.list_all():
                continue
            rules_executed.append(rule_name)
            for hunk in state.hunks:
                # Try to get the complete function code from RAG
                # context so AST rules can parse valid Python.
                effective_hunk = self._enrich_hunk(hunk, codebase)
                try:
                    evidence_pool.extend(
                        self.registry.execute(rule_name, effective_hunk)
                    )
                except Exception as exc:
                    logger.warning(
                        "Rule %s failed on %s:%s: %s: %s",
                        rule_name,
                        hunk.file_path,
                        hunk.new_start,
                        type(exc).__name__,
                        exc,
                    )
                    evidence_pool.append(
                        Evidence(
                            source=rule_name,
                            rule_id=None,
                            category=RiskCategory.BUG_RISK,
                            severity=Severity.LOW,
                            message=(
                                f"Rule '{rule_name}' failed to execute "
                                f"on {hunk.file_path} (line {hunk.new_start}): "
                                f"{type(exc).__name__}: {exc}. "
                                f"Potential undetected risks may exist."
                            ),
                            line_range=(hunk.new_start, hunk.new_start),
                            snippet="",
                            confidence=0.3,
                            source_type="error",
                        )
                    )

        rag_context = dict(state.rag_context)
        rag_context["codebase"] = codebase
        return {
            "evidence_pool": evidence_pool,
            "rules_executed": rules_executed,
            # Consume the queue: what actually ran is already recorded in
            # rules_executed (accumulated); the queue must be empty until
            # Reflection decides to loop and refills it.
            "pending_tools": [],
            "phase": AgentPhase.JUDGING,
            "needs_more_analysis": False,
            "rag_context": rag_context,
        }

    def _enrich_hunk(
        self,
        hunk: HunkInfo,
        codebase: dict[str, list[dict]],
    ) -> HunkInfo:
        """Return a hunk with full function code if available from RAG.

        Looks up the hunk's file path in the codebase index and finds
        symbols whose line range contains the hunk's changed lines.
        If found, replaces ``added_code`` with the full function source
        so AST-based rules can parse valid Python.

        - Hunk spanning two functions: both are checked separately;
          each function's full code is appended as a synthetic hunk-like
          DiffLine list so rules run once per overlapping function.
        - Nested functions: outer function's content includes inner ones.
        - Global code (no matching symbol): original hunk unchanged.
        """
        symbols = codebase.get(hunk.file_path, [])
        if not symbols:
            return hunk

        hunk_start = hunk.new_start
        hunk_end = hunk.new_start + hunk.new_count

        # Find symbols whose line range overlaps with the hunk
        matching_contents: list[str] = []
        for sym in symbols:
            lr = sym.get("line_range", "")
            if not lr or "-" not in lr:
                continue
            try:
                parts = lr.split("-")
                sym_start = int(parts[0])
                sym_end = int(parts[1])
            except (ValueError, IndexError):
                continue

            # Check if hunk lines fall within this symbol's range
            if sym_start <= hunk_end and sym_end >= hunk_start:
                content = sym.get("content", "")
                if content:
                    matching_contents.append(content)

        if not matching_contents:
            # No matching symbol — use original hunk (global code or
            # file not indexed). AST rules may fail, regex still works.
            return hunk

        # Build a synthetic hunk with the full function code as a
        # single added line. This lets rules that use added_code or
        # added_lines see the complete function.
        full_code = "\n\n".join(matching_contents)
        enriched = HunkInfo(
            file_path=hunk.file_path,
            old_start=hunk.old_start,
            old_count=hunk.old_count,
            new_start=hunk.new_start,
            new_count=hunk.new_count,
            section_header=hunk.section_header,
            lines=[
                DiffLine(
                    content=full_code,
                    change_type=hunk.lines[0].change_type if hunk.lines else None,
                    old_line_no=hunk.lines[0].old_line_no if hunk.lines else None,
                    new_line_no=hunk.lines[0].new_line_no if hunk.lines else None,
                )
            ],
        )
        return enriched
