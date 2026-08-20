"""LLM-driven risk-judgment node."""

from __future__ import annotations

import ast
import json

from src.llm.client import LLMClient
from src.models import (
    AgentPhase, AgentState, DismissedEvidence, RiskCategory, RiskItem, Severity,
)
from src.rag.retriever import RAGRetriever


def _parse_symbol_range(lr):
    """Parse "12-40"-style line ranges; return None when malformed."""
    if not lr or "-" not in lr:
        return None
    try:
        start_s, end_s = lr.split("-", 1)
        return int(start_s), int(end_s)
    except ValueError:
        return None


def _called_names(source: str) -> set[str]:
    """Collect function/method names invoked in a code snippet (AST-level)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def _trim_symbol(sym: dict, max_chars: int = 1000) -> dict:
    """Keep prompt-relevant fields only; drop id/imports noise; cap content."""
    content = sym.get("content", "") or ""
    trimmed = {
        "file_path": sym.get("file_path"),
        "symbol_name": sym.get("symbol_name"),
        "symbol_type": sym.get("symbol_type"),
        "line_range": sym.get("line_range"),
        "source": sym.get("source", "diff_file"),
    }
    if len(content) > max_chars:
        trimmed["content"] = content[:max_chars] + "\n... [truncated]"
    else:
        trimmed["content"] = content
    return trimmed


def _select_codebase_context(hunks, codebase: dict, max_chars: int = 1000) -> dict:
    """Prune the RAG codebase dict to what judgment actually needs.

    Part 1 — diff-file symbols whose line_range overlaps any hunk's
    [new_start, new_start + new_count] interval (same predicate as
    ToolRouter._enrich_hunk, recomputed here so ToolRouter stays untouched).

    Part 2 — cross-file symbols whose symbol_name matches a function/method
    actually called from the part-1 function sources (AST Call nodes),
    instead of whole symbol tables of every referenced file.

    Fallback: if no hunk overlaps any symbol (global code / unindexed file),
    return all symbols trimmed, so the judge never loses context entirely.
    """
    if not codebase:
        return {}

    selected: list[dict] = []
    for hunk in hunks:
        h_start = hunk.new_start
        h_end = hunk.new_start + hunk.new_count
        for sym in codebase.get(hunk.file_path, []):
            if sym.get("source") == "cross_file":
                continue
            rng = _parse_symbol_range(sym.get("line_range"))
            if rng and rng[0] <= h_end and rng[1] >= h_start:
                selected.append(sym)

    called: set[str] = set()
    for sym in selected:
        called |= _called_names(sym.get("content", ""))

    for symbols in codebase.values():
        for sym in symbols:
            if sym.get("source") != "cross_file":
                continue
            if sym.get("symbol_name") in called:
                selected.append(sym)

    if not selected:
        selected = [
            sym for symbols in codebase.values() for sym in symbols
        ]

    grouped: dict[str, list[dict]] = {}
    for sym in selected:
        key = sym.get("file_path") or "unknown"
        grouped.setdefault(key, []).append(_trim_symbol(sym, max_chars))
    return grouped


def _codebase_is_fallback(hunks, codebase: dict, ctx: dict) -> bool:
    """True when _select_codebase_context took its fallback path.

    Fallback means no diff-file symbol overlapped any hunk, so ctx contains
    ALL symbols — in that case the evidence snippet is the judge's only
    line-level anchor and must be kept.
    """
    for hunk in hunks:
        h_start = hunk.new_start
        h_end = hunk.new_start + hunk.new_count
        for sym in codebase.get(hunk.file_path, []):
            if sym.get("source") == "cross_file":
                continue
            rng = _parse_symbol_range(sym.get("line_range"))
            if rng and rng[0] <= h_end and rng[1] >= h_start:
                return False  # overlap found -> selection path, not fallback
    return bool(codebase)  # no overlap: fallback only if there was any symbol


class JudgeNode:
    """Convert evidence and RAG context into structured risk items.

    Zero-hallucination invariant: every emitted RiskItem must carry a
    non-empty, index-valid evidence chain. Risks without valid refs are
    dropped at parse time — the judge may reject or downplay evidence,
    but never invent findings without it.
    """

    def __init__(self, llm: LLMClient, rag: RAGRetriever) -> None:
        self.llm = llm
        self.rag = rag

    def __call__(self, state: AgentState) -> dict:
        rule_ids = list({item.rule_id for item in state.evidence_pool if item.rule_id})
        try:
            security_raw = self.rag.search_security(
                " ".join(item.message for item in state.evidence_pool), rule_ids
            )
        except Exception:
            security_raw = []
        # Slim knowledge: title + best_practice + rule_id only. The incident
        # narrative (content) rarely changes attribution or scoring but is
        # the single largest token block in this prompt.
        security = [
            {
                "rule_id": k.get("rule_id"),
                "title": k.get("title"),
                "best_practice": k.get("best_practice"),
            }
            for k in (security_raw or [])
        ]

        codebase_ctx = _select_codebase_context(
            state.hunks, state.rag_context.get("codebase", {})
        )

        # Evidence snippets are only needed when the codebase context could
        # not provide symbol-level source (fallback path or unindexed file);
        # when symbols are present the snippet is redundant triple context.
        symbols_present = bool(codebase_ctx) and not _codebase_is_fallback(
            state.hunks, state.rag_context.get("codebase", {}), codebase_ctx
        )
        evidence = []
        for idx, item in enumerate(state.evidence_pool):
            d = item.model_dump(mode="json")
            if symbols_present:
                d["snippet"] = None
            evidence.append(d)

        # History as attribution prior only: file + risk titles, no narrative.
        history_slim = [
            {
                "file_path": h.get("file_path"),
                "risk_titles": h.get("risk_titles"),
            }
            for h in (state.rag_context.get("history") or [])
        ]

        prompt = (
            "Judge these code-risk evidences. Return JSON only: "
            "{\"risks\": [{\"title\": str, \"category\": str, "
            "\"severity\": str, \"description\": str, \"evidence_refs\": [int], "
            "\"suggestion\": str, \"file_path\": str, \"line_range\": [int, int], "
            "\"risk_score\": float (0.0-1.0)}], "
            "\"dismissed_evidence\": [{\"index\": int, \"reason\": str}], "
            "\"overall_risk_score\": float (0.0-1.0)}.\n"
            "Not all evidence constitutes a real risk. Before confirming any "
            "evidence as a risk, VERIFY it against the provided codebase "
            "context — check whether the flagged pattern is actually present "
            "in the code, whether it is a false positive (e.g. test fixture, "
            "documentation example, or the symbol is in fact used elsewhere), "
            "and whether the severity is warranted given the context.\n"
            "If an evidence item is a false positive or irrelevant, put its "
            "index in dismissed_evidence with a reason — do NOT create a "
            "risk for it. Every risk MUST reference at least one evidence "
            "index in evidence_refs; risks without valid evidence references "
            "will be discarded.\n\n"
            f"Evidence (reference by zero-based index; each item carries its own file_path):\n{json.dumps(evidence)}\n\n"
            f"Security knowledge:\n{json.dumps(security)}\n\n"
            f"Codebase context (diff-file symbols overlapping changed lines, plus cross-file symbols actually called from them):\n{json.dumps(codebase_ctx)}"
        )
        if history_slim:
            prompt += (
                "\n\nHistorical risks previously flagged for these files "
                f"(use as attribution signal, not as new findings):\n{json.dumps(history_slim)}"
            )
        try:
            response = self.llm.chat_json(
                "You are a code risk judge.", prompt
            )
        except Exception:
            response = None

        # Degradation path: no parseable judgment (empty, unparseable or
        # truncated LLM output). Return WITHOUT the ``risks`` /
        # ``dismissed_evidence`` keys: GraphState uses whole-value
        # replacement for these fields, so returning empty lists would
        # erase a previous round's valid adjudication. An unread judgment
        # must never wipe an earlier one — omitting the keys preserves it.
        if response is None:
            rag_context = dict(state.rag_context)
            rag_context["security"] = security_raw
            return {
                "phase": AgentPhase.REFLECTING,
                "rag_context": rag_context,
            }

        raw_risks = response.get("risks", [])
        raw_dismissed = response.get("dismissed_evidence", [])

        # --- Parse dismissed evidence (index-validated) ---
        dismissed_idx: set[int] = set()
        dismissed: list[DismissedEvidence] = []
        for d in raw_dismissed:
            idx = d.get("index")
            if isinstance(idx, int) and 0 <= idx < len(state.evidence_pool):
                dismissed_idx.add(idx)
                dismissed.append(DismissedEvidence(
                    evidence=state.evidence_pool[idx],
                    reason=d.get("reason", "Dismissed by judge"),
                ))

        risks: list[RiskItem] = []
        for item in raw_risks:
            refs = item.get("evidence_refs", [])
            valid_refs = [
                index for index in refs
                if isinstance(index, int) and 0 <= index < len(state.evidence_pool)
            ]
            # A risk must retain at least one non-dismissed valid reference.
            # If the LLM both cites and dismisses an index, dismissal wins for
            # that item; only its remaining evidence can support the risk.
            surviving_refs = [index for index in valid_refs if index not in dismissed_idx]
            chain = [state.evidence_pool[index] for index in surviving_refs]
            # Zero-hallucination guard: after filtering dismissed / invalid
            # refs, a risk must retain a real evidence chain.
            if not chain:
                continue
            line_range = item.get("line_range")
            if isinstance(line_range, list) and len(line_range) == 2:
                line_range = tuple(line_range)
            else:
                line_range = None
            try:
                category = RiskCategory(item.get("category", "security"))
            except ValueError:
                category = RiskCategory.SECURITY
            try:
                severity = Severity(item.get("severity", "medium"))
            except ValueError:
                severity = Severity.MEDIUM
            raw_score = item.get("risk_score", 0.0)
            if isinstance(raw_score, (int, float)) and raw_score > 1.0:
                raw_score = min(raw_score / 10.0, 1.0)
            risks.append(RiskItem(
                title=item.get("title", "Unknown risk"),
                category=category,
                severity=severity,
                description=item.get("description", ""),
                evidence_chain=chain,
                suggestion=item.get("suggestion"),
                file_path=item.get("file_path"),
                line_range=line_range,
                risk_score=raw_score,
            ))

        rag_context = dict(state.rag_context)
        rag_context["security"] = security_raw
        return {
            "risks": risks,
            "dismissed_evidence": dismissed,
            "phase": AgentPhase.REFLECTING,
            "rag_context": rag_context,
        }
