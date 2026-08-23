"""LLM-driven risk-judgment node."""

from __future__ import annotations

import ast
import json
from typing import Any

from src.llm.client import LLMClient
from src.memory.long_term import LongTermMemory
from src.models import (
    AgentPhase, AgentState, DismissedEvidence, RiskCategory, RiskItem, Severity,
)
from src.rag.retriever import RAGRetriever

_SEVERITY_VALUES = sorted(s.value for s in Severity)


def _validate_judge_response(parsed: Any) -> None:
    """Business-level contract check run inside the chat_json retry loop.

    Raises ValueError on the first contract violation (e.g. an unknown
    severity), which chat_json converts into a repair-retry prompt for
    the LLM instead of silently falling back to a default.
    """
    if not isinstance(parsed, dict):
        raise ValueError("response must be a JSON object")
    risks = parsed.get("risks", [])
    if not isinstance(risks, list):
        raise ValueError("'risks' must be a list")
    for i, item in enumerate(risks):
        if not isinstance(item, dict):
            raise ValueError(f"risks[{i}] must be an object")
        severity = item.get("severity")
        if severity not in _SEVERITY_VALUES:
            raise ValueError(
                f"risks[{i}].severity must be one of {_SEVERITY_VALUES}; "
                f"got {severity!r}"
            )


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


def _evidence_symbols(codebase: dict, evidence_pool) -> dict[str, list[str]]:
    """Map evidence file_path -> symbol names the evidence line ranges touch.

    Long-term feedback recall keys on symbols: a feedback row is only
    relevant to the Judge when the code it corrected is the code the
    evidence points at. Overlap uses the same interval predicate as
    _select_codebase_context, restricted to diff-file symbols (evidence
    never originates in cross-file symbols).
    """
    by_file: dict[str, list[str]] = {}
    for file_path, symbols in codebase.items():
        ranges = [
            (sym.get("symbol_name"), _parse_symbol_range(sym.get("line_range")))
            for sym in symbols
            if sym.get("source") != "cross_file"
        ]
        if not ranges:
            continue
        names: list[str] = []
        for item in evidence_pool:
            if item.file_path != file_path:
                continue
            for name, rng in ranges:
                if name in names:
                    continue
                # No evidence line_range: file-level attribution only.
                if item.line_range is None:
                    names.append(name)
                    continue
                if rng and rng[0] <= item.line_range[1] and rng[1] >= item.line_range[0]:
                    names.append(name)
        if names:
            by_file[file_path] = names[:20]
    return by_file


class JudgeNode:
    """Convert evidence and RAG context into structured risk items.

    Zero-hallucination invariant: every emitted RiskItem must carry a
    non-empty, index-valid evidence chain. Risks without valid refs are
    dropped at parse time — the judge may reject or downplay evidence,
    but never invent findings without it.

    Evidence is adjudicated in batches (``_BATCH_SIZE`` per LLM call,
    grouped by file_path): a prompt covering hundreds of evidence items
    cannot be answered within the timeout window, which silently degraded
    every judgment in a real run. Every evidence item keeps its GLOBAL id
    across batches, so per-batch results merge without re-indexing. A
    degraded batch is isolated: its evidence is counted as unadjudicated
    (``judge_unadjudicated_evidence``) while the remaining batches still
    get judged.
    """

    _BATCH_SIZE = 50

    def __init__(
        self,
        llm: LLMClient,
        rag: RAGRetriever,
        ltm: LongTermMemory | None = None,
    ) -> None:
        self.llm = llm
        self.rag = rag
        self.ltm = ltm

    def _recall_feedback_precedents(self, state: AgentState) -> list[dict]:
        """Recall human feedback tied to the symbols under judgment.

        Per file with evidence: file_pattern is filtered in SQL against
        the evidence file_path (equality/prefix, 'missing' type excluded),
        then the evidence-involved symbol names are FTS-matched against
        feedback_content, bm25-ranked. Deduped across files, capped at 10.
        Any storage failure degrades to no precedents — never crashes
        the judgment.
        """
        if self.ltm is None:
            return []
        try:
            file_symbols = _evidence_symbols(
                state.rag_context.get("codebase", {}), state.evidence_pool
            )
            seen: set = set()
            precedents: list[dict] = []
            for file_path, symbols in file_symbols.items():
                for row in self.ltm.search_feedback(file_path, symbols, limit=10):
                    if row.get("id") in seen:
                        continue
                    seen.add(row.get("id"))
                    precedents.append({
                        "file_pattern": row.get("file_pattern"),
                        "rule_id": row.get("rule_id"),
                        "feedback_type": row.get("feedback_type"),
                        "feedback_content": row.get("feedback_content"),
                    })
            return precedents[:10]
        except Exception:
            return []

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
        # Global view of the pool: index == global evidence id. Batch
        # prompts embed each item's id so per-batch refs merge globally.
        evidence: list[dict] = []
        for idx, item in enumerate(state.evidence_pool):
            d = item.model_dump(mode="json")
            d["id"] = idx
            if symbols_present:
                d["snippet"] = None
            evidence.append(d)

        # -- Batch construction: group by file_path, then chunk to
        # _BATCH_SIZE. File grouping keeps each prompt's codebase context
        # coherent (one file's symbols verify one file's evidence).
        by_file: dict[str, list[int]] = {}
        for idx, item in enumerate(state.evidence_pool):
            by_file.setdefault(item.file_path or "unknown", []).append(idx)
        batches: list[list[int]] = []
        for file_ids in by_file.values():
            for i in range(0, len(file_ids), self._BATCH_SIZE):
                batches.append(file_ids[i:i + self._BATCH_SIZE])

        feedback_precedents = self._recall_feedback_precedents(state)

        risks: list[RiskItem] = []
        dismissed: list[DismissedEvidence] = []
        unadjudicated = 0
        for batch_ids in batches:
            id_set = set(batch_ids)
            batch_evidence = [evidence[i] for i in batch_ids]
            prompt = (
                "Judge these code-risk evidences. Return JSON only: "
                "{\"risks\": [{\"title\": str, \"category\": str, "
                "\"severity\": one of \"info\", \"low\", \"medium\", "
                "\"high\", \"critical\", "
                "\"description\": str, \"evidence_refs\": [int], "
                "\"suggestion\": str, \"file_path\": str, \"line_range\": [int, int], "
                "\"risk_score\": float (0.0-1.0)}], "
                "\"dismissed_evidence\": [{\"index\": int, \"reason\": str}]}.\n"
                "Not all evidence constitutes a real risk. Before confirming any "
                "evidence as a risk, VERIFY it against the provided codebase "
                "context — check whether the flagged pattern is actually present "
                "in the code, whether it is a false positive (e.g. test fixture, "
                "documentation example, or the symbol is in fact used elsewhere), "
                "and whether the severity is warranted given the context.\n"
                "If an evidence item is a false positive or irrelevant, put its "
                "id in dismissed_evidence with a reason — do NOT create a "
                "risk for it. Every risk MUST reference at least one evidence "
                "id (the \"id\" field) in evidence_refs, using ONLY ids from "
                "this batch; risks without valid evidence references "
                "will be discarded.\n\n"
                "Evidence for this batch (reference by its \"id\" field; each "
                "item carries its own file_path):\n"
                f"{json.dumps(batch_evidence)}\n\n"
                f"Security knowledge:\n{json.dumps(security)}\n\n"
                f"Codebase context (diff-file symbols overlapping changed lines, plus cross-file symbols actually called from them):\n{json.dumps(codebase_ctx)}"
            )
            if feedback_precedents:
                prompt += (
                    "\n\nHuman feedback precedents for these files and symbols "
                    "(past reviewer corrections; a false_positive entry means a "
                    "similar finding was rejected before — do not re-confirm the "
                    "same pattern unless this instance clearly differs, and weigh "
                    f"confirmed entries accordingly):\n{json.dumps(feedback_precedents)}"
                )
            try:
                response = self.llm.chat_json(
                    "You are a code risk judge.", prompt,
                    validator=_validate_judge_response,
                )
            except Exception:
                response = None

            if response is None:
                # Single-batch degradation isolation: this batch's evidence
                # stays unadjudicated, but the remaining batches are still
                # judged. The count is surfaced to Reporter so a degraded
                # run is reported as "degraded", never as a clean ✅.
                unadjudicated += len(batch_ids)
                continue

            batch_risks, batch_dismissed = self._parse_batch(
                response, state.evidence_pool, id_set
            )
            self._merge_batch(risks, batch_risks)
            dismissed.extend(batch_dismissed)

        rag_context = dict(state.rag_context)
        rag_context["security"] = security_raw

        # Every batch degraded: nothing was adjudicated this round. Return
        # WITHOUT the ``risks`` / ``dismissed_evidence`` keys: GraphState
        # uses whole-value replacement for these fields, so returning
        # empty lists would erase a previous round's valid adjudication.
        # An unread judgment must never wipe an earlier one — omitting the
        # keys preserves it.
        total = sum(len(b) for b in batches)
        if batches and unadjudicated == total:
            return {
                "phase": AgentPhase.REFLECTING,
                "rag_context": rag_context,
                "judge_unadjudicated_evidence": unadjudicated,
            }

        return {
            "risks": risks,
            "dismissed_evidence": dismissed,
            "phase": AgentPhase.REFLECTING,
            "rag_context": rag_context,
            "judge_unadjudicated_evidence": unadjudicated,
        }

    @staticmethod
    def _parse_batch(
        response: dict,
        pool: list,
        id_set: set[int],
    ) -> tuple[list[RiskItem], list[DismissedEvidence]]:
        """Parse one batch's judgment against the GLOBAL evidence pool.

        Refs are restricted to ``id_set`` (ids this batch actually saw):
        any other id would be a hallucinated reference. Dismissal still
        wins over citation (an id both cited and dismissed supports no
        risk), and a risk must keep at least one surviving ref.
        """
        raw_risks = response.get("risks", [])
        raw_dismissed = response.get("dismissed_evidence", [])

        # --- Parse dismissed evidence (id-validated against the batch) ---
        dismissed_idx: set[int] = set()
        dismissed: list[DismissedEvidence] = []
        for d in raw_dismissed:
            idx = d.get("index")
            if isinstance(idx, int) and idx in id_set:
                dismissed_idx.add(idx)
                dismissed.append(DismissedEvidence(
                    evidence=pool[idx],
                    reason=d.get("reason", "Dismissed by judge"),
                ))

        risks: list[RiskItem] = []
        for item in raw_risks:
            refs = item.get("evidence_refs", [])
            valid_refs = [
                index for index in refs
                if isinstance(index, int) and index in id_set
            ]
            # A risk must retain at least one non-dismissed valid reference.
            # If the LLM both cites and dismisses an index, dismissal wins for
            # that item; only its remaining evidence can support the risk.
            surviving_refs = [index for index in valid_refs if index not in dismissed_idx]
            chain = [pool[index] for index in surviving_refs]
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
        return risks, dismissed

    @staticmethod
    def _merge_batch(
        risks: list[RiskItem], batch_risks: list[RiskItem],
    ) -> None:
        """Merge one batch's risks into the accumulating global list.

        Global evidence ids mean two batches can independently surface the
        "same" finding (same title + file_path) with disjoint evidence
        chains. Merge those into one risk carrying the union of both
        chains instead of two near-duplicate report entries.
        """
        for risk in batch_risks:
            twin = next(
                (r for r in risks
                 if r.title == risk.title and r.file_path == risk.file_path),
                None,
            )
            if twin is None:
                risks.append(risk)
                continue
            known = {id(ev) for ev in twin.evidence_chain}
            twin.evidence_chain = [
                *twin.evidence_chain,
                *(ev for ev in risk.evidence_chain if id(ev) not in known),
            ]
            twin.risk_score = max(twin.risk_score, risk.risk_score)

