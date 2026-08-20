"""Rule-routing node: execute hunk-targeted rules and record outcomes."""

from __future__ import annotations

import logging

from src.models import (
    AgentPhase, AgentState, ChangeType, DiffLine, Evidence, HunkInfo,
    RuleOutcome, RuleOutcomeStatus,
)
from src.rag.retriever import RAGRetriever
from src.rules.llm_assisted import LLMAnalysisDegraded
from src.rules.registry import ToolRegistry

logger = logging.getLogger(__name__)


def hunk_key(hunk: HunkInfo) -> str:
    """Stable key for rule scheduling and outcome attribution."""
    return f"{hunk.file_path}:{hunk.new_start}"


class ToolRouterNode:
    """Run only the rules scheduled for each hunk and audit every result."""

    def __init__(self, registry: ToolRegistry, rag: RAGRetriever) -> None:
        self.registry = registry
        self.rag = rag

    def __call__(self, state: AgentState) -> dict:
        evidence_pool: list[Evidence] = []
        outcomes = list(state.rule_outcomes)
        executed = {
            key: list(rules)
            for key, rules in state.executed_tools_by_hunk.items()
        }
        codebase = self._get_codebase(state)
        available = set(self.registry.list_all())
        hunk_by_key = {hunk_key(hunk): hunk for hunk in state.hunks}

        # Dedup cache: when two hunks fall inside the same indexed symbol,
        # _enrich_hunk produces identical added_code for both. Running the
        # same rule on the same content twice wastes compute and creates
        # duplicate Evidence. Cache by (rule, enriched_code) so the second
        # hunk reuses the first's result.
        exec_cache: dict[tuple[str, str], tuple[list[Evidence], RuleOutcomeStatus]] = {}

        for key, rule_names in state.pending_tools_by_hunk.items():
            hunk = hunk_by_key.get(key)
            if hunk is None:
                logger.warning("Skipping rules for unknown hunk key %s", key)
                continue
            effective_hunk = self._enrich_hunk(hunk, codebase)
            enriched_code = effective_hunk.added_code

            for rule_name in dict.fromkeys(rule_names):
                if rule_name not in available:
                    logger.warning("Skipping unknown rule %s for %s", rule_name, key)
                    continue

                cache_key = (rule_name, enriched_code)
                if cache_key in exec_cache:
                    evidences, status = exec_cache[cache_key]
                else:
                    try:
                        evidences = self.registry.execute(rule_name, effective_hunk) or []
                    except LLMAnalysisDegraded as exc:
                        outcomes = self._replace_outcome(
                            outcomes,
                            RuleOutcome(
                                hunk_key=key, rule=rule_name,
                                status=RuleOutcomeStatus.DEGRADED, detail=str(exc),
                            ),
                        )
                        continue
                    except Exception as exc:
                        logger.warning(
                            "Rule %s failed on %s: %s: %s",
                            rule_name, key, type(exc).__name__, exc,
                        )
                        outcomes = self._replace_outcome(
                            outcomes,
                            RuleOutcome(
                                hunk_key=key, rule=rule_name,
                                status=RuleOutcomeStatus.FAILED,
                                detail=f"{type(exc).__name__}: {exc}",
                            ),
                        )
                        continue

                    status = (
                        RuleOutcomeStatus.EVIDENCE_PRODUCED
                        if evidences else RuleOutcomeStatus.CLEAN
                    )
                    exec_cache[cache_key] = (evidences, status)

                for ev in evidences:
                    if ev.file_path is None:
                        ev = ev.model_copy(update={"file_path": hunk.file_path})
                    evidence_pool.append(ev)
                outcomes = self._replace_outcome(
                    outcomes,
                    RuleOutcome(hunk_key=key, rule=rule_name, status=status),
                )
                # Only conclusive results (clean / evidence_produced) are
                # recorded as "executed". Degraded and failed attempts stay
                # in rule_outcomes for auditability but do NOT count as
                # completed — so the anti-idle check in Reflection will
                # still allow a retry for degraded hunks.
                executed.setdefault(key, [])
                if rule_name not in executed[key]:
                    executed[key].append(rule_name)

        rag_context = dict(state.rag_context)
        rag_context["codebase"] = codebase
        return {
            "evidence_pool": evidence_pool,
            "rule_outcomes": outcomes,
            "executed_tools_by_hunk": executed,
            "pending_tools_by_hunk": {},
            "phase": AgentPhase.JUDGING,
            "rag_context": rag_context,
        }

    def _get_codebase(self, state: AgentState) -> dict[str, list[dict]]:
        """Fetch codebase context once and cache it for later rounds."""
        existing = state.rag_context.get("codebase", {})
        if existing and state.reflection_round > 0:
            return dict(existing)
        codebase: dict[str, list[dict]] = {}
        for hunk in state.hunks:
            if hunk.file_path in codebase:
                continue
            try:
                codebase[hunk.file_path] = self.rag.search_codebase(hunk.file_path)
            except Exception:
                codebase[hunk.file_path] = []
        return codebase

    @staticmethod
    def _replace_outcome(
        outcomes: list[RuleOutcome], replacement: RuleOutcome,
    ) -> list[RuleOutcome]:
        """Keep exactly the latest outcome for one hunk/rule execution."""
        return [
            outcome for outcome in outcomes
            if not (
                outcome.hunk_key == replacement.hunk_key
                and outcome.rule == replacement.rule
            )
        ] + [replacement]

    def _enrich_hunk(
        self, hunk: HunkInfo, codebase: dict[str, list[dict]],
    ) -> HunkInfo:
        """Use same-file indexed symbols overlapping the changed line range.

        Complete function source lets AST rules parse partial diffs. The explicit
        file-path filter prevents a fuzzy RAG hit from a different file being
        attributed to this hunk.
        """
        symbols = codebase.get(hunk.file_path, [])
        if not symbols:
            return hunk
        hunk_start = hunk.new_start
        hunk_end = hunk.new_start + hunk.new_count
        # Collect (start, end, content) for all overlapping same-file symbols.
        raw: list[tuple[int, int, str]] = []
        for symbol in symbols:
            if symbol.get("file_path") not in (None, hunk.file_path):
                continue
            line_range = symbol.get("line_range", "")
            if "-" not in line_range:
                continue
            try:
                start, end = (int(part) for part in line_range.split("-", 1))
            except ValueError:
                continue
            if start <= hunk_end and end >= hunk_start and symbol.get("content"):
                raw.append((start, end, symbol["content"]))

        # Deduplicate nested symbols: if symbol A's line range fully
        # contains symbol B's range, B's content is already a substring
        # of A's content (AST indexer captures the full outer function
        # including nested defs). Keeping both would feed AST rules
        # duplicate definitions. Sort by range width descending so the
        # outermost symbols are considered first.
        raw.sort(key=lambda t: (t[0], -(t[1] - t[0])))
        contents: list[str] = []
        kept_ranges: list[tuple[int, int]] = []
        for s, e, content in raw:
            if any(ks <= s and e <= ke for ks, ke in kept_ranges):
                continue  # fully contained in an already-kept symbol
            contents.append(content)
            kept_ranges.append((s, e))
        if not contents:
            return hunk
        first = hunk.lines[0] if hunk.lines else None
        return HunkInfo(
            file_path=hunk.file_path, old_start=hunk.old_start,
            old_count=hunk.old_count, new_start=hunk.new_start,
            new_count=hunk.new_count, section_header=hunk.section_header,
            lines=[DiffLine(
                content="\n\n".join(contents), change_type=ChangeType.ADDED,
                old_line_no=first.old_line_no if first else None,
                new_line_no=first.new_line_no if first else hunk.new_start,
            )],
        )
