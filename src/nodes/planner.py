"""LLM-driven planner that schedules deterministic rules per diff hunk."""

from __future__ import annotations

import json
import logging

from src.llm.client import LLMClient
from src.models import AgentPhase, AgentState, HunkInfo
from src.rag.retriever import RAGRetriever
from src.rules import registry
from src.nodes.tool_router import hunk_key

logger = logging.getLogger(__name__)


class PlannerNode:
    """Load historical-risk context and choose deterministic checks per hunk."""

    def __init__(self, llm: LLMClient, rag: RAGRetriever) -> None:
        self.llm = llm
        self.rag = rag

    def __call__(self, state: AgentState) -> dict:
        history: list[dict] = []
        seen_history_ids: set = set()
        # Embed-call sharing: identical (file_path, added_code) hunks
        # need identical retrieval, and each embed() is a real API call.
        # NOTE: the key must include file_path — the same code in two
        # files recalls that file's history, not the other's (same
        # lesson as the exec_cache scoping fix).
        history_cache: dict[tuple[str, str], list[dict]] = {}
        for hunk in state.hunks:
            cache_key = (hunk.file_path, hunk.added_code)
            if cache_key in history_cache:
                hunk_history = history_cache[cache_key]
            else:
                try:
                    hunk_history = self.rag.search_history(
                        f"{hunk.file_path} {hunk.added_code}", hunk.file_path
                    )
                except Exception:
                    hunk_history = []
                history_cache[cache_key] = hunk_history
            for row in hunk_history:
                # Cross-hunk dedup: the same history row recalled for
                # several hunks of one file must appear once in the
                # prompt, not once per hunk.
                row_id = row.get("id")
                if row_id in seen_history_ids:
                    continue
                seen_history_ids.add(row_id)
                history.append(row)

        # llm_assisted is excluded from the initial plan by construction:
        # it never appears in the available list, and _normalize_plan only
        # accepts names from that list — no prompt-level prohibition needed.
        available = [rule for rule in registry.list_all() if rule != "llm_assisted"]
        changed = "\n\n".join(
            f"Hunk key: {hunk_key(hunk)}\nFile: {hunk.file_path}\n"
            f"Added code:\n{hunk.added_code or '(none)'}"
            for hunk in state.hunks
        )
        # Prompt slimming: keep the fields the Planner LLM can actually
        # reason over. id/thread_id/created_at stay on the dict (dedup /
        # audit) but never enter the prompt — same pattern as the Judge
        # and feedback channels' slim-before-prompt mapping.
        history_slim = [
            {
                "file_path": h.get("file_path"),
                "diff_summary": h.get("diff_summary"),
                "risk_titles": h.get("risk_titles", []),
                "risk_categories": h.get("risk_categories", []),
                "overall_score": h.get("overall_score"),
            }
            for h in history
        ]
        prompt = (
            "Analyze each diff hunk and select relevant deterministic analysis "
            "rules. Return JSON only: {\"plan_by_hunk\": "
            "{\"file_path:new_start\": {\"tools\": [str], \"reason\": str}}}. "
            "Every plan_by_hunk key must be one supplied Hunk key. \"tools\" "
            "must only contain rule names from the available rules list. "
            "\"reason\" is a short justification of why these tools were "
            "selected for this hunk.\n\n"
            f"Changed hunks:\n{changed}\n\n"
            f"Historical risks:\n{json.dumps(history_slim)}\n\n"
            f"Available deterministic rules:\n{json.dumps(available)}"
        )
        exc: Exception | None = None
        try:
            response = self.llm.chat_json("You are a code risk planner.", prompt)
        except Exception as caught:
            response = None
            exc = caught

        # Fail-fast: chat_json returning None means the LLM call degraded
        # after its internal repair retries — we do NOT know what the model
        # would have planned. Folding that into an empty plan would disguise
        # "could not plan" as "nothing worth checking" and the final report
        # would show a bogus "no risks" verdict (observed in a real run).
        # A legitimately empty plan (LLM returned JSON with zero valid
        # assignments) is NOT a failure and continues through the pipeline.
        if response is None:
            logger.error(
                "Planner LLM call degraded after repair retries: %s", exc,
            )
            detail = f" ({type(exc).__name__}: {exc})" if exc else ""
            return {
                "fatal_error": (
                    "Planning failed: LLM unavailable after repair retries"
                    + detail
                ),
                "phase": AgentPhase.REPORTING,
                "rag_context": {"history": history},
            }

        raw_plan = response.get("plan_by_hunk", {}) if response else {}

        valid_keys = {hunk_key(hunk) for hunk in state.hunks}
        pending, reasons = self._normalize_plan(
            raw_plan, valid_keys, set(available)
        )
        return {
            "pending_tools_by_hunk": pending,
            "planning_reasons": reasons,
            "phase": AgentPhase.TOOL_ROUTING,
            "rag_context": {"history": history},
        }

    @staticmethod
    def _normalize_plan(
        raw_plan: object, valid_keys: set[str], available: set[str],
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        """Validate hunk assignments; return (plan, reasons) pair.

        Drops malformed entries, unknown hunk keys, non-string rule names,
        rules not in the available list, and duplicate assignments. The
        reason is kept as an empty string when the LLM did not provide one.
        """
        if not isinstance(raw_plan, dict):
            return {}, {}
        plan: dict[str, list[str]] = {}
        reasons: dict[str, str] = {}
        for key, entry in raw_plan.items():
            if not isinstance(key, str) or key not in valid_keys:
                continue
            if isinstance(entry, list):
                # Tolerate the legacy bare-list form.
                tools_raw: list = entry
                reason = ""
            elif isinstance(entry, dict):
                tools_raw = entry.get("tools", [])
                reason = entry.get("reason", "")
                if not isinstance(reason, str):
                    reason = ""
            else:
                continue
            if not isinstance(tools_raw, list):
                continue
            accepted = list(dict.fromkeys(
                rule for rule in tools_raw
                if isinstance(rule, str) and rule in available
            ))
            if accepted:
                plan[key] = accepted
                reasons[key] = reason
        return plan, reasons
