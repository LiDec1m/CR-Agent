"""Reporter node: build the final RiskReport from the agent state."""

from __future__ import annotations

from src.models import (
    AgentPhase, AgentState, HunkSummary, RiskReport, RuleOutcomeStatus,
)
from src.nodes.tool_router import hunk_key


class ReporterNode:
    """Assemble the final RiskReport from the current state."""

    def __call__(self, state: AgentState) -> dict:
        return {"phase": AgentPhase.DONE, "report": self._build_report(state)}

    @staticmethod
    def _build_report(state: AgentState) -> RiskReport:
        """Build the report and derive global summaries from hunk-level state."""
        # Status is derived, never trusted from elsewhere: failed when the
        # pipeline aborted (planner fail-fast), degraded when the Judge
        # could not adjudicate every evidence item, completed otherwise.
        if state.fatal_error:
            status = "failed"
        elif state.judge_unadjudicated_evidence > 0:
            status = "degraded"
        else:
            status = "completed"

        files_scanned = list(dict.fromkeys(hunk.file_path for hunk in state.hunks))
        # rules_executed is derived from the durable outcome ledger: every
        # outcome records one rule execution (any status), so the global
        # list is the set of rules that actually ran somewhere.
        rule_names = sorted({
            outcome.rule for outcome in state.rule_outcomes
        })
        outcomes = {(outcome.hunk_key, outcome.rule): outcome.status
                    for outcome in state.rule_outcomes}
        limited = 0
        for hunk in state.hunks:
            statuses = [
                status for (key, _), status in outcomes.items()
                if key == hunk_key(hunk)
            ]
            if not any(status in {
                RuleOutcomeStatus.CLEAN, RuleOutcomeStatus.EVIDENCE_PRODUCED,
            } for status in statuses):
                limited += 1
        overall_risk_score = max((risk.risk_score for risk in state.risks), default=0.0)
        if status == "failed":
            summary = f"Analysis aborted: {state.fatal_error}"
        elif status == "degraded":
            summary = (
                f"Judgment incomplete: {state.judge_unadjudicated_evidence} "
                "evidence item(s) were never adjudicated. "
                + (
                    f"Detected {len(state.risks)} risk(s)."
                    if state.risks else "No significant risks detected."
                )
            )
        else:
            summary = (
                f"Detected {len(state.risks)} risk(s)."
                if state.risks else "No significant risks detected."
            )
        return RiskReport(
            repo=state.repo, commit_sha=state.commit_sha, summary=summary,
            status=status,
            risks=state.risks, dismissed_evidence=state.dismissed_evidence,
            overall_risk_score=overall_risk_score, files_scanned=files_scanned,
            total_hunks=len(state.hunks), rules_executed=rule_names,
            coverage_limited_hunks=limited,
            reflection_rounds=state.reflection_round,
            unadjudicated_evidence=state.judge_unadjudicated_evidence,
            hunk_summaries=ReporterNode._build_hunk_summaries(state),
        )

    @staticmethod
    def _build_hunk_summaries(state: AgentState) -> list[HunkSummary]:
        """Roll the state up to one HunkSummary per diff hunk.

        Rule statuses come from the durable outcome ledger; evidence counts
        from each evidence's ``hunk_keys`` (merge-style dedup means one
        evidence can cover several hunks); risk titles from risks whose
        evidence chains touch the hunk. Hunks with no outcome, evidence
        or risk still get a row (rule_statuses empty -> "unexamined" in
        rendering) so the summary covers every hunk in the diff.
        """
        statuses: dict[str, dict[str, str]] = {}
        for outcome in state.rule_outcomes:
            statuses.setdefault(outcome.hunk_key, {})[outcome.rule] = (
                outcome.status.value if hasattr(outcome.status, "value")
                else str(outcome.status)
            )
        evidence_counts: dict[str, int] = {}
        for ev in state.evidence_pool:
            for key in ev.hunk_keys:
                evidence_counts[key] = evidence_counts.get(key, 0) + 1
        risk_titles: dict[str, list[str]] = {}
        for risk in state.risks:
            for key in {hk for ev in risk.evidence_chain for hk in ev.hunk_keys}:
                risk_titles.setdefault(key, [])
                if risk.title not in risk_titles[key]:
                    risk_titles[key].append(risk.title)

        summaries = []
        for hunk in state.hunks:
            key = hunk_key(hunk)
            summaries.append(HunkSummary(
                hunk_key=key,
                rule_statuses=statuses.get(key, {}),
                evidence_count=evidence_counts.get(key, 0),
                risk_titles=risk_titles.get(key, []),
            ))
        return summaries
