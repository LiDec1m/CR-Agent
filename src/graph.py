"""LangGraph StateGraph assembly: connect nodes with conditional edges."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END

from src.models import AgentState, AgentPhase
from src.nodes import (
    JudgeNode,
    PlannerNode,
    ReflectionNode,
    ReporterNode,
    ToolRouterNode,
)
from src.rules.registry import ToolRegistry
from src.llm.client import LLMClient
from src.rag.retriever import RAGRetriever
from src.memory.long_term import LongTermMemory


# -- State schema for LangGraph -------------------------------------------
#
# List fields annotated with ``operator.add`` are accumulated across
# reflection rounds (evidence and notes). Hunk-keyed tool queues, execution
# history and outcomes use replacement: their writer returns the complete
# merged mapping/list to keep one authoritative per-hunk state.

class GraphState(TypedDict, total=False):
    repo: str
    commit_sha: str
    raw_diff: str
    hunks: list
    # Queue for the CURRENT ToolRouter round, keyed by ``file_path:new_start``.
    pending_tools_by_hunk: dict
    planning_reasons: dict
    rule_outcomes: list
    phase: AgentPhase
    fatal_error: str
    judge_unadjudicated_evidence: int
    evidence_pool: Annotated[list, operator.add]
    risks: list
    dismissed_evidence: list
    reflection_round: int
    reflection_notes: Annotated[list, operator.add]
    needs_more_analysis: bool
    rag_context: dict
    report: object


def _dict_to_state(data: dict) -> AgentState:
    """Convert a plain dict back to an AgentState instance.

    Any keys that are not valid AgentState fields are silently dropped
    so that LangGraph-internal bookkeeping does not cause validation errors.
    """
    valid_fields = set(AgentState.model_fields.keys())
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return AgentState(**filtered)


def build_graph(
    llm: LLMClient,
    rag: RAGRetriever,
    ltm: LongTermMemory,
    registry: ToolRegistry,
    max_rounds: int = 3,
    checkpointer=None,
):
    """Build and compile the LangGraph workflow.

    The graph wires five nodes -- planner, tool_router, judge, reflection,
    reporter -- with a conditional edge from reflection back to tool_router
    (when more analysis is needed) or to reporter (which builds the final
    RiskReport and routes to END). Routing every exit through reporter
    makes a report-less finish structurally impossible.

    ``ltm`` is consumed only by the Judge (false-positive feedback recall).
    ``checkpointer`` enables short-term memory persistence: every node
    transition is checkpointed per ``thread_id``, enabling state replay
    (``graph.get_state_history``) and resumption of interrupted runs.
    """
    planner = PlannerNode(llm, rag)
    tool_router = ToolRouterNode(registry, rag)
    judge = JudgeNode(llm, rag, ltm=ltm)
    reflection = ReflectionNode(llm, max_rounds=max_rounds)
    reporter = ReporterNode()

    # -- Node wrappers: accept dict, convert to AgentState, return dict --

    def planner_node(state: dict) -> dict:
        s = _dict_to_state(state)
        return planner(s)

    def tool_router_node(state: dict) -> dict:
        s = _dict_to_state(state)
        return tool_router(s)

    def judge_node(state: dict) -> dict:
        s = _dict_to_state(state)
        return judge(s)

    def reflection_node(state: dict) -> dict:
        s = _dict_to_state(state)
        return reflection(s)

    def reporter_node(state: dict) -> dict:
        s = _dict_to_state(state)
        return reporter(s)

    # -- Conditional edge: decide where to go after reflection --

    def reflection_route(state: dict) -> str:
        s = _dict_to_state(state)
        if s.needs_more_analysis and s.reflection_round < max_rounds:
            return "tool_router"
        return "reporter"

    # -- Conditional edge: planner fail-fast. When chat_json degraded
    # after its repair retries there is nothing to schedule -- routing on
    # would silently run an empty plan. Route straight to reporter so a
    # failed report is structurally guaranteed. A legitimately empty plan
    # (valid JSON, zero assignments) does NOT take this path.

    def planner_route(state: dict) -> str:
        s = _dict_to_state(state)
        if s.fatal_error:
            return "reporter"
        return "tool_router"

    # -- Assemble the workflow --

    workflow = StateGraph(GraphState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("tool_router", tool_router_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("reporter", reporter_node)

    workflow.set_entry_point("planner")
    workflow.add_conditional_edges(
        "planner",
        planner_route,
        {
            "tool_router": "tool_router",
            "reporter": "reporter",
        },
    )
    workflow.add_edge("tool_router", "judge")
    workflow.add_edge("judge", "reflection")
    workflow.add_conditional_edges(
        "reflection",
        reflection_route,
        {
            "tool_router": "tool_router",
            "reporter": "reporter",
        },
    )
    workflow.add_edge("reporter", END)

    return workflow.compile(checkpointer=checkpointer)
