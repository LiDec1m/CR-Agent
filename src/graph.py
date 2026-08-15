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
# List fields annotated with ``operator.add`` are *accumulated* across
# reflection rounds (evidence, executed rules, notes).  All other fields
# use the default replacement strategy so each node's return value
# overwrites the previous value.

class GraphState(TypedDict, total=False):
    repo: str
    commit_sha: str
    raw_diff: str
    hunks: list
    # Queue of rules for the CURRENT tool_router round: written by
    # Planner / Reflection, consumed and cleared by ToolRouter.
    pending_tools: list
    phase: AgentPhase
    evidence_pool: Annotated[list, operator.add]
    rules_executed: Annotated[list, operator.add]
    risks: list
    reflection_round: int
    reflection_notes: Annotated[list, operator.add]
    needs_more_analysis: bool
    long_term_feedback: list
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

    ``checkpointer`` enables short-term memory persistence: every node
    transition is checkpointed per ``thread_id``, enabling state replay
    (``graph.get_state_history``) and resumption of interrupted runs.
    """
    planner = PlannerNode(llm, rag, ltm)
    tool_router = ToolRouterNode(registry, rag)
    judge = JudgeNode(llm, rag)
    reflection = ReflectionNode(llm, ltm=ltm, max_rounds=max_rounds)
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

    # -- Assemble the workflow --

    workflow = StateGraph(GraphState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("tool_router", tool_router_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("reporter", reporter_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "tool_router")
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
