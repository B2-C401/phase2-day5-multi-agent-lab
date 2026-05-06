"""Multi-agent orchestration workflow."""

import logging
from typing import Any

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Orchestration uses a supervisor-driven while-loop (the canonical
    pattern from Anthropic's "Building effective agents" guide).

    For LangGraph integration see ``build()`` which returns a compiled
    StateGraph that can be visualised and traced via LangSmith.
    """

    def build(self) -> Any:
        """Create a LangGraph StateGraph and return the compiled graph.

        Requires ``pip install langgraph``.  The compiled graph supports
        ``.invoke()``, ``.stream()``, and LangSmith tracing.
        """
        from langgraph.graph import END, StateGraph  # type: ignore[import]

        supervisor = SupervisorAgent()
        researcher = ResearcherAgent()
        analyst = AnalystAgent()
        writer = WriterAgent()

        def _wrap(agent: BaseAgent):
            def node(state: dict[str, Any]) -> dict[str, Any]:
                rs = ResearchState.model_validate(state)
                updated = agent.run(rs)
                return updated.model_dump()

            node.__name__ = agent.name
            return node

        def _route(state: dict[str, Any]) -> str:
            history: list[str] = state.get("route_history", [])
            if not history:
                return END  # type: ignore[return-value]
            last = history[-1]
            return END if last == "done" else last  # type: ignore[return-value]

        graph: StateGraph = StateGraph(dict)
        graph.add_node("supervisor", _wrap(supervisor))
        graph.add_node("researcher", _wrap(researcher))
        graph.add_node("analyst", _wrap(analyst))
        graph.add_node("writer", _wrap(writer))

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            _route,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                END: END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the supervisor-loop workflow and return final state."""

        supervisor = SupervisorAgent()
        workers: dict[str, BaseAgent] = {
            "researcher": ResearcherAgent(),
            "analyst": AnalystAgent(),
            "writer": WriterAgent(),
        }
        settings = get_settings()

        with trace_span("multi_agent_workflow", {"query": state.request.query}):
            while state.iteration < settings.max_iterations:
                with trace_span("supervisor", {"iteration": state.iteration}):
                    state = supervisor.run(state)
                last_route = state.route_history[-1] if state.route_history else "done"

                if last_route == "done":
                    logger.info(
                        "Workflow complete after %d iteration(s)", state.iteration
                    )
                    break

                agent = workers.get(last_route)
                if agent is None:
                    logger.error("Unknown route '%s' — aborting", last_route)
                    state.errors.append(f"Unknown route: {last_route}")
                    break

                try:
                    with trace_span(last_route, {"iteration": state.iteration}):
                        state = agent.run(state)
                except AgentExecutionError as exc:
                    logger.error("Agent '%s' raised error: %s", last_route, exc)
                    state.errors.append(str(exc))
                    # Leave the state field (e.g. research_notes) as None so the
                    # supervisor's consecutive-failure guard will eventually abort.
            else:
                logger.warning(
                    "Reached max_iterations=%d without finishing", settings.max_iterations
                )

        return state
