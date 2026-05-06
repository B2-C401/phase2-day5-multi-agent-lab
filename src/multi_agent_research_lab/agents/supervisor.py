"""Supervisor / router agent."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update ``state.route_history`` with the next route.

        Routing order: researcher → analyst → writer → done.
        Guards: max_iterations cap, consecutive-failure abort.
        """
        settings = get_settings()

        # Hard cap on total iterations
        if state.iteration >= settings.max_iterations:
            logger.warning("Max iterations (%d) reached — forcing done", settings.max_iterations)
            state.record_route("done")
            return state

        # Abort if the same agent failed three iterations in a row
        if len(state.route_history) >= 3:
            last_three = state.route_history[-3:]
            if len(set(last_three)) == 1 and last_three[0] not in ("done",):
                stuck_agent = last_three[0]
                logger.error("Agent '%s' stuck for 3 iterations — aborting", stuck_agent)
                state.errors.append(
                    f"Agent '{stuck_agent}' failed repeatedly; workflow aborted."
                )
                state.record_route("done")
                return state

        # Core routing decision
        if state.research_notes is None:
            next_route = "researcher"
        elif state.analysis_notes is None:
            next_route = "analyst"
        elif state.final_answer is None:
            next_route = "writer"
        else:
            next_route = "done"

        logger.info(
            "Supervisor → %s  (iteration %d / %d)",
            next_route,
            state.iteration,
            settings.max_iterations,
        )
        state.add_trace_event(
            "supervisor_route",
            {"next": next_route, "iteration": state.iteration},
        )
        state.record_route(next_route)
        return state
