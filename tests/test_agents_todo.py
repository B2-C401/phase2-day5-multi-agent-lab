"""Tests for agent routing and state transitions."""

import pytest

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _state(query: str = "Explain multi-agent systems") -> ResearchState:
    return ResearchState(request=ResearchQuery(query=query))


class TestSupervisorRouting:
    def test_routes_researcher_when_no_notes(self) -> None:
        state = _state()
        result = SupervisorAgent().run(state)
        assert result.route_history[-1] == "researcher"

    def test_routes_analyst_when_research_done(self) -> None:
        state = _state()
        state.research_notes = "Some research notes."
        result = SupervisorAgent().run(state)
        assert result.route_history[-1] == "analyst"

    def test_routes_writer_when_analysis_done(self) -> None:
        state = _state()
        state.research_notes = "Some research notes."
        state.analysis_notes = "Some analysis."
        result = SupervisorAgent().run(state)
        assert result.route_history[-1] == "writer"

    def test_routes_done_when_all_complete(self) -> None:
        state = _state()
        state.research_notes = "Notes."
        state.analysis_notes = "Analysis."
        state.final_answer = "Answer."
        result = SupervisorAgent().run(state)
        assert result.route_history[-1] == "done"

    def test_iteration_increments(self) -> None:
        state = _state()
        assert state.iteration == 0
        result = SupervisorAgent().run(state)
        assert result.iteration == 1

    def test_max_iterations_guard(self) -> None:
        state = _state()
        # Simulate being at the iteration cap
        from multi_agent_research_lab.core.config import get_settings

        state.iteration = get_settings().max_iterations
        result = SupervisorAgent().run(state)
        assert result.route_history[-1] == "done"

    def test_consecutive_failure_guard(self) -> None:
        state = _state()
        # Simulate researcher failing 3 times in a row
        state.route_history = ["researcher", "researcher", "researcher"]
        state.iteration = 3
        result = SupervisorAgent().run(state)
        assert result.route_history[-1] == "done"
        assert any("researcher" in e for e in result.errors)
