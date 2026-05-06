"""Analyst agent — turns research notes into structured insights."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a critical analyst. Given research notes on a topic, produce structured analysis
(200–350 words) tailored for: {audience}.

Your analysis must include:
- **Key claims** (bulleted list)
- **Evidence strength** for each claim: Strong / Moderate / Weak
- **Conflicting viewpoints or gaps** in the research
- **Synthesis / recommendation** for the target audience

Be concise and rigorous. Do not repeat the research notes verbatim.\
"""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.analysis_notes``."""

        if not state.research_notes:
            logger.warning("AnalystAgent: research_notes is empty, skipping")
            state.errors.append("AnalystAgent called with empty research_notes")
            return state

        system = _SYSTEM_PROMPT.format(audience=state.request.audience)
        user_prompt = (
            f"Research notes:\n{state.research_notes}\n\n"
            "Provide structured analysis."
        )

        try:
            resp = self._llm.complete(system, user_prompt)
            state.analysis_notes = resp.content
            state.add_trace_event(
                "analyst_done",
                {
                    "analysis_length": len(resp.content),
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cost_usd": resp.cost_usd,
                },
            )
            logger.info("AnalystAgent: analysis written (%d chars)", len(resp.content))
        except Exception as exc:
            logger.error("LLM call failed in AnalystAgent: %s", exc)
            state.errors.append(f"AnalystAgent LLM error: {exc}")
            raise AgentExecutionError(f"AnalystAgent failed: {exc}") from exc

        return state
