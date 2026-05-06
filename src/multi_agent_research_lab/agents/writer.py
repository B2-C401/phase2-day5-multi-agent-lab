"""Writer agent — synthesises a final answer from research and analysis."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a technical writer. Write a clear, well-structured response (~500 words)
for the following audience: {audience}.

Guidelines:
- Answer the original query directly in the opening paragraph
- Integrate findings from the research notes and analyst insights
- Use markdown headings (##) and bullet points where they aid clarity
- Cite sources inline as [1], [2], etc. where relevant
- End with a "## Key Takeaways" section (3–5 bullets)

Do not pad the response. Prioritise clarity over length.\
"""


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.final_answer``."""

        if not state.research_notes:
            logger.warning("WriterAgent: research_notes is empty, skipping")
            state.errors.append("WriterAgent called with empty research_notes")
            return state

        context_parts = [
            f"Original Query: {state.request.query}",
            f"Research Notes:\n{state.research_notes}",
        ]
        if state.analysis_notes:
            context_parts.append(f"Analyst Notes:\n{state.analysis_notes}")

        source_refs = "\n".join(
            f"[{i + 1}] {s.title} — {s.url or 'no URL'}"
            for i, s in enumerate(state.sources)
        )
        if source_refs:
            context_parts.append(f"Source list:\n{source_refs}")

        system = _SYSTEM_PROMPT.format(audience=state.request.audience)
        user_prompt = "\n\n".join(context_parts) + "\n\nWrite the final response."

        try:
            resp = self._llm.complete(system, user_prompt)
            state.final_answer = resp.content
            state.add_trace_event(
                "writer_done",
                {
                    "answer_length": len(resp.content),
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cost_usd": resp.cost_usd,
                },
            )
            logger.info("WriterAgent: final answer written (%d chars)", len(resp.content))
        except Exception as exc:
            logger.error("LLM call failed in WriterAgent: %s", exc)
            state.errors.append(f"WriterAgent LLM error: {exc}")
            raise AgentExecutionError(f"WriterAgent failed: {exc}") from exc

        return state
