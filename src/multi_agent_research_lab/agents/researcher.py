"""Researcher agent — searches for sources and writes research notes."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a research assistant. Given a user query and a set of source documents,
write concise research notes (300–500 words) that:
- Summarise the most relevant findings from the provided sources
- Highlight key facts, data points, and specific claims
- Flag any conflicting information or gaps across sources
- Cite sources inline as [1], [2], etc.

Keep the notes factual and objective. Do not invent information not present in the sources.\
"""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._search = SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.sources`` and ``state.research_notes``."""

        query = state.request.query
        logger.info("ResearcherAgent: searching for '%s'", query[:80])

        # Search phase
        try:
            sources = self._search.search(query, max_results=state.request.max_sources)
            state.sources = sources
            logger.info("ResearcherAgent: retrieved %d source(s)", len(sources))
        except Exception as exc:
            logger.error("Search failed: %s", exc)
            state.errors.append(f"Search error: {exc}")
            sources = []

        # Summarise phase
        source_text = "\n\n".join(
            f"[{i + 1}] {s.title}\n{s.snippet}" for i, s in enumerate(sources)
        )
        user_prompt = (
            f"Query: {query}\n\n"
            f"Sources:\n{source_text or '(no sources retrieved)'}\n\n"
            "Write research notes."
        )

        try:
            resp = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            state.research_notes = resp.content
            state.add_trace_event(
                "researcher_done",
                {
                    "sources_count": len(sources),
                    "notes_length": len(resp.content),
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cost_usd": resp.cost_usd,
                },
            )
            logger.info("ResearcherAgent: notes written (%d chars)", len(resp.content))
        except Exception as exc:
            logger.error("LLM call failed in ResearcherAgent: %s", exc)
            state.errors.append(f"ResearcherAgent LLM error: {exc}")
            raise AgentExecutionError(f"ResearcherAgent failed: {exc}") from exc

        return state
