"""Critic agent — optional fact-check and quality gate."""

import logging
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a fact-checking editor. Review the draft answer below and return a JSON object
with these fields:
  "citation_coverage": int   — number of [n] inline citations found
  "unsupported_claims": list[str]  — claims that lack any citation
  "hallucination_flags": list[str] — claims that contradict or go beyond the provided sources
  "quality_score": float  — overall quality 0.0–10.0
  "feedback": str  — one paragraph of constructive feedback

Return ONLY valid JSON, no markdown fences.\
"""


class CriticAgent(BaseAgent):
    """Validates final answer quality; appends findings to state trace."""

    name = "critic"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Check ``state.final_answer`` and record quality metrics in trace."""

        if not state.final_answer:
            logger.warning("CriticAgent: final_answer is empty, skipping")
            return state

        context = (
            f"Sources provided:\n"
            + "\n".join(f"[{i+1}] {s.title}: {s.snippet}" for i, s in enumerate(state.sources))
            + f"\n\nDraft answer:\n{state.final_answer}"
        )

        try:
            resp = self._llm.complete(_SYSTEM_PROMPT, context)
            import json

            review = json.loads(resp.content)
            state.add_trace_event("critic_review", review)
            logger.info(
                "CriticAgent: quality_score=%.1f citations=%d",
                review.get("quality_score", 0),
                review.get("citation_coverage", 0),
            )
        except Exception as exc:
            logger.error("CriticAgent failed: %s", exc)
            state.errors.append(f"CriticAgent error: {exc}")
            raise AgentExecutionError(f"CriticAgent failed: {exc}") from exc

        return state

    # ------------------------------------------------------------------
    # Lightweight heuristics (no LLM call needed)
    # ------------------------------------------------------------------

    @staticmethod
    def count_citations(text: str) -> int:
        """Count distinct [n] citation markers in *text*."""
        return len(set(re.findall(r"\[\d+\]", text)))
