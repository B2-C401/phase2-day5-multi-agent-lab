"""Search client abstraction for ResearcherAgent."""

import logging

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client.

    Uses Tavily when TAVILY_API_KEY is set; falls back to a deterministic mock.
    """

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        settings = get_settings()
        if settings.tavily_api_key:
            return self._tavily_search(query, max_results, settings.tavily_api_key)
        logger.warning("TAVILY_API_KEY not set — using mock search results")
        return self._mock_search(query, max_results)

    # ------------------------------------------------------------------
    # Providers
    # ------------------------------------------------------------------

    def _tavily_search(self, query: str, max_results: int, api_key: str) -> list[SourceDocument]:
        from tavily import TavilyClient  # optional dependency

        client = TavilyClient(api_key=api_key)
        resp = client.search(query=query, max_results=max_results)
        return [
            SourceDocument(
                title=r.get("title", "Untitled"),
                url=r.get("url"),
                snippet=r.get("content", ""),
                metadata={"score": r.get("score")},
            )
            for r in resp.get("results", [])
        ]

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Return plausible-looking placeholder documents for offline development."""
        short_q = query[:60]
        docs = [
            SourceDocument(
                title=f"Overview of {short_q}",
                url="https://arxiv.org/mock/2401.00001",
                snippet=(
                    f"Recent advances in {short_q} have shown significant progress. "
                    "Multiple studies indicate strong performance gains when applying "
                    "modern techniques to this domain, with notable improvements in "
                    "accuracy, efficiency, and scalability."
                ),
                metadata={"source": "mock"},
            ),
            SourceDocument(
                title=f"Practical guide to {short_q}",
                url="https://blog.example.com/mock/guide",
                snippet=(
                    f"This guide covers the fundamentals of {short_q}, including "
                    "best practices, common pitfalls, and production deployment "
                    "considerations. The community has converged on several patterns "
                    "that improve reliability and reduce operational overhead."
                ),
                metadata={"source": "mock"},
            ),
            SourceDocument(
                title=f"Benchmarking {short_q}: a comparative study",
                url="https://papers.example.com/mock/bench",
                snippet=(
                    f"Comparative benchmarks for {short_q} reveal tradeoffs between "
                    "latency, cost, and quality. Systems using specialised components "
                    "outperform monolithic approaches by up to 40% on standard metrics "
                    "while reducing end-to-end latency."
                ),
                metadata={"source": "mock"},
            ),
            SourceDocument(
                title=f"Production lessons: {short_q}",
                url="https://engineering.example.com/mock/lessons",
                snippet=(
                    f"Engineering teams deploying {short_q} at scale have reported "
                    "several recurring failure modes: context window overflow, "
                    "cascading retries, and non-deterministic outputs. Mitigation "
                    "strategies include circuit breakers, output validation, and "
                    "incremental rollouts."
                ),
                metadata={"source": "mock"},
            ),
            SourceDocument(
                title=f"Future directions in {short_q}",
                url="https://research.example.com/mock/future",
                snippet=(
                    f"The field of {short_q} is rapidly evolving. Key open challenges "
                    "include improving factual grounding, reducing hallucinations, "
                    "and enabling more nuanced multi-step reasoning. Researchers "
                    "anticipate breakthrough results within the next 12-18 months."
                ),
                metadata={"source": "mock"},
            ),
        ]
        return docs[:max_results]
