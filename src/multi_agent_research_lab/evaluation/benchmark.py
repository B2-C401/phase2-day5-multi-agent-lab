"""Benchmark helpers for single-agent vs multi-agent comparison."""

import re
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Time *runner*, then compute all available metrics."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    cost = _total_cost(state)
    quality = _heuristic_quality(state)
    notes = _build_notes(state)

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost,
        quality_score=quality,
        notes=notes,
    )
    return state, metrics


# ------------------------------------------------------------------
# Metric helpers
# ------------------------------------------------------------------


def _total_cost(state: ResearchState) -> float | None:
    """Sum cost_usd from all trace events that carry it."""
    total = 0.0
    found = False
    for event in state.trace:
        cost = event.get("payload", {}).get("cost_usd")
        if cost is not None:
            total += cost
            found = True
    return round(total, 6) if found else None


def _count_citations(text: str) -> int:
    """Count distinct [n] citation markers."""
    return len(set(re.findall(r"\[\d+\]", text)))


def _heuristic_quality(state: ResearchState) -> float | None:
    """Score 0–10 based on word count, citation presence, and error count.

    This is a rough heuristic for automated benchmarks.
    Replace with peer-review scores for production evaluation.
    """
    if not state.final_answer:
        return 0.0

    answer = state.final_answer
    word_count = len(answer.split())
    citations = _count_citations(answer)
    errors = len(state.errors)

    # Base score from word count (target ~500 words)
    length_score = min(word_count / 500, 1.0) * 5.0

    # Citation bonus (up to 2 points, saturates at 4 citations)
    citation_score = min(citations / 4, 1.0) * 2.0

    # Structure bonus: has at least one ## heading and a Key Takeaways section
    structure_score = 0.0
    if "##" in answer:
        structure_score += 1.0
    if "takeaway" in answer.lower() or "key takeaway" in answer.lower():
        structure_score += 1.0

    # Error penalty
    error_penalty = min(errors * 0.5, 2.0)

    raw = length_score + citation_score + structure_score - error_penalty
    return round(max(0.0, min(raw, 10.0)), 2)


def _build_notes(state: ResearchState) -> str:
    parts: list[str] = []
    parts.append(f"iterations={state.iteration}")
    if state.final_answer:
        parts.append(f"words={len(state.final_answer.split())}")
        parts.append(f"citations={_count_citations(state.final_answer)}")
    if state.errors:
        parts.append(f"errors={len(state.errors)}")
    return " | ".join(parts)
