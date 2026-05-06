"""Tracing hooks with LangSmith integration and local JSON fallback.

Priority:
  1. LangSmith  — when LANGSMITH_API_KEY is set in environment
  2. Local span  — wall-clock timing written to ResearchState.trace

To switch to Langfuse instead of LangSmith, replace ``_langsmith_span``
with a Langfuse observation context (see langfuse.com/docs).
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager that records a named span.

    Automatically chooses LangSmith when configured, otherwise local.

    Usage::

        with trace_span("researcher", {"query": q}) as span:
            do_work()
        print(span["duration_seconds"], span["status"])
    """
    from multi_agent_research_lab.core.config import get_settings

    settings = get_settings()
    if settings.langsmith_api_key:
        with _langsmith_span(name, attributes or {}, settings) as span:
            yield span
    else:
        with _local_span(name, attributes or {}) as span:
            yield span


# ------------------------------------------------------------------
# LangSmith provider
# ------------------------------------------------------------------


@contextmanager
def _langsmith_span(
    name: str, attributes: dict[str, Any], settings: Any
) -> Iterator[dict[str, Any]]:
    """Wrap execution in a LangSmith RunTree span."""
    try:
        from langsmith import Client  # type: ignore[import]
        from langsmith.run_trees import RunTree  # type: ignore[import]
    except ImportError:
        logger.warning("langsmith package not installed — falling back to local span")
        with _local_span(name, attributes) as span:
            yield span
        return

    # Pass api_key explicitly — pydantic-settings loads .env into Python objects
    # but does NOT write to os.environ, so LangSmith SDK won't see it otherwise.
    ls_client = Client(
        api_key=settings.langsmith_api_key,
        api_url="https://api.smith.langchain.com",
    )
    run = RunTree(
        name=name,
        inputs=attributes,
        run_type="chain",
        project_name=settings.langsmith_project,
        client=ls_client,
    )
    run.post()
    logger.debug("LangSmith span START name=%s project=%s", name, settings.langsmith_project)

    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes,
        "duration_seconds": None,
        "status": "ok",
        "provider": "langsmith",
    }
    try:
        yield span
        span["duration_seconds"] = perf_counter() - started
        run.end(outputs={"duration_seconds": span["duration_seconds"], "status": "ok"})
        run.patch()
        logger.debug(
            "LangSmith span END name=%s duration=%.3fs url=%s",
            name,
            span["duration_seconds"],
            _run_url(run, settings),
        )
    except Exception as exc:
        span["status"] = "error"
        span["error"] = str(exc)
        span["duration_seconds"] = perf_counter() - started
        run.end(error=str(exc))
        run.patch()
        logger.debug("LangSmith span ERROR name=%s error=%s", name, exc)
        raise


def _run_url(run: Any, settings: Any) -> str:
    """Best-effort URL for the LangSmith run."""
    try:
        return f"https://smith.langchain.com/projects/{settings.langsmith_project}/runs/{run.id}"
    except Exception:
        return ""


# ------------------------------------------------------------------
# Local fallback
# ------------------------------------------------------------------


@contextmanager
def _local_span(name: str, attributes: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Record timing locally; no external calls."""
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes,
        "duration_seconds": None,
        "status": "ok",
        "provider": "local",
    }
    logger.debug("local span START name=%s", name)
    try:
        yield span
    except Exception as exc:
        span["status"] = "error"
        span["error"] = str(exc)
        logger.debug("local span ERROR name=%s error=%s", name, exc)
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        logger.debug(
            "local span END name=%s duration=%.3fs status=%s",
            name,
            span["duration_seconds"],
            span["status"],
        )


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------


def spans_to_markdown(spans: list[dict[str, Any]]) -> str:
    """Render a list of trace spans as a simple markdown table."""
    lines = [
        "| Span | Duration (s) | Status | Provider |",
        "|---|---:|---|---|",
    ]
    for s in spans:
        dur = f"{s.get('duration_seconds', 0):.3f}" if s.get("duration_seconds") else "—"
        lines.append(
            f"| {s['name']} | {dur} | {s.get('status', '—')} | {s.get('provider', '—')} |"
        )
    return "\n".join(lines)
