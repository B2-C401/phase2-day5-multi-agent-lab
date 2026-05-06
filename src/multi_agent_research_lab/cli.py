"""Command-line entrypoint for the lab."""

import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import generate_html_report, render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()

logger = logging.getLogger(__name__)

_BASELINE_SYSTEM = """\
You are a research assistant. Answer the user's query thoroughly in ~500 words.
Include key facts, mention relevant tradeoffs, and end with a Key Takeaways section.\
"""


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline (one LLM call, no specialist agents)."""

    _init()
    console.print(f"[bold]Single-Agent Baseline[/bold] — query: {query[:80]}")

    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    llm = LLMClient()
    resp = llm.complete(_BASELINE_SYSTEM, query)
    state.final_answer = resp.content

    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))
    console.print(
        f"[dim]tokens in={resp.input_tokens} out={resp.output_tokens} "
        f"cost=${resp.cost_usd:.5f}[/dim]"
        if resp.cost_usd is not None
        else ""
    )


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    trace: Annotated[bool, typer.Option("--trace", help="Print trace events")] = False,
) -> None:
    """Run the full multi-agent workflow (Supervisor → Researcher → Analyst → Writer)."""

    _init()
    console.print(f"[bold]Multi-Agent Workflow[/bold] — query: {query[:80]}")

    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)

    if result.final_answer:
        console.print(Panel.fit(result.final_answer, title="Multi-Agent Result"))
    else:
        console.print("[red]No final answer produced.[/red]")

    if result.errors:
        console.print(f"[yellow]Errors: {result.errors}[/yellow]")

    if trace and result.trace:
        console.print("\n[bold]Trace events:[/bold]")
        console.print_json(json.dumps(result.trace, indent=2))

    console.print(
        f"[dim]Completed in {result.iteration} iteration(s). "
        f"Route: {' → '.join(result.route_history)}[/dim]"
    )


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")] = (
        "Research GraphRAG state-of-the-art and write a 500-word summary"
    ),
    output: Annotated[Path, typer.Option("--output", "-o", help="Output markdown path")] = Path(
        "reports/benchmark_report.md"
    ),
) -> None:
    """Run both baseline and multi-agent, compare metrics, and save a report."""

    _init()
    console.print(f"[bold]Benchmark[/bold] — query: {query[:80]}\n")

    # Single-agent runner
    def _baseline_runner(q: str) -> ResearchState:
        request = ResearchQuery(query=q)
        st = ResearchState(request=request)
        llm = LLMClient()
        resp = llm.complete(_BASELINE_SYSTEM, q)
        st.final_answer = resp.content
        if resp.input_tokens and resp.output_tokens:
            from multi_agent_research_lab.services.llm_client import (
                _INPUT_COST_PER_TOKEN,
                _OUTPUT_COST_PER_TOKEN,
            )

            cost = (
                resp.input_tokens * _INPUT_COST_PER_TOKEN
                + resp.output_tokens * _OUTPUT_COST_PER_TOKEN
            )
            st.add_trace_event(
                "baseline_done",
                {
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cost_usd": cost,
                },
            )
        return st

    # Multi-agent runner
    def _multi_runner(q: str) -> ResearchState:
        st = ResearchState(request=ResearchQuery(query=q))
        return MultiAgentWorkflow().run(st)

    console.print("Running single-agent baseline…")
    baseline_state, baseline_metrics = run_benchmark("single-agent", query, _baseline_runner)

    console.print("Running multi-agent workflow…")
    multi_state, multi_metrics = run_benchmark("multi-agent", query, _multi_runner)

    # Rich table
    table = Table(title="Benchmark Results", show_header=True)
    table.add_column("Run")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Quality /10", justify="right")
    table.add_column("Notes")

    for m in [baseline_metrics, multi_metrics]:
        cost = "—" if m.estimated_cost_usd is None else f"{m.estimated_cost_usd:.4f}"
        quality = "—" if m.quality_score is None else f"{m.quality_score:.1f}"
        table.add_row(m.run_name, f"{m.latency_seconds:.2f}", cost, quality, m.notes)

    console.print(table)

    # Save markdown report
    output.parent.mkdir(parents=True, exist_ok=True)
    report_md = render_markdown_report([baseline_metrics, multi_metrics])
    output.write_text(report_md, encoding="utf-8")
    console.print(f"[green]Markdown report → {output}[/green]")

    # Save HTML report
    html_path = output.with_suffix(".html")
    html_content = generate_html_report(
        [baseline_metrics, multi_metrics], baseline_state, multi_state, query
    )
    html_path.write_text(html_content, encoding="utf-8")
    console.print(f"[green]HTML report    → {html_path}[/green]")


if __name__ == "__main__":
    app()
