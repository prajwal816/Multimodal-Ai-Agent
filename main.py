"""
main.py
────────
CLI entry point for the Multimodal AI Agent.

Usage
─────
  # Standard task (stub mode, no GPU/API key needed)
  python main.py --task "Analyze image and summarize insights" --image data/sample_images/test.jpg

  # RAG-only query
  python main.py --task "What is retrieval augmented generation?" --rag-only

  # Memory benchmark
  python main.py --benchmark-memory --n 50000

  # Custom config
  python main.py --task "Describe the scene" --config configs/config.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

# Ensure the project root is on sys.path so `src` imports work
sys.path.insert(0, str(Path(__file__).parent))


@click.command()
@click.option(
    "--task", "-t",
    default="Analyze the provided content and summarize key insights.",
    show_default=True,
    help="Natural-language task for the agent to complete.",
)
@click.option(
    "--image", "-i",
    default=None,
    help="Path to an image file for multimodal analysis.",
)
@click.option(
    "--config", "-c",
    default="configs/config.yaml",
    show_default=True,
    help="Path to the YAML config file.",
)
@click.option(
    "--rag-only",
    is_flag=True,
    default=False,
    help="Run only the RAG pipeline (skip planning & execution).",
)
@click.option(
    "--benchmark-memory",
    is_flag=True,
    default=False,
    help="Populate FAISS with synthetic entries and report performance.",
)
@click.option(
    "--n",
    default=100_000,
    show_default=True,
    help="Number of synthetic entries for --benchmark-memory.",
)
@click.option(
    "--save-memory",
    is_flag=True,
    default=False,
    help="Persist the FAISS index to disk after the run.",
)
@click.option(
    "--output-json",
    default=None,
    help="Write the full result as JSON to this file path.",
)
def main(
    task: str,
    image: str,
    config: str,
    rag_only: bool,
    benchmark_memory: bool,
    n: int,
    save_memory: bool,
    output_json: str,
) -> None:
    """Multimodal AI Agent — Vision & Language Reasoning."""

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.syntax import Syntax
        console = Console()
        _rich = True
    except ImportError:
        _rich = False

    def print_header() -> None:
        if _rich:
            console.print(Panel.fit(
                "[bold cyan]🤖 Multimodal AI Agent[/bold cyan]\n"
                "[dim]Vision · Language · Memory · RAG[/dim]",
                border_style="cyan",
            ))
        else:
            print("=" * 60)
            print("  Multimodal AI Agent — Vision & Language Reasoning")
            print("=" * 60)

    def print_result(result: dict) -> None:
        if _rich:
            console.print("\n[bold green]✅ Result[/bold green]")
            console.print(Panel(result.get("answer", ""), title="Answer", border_style="green"))

            if result.get("plan_steps"):
                console.print("\n[bold yellow]📋 Plan[/bold yellow]")
                for s in result["plan_steps"]:
                    console.print(f"  {s['index']}. [{s['tool']}] {s['description']}")

            if result.get("sources"):
                console.print(f"\n[bold blue]📚 Sources ({len(result['sources'])} retrieved)[/bold blue]")
                for src in result["sources"][:3]:
                    console.print(f"  • [{src['rank']}] (score={src['score']:.3f}) {src['text'][:100]}…")

            metrics = result.get("metrics", {})
            if metrics:
                console.print(
                    f"\n[dim]⏱ Latency: {metrics.get('total_latency_ms', 0):.0f} ms | "
                    f"Steps: {metrics.get('successful_steps', 0)}/{metrics.get('total_steps', 0)} | "
                    f"Goal: {'✅' if metrics.get('goal_completed') else '❌'}[/dim]"
                )
        else:
            print("\n=== ANSWER ===")
            print(result.get("answer", ""))
            print(f"\nLatency: {result.get('metrics', {}).get('total_latency_ms', 0):.0f} ms")

    print_header()

    # Lazy import to keep startup fast
    from src.agent.agent import MultimodalAgent

    agent = MultimodalAgent(config_path=config)

    # ── Benchmark mode ─────────────────────────────────────────────────────────
    if benchmark_memory:
        if _rich:
            console.print(f"[yellow]📊 Benchmarking memory with {n:,} entries…[/yellow]")
        else:
            print(f"Benchmarking memory with {n:,} entries…")

        result = agent.benchmark_memory(n=n)

        if _rich:
            console.print(Panel(
                f"Index size   : {result['index_size']:,}\n"
                f"Build time   : {result['build_time_s']:.2f}s\n"
                f"Throughput   : {result['throughput_entries_per_s']:,} entries/s",
                title="Memory Benchmark",
                border_style="yellow",
            ))
        else:
            print(json.dumps(result, indent=2))

        if save_memory:
            agent.save_memory()
        return

    # ── RAG-only mode ─────────────────────────────────────────────────────────
    if rag_only:
        if _rich:
            console.print(f"[blue]🔍 RAG query: {task}[/blue]")
        result = agent.query_rag(task)
        print_result({"answer": result.get("answer", ""), "sources": result.get("sources", [])})

    # ── Full agent run ─────────────────────────────────────────────────────────
    else:
        if _rich:
            console.print(f"[cyan]▶ Task: {task}[/cyan]")
            if image:
                console.print(f"[cyan]🖼 Image: {image}[/cyan]")

        result = agent.run(task=task, image_path=image)
        print_result(result)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    if output_json:
        Path(output_json).write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        if _rich:
            console.print(f"[dim]Result saved → {output_json}[/dim]")

    if save_memory:
        agent.save_memory()


if __name__ == "__main__":
    main()
