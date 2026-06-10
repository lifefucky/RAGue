"""CLI entrypoint for evaluation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rague.evaluation.agent import run_agent_evaluation_cases
from rague.evaluation.agent_trace import run_traced_agent_evaluation_cases
from rague.evaluation.dataset import load_evaluation_cases
from rague.evaluation.hnsw_benchmark import HnswBenchmarkConfig, run_hnsw_benchmark
from rague.evaluation.reporting import render_evaluation_summary_markdown
from rague.evaluation.retrieval import evaluate_retriever_cases, retriever_to_retrieve_ids
from rague.evaluation.tracing import (
    default_trace_output_path,
    render_trace_summary_markdown,
    write_trace_jsonl,
)


def _default_dataset_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    corpus_dataset = repo_root / "data" / "evaluation" / "basic_cases.json"
    if corpus_dataset.exists():
        return corpus_dataset
    return repo_root / "tests" / "fixtures" / "evaluation" / "basic_cases.json"


def _write_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_retrieval_command(args: argparse.Namespace) -> dict[str, object]:
    from rague.retrieval.hybrid_reranker import create_retriever_from_env

    cases = load_evaluation_cases(args.dataset)
    retriever = create_retriever_from_env()
    results = evaluate_retriever_cases(cases, retriever, k_values=tuple(args.k_values))
    return {"retrieval": results}


def run_agent_trace_command(args: argparse.Namespace) -> dict[str, object]:
    cases = load_evaluation_cases(args.dataset)
    limited_cases = cases[: args.limit]
    results = run_traced_agent_evaluation_cases(limited_cases)

    output_jsonl = args.output_jsonl or default_trace_output_path()
    write_trace_jsonl(output_jsonl, results["traces"])
    results["output_jsonl"] = str(output_jsonl)

    summary_markdown = render_trace_summary_markdown(
        traces=results["traces"],
        run_metadata=results["run_metadata"],
        aggregate={
            "routing": results["routing"],
            "generation": results["generation"],
        },
    )
    results["summary_markdown"] = summary_markdown

    if args.summary is not None:
        _write_output(args.summary, summary_markdown)
    elif args.output is not None:
        _write_output(args.output, summary_markdown)

    return results


def run_agent_command(args: argparse.Namespace) -> dict[str, object]:
    from rague.agents.workflows import run_agentic_rag_from_env

    cases = load_evaluation_cases(args.dataset)
    limited_cases = cases[: args.limit]
    agent_results = run_agent_evaluation_cases(limited_cases, run_agentic_rag_from_env)
    return {
        "routing": agent_results["routing"],
        "generation": agent_results["generation"],
    }


def run_hnsw_benchmark_command(args: argparse.Namespace) -> dict[str, object]:
    from dataclasses import replace

    from rague.retrieval.hybrid_reranker import create_retriever_from_config, _config_from_env

    cases = load_evaluation_cases(args.dataset)

    def retrieve_ids_for_ef(question: str, hnsw_ef: int) -> list[str]:
        config = replace(_config_from_env(), hnsw_ef_search=hnsw_ef)
        retriever = create_retriever_from_config(config)
        retrieve_ids = retriever_to_retrieve_ids(retriever, id_field=args.id_field)
        return retrieve_ids(question)

    benchmark = run_hnsw_benchmark(
        cases,
        retrieve_ids_for_ef,
        config=HnswBenchmarkConfig(
            hnsw_ef=args.hnsw_ef,
            top_k=args.top_k,
            query_limit=args.query_limit,
            id_field=args.id_field,
            baseline_hnsw_ef=args.baseline_hnsw_ef,
        ),
    )
    return {"hnsw_benchmark": benchmark}


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_default_dataset_path(),
        help="Path to labeled evaluation dataset JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for Markdown summary.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON results to stdout.",
    )


def build_parser() -> argparse.ArgumentParser:
    common_parser = argparse.ArgumentParser(add_help=False)
    _add_common_arguments(common_parser)

    parser = argparse.ArgumentParser(description="Run RAGue evaluation commands.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    retrieval_parser = subparsers.add_parser(
        "retrieval",
        help="Evaluate retrieval metrics.",
        parents=[common_parser],
    )
    retrieval_parser.add_argument("--id-field", default="chunk_id")
    retrieval_parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5])
    retrieval_parser.set_defaults(handler=run_retrieval_command)

    agent_parser = subparsers.add_parser(
        "agent",
        help="Evaluate agent routing and generation.",
        parents=[common_parser],
    )
    agent_parser.add_argument("--limit", type=int, default=2)
    agent_parser.set_defaults(handler=run_agent_command)

    agent_trace_parser = subparsers.add_parser(
        "agent-trace",
        help="Run agent evaluation with per-case JSONL tracing.",
        parents=[common_parser],
    )
    agent_trace_parser.add_argument("--limit", type=int, default=15)
    agent_trace_parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Path for per-case JSONL trace output.",
    )
    agent_trace_parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional Markdown summary path for trace diagnostics.",
    )
    agent_trace_parser.set_defaults(handler=run_agent_trace_command)

    hnsw_parser = subparsers.add_parser(
        "hnsw-benchmark",
        help="Benchmark HNSW query settings.",
        parents=[common_parser],
    )
    hnsw_parser.add_argument("--id-field", default="page_id")
    hnsw_parser.add_argument("--top-k", type=int, default=5)
    hnsw_parser.add_argument("--query-limit", type=int, default=5)
    hnsw_parser.add_argument("--hnsw-ef", type=int, default=64)
    hnsw_parser.add_argument("--baseline-hnsw-ef", type=int, default=512)
    hnsw_parser.set_defaults(handler=run_hnsw_benchmark_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    results = args.handler(args)

    if args.json:
        payload = dict(results)
        payload.pop("summary_markdown", None)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "agent-trace":
        print(results.get("summary_markdown", ""))
    else:
        print(render_evaluation_summary_markdown(results))

    if args.output is not None:
        _write_output(args.output, render_evaluation_summary_markdown(results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
