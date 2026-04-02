# python evaluate_retrieval.py --repo astapi-realworld-example-app --repo flask

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

CURRENT_DIR = Path(__file__).resolve().parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from code_indexer import CodeIndexer
from graph_builder import CodeGraphBuilder
from parser import RepoParser
from retriever import HybridRetriever


REPO_PATH = CURRENT_DIR.parent / "test_repo"
BENCHMARK_PATH = CURRENT_DIR / "benchmarks" / "benchmark.jsonl"
TOP_K = 5


def load_benchmark(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def symbol_to_qualified_name(unit: dict) -> str:
    parent = unit.get("parent")
    parts = [unit.get("module", "")]
    if parent:
        parts.append(parent)
    parts.append(unit["name"])
    return ".".join(part for part in parts if part)


def evaluate_method(method_name: str, benchmark: Iterable[dict], retriever: HybridRetriever, w_vec: float = 0.8, w_bm25: float = 0.2) -> dict:
    symbol_recall_total = 0.0
    symbol_precision_total = 0.0
    symbol_mrr_total = 0.0
    file_recall_total = 0.0
    file_precision_total = 0.0
    file_mrr_total = 0.0
    secondary_hit_count = 0
    secondary_file_hit_count = 0
    context_gain_total = 0.0
    noise_rate_total = 0.0
    total_queries = 0

    for sample in benchmark:
        query = sample["query"]
        if method_name == "bm25":
            results = retriever.search(query, top_k=TOP_K, w_vec=0.0, w_bm25=1.0)
        elif method_name == "vector":
            results = retriever.search(query, top_k=TOP_K, w_vec=1.0, w_bm25=0.0)
        elif method_name == "vector_bm25_rerank":
            # Now "rerank" means using the BGE cross-encoder
            results = retriever.search(query, top_k=TOP_K, w_vec=w_vec, w_bm25=w_bm25)
        elif method_name == "vector_bm25_rerank_graph":
            base_results = retriever.search(query, top_k=TOP_K, w_vec=w_vec, w_bm25=w_bm25)
            expanded_context = retriever.expand_context(base_results, expand_hops=1)
            
            # Combine base results and expanded context, then sort by retrieval_score descending
            combined = base_results + expanded_context
            
            # Sort all combined results by score properly, avoiding duplicate IDs
            seen_ids = set()
            unique_combined = []
            
            # We sort first so that if there are duplicates, the highest score wins
            for res in sorted(combined, key=lambda x: x.get("retrieval_score", 0.0), reverse=True):
                if res["id"] not in seen_ids:
                    seen_ids.add(res["id"])
                    unique_combined.append(res)
                    
            results = unique_combined[:TOP_K]
        else:
            results = retriever.search(query, top_k=TOP_K, expand_hops=1)[:TOP_K]

        returned_symbols = {symbol_to_qualified_name(result) for result in results}
        returned_files = {result["file"] for result in results}

        primary_symbols = set(sample.get("primary_symbols", []))
        primary_files = set(sample.get("primary_files", []))

        symbol_hit_count = len(returned_symbols & primary_symbols)
        file_hit_count = len(returned_files & primary_files)

        symbol_recall = (symbol_hit_count / len(primary_symbols)) if primary_symbols else 0.0
        symbol_precision = (symbol_hit_count / len(results)) if results else 0.0
        file_recall = (file_hit_count / len(primary_files)) if primary_files else 0.0
        file_precision = (file_hit_count / len(results)) if results else 0.0

        symbol_recall_total += symbol_recall
        symbol_precision_total += symbol_precision
        file_recall_total += file_recall
        file_precision_total += file_precision

        symbol_mrr = 0.0
        file_mrr = 0.0
        for rank, result in enumerate(results, start=1):
            qualified_name = symbol_to_qualified_name(result)
            if symbol_mrr == 0.0 and qualified_name in primary_symbols:
                symbol_mrr = 1.0 / rank
            if file_mrr == 0.0 and result["file"] in primary_files:
                file_mrr = 1.0 / rank
            if symbol_mrr > 0.0 and file_mrr > 0.0:
                break

        symbol_mrr_total += symbol_mrr
        file_mrr_total += file_mrr

        if method_name == "vector_bm25_rerank_graph":
            # Graph evaluations need separate tracking of expanded context
            primary_results = retriever.search(query, top_k=TOP_K, w_vec=w_vec, w_bm25=w_bm25)
            expanded_context = retriever.expand_context(primary_results, expand_hops=1)[:TOP_K]
            expanded_symbols = {symbol_to_qualified_name(result) for result in expanded_context}
            expanded_files = {result["file"] for result in expanded_context}
            secondary_symbols = set(sample.get("secondary_symbols", []))
            secondary_files = set(sample.get("secondary_files", []))

            if expanded_symbols & secondary_symbols:
                secondary_hit_count += 1
            if expanded_files & secondary_files:
                secondary_file_hit_count += 1

            expanded_count = max(len(expanded_context), 1)
            useful_context_count = len(expanded_symbols & secondary_symbols)
            noise_count = len(expanded_symbols - primary_symbols - secondary_symbols)

            context_gain_total += useful_context_count / expanded_count
            noise_rate_total += noise_count / expanded_count
        total_queries += 1

    metrics = {
        "method": method_name,
        "queries": total_queries,
        "primary_symbol_recall_at_5": round(symbol_recall_total / total_queries, 4),
        "primary_symbol_precision_at_5": round(symbol_precision_total / total_queries, 4),
        "primary_symbol_mrr": round(symbol_mrr_total / total_queries, 4),
        "primary_file_recall_at_5": round(file_recall_total / total_queries, 4),
        "primary_file_precision_at_5": round(file_precision_total / total_queries, 4),
        "primary_file_mrr": round(file_mrr_total / total_queries, 4),
    }

    if method_name == "vector_lexical_rerank_graph":
        avg_context_gain = context_gain_total / total_queries
        avg_noise_rate = noise_rate_total / total_queries
        metrics.update(
            {
                "secondary_symbol_hit_at_5": round(secondary_hit_count / total_queries, 4),
                "secondary_file_hit_at_5": round(secondary_file_hit_count / total_queries, 4),
                "context_gain": round(avg_context_gain, 4),
                "noise_rate": round(avg_noise_rate, 4),
                "context_precision": round(max(0.0, 1.0 - avg_noise_rate), 4),
            }
        )

    return metrics


def main() -> None:
    parser_args = argparse.ArgumentParser()
    parser_args.add_argument("--repo", action="append", default=[], help="Repository path to run evaluation on")
    args = parser_args.parse_args()

    repos_to_run = args.repo if args.repo else [str(REPO_PATH / "fastapi-realworld-example-app")]

    for repo_name_or_path in repos_to_run:
        repo_path_str = str(REPO_PATH / repo_name_or_path) if not Path(repo_name_or_path).is_absolute() else repo_name_or_path
        print(f"\\n--- Evaluating repository: {repo_path_str} ---")
        parser = RepoParser(str(repo_path_str))
        try:
            units = parser.parse()
        except Exception as e:
            print(f"Error parsing repository {repo_path_str}: {e}")
            continue
            
        if not units:
            print(f"No code units found for repository {repo_path_str}. Skipping...")
            continue
        
        graph_builder = CodeGraphBuilder(units)
        graph_builder.build()
        code_indexer = CodeIndexer(units)
        code_indexer.build()
        retriever = HybridRetriever(units, graph_builder, code_indexer)

        benchmark = load_benchmark(BENCHMARK_PATH)
        
        repo_name = Path(repo_path_str).name
        repo_benchmark = [sample for sample in benchmark if sample.get("repo", repo_name) == repo_name]
        
        methods = [
            "bm25",
            "vector",
            "vector_bm25_rerank",
            "vector_bm25_rerank_graph",
        ]

        print("Evaluation benchmark:", BENCHMARK_PATH)
        print(f"Queries for {repo_name}:", len(repo_benchmark))
        
        # Collect results for table
        all_metrics = []
        for method in methods:
            metrics = evaluate_method(method, repo_benchmark, retriever)
            all_metrics.append(metrics)
            
        from tabulate import tabulate
        
        # Prepare headers map to shorten column names for display
        header_map = {
            "method": "Method",
            "queries": "Q",
            "primary_symbol_recall_at_5": "Sym Rec",
            "primary_symbol_precision_at_5": "Sym Prec",
            "primary_symbol_mrr": "Sym MRR",
            "primary_file_recall_at_5": "File Rec",
            "primary_file_precision_at_5": "File Prec",
            "primary_file_mrr": "File MRR",
            "secondary_symbol_hit_at_5": "Sec Sym",
            "secondary_file_hit_at_5": "Sec File",
            "context_gain": "Ctx Gain",
            "noise_rate": "Noise",
            "context_precision": "Ctx Prec"
        }
        
        # Format the table using tabulate
        # Rebuild dictionaries so headers look nice
        formatted_metrics = []
        for m in all_metrics:
            formatted_m = {}
            for k, v in m.items():
                if k in header_map:
                    # Keep strings as is, format floats to 4 decimals
                    if isinstance(v, float):
                        formatted_m[header_map[k]] = f"{v:.4f}"
                    else:
                        formatted_m[header_map[k]] = v
            formatted_metrics.append(formatted_m)
            
        print("\n" + tabulate(formatted_metrics, headers="keys", tablefmt="grid") + "\n")

if __name__ == "__main__":
    main()
