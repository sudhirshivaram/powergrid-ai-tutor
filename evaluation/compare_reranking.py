"""
Compare evaluation metrics with and without LLM reranking.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.pipeline import RAGPipeline
from evaluation.evaluators.hit_rate import HitRateEvaluator
from evaluation.evaluators.mrr import MRREvaluator


def load_test_queries(file_path: str = "evaluation/datasets/test_queries.json"):
 """Load test queries from JSON file."""
 with open(file_path, 'r') as f:
 data = json.load(f)
 return data['queries']


def run_evaluation(use_reranking: bool = False, top_k: int = 5):
 """
 Run evaluation on the RAG system.

 Args:
 use_reranking: Whether to use LLM reranking
 top_k: Number of final chunks to evaluate

 Returns:
 Dictionary with evaluation results
 """
 print(f"\nRunning evaluation {'WITH' if use_reranking else 'WITHOUT'} reranking...")
 print("=" * 70)

 # Load test queries
 queries = load_test_queries()

 # Initialize RAG pipeline
 print(f"Initializing RAG pipeline (reranking={'ON' if use_reranking else 'OFF'})...")
 pipeline = RAGPipeline(use_reranking=use_reranking)
 pipeline.load_existing(persist_dir="data/vector_stores/faiss_full")

 # Initialize evaluators
 hit_rate_eval = HitRateEvaluator(top_k=top_k)
 mrr_eval = MRREvaluator(top_k=top_k)

 # Track results
 results = {
 'hits': [],
 'reciprocal_ranks': [],
 'query_details': []
 }

 # Evaluate each query
 print(f"\nEvaluating {len(queries)} queries...")
 for i, query_data in enumerate(queries, 1):
 query = query_data['query']
 expected_topics = query_data['expected_topics']

 # Retrieve chunks
 retrieved_nodes = pipeline.retrieve_only(query)

 # Evaluate
 hit_result = hit_rate_eval.evaluate_query(retrieved_nodes, expected_topics)
 mrr_result = mrr_eval.evaluate_query(retrieved_nodes, expected_topics)

 results['hits'].append(hit_result['hit'])
 results['reciprocal_ranks'].append(mrr_result['reciprocal_rank'])
 results['query_details'].append({
 'id': query_data['id'],
 'query': query,
 'hit': hit_result['hit'],
 'rr': mrr_result['reciprocal_rank']
 })

 print(f" [{i}/{len(queries)}] Hit: {hit_result['hit']}, RR: {mrr_result['reciprocal_rank']:.3f}")

 # Calculate aggregate metrics
 hit_rate = sum(results['hits']) / len(results['hits'])
 mrr = sum(results['reciprocal_ranks']) / len(results['reciprocal_ranks'])

 print("\n" + "=" * 70)
 print(f"RESULTS {'WITH' if use_reranking else 'WITHOUT'} RERANKING:")
 print(f" Hit Rate: {hit_rate:.3f} ({hit_rate*100:.1f}%)")
 print(f" MRR: {mrr:.3f} ({mrr*100:.1f}%)")
 print("=" * 70)

 return {
 'use_reranking': use_reranking,
 'hit_rate': hit_rate,
 'mrr': mrr,
 'details': results
 }


def main():
 """
 Run comparison evaluation: baseline vs reranked.
 """
 print("\n" + "=" * 70)
 print("PowerGrid AI Tutor - Reranking Comparison Evaluation")
 print("=" * 70)

 # Run baseline (without reranking)
 print("\n" + "=" * 70)
 print("BASELINE EVALUATION (No Reranking)")
 print("=" * 70)
 baseline_results = run_evaluation(use_reranking=False, top_k=5)

 # Run with reranking
 print("\n" + "=" * 70)
 print("ENHANCED EVALUATION (With LLM Reranking)")
 print("=" * 70)
 reranked_results = run_evaluation(use_reranking=True, top_k=5)

 # Compare results
 print("\n" + "=" * 70)
 print("COMPARISON SUMMARY")
 print("=" * 70)
 print(f"\nBaseline (No Reranking):")
 print(f" Hit Rate: {baseline_results['hit_rate']:.3f} ({baseline_results['hit_rate']*100:.1f}%)")
 print(f" MRR: {baseline_results['mrr']:.3f} ({baseline_results['mrr']*100:.1f}%)")

 print(f"\nWith LLM Reranking:")
 print(f" Hit Rate: {reranked_results['hit_rate']:.3f} ({reranked_results['hit_rate']*100:.1f}%)")
 print(f" MRR: {reranked_results['mrr']:.3f} ({reranked_results['mrr']*100:.1f}%)")

 # Calculate improvements
 hit_rate_improvement = reranked_results['hit_rate'] - baseline_results['hit_rate']
 mrr_improvement = reranked_results['mrr'] - baseline_results['mrr']

 print(f"\nImprovement:")
 print(f" Hit Rate: {hit_rate_improvement:+.3f} ({hit_rate_improvement*100:+.1f}%)")
 print(f" MRR: {mrr_improvement:+.3f} ({mrr_improvement*100:+.1f}%)")
 print("=" * 70)

 # Save results
 output_file = f"evaluation/results/reranking_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
 Path(output_file).parent.mkdir(parents=True, exist_ok=True)

 with open(output_file, 'w') as f:
 json.dump({
 'timestamp': datetime.now().isoformat(),
 'baseline': baseline_results,
 'reranked': reranked_results,
 'improvement': {
 'hit_rate': hit_rate_improvement,
 'mrr': mrr_improvement
 }
 }, f, indent=2)

 print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
 main()
