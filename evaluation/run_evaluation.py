"""
Run evaluation on the RAG system.
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


def run_retrieval_evaluation(top_k: int = 5):
 """
 Run retrieval evaluation on the RAG system.
 
 Args:
 top_k: Number of chunks to retrieve
 """
 print("PowerGrid AI Tutor - Retrieval Evaluation")
 print("=" * 70)
 
 # Load test queries
 print("\n1. Loading test queries...")
 queries = load_test_queries()
 print(f" Loaded {len(queries)} test queries")
 
 # Initialize RAG pipeline
 print("\n2. Initializing RAG pipeline...")
 pipeline = RAGPipeline()
 pipeline.load_existing(persist_dir="data/vector_stores/faiss_full")
 
 # Initialize evaluators
 print("\n3. Initializing evaluators...")
 hit_rate_eval = HitRateEvaluator(top_k=top_k)
 mrr_eval = MRREvaluator(top_k=top_k)
 
 # Run evaluation on each query
 print(f"\n4. Running evaluation on {len(queries)} queries...")
 print("-" * 70)
 
 hit_rate_results = []
 mrr_results = []
 
 for i, query_data in enumerate(queries, 1):
 query = query_data['query']
 expected_topics = query_data['expected_topics']
 
 print(f"\n[{i}/{len(queries)}] Query: {query[:60]}...")
 
 # Retrieve relevant chunks (no generation, just retrieval)
 retrieved_nodes = pipeline.retrieve_only(query)
 
 # Evaluate Hit Rate
 hit_result = hit_rate_eval.evaluate_query(retrieved_nodes, expected_topics)
 hit_rate_results.append(hit_result)
 
 # Evaluate MRR
 mrr_result = mrr_eval.evaluate_query(retrieved_nodes, expected_topics)
 mrr_results.append(mrr_result)
 
 # Print results
 print(f" Hit: {'✓' if hit_result['hit'] else '✗'}")
 print(f" Reciprocal Rank: {mrr_result['reciprocal_rank']:.3f}")
 if mrr_result['first_relevant_rank']:
 print(f" First relevant at rank: {mrr_result['first_relevant_rank']}")
 
 # Calculate overall metrics
 print("\n\n" + "=" * 70)
 print("EVALUATION RESULTS")
 print("=" * 70)
 
 hit_rate_metrics = hit_rate_eval.evaluate_dataset(hit_rate_results)
 mrr_metrics = mrr_eval.evaluate_dataset(mrr_results)
 
 print(f"\nHit Rate @ {top_k}:")
 print(f" Score: {hit_rate_metrics['hit_rate']:.3f}")
 print(f" Queries with hits: {hit_rate_metrics['total_hits']}/{hit_rate_metrics['total_queries']}")
 print(f" Queries missed: {hit_rate_metrics['queries_missed']}")
 
 print(f"\nMean Reciprocal Rank (MRR):")
 print(f" Score: {mrr_metrics['mrr']:.3f}")
 print(f" Coverage: {mrr_metrics['coverage']:.3f}")
 print(f" Queries with relevant results: {mrr_metrics['queries_with_relevant']}/{mrr_metrics['total_queries']}")
 
 # Save results
 results_dir = Path("evaluation/results")
 results_dir.mkdir(parents=True, exist_ok=True)
 
 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 results_file = results_dir / f"retrieval_eval_{timestamp}.json"
 
 results = {
 "timestamp": timestamp,
 "top_k": top_k,
 "total_queries": len(queries),
 "hit_rate": hit_rate_metrics,
 "mrr": mrr_metrics,
 "individual_results": [
 {
 "query_id": queries[i]['id'],
 "query": queries[i]['query'],
 "hit": hit_rate_results[i]['hit'],
 "reciprocal_rank": mrr_results[i]['reciprocal_rank']
 }
 for i in range(len(queries))
 ]
 }
 
 with open(results_file, 'w') as f:
 json.dump(results, f, indent=2)
 
 print(f"\nResults saved to: {results_file}")
 print("=" * 70)
 
 return results


if __name__ == "__main__":
 run_retrieval_evaluation(top_k=5)