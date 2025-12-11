"""
Mean Reciprocal Rank (MRR) evaluator for retrieval quality.
"""

from typing import List, Dict


class MRREvaluator:
 """
 Evaluates retrieval quality using Mean Reciprocal Rank (MRR).
 
 MRR measures: How high is the first relevant result ranked?
 Score: 1/rank of first relevant result
 Example: First relevant at rank 3 → MRR = 1/3 = 0.333
 """
 
 def __init__(self, top_k: int = 5):
 """
 Initialize MRR evaluator.
 
 Args:
 top_k: Number of top results to consider
 """
 self.top_k = top_k
 
 def evaluate_query(self, retrieved_nodes: List, expected_topics: List[str]) -> Dict:
 """
 Evaluate a single query's retrieval using MRR.
 
 Args:
 retrieved_nodes: List of retrieved nodes from the system
 expected_topics: List of expected topics/keywords
 
 Returns:
 Dictionary with reciprocal rank and details
 """
 # Find rank of first relevant result
 first_relevant_rank = None
 
 for i, node in enumerate(retrieved_nodes[:self.top_k]):
 node_text = node.text.lower() if hasattr(node, 'text') else str(node).lower()
 
 # Check if any expected topic appears in this chunk
 for topic in expected_topics:
 if topic.lower() in node_text:
 first_relevant_rank = i + 1 # Rank starts at 1
 break
 
 if first_relevant_rank:
 break
 
 # Calculate reciprocal rank
 reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
 
 return {
 'reciprocal_rank': reciprocal_rank,
 'first_relevant_rank': first_relevant_rank,
 'found_relevant': first_relevant_rank is not None
 }
 
 def evaluate_dataset(self, results: List[Dict]) -> Dict:
 """
 Evaluate all queries in a dataset using MRR.
 
 Args:
 results: List of query results with reciprocal ranks
 
 Returns:
 Mean Reciprocal Rank across all queries
 """
 total_queries = len(results)
 
 if total_queries == 0:
 return {'mrr': 0.0, 'total_queries': 0}
 
 total_rr = sum(r.get('reciprocal_rank', 0) for r in results)
 mrr = total_rr / total_queries
 
 # Additional stats
 queries_with_relevant = sum(1 for r in results if r.get('found_relevant', False))
 
 return {
 'mrr': mrr,
 'total_queries': total_queries,
 'queries_with_relevant': queries_with_relevant,
 'coverage': queries_with_relevant / total_queries if total_queries > 0 else 0
 }