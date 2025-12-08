"""
Hit Rate evaluator for retrieval quality.
"""

from typing import List, Dict


class HitRateEvaluator:
    """
    Evaluates retrieval quality using Hit Rate metric.
    
    Hit Rate measures: Did the retrieved results contain a relevant chunk?
    Score: 1 if at least one relevant chunk is in top-K, 0 otherwise
    """
    
    def __init__(self, top_k: int = 5):
        """
        Initialize Hit Rate evaluator.
        
        Args:
            top_k: Number of top results to consider
        """
        self.top_k = top_k
    
    def evaluate_query(self, retrieved_nodes: List, expected_topics: List[str]) -> Dict:
        """
        Evaluate a single query's retrieval.
        
        Args:
            retrieved_nodes: List of retrieved nodes from the system
            expected_topics: List of expected topics/keywords
            
        Returns:
            Dictionary with hit score and details
        """
        # Check if any retrieved node contains expected topics
        hit = False
        matching_chunks = []
        
        for i, node in enumerate(retrieved_nodes[:self.top_k]):
            node_text = node.text.lower() if hasattr(node, 'text') else str(node).lower()
            
            # Check if any expected topic appears in this chunk
            for topic in expected_topics:
                if topic.lower() in node_text:
                    hit = True
                    matching_chunks.append({
                        'rank': i + 1,
                        'topic': topic,
                        'score': node.score if hasattr(node, 'score') else None
                    })
                    break
        
        return {
            'hit': 1 if hit else 0,
            'matching_chunks': matching_chunks,
            'total_retrieved': len(retrieved_nodes[:self.top_k])
        }
    
    def evaluate_dataset(self, results: List[Dict]) -> Dict:
        """
        Evaluate all queries in a dataset.
        
        Args:
            results: List of query results with retrieved nodes and expected topics
            
        Returns:
            Aggregated hit rate metrics
        """
        total_queries = len(results)
        total_hits = sum(1 for r in results if r.get('hit', 0) == 1)
        
        hit_rate = total_hits / total_queries if total_queries > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'total_queries': total_queries,
            'total_hits': total_hits,
            'queries_missed': total_queries - total_hits
        }