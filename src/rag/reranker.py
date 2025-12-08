"""
Reranker module for improving retrieval relevance using LLM-based scoring.

This module uses an LLM to evaluate and rerank retrieved chunks based on
their relevance to the user's query, improving answer quality.
"""

from typing import List
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.schema import NodeWithScore


class Reranker:
    """
    LLM-based reranker for retrieved chunks.

    The reranker takes initial retrieval results and uses an LLM to score
    each chunk's relevance to the query, returning only the most relevant ones.
    """

    def __init__(self, top_n: int = 5, choice_batch_size: int = 10):
        """
        Initialize the reranker.

        Args:
            top_n: Number of top-ranked chunks to return after reranking
            choice_batch_size: Number of chunks to consider in each batch
        """
        self.top_n = top_n
        self.reranker = LLMRerank(
            top_n=top_n,
            choice_batch_size=choice_batch_size
        )
        print(f"Initialized LLM Reranker (top_n={top_n})")

    def rerank(self, nodes: List[NodeWithScore], query_str: str) -> List[NodeWithScore]:
        """
        Rerank nodes based on their relevance to the query.

        Args:
            nodes: List of retrieved nodes with similarity scores
            query_str: The user's query

        Returns:
            List of reranked nodes (top_n most relevant)
        """
        if not nodes:
            return nodes

        # Use LLM to score and rerank
        reranked_nodes = self.reranker.postprocess_nodes(
            nodes,
            query_str=query_str
        )

        return reranked_nodes
