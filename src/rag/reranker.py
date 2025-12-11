"""
Reranker module for improving retrieval relevance using LLM-based scoring.

This module uses an LLM to evaluate and rerank retrieved chunks based on
their relevance to the user's query, improving answer quality.
"""

from typing import List
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.schema import NodeWithScore
import sys
from pathlib import Path

# Add project root to path for config import
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from config import RERANKING
except ImportError:
    RERANKING = {"top_n": 3, "choice_batch_size": 10}


class Reranker:
    """
    LLM-based reranker for retrieved chunks.

    The reranker takes initial retrieval results and uses an LLM to score
    each chunk's relevance to the query, returning only the most relevant ones.
    """

    def __init__(self, top_n: int = None, choice_batch_size: int = None):
        """
        Initialize the reranker.

        Args:
            top_n: Number of top-ranked chunks to return after reranking (uses config.py if not provided)
            choice_batch_size: Number of chunks to consider in each batch (uses config.py if not provided)
        """
        self.top_n = top_n if top_n is not None else RERANKING["top_n"]
        choice_batch_size = choice_batch_size if choice_batch_size is not None else RERANKING["choice_batch_size"]
        self.reranker = LLMRerank(
            top_n=self.top_n,
            choice_batch_size=choice_batch_size
        )
        print(f"Initialized LLM Reranker (top_n={self.top_n})")

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
