"""
Retrieval module for finding relevant document chunks.
"""

from typing import List, Optional, Dict
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter, FilterOperator
from rank_bm25 import BM25Okapi

class Retriever:
 """
 Handles retrieval of relevant document chunks from vector store.
 """
 
 def __init__(self, index: VectorStoreIndex, top_k: int = 5, use_hybrid: bool = False):
 """
 Initialize retriever with a vector index.
 
 Args:
 index: VectorStoreIndex to query
 top_k: Number of most relevant chunks to retrieve
 """
 self.index = index
 self.top_k = top_k
 self.use_hybrid = use_hybrid
 
 # Create retriever from index
 self.retriever = index.as_retriever(similarity_top_k=top_k)
 if self.use_hybrid:
 self._initialize_bm25()
 
 def retrieve(self, query: str, filters: Optional[Dict[str, str]] = None) -> List[NodeWithScore]:
 """
 Retrieve relevant chunks for a query with optional metadata filtering.

 Process:
 1. If hybrid search enabled: Use BM25 + semantic fusion
 2. Otherwise: Use semantic search only
 3. Apply metadata filters (if provided) by filtering results
 4. Return top_k most similar filtered chunks

 Args:
 query: User's question
 filters: Optional dictionary of metadata filters
 e.g., {"topic": "solar", "source": "paper_name.pdf"}

 Returns:
 List of NodeWithScore objects (chunks with similarity scores)
 """
 # Use hybrid search if enabled
 if self.use_hybrid:
 return self.retrieve_hybrid(query, filters=filters)
 
 # Otherwise use semantic search only
 # If filters provided, retrieve more chunks then filter
 if filters:
 # Retrieve more chunks to ensure we have enough after filtering
 # Get 3x the requested amount to account for filtering
 large_retriever = self.index.as_retriever(similarity_top_k=self.top_k * 3)
 all_nodes = large_retriever.retrieve(query)

 # Filter nodes by metadata
 filtered_nodes = self._filter_nodes_by_metadata(all_nodes, filters)

 # Return top_k from filtered results
 nodes = filtered_nodes[:self.top_k]
 else:
 # Retrieve without filters
 nodes = self.retriever.retrieve(query)

 return nodes

 def _filter_nodes_by_metadata(self, nodes: List[NodeWithScore], filters: Dict[str, str]) -> List[NodeWithScore]:
 """
 Filter nodes based on metadata criteria.

 Args:
 nodes: List of nodes to filter
 filters: Dictionary of metadata key-value pairs to match

 Returns:
 Filtered list of nodes matching all filter criteria
 """
 filtered_nodes = []
 for node in nodes:
 # Check if node matches all filter criteria
 matches_all = True
 for key, value in filters.items():
 if key not in node.node.metadata or node.node.metadata[key] != value:
 matches_all = False
 break

 if matches_all:
 filtered_nodes.append(node)

 return filtered_nodes

 def _build_metadata_filters(self, filters: Dict[str, str]) -> MetadataFilters:
 """
 Build MetadataFilters object from filter dictionary.
 NOTE: Not used with FAISS as it doesn't support metadata filtering.
 Kept for compatibility with other vector stores.

 Args:
 filters: Dictionary of field:value pairs to filter on

 Returns:
 MetadataFilters object for LlamaIndex retrieval
 """
 filter_list = []
 for key, value in filters.items():
 if value: # Only add non-empty values
 filter_list.append(
 ExactMatchFilter(key=key, value=value)
 )

 return MetadataFilters(filters=filter_list)
 
 def _initialize_bm25(self):
 """
 Initialize BM25 index from all documents in the vector store.
 BM25 requires tokenized corpus for ranking.
 """
 # Get all nodes from index
 all_nodes = list(self.index.docstore.docs.values())
 
 # Tokenize all documents (simple whitespace tokenization)
 self.bm25_corpus = [node.get_content().lower().split() for node in all_nodes]
 
 # Store node IDs for later retrieval
 self.bm25_node_ids = [node.node_id for node in all_nodes]
 
 # Create BM25 index
 self.bm25 = BM25Okapi(self.bm25_corpus)
 
 print(f"BM25 initialized with {len(self.bm25_corpus)} documents")
 
 def _bm25_search(self, query: str, top_k: int = None) -> List[NodeWithScore]:
 """
 Perform BM25 keyword search.
 
 Args:
 query: Search query
 top_k: Number of results (defaults to self.top_k)
 
 Returns:
 List of NodeWithScore objects ranked by BM25 score
 """
 if top_k is None:
 top_k = self.top_k
 
 # Tokenize query
 query_tokens = query.lower().split()
 
 # Get BM25 scores for all documents
 scores = self.bm25.get_scores(query_tokens)
 
 # Get top-k indices
 top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
 
 # Convert to NodeWithScore objects
 results = []
 for idx in top_indices:
 node_id = self.bm25_node_ids[idx]
 node = self.index.docstore.get_node(node_id)
 score = float(scores[idx])
 results.append(NodeWithScore(node=node, score=score))
 
 return results

 def _reciprocal_rank_fusion(self, semantic_results: List[NodeWithScore],
 bm25_results: List[NodeWithScore],
 k: int = 60) -> List[NodeWithScore]:
 """
 Combine results from semantic and BM25 search using Reciprocal Rank Fusion.

 RRF Formula: score(d) = sum(1 / (k + rank(d))) for each ranking

 Args:
 semantic_results: Results from semantic (vector) search
 bm25_results: Results from BM25 keyword search
 k: Constant for RRF (default 60, standard value from literature)

 Returns:
 Combined and re-ranked results (returns all fused results, not limited to top_k)
 """
 # Create a dictionary to accumulate RRF scores
 rrf_scores = {}

 # Process semantic search results
 for rank, node in enumerate(semantic_results):
 node_id = node.node.node_id
 rrf_scores[node_id] = rrf_scores.get(node_id, 0) + 1 / (k + rank + 1)

 # Process BM25 results
 for rank, node in enumerate(bm25_results):
 node_id = node.node.node_id
 rrf_scores[node_id] = rrf_scores.get(node_id, 0) + 1 / (k + rank + 1)

 # Sort by RRF score (descending)
 sorted_node_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

 # Convert back to NodeWithScore objects (return all, let caller limit)
 fused_results = []
 for node_id, score in sorted_node_ids:
 node = self.index.docstore.get_node(node_id)
 fused_results.append(NodeWithScore(node=node, score=score))

 return fused_results
 
 
 def retrieve_hybrid(self, query: str, filters: Optional[Dict[str, str]] = None) -> List[NodeWithScore]:
 """
 Perform hybrid retrieval combining semantic and BM25 search with RRF fusion.

 Args:
 query: Search query
 filters: Optional metadata filters

 Returns:
 Fused and ranked results from both search methods
 """
 # If filters are provided, retrieve more results for filtering
 retrieval_multiplier = 3 if filters else 2

 # Get semantic search results
 semantic_retriever = self.index.as_retriever(similarity_top_k=self.top_k * retrieval_multiplier)
 semantic_results = semantic_retriever.retrieve(query)

 # Get BM25 keyword search results
 bm25_results = self._bm25_search(query, top_k=self.top_k * retrieval_multiplier)

 # Fuse results using RRF
 fused_results = self._reciprocal_rank_fusion(semantic_results, bm25_results)

 # Apply metadata filtering if needed
 if filters:
 fused_results = self._filter_nodes_by_metadata(fused_results, filters)

 return fused_results[:self.top_k]
 
 
 def get_retrieved_text(self, nodes: List[NodeWithScore]) -> str:
 """
 Extract text content from retrieved nodes.

 Args:
 nodes: List of retrieved nodes with scores

 Returns:
 Combined text from all retrieved chunks
 """
 texts = []
 for i, node in enumerate(nodes):
 texts.append(f"Chunk {i+1} (Score: {node.score:.4f}):\n{node.text}\n")

 return "\n---\n".join(texts)
 
"""
Explanation:
 
 What this does:
 * as_retriever(): Creates a retriever from the vector index
 * retrieve(): Converts query to embedding, finds similar chunks
 * similarity_top_k: Returns the top K most similar chunks
 * get_retrieved_text(): Helper to see what was retrieved
 
 The retrieval process:
 * Query: "What is solar energy?"
 * Embedding: Convert query to 384-dimensional vector
 * Similarity: Calculate distance to all 21 chunk vectors
 * Top K: Return 5 closest chunks
 
 """