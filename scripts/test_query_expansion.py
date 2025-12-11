"""
Test script for Query Expansion functionality.

Tests query expansion with different configurations:
1. No expansion (baseline)
2. Query expansion only
3. Query expansion + Hybrid search
4. Full pipeline (Query expansion + Hybrid + Reranking)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.pipeline import RAGPipeline
from src.rag.query_expander import QueryExpander


def test_query_expansion():
 """Test query expansion with various configurations."""

 print("Testing Query Expansion")
 print("=" * 70)

 # Test queries (natural language, non-technical)
 test_queries = [
 "How do solar panels work?",
 "What is battery storage?",
 "Explain wind power generation",
 ]

 print("\n" + "=" * 70)
 print("PART 1: Query Expansion Examples")
 print("=" * 70)

 # Initialize query expander
 expander = QueryExpander(max_expansions=5)

 for i, query in enumerate(test_queries, 1):
 print(f"\n{i}. Original Query: '{query}'")
 print("-" * 70)

 # Get expansion details
 expansion_info = expander.expand_with_details(query)

 print(f"\nExpansion Terms:")
 for j, term in enumerate(expansion_info['expansion_terms'], 1):
 print(f" {j}. {term}")

 print(f"\nExpanded Query:")
 print(f" {expansion_info['expanded_query']}")

 print("\n\n" + "=" * 70)
 print("PART 2: Retrieval Comparison (With vs Without Expansion)")
 print("=" * 70)

 # Initialize pipelines
 print("\nInitializing pipelines...")
 print("-" * 70)

 print("\n1. Baseline (No expansion, Semantic only)...")
 pipeline_baseline = RAGPipeline(
 use_reranking=False,
 use_hybrid=False,
 use_query_expansion=False
 )
 pipeline_baseline.load_existing(persist_dir="data/vector_stores/faiss_full")

 print("\n2. Query Expansion + Semantic search...")
 pipeline_expansion = RAGPipeline(
 use_reranking=False,
 use_hybrid=False,
 use_query_expansion=True
 )
 pipeline_expansion.load_existing(persist_dir="data/vector_stores/faiss_full")

 print("\n3. Query Expansion + Hybrid search...")
 pipeline_expansion_hybrid = RAGPipeline(
 use_reranking=False,
 use_hybrid=True,
 use_query_expansion=True
 )
 pipeline_expansion_hybrid.load_existing(persist_dir="data/vector_stores/faiss_full")

 print("\n4. Full pipeline (Expansion + Hybrid + Reranking)...")
 pipeline_full = RAGPipeline(
 use_reranking=True,
 use_hybrid=True,
 use_query_expansion=True
 )
 pipeline_full.load_existing(persist_dir="data/vector_stores/faiss_full")

 print("\n" + "=" * 70)
 print("All pipelines initialized!")
 print("=" * 70)

 # Test retrieval for first query
 test_query = test_queries[0]

 print(f"\n\n{'=' * 70}")
 print(f"RETRIEVAL TEST: '{test_query}'")
 print("=" * 70)

 # Configuration 1: Baseline
 print("\n\nConfiguration 1: Baseline (No expansion)")
 print("-" * 70)
 nodes_baseline = pipeline_baseline.retrieve_only(test_query)
 print(f"Retrieved {len(nodes_baseline)} chunks")
 print("\nTop 3 chunks:")
 for i, node in enumerate(nodes_baseline[:3]):
 topic = node.node.metadata.get('topic', 'unknown')
 score = node.score
 preview = node.node.text[:100].replace('\n', ' ')
 print(f" {i+1}. Score: {score:.4f} | Topic: {topic}")
 print(f" Preview: {preview}...")

 # Configuration 2: Query Expansion + Semantic
 print("\n\nConfiguration 2: Query Expansion + Semantic")
 print("-" * 70)
 nodes_expansion = pipeline_expansion.retrieve_only(test_query)
 print(f"Retrieved {len(nodes_expansion)} chunks")
 print("\nTop 3 chunks:")
 for i, node in enumerate(nodes_expansion[:3]):
 topic = node.node.metadata.get('topic', 'unknown')
 score = node.score
 preview = node.node.text[:100].replace('\n', ' ')
 print(f" {i+1}. Score: {score:.4f} | Topic: {topic}")
 print(f" Preview: {preview}...")

 # Configuration 3: Query Expansion + Hybrid
 print("\n\nConfiguration 3: Query Expansion + Hybrid")
 print("-" * 70)
 nodes_exp_hybrid = pipeline_expansion_hybrid.retrieve_only(test_query)
 print(f"Retrieved {len(nodes_exp_hybrid)} chunks")
 print("\nTop 3 chunks:")
 for i, node in enumerate(nodes_exp_hybrid[:3]):
 topic = node.node.metadata.get('topic', 'unknown')
 score = node.score
 preview = node.node.text[:100].replace('\n', ' ')
 print(f" {i+1}. Score: {score:.4f} | Topic: {topic}")
 print(f" Preview: {preview}...")

 # Configuration 4: Full Pipeline
 print("\n\nConfiguration 4: Full Pipeline (Expansion + Hybrid + Reranking)")
 print("-" * 70)
 nodes_full = pipeline_full.retrieve_only(test_query)
 print(f"Retrieved {len(nodes_full)} chunks (after reranking)")
 print("\nTop 3 chunks:")
 for i, node in enumerate(nodes_full[:3]):
 topic = node.node.metadata.get('topic', 'unknown')
 score = node.score
 preview = node.node.text[:100].replace('\n', ' ')
 print(f" {i+1}. Score: {score:.4f} | Topic: {topic}")
 print(f" Preview: {preview}...")

 # Generate answers
 print("\n\n" + "=" * 70)
 print("ANSWER COMPARISON")
 print("=" * 70)

 print(f"\nQuery: '{test_query}'")

 print("\n\nBaseline (No expansion):")
 print("-" * 70)
 answer_baseline = pipeline_baseline.query(test_query)
 print(answer_baseline[:300] + "...")

 print("\n\nQuery Expansion + Semantic:")
 print("-" * 70)
 answer_expansion = pipeline_expansion.query(test_query)
 print(answer_expansion[:300] + "...")

 print("\n\nQuery Expansion + Hybrid:")
 print("-" * 70)
 answer_exp_hybrid = pipeline_expansion_hybrid.query(test_query)
 print(answer_exp_hybrid[:300] + "...")

 print("\n\nFull Pipeline (Expansion + Hybrid + Reranking):")
 print("-" * 70)
 answer_full = pipeline_full.query(test_query)
 print(answer_full[:300] + "...")

 print("\n\n" + "=" * 70)
 print("TEST COMPLETE!")
 print("=" * 70)
 print("\nKey Observations:")
 print("1. How do expansion terms help improve retrieval?")
 print("2. Does expansion + hybrid work better than expansion alone?")
 print("3. Are answers more accurate with query expansion?")
 print("4. What is the LLM API cost trade-off?")
 print(" - Query expansion: 1 LLM call per query (cheap)")
 print(" - Reranking: 1 LLM call per query (moderate)")
 print(" - Total for full pipeline: 3 LLM calls (expansion + reranking + generation)")


if __name__ == "__main__":
 test_query_expansion()
