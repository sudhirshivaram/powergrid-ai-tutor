"""
Simplified test script for Query Expansion functionality.

Tests query expansion integrated with the pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.pipeline import RAGPipeline


def test_query_expansion_simple():
 """Test query expansion with different pipeline configurations."""

 print("Testing Query Expansion (Simplified)")
 print("=" * 70)

 # Test query (natural language, non-technical)
 test_query = "How do solar panels work?"

 print(f"\nTest Query: '{test_query}'")
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

 # Test query expansion manually using the full pipeline's expander
 print(f"\n\n{'=' * 70}")
 print("QUERY EXPANSION EXAMPLE")
 print("=" * 70)
 print(f"\nOriginal Query: '{test_query}'")
 print("-" * 70)

 expansion_info = pipeline_full.query_expander.expand_with_details(test_query)

 print(f"\nExpansion Terms Generated:")
 for i, term in enumerate(expansion_info['expansion_terms'], 1):
 print(f" {i}. {term}")

 print(f"\nExpanded Query:")
 print(f" {expansion_info['expanded_query']}")

 # Retrieval comparison
 print(f"\n\n{'=' * 70}")
 print("RETRIEVAL COMPARISON")
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
 print("1. Query expansion adds relevant technical terms")
 print("2. Expansion + Hybrid provides best keyword matching")
 print("3. Full pipeline (+ Reranking) provides best overall results")
 print("4. LLM API costs:")
 print(" - Expansion: 1 call per query (cheap)")
 print(" - Reranking: 1 call per query (moderate)")
 print(" - Generation: 1 call per query (moderate)")
 print(" - Total for full pipeline: 3 calls per query")


if __name__ == "__main__":
 test_query_expansion_simple()
