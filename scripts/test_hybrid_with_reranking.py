"""
Test script to evaluate Hybrid Search + LLM Reranking combination.

This tests the full pipeline:
1. Hybrid Search (BM25 + Semantic + RRF) → Top 10 chunks
2. LLM Reranking → Top 5 chunks
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.pipeline import RAGPipeline


def test_hybrid_with_reranking():
 """Test hybrid search combined with LLM reranking."""

 print("Testing Hybrid Search + LLM Reranking Combination")
 print("=" * 70)

 # Test queries
 test_queries = [
 "What are MPPT algorithms in solar systems?",
 "How does battery energy storage work?",
 "Explain wind turbine control systems",
 ]

 # Initialize 3 different pipeline configurations
 print("\nInitializing pipelines...")
 print("-" * 70)

 # Configuration 1: Semantic only (baseline)
 print("\n1. Semantic-only (baseline)...")
 pipeline_semantic = RAGPipeline(use_reranking=False, use_hybrid=False)
 pipeline_semantic.load_existing(persist_dir="data/vector_stores/faiss_full")

 # Configuration 2: Hybrid search only
 print("\n2. Hybrid search only...")
 pipeline_hybrid = RAGPipeline(use_reranking=False, use_hybrid=True)
 pipeline_hybrid.load_existing(persist_dir="data/vector_stores/faiss_full")

 # Configuration 3: Hybrid + Reranking (best of both)
 print("\n3. Hybrid + Reranking (full pipeline)...")
 pipeline_full = RAGPipeline(use_reranking=True, use_hybrid=True)
 pipeline_full.load_existing(persist_dir="data/vector_stores/faiss_full")

 print("\n" + "=" * 70)
 print("All pipelines initialized!")
 print("=" * 70)

 # Test each query
 for i, query in enumerate(test_queries, 1):
 print(f"\n\n{'=' * 70}")
 print(f"TEST {i}: {query}")
 print("=" * 70)

 # Semantic only
 print("\nConfiguration 1: Semantic Only")
 print("-" * 70)
 nodes_semantic = pipeline_semantic.retrieve_only(query)
 print(f"Retrieved {len(nodes_semantic)} chunks")
 print("\nTop 3 chunks:")
 for j, node in enumerate(nodes_semantic[:3]):
 topic = node.node.metadata.get('topic', 'unknown')
 score = node.score
 preview = node.node.text[:100].replace('\n', ' ')
 print(f" {j+1}. Score: {score:.4f} | Topic: {topic}")
 print(f" Preview: {preview}...")

 # Hybrid only
 print("\n\nConfiguration 2: Hybrid Search Only")
 print("-" * 70)
 nodes_hybrid = pipeline_hybrid.retrieve_only(query)
 print(f"Retrieved {len(nodes_hybrid)} chunks")
 print("\nTop 3 chunks:")
 for j, node in enumerate(nodes_hybrid[:3]):
 topic = node.node.metadata.get('topic', 'unknown')
 score = node.score
 preview = node.node.text[:100].replace('\n', ' ')
 print(f" {j+1}. Score: {score:.4f} | Topic: {topic}")
 print(f" Preview: {preview}...")

 # Hybrid + Reranking (FULL)
 print("\n\nConfiguration 3: Hybrid + Reranking (FULL)")
 print("-" * 70)
 nodes_full = pipeline_full.retrieve_only(query)
 print(f"Retrieved {len(nodes_full)} chunks (after reranking)")
 print("\nTop 3 chunks:")
 for j, node in enumerate(nodes_full[:3]):
 topic = node.node.metadata.get('topic', 'unknown')
 score = node.score
 preview = node.node.text[:100].replace('\n', ' ')
 print(f" {j+1}. Score: {score:.4f} | Topic: {topic}")
 print(f" Preview: {preview}...")

 # Generate answers with each configuration
 print("\n\n" + "=" * 70)
 print("ANSWER COMPARISON")
 print("=" * 70)

 print("\nSemantic Only:")
 print("-" * 70)
 answer_semantic = pipeline_semantic.query(query)
 print(answer_semantic[:300] + "...")

 print("\n\nHybrid Search Only:")
 print("-" * 70)
 answer_hybrid = pipeline_hybrid.query(query)
 print(answer_hybrid[:300] + "...")

 print("\n\nHybrid + Reranking (FULL):")
 print("-" * 70)
 answer_full = pipeline_full.query(query)
 print(answer_full[:300] + "...")

 print("\n\n" + "=" * 70)
 print("TEST COMPLETE!")
 print("=" * 70)
 print("\nKey Observations:")
 print("1. How do retrieved chunks differ between configurations?")
 print("2. Does reranking improve chunk relevance?")
 print("3. Are generated answers more accurate with hybrid + reranking?")
 print("4. Is the quality improvement worth the LLM API cost?")


if __name__ == "__main__":
 test_hybrid_with_reranking()
