"""
Test script for hybrid search functionality (BM25 + Semantic + RRF).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.pipeline import RAGPipeline


def test_hybrid_search():
    """Test hybrid search with comparison to semantic-only search."""

    print("Testing Hybrid Search (BM25 + Semantic + RRF)")
    print("=" * 70)

    # Test query that should benefit from keyword matching
    query = "What are MPPT algorithms in solar PV systems?"

    # Test 1: Semantic search only (baseline)
    print("\n" + "=" * 70)
    print("TEST 1: Semantic Search Only (Baseline)")
    print("=" * 70)
    print(f"Query: {query}\n")

    pipeline_semantic = RAGPipeline(use_reranking=False, use_hybrid=False)
    pipeline_semantic.load_existing(persist_dir="data/vector_stores/faiss_full")

    nodes_semantic = pipeline_semantic.retrieve_only(query)
    print(f"Retrieved {len(nodes_semantic)} chunks\n")
    print("Top 3 chunks:")
    for i, node in enumerate(nodes_semantic[:3]):
        topic = node.node.metadata.get('topic', 'unknown')
        source = node.node.metadata.get('source', 'unknown')
        score = node.score
        preview = node.node.text[:150].replace('\n', ' ')
        print(f"\n{i+1}. Score: {score:.4f} | Topic: {topic}")
        print(f"   Source: {source[:50]}...")
        print(f"   Preview: {preview}...")

    # Test 2: Hybrid search (BM25 + Semantic + RRF)
    print("\n\n" + "=" * 70)
    print("TEST 2: Hybrid Search (BM25 + Semantic + RRF)")
    print("=" * 70)
    print(f"Query: {query}\n")

    pipeline_hybrid = RAGPipeline(use_reranking=False, use_hybrid=True)
    pipeline_hybrid.load_existing(persist_dir="data/vector_stores/faiss_full")

    nodes_hybrid = pipeline_hybrid.retrieve_only(query)
    print(f"Retrieved {len(nodes_hybrid)} chunks\n")
    print("Top 3 chunks:")
    for i, node in enumerate(nodes_hybrid[:3]):
        topic = node.node.metadata.get('topic', 'unknown')
        source = node.node.metadata.get('source', 'unknown')
        score = node.score
        preview = node.node.text[:150].replace('\n', ' ')
        print(f"\n{i+1}. Score: {score:.4f} | Topic: {topic}")
        print(f"   Source: {source[:50]}...")
        print(f"   Preview: {preview}...")

    # Test 3: Hybrid search with metadata filtering
    print("\n\n" + "=" * 70)
    print("TEST 3: Hybrid Search + Metadata Filter (Solar)")
    print("=" * 70)
    query_filtered = "How does MPPT work?"
    filters = {"topic": "solar"}
    print(f"Query: {query_filtered}")
    print(f"Filters: {filters}\n")

    nodes_filtered = pipeline_hybrid.retrieve_only(query_filtered, filters=filters)
    print(f"Retrieved {len(nodes_filtered)} chunks\n")
    print("Top 3 chunks:")
    for i, node in enumerate(nodes_filtered[:3]):
        topic = node.node.metadata.get('topic', 'unknown')
        source = node.node.metadata.get('source', 'unknown')
        score = node.score
        preview = node.node.text[:150].replace('\n', ' ')
        print(f"\n{i+1}. Score: {score:.4f} | Topic: {topic}")
        print(f"   Source: {source[:50]}...")
        print(f"   Preview: {preview}...")

    # Test 4: Generate answer with hybrid search
    print("\n\n" + "=" * 70)
    print("TEST 4: Generate Answer with Hybrid Search")
    print("=" * 70)
    query_gen = "What are the main types of renewable energy sources?"
    print(f"Query: {query_gen}\n")

    answer = pipeline_hybrid.query(query_gen)
    print(f"Answer:\n{answer}")

    # Test 5: Compare semantic vs hybrid for keyword-heavy query
    print("\n\n" + "=" * 70)
    print("TEST 5: Keyword Query - Semantic vs Hybrid Comparison")
    print("=" * 70)
    keyword_query = "battery energy storage system BESS"
    print(f"Query: {keyword_query}\n")

    print("Semantic Search Results:")
    nodes_sem = pipeline_semantic.retrieve_only(keyword_query)
    for i, node in enumerate(nodes_sem[:3]):
        topic = node.node.metadata.get('topic', 'unknown')
        score = node.score
        preview = node.node.text[:100].replace('\n', ' ')
        print(f"  {i+1}. Score: {score:.4f} | Topic: {topic} | {preview}...")

    print("\nHybrid Search Results:")
    nodes_hyb = pipeline_hybrid.retrieve_only(keyword_query)
    for i, node in enumerate(nodes_hyb[:3]):
        topic = node.node.metadata.get('topic', 'unknown')
        score = node.score
        preview = node.node.text[:100].replace('\n', ' ')
        print(f"  {i+1}. Score: {score:.4f} | Topic: {topic} | {preview}...")

    print("\n\n" + "=" * 70)
    print("HYBRID SEARCH TEST COMPLETE!")
    print("=" * 70)
    print("\nKey observations to make:")
    print("1. Do hybrid results include more keyword-relevant chunks?")
    print("2. Are RRF scores different from pure similarity scores?")
    print("3. Does hybrid search work with metadata filtering?")
    print("4. Is the generated answer accurate with hybrid search?")
    print("5. Does hybrid search better handle keyword-heavy queries (Test 5)?")


if __name__ == "__main__":
    test_hybrid_search()
