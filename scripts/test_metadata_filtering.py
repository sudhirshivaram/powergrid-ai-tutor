"""
Test script for metadata filtering functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.pipeline import RAGPipeline


def test_metadata_filtering():
    """Test metadata filtering with different queries."""

    print("Testing Metadata Filtering")
    print("=" * 70)

    # Initialize pipeline
    print("\nInitializing RAG pipeline...")
    pipeline = RAGPipeline(use_reranking=False)
    pipeline.load_existing(persist_dir="data/vector_stores/faiss_full")

    # Test 1: No filter (baseline)
    print("\n" + "=" * 70)
    print("TEST 1: No Filter (Baseline)")
    print("=" * 70)
    query = "What are MPPT algorithms in solar systems?"
    print(f"Query: {query}")
    nodes = pipeline.retrieve_only(query)
    print(f"Retrieved {len(nodes)} chunks")
    for i, node in enumerate(nodes[:3]):
        topic = node.node.metadata.get('topic', 'unknown')
        source = node.node.metadata.get('source', 'unknown')
        print(f"  {i+1}. Topic: {topic}, Source: {source[:40]}...")

    # Test 2: Filter by topic (solar)
    print("\n" + "=" * 70)
    print("TEST 2: Filter by Topic (Solar)")
    print("=" * 70)
    query = "What are MPPT algorithms?"
    filters = {"topic": "solar"}
    print(f"Query: {query}")
    print(f"Filters: {filters}")
    nodes = pipeline.retrieve_only(query, filters=filters)
    print(f"Retrieved {len(nodes)} chunks")
    for i, node in enumerate(nodes[:3]):
        topic = node.node.metadata.get('topic', 'unknown')
        source = node.node.metadata.get('source', 'unknown')
        print(f"  {i+1}. Topic: {topic}, Source: {source[:40]}...")

    # Test 3: Filter by topic (wind)
    print("\n" + "=" * 70)
    print("TEST 3: Filter by Topic (Wind)")
    print("=" * 70)
    query = "How do turbines generate power?"
    filters = {"topic": "wind"}
    print(f"Query: {query}")
    print(f"Filters: {filters}")
    nodes = pipeline.retrieve_only(query, filters=filters)
    print(f"Retrieved {len(nodes)} chunks")
    for i, node in enumerate(nodes[:3]):
        topic = node.node.metadata.get('topic', 'unknown')
        source = node.node.metadata.get('source', 'unknown')
        print(f"  {i+1}. Topic: {topic}, Source: {source[:40]}...")

    # Test 4: Check metadata distribution
    print("\n" + "=" * 70)
    print("METADATA DISTRIBUTION")
    print("=" * 70)
    all_nodes = pipeline.index.docstore.docs.values()
    topic_counts = {}
    for node in all_nodes:
        if hasattr(node, 'metadata') and 'topic' in node.metadata:
            topic = node.metadata['topic']
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

    print("\nTopic Distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(list(all_nodes))) * 100
        print(f"  {topic}: {count} chunks ({percentage:.1f}%)")

    print("\n" + "=" * 70)
    print("METADATA FILTERING TEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    test_metadata_filtering()
