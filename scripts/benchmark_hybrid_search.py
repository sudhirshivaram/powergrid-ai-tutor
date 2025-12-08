"""
Benchmark script to measure actual query time for semantic vs hybrid search.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.pipeline import RAGPipeline


def benchmark_search():
    """Benchmark semantic vs hybrid search query times."""

    print("Benchmarking Search Performance")
    print("=" * 70)

    # Test queries
    queries = [
        "What are MPPT algorithms?",
        "battery energy storage system",
        "wind turbine power generation",
        "solar panel efficiency factors",
        "grid stability and frequency regulation"
    ]

    # Initialize pipelines
    print("\nInitializing pipelines...")

    print("\n1. Semantic-only pipeline...")
    start = time.time()
    pipeline_semantic = RAGPipeline(use_reranking=False, use_hybrid=False)
    pipeline_semantic.load_existing(persist_dir="data/vector_stores/faiss_full")
    semantic_init_time = time.time() - start
    print(f"   Initialization time: {semantic_init_time:.2f}s")

    print("\n2. Hybrid search pipeline...")
    start = time.time()
    pipeline_hybrid = RAGPipeline(use_reranking=False, use_hybrid=True)
    pipeline_hybrid.load_existing(persist_dir="data/vector_stores/faiss_full")
    hybrid_init_time = time.time() - start
    print(f"   Initialization time: {hybrid_init_time:.2f}s")
    print(f"   BM25 overhead: {hybrid_init_time - semantic_init_time:.2f}s")

    # Benchmark queries
    print("\n" + "=" * 70)
    print("Query Performance Benchmark")
    print("=" * 70)

    semantic_times = []
    hybrid_times = []

    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: '{query}'")

        # Semantic search
        start = time.time()
        _ = pipeline_semantic.retrieve_only(query)
        semantic_time = (time.time() - start) * 1000  # Convert to ms
        semantic_times.append(semantic_time)
        print(f"  Semantic: {semantic_time:.1f}ms")

        # Hybrid search
        start = time.time()
        _ = pipeline_hybrid.retrieve_only(query)
        hybrid_time = (time.time() - start) * 1000  # Convert to ms
        hybrid_times.append(hybrid_time)
        print(f"  Hybrid:   {hybrid_time:.1f}ms")
        print(f"  Overhead: +{hybrid_time - semantic_time:.1f}ms ({((hybrid_time/semantic_time - 1) * 100):.1f}%)")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    avg_semantic = sum(semantic_times) / len(semantic_times)
    avg_hybrid = sum(hybrid_times) / len(hybrid_times)
    avg_overhead = avg_hybrid - avg_semantic

    print(f"\nSemantic Search:")
    print(f"  Average: {avg_semantic:.1f}ms")
    print(f"  Min: {min(semantic_times):.1f}ms")
    print(f"  Max: {max(semantic_times):.1f}ms")

    print(f"\nHybrid Search:")
    print(f"  Average: {avg_hybrid:.1f}ms")
    print(f"  Min: {min(hybrid_times):.1f}ms")
    print(f"  Max: {max(hybrid_times):.1f}ms")

    print(f"\nOverhead:")
    print(f"  Average: +{avg_overhead:.1f}ms ({((avg_hybrid/avg_semantic - 1) * 100):.1f}%)")
    print(f"  Initialization: +{hybrid_init_time - semantic_init_time:.2f}s (one-time)")

    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_search()
