"""
Comparison test: Gemini (API) vs Ollama (Local) for RAG

This script compares:
1. Answer quality
2. Response time
3. Cost implications

Test query: Simple question about solar energy
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.pipeline import RAGPipeline


def main():
 print("=" * 70)
 print("GEMINI (API) vs OLLAMA (LOCAL) COMPARISON")
 print("=" * 70)

 test_query = "What are the main advantages of solar energy?"

 print(f"\nTest Query: '{test_query}'")
 print("=" * 70)

 # Initialize pipelines
 print("\n\nInitializing Gemini pipeline...")
 print("-" * 70)
 pipeline_gemini = RAGPipeline(llm_provider="gemini")
 pipeline_gemini.load_existing(persist_dir="data/vector_stores/faiss_full")

 print("\n\nInitializing Ollama pipeline...")
 print("-" * 70)
 pipeline_ollama = RAGPipeline(llm_provider="ollama")
 pipeline_ollama.load_existing(persist_dir="data/vector_stores/faiss_full")

 print("\n\n" + "=" * 70)
 print("RUNNING COMPARISON TESTS")
 print("=" * 70)

 # Test 1: Gemini
 print("\n\nTEST 1: Gemini (API-based)")
 print("-" * 70)
 start_time = time.time()
 answer_gemini = pipeline_gemini.query(test_query)
 gemini_time = time.time() - start_time

 print(f"\nAnswer:\n{answer_gemini}")
 print(f"\nResponse Time: {gemini_time:.2f} seconds")

 # Test 2: Ollama
 print("\n\nTEST 2: Ollama (Local)")
 print("-" * 70)
 start_time = time.time()
 answer_ollama = pipeline_ollama.query(test_query)
 ollama_time = time.time() - start_time

 print(f"\nAnswer:\n{answer_ollama}")
 print(f"\nResponse Time: {ollama_time:.2f} seconds")

 # Summary comparison
 print("\n\n" + "=" * 70)
 print("COMPARISON SUMMARY")
 print("=" * 70)

 print("\n1. RESPONSE TIME:")
 print(f" Gemini: {gemini_time:.2f}s")
 print(f" Ollama: {ollama_time:.2f}s")
 if ollama_time < gemini_time:
 speedup = (gemini_time - ollama_time) / gemini_time * 100
 print(f" Winner: Ollama (faster by {speedup:.1f}%)")
 else:
 slowdown = (ollama_time - gemini_time) / gemini_time * 100
 print(f" Winner: Gemini (Ollama slower by {slowdown:.1f}%)")

 print("\n2. COST:")
 print(" Gemini: ~$0.003 per query (API cost)")
 print(" Ollama: $0 (local, just electricity)")
 print(" Winner: Ollama (free after setup)")

 print("\n3. ANSWER QUALITY:")
 print(" Gemini: Generally higher quality (larger model)")
 print(" Ollama: Good quality (7B model, runs locally)")
 print(" Note: Compare answers above to judge yourself")

 print("\n4. RATE LIMITS:")
 print(" Gemini: 15-20 requests/minute (free tier)")
 print(" Ollama: Unlimited (local)")
 print(" Winner: Ollama (no limits)")

 print("\n5. PRIVACY:")
 print(" Gemini: Data sent to Google servers")
 print(" Ollama: 100% private (stays on your machine)")
 print(" Winner: Ollama (complete privacy)")

 print("\n" + "=" * 70)
 print("RECOMMENDATIONS")
 print("=" * 70)
 print("\nUse GEMINI when:")
 print(" - Need best possible answer quality")
 print(" - Queries are infrequent (<15/min)")
 print(" - Willing to pay ~$0.003 per query")
 print(" - OK with sending data to API")

 print("\nUse OLLAMA when:")
 print(" - Want to avoid API costs")
 print(" - Need unlimited queries")
 print(" - Privacy is important")
 print(" - Have decent hardware (8GB+ RAM)")
 print(" - OK with slightly longer response times")

 print("\n" + "=" * 70)
 print("COMPARISON COMPLETE!")
 print("=" * 70)


if __name__ == "__main__":
 main()
