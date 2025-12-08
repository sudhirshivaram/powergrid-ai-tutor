"""
Quick test to verify Ollama integration with LlamaIndex.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llama_index.llms.ollama import Ollama

print("Testing Ollama Integration")
print("=" * 70)

# Initialize Ollama with Qwen 2.5
llm = Ollama(model="qwen2.5:7b", request_timeout=120.0)

print("\nModel: qwen2.5:7b")
print("Testing simple completion...")
print("-" * 70)

# Test simple completion
prompt = "What is photovoltaic effect in solar panels? Answer in 2 sentences."
response = llm.complete(prompt)

print(f"\nPrompt: {prompt}")
print(f"\nResponse:\n{response.text}")

print("\n" + "=" * 70)
print("Ollama test complete!")
print("=" * 70)
