"""
Embedding generation for converting text to vector representations.
"""

import os
from typing import List, Literal
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

# Load environment variables from .env file
load_dotenv()


class EmbeddingManager:
 """
 Manages embedding generation and LLM configuration.
 Sets up global LlamaIndex settings for embeddings and LLM.
 """
 
 def __init__(self,
 embedding_model: str = "BAAI/bge-small-en-v1.5",
 llm_provider: Literal["gemini", "ollama", "openai"] = "gemini",
 llm_model: str = None,
 temperature: float = 0.1,
 api_key: str = None):
 """
 Initialize embedding and LLM models.
 Uses local HuggingFace embeddings (no API calls, no rate limits).

 Args:
 embedding_model: HuggingFace embedding model to use
 llm_provider: Which LLM provider to use ("gemini", "ollama", or "openai")
 llm_model: Specific model to use. Defaults:
 - "models/gemini-2.5-flash" for gemini
 - "qwen2.5:7b" for ollama
 - "gpt-4o-mini" for openai
 
 temperature: LLM temperature (lower = more deterministic)
 api_key: API key for the LLM provider (if using Gemini or OpenAI). If None, will try to load from .env
 """
 # Set default models based on provider
 if llm_model is None:
 if llm_provider == "gemini":
 llm_model = "models/gemini-2.5-flash"
 elif llm_provider == "openai":
 llm_model = "gpt-4o-mini"
 else:
 llm_model = "qwen2.5:7b"

 # Initialize LOCAL embedding model (runs on your machine)
 # BAAI/bge-small-en-v1.5 is small, fast, and effective
 print("Loading local embedding model (this may take a moment on first run)...")
 self.embed_model = HuggingFaceEmbedding(
 model_name=embedding_model
 )

 # Initialize LLM based on provider
 if llm_provider == "gemini":
 # Get API key for Gemini (from parameter or environment)
 if api_key is None:
 api_key = os.getenv("GOOGLE_API_KEY")
 if not api_key:
 raise ValueError("GOOGLE_API_KEY is required. Please provide it via UI or environment variables")

 self.llm = Gemini(
 model=llm_model,
 api_key=api_key,
 temperature=temperature
 )
 print(f"Initialized Gemini LLM: {llm_model}")

 elif llm_provider == "ollama":
 # Initialize Ollama (local, no API key needed)
 self.llm = Ollama(
 model=llm_model,
 request_timeout=300.0, # 5 minutes for local model
 temperature=temperature
 )
 print(f"Initialized Ollama LLM: {llm_model}")

 elif llm_provider == "openai":
 # Get API key for OpenAI (from parameter or environment)
 if api_key is None:
 api_key = os.getenv("OPENAI_API_KEY")
 if not api_key:
 raise ValueError("OPENAI_API_KEY is required. Please provide it via UI or environment variables")

 self.llm = OpenAI(
 model=llm_model,
 api_key=api_key,
 temperature=temperature
 )
 print(f"Initialized OpenAI LLM: {llm_model}")

 else:
 raise ValueError(f"Unknown LLM provider: {llm_provider}. Use 'gemini', 'ollama', or 'openai'")

 # Set global settings for LlamaIndex
 Settings.embed_model = self.embed_model
 Settings.llm = self.llm

 print(f"Initialized local embeddings: {embedding_model}")
 
 def get_embedding_model(self):
 """Return the embedding model instance"""
 return self.embed_model
 
 def get_llm(self):
 """Return the LLM instance"""
 return self.llm