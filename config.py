"""
Central configuration file for PowerGrid AI Tutor.
All configurable parameters in one place for easy adjustment.
"""

# =============================================================================
# DATA COLLECTION SETTINGS
# =============================================================================

DATA_COLLECTION = {
    # ArXiv paper collection
    "max_papers_per_topic": 20,
    "arxiv_categories": ["eess.SY", "cs.AI"],  # Electrical Engineering, AI
    "download_timeout": 30,  # seconds
    
    # Data directories
    "raw_data_dir": "data/raw/papers",
    "processed_data_dir": "data/processed",
    "vector_store_dir": "data/vector_stores",
}

# =============================================================================
# CHUNKING SETTINGS
# =============================================================================

CHUNKING = {
    "chunk_size": 512,  # tokens per chunk
    "chunk_overlap": 50,  # overlap between chunks (maintains context)
    "min_chunk_size": 100,  # minimum viable chunk size
    "respect_sentence_boundary": True,  # split on sentences, not mid-sentence
}

# =============================================================================
# EMBEDDING SETTINGS
# =============================================================================

EMBEDDING = {
    "model_name": "BAAI/bge-small-en-v1.5",
    "embedding_dimension": 384,
    "batch_size": 32,  # for batch embedding generation
    "normalize_embeddings": True,
}

# =============================================================================
# RETRIEVAL SETTINGS
# =============================================================================

RETRIEVAL = {
    # Basic retrieval
    "top_k": 5,  # number of chunks to retrieve initially
    "similarity_threshold": 0.5,  # minimum similarity score (0-1)
    
    # Hybrid search weights
    "semantic_weight": 0.7,  # weight for vector search
    "bm25_weight": 0.3,  # weight for keyword search
    
    # BM25 parameters
    "bm25_k1": 1.5,  # term frequency saturation
    "bm25_b": 0.75,  # length normalization
}

# =============================================================================
# RERANKING SETTINGS
# =============================================================================

RERANKING = {
    "enabled": True,
    "top_n": 3,  # final number of chunks after reranking
    "choice_batch_size": 10,  # chunks to consider per batch
    "model": "llm",  # or "cohere" for Cohere reranker
}

# =============================================================================
# QUERY EXPANSION SETTINGS
# =============================================================================

QUERY_EXPANSION = {
    "enabled": True,
    "max_expansions": 5,  # maximum expansion terms to generate
    "temperature": 0.3,  # LLM temperature for expansion
    "min_query_length": 3,  # minimum words before expansion
}

# =============================================================================
# LLM SETTINGS
# =============================================================================

LLM = {
    # Model selection
    "default_provider": "openai",  # "openai" or "gemini"
    
    # OpenAI
    "openai_model": "gpt-4o-mini",
    "openai_temperature": 0.1,
    "openai_max_tokens": 1000,
    
    # Gemini
    "gemini_model": "gemini-1.5-flash",
    "gemini_temperature": 0.1,
    "gemini_max_tokens": 1000,
    
    # Generation parameters
    "context_window": 4096,  # max context for LLM
    "response_timeout": 30,  # seconds
}

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

PROMPTS = {
    "qa_system_instruction": (
        "You are an AI assistant specialized in electrical engineering, "
        "renewable energy, power systems, and smart grids."
    ),
    
    "qa_template": (
        "Instructions:\n"
        "1. Read the context provided below.\n"
        "2. If the question is related to electrical engineering, renewable energy, "
        "power systems, or smart grids, answer using the context - even if the context "
        "is partial or general.\n"
        "3. For broad questions (e.g., 'batteries', 'transformers'), provide what "
        "information you can find in the context.\n"
        "4. Only if the question is completely unrelated to these domains (e.g., cooking, "
        "sports, entertainment), respond with: 'I don't have information on this topic. "
        "Please ask about electrical engineering, renewable energy, power systems, or smart grids.'\n\n"
        "Context:\n{context_str}\n\n"
        "Question: {query_str}\n\n"
        "Answer:"
    ),
    
    "rejection_message": (
        "I don't have information on this topic. Please ask about electrical "
        "engineering, renewable energy, power systems, or smart grids."
    ),
}

# =============================================================================
# EVALUATION SETTINGS
# =============================================================================

EVALUATION = {
    "test_queries_file": "evaluation/datasets/test_queries.json",
    "hit_rate_k": 3,  # top-k for hit rate calculation
    "mrr_k": 3,  # top-k for MRR calculation
    "min_test_queries": 20,  # minimum number of test queries
}

# =============================================================================
# UI SETTINGS
# =============================================================================

UI = {
    "default_features": {
        "use_query_expansion": True,
        "use_hybrid_search": True,
        "use_reranking": True,
    },
    
    "available_topics": ["All Topics", "Solar", "Wind", "Battery", "Grid", "Smart Grid"],
    "available_sources": ["All Sources"],  # dynamically populated
    
    # Display settings
    "show_sources": True,
    "show_relevance_scores": True,
    "show_processing_time": True,
    "max_sources_display": 3,
}

# =============================================================================
# COST TRACKING
# =============================================================================

COST = {
    # Per 1M tokens
    "openai_input_price": 0.150,  # $/1M tokens
    "openai_output_price": 0.600,  # $/1M tokens
    
    "gemini_input_price": 0.075,  # $/1M tokens (free tier available)
    "gemini_output_price": 0.300,  # $/1M tokens
    
    "cohere_rerank_price": 0.001,  # $/1K searches
    
    # Estimated tokens per query
    "avg_input_tokens": 2000,  # context + query
    "avg_output_tokens": 500,  # answer
}

# =============================================================================
# METADATA SCHEMA
# =============================================================================

METADATA = {
    "required_fields": ["source", "page", "topic"],
    "optional_fields": ["authors", "published_date", "arxiv_id"],
    
    "topic_tags": {
        "solar": ["solar", "photovoltaic", "pv", "solar panel"],
        "wind": ["wind", "turbine", "wind energy", "wind farm"],
        "battery": ["battery", "storage", "bess", "energy storage"],
        "grid": ["grid", "power system", "transmission", "distribution"],
        "smart_grid": ["smart grid", "microgrid", "demand response", "v2g"],
    },
}

# =============================================================================
# LOGGING SETTINGS
# =============================================================================

LOGGING = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": "logs/powergrid_tutor.log",
    "log_to_console": True,
    "log_to_file": False,
}

# =============================================================================
# DEPLOYMENT SETTINGS
# =============================================================================

DEPLOYMENT = {
    "huggingface_space": "sudhirshivaram/powergrid-ai-tutor",
    "github_repo": "sushiva/powergrid-ai-tutor",
    "port": 7860,  # Gradio default
    "share": False,  # for local testing
    "enable_queue": True,  # for HF Spaces
}
