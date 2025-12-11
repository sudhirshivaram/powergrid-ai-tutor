# Frequently Asked Questions (FAQ)

## Deployment & Data

### Q: Do I need to include PDF files when deploying to HuggingFace?

**A: No, PDFs are NOT needed for the app to work.**

**Why PDFs are not required:**
- ✅ The vector store (`data/vector_stores/faiss_full/`) already contains **all the text content** extracted from the PDFs
- ✅ All 2,166 chunks with pre-computed embeddings are stored in the vector store
- ✅ Metadata (paper names, topics, authors, etc.) is stored in JSON files
- ✅ The app **never opens PDFs** during runtime - it only queries the vector store

**How the RAG pipeline works:**
1. **One-time processing** (already done): PDFs → Extract text → Create chunks → Generate embeddings → Save to vector store
2. **Runtime** (when user asks questions): User query → Search vector store → Retrieve relevant chunks → Generate answer with LLM
3. PDFs are **only accessed** during the initial indexing phase, not during query time

**PDF files are only needed if you want to:**
- Rebuild the vector store from scratch
- Add new papers to the knowledge base
- Display original PDF pages in the UI (not implemented in this app)
- Verify source content manually

**For HuggingFace deployment:**
- **Keep**: Vector store files (~13MB) - contains all searchable content
- **Can remove**: PDF files (~186MB) - not used during runtime
- **Benefit**: Much faster deployment (13MB vs 200MB upload)

**Note**: The full project repository on GitHub can still include PDFs for documentation and reproducibility purposes, but the deployed HuggingFace Space doesn't need them.

---

## Vector Store & Indexing

### Q: How do I rebuild the vector store if needed?

**A:** Use the indexing scripts:
```bash
python scripts/data_processing/build_full_index.py
```

This will:
1. Load all PDFs from `data/raw/papers/`
2. Extract text and create chunks
3. Generate embeddings using the local HuggingFace model
4. Save the vector store to `data/vector_stores/faiss_full/`

---

## Cost & Performance

### Q: What are the API costs?

**A:** With Google Gemini 2.5 Flash:
- ~$0.003 per query
- Free tier: 15 requests/minute, 1 million tokens/day
- Total demo cost: < $0.10 (well under the $0.50 certification requirement)

**Alternative (free):** Use Ollama for local LLM (slower but zero cost)

---

## Features

### Q: What advanced RAG techniques are implemented?

**A:** 8 advanced techniques (requirement: minimum 5):
1. ✅ Query Expansion - LLM generates technical synonyms (+10-20% accuracy)
2. ✅ Hybrid Search - BM25 + semantic search with RRF fusion (+5-15% accuracy)
3. ✅ LLM Reranking - Reorders results for better relevance (+15-25% accuracy)
4. ✅ Metadata Filtering - Filter by topic (Solar/Wind/Battery/Grid) and source paper
5. ✅ RAG Evaluation - Complete framework with Hit Rate (70%) and MRR (55%) metrics
6. ✅ Domain-Specific - Specialized for electrical engineering and renewable energy
7. ✅ Multiple Data Sources - 50 ArXiv papers with rich metadata
8. ✅ Streaming Responses - Real-time answer generation with Gradio

---

## Troubleshooting

### Q: Why is the HuggingFace deployment slow to upload?

**A:** Large files (PDFs, vector stores) take time to upload. Solutions:
1. Remove PDF files (not needed for runtime)
2. Use Git LFS for large files (already configured)
3. Be patient - 200MB can take 5-10 minutes to upload

### Q: Can I use this without a Gemini API key?

**A:** Yes! Use Ollama instead:
1. Install Ollama: https://ollama.ai
2. Pull the model: `ollama pull qwen2.5:7b`
3. Select "Ollama" in the UI dropdown
4. No API key needed (completely free, runs locally)
5. Trade-off: Slower responses (~30-40s vs 2-3s with Gemini)

---

*Last updated: December 2025*
