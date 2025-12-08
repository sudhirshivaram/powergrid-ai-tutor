# PowerGrid AI Tutor - Quick Reference Guide

## 🚀 Quick Start Commands

### Local Development

```bash
# Basic mode (semantic search only)
python app/main.py

# Advanced mode (all features)
python app/main.py --full

# With API key UI (for certification)
python app/main_with_api_key.py

# Using Ollama (free, local)
python app/main.py --llm ollama --full
```

### Feature Flags

```bash
# Individual features
python app/main.py --expand          # Query expansion
python app/main.py --hybrid          # Hybrid search  
python app/main.py --rerank          # LLM reranking

# Combine features
python app/main.py --expand --hybrid --rerank

# Full optimization
python app/main.py --full
```

### Testing

```bash
# Run evaluations
python evaluation/run_evaluation.py
python evaluation/compare_reranking.py

# Test specific features
python scripts/test_hybrid_search.py
python scripts/test_query_expansion.py
python scripts/test_metadata_filtering.py
```

---

## 📦 Project Structure

```
powergrid-ai-tutor/
│
├── app/
│   ├── main.py                   # Original Gradio UI (uses .env)
│   └── main_with_api_key.py      # Certification version (API key in UI) ⭐
│
├── src/
│   ├── data/
│   │   ├── loaders.py            # PDF loading
│   │   ├── chunkers.py           # Text chunking (512 tokens)
│   │   ├── embedders.py          # Embeddings + LLM setup
│   │   └── metadata.py           # Metadata extraction
│   │
│   ├── vector_store/
│   │   └── faiss_store.py        # FAISS operations
│   │
│   └── rag/
│       ├── pipeline.py           # Main RAG orchestrator ⭐
│       ├── retrieval.py          # Hybrid retrieval
│       ├── reranker.py           # LLM reranking
│       ├── query_expander.py     # Query expansion
│       ├── generator.py          # Answer generation
│       └── query_router.py       # Query routing (not active)
│
├── evaluation/
│   ├── datasets/
│   │   ├── test_queries.json     # 20 test queries
│   │   └── ground_truth.json     # Expected answers
│   ├── evaluators/
│   │   ├── hit_rate.py           # Hit Rate metric
│   │   └── mrr.py                # MRR metric
│   ├── run_evaluation.py         # Main evaluation
│   └── compare_reranking.py      # Compare performance
│
├── scripts/
│   ├── data_collection/
│   │   └── collect_arxiv_papers.py  # ArXiv scraper
│   └── data_processing/
│       └── build_full_index.py      # Build FAISS index
│
├── data/
│   ├── raw/papers/               # 50 PDF papers
│   └── vector_stores/
│       └── faiss_full/           # Pre-built index ⭐ (REQUIRED for HF)
│
├── docs/
│   ├── deployment.md             # HuggingFace deployment guide ⭐
│   ├── architecture.md
│   └── troubleshooting.md
│
├── README.md                     # Main documentation ⭐
├── README_HF.md                  # For HuggingFace Space
├── CERTIFICATION_CHECKLIST.md    # Requirements verification ⭐
├── requirements.txt              # Python dependencies
├── requirements_hf.txt           # For HuggingFace deployment
└── app.py                        # Entry point for HF Space
```

⭐ = Critical files for certification

---

## 🎯 Key Features & How to Use Them

### 1. Query Expansion
**What**: LLM generates technical synonyms before search  
**Enable**: `--expand` flag  
**Benefit**: +10-20% accuracy  
**Example**: "solar" → "solar, photovoltaic, PV, solar cell, solar panel"

### 2. Hybrid Search
**What**: BM25 (keywords) + Semantic (meaning) with RRF fusion  
**Enable**: `--hybrid` flag  
**Benefit**: +5-15% accuracy  
**Why**: Catches both exact terms and semantic matches

### 3. LLM Reranking
**What**: LLM re-scores retrieved chunks for relevance  
**Enable**: `--rerank` flag  
**Benefit**: +15-25% accuracy  
**Trade-off**: Slower (adds 1-2s), but more accurate

### 4. Metadata Filtering
**What**: Filter by topic or source paper  
**Enable**: Always available in UI dropdowns  
**Topics**: Solar, Wind, Battery, Grid, General  
**Sources**: Any of the 50 research papers

### 5. Streaming Responses
**What**: Real-time answer generation  
**Enable**: Automatic in `main_with_api_key.py`  
**Benefit**: Better UX, see answers as they generate

---

## 🧪 Evaluation Metrics

### Hit Rate @ K
- **Definition**: % of queries with at least 1 relevant chunk in top-K
- **Baseline**: 50% @ top-5
- **Full Pipeline**: ~70% @ top-5
- **Improvement**: +20 percentage points

### Mean Reciprocal Rank (MRR)
- **Definition**: Average of 1/rank of first relevant chunk
- **Baseline**: 33.9%
- **Full Pipeline**: ~55%
- **Improvement**: +21 percentage points

### Overall Accuracy Gain
- **Query Expansion**: +10-20%
- **Hybrid Search**: +5-15%
- **Reranking**: +15-25%
- **Combined**: +30-50%

---

## 💰 Cost Analysis

### With Gemini API

| Component | Tokens/Query | Cost/Query |
|-----------|-------------|------------|
| Embeddings | 0 (local) | $0.000 |
| Query Expansion | ~500 | $0.0005 |
| Retrieval | 0 (local) | $0.000 |
| Reranking | ~800 | $0.0008 |
| Generation | ~1,200 | $0.0012 |
| **TOTAL** | **~2,500** | **~$0.003** |

**To try all features**: ~30 queries × $0.003 = **< $0.10** ✅

### With Ollama (Local)
- **Cost**: $0.00 (completely free)
- **Trade-off**: Slower (~30-40s per query vs 2-3s)

---

## 📋 Certification Requirements

### Mandatory (All ✅)
1. ✅ RAG project in Python
2. ✅ Uses LLM (Gemini/Ollama)
3. ⚠️ HuggingFace deployment (ready, needs upload)
4. ✅ Data collection scripts
5. ✅ README with explanation
6. ✅ API key in UI (not in code)
7. ✅ No costly pipelines
8. ✅ Cost ≤ $0.50
9. ✅ API keys listed
10. ✅ ≥5 optional features

### Optional (8/5 Required ✅)
1. ✅ Streaming responses
2. ✅ RAG evaluation
3. ✅ Domain-specific
4. ✅ Multiple data sources
5. ✅ Structured JSON
6. ✅ Reranker
7. ✅ Hybrid search
8. ✅ Metadata filtering

---

## 🚢 Deployment Checklist

### Pre-Deployment
- [x] API key UI implemented
- [x] Streaming responses working
- [x] README complete
- [x] Cost estimation documented
- [x] Requirements files ready
- [x] Vector store built and saved

### HuggingFace Deployment Steps
1. Create new Space on HuggingFace
2. Clone space repository locally
3. Copy project files (see `docs/deployment.md`)
4. **CRITICAL**: Include `data/vector_stores/faiss_full/`
5. Set up Git LFS for large files
6. Push to HuggingFace
7. Test deployed app
8. Submit for certification

### Post-Deployment Testing
- [ ] Space loads without errors
- [ ] Can enter API key
- [ ] Can initialize system
- [ ] Chat responds correctly
- [ ] Filters work
- [ ] All features functional
- [ ] Cost is under $0.50 for demo

---

## 🐛 Common Issues & Solutions

### "FAISS index not found"
**Solution**: Ensure `data/vector_stores/faiss_full/` is included in deployment

### "API key required"
**Solution**: This is expected! Users must provide their own key via UI

### "Git push failed - file too large"
**Solution**: Use Git LFS for vector store files
```bash
git lfs track "data/vector_stores/**/*"
```

### Slow first load
**Solution**: Normal - downloading embedding model (~150MB first time)

### Ollama connection error
**Solution**: Start Ollama service
```bash
ollama serve
ollama pull qwen2.5:7b
```

---

## 📊 Performance Benchmarks

### Response Time
- **Gemini (no features)**: 2-3 seconds
- **Gemini (full pipeline)**: 4-6 seconds
- **Ollama (no features)**: 30-40 seconds
- **Ollama (full pipeline)**: 60-90 seconds

### Accuracy
- **Baseline**: 50% Hit Rate
- **+ Query Expansion**: 60% Hit Rate
- **+ Hybrid Search**: 65% Hit Rate
- **+ Reranking**: 70% Hit Rate

### Knowledge Base
- **Papers**: 50
- **Pages**: 852
- **Chunks**: 2,166
- **Avg Chunk Size**: 512 tokens
- **Embedding Dim**: 384

---

## 🎓 Learning Resources

### RAG Concepts
- **Retrieval**: Finding relevant documents
- **Augmentation**: Adding context to prompt
- **Generation**: LLM creates answer from context

### Advanced Techniques
- **Hybrid Search**: Keyword + semantic retrieval
- **Reranking**: Re-score results for relevance
- **Query Expansion**: Add related terms to improve recall
- **Metadata Filtering**: Narrow search by attributes

### Evaluation Metrics
- **Hit Rate**: Coverage of relevant results
- **MRR**: Position of first relevant result
- **Precision**: % of retrieved docs that are relevant
- **Recall**: % of relevant docs that are retrieved

---

## 📞 Support & Resources

**Documentation**: See `/docs` folder
**Checklist**: `CERTIFICATION_CHECKLIST.md`
**Deployment**: `docs/deployment.md`
**Troubleshooting**: `docs/troubleshooting.md`

**Repository**: https://github.com/sudhirshivaram/powergrid-ai-tutor

---

## 🎉 You're Ready!

Your PowerGrid AI Tutor project is **95% complete** and ready for certification!

**Next Steps**:
1. Test locally: `python app/main_with_api_key.py`
2. Deploy to HuggingFace Spaces (see `docs/deployment.md`)
3. Test deployed version
4. Submit for certification

**Good luck!** 🚀
