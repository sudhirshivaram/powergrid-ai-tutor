# HuggingFace Space Deployment Guide

This guide explains how to deploy PowerGrid AI Tutor to HuggingFace Spaces for the LLM Developer Certification.

## Prerequisites

1. HuggingFace account: https://huggingface.co/join
2. Git installed on your machine
3. Git LFS installed: `git lfs install`

## Deployment Steps

### 1. Create a New Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in details:
   - **Name**: `powergrid-ai-tutor`
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (free tier)
   - **Visibility**: Public

### 2. Clone Your Space Repository

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/powergrid-ai-tutor
cd powergrid-ai-tutor
```

### 3. Copy Project Files

Copy these files from your local project:

```bash
# From your powergrid-ai-tutor directory

# Core application files
cp -r src/ <space-directory>/
cp -r app/ <space-directory>/
cp app.py <space-directory>/

# Data files (IMPORTANT!)
cp -r data/vector_stores/ <space-directory>/data/

# Configuration files
cp README_HF.md <space-directory>/README.md
cp requirements_hf.txt <space-directory>/requirements.txt
cp LICENSE <space-directory>/

# Optional: Example assets
cp -r assets/ <space-directory>/ 
```

### 4. Add Git LFS for Large Files

The FAISS vector store files are large and need Git LFS:

```bash
cd <space-directory>

# Track large files with LFS
git lfs track "data/vector_stores/**/*.faiss"
git lfs track "data/vector_stores/**/*.pkl"
git add .gitattributes
```

### 5. Commit and Push

```bash
git add .
git commit -m "Initial deployment of PowerGrid AI Tutor"
git push
```

### 6. Wait for Build

- HuggingFace will automatically build your space
- Check the build logs on your space page
- Build typically takes 3-5 minutes

### 7. Test Your Deployment

1. Visit: `https://huggingface.co/spaces/YOUR_USERNAME/powergrid-ai-tutor`
2. Enter a Gemini API key in the UI
3. Click "Initialize System"
4. Try example questions

## File Structure for HuggingFace Space

```
powergrid-ai-tutor/  (HF Space root)
├── README.md                 # From README_HF.md
├── requirements.txt          # From requirements_hf.txt
├── app.py                    # Entry point for Gradio
├── app/
│   └── main_with_api_key.py  # Main application
├── src/
│   ├── data/
│   ├── vector_store/
│   └── rag/
├── data/
│   └── vector_stores/
│       └── faiss_full/       # Pre-built FAISS index (REQUIRED!)
│           ├── default__vector_store.json
│           ├── docstore.json
│           ├── image__vector_store.json
│           └── index_store.json
└── LICENSE
```

## Important Notes

### ⚠️ Critical Requirements

1. **Vector Store Must Be Included**
   - The `data/vector_stores/faiss_full/` directory MUST be uploaded
   - This contains the pre-built index (2,166 chunks from 50 papers)
   - Without it, the app won't work

2. **No API Keys in Code**
   - Never commit `.env` files
   - Users provide their own API keys via the UI
   - This meets certification requirements

3. **Git LFS Required**
   - FAISS index files are typically > 100MB
   - Use Git LFS to track them
   - Without LFS, push will fail

### Cost Control

The app is designed to meet the "$0.50 or less" requirement:
- Embeddings are LOCAL (no API costs)
- Only LLM calls use API
- Average query: ~2,500 tokens = $0.003
- Users can test all features for < $0.10

### Troubleshooting

**Build fails with "File too large"**
- Solution: Use Git LFS for vector store files
- Run: `git lfs track "data/vector_stores/**/*"`

**"FAISS index not found" error**
- Solution: Ensure `data/vector_stores/faiss_full/` is included
- Check file paths are correct

**"API key required" error**
- This is expected! Users must provide their own key
- Make sure UI has API key input field

**Slow cold start**
- First load downloads embedding model (~150MB)
- Subsequent loads are faster
- Consider using persistent storage on HF

## Alternative: Using Docker

If you prefer Docker deployment:

```bash
# Create Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements_hf.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
```

Then select "Docker" SDK when creating your HuggingFace Space.

## Post-Deployment Checklist

- [ ] Space is public and accessible
- [ ] README explains the project clearly
- [ ] API key input works
- [ ] Can initialize system with Gemini key
- [ ] Chat responds to questions
- [ ] All RAG features work (expansion, hybrid, reranking)
- [ ] Filters work (topic, source)
- [ ] Streaming responses display
- [ ] Cost is under $0.10 for full testing
- [ ] No API keys in code/config files

## Certification Submission

When submitting for certification:

1. **HuggingFace Space URL**: `https://huggingface.co/spaces/YOUR_USERNAME/powergrid-ai-tutor`
2. **Repository URL**: `https://github.com/sudhirshivaram/powergrid-ai-tutor`
3. **README**: Includes all required sections (features, API keys, cost estimation)
4. **Demo**: Test with your own API key first

## Support

If you encounter issues:
1. Check HuggingFace Space build logs
2. Review `docs/troubleshooting.md`
3. Test locally first: `python app/main_with_api_key.py`

Good luck with your certification! 🚀
