# HuggingFace Spaces Deployment - Challenges & Solutions

## Summary
Successfully deployed PowerGrid AI Tutor to HuggingFace Spaces after resolving multiple dependency and configuration issues.

**Final Working Configuration:**
- Gradio: 6.0.2
- Python: 3.10 (HuggingFace default)
- Total deployment time: ~4 hours
- Total commits: 15+

---

## Challenges Encountered & Solutions

### 1. Authentication Issues
**Problem:** 
- SSH authentication failing
- "Invalid username or password" errors
- Git LFS uploads stuck at 0%

**Solution:**
- Switched from SSH to HTTPS: `https://huggingface.co/spaces/sudhirshivaram/powergrid-ai-tutor`
- Created new HuggingFace Write token
- Used VS Code credential dialog for authentication
- Commits: `268e949`, `afd22b1`

---

### 2. Binary File Rejection
**Problem:**
```
remote: error: push rejected because it contains binary files
```
- HuggingFace Spaces doesn't accept PDF files in git history
- 49 PDFs in `data/raw/papers/` (186MB total)

**Solution:**
```bash
# Remove PDFs from git history
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch data/raw/standards/EPA02461_2010.pdf' \
  HEAD

# Remove all PDFs
rm -rf data/raw/papers/*.pdf
```
- Created `docs/FAQ.md` explaining why PDFs aren't needed (vector store already built)
- Commit: `268e949`

---

### 3. README Metadata Validation
**Problem:**
```
error: short_description length must be ≤60 characters (was 71)
error: app_file path incorrect
```

**Solution:**
- Fixed `short_description`: "AI tutor for electrical engineering & renewables" (54 chars)
- Corrected `app_file`: Changed from `app/main_with_api_key.py` to `app.py`
- Updated YAML frontmatter in README.md
- Commit: `7773bdb`

---

### 4. Gradio Version Compatibility
**Problem 1:** Gradio 5.0.1 incompatible with HuggingFace's huggingface_hub
```python
ImportError: cannot import name 'HfFolder' from 'huggingface_hub'
```

**Solution 1:**
- Downgraded to Gradio 4.44.0
- Pinned `huggingface-hub<0.23.0`
- Commit: `04a80f9`, `fdc5a14`

**Problem 2:** SDK version mismatch
- README metadata had `sdk_version: 5.0.1`
- requirements.txt had `gradio==4.44.0`
- HuggingFace uses README metadata, causing conflict

**Solution 2:**
- Synchronized both files to use same version
- Commit: `7773bdb`

**Problem 3:** Gradio 4.x schema TypeError
```python
TypeError: argument of type 'bool' is not iterable
```

**Solution 3:**
- Upgraded to Gradio 6.0.2 (latest stable)
- Changed chat history format from tuples to dictionaries:
```python
# Old (Gradio 4.x):
history.append((message, response))

# New (Gradio 6.x):
history.append({"role": "user", "content": message})
history.append({"role": "assistant", "content": response})
```
- Commit: `2dddf42`, `189e2ac`

---

### 5. Missing Dependencies
**Problem:**
Multiple `ModuleNotFoundError` during deployment:
- `llama_index.embeddings.huggingface`
- `llama_index.llms.ollama`
- `rank_bm25`

**Root Cause:**
- Local development used `pip install` without updating requirements.txt
- HuggingFace only installs packages listed in requirements.txt

**Solution:**
Added all missing packages:
```python
llama-index-embeddings-huggingface
llama-index-llms-ollama
rank-bm25
```
- Commits: `fdc5a14`, `05ad922`, `ead7596`

**Lesson Learned:**
Always update requirements.txt when installing packages:
```bash
echo "package-name==version" >> requirements.txt
# OR
pip freeze > requirements.txt
```

---

### 6. API Key Initialization Error
**Problem:**
```python
ValueError: GOOGLE_API_KEY is required. Please provide it via UI or environment variables
```
- Pipeline initialized in `__init__` before user provides API key
- App crashed on startup

**Solution:**
Implemented lazy initialization:
```python
def __init__(self):
    self.pipeline = None  # Don't initialize yet
    
def initialize_pipeline(self, api_key):
    """Initialize only after user provides API key"""
    os.environ['GOOGLE_API_KEY'] = api_key
    self.pipeline = RAGPipeline(...)
```
- Added API key input UI before chat interface
- Commit: `0a7768e`

---

### 7. Asyncio File Descriptor Error
**Problem:**
```python
ValueError: Invalid file descriptor: -1
Exception in: <function BaseEventLoop.__del__>
```

**Cause:**
- Event loop cleanup issue on HuggingFace Spaces
- Uvicorn + FastAPI environment compatibility

**Solution 1:** Enable Gradio queue
```python
interface.queue()
interface.launch()
```
- Commit: `a391d30`

**Solution 2:** Suppress warnings (cosmetic)
```python
import warnings
warnings.filterwarnings("ignore", message="Invalid file descriptor")
```
- Note: Error is harmless, only appears during shutdown

---

### 8. Missing Source Attribution
**Problem:**
- Responses didn't include source citations
- `return_sources=False` in query call

**Solution:**
```python
# Change from:
response = self.pipeline.query(message, return_sources=False)

# To:
response = self.pipeline.query(message, return_sources=True)
```
- Commit: `6c386a0`

---

## Final Working Files

### requirements.txt
```python
# Core dependencies
llama-index-core==0.11.0
llama-index-embeddings-gemini
llama-index-embeddings-huggingface
llama-index-llms-gemini
llama-index-llms-ollama
llama-index-readers-file

# Vector stores
faiss-cpu
llama-index-vector-stores-faiss  

# Retrieval
rank-bm25

# Data processing
arxiv
requests
beautifulsoup4
pypdf
python-dotenv

# UI - Gradio 6.x
gradio==6.0.2
gradio_client==2.0.1
```

### README.md (metadata)
```yaml
---
title: PowerGrid AI Tutor
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.0.2
app_file: app.py
pinned: false
short_description: AI tutor for electrical engineering & renewables
---
```

### app.py
```python
import warnings
warnings.filterwarnings("ignore", message="Invalid file descriptor")

from app.main import PowerGridTutorUI

if __name__ == "__main__":
    tutor = PowerGridTutorUI()
    interface = tutor.create_interface()
    interface.queue()
    interface.launch()
```

---

## Key Lessons Learned

### 1. Version Management
- **Always sync versions** between README metadata and requirements.txt
- HuggingFace reads `sdk_version` from README, not requirements.txt
- Use exact versions (`==`) not ranges (`>=`) for critical dependencies

### 2. Dependency Tracking
- Keep requirements.txt updated during development
- Test deployment early to catch missing dependencies
- Consider using `pip freeze` or `poetry` for better dependency management

### 3. Platform Differences
- Local environment ≠ deployment environment
- HuggingFace uses Python 3.10 (not 3.13)
- Different event loop handling (asyncio quirks)
- Git LFS required for large files (but PDFs rejected)

### 4. API Key Management
- Lazy initialization pattern for user-provided API keys
- Don't initialize services in `__init__` that require secrets
- Provide clear UI feedback during initialization

### 5. Gradio Evolution
- Gradio 4.x vs 6.x have different Chatbot schemas
- Tuples `(user, bot)` vs Dictionaries `{"role": ..., "content": ...}`
- Check Gradio version when copying code examples

---

## Future Improvements

### Docker Deployment
Using Docker would have avoided most of these issues:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]
```

**Benefits:**
- ✅ Exact environment replication
- ✅ No version drift
- ✅ Platform consistency
- ✅ Easier debugging

**Trade-offs:**
- ❌ Larger deployment size
- ❌ Slightly more complex setup
- ❌ Learning curve

### Better Dependency Management
Consider switching to Poetry:
```toml
[tool.poetry.dependencies]
python = "^3.10"
gradio = "6.0.2"
llama-index-core = "0.11.0"
```

### Environment Variables
Externalize configuration:
```python
SERVER_NAME = os.getenv("SERVER_NAME", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))
```

---

## Deployment Checklist

Before deploying to HuggingFace Spaces:

- [ ] All dependencies in requirements.txt with exact versions
- [ ] README metadata matches requirements.txt (sdk_version)
- [ ] No binary files (PDFs, large models) in git history
- [ ] API keys handled via UI input or HF Secrets
- [ ] Lazy initialization for services requiring credentials
- [ ] Test locally with same Python version as deployment (3.10)
- [ ] Gradio chat history format matches Gradio version
- [ ] Git LFS configured for vector stores/large JSON files
- [ ] app_file path correct in README
- [ ] short_description ≤60 characters

---

## Timeline

| Time | Action | Result |
|------|--------|--------|
| T+0h | Initial push | Authentication failed |
| T+0.5h | Fixed auth, pushed | Binary files rejected |
| T+1h | Removed PDFs | Metadata validation failed |
| T+1.5h | Fixed metadata | Gradio import error |
| T+2h | Fixed Gradio version | Missing llama-index packages |
| T+2.5h | Added dependencies | API key initialization error |
| T+3h | Lazy initialization | Gradio schema error |
| T+3.5h | Updated to Gradio 6.x | Asyncio file descriptor warning |
| T+4h | Enabled queue, suppressed warnings | ✅ **SUCCESS** |

---

## Resources

- [HuggingFace Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Gradio 6.x Migration Guide](https://www.gradio.app/guides/gradio-6-migration-guide)
- [Git LFS Documentation](https://git-lfs.github.com/)
- [HuggingFace Git Password Deprecation](https://huggingface.co/blog/password-git-deprecation)

---

**Deployment URL:** https://huggingface.co/spaces/sudhirshivaram/powergrid-ai-tutor  
**Final Status:** ✅ Running successfully  
**Last Updated:** December 8, 2025
