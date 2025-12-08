# 🚀 HuggingFace Deployment - Step by Step

## Time Required: 30-45 minutes

This is your final step before certification submission!

---

## ✅ Pre-Deployment Verification

Run the verification script:
```bash
python verify_deployment.py
```

**Expected Output**: "✅ ALL CHECKS PASSED!"

If any checks fail, review `CERTIFICATION_CHECKLIST.md` and fix issues.

---

## 📝 Step 1: Create HuggingFace Account (If Needed)

1. Go to: https://huggingface.co/join
2. Sign up with email or GitHub
3. Verify your email address

**Time**: 5 minutes

---

## 🆕 Step 2: Create New Space

1. Go to: https://huggingface.co/spaces
2. Click **"Create new Space"** button
3. Fill in details:

   | Field | Value |
   |-------|-------|
   | **Owner** | Your username |
   | **Space name** | `powergrid-ai-tutor` |
   | **License** | MIT |
   | **Select SDK** | Gradio |
   | **Space hardware** | CPU basic - Free! |
   | **Repo type** | Public |

4. Click **"Create Space"**

**Time**: 2 minutes

---

## 💻 Step 3: Clone Your New Space

Open a terminal and run:

```bash
# Clone your space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/powergrid-ai-tutor
cd powergrid-ai-tutor

# Verify you're in the space directory
pwd
# Should show: .../powergrid-ai-tutor
```

Replace `YOUR_USERNAME` with your HuggingFace username.

**Time**: 1 minute

---

## 📁 Step 4: Copy Project Files

From your PowerGrid AI Tutor project directory:

```bash
# Set variables (adjust paths as needed)
PROJECT_DIR="/home/bhargav/portfolio-project/powergrid-ai-tutor"
SPACE_DIR="/path/to/powergrid-ai-tutor"  # Your cloned HF space

# Copy core application
cp -r $PROJECT_DIR/src/ $SPACE_DIR/
cp -r $PROJECT_DIR/app/ $SPACE_DIR/
cp $PROJECT_DIR/app.py $SPACE_DIR/

# Copy CRITICAL vector store (required for app to work!)
cp -r $PROJECT_DIR/data/vector_stores/ $SPACE_DIR/data/

# Copy configuration
cp $PROJECT_DIR/README_HF.md $SPACE_DIR/README.md
cp $PROJECT_DIR/requirements_hf.txt $SPACE_DIR/requirements.txt
cp $PROJECT_DIR/LICENSE $SPACE_DIR/

# Optional: Copy assets for better UI
cp -r $PROJECT_DIR/assets/ $SPACE_DIR/ 2>/dev/null || true
```

**Alternative - Manual Copy**:
1. Navigate to your HF space directory
2. Copy these folders/files from your project:
   - `src/` → space root
   - `app/` → space root
   - `data/vector_stores/` → `data/` (CREATE data folder if needed)
   - `app.py` → space root
   - `README_HF.md` → rename to `README.md`
   - `requirements_hf.txt` → rename to `requirements.txt`
   - `LICENSE` → space root

**⚠️ CRITICAL**: The `data/vector_stores/faiss_full/` directory MUST be included!

**Time**: 3 minutes

---

## 📦 Step 5: Setup Git LFS (For Large Files)

The FAISS vector store files are large and need Git LFS:

```bash
cd $SPACE_DIR  # Your HF space directory

# Install Git LFS if not already installed
# Ubuntu/Debian:
sudo apt-get install git-lfs

# macOS:
brew install git-lfs

# Windows: Download from https://git-lfs.github.com/

# Initialize Git LFS
git lfs install

# Track large files
git lfs track "data/vector_stores/**/*.faiss"
git lfs track "data/vector_stores/**/*.pkl"
git lfs track "data/vector_stores/**/*.json"

# Add .gitattributes
git add .gitattributes
```

**Verify LFS tracking**:
```bash
cat .gitattributes
# Should show lines like: data/vector_stores/**/*.faiss filter=lfs diff=lfs merge=lfs -text
```

**Time**: 5 minutes

---

## 🚀 Step 6: Commit and Push to HuggingFace

```bash
cd $SPACE_DIR

# Check what's being added
git status

# Add all files
git add .

# Commit with message
git commit -m "Deploy PowerGrid AI Tutor for LLM Certification"

# Push to HuggingFace
git push

# If push fails with auth error, use:
git push https://YOUR_USERNAME:YOUR_HF_TOKEN@huggingface.co/spaces/YOUR_USERNAME/powergrid-ai-tutor
```

**Get HuggingFace Token** (if needed):
1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: "Space Deployment"
4. Role: Write
5. Copy token

**Expected output**:
```
Uploading LFS objects: 100% (X/X), Y MB | Z MB/s
...
To https://huggingface.co/spaces/YOUR_USERNAME/powergrid-ai-tutor
   abc1234..def5678  main -> main
```

**Time**: 5-10 minutes (depending on upload speed)

---

## ⏳ Step 7: Wait for Build

1. Go to your space: `https://huggingface.co/spaces/YOUR_USERNAME/powergrid-ai-tutor`
2. You'll see "Building..." status
3. Click on "Build" tab to see logs
4. Wait for "Running" status (usually 3-5 minutes)

**Watch for errors in build logs**:
- "No module named..." → Check `requirements.txt`
- "File not found..." → Check file paths
- "Out of storage" → Vector store too large (shouldn't happen)

**Time**: 3-5 minutes

---

## 🧪 Step 8: Test Your Deployed App

1. Visit: `https://huggingface.co/spaces/YOUR_USERNAME/powergrid-ai-tutor`
2. Wait for app to load (first load takes ~30 seconds - downloading embedding model)

**Test Checklist**:

### Test 1: API Key Input
- [ ] See "System Configuration" section
- [ ] Can enter Gemini API key
- [ ] Key input is password-masked
- [ ] See LLM provider radio buttons

### Test 2: Initialize System
- [ ] Get your Gemini API key: https://makersuite.google.com/app/apikey
- [ ] Paste key into API key field
- [ ] Select features (query expansion, hybrid search)
- [ ] Click "Initialize System"
- [ ] See success message: "✅ PowerGrid AI Tutor Ready!"

### Test 3: Ask Questions
- [ ] Try example: "What are the main challenges in integrating solar power?"
- [ ] See streaming response (text appears gradually)
- [ ] Response is relevant and detailed
- [ ] No errors in console

### Test 4: Filters
- [ ] Select topic filter: "Solar"
- [ ] Ask: "How do solar panels work?"
- [ ] Should get solar-specific results
- [ ] Try source filter (select a paper)
- [ ] Results should be from that paper only

### Test 5: All Features
- [ ] Enable all features (expansion, hybrid, reranking)
- [ ] Ask: "Explain battery energy storage systems"
- [ ] Response quality should be high
- [ ] Cost should be ~$0.003-0.005 per query

**If any test fails**: Check the logs in HF Space, review code, and redeploy.

**Time**: 10 minutes

---

## ✅ Step 9: Final Checklist

Before submitting for certification:

### Documentation
- [ ] README on HF Space explains the project
- [ ] API keys required are listed
- [ ] Cost estimation is visible (< $0.50)
- [ ] Quick start instructions are clear

### Functionality
- [ ] App loads without errors
- [ ] API key input works
- [ ] Can initialize system with Gemini key
- [ ] Chat responds to questions
- [ ] Streaming works
- [ ] All features toggleable (expansion, hybrid, reranking)
- [ ] Filters work (topic, source)

### Security
- [ ] No API keys in code
- [ ] .env not in repository
- [ ] Only user-provided keys are used

### Cost
- [ ] Test with ~30 queries
- [ ] Total cost should be < $0.10
- [ ] Document actual cost in README

**Time**: 5 minutes

---

## 📋 Step 10: Submit for Certification

Once all tests pass:

1. **Gather submission info**:
   - ✅ HuggingFace Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/powergrid-ai-tutor`
   - ✅ GitHub Repository: `https://github.com/sudhirshivaram/powergrid-ai-tutor`
   - ✅ List of 8 optional features implemented
   - ✅ Cost estimate: < $0.10

2. **Submit through course portal**:
   - Paste Space URL
   - Paste GitHub URL
   - List features
   - Confirm cost < $0.50
   - Confirm API keys only in UI

3. **Wait for review**:
   - Reviewers will test your app
   - They'll verify code quality
   - They'll check all requirements

**Time**: 5 minutes

---

## 🎉 You're Done!

Congratulations! You've completed the LLM Developer Certification project!

**What You Built**:
- ✅ Advanced RAG system with 8 features
- ✅ Production-ready deployment
- ✅ Comprehensive documentation
- ✅ Rigorous evaluation
- ✅ Cost-optimized design

**Skills Demonstrated**:
- RAG implementation
- System architecture
- API integration
- Deployment
- Evaluation methodology
- Technical documentation

---

## 🐛 Troubleshooting

### Issue: "Build failed"
**Solution**: Check build logs on HF Space, verify `requirements.txt` is correct

### Issue: "No module named 'src'"
**Solution**: Ensure `src/` folder is in space root, check import paths

### Issue: "FAISS index not found"
**Solution**: Verify `data/vector_stores/faiss_full/` was copied and pushed with Git LFS

### Issue: "File too large to push"
**Solution**: Set up Git LFS (see Step 5), track large files

### Issue: "App is slow on first load"
**Solution**: Normal - HuggingFace downloads embedding model first time (~150MB)

### Issue: "API key doesn't work"
**Solution**: Verify key is correct, check Gemini API is enabled, try regenerating key

### Issue: "Out of memory"
**Solution**: Reduce chunk size or use smaller model (should not happen with CPU basic)

---

## 📞 Need Help?

1. **Review docs**: `docs/deployment.md`, `CERTIFICATION_CHECKLIST.md`
2. **Test locally first**: `python app/main_with_api_key.py`
3. **Check HF logs**: Build tab on your Space page
4. **Verify files**: Run `python verify_deployment.py`

---

## 🎯 Success Criteria

Your deployment is successful when:
- ✅ Space URL is publicly accessible
- ✅ App loads without errors
- ✅ Can enter API key and initialize
- ✅ Answers questions correctly
- ✅ All features work
- ✅ Cost is under $0.50 for demo
- ✅ Code has no API keys

**Ready to deploy? Follow the steps above and good luck!** 🚀
