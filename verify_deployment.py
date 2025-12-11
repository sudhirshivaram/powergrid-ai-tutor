#!/usr/bin/env python3
"""
Pre-deployment verification script for PowerGrid AI Tutor.
Checks all critical components before HuggingFace deployment.
"""

import sys
import os
from pathlib import Path
import json

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check(condition, message, details=""):
 """Print check result with color."""
 if condition:
 print(f"{GREEN} {message}{RESET}")
 if details:
 print(f" {BLUE}{details}{RESET}")
 return True
 else:
 print(f"{RED} {message}{RESET}")
 if details:
 print(f" {YELLOW}{details}{RESET}")
 return False

def main():
 print(f"\n{BLUE}{'='*70}{RESET}")
 print(f"{BLUE}PowerGrid AI Tutor - Pre-Deployment Verification{RESET}")
 print(f"{BLUE}{'='*70}{RESET}\n")
 
 project_root = Path(__file__).parent
 os.chdir(project_root)
 
 all_passed = True
 
 # 1. Check critical files exist
 print(f"\n{BLUE}[1] Checking Critical Files{RESET}")
 print("-" * 70)
 
 critical_files = {
 "README.md": "Main documentation",
 "CERTIFICATION_CHECKLIST.md": "Requirements verification",
 "requirements.txt": "Python dependencies",
 "app/main_with_api_key.py": "Main UI with API key support",
 "app.py": "HuggingFace entry point",
 "requirements_hf.txt": "HF deployment dependencies",
 "README_HF.md": "HF Space README",
 "src/rag/pipeline.py": "Core RAG pipeline",
 "src/rag/retrieval.py": "Hybrid search",
 "src/rag/reranker.py": "LLM reranking",
 "src/rag/query_expander.py": "Query expansion",
 "evaluation/run_evaluation.py": "Evaluation script",
 "evaluation/datasets/test_queries.json": "Test dataset",
 "scripts/data_collection/collect_arxiv_papers.py": "Data collection",
 }
 
 for file_path, description in critical_files.items():
 exists = Path(file_path).exists()
 all_passed &= check(exists, f"{file_path}", description)
 
 # 2. Check vector store exists (CRITICAL for deployment)
 print(f"\n{BLUE}[2] Checking Vector Store (CRITICAL){RESET}")
 print("-" * 70)
 
 vector_store_path = Path("data/vector_stores/faiss_full")
 vs_exists = vector_store_path.exists() and vector_store_path.is_dir()
 all_passed &= check(vs_exists, "Vector store directory exists", str(vector_store_path))
 
 if vs_exists:
 required_vs_files = [
 "default__vector_store.json",
 "docstore.json",
 "image__vector_store.json",
 "index_store.json"
 ]
 for vs_file in required_vs_files:
 file_path = vector_store_path / vs_file
 exists = file_path.exists()
 all_passed &= check(exists, f" {vs_file}", f"Size: {file_path.stat().st_size if exists else 0} bytes")
 
 # 3. Check no API keys in code
 print(f"\n{BLUE}[3] Checking API Key Security{RESET}")
 print("-" * 70)
 
 env_file = Path(".env")
 env_in_gitignore = False
 if Path(".gitignore").exists():
 with open(".gitignore", 'r') as f:
 env_in_gitignore = ".env" in f.read()
 
 all_passed &= check(env_in_gitignore, ".env is in .gitignore", "API keys won't be committed")
 
 env_exists = env_file.exists()
 check(not env_exists or env_in_gitignore, 
 ".env file handling", 
 "Present locally but gitignored" if env_exists else "Not present (good for deployment)")
 
 # 4. Check README completeness
 print(f"\n{BLUE}[4] Checking README Completeness{RESET}")
 print("-" * 70)
 
 readme_path = Path("README.md")
 if readme_path.exists():
 readme_content = readme_path.read_text()
 
 required_sections = {
 "API Keys Required": "Lists required API keys",
 "Cost Estimation": "Shows cost < $0.50",
 "Features": "Lists implemented features",
 "Quick Start": "Usage instructions",
 "Evaluation Results": "Performance metrics",
 }
 
 for section, description in required_sections.items():
 has_section = section.lower() in readme_content.lower()
 all_passed &= check(has_section, f"Section: {section}", description)
 else:
 all_passed = False
 print(f"{RED} README.md not found{RESET}")
 
 # 5. Check evaluation dataset
 print(f"\n{BLUE}[5] Checking Evaluation Dataset{RESET}")
 print("-" * 70)
 
 test_queries_path = Path("evaluation/datasets/test_queries.json")
 if test_queries_path.exists():
 with open(test_queries_path, 'r') as f:
 data = json.load(f)
 num_queries = len(data.get('queries', []))
 has_enough = num_queries >= 10
 all_passed &= check(has_enough, 
 f"Test queries ({num_queries})", 
 "At least 10 queries for evaluation")
 else:
 all_passed &= check(False, "Test queries file", "evaluation/datasets/test_queries.json")
 
 # 6. Check Python dependencies
 print(f"\n{BLUE}[6] Checking Dependencies{RESET}")
 print("-" * 70)
 
 requirements_path = Path("requirements_hf.txt")
 if requirements_path.exists():
 with open(requirements_path, 'r') as f:
 requirements = f.read()
 
 critical_packages = {
 "llama-index-core": "Core RAG framework",
 "faiss-cpu": "Vector store",
 "gradio": "UI framework",
 "llama-index-llms-gemini": "Gemini LLM support",
 }
 
 for package, description in critical_packages.items():
 has_package = package in requirements
 all_passed &= check(has_package, f"Package: {package}", description)
 else:
 all_passed &= check(False, "requirements_hf.txt", "Deployment dependencies")
 
 # 7. Check deployment files
 print(f"\n{BLUE}[7] Checking HuggingFace Deployment Files{RESET}")
 print("-" * 70)
 
 app_py_path = Path("app.py")
 if app_py_path.exists():
 with open(app_py_path, 'r') as f:
 app_content = f.read()
 has_main = "main_with_api_key" in app_content
 all_passed &= check(has_main, "app.py imports correct entry point", 
 "Uses main_with_api_key.py")
 else:
 all_passed &= check(False, "app.py exists", "HuggingFace entry point")
 
 # 8. Check optional features
 print(f"\n{BLUE}[8] Checking Optional Features (Need 5, Have 8+){RESET}")
 print("-" * 70)
 
 features = {
 "Streaming": Path("app/main_with_api_key.py").exists() and "yield" in Path("app/main_with_api_key.py").read_text(),
 "Reranking": Path("src/rag/reranker.py").exists(),
 "Hybrid Search": Path("src/rag/retrieval.py").exists() and "bm25" in Path("src/rag/retrieval.py").read_text().lower(),
 "Query Expansion": Path("src/rag/query_expander.py").exists(),
 "Metadata Filtering": "filters" in Path("src/rag/retrieval.py").read_text(),
 "Evaluation": Path("evaluation/run_evaluation.py").exists(),
 "Domain-Specific": "electrical engineering" in readme_content.lower() or "renewable energy" in readme_content.lower(),
 "Data Collection": Path("scripts/data_collection/collect_arxiv_papers.py").exists(),
 }
 
 implemented_count = sum(features.values())
 for feature, implemented in features.items():
 check(implemented, f"Feature: {feature}", "✓ Implemented" if implemented else "✗ Not found")
 
 has_enough_features = implemented_count >= 5
 all_passed &= check(has_enough_features, 
 f"Total optional features: {implemented_count}/5 required",
 f"{'Exceeds' if implemented_count > 5 else 'Meets'} requirement")
 
 # Final Summary
 print(f"\n{BLUE}{'='*70}{RESET}")
 print(f"{BLUE}Verification Summary{RESET}")
 print(f"{BLUE}{'='*70}{RESET}\n")
 
 if all_passed:
 print(f"{GREEN} ALL CHECKS PASSED!{RESET}")
 print(f"\n{GREEN}Your project is ready for HuggingFace deployment!{RESET}")
 print(f"\n{BLUE}Next steps:{RESET}")
 print(f" 1. Follow docs/deployment.md to deploy to HuggingFace Spaces")
 print(f" 2. Test the deployed app with a Gemini API key")
 print(f" 3. Submit for certification\n")
 return 0
 else:
 print(f"{RED} SOME CHECKS FAILED{RESET}")
 print(f"\n{YELLOW}Please fix the issues above before deployment.{RESET}")
 print(f"{YELLOW}See CERTIFICATION_CHECKLIST.md for detailed requirements.{RESET}\n")
 return 1

if __name__ == "__main__":
 sys.exit(main())
