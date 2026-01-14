# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Medical Literature RAG System (医疗文献 RAG 系统) - A semantic retrieval system for medical literature using Qwen3-Embedding, Qwen3-Reranker, and Milvus vector database. Supports hybrid search (dense vectors + BM25), reranking, and AI-powered answer generation.

## Development Commands

**IMPORTANT**: Always set PYTHONPATH before running any script:
```powershell
$env:PYTHONPATH = "D:\Project\medical_embedding"
```

### Run Tests
```powershell
pytest tests/ -v
```

Run a single test file:
```powershell
pytest tests/test_adapters.py -v
```

### Start API Server
```powershell
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
API docs available at http://localhost:8000/docs

### Index Documents
```powershell
python scripts/index_documents.py --input-dir data/documents
```

### Interactive Search
```powershell
python scripts/search.py
# Or single query:
python scripts/search.py -q "糖尿病的治疗方法"
```

### Start Milvus (Docker)
```powershell
docker compose -f docker/milvus-standalone.yml up -d
```

## Architecture

### Core RAG Pipeline (`src/`)
- `pipeline.py` - End-to-end RAG orchestration: Query → Embedding → Hybrid Search → Rerank → Generate
- `retriever.py` - Retriever class integrating embedder, vector store, and reranker
- `embedder.py` - Qwen3-Embedding wrapper with medical domain instruction support
- `vector_store.py` - Milvus client with hybrid search (dense + sparse BM25)
- `reranker.py` - Qwen3-Reranker for result reranking
- `generator.py` - LLM-based answer generation (DeepSeek/OpenAI)
- `api.py` - FastAPI endpoints for upload, index, search

### Agentic Components (`src/agentic/`)
- `query_router.py` - Routes queries to optimal retrieval strategy (VECTOR/BM25/HYBRID/DIRECT/WEB) based on query characteristics
- `query_rewriter.py` - Query optimization with medical term standardization, synonym expansion, and follow-up completion
- `result_evaluator.py` - Evaluates retrieval quality and decides next action (SUFFICIENT/PARTIAL/INSUFFICIENT)

### LlamaIndex Integration (`src/adapters/`)
- `llama_retriever.py` - Wraps MedicalRetriever as LlamaIndex BaseRetriever
- `llama_tools.py` - FunctionTool wrappers for agent integration

## Key Patterns

### Configuration
All settings in `config/settings.py`, loaded from environment variables or `.env` file. Access via:
```python
from config import settings as cfg
# or
from config.settings import settings
```

### Lazy Model Loading
Components use lazy loading to defer expensive model initialization:
```python
@property
def embedder(self) -> MedicalEmbedder:
    if self._embedder is None:
        self._embedder = MedicalEmbedder(...)
    return self._embedder
```

### Testing Without Models
Tests use mocks to avoid requiring actual models or Milvus:
```python
@pytest.fixture
def mock_retriever(self):
    mock = MagicMock()
    mock.search.return_value = [{"entity": {...}, "rerank_score": 0.95}]
    return mock
```

### Document Result Format
Retrieval results use this structure:
```python
{
    "entity": {
        "original_text": "原始文本",
        "text": "分词后文本",
        "source": "filename.pdf",
        "title": "文献标题",
        "year": "2024",
        "doi": "10.xxx/xxx",
        "keywords": '["关键词1", "关键词2"]',  # JSON string
        "chunk_index": 0
    },
    "rerank_score": 0.95,  # or "score" or "distance"
}
```

## RAG Evaluation

Three-tier evaluation architecture for bilingual (Chinese/English) medical RAG:

### Retrieval Evaluation (existing)
```powershell
python scripts/evaluate.py --test-file data/evaluation/test_queries.json --alpha 0.7
```
Metrics: MRR@K, Recall@K, NDCG@K

### DeepEval End-to-End Evaluation
```powershell
pip install deepeval
python scripts/evaluate_deepeval.py --test-file data/evaluation/test_queries_template.json
python scripts/evaluate_deepeval.py --test-file data/evaluation/test_queries_template.json --metrics faithfulness relevancy precision
```
Metrics: Faithfulness (hallucination detection), Answer Relevancy, Contextual Precision/Recall

### MIRAGE Medical Benchmark
```powershell
# Download benchmark (7,663 English medical QA questions)
python scripts/evaluate_mirage.py --download

# Quick test
python scripts/evaluate_mirage.py --dataset mmlu --limit 50

# Full evaluation with generation
python scripts/evaluate_mirage.py --dataset all --use-generation
```
Datasets: MMLU-Med, MedQA, MedMCQA, PubMedQA, BioASQ

## Data Directories
- `data/documents/` - Source PDF files
- `data/parsed/` - MinerU parsed output
- `data/cache/embeddings/` - Embedding cache
- `data/evaluation/` - Test datasets and MIRAGE benchmark

## Environment Setup
1. Copy `.env.example` to `.env`
2. Configure model paths (local) or HuggingFace model names (auto-download)
3. Set LLM API key for answer generation (optional)
4. For DeepEval: `pip install deepeval` and optionally set `CONFIDENT_API_KEY` for web dashboard
