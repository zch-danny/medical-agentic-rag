# 医疗文献 RAG 系统

基于 Qwen3-Embedding + Qwen3-Reranker + Milvus 的医疗文献语义检索系统。

## 功能

- PDF 文献解析（MinerU）
- 混合检索（向量 + BM25）
- 语义重排序
- AI 答案生成（DeepSeek/OpenAI）
- REST API 服务
- RAG 评估框架（DeepEval + MIRAGE 医学基准）

## 项目结构

```
medical_embedding/
├── config/
│   └── settings.py      # 配置管理
├── src/
│   ├── api.py           # FastAPI 服务
│   ├── document_loader.py
│   ├── embedder.py
│   ├── embedding_cache.py
│   ├── vector_store.py
│   ├── reranker.py
│   ├── retriever.py
│   ├── generator.py
│   ├── pipeline.py
│   └── health.py
├── scripts/
│   ├── index_documents.py
│   ├── search.py
│   ├── evaluate.py           # 检索指标评估
│   ├── evaluate_deepeval.py  # DeepEval 端到端评估
│   └── evaluate_mirage.py    # MIRAGE 医学基准评估
├── docker/
│   └── milvus-standalone.yml
├── data/
│   ├── documents/       # PDF 文件
│   ├── parsed/          # 解析结果
│   ├── cache/           # 嵌入缓存
│   └── evaluation/      # 评估数据集
└── tests/
```

## 快速开始

### 1. 环境准备

```powershell
# 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

复制 `.env.example` 为 `.env`，填写配置：

```env
# 模型路径（本地模型）
EMBEDDING_MODEL_PATH=D:/models/Qwen3-Embedding-8B
RERANKER_MODEL_PATH=D:/models/Qwen3-Reranker-8B

# 或使用 HuggingFace 模型名（自动下载）
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
RERANKER_MODEL=Qwen/Qwen3-Reranker-8B

# Milvus
MILVUS_URI=http://127.0.0.1:19530

# LLM API（可选，用于答案生成）
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

### 3. 启动 Milvus

```powershell
docker compose -f docker/milvus-standalone.yml up -d
```

### 4. 使用方式

#### 方式一：命令行

```powershell
$env:PYTHONPATH = "D:\Project\medical_embedding"

# 索引文档
python scripts/index_documents.py --input-dir data/documents

# 交互式检索
python scripts/search.py

# 单次查询
python scripts/search.py -q "糖尿病的治疗方法"
```

#### 方式二：API 服务

```powershell
$env:PYTHONPATH = "D:\Project\medical_embedding"
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

API 文档：http://localhost:8000/docs

## API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/upload` | POST | 上传 PDF 文件 |
| `/files` | GET | 列出已上传文件 |
| `/files/{id}` | DELETE | 删除文件 |
| `/index` | POST | 索引文件到向量库 |
| `/search` | POST/GET | 检索文献 |
| `/health` | GET | 健康检查 |
| `/paper2figure` | POST | 生成论文图表（架构图/流程图等） |
| `/paper2ppt` | POST | 从论文生成完整 PPT 演示文稿 |
| `/ppt-polish` | POST | PPT 美化（配色/字体/页码） |

### 示例

```bash
# 上传
curl -X POST http://localhost:8000/upload -F "file=@文献.pdf"

# 检索
curl "http://localhost:8000/search?q=高血压治疗&top_k=5"
```

## 配置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| CHUNK_SIZE | 512 | 文本分块大小（字符） |
| CHUNK_OVERLAP | 64 | 分块重叠 |
| TOP_K | 50 | 粗召回数量 |
| RERANK_TOP_K | 10 | 重排后返回数量 |
| HYBRID_ALPHA | 0.7 | 向量权重（0=纯BM25，1=纯向量） |

## 硬件需求

| 配置 | GPU | 内存 | 说明 |
|------|-----|------|------|
| 最低 | RTX 3090 24GB | 32GB | 模型分时加载 |
| 推荐 | A100 80GB | 64GB | 双模型同时加载 |
| CPU-only | 无 | 32GB | 使用量化模型或 API |

## 测试

```powershell
pytest tests/ -v
```

## 评估

### 检索质量评估
```powershell
$env:PYTHONPATH = "D:\Project\medical_embedding"
python scripts/evaluate.py --test-file data/evaluation/test_queries.json
```

### DeepEval 端到端评估
评估 Faithfulness（幻觉检测）、Answer Relevancy、Context Precision 等指标：
```powershell
pip install deepeval
python scripts/evaluate_deepeval.py --test-file data/evaluation/test_queries_template.json
```

### MIRAGE 医学基准评估
使用 7,663 道英文医学问答题（MMLU-Med、MedQA、MedMCQA、PubMedQA、BioASQ）：
```powershell
# 下载基准数据
python scripts/evaluate_mirage.py --download

# 快速测试
python scripts/evaluate_mirage.py --dataset mmlu --limit 50

# 完整评估（含答案生成）
python scripts/evaluate_mirage.py --dataset all --use-generation
```

## Paper2Any 论文多模态输出

自动从论文内容生成科研图表、PPT 演示文稿等。

### Paper2Figure - 图表生成

```powershell
$env:PYTHONPATH = "D:\Project\medical_embedding"

# 从 PDF 生成架构图
python scripts/paper2figure.py --pdf paper.pdf --type architecture

# 从文本生成流程图
python scripts/paper2figure.py --text "本文提出了一种..." --type flowchart

# 自动检测图表类型并生成多种格式
python scripts/paper2figure.py --pdf paper.pdf --formats html,pptx,svg --preview
```

图表类型：`auto`（自动检测）、`architecture`（架构图）、`roadmap`（路线图）、`flowchart`（流程图）、`experiment`（数据图）

### Paper2PPT - PPT 生成

```powershell
# 从 PDF 生成完整 PPT
python scripts/paper2ppt.py --pdf paper.pdf --style academic

# 从文本生成 PPT
python scripts/paper2ppt.py --text "论文内容..." --style business

# 指定输出路径
python scripts/paper2ppt.py --pdf paper.pdf --output ./presentation.pptx
```

PPT 风格：`academic`（学术）、`business`（商务）、`modern`（现代）、`colorful`（多彩）

### PPTPolish - PPT 美化

```powershell
# 美化已有 PPT
python scripts/paper2ppt.py --polish input.pptx --color academic_blue --font professional

# 查看可用配色方案
python scripts/paper2ppt.py --list-schemes
```

配色方案：`academic_blue`、`modern_green`、`elegant_purple`、`business_navy`、`warm_orange`、`minimal_gray`

### API 调用

```bash
# 生成图表
curl -X POST "http://localhost:8000/paper2figure" \
  -H "Content-Type: application/json" \
  -d '{"content": "论文内容...", "figure_type": "architecture"}'

# 生成 PPT
curl -X POST "http://localhost:8000/paper2ppt" \
  -H "Content-Type: application/json" \
  -d '{"content": "论文内容...", "style": "academic"}'

# 美化 PPT
curl -X POST "http://localhost:8000/ppt-polish" \
  -H "Content-Type: application/json" \
  -d '{"pptx_path": "./output/presentation.pptx", "color_scheme": "modern_green"}'
```

## License

MIT
