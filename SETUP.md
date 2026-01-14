# 项目设置指南

## 1. 安装 Git

### Windows
下载并安装 Git for Windows：
https://git-scm.com/download/win

安装后重启 PowerShell。

### 验证安装
```powershell
git --version
```

## 2. 初始化仓库

```powershell
cd D:\Project\medical_embedding

# 初始化 Git
git init

# 配置用户信息（如果还没配置）
git config user.name "你的名字"
git config user.email "你的邮箱"

# 添加所有文件
git add .

# 首次提交
git commit -m "feat: 初始化医疗文献 Agentic RAG 系统

- 基础 RAG 流程（Embedding + Hybrid Search + Rerank + Generation）
- LlamaIndex 适配器（Phase 1）
- 单元测试
- 使用示例

Co-Authored-By: Warp <agent@warp.dev>"
```

## 3. 创建 GitHub 仓库

### 方法 1：使用 GitHub CLI (推荐)
```powershell
# 安装 GitHub CLI
winget install GitHub.cli

# 登录
gh auth login

# 创建仓库并推送
gh repo create medical-agentic-rag --public --source=. --push
```

### 方法 2：手动创建
1. 访问 https://github.com/new
2. 创建名为 `medical-agentic-rag` 的仓库
3. 推送代码：

```powershell
git remote add origin https://github.com/你的用户名/medical-agentic-rag.git
git branch -M main
git push -u origin main
```

## 4. 安装依赖

```powershell
cd D:\Project\medical_embedding

# 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

## 5. 验证安装

```powershell
# 设置 PYTHONPATH
$env:PYTHONPATH = "D:\Project\medical_embedding"

# 运行测试
pytest tests/test_adapters.py -v
```

## 6. 使用示例

```powershell
# 运行示例脚本
python scripts/example_llama_agent.py --example 2
```

## 项目结构

```
medical_embedding/
├── config/                 # 配置
├── src/
│   ├── adapters/          # 🆕 LlamaIndex 适配器
│   │   ├── llama_retriever.py
│   │   └── llama_tools.py
│   ├── embedder.py        # Qwen3-Embedding
│   ├── vector_store.py    # Milvus Hybrid Search
│   ├── reranker.py        # Qwen3-Reranker
│   ├── retriever.py       # 检索器
│   ├── generator.py       # 答案生成
│   └── pipeline.py        # RAG 管道
├── scripts/
│   └── example_llama_agent.py  # 🆕 使用示例
├── tests/
│   └── test_adapters.py   # 🆕 适配器测试
└── requirements.txt       # 依赖（已更新）
```

## 下一步

Phase 1 已完成。继续实现：
- Phase 2: Agentic 核心组件 (QueryRouter, QueryRewriter, ResultEvaluator)
- Phase 3: LlamaIndex Agent 集成
- Phase 4: DSPy 优化
