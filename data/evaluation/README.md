# 评估数据目录

## 文件说明

### test_queries_template.json
中英双语测试数据模板，用于 DeepEval 评估。

**格式：**
```json
[
    {
        "question": "问题文本",
        "ground_truth": "参考答案（可选，用于 Contextual Recall）",
        "language": "zh 或 en",
        "category": "分类标签（可选）"
    }
]
```

**分类建议：**
- `diagnosis` - 诊断类问题
- `treatment` - 治疗类问题
- `mechanism` - 机制类问题
- `adverse_effects` - 不良反应类问题
- `prevention` - 预防类问题

### mirage_benchmark.json
MIRAGE 医学基准数据集（自动下载）。包含 7,663 个英文医学问答题。

**数据集：**
- MMLU-Med: 医学考试题
- MedQA: 美国医师执照考试
- MedMCQA: 印度医学入学考试
- PubMedQA: 生物医学研究问答
- BioASQ: 生物医学语义问答

## 使用方法

### DeepEval 评估
```powershell
$env:PYTHONPATH = "D:\Project\medical_embedding"

# 使用模板运行评估
python scripts/evaluate_deepeval.py --test-file data/evaluation/test_queries_template.json

# 指定评估指标
python scripts/evaluate_deepeval.py --test-file data/evaluation/test_queries_template.json --metrics faithfulness relevancy
```

### MIRAGE 基准评估
```powershell
$env:PYTHONPATH = "D:\Project\medical_embedding"

# 下载基准数据
python scripts/evaluate_mirage.py --download

# 快速测试 (限制 50 题)
python scripts/evaluate_mirage.py --dataset mmlu --limit 50

# 完整评估
python scripts/evaluate_mirage.py --dataset all --use-generation
```

## 构建自定义测试集

1. 复制 `test_queries_template.json` 为你的测试文件
2. 从你的文献中提取 50-100 个典型问题
3. 为每个问题编写参考答案 (ground_truth)
4. 标注语言和分类

**建议测试集规模：**
- 快速验证: 20-50 题
- 常规评估: 50-100 题
- 完整基准: 100+ 题
