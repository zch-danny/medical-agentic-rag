"""
医学 Embedding 训练数据构建器

支持从多种数据源构建对比学习训练对:
1. 本地已索引的医学文献
2. PubMed 摘要 (标题-摘要对)
3. 医学 QA 数据集 (MedQA, PubMedQA, MedMCQA)
4. 自定义 query-passage 对
"""
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from datasets import Dataset, load_dataset
from loguru import logger
from tqdm import tqdm


@dataclass
class TrainingExample:
    """训练样本"""
    query: str
    positive: str  # 正样本
    negatives: List[str] = field(default_factory=list)  # 负样本（可选）
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "positive": self.positive,
            "negatives": self.negatives,
        }


@dataclass
class DataBuilderConfig:
    """数据构建配置"""
    # 数据源开关
    use_pubmed: bool = True
    use_medqa: bool = True
    use_pubmedqa: bool = True
    use_medmcqa: bool = False
    use_local_docs: bool = True
    
    # 采样配置
    max_samples_per_source: int = 10000
    min_query_length: int = 10
    min_passage_length: int = 50
    max_passage_length: int = 2000
    
    # 负样本配置
    num_hard_negatives: int = 0  # 难负样本数量 (需要额外计算)
    num_random_negatives: int = 3  # 随机负样本数量
    
    # 输出配置
    output_dir: Path = Path("data/training")
    train_ratio: float = 0.9
    seed: int = 42


class MedicalDataBuilder:
    """
    医学训练数据构建器
    
    使用示例:
    ```python
    builder = MedicalDataBuilder()
    
    # 从多个数据源构建
    examples = builder.build_all()
    
    # 保存为 sentence-transformers 格式
    builder.save_for_sentence_transformers(examples, "data/training")
    
    # 或保存为 JSON
    builder.save_as_json(examples, "data/training/medical_pairs.json")
    ```
    """
    
    def __init__(self, config: Optional[DataBuilderConfig] = None):
        self.config = config or DataBuilderConfig()
        random.seed(self.config.seed)
        
        # 确保输出目录存在
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def build_all(self) -> List[TrainingExample]:
        """从所有启用的数据源构建训练数据"""
        all_examples: List[TrainingExample] = []
        
        if self.config.use_pubmedqa:
            logger.info("加载 PubMedQA 数据...")
            all_examples.extend(self._build_from_pubmedqa())
        
        if self.config.use_medqa:
            logger.info("加载 MedQA 数据...")
            all_examples.extend(self._build_from_medqa())
        
        if self.config.use_medmcqa:
            logger.info("加载 MedMCQA 数据...")
            all_examples.extend(self._build_from_medmcqa())
        
        if self.config.use_local_docs:
            logger.info("加载本地文档数据...")
            all_examples.extend(self._build_from_local_docs())
        
        # 添加随机负样本
        if self.config.num_random_negatives > 0:
            all_examples = self._add_random_negatives(all_examples)
        
        # 打乱顺序
        random.shuffle(all_examples)
        
        logger.info(f"总共构建 {len(all_examples)} 个训练样本")
        return all_examples
    
    def _build_from_pubmedqa(self) -> List[TrainingExample]:
        """从 PubMedQA 构建训练对 (问题-摘要)"""
        examples = []
        
        try:
            # 加载 PubMedQA labeled 子集
            dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
            
            for item in tqdm(dataset, desc="处理 PubMedQA"):
                question = item.get("question", "")
                context = item.get("context", {})
                
                # 合并所有上下文段落
                if isinstance(context, dict):
                    passages = context.get("contexts", [])
                    passage = " ".join(passages) if passages else ""
                else:
                    passage = str(context)
                
                if self._is_valid_pair(question, passage):
                    examples.append(TrainingExample(
                        query=question,
                        positive=passage[:self.config.max_passage_length],
                    ))
                
                if len(examples) >= self.config.max_samples_per_source:
                    break
                    
        except Exception as e:
            logger.warning(f"加载 PubMedQA 失败: {e}")
        
        logger.info(f"从 PubMedQA 构建 {len(examples)} 个样本")
        return examples
    
    def _build_from_medqa(self) -> List[TrainingExample]:
        """从 MedQA 构建训练对 (问题+选项-解释)"""
        examples = []
        
        try:
            # MedQA-USMLE
            dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
            
            for item in tqdm(dataset, desc="处理 MedQA"):
                question = item.get("question", "")
                options = item.get("options", {})
                answer_idx = item.get("answer_idx", "")
                
                # 构建完整问题
                if options:
                    option_text = " ".join([f"{k}: {v}" for k, v in options.items()])
                    full_question = f"{question} {option_text}"
                else:
                    full_question = question
                
                # 正确答案作为"正样本"的一部分
                correct_answer = options.get(answer_idx, "") if options else ""
                if correct_answer:
                    passage = f"{question} 答案是: {correct_answer}"
                else:
                    passage = question
                
                if self._is_valid_pair(full_question, passage):
                    examples.append(TrainingExample(
                        query=full_question[:500],
                        positive=passage,
                    ))
                
                if len(examples) >= self.config.max_samples_per_source:
                    break
                    
        except Exception as e:
            logger.warning(f"加载 MedQA 失败: {e}")
        
        logger.info(f"从 MedQA 构建 {len(examples)} 个样本")
        return examples
    
    def _build_from_medmcqa(self) -> List[TrainingExample]:
        """从 MedMCQA 构建训练对"""
        examples = []
        
        try:
            dataset = load_dataset("openlifescienceai/medmcqa", split="train")
            
            for item in tqdm(dataset, desc="处理 MedMCQA"):
                question = item.get("question", "")
                explanation = item.get("exp", "")
                
                # 使用解释作为正样本
                if explanation and self._is_valid_pair(question, explanation):
                    examples.append(TrainingExample(
                        query=question,
                        positive=explanation[:self.config.max_passage_length],
                    ))
                
                if len(examples) >= self.config.max_samples_per_source:
                    break
                    
        except Exception as e:
            logger.warning(f"加载 MedMCQA 失败: {e}")
        
        logger.info(f"从 MedMCQA 构建 {len(examples)} 个样本")
        return examples
    
    def _build_from_local_docs(self) -> List[TrainingExample]:
        """从本地已索引文档构建训练对"""
        examples = []
        
        # 检查本地数据目录
        parsed_dir = self.config.output_dir.parent / "parsed"
        if not parsed_dir.exists():
            logger.warning(f"本地解析目录不存在: {parsed_dir}")
            return examples
        
        # 遍历所有解析后的 markdown 文件
        md_files = list(parsed_dir.glob("**/*.md"))
        
        for md_file in tqdm(md_files, desc="处理本地文档"):
            try:
                content = md_file.read_text(encoding="utf-8")
                # 使用标题作为 query，正文作为 passage
                pairs = self._extract_title_passage_pairs(content)
                examples.extend(pairs)
                
                if len(examples) >= self.config.max_samples_per_source:
                    break
            except Exception as e:
                logger.debug(f"处理文件失败 {md_file}: {e}")
        
        logger.info(f"从本地文档构建 {len(examples)} 个样本")
        return examples
    
    def _extract_title_passage_pairs(self, content: str) -> List[TrainingExample]:
        """从 Markdown 内容提取标题-正文对"""
        import re
        
        examples = []
        
        # 按标题切分
        sections = re.split(r'\n(#{1,3})\s+(.+)\n', content)
        
        current_title = ""
        for i, section in enumerate(sections):
            if section.startswith("#"):
                continue
            elif i > 0 and sections[i-1].startswith("#"):
                current_title = sections[i] if i < len(sections) else ""
            else:
                # 这是正文部分
                passage = section.strip()
                if current_title and self._is_valid_pair(current_title, passage):
                    examples.append(TrainingExample(
                        query=current_title,
                        positive=passage[:self.config.max_passage_length],
                    ))
        
        return examples
    
    def _add_random_negatives(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """为每个样本添加随机负样本"""
        all_passages = [e.positive for e in examples]
        
        for example in tqdm(examples, desc="添加负样本"):
            # 随机选择不同的段落作为负样本
            candidates = [p for p in all_passages if p != example.positive]
            if candidates:
                negatives = random.sample(
                    candidates, 
                    min(self.config.num_random_negatives, len(candidates))
                )
                example.negatives = negatives
        
        return examples
    
    def _is_valid_pair(self, query: str, passage: str) -> bool:
        """验证训练对是否有效"""
        if not query or not passage:
            return False
        if len(query) < self.config.min_query_length:
            return False
        if len(passage) < self.config.min_passage_length:
            return False
        return True
    
    def save_for_sentence_transformers(
        self, 
        examples: List[TrainingExample],
        output_dir: Union[str, Path],
    ) -> Tuple[Path, Path]:
        """
        保存为 sentence-transformers 训练格式
        
        生成两个文件:
        - train.jsonl: 训练集
        - val.jsonl: 验证集
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 划分训练集和验证集
        split_idx = int(len(examples) * self.config.train_ratio)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        train_path = output_dir / "train.jsonl"
        val_path = output_dir / "val.jsonl"
        
        # 保存训练集
        with open(train_path, "w", encoding="utf-8") as f:
            for ex in train_examples:
                # sentence-transformers InputExample 格式
                line = json.dumps({
                    "query": ex.query,
                    "pos": [ex.positive],
                    "neg": ex.negatives,
                }, ensure_ascii=False)
                f.write(line + "\n")
        
        # 保存验证集
        with open(val_path, "w", encoding="utf-8") as f:
            for ex in val_examples:
                line = json.dumps({
                    "query": ex.query,
                    "pos": [ex.positive],
                    "neg": ex.negatives,
                }, ensure_ascii=False)
                f.write(line + "\n")
        
        logger.info(f"训练集: {len(train_examples)} 样本 -> {train_path}")
        logger.info(f"验证集: {len(val_examples)} 样本 -> {val_path}")
        
        return train_path, val_path
    
    def save_as_json(
        self, 
        examples: List[TrainingExample], 
        output_path: Union[str, Path]
    ) -> Path:
        """保存为 JSON 格式"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [e.to_dict() for e in examples]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存 {len(examples)} 个样本到 {output_path}")
        return output_path
    
    def load_from_json(self, input_path: Union[str, Path]) -> List[TrainingExample]:
        """从 JSON 加载训练数据"""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return [
            TrainingExample(
                query=item["query"],
                positive=item["positive"],
                negatives=item.get("negatives", []),
            )
            for item in data
        ]


def build_medical_training_data(
    output_dir: str = "data/training",
    max_samples: int = 10000,
) -> Tuple[Path, Path]:
    """
    快速构建医学训练数据的便捷函数
    
    Args:
        output_dir: 输出目录
        max_samples: 每个数据源的最大样本数
    
    Returns:
        (train_path, val_path)
    """
    config = DataBuilderConfig(
        max_samples_per_source=max_samples,
        output_dir=Path(output_dir),
    )
    
    builder = MedicalDataBuilder(config)
    examples = builder.build_all()
    
    return builder.save_for_sentence_transformers(examples, output_dir)


if __name__ == "__main__":
    # 测试
    train_path, val_path = build_medical_training_data(
        output_dir="data/training",
        max_samples=1000,
    )
    print(f"训练数据: {train_path}")
    print(f"验证数据: {val_path}")
