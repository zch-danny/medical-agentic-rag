"""
DSPy 优化器封装

提供便捷的优化流程：
- 训练数据加载
- 优化器配置
- 模型编译和保存
- 编译模型加载
"""

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import dspy
from loguru import logger

from .modules import (
    OptimizedRAG,
    OptimizedRewriter,
    OptimizedEvaluator,
    ChainOfThoughtRAG,
    MedicalRAGPipeline,
)
from .metrics import MedicalQAMetrics, create_dspy_metric


@dataclass
class OptimizationConfig:
    """优化配置"""
    # 优化器设置
    optimizer_type: str = "BootstrapFewShot"  # BootstrapFewShot, MIPRO, BootstrapFewShotWithRandomSearch
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 8
    max_rounds: int = 1
    
    # 训练设置
    num_threads: int = 4
    
    # 评估设置
    metric_type: str = "semantic_f1"
    metric_threshold: float = 0.7
    
    # 保存设置
    save_path: Optional[Path] = None
    
    # LLM 设置
    teacher_model: Optional[str] = None  # 用于生成 demo 的模型


@dataclass
class TrainingExample:
    """训练样本"""
    question: str
    context: str
    answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dspy_example(self) -> dspy.Example:
        """转换为 DSPy Example"""
        return dspy.Example(
            question=self.question,
            context=self.context,
            answer=self.answer,
        ).with_inputs("question", "context")


class DSPyOptimizer:
    """
    DSPy 优化器封装
    
    提供便捷的模块优化流程。
    
    示例:
        ```python
        # 创建优化器
        optimizer = DSPyOptimizer(
            module=OptimizedRAG(),
            config=OptimizationConfig(),
        )
        
        # 加载训练数据
        optimizer.load_training_data("data/training/qa_pairs.json")
        
        # 执行优化
        compiled_module = optimizer.compile()
        
        # 保存编译结果
        optimizer.save("compiled/optimized_rag.pkl")
        ```
    """
    
    def __init__(
        self,
        module: dspy.Module,
        config: Optional[OptimizationConfig] = None,
        metric: Optional[Callable] = None,
    ):
        """
        Args:
            module: 要优化的 DSPy 模块
            config: 优化配置
            metric: 评估函数
        """
        self.module = module
        self.config = config or OptimizationConfig()
        
        # 设置评估函数
        if metric is None:
            self.metric = create_dspy_metric(
                metric_type=self.config.metric_type,
                threshold=self.config.metric_threshold,
            )
        else:
            self.metric = metric
        
        # 训练数据
        self.train_set: List[dspy.Example] = []
        self.dev_set: List[dspy.Example] = []
        
        # 编译结果
        self.compiled_module: Optional[dspy.Module] = None
        
        # 优化历史
        self.optimization_history: List[Dict] = []
    
    def load_training_data(
        self,
        path: Union[str, Path],
        dev_ratio: float = 0.2,
    ) -> None:
        """
        加载训练数据
        
        Args:
            path: 数据文件路径（JSON 格式）
            dev_ratio: 验证集比例
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"训练数据文件不存在: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 转换为 DSPy Example
        examples = []
        for item in data:
            example = dspy.Example(
                question=item.get("question", ""),
                context=item.get("context", ""),
                answer=item.get("answer", ""),
            ).with_inputs("question", "context")
            examples.append(example)
        
        # 划分训练集和验证集
        split_idx = int(len(examples) * (1 - dev_ratio))
        self.train_set = examples[:split_idx]
        self.dev_set = examples[split_idx:]
        
        logger.info(f"加载训练数据: {len(self.train_set)} 训练样本, {len(self.dev_set)} 验证样本")
    
    def add_examples(
        self,
        examples: List[TrainingExample],
        is_dev: bool = False,
    ) -> None:
        """
        添加训练样本
        
        Args:
            examples: 训练样本列表
            is_dev: 是否添加到验证集
        """
        dspy_examples = [ex.to_dspy_example() for ex in examples]
        
        if is_dev:
            self.dev_set.extend(dspy_examples)
        else:
            self.train_set.extend(dspy_examples)
        
        logger.info(f"添加 {len(dspy_examples)} 个样本到 {'验证' if is_dev else '训练'}集")
    
    def compile(
        self,
        verbose: bool = True,
    ) -> dspy.Module:
        """
        编译优化模块
        
        Args:
            verbose: 是否输出详细日志
            
        Returns:
            编译后的模块
        """
        if not self.train_set:
            raise ValueError("没有训练数据，请先调用 load_training_data 或 add_examples")
        
        logger.info(f"开始优化，使用 {self.config.optimizer_type} 优化器")
        
        # 创建优化器
        optimizer = self._create_optimizer()
        
        # 记录开始时间
        start_time = datetime.now()
        
        # 执行编译
        try:
            self.compiled_module = optimizer.compile(
                self.module,
                trainset=self.train_set,
                valset=self.dev_set if self.dev_set else None,
            )
            
            # 记录优化结果
            end_time = datetime.now()
            self.optimization_history.append({
                "timestamp": end_time.isoformat(),
                "duration_seconds": (end_time - start_time).total_seconds(),
                "optimizer_type": self.config.optimizer_type,
                "train_size": len(self.train_set),
                "dev_size": len(self.dev_set),
                "success": True,
            })
            
            logger.info(f"优化完成，耗时 {(end_time - start_time).total_seconds():.2f} 秒")
            
        except Exception as e:
            logger.error(f"优化失败: {e}")
            self.optimization_history.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "success": False,
            })
            raise
        
        return self.compiled_module
    
    def _create_optimizer(self) -> Any:
        """创建优化器实例"""
        optimizer_type = self.config.optimizer_type
        
        if optimizer_type == "BootstrapFewShot":
            return dspy.BootstrapFewShot(
                metric=self.metric,
                max_bootstrapped_demos=self.config.max_bootstrapped_demos,
                max_labeled_demos=self.config.max_labeled_demos,
                max_rounds=self.config.max_rounds,
            )
        
        elif optimizer_type == "BootstrapFewShotWithRandomSearch":
            return dspy.BootstrapFewShotWithRandomSearch(
                metric=self.metric,
                max_bootstrapped_demos=self.config.max_bootstrapped_demos,
                max_labeled_demos=self.config.max_labeled_demos,
                num_threads=self.config.num_threads,
            )
        
        elif optimizer_type == "MIPRO":
            # MIPRO 需要额外配置
            try:
                return dspy.MIPRO(
                    metric=self.metric,
                    num_threads=self.config.num_threads,
                )
            except Exception as e:
                logger.warning(f"MIPRO 初始化失败，回退到 BootstrapFewShot: {e}")
                return dspy.BootstrapFewShot(
                    metric=self.metric,
                    max_bootstrapped_demos=self.config.max_bootstrapped_demos,
                    max_labeled_demos=self.config.max_labeled_demos,
                )
        
        else:
            logger.warning(f"未知优化器类型: {optimizer_type}，使用 BootstrapFewShot")
            return dspy.BootstrapFewShot(
                metric=self.metric,
                max_bootstrapped_demos=self.config.max_bootstrapped_demos,
                max_labeled_demos=self.config.max_labeled_demos,
            )
    
    def evaluate(
        self,
        test_set: Optional[List[dspy.Example]] = None,
    ) -> Dict[str, Any]:
        """
        评估编译后的模块
        
        Args:
            test_set: 测试集（默认使用验证集）
            
        Returns:
            评估结果
        """
        if self.compiled_module is None:
            raise ValueError("模块尚未编译，请先调用 compile")
        
        test_data = test_set or self.dev_set
        if not test_data:
            raise ValueError("没有测试数据")
        
        results = []
        for example in test_data:
            try:
                prediction = self.compiled_module(
                    question=example.question,
                    context=example.context,
                )
                score = self.metric(example, prediction)
                results.append({
                    "question": example.question,
                    "prediction": getattr(prediction, "answer", str(prediction)),
                    "expected": example.answer,
                    "score": score,
                })
            except Exception as e:
                results.append({
                    "question": example.question,
                    "error": str(e),
                    "score": 0.0,
                })
        
        # 计算统计信息
        scores = [r["score"] for r in results if "error" not in r]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "average_score": avg_score,
            "num_samples": len(test_data),
            "num_errors": len([r for r in results if "error" in r]),
            "results": results,
        }
    
    def save(
        self,
        path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        保存编译后的模块
        
        Args:
            path: 保存路径
            
        Returns:
            保存的文件路径
        """
        if self.compiled_module is None:
            raise ValueError("模块尚未编译，请先调用 compile")
        
        if path is None:
            path = self.config.save_path
        
        if path is None:
            # 默认保存路径
            path = Path("src/optimization/compiled") / f"compiled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存模块
        save_compiled_module(self.compiled_module, path)
        
        # 保存元数据
        meta_path = path.with_suffix(".meta.json")
        metadata = {
            "compiled_at": datetime.now().isoformat(),
            "module_type": type(self.module).__name__,
            "config": {
                "optimizer_type": self.config.optimizer_type,
                "max_bootstrapped_demos": self.config.max_bootstrapped_demos,
                "max_labeled_demos": self.config.max_labeled_demos,
            },
            "training_stats": {
                "train_size": len(self.train_set),
                "dev_size": len(self.dev_set),
            },
            "optimization_history": self.optimization_history,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存编译模块: {path}")
        return path
    
    def load(
        self,
        path: Union[str, Path],
    ) -> dspy.Module:
        """
        加载编译后的模块
        
        Args:
            path: 模块文件路径
            
        Returns:
            加载的模块
        """
        self.compiled_module = load_compiled_module(path)
        logger.info(f"加载编译模块: {path}")
        return self.compiled_module


def save_compiled_module(
    module: dspy.Module,
    path: Union[str, Path],
) -> None:
    """
    保存编译后的 DSPy 模块
    
    Args:
        module: DSPy 模块
        path: 保存路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(module, f)


def load_compiled_module(
    path: Union[str, Path],
) -> dspy.Module:
    """
    加载编译后的 DSPy 模块
    
    Args:
        path: 模块文件路径
        
    Returns:
        加载的模块
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"模块文件不存在: {path}")
    
    with open(path, "rb") as f:
        module = pickle.load(f)
    
    return module


def create_optimizer(
    module_type: str = "rag",
    config: Optional[OptimizationConfig] = None,
    **module_kwargs,
) -> DSPyOptimizer:
    """
    创建优化器的便捷函数
    
    Args:
        module_type: 模块类型 (rag, rewriter, evaluator, cot_rag, pipeline)
        config: 优化配置
        **module_kwargs: 模块参数
        
    Returns:
        DSPyOptimizer 实例
    """
    # 创建模块
    if module_type == "rag":
        module = OptimizedRAG(**module_kwargs)
    elif module_type == "cot_rag":
        module = ChainOfThoughtRAG()
    elif module_type == "rewriter":
        module = OptimizedRewriter(**module_kwargs)
    elif module_type == "evaluator":
        module = OptimizedEvaluator()
    elif module_type == "pipeline":
        module = MedicalRAGPipeline(**module_kwargs)
    else:
        raise ValueError(f"未知模块类型: {module_type}")
    
    return DSPyOptimizer(module=module, config=config)
