"""
Embedding 微调配置
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


def _get_env(key: str, default: str = None, cast_type: type = str):
    """获取环境变量"""
    value = os.getenv(key, default)
    if value is None:
        return None
    if cast_type == bool:
        return value.lower() in ("true", "1", "yes")
    return cast_type(value)


# 项目根目录
BASE_DIR = Path(__file__).parent.parent


@dataclass
class TrainingConfig:
    """微调配置"""
    
    # ========== 模型配置 ==========
    # 基础模型 (HuggingFace 模型名或本地路径)
    base_model: str = _get_env("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
    
    # 输出目录
    output_dir: Path = BASE_DIR / "models" / "medical-embedding"
    
    # ========== 数据配置 ==========
    train_file: Path = BASE_DIR / "data" / "training" / "train.jsonl"
    val_file: Path = BASE_DIR / "data" / "training" / "val.jsonl"
    
    # ========== 训练参数 ==========
    # 批大小 (根据 GPU 内存调整)
    # RTX 3090 24GB: batch_size=4, grad_accum=8 → effective_batch=32
    # A100 80GB: batch_size=16, grad_accum=2 → effective_batch=32
    batch_size: int = _get_env("TRAIN_BATCH_SIZE", "4", int)
    gradient_accumulation_steps: int = _get_env("GRAD_ACCUM_STEPS", "8", int)
    
    # 学习率
    learning_rate: float = _get_env("TRAIN_LR", "2e-5", float)
    warmup_ratio: float = 0.1
    
    # 训练轮数
    num_epochs: int = _get_env("TRAIN_EPOCHS", "3", int)
    
    # 最大序列长度
    max_length: int = _get_env("MAX_SEQ_LENGTH", "512", int)
    
    # ========== 损失函数配置 ==========
    # 可选: "mnrl" (MultipleNegativesRankingLoss), "cosine", "triplet", "contrastive"
    loss_type: str = "mnrl"
    
    # InfoNCE 温度参数
    temperature: float = 0.05
    
    # ========== 优化器配置 ==========
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # ========== 评估配置 ==========
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100
    
    # ========== 硬件配置 ==========
    fp16: bool = True  # 混合精度训练
    bf16: bool = False  # 如果 GPU 支持 BF16，可以开启
    
    # 多卡训练
    local_rank: int = -1
    
    # ========== 其他 ==========
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None


@dataclass 
class DataConfig:
    """数据构建配置"""
    
    # 数据源开关
    use_pubmedqa: bool = True
    use_medqa: bool = True
    use_medmcqa: bool = False
    use_local_docs: bool = True
    
    # 采样配置
    max_samples_per_source: int = _get_env("MAX_TRAIN_SAMPLES", "10000", int)
    
    # 负样本
    num_random_negatives: int = 3
    
    # 输出
    output_dir: Path = BASE_DIR / "data" / "training"


# 默认配置实例
training_config = TrainingConfig()
data_config = DataConfig()
