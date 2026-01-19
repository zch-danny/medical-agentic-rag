#!/usr/bin/env python
"""
医学 Embedding 微调脚本

使用 sentence-transformers 库对 Qwen3-Embedding 进行医学领域微调。

使用方式:
1. 准备数据:
   python -c "from src.training.data_builder import build_medical_training_data; build_medical_training_data()"

2. 运行微调:
   python scripts/finetune_embedding.py

3. 使用微调后的模型:
   在 .env 中设置 EMBEDDING_MODEL=./models/medical-embedding
"""
import argparse
import json
import os
import sys
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from loguru import logger

# 检查依赖
try:
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.training_args import BatchSamplers
    HAS_ST = True
except ImportError:
    HAS_ST = False
    logger.warning("sentence-transformers 未安装，将使用原生 PyTorch 训练")

try:
    from datasets import Dataset, load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    logger.error("datasets 库未安装，请运行: pip install datasets")
    sys.exit(1)


def setup_logging(verbose: bool = False):
    """配置日志"""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)


def load_training_data(train_file: str, val_file: str = None):
    """
    加载训练数据
    
    数据格式 (JSONL):
    {"query": "...", "pos": ["..."], "neg": ["...", "..."]}
    """
    def load_jsonl(filepath):
        data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    train_data = load_jsonl(train_file)
    val_data = load_jsonl(val_file) if val_file and Path(val_file).exists() else None
    
    logger.info(f"加载训练数据: {len(train_data)} 样本")
    if val_data:
        logger.info(f"加载验证数据: {len(val_data)} 样本")
    
    return train_data, val_data


def create_dataset_for_mnrl(data: list) -> Dataset:
    """
    创建适用于 MultipleNegativesRankingLoss 的数据集
    
    返回格式: {"anchor": query, "positive": pos_passage}
    """
    anchors = []
    positives = []
    
    for item in data:
        query = item["query"]
        pos_list = item.get("pos", [])
        
        for pos in pos_list:
            anchors.append(query)
            positives.append(pos)
    
    return Dataset.from_dict({
        "anchor": anchors,
        "positive": positives,
    })


def create_dataset_with_negatives(data: list) -> Dataset:
    """
    创建包含负样本的数据集 (用于 TripletLoss 或 ContrastiveLoss)
    
    返回格式: {"anchor": query, "positive": pos, "negative": neg}
    """
    anchors = []
    positives = []
    negatives = []
    
    for item in data:
        query = item["query"]
        pos_list = item.get("pos", [])
        neg_list = item.get("neg", [])
        
        if not neg_list:
            continue
            
        for pos in pos_list:
            for neg in neg_list:
                anchors.append(query)
                positives.append(pos)
                negatives.append(neg)
    
    return Dataset.from_dict({
        "anchor": anchors,
        "positive": positives,
        "negative": negatives,
    })


def train_with_sentence_transformers(
    base_model: str,
    train_file: str,
    val_file: str,
    output_dir: str,
    batch_size: int = 4,
    gradient_accumulation: int = 8,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    max_length: int = 512,
    loss_type: str = "mnrl",
    warmup_ratio: float = 0.1,
    fp16: bool = True,
    seed: int = 42,
):
    """
    使用 sentence-transformers 进行微调
    """
    if not HAS_ST:
        raise ImportError("请安装 sentence-transformers: pip install sentence-transformers>=3.0.0")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载基础模型
    logger.info(f"加载基础模型: {base_model}")
    model = SentenceTransformer(
        base_model,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.float16 if fp16 else torch.float32},
    )
    
    # 设置最大序列长度
    model.max_seq_length = max_length
    
    # 2. 加载训练数据
    train_data, val_data = load_training_data(train_file, val_file)
    
    # 3. 创建数据集
    if loss_type == "mnrl":
        train_dataset = create_dataset_for_mnrl(train_data)
        loss = losses.MultipleNegativesRankingLoss(model)
        logger.info("使用 MultipleNegativesRankingLoss")
    elif loss_type == "triplet":
        train_dataset = create_dataset_with_negatives(train_data)
        loss = losses.TripletLoss(model)
        logger.info("使用 TripletLoss")
    elif loss_type == "contrastive":
        train_dataset = create_dataset_with_negatives(train_data)
        loss = losses.ContrastiveLoss(model)
        logger.info("使用 ContrastiveLoss")
    else:
        raise ValueError(f"不支持的损失函数: {loss_type}")
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    
    # 4. 创建评估器 (可选)
    evaluator = None
    if val_data:
        # 构建 IR 评估数据
        queries = {}
        corpus = {}
        relevant_docs = {}
        
        for i, item in enumerate(val_data[:1000]):  # 限制评估集大小
            qid = f"q{i}"
            queries[qid] = item["query"]
            
            for j, pos in enumerate(item.get("pos", [])):
                did = f"d{i}_{j}"
                corpus[did] = pos
                if qid not in relevant_docs:
                    relevant_docs[qid] = set()
                relevant_docs[qid].add(did)
        
        if queries and corpus:
            evaluator = InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name="medical-eval",
                score_functions={"cosine": lambda a, b: torch.cosine_similarity(a, b, dim=-1)},
            )
            logger.info(f"创建评估器: {len(queries)} 查询, {len(corpus)} 文档")
    
    # 5. 训练参数
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=fp16,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps" if evaluator else "no",
        eval_steps=500 if evaluator else None,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        logging_steps=100,
        seed=seed,
        dataloader_num_workers=0,  # Windows 兼容
        report_to="none",  # 禁用 wandb 等
    )
    
    # 6. 创建 Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    
    # 7. 开始训练
    logger.info("开始训练...")
    trainer.train()
    
    # 8. 保存最终模型
    final_path = output_dir / "final"
    model.save(str(final_path))
    logger.info(f"模型已保存到: {final_path}")
    
    return str(final_path)


def train_native_pytorch(
    base_model: str,
    train_file: str,
    output_dir: str,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    max_length: int = 512,
    fp16: bool = True,
):
    """
    原生 PyTorch 训练 (备选方案)
    """
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    logger.info(f"加载模型: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if fp16 else torch.float32,
        trust_remote_code=True,
    ).to(device)
    
    # 加载数据
    train_data, _ = load_training_data(train_file)
    
    # 准备 DataLoader
    def collate_fn(batch):
        queries = [item["query"] for item in batch]
        positives = [item["pos"][0] if item["pos"] else "" for item in batch]
        
        query_enc = tokenizer(
            queries, padding=True, truncation=True, 
            max_length=max_length, return_tensors="pt"
        )
        pos_enc = tokenizer(
            positives, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        
        return {
            "query_input_ids": query_enc["input_ids"],
            "query_attention_mask": query_enc["attention_mask"],
            "pos_input_ids": pos_enc["input_ids"],
            "pos_attention_mask": pos_enc["attention_mask"],
        }
    
    dataloader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # InfoNCE Loss
    def info_nce_loss(query_emb, pos_emb, temperature=0.05):
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
        pos_emb = torch.nn.functional.normalize(pos_emb, p=2, dim=1)
        
        similarity = torch.matmul(query_emb, pos_emb.T) / temperature
        labels = torch.arange(query_emb.size(0), device=query_emb.device)
        
        return torch.nn.functional.cross_entropy(similarity, labels)
    
    # Mean pooling
    def mean_pooling(hidden_states, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask
    
    # 训练循环
    model.train()
    global_step = 0
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in dataloader:
            # 移动到设备
            query_ids = batch["query_input_ids"].to(device)
            query_mask = batch["query_attention_mask"].to(device)
            pos_ids = batch["pos_input_ids"].to(device)
            pos_mask = batch["pos_attention_mask"].to(device)
            
            # 前向传播
            query_out = model(input_ids=query_ids, attention_mask=query_mask)
            pos_out = model(input_ids=pos_ids, attention_mask=pos_mask)
            
            query_emb = mean_pooling(query_out.last_hidden_state, query_mask)
            pos_emb = mean_pooling(pos_out.last_hidden_state, pos_mask)
            
            # 计算损失
            loss = info_nce_loss(query_emb, pos_emb)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            global_step += 1
            
            if global_step % 100 == 0:
                avg_loss = total_loss / (global_step % len(dataloader) or len(dataloader))
                logger.info(f"Epoch {epoch+1}, Step {global_step}, Loss: {avg_loss:.4f}")
        
        # 保存 checkpoint
        checkpoint_dir = output_dir / f"checkpoint-{epoch+1}"
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        logger.info(f"保存 checkpoint: {checkpoint_dir}")
    
    # 保存最终模型
    final_path = output_dir / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"模型已保存到: {final_path}")
    
    return str(final_path)


def main():
    parser = argparse.ArgumentParser(description="医学 Embedding 微调")
    
    # 模型参数
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-Embedding-8B",
        help="基础模型名称或路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/medical-embedding",
        help="输出目录",
    )
    
    # 数据参数
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/training/train.jsonl",
        help="训练数据文件",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="data/training/val.jsonl",
        help="验证数据文件",
    )
    
    # 训练参数
    parser.add_argument("--batch-size", type=int, default=4, help="批大小")
    parser.add_argument("--grad-accum", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--max-length", type=int, default=512, help="最大序列长度")
    parser.add_argument(
        "--loss", 
        type=str, 
        default="mnrl",
        choices=["mnrl", "triplet", "contrastive"],
        help="损失函数类型",
    )
    
    # 其他
    parser.add_argument("--fp16", action="store_true", default=True, help="使用 FP16")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16", help="禁用 FP16")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--native", action="store_true", help="使用原生 PyTorch 训练")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # 检查训练数据
    train_path = Path(args.train_file)
    if not train_path.exists():
        logger.error(f"训练数据不存在: {train_path}")
        logger.info("请先运行数据构建:")
        logger.info('  python -c "from src.training.data_builder import build_medical_training_data; build_medical_training_data()"')
        sys.exit(1)
    
    # 选择训练方式
    if args.native or not HAS_ST:
        logger.info("使用原生 PyTorch 训练")
        model_path = train_native_pytorch(
            base_model=args.base_model,
            train_file=args.train_file,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            max_length=args.max_length,
            fp16=args.fp16,
        )
    else:
        logger.info("使用 sentence-transformers 训练")
        model_path = train_with_sentence_transformers(
            base_model=args.base_model,
            train_file=args.train_file,
            val_file=args.val_file,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            gradient_accumulation=args.grad_accum,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            max_length=args.max_length,
            loss_type=args.loss,
            fp16=args.fp16,
            seed=args.seed,
        )
    
    print("\n" + "=" * 50)
    print("训练完成!")
    print("=" * 50)
    print(f"模型路径: {model_path}")
    print("\n使用微调模型:")
    print(f'  在 .env 中设置: EMBEDDING_MODEL={model_path}')
    print("  或在代码中指定:")
    print(f'    embedder = MedicalEmbedder(model_name="{model_path}")')


if __name__ == "__main__":
    main()
