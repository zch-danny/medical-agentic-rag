"""
重排序模型 - Qwen3-Reranker
"""
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen3Reranker:
    """
    基于 Qwen3-Reranker 的重排序模型

    支持批处理以避免大量候选时显存溢出
    """

    # 系统提示词
    SYSTEM_PROMPT = 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'

    def __init__(
        self,
        model_name: str,
        instruction: Optional[str] = None,
        max_length: int = 8192,
        batch_size: int = 8,
    ):
        """
        Args:
            model_name: 模型名称，如 Qwen/Qwen3-Reranker-8B
            instruction: 自定义任务指令
            max_length: 最大序列长度
            batch_size: 批处理大小，防止显存溢出
        """
        self.instruction = instruction or "Given a web search query, retrieve relevant passages that answer the query"
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"加载重排序模型: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

        # 获取 yes/no token ID
        yes_ids = self.tokenizer.encode(" yes", add_special_tokens=False)
        no_ids = self.tokenizer.encode(" no", add_special_tokens=False)
        if len(yes_ids) != 1 or len(no_ids) != 1:
            logger.warning(f"yes/no 不是单 token：yes={yes_ids}, no={no_ids}，将使用最后一个 token")
        self.token_true_id = yes_ids[-1]
        self.token_false_id = no_ids[-1]

        # 构建 prefix 和 suffix tokens
        self.prefix = f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        logger.info(f"重排序模型已加载，显存占用: {self._get_gpu_memory():.1f}GB")

    def _get_gpu_memory(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0

    def _format_input(self, query: str, document: str) -> str:
        """格式化输入"""
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"

    def _compute_score_batch(self, formatted_inputs: List[str]) -> List[float]:
        """计算一批输入的重排序分数"""
        # Tokenize
        inputs = self.tokenizer(
            formatted_inputs,
            padding=False,
            truncation=True,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
            return_attention_mask=False
        )

        # 添加 prefix 和 suffix tokens
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ids + self.suffix_tokens

        # Padding
        inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if torch.cuda.is_available()
            else nullcontext()
        )

        with torch.no_grad(), autocast_ctx:
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]

            true_logits = logits[:, self.token_true_id]
            false_logits = logits[:, self.token_false_id]

            scores = torch.softmax(
                torch.stack([false_logits, true_logits], dim=1),
                dim=1
            )[:, 1]

        return scores.cpu().float().tolist()

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
        show_progress: bool = False,
    ) -> List[Dict]:
        """
        对候选结果进行重排序

        Args:
            query: 查询文本
            candidates: 候选文档列表
            top_k: 返回数量
            show_progress: 是否显示进度条

        Returns:
            重排序后的 top_k 结果
        """
        if not candidates:
            return []

        # 提取文本并格式化输入
        formatted_inputs = []
        for c in candidates:
            # 优先使用原始文本（未分词）
            text = (
                c.get("entity", {}).get("original_text")
                or c.get("entity", {}).get("text", "")
                or c.get("original_text")
                or c.get("text", "")
            )
            formatted_inputs.append(self._format_input(query, text))

        # 分批处理，避免显存溢出
        all_scores = []
        num_batches = (len(formatted_inputs) + self.batch_size - 1) // self.batch_size

        iterator = range(0, len(formatted_inputs), self.batch_size)
        if show_progress and num_batches > 1:
            iterator = tqdm(iterator, desc="重排序", total=num_batches)

        for i in iterator:
            batch = formatted_inputs[i : i + self.batch_size]
            batch_scores = self._compute_score_batch(batch)
            all_scores.extend(batch_scores)

            # 每批次后清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 排序并返回 top_k
        scored = list(zip(candidates, all_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for item, score in scored[:top_k]:
            item["rerank_score"] = score
            results.append(item)

        return results


# 别名，保持后向兼容
Reranker = Qwen3Reranker
