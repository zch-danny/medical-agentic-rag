"""
嵌入模型 - 基于 Qwen3-Embedding
"""
from typing import List, Optional

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class MedicalEmbedder:
    """
    医疗文献嵌入模型

    使用 Qwen3-Embedding 生成文档和查询的嵌入向量
    支持自定义医疗领域指令
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        instruction: Optional[str] = None,
        max_length: int = 8192,
    ):
        """
        Args:
            model_name: HuggingFace 模型名称
            instruction: 查询时的任务指令
            max_length: 最大序列长度
        """
        self.model_name = model_name
        self.instruction = instruction or (
            "Given a medical or clinical question, retrieve relevant passages from "
            "medical literature that answer the query"
        )
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"加载嵌入模型: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # 尝试使用 Flash Attention，失败则回退
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
            logger.info("使用 Flash Attention 2")
        except Exception:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("使用默认 Attention (sdpa)")

        self.model.eval()
        logger.info(f"嵌入模型已加载，维度: {self.model.config.hidden_size}")

    def _mean_pooling(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """对 token 嵌入做 mean pooling"""
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def encode_query(self, query: str) -> List[float]:
        """
        编码查询文本（带指令）

        Args:
            query: 查询文本

        Returns:
            嵌入向量
        """
        # 格式化带指令的查询
        formatted = f"Instruct: {self.instruction}\nQuery: {query}"

        inputs = self.tokenizer(
            [formatted],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings[0].cpu().float().tolist()

    def encode_documents(
        self,
        documents: List[str],
        batch_size: int = 4,
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        批量编码文档（不带指令）

        Args:
            documents: 文档列表
            batch_size: 批大小
            show_progress: 是否显示进度条

        Returns:
            嵌入向量列表
        """
        all_embeddings = []

        iterator = range(0, len(documents), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="生成嵌入", total=(len(documents) + batch_size - 1) // batch_size)

        for i in iterator:
            batch = documents[i : i + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self._mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.extend(embeddings.cpu().float().tolist())

        return all_embeddings

    @property
    def embedding_dim(self) -> int:
        """返回嵌入维度"""
        return self.model.config.hidden_size
