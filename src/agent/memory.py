"""
对话记忆管理

支持：
- 短期记忆：当前会话历史
- 长期记忆：用户偏好（可选）
- 摘要机制：长对话自动摘要
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

from loguru import logger


@dataclass
class Message:
    """单条消息"""
    role: str           # "user" 或 "assistant"
    content: str        # 消息内容
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    """记忆配置"""
    max_history_length: int = 20        # 最大历史消息数
    summarize_threshold: int = 15       # 超过此数量触发摘要
    enable_persistence: bool = False    # 是否持久化
    storage_path: Optional[Path] = None # 持久化路径


class ConversationMemory:
    """
    对话记忆管理器
    
    管理对话历史，支持摘要和持久化
    """
    
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        session_id: Optional[str] = None,
        llm=None,
    ):
        """
        Args:
            config: 记忆配置
            session_id: 会话ID
            llm: LLM 实例，用于生成摘要（可选）
        """
        self.config = config or MemoryConfig()
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._llm = llm
        
        # 消息历史
        self._messages: List[Message] = []
        
        # 摘要（当历史过长时生成）
        self._summary: Optional[str] = None
        
        # 上下文变量（用于追问等场景）
        self._context: Dict[str, Any] = {}
        
        # 加载持久化数据
        if self.config.enable_persistence:
            self._load()
    
    def add_user_message(self, content: str, metadata: Optional[Dict] = None) -> None:
        """添加用户消息"""
        msg = Message(
            role="user",
            content=content,
            metadata=metadata or {},
        )
        self._messages.append(msg)
        self._check_and_summarize()
        self._save_if_needed()
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict] = None) -> None:
        """添加助手消息"""
        msg = Message(
            role="assistant",
            content=content,
            metadata=metadata or {},
        )
        self._messages.append(msg)
        self._check_and_summarize()
        self._save_if_needed()
    
    def get_history(self, max_messages: Optional[int] = None) -> List[Dict[str, str]]:
        """
        获取对话历史
        
        Args:
            max_messages: 最大返回消息数
            
        Returns:
            消息列表，格式 [{"role": "user", "content": "..."}]
        """
        messages = self._messages
        if max_messages:
            messages = messages[-max_messages:]
        
        return [{"role": m.role, "content": m.content} for m in messages]
    
    def get_history_text(self, max_messages: Optional[int] = None) -> str:
        """
        获取对话历史的文本格式
        
        Args:
            max_messages: 最大返回消息数
            
        Returns:
            格式化的对话历史文本
        """
        history = self.get_history(max_messages)
        lines = []
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)
    
    def get_last_user_query(self) -> Optional[str]:
        """获取最后一个用户查询"""
        for msg in reversed(self._messages):
            if msg.role == "user":
                return msg.content
        return None
    
    def get_previous_query(self) -> Optional[str]:
        """获取倒数第二个用户查询（用于追问场景）"""
        user_queries = [m.content for m in self._messages if m.role == "user"]
        if len(user_queries) >= 2:
            return user_queries[-2]
        return None
    
    def get_summary(self) -> Optional[str]:
        """获取对话摘要"""
        return self._summary
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """获取上下文变量"""
        return self._context.get(key, default)
    
    def set_context(self, key: str, value: Any) -> None:
        """设置上下文变量"""
        self._context[key] = value
    
    def clear_context(self) -> None:
        """清除上下文变量"""
        self._context.clear()
    
    def clear_history(self) -> None:
        """清除对话历史"""
        self._messages.clear()
        self._summary = None
        self._context.clear()
        self._save_if_needed()
    
    def _check_and_summarize(self) -> None:
        """检查是否需要摘要"""
        if len(self._messages) > self.config.summarize_threshold:
            self._generate_summary()
    
    def _generate_summary(self) -> None:
        """生成对话摘要"""
        if self._llm is None:
            # 无 LLM 时使用简单摘要
            self._simple_summarize()
            return
        
        try:
            history_text = self.get_history_text()
            prompt = f"""请简洁地总结以下医疗对话的要点（不超过100字）：

{history_text}

总结："""
            
            response = self._llm.complete(prompt)
            self._summary = response.text.strip()
            
            # 保留最近的消息，删除较早的
            keep_count = self.config.max_history_length // 2
            self._messages = self._messages[-keep_count:]
            
            logger.debug(f"生成对话摘要: {self._summary[:50]}...")
            
        except Exception as e:
            logger.error(f"生成摘要失败: {e}")
            self._simple_summarize()
    
    def _simple_summarize(self) -> None:
        """简单摘要（无 LLM 时使用）"""
        # 提取关键信息
        topics = set()
        for msg in self._messages:
            if msg.role == "user":
                # 提取可能的主题词
                for keyword in ["糖尿病", "高血压", "治疗", "症状", "预防", "诊断"]:
                    if keyword in msg.content:
                        topics.add(keyword)
        
        if topics:
            self._summary = f"讨论主题: {', '.join(topics)}"
        else:
            self._summary = f"对话包含 {len(self._messages)} 条消息"
        
        # 保留最近的消息
        keep_count = self.config.max_history_length // 2
        self._messages = self._messages[-keep_count:]
    
    def _save_if_needed(self) -> None:
        """如果启用持久化则保存"""
        if self.config.enable_persistence:
            self._save()
    
    def _get_storage_file(self) -> Path:
        """获取存储文件路径"""
        if self.config.storage_path:
            base_path = self.config.storage_path
        else:
            base_path = Path("data/sessions")
        
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path / f"{self.session_id}.json"
    
    def _save(self) -> None:
        """保存到文件"""
        try:
            data = {
                "session_id": self.session_id,
                "messages": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "timestamp": m.timestamp,
                        "metadata": m.metadata,
                    }
                    for m in self._messages
                ],
                "summary": self._summary,
                "context": self._context,
            }
            
            file_path = self._get_storage_file()
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"保存会话: {file_path}")
            
        except Exception as e:
            logger.error(f"保存会话失败: {e}")
    
    def _load(self) -> None:
        """从文件加载"""
        try:
            file_path = self._get_storage_file()
            if not file_path.exists():
                return
            
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self._messages = [
                Message(
                    role=m["role"],
                    content=m["content"],
                    timestamp=m.get("timestamp", ""),
                    metadata=m.get("metadata", {}),
                )
                for m in data.get("messages", [])
            ]
            self._summary = data.get("summary")
            self._context = data.get("context", {})
            
            logger.debug(f"加载会话: {file_path}, {len(self._messages)} 条消息")
            
        except Exception as e:
            logger.error(f"加载会话失败: {e}")
    
    def __len__(self) -> int:
        return len(self._messages)
    
    def __bool__(self) -> bool:
        return len(self._messages) > 0
