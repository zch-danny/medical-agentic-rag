"""
Skill 加载器

支持将 Agent Skills 注入任意 LLM（DeepSeek、GPT、Claude 等）
"""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional


class SkillLoader:
    """
    Skill 加载器
    
    支持渐进式加载（Progressive Disclosure）：
    1. 启动时只加载 name + description（轻量）
    2. 需要时再加载完整 SKILL.md 内容
    """
    
    def __init__(self, skills_dir: str | Path):
        self.skills_dir = Path(skills_dir)
        self.skill_index: Dict[str, dict] = {}
        self._build_index()
    
    def _build_index(self):
        """构建 Skill 索引（只加载元数据）"""
        for skill_path in self.skills_dir.iterdir():
            if not skill_path.is_dir():
                continue
            
            skill_md = skill_path / "SKILL.md"
            if not skill_md.exists():
                continue
            
            content = skill_md.read_text(encoding="utf-8")
            meta = self._parse_frontmatter(content)
            
            if meta and "name" in meta:
                self.skill_index[meta["name"]] = {
                    "path": skill_path,
                    "description": meta.get("description", ""),
                    "meta": meta
                }
    
    def _parse_frontmatter(self, content: str) -> Optional[dict]:
        """解析 YAML frontmatter"""
        match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if match:
            try:
                return yaml.safe_load(match.group(1))
            except yaml.YAMLError:
                return None
        return None
    
    def _get_body(self, content: str) -> str:
        """获取 frontmatter 之后的正文"""
        match = re.match(r"^---\s*\n.*?\n---\s*\n", content, re.DOTALL)
        if match:
            return content[match.end():]
        return content
    
    def list_skills(self) -> List[dict]:
        """列出所有可用 Skill"""
        return [
            {"name": name, "description": info["description"]}
            for name, info in self.skill_index.items()
        ]
    
    def get_index_prompt(self) -> str:
        """
        生成 Skill 索引提示词（放入 System Prompt）
        
        轻量级，只包含名称和描述
        """
        lines = ["## 可用 Skills\n"]
        lines.append("以下 Skills 可按需加载使用：\n")
        
        for name, info in self.skill_index.items():
            lines.append(f"- **{name}**: {info['description']}")
        
        lines.append("\n如需使用某个 Skill，请先阅读其完整说明。")
        return "\n".join(lines)
    
    def load_skill(self, name: str) -> str:
        """
        加载完整 Skill 内容
        
        Args:
            name: Skill 名称
            
        Returns:
            完整的 SKILL.md 内容（不含 frontmatter）
        """
        if name not in self.skill_index:
            raise ValueError(f"Skill '{name}' 不存在")
        
        skill_path = self.skill_index[name]["path"]
        skill_md = skill_path / "SKILL.md"
        content = skill_md.read_text(encoding="utf-8")
        
        return self._get_body(content)
    
    def load_resource(self, skill_name: str, resource_name: str) -> str:
        """
        加载 Skill 附属资源文件
        
        Args:
            skill_name: Skill 名称
            resource_name: 资源文件名（如 MEDICAL_TERMS.md）
        """
        if skill_name not in self.skill_index:
            raise ValueError(f"Skill '{skill_name}' 不存在")
        
        skill_path = self.skill_index[skill_name]["path"]
        resource_path = skill_path / resource_name
        
        if not resource_path.exists():
            raise FileNotFoundError(f"资源 '{resource_name}' 不存在")
        
        return resource_path.read_text(encoding="utf-8")
    
    def get_skill_path(self, name: str) -> Path:
        """获取 Skill 目录路径"""
        if name not in self.skill_index:
            raise ValueError(f"Skill '{name}' 不存在")
        return self.skill_index[name]["path"]


class LLMWithSkills:
    """
    带 Skill 支持的 LLM 封装
    
    支持 DeepSeek、GPT、Claude 等 OpenAI 兼容 API
    """
    
    def __init__(
        self,
        client,  # OpenAI 兼容的 client
        model: str,
        skills_dir: str | Path,
        system_prompt: str = ""
    ):
        self.client = client
        self.model = model
        self.skill_loader = SkillLoader(skills_dir)
        self.base_system_prompt = system_prompt
        self.loaded_skills: List[str] = []
    
    def _build_system_prompt(self) -> str:
        """构建包含 Skill 信息的 System Prompt"""
        parts = []
        
        if self.base_system_prompt:
            parts.append(self.base_system_prompt)
        
        # 添加 Skill 索引
        parts.append(self.skill_loader.get_index_prompt())
        
        # 添加已加载的完整 Skill 内容
        for skill_name in self.loaded_skills:
            skill_content = self.skill_loader.load_skill(skill_name)
            parts.append(f"\n## Skill: {skill_name}\n\n{skill_content}")
        
        return "\n\n".join(parts)
    
    def load_skill(self, name: str):
        """加载 Skill 到上下文"""
        if name not in self.loaded_skills:
            self.loaded_skills.append(name)
    
    def unload_skill(self, name: str):
        """卸载 Skill"""
        if name in self.loaded_skills:
            self.loaded_skills.remove(name)
    
    def chat(self, messages: List[dict], **kwargs) -> str:
        """
        发送对话请求
        
        Args:
            messages: 对话消息列表
            **kwargs: 传递给 API 的其他参数
        """
        system_prompt = self._build_system_prompt()
        
        full_messages = [
            {"role": "system", "content": system_prompt},
            *messages
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            **kwargs
        )
        
        return response.choices[0].message.content


# 使用示例
if __name__ == "__main__":
    # 示例：使用 DeepSeek
    from openai import OpenAI
    
    # 初始化 DeepSeek client
    client = OpenAI(
        api_key="your-deepseek-api-key",
        base_url="https://api.deepseek.com"
    )
    
    # 创建带 Skill 的 LLM
    llm = LLMWithSkills(
        client=client,
        model="deepseek-chat",
        skills_dir="./skills",
        system_prompt="你是一个专业的医疗问答助手。"
    )
    
    # 加载医疗检索 Skill
    llm.load_skill("medical-literature-rag")
    
    # 对话
    response = llm.chat([
        {"role": "user", "content": "糖尿病的一线治疗药物是什么？"}
    ])
    
    print(response)
