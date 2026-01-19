"""
Paper2Figure 核心模块

使用 LLM 分析论文内容，生成图表的结构化描述和代码
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class FigureType(str, Enum):
    """图表类型"""
    ARCHITECTURE = "architecture"      # 模型架构图
    ROADMAP = "roadmap"               # 技术路线图
    FLOWCHART = "flowchart"           # 流程图
    EXPERIMENT = "experiment"         # 实验数据图
    COMPARISON = "comparison"         # 对比图
    AUTO = "auto"                     # 自动检测


@dataclass
class FigureResult:
    """图表生成结果"""
    figure_type: FigureType
    title: str
    description: str
    mermaid_code: str                 # Mermaid 格式代码
    svg_code: Optional[str] = None    # SVG 代码（如果已渲染）
    pptx_path: Optional[str] = None   # PPTX 文件路径
    metadata: Dict = field(default_factory=dict)


class Paper2Figure:
    """
    论文图表生成器

    将论文内容转换为可编辑的架构图、流程图等。
    优先使用在线免费服务，回退到本地 DeepSeek API。
    """

    # 架构图生成 Prompt
    ARCHITECTURE_PROMPT = """你是一个专业的科研图表设计师。请分析以下论文内容，生成一个清晰的模型架构图。

要求：
1. 使用 Mermaid flowchart 语法
2. 识别论文中的主要模块/组件
3. 用方框表示模块，箭头表示数据流或依赖关系
4. 添加中文标注说明每个模块的功能
5. 保持图表简洁，突出核心架构

论文内容：
{content}

请直接输出 Mermaid 代码，格式如下：
```mermaid
flowchart TD
    A[模块A] --> B[模块B]
    ...
```

只输出代码，不要其他解释。"""

    # 技术路线图 Prompt
    ROADMAP_PROMPT = """你是一个专业的科研图表设计师。请分析以下论文内容，生成一个技术路线图。

要求：
1. 使用 Mermaid flowchart 语法（从左到右 LR 或从上到下 TD）
2. 按时间或逻辑顺序展示研究步骤
3. 每个节点包含简短描述
4. 可以使用子图(subgraph)分组相关步骤

论文内容：
{content}

请直接输出 Mermaid 代码，格式如下：
```mermaid
flowchart LR
    subgraph 阶段1
        A[步骤1] --> B[步骤2]
    end
    ...
```

只输出代码，不要其他解释。"""

    # 流程图 Prompt
    FLOWCHART_PROMPT = """你是一个专业的科研图表设计师。请分析以下论文内容，生成一个方法流程图。

要求：
1. 使用 Mermaid flowchart 语法
2. 展示论文方法的完整流程
3. 包含输入、处理步骤、输出
4. 使用菱形表示判断/条件，圆角矩形表示开始/结束

论文内容：
{content}

请直接输出 Mermaid 代码，格式如下：
```mermaid
flowchart TD
    Start([开始]) --> A[步骤1]
    A --> B{{判断}}
    B -->|是| C[步骤2]
    B -->|否| D[步骤3]
    C --> End([结束])
```

只输出代码，不要其他解释。"""

    # 实验对比图 Prompt
    EXPERIMENT_PROMPT = """你是一个专业的数据可视化专家。请分析以下论文内容，提取实验数据并生成图表。

要求：
1. 识别论文中的实验结果数据（准确率、F1分数等）
2. 使用 Mermaid xychart 或 pie 语法
3. 如果是多组对比数据，使用柱状图
4. 如果是比例数据，使用饼图

论文内容：
{content}

请直接输出 Mermaid 代码。如果是柱状图：
```mermaid
xychart-beta
    title "实验结果对比"
    x-axis [方法A, 方法B, 方法C]
    y-axis "准确率 (%)" 0 --> 100
    bar [85, 90, 88]
```

如果是饼图：
```mermaid
pie title 数据分布
    "类别A" : 30
    "类别B" : 45
    "类别C" : 25
```

只输出代码，不要其他解释。"""

    # 自动检测并生成 Prompt
    AUTO_PROMPT = """你是一个专业的科研图表设计师。请分析以下论文内容，判断最适合的图表类型并生成。

图表类型选择：
- 如果论文描述了模型/系统架构 -> 生成架构图 (flowchart TD)
- 如果论文描述了研究步骤/方法流程 -> 生成流程图 (flowchart TD/LR)
- 如果论文包含实验数据对比 -> 生成数据图 (xychart/pie)
- 如果论文描述了技术演进/发展路线 -> 生成路线图 (flowchart LR)

论文内容：
{content}

请先在注释中说明选择的图表类型，然后输出 Mermaid 代码：
```mermaid
%% 图表类型: [architecture/flowchart/experiment/roadmap]
%% 标题: [图表标题]
...你的代码...
```

只输出代码块，不要其他解释。"""

    PROMPTS = {
        FigureType.ARCHITECTURE: ARCHITECTURE_PROMPT,
        FigureType.ROADMAP: ROADMAP_PROMPT,
        FigureType.FLOWCHART: FLOWCHART_PROMPT,
        FigureType.EXPERIMENT: EXPERIMENT_PROMPT,
        FigureType.AUTO: AUTO_PROMPT,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        use_online_first: bool = True,
    ):
        """
        初始化 Paper2Figure

        Args:
            api_key: LLM API Key（默认从环境变量读取）
            base_url: LLM API Base URL
            model: 模型名称
            use_online_first: 是否优先使用在线免费服务
        """
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.getenv("LLM_MODEL", "deepseek-chat")
        self.use_online_first = use_online_first

        # Paper2Any 在线服务（公测期间免费）
        self.online_api_url = "http://dcai-paper2any.nas.cpolar.cn"

        if not self.api_key and not use_online_first:
            raise ValueError("LLM_API_KEY 未设置，且未启用在线服务")

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            logger.info(f"Paper2Figure 初始化完成: {self.base_url}, model={self.model}")
        else:
            self.client = None
            logger.info("Paper2Figure 初始化完成: 仅使用在线服务")

    def _extract_mermaid_code(self, response: str) -> tuple[str, str, str]:
        """从 LLM 响应中提取 Mermaid 代码和元信息"""
        import re

        # 提取 mermaid 代码块
        pattern = r'```mermaid\s*([\s\S]*?)```'
        match = re.search(pattern, response)

        if not match:
            # 尝试直接匹配 flowchart/pie/xychart
            if any(kw in response for kw in ['flowchart', 'pie', 'xychart', 'graph']):
                code = response.strip()
            else:
                raise ValueError("无法从响应中提取 Mermaid 代码")
        else:
            code = match.group(1).strip()

        # 提取图表类型和标题（从注释中）
        figure_type = "flowchart"
        title = "论文图表"

        type_match = re.search(r'%%\s*图表类型:\s*(\w+)', code)
        if type_match:
            figure_type = type_match.group(1)

        title_match = re.search(r'%%\s*标题:\s*(.+)', code)
        if title_match:
            title = title_match.group(1).strip()
        else:
            # 从 title 属性提取
            title_attr_match = re.search(r'title\s+["\'](.+?)["\']', code)
            if title_attr_match:
                title = title_attr_match.group(1)

        return code, figure_type, title

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_llm(self, prompt: str) -> str:
        """调用 LLM 生成图表代码"""
        if not self.client:
            raise ValueError("LLM 客户端未初始化")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个专业的科研图表设计师，擅长使用 Mermaid 语法生成清晰的图表。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    def _try_online_service(self, content: str, figure_type: FigureType) -> Optional[FigureResult]:
        """尝试使用在线 Paper2Any 服务（公测期间免费）"""
        # 注意：Paper2Any 在线版目前只提供 Web 界面，没有公开 API
        # 这里预留接口，后续如果开放 API 可以接入
        logger.debug("在线服务暂不可用，回退到本地 LLM")
        return None

    def generate(
        self,
        content: str,
        figure_type: FigureType = FigureType.AUTO,
        title: Optional[str] = None,
    ) -> FigureResult:
        """
        从论文内容生成图表

        Args:
            content: 论文文本内容
            figure_type: 图表类型（默认自动检测）
            title: 图表标题（可选）

        Returns:
            FigureResult: 包含 Mermaid 代码的结果
        """
        # 1. 尝试在线服务（免费额度）
        if self.use_online_first:
            result = self._try_online_service(content, figure_type)
            if result:
                return result

        # 2. 使用本地 LLM
        if not self.client:
            raise ValueError("在线服务不可用，且本地 LLM 未配置")

        # 截断过长的内容
        max_content_length = 8000
        if len(content) > max_content_length:
            logger.warning(f"内容过长 ({len(content)} 字符)，截断到 {max_content_length}")
            content = content[:max_content_length] + "\n...[内容已截断]..."

        # 获取对应的 Prompt
        prompt_template = self.PROMPTS.get(figure_type, self.PROMPTS[FigureType.AUTO])
        prompt = prompt_template.format(content=content)

        # 调用 LLM
        logger.info(f"调用 LLM 生成 {figure_type.value} 图表...")
        response = self._call_llm(prompt)

        # 解析响应
        mermaid_code, detected_type, detected_title = self._extract_mermaid_code(response)

        return FigureResult(
            figure_type=FigureType(detected_type) if detected_type in [e.value for e in FigureType] else figure_type,
            title=title or detected_title,
            description=f"从论文内容自动生成的{figure_type.value}图表",
            mermaid_code=mermaid_code,
            metadata={
                "model": self.model,
                "content_length": len(content),
            }
        )

    def generate_from_pdf(
        self,
        pdf_path: Union[str, Path],
        figure_type: FigureType = FigureType.AUTO,
        title: Optional[str] = None,
    ) -> FigureResult:
        """
        从 PDF 文件生成图表

        Args:
            pdf_path: PDF 文件路径
            figure_type: 图表类型
            title: 图表标题

        Returns:
            FigureResult
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        # 使用项目现有的文档加载器
        from src.document_loader import MinerUDocumentLoader

        loader = MinerUDocumentLoader()

        # 加载并合并文本
        chunks = loader.load(str(pdf_path), chunk_size=10000, chunk_overlap=0)
        if not chunks:
            raise ValueError(f"无法从 PDF 提取文本: {pdf_path}")

        # 合并所有 chunk
        content = "\n\n".join(chunk["text"] for chunk in chunks)

        # 使用文件名作为默认标题
        if not title:
            title = pdf_path.stem

        return self.generate(content, figure_type, title)

    def generate_multiple(
        self,
        content: str,
        figure_types: Optional[List[FigureType]] = None,
    ) -> List[FigureResult]:
        """
        生成多种类型的图表

        Args:
            content: 论文内容
            figure_types: 要生成的图表类型列表（默认生成架构图和流程图）

        Returns:
            List[FigureResult]
        """
        if figure_types is None:
            figure_types = [FigureType.ARCHITECTURE, FigureType.FLOWCHART]

        results = []
        for ft in figure_types:
            try:
                result = self.generate(content, ft)
                results.append(result)
            except Exception as e:
                logger.error(f"生成 {ft.value} 图表失败: {e}")

        return results
