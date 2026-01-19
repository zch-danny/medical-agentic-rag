"""
Paper2Any - 论文多模态输出模块

功能：
- Paper2Figure: 生成架构图、流程图、技术路线图
- Paper2PPT: 从论文生成完整 PPT 演示文稿
- PPTPolish: PPT 美化和优化

输出格式：SVG、Mermaid、PPTX、HTML
"""

from .core import Paper2Figure, FigureType, FigureResult
from .renderer import FigureRenderer
from .paper2ppt import Paper2PPT, PPTStyle, PPTContent, SlideContent
from .ppt_polish import PPTPolish, PolishMode, PolishResult

__all__ = [
    # Paper2Figure
    "Paper2Figure",
    "FigureType",
    "FigureResult",
    "FigureRenderer",
    # Paper2PPT
    "Paper2PPT",
    "PPTStyle",
    "PPTContent",
    "SlideContent",
    # PPTPolish
    "PPTPolish",
    "PolishMode",
    "PolishResult",
]
