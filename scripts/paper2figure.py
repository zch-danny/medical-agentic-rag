#!/usr/bin/env python3
"""
Paper2Figure 命令行工具

从论文 PDF 或文本生成架构图、流程图等科研图表

用法:
    python scripts/paper2figure.py --pdf paper.pdf --type architecture
    python scripts/paper2figure.py --text "论文内容..." --type flowchart
    python scripts/paper2figure.py --pdf paper.pdf --output ./output --formats html,pptx,svg
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="Paper2Figure - 从论文生成科研图表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从 PDF 生成架构图
  python scripts/paper2figure.py --pdf paper.pdf --type architecture

  # 从文本生成流程图
  python scripts/paper2figure.py --text "本文提出了一种..." --type flowchart

  # 生成多种格式输出
  python scripts/paper2figure.py --pdf paper.pdf --formats html,pptx,svg

  # 自动检测最佳图表类型
  python scripts/paper2figure.py --pdf paper.pdf --type auto

图表类型:
  auto         - 自动检测（默认）
  architecture - 模型架构图
  roadmap      - 技术路线图
  flowchart    - 方法流程图
  experiment   - 实验数据图
        """,
    )

    # 输入选项（二选一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--pdf",
        type=str,
        help="PDF 文件路径",
    )
    input_group.add_argument(
        "--text",
        type=str,
        help="论文文本内容",
    )
    input_group.add_argument(
        "--file",
        type=str,
        help="文本文件路径（.txt/.md）",
    )

    # 图表选项
    parser.add_argument(
        "--type", "-t",
        type=str,
        default="auto",
        choices=["auto", "architecture", "roadmap", "flowchart", "experiment"],
        help="图表类型（默认: auto）",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="图表标题（默认从文件名/内容推断）",
    )

    # 输出选项
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="输出目录（默认: ./output）",
    )
    parser.add_argument(
        "--formats", "-f",
        type=str,
        default="html,pptx",
        help="输出格式，逗号分隔（默认: html,pptx）",
    )

    # 其他选项
    parser.add_argument(
        "--preview",
        action="store_true",
        help="生成后在浏览器中预览 HTML",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细日志",
    )

    args = parser.parse_args()

    # 配置日志
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO", format="{message}")

    # 导入模块
    from src.paper2figure import Paper2Figure, FigureType, FigureRenderer

    # 获取输入内容
    content = None
    title = args.title

    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            logger.error(f"❌ PDF 文件不存在: {pdf_path}")
            sys.exit(1)

        logger.info(f"📄 解析 PDF: {pdf_path}")

        # 使用文档加载器
        try:
            from src.document_loader import MinerUDocumentLoader
            loader = MinerUDocumentLoader()
            chunks = loader.load(str(pdf_path), chunk_size=10000, chunk_overlap=0)
            if chunks:
                content = "\n\n".join(c["text"] for c in chunks)
                if not title:
                    title = pdf_path.stem
            else:
                logger.error("❌ PDF 解析失败，未提取到文本")
                sys.exit(1)
        except Exception as e:
            logger.error(f"❌ PDF 解析失败: {e}")
            sys.exit(1)

    elif args.text:
        content = args.text
        if not title:
            title = "论文图表"

    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"❌ 文件不存在: {file_path}")
            sys.exit(1)

        content = file_path.read_text(encoding="utf-8")
        if not title:
            title = file_path.stem

    if not content or len(content) < 50:
        logger.error("❌ 输入内容太短，请提供更多论文内容")
        sys.exit(1)

    logger.info(f"📊 内容长度: {len(content)} 字符")

    # 初始化 Paper2Figure
    try:
        p2f = Paper2Figure()
    except ValueError as e:
        logger.error(f"❌ 初始化失败: {e}")
        logger.info("💡 请确保已配置 LLM_API_KEY 环境变量")
        sys.exit(1)

    # 解析图表类型
    figure_type = FigureType(args.type)

    # 生成图表
    logger.info(f"🔄 正在生成 {figure_type.value} 图表...")

    try:
        result = p2f.generate(content, figure_type, title)
        logger.info(f"✅ 图表生成成功: {result.title}")
    except Exception as e:
        logger.error(f"❌ 图表生成失败: {e}")
        sys.exit(1)

    # 打印 Mermaid 代码
    print("\n" + "=" * 50)
    print("📊 Mermaid 代码:")
    print("=" * 50)
    print(result.mermaid_code)
    print("=" * 50 + "\n")

    # 渲染输出
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    formats = [f.strip() for f in args.formats.split(",")]

    renderer = FigureRenderer(output_dir=output_dir)
    outputs = renderer.render_all(result, formats=formats)

    logger.info("📁 输出文件:")
    for fmt, path in outputs.items():
        logger.info(f"   {fmt.upper()}: {path}")

    # 预览
    if args.preview and "html" in outputs:
        import webbrowser
        webbrowser.open(f"file://{Path(outputs['html']).absolute()}")
        logger.info("🌐 已在浏览器中打开预览")

    logger.info("✨ 完成!")


if __name__ == "__main__":
    main()
