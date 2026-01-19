#!/usr/bin/env python3
"""
Paper2PPT 命令行工具

从论文 PDF 或文本生成完整 PPT 演示文稿

用法:
    python scripts/paper2ppt.py --pdf paper.pdf --style academic
    python scripts/paper2ppt.py --text "论文内容..." --output ./output/presentation.pptx
    python scripts/paper2ppt.py --polish input.pptx --color modern_green
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="Paper2PPT - 从论文生成 PPT 演示文稿",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从 PDF 生成 PPT（学术风格）
  python scripts/paper2ppt.py --pdf paper.pdf --style academic

  # 从文本生成 PPT（商务风格）
  python scripts/paper2ppt.py --text "论文内容..." --style business

  # 美化已有 PPT
  python scripts/paper2ppt.py --polish input.pptx --color academic_blue --font professional

  # 列出可用配色方案
  python scripts/paper2ppt.py --list-schemes

PPT 风格:
  academic  - 学术风格（简洁专业）
  business  - 商务风格
  modern    - 现代简约
  colorful  - 多彩活泼

配色方案:
  academic_blue   - 学术蓝
  modern_green    - 现代绿
  elegant_purple  - 优雅紫
  business_navy   - 商务蓝
  warm_orange     - 温暖橙
  minimal_gray    - 极简灰
        """,
    )

    # 输入选项
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--pdf", type=str, help="PDF 文件路径")
    input_group.add_argument("--text", type=str, help="论文文本内容")
    input_group.add_argument("--file", type=str, help="文本文件路径（.txt/.md）")
    input_group.add_argument("--polish", type=str, help="要美化的 PPT 文件路径")
    input_group.add_argument("--list-schemes", action="store_true", help="列出可用配色方案")

    # PPT 生成选项
    parser.add_argument("--title", type=str, help="PPT 标题")
    parser.add_argument(
        "--style", "-s",
        type=str,
        default="academic",
        choices=["academic", "business", "modern", "colorful"],
        help="PPT 风格（默认: academic）",
    )

    # 美化选项
    parser.add_argument(
        "--color",
        type=str,
        default="academic_blue",
        help="配色方案（默认: academic_blue）",
    )
    parser.add_argument(
        "--font",
        type=str,
        default="professional",
        choices=["professional", "elegant", "modern"],
        help="字体方案（默认: professional）",
    )
    parser.add_argument(
        "--no-page-numbers",
        action="store_true",
        help="不添加页码",
    )

    # 输出选项
    parser.add_argument("--output", "-o", type=str, help="输出文件路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")

    args = parser.parse_args()

    # 配置日志
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO", format="{message}")

    # 列出配色方案
    if args.list_schemes:
        from src.paper2figure import PPTPolish
        print("\n📎 可用配色方案:")
        for key, name in PPTPolish.list_color_schemes().items():
            print(f"   {key}: {name}")
        print("\n📝 可用字体方案:")
        for key, name in PPTPolish.list_font_schemes().items():
            print(f"   {key}: {name}")
        return

    # 美化 PPT
    if args.polish:
        from src.paper2figure import PPTPolish, PolishMode

        pptx_path = Path(args.polish)
        if not pptx_path.exists():
            logger.error(f"❌ 文件不存在: {pptx_path}")
            sys.exit(1)

        output_path = args.output or str(pptx_path.parent / f"{pptx_path.stem}_polished.pptx")

        logger.info(f"🎨 美化 PPT: {pptx_path}")

        try:
            polisher = PPTPolish()
            result = polisher.polish(
                pptx_path,
                output_path=output_path,
                mode=PolishMode.FULL,
                color_scheme=args.color,
                font_scheme=args.font,
                add_numbers=not args.no_page_numbers,
            )

            logger.info(f"✅ 美化完成: {result.output_path}")
            logger.info("📋 修改内容:")
            for change in result.changes:
                logger.info(f"   • {change}")

            if result.suggestions:
                logger.info("💡 优化建议:")
                for suggestion in result.suggestions:
                    logger.info(f"   • {suggestion}")

        except Exception as e:
            logger.error(f"❌ 美化失败: {e}")
            sys.exit(1)

        return

    # 生成 PPT
    if not any([args.pdf, args.text, args.file]):
        parser.print_help()
        sys.exit(1)

    from src.paper2figure import Paper2PPT, PPTStyle

    content = None
    title = args.title

    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            logger.error(f"❌ PDF 文件不存在: {pdf_path}")
            sys.exit(1)

        logger.info(f"📄 解析 PDF: {pdf_path}")

        try:
            from src.document_loader import MinerUDocumentLoader
            loader = MinerUDocumentLoader()
            chunks = loader.load(str(pdf_path), chunk_size=10000, chunk_overlap=0)
            if chunks:
                content = "\n\n".join(c["text"] for c in chunks)
                if not title:
                    title = pdf_path.stem
            else:
                logger.error("❌ PDF 解析失败")
                sys.exit(1)
        except Exception as e:
            logger.error(f"❌ PDF 解析失败: {e}")
            sys.exit(1)

    elif args.text:
        content = args.text
        if not title:
            title = "演示文稿"

    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"❌ 文件不存在: {file_path}")
            sys.exit(1)
        content = file_path.read_text(encoding="utf-8")
        if not title:
            title = file_path.stem

    if not content or len(content) < 100:
        logger.error("❌ 内容太短，请提供更多论文内容")
        sys.exit(1)

    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        safe_title = "".join(c for c in title if c.isalnum() or c in " _-")[:50]
        output_path = Path("./output") / f"{safe_title}.pptx"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 初始化
    try:
        p2ppt = Paper2PPT()
    except ValueError as e:
        logger.error(f"❌ 初始化失败: {e}")
        logger.info("💡 请确保已配置 LLM_API_KEY 环境变量")
        sys.exit(1)

    # 解析风格
    style = PPTStyle(args.style)

    # 生成 PPT
    logger.info(f"🔄 正在生成 PPT（{style.value} 风格）...")
    logger.info(f"📊 内容长度: {len(content)} 字符")

    try:
        ppt_content = p2ppt.analyze(content, title)
        logger.info(f"📝 生成 {len(ppt_content.slides)} 页幻灯片")

        result_path = p2ppt.generate_pptx(ppt_content, output_path, style)

        logger.info(f"✅ PPT 生成成功: {result_path}")
        logger.info(f"📄 标题: {ppt_content.title}")
        logger.info(f"📑 页数: {len(ppt_content.slides)}")

        # 显示幻灯片概览
        print("\n" + "=" * 50)
        print("📋 幻灯片概览:")
        print("=" * 50)
        for i, slide in enumerate(ppt_content.slides, 1):
            print(f"  {i}. [{slide.slide_type}] {slide.title}")
        print("=" * 50)

    except Exception as e:
        logger.error(f"❌ PPT 生成失败: {e}")
        sys.exit(1)

    logger.info("✨ 完成!")


if __name__ == "__main__":
    main()
