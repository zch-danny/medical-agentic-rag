"""
Medical Agentic RAG - Gradio 前端

提供可视化对话界面，支持:
- 医疗问答对话
- PubMed 联网搜索
- 结构化信息提取
- 参数调节
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Generator, List, Tuple

import gradio as gr
from loguru import logger

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# 导入项目模块
from config import settings
from src.generator import AnswerGenerator, GenerationConfig
from src.advanced.extractor import MedicalExtractor, extract_medical_info
from src.advanced.pubmed import PubMedClient, search_pubmed_sync


# ============== 全局状态 ==============

class AppState:
    """应用状态管理"""
    
    def __init__(self):
        self._generator = None
        self._extractor = MedicalExtractor()
        self._pipeline = None
        
    @property
    def generator(self) -> AnswerGenerator:
        if self._generator is None:
            try:
                self._generator = AnswerGenerator()
            except Exception as e:
                logger.error(f"初始化 Generator 失败: {e}")
                raise
        return self._generator
    
    @property
    def extractor(self) -> MedicalExtractor:
        return self._extractor
    
    def get_pipeline(self):
        """懒加载 RAG Pipeline"""
        if self._pipeline is None:
            try:
                from src.pipeline import MedicalRAGPipeline
                self._pipeline = MedicalRAGPipeline()
                logger.info("RAG Pipeline 初始化成功")
            except Exception as e:
                logger.warning(f"RAG Pipeline 初始化失败: {e}")
        return self._pipeline


# 全局状态实例
app_state = AppState()


# ============== 核心功能 ==============

def chat_response(
    message: str,
    history: List[Tuple[str, str]],
    use_rag: bool,
    use_pubmed: bool,
    top_k: int,
    alpha: float,
) -> Generator[str, None, None]:
    """
    处理聊天请求
    
    Args:
        message: 用户消息
        history: 对话历史
        use_rag: 是否使用本地 RAG
        use_pubmed: 是否使用 PubMed 补充
        top_k: 检索数量
        alpha: 混合检索权重
    """
    if not message.strip():
        yield "请输入问题。"
        return
    
    documents = []
    sources_info = []
    
    # 1. 本地 RAG 检索
    if use_rag:
        pipeline = app_state.get_pipeline()
        if pipeline:
            try:
                yield "🔍 正在检索本地知识库..."
                docs = pipeline.retrieve(
                    query=message,
                    alpha=alpha,
                    final_top_k=top_k,
                )
                documents.extend(docs)
                sources_info.append(f"本地检索: {len(docs)} 篇文档")
                logger.info(f"本地检索返回 {len(docs)} 篇文档")
            except Exception as e:
                logger.error(f"本地检索失败: {e}")
                sources_info.append(f"本地检索失败: {e}")
    
    # 2. PubMed 联网搜索
    if use_pubmed:
        try:
            yield "🌐 正在搜索 PubMed..."
            pubmed_articles = search_pubmed_sync(message, max_results=min(5, top_k))
            
            for article in pubmed_articles:
                if article.abstract:
                    documents.append({
                        "entity": {
                            "original_text": f"{article.title}\n\n{article.abstract}",
                            "source": f"PubMed: {article.pmid}",
                        },
                        "score": 0.8,
                    })
            
            sources_info.append(f"PubMed: {len(pubmed_articles)} 篇文章")
            logger.info(f"PubMed 返回 {len(pubmed_articles)} 篇文章")
        except Exception as e:
            logger.error(f"PubMed 搜索失败: {e}")
            sources_info.append(f"PubMed 搜索失败: {e}")
    
    # 3. 生成回答
    if not documents:
        yield "❌ 未找到相关文献。请尝试启用 PubMed 搜索或检查本地知识库。"
        return
    
    yield f"📚 找到 {len(documents)} 篇相关文献，正在生成回答...\n\n"
    
    try:
        generator = app_state.generator
        response_text = ""
        
        for chunk in generator.generate_stream(message, documents):
            response_text += chunk
            yield response_text
        
        # 添加来源信息
        yield response_text + f"\n\n---\n📖 **来源**: {', '.join(sources_info)}"
        
    except Exception as e:
        logger.error(f"生成回答失败: {e}")
        yield f"❌ 生成回答失败: {e}"


def extract_info(text: str) -> str:
    """提取结构化医疗信息"""
    if not text.strip():
        return "请输入医疗文本。"
    
    try:
        info = extract_medical_info(text)
        
        result_parts = ["## 📋 提取结果\n"]
        
        if info.diseases:
            result_parts.append("### 🏥 疾病")
            for d in info.diseases:
                result_parts.append(f"- {d.normalized}")
        
        if info.symptoms:
            result_parts.append("\n### 🤒 症状")
            for s in info.symptoms:
                result_parts.append(f"- {s.normalized}")
        
        if info.medications:
            result_parts.append("\n### 💊 药物")
            for m in info.medications:
                result_parts.append(f"- {m.normalized}")
        
        if info.treatments:
            result_parts.append("\n### 🩺 治疗")
            for t in info.treatments:
                result_parts.append(f"- {t.normalized}")
        
        if info.examinations:
            result_parts.append("\n### 🔬 检查")
            for e in info.examinations:
                result_parts.append(f"- {e.normalized}")
        
        if info.total_entities == 0:
            result_parts.append("\n未识别到医疗实体。")
        else:
            result_parts.append(f"\n\n---\n**共识别 {info.total_entities} 个实体**")
        
        return "\n".join(result_parts)
        
    except Exception as e:
        logger.error(f"信息提取失败: {e}")
        return f"❌ 提取失败: {e}"


def search_pubmed_ui(query: str, max_results: int) -> str:
    """PubMed 搜索界面"""
    if not query.strip():
        return "请输入搜索词。"
    
    try:
        articles = search_pubmed_sync(query, max_results=max_results)
        
        if not articles:
            return "未找到相关文章。"
        
        result_parts = [f"## 🔬 PubMed 搜索结果 ({len(articles)} 篇)\n"]
        
        for i, article in enumerate(articles, 1):
            result_parts.append(f"### {i}. {article.title or 'No Title'}")
            result_parts.append(f"**PMID**: {article.pmid}")
            
            if article.authors:
                authors = ", ".join(article.authors[:3])
                if len(article.authors) > 3:
                    authors += " et al."
                result_parts.append(f"**作者**: {authors}")
            
            if article.journal:
                result_parts.append(f"**期刊**: {article.journal}")
            
            if article.pub_date:
                result_parts.append(f"**发表日期**: {article.pub_date}")
            
            if article.abstract:
                abstract = article.abstract[:500] + "..." if len(article.abstract) > 500 else article.abstract
                result_parts.append(f"\n{abstract}")
            
            result_parts.append("\n---\n")
        
        return "\n".join(result_parts)
        
    except Exception as e:
        logger.error(f"PubMed 搜索失败: {e}")
        return f"❌ 搜索失败: {e}"


# ============== Gradio 界面 ==============

def create_ui() -> gr.Blocks:
    """创建 Gradio 界面"""
    
    with gr.Blocks(
        title="Medical Agentic RAG",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .title { text-align: center; margin-bottom: 20px; }
        """
    ) as app:
        
        gr.Markdown(
            """
            # 🏥 Medical Agentic RAG
            
            基于 LlamaIndex + DSPy 的医疗文献智能问答系统
            """,
            elem_classes="title"
        )
        
        with gr.Tabs():
            # Tab 1: 对话问答
            with gr.Tab("💬 医疗问答"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="对话",
                            height=500,
                            show_copy_button=True,
                        )
                        
                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="输入问题",
                                placeholder="请输入您的医疗问题，例如：糖尿病的症状有哪些？",
                                lines=2,
                                scale=4,
                            )
                            submit_btn = gr.Button("发送", variant="primary", scale=1)
                        
                        with gr.Row():
                            clear_btn = gr.Button("清空对话")
                            
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ 设置")
                        
                        use_rag = gr.Checkbox(
                            label="使用本地知识库",
                            value=True,
                            info="检索本地 Milvus 向量库"
                        )
                        
                        use_pubmed = gr.Checkbox(
                            label="联网搜索 PubMed",
                            value=False,
                            info="从 PubMed 获取最新文献"
                        )
                        
                        top_k = gr.Slider(
                            label="检索数量",
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                        )
                        
                        alpha = gr.Slider(
                            label="混合检索权重",
                            minimum=0,
                            maximum=1,
                            value=0.7,
                            step=0.1,
                            info="0=纯BM25, 1=纯向量"
                        )
                
                # 事件绑定
                def user_message(message, history):
                    return "", history + [[message, None]]
                
                def bot_response(history, use_rag, use_pubmed, top_k, alpha):
                    if not history:
                        return history
                    
                    message = history[-1][0]
                    history[-1][1] = ""
                    
                    for chunk in chat_response(message, history[:-1], use_rag, use_pubmed, top_k, alpha):
                        history[-1][1] = chunk
                        yield history
                
                msg_input.submit(
                    user_message,
                    [msg_input, chatbot],
                    [msg_input, chatbot],
                    queue=False,
                ).then(
                    bot_response,
                    [chatbot, use_rag, use_pubmed, top_k, alpha],
                    chatbot,
                )
                
                submit_btn.click(
                    user_message,
                    [msg_input, chatbot],
                    [msg_input, chatbot],
                    queue=False,
                ).then(
                    bot_response,
                    [chatbot, use_rag, use_pubmed, top_k, alpha],
                    chatbot,
                )
                
                clear_btn.click(lambda: [], None, chatbot, queue=False)
            
            # Tab 2: 信息提取
            with gr.Tab("📋 信息提取"):
                gr.Markdown(
                    """
                    ### 从医疗文本中提取结构化信息
                    
                    支持提取：疾病、症状、药物、治疗方案、检查项目等
                    """
                )
                
                with gr.Row():
                    with gr.Column():
                        extract_input = gr.Textbox(
                            label="输入医疗文本",
                            placeholder="请输入医疗相关文本，例如：患者诊断为2型糖尿病合并高血压，建议服用二甲双胍500mg tid，定期监测血糖。",
                            lines=8,
                        )
                        extract_btn = gr.Button("提取信息", variant="primary")
                    
                    with gr.Column():
                        extract_output = gr.Markdown(label="提取结果")
                
                extract_btn.click(
                    extract_info,
                    inputs=extract_input,
                    outputs=extract_output,
                )
                
                # 示例
                gr.Examples(
                    examples=[
                        ["患者诊断为2型糖尿病合并高血压，既往有冠心病病史。建议服用二甲双胍500mg tid，阿司匹林100mg qd。定期监测血糖、血压，完善心电图和肝肾功能检查。"],
                        ["主诉：头痛、发热3天，伴咳嗽、咳痰。查体：体温38.5℃，咽部充血。诊断：急性上呼吸道感染。处方：布洛芬退热，阿莫西林抗感染。"],
                    ],
                    inputs=extract_input,
                )
            
            # Tab 3: PubMed 搜索
            with gr.Tab("🔬 PubMed 搜索"):
                gr.Markdown(
                    """
                    ### 搜索 PubMed 医学文献数据库
                    
                    获取最新的医学研究和临床指南
                    """
                )
                
                with gr.Row():
                    pubmed_query = gr.Textbox(
                        label="搜索词",
                        placeholder="输入搜索词，例如：diabetes treatment 2024",
                        scale=4,
                    )
                    pubmed_max = gr.Slider(
                        label="最大结果数",
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        scale=1,
                    )
                    pubmed_btn = gr.Button("搜索", variant="primary", scale=1)
                
                pubmed_output = gr.Markdown(label="搜索结果")
                
                pubmed_btn.click(
                    search_pubmed_ui,
                    inputs=[pubmed_query, pubmed_max],
                    outputs=pubmed_output,
                )
                
                # 示例搜索
                gr.Examples(
                    examples=[
                        ["diabetes treatment guidelines", 5],
                        ["COVID-19 vaccine efficacy", 5],
                        ["hypertension management", 5],
                    ],
                    inputs=[pubmed_query, pubmed_max],
                )
            
            # Tab 4: 关于
            with gr.Tab("ℹ️ 关于"):
                gr.Markdown(
                    """
                    ## Medical Agentic RAG 系统
                    
                    ### 🎯 功能特性
                    
                    - **智能问答**: 基于医疗文献的专业问答
                    - **混合检索**: BM25 + 向量检索 + 重排序
                    - **PubMed 集成**: 联网获取最新研究
                    - **信息提取**: 从文本中提取结构化医疗信息
                    - **DSPy 优化**: 自动优化提示词
                    
                    ### 🏗️ 技术架构
                    
                    - **向量数据库**: Milvus
                    - **嵌入模型**: Qwen3-Embedding-8B
                    - **重排序**: Qwen3-Reranker-8B
                    - **LLM**: DeepSeek / OpenAI 兼容
                    - **Agent 框架**: LlamaIndex
                    - **优化框架**: DSPy
                    
                    ### ⚠️ 免责声明
                    
                    本系统仅供学习和研究使用，**不能替代专业医疗建议**。
                    如有健康问题，请咨询专业医生。
                    
                    ---
                    
                    GitHub: [medical-agentic-rag](https://github.com/zch-danny/medical-agentic-rag)
                    """
                )
        
        gr.Markdown(
            """
            ---
            <center>
            Medical Agentic RAG © 2024 | Powered by LlamaIndex + DSPy
            </center>
            """,
            elem_classes="footer"
        )
    
    return app


# ============== 主入口 ==============

def main():
    """启动应用"""
    logger.info("启动 Medical Agentic RAG 前端...")
    
    app = create_ui()
    app.queue()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
