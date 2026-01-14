"""
PubMed API 集成

提供从 PubMed/NCBI 获取最新医学文献的功能

API 文档: https://www.ncbi.nlm.nih.gov/books/NBK25500/
"""

import asyncio
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import httpx
from loguru import logger


# PubMed E-utilities 基础 URL
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_SEARCH_URL = f"{PUBMED_BASE_URL}/esearch.fcgi"
PUBMED_FETCH_URL = f"{PUBMED_BASE_URL}/efetch.fcgi"
PUBMED_SUMMARY_URL = f"{PUBMED_BASE_URL}/esummary.fcgi"


@dataclass
class PubMedArticle:
    """PubMed 文章数据结构"""
    
    pmid: str                                    # PubMed ID
    title: str = ""                              # 标题
    abstract: str = ""                           # 摘要
    authors: List[str] = field(default_factory=list)  # 作者列表
    journal: str = ""                            # 期刊名
    pub_date: Optional[str] = None               # 发表日期
    doi: Optional[str] = None                    # DOI
    keywords: List[str] = field(default_factory=list)  # 关键词
    mesh_terms: List[str] = field(default_factory=list)  # MeSH 主题词
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "journal": self.journal,
            "pub_date": self.pub_date,
            "doi": self.doi,
            "keywords": self.keywords,
            "mesh_terms": self.mesh_terms,
        }
    
    def to_document(self) -> Dict[str, Any]:
        """转换为文档格式（用于 RAG）"""
        content = f"{self.title}\n\n{self.abstract}"
        return {
            "content": content,
            "metadata": {
                "source": "pubmed",
                "pmid": self.pmid,
                "title": self.title,
                "authors": ", ".join(self.authors[:3]) + ("..." if len(self.authors) > 3 else ""),
                "journal": self.journal,
                "pub_date": self.pub_date,
                "doi": self.doi,
            }
        }


@dataclass
class PubMedConfig:
    """PubMed 客户端配置"""
    
    api_key: Optional[str] = None      # NCBI API Key（可选，提高请求限制）
    email: Optional[str] = None         # 联系邮箱（NCBI 推荐提供）
    tool: str = "medical_rag"           # 工具名称
    max_results: int = 20               # 默认最大结果数
    timeout: float = 30.0               # 请求超时（秒）
    retry_count: int = 3                # 重试次数
    retry_delay: float = 1.0            # 重试延迟（秒）


class PubMedClient:
    """
    PubMed API 客户端
    
    支持:
    - 关键词搜索
    - 按 PMID 获取文章详情
    - 批量异步获取
    - 日期范围过滤
    
    示例:
        ```python
        client = PubMedClient()
        
        # 搜索文章
        articles = await client.search("diabetes treatment", max_results=10)
        
        # 获取特定文章
        article = await client.fetch_article("12345678")
        
        # 批量获取
        articles = await client.fetch_articles(["12345678", "87654321"])
        ```
    """
    
    def __init__(self, config: Optional[PubMedConfig] = None):
        self.config = config or PubMedConfig()
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建 HTTP 客户端"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout,
                follow_redirects=True,
            )
        return self._client
    
    async def close(self):
        """关闭客户端"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    def _build_params(self, extra_params: Dict[str, Any]) -> Dict[str, Any]:
        """构建请求参数"""
        params = {
            "db": "pubmed",
            "retmode": "xml",
            "tool": self.config.tool,
        }
        if self.config.api_key:
            params["api_key"] = self.config.api_key
        if self.config.email:
            params["email"] = self.config.email
        params.update(extra_params)
        return params
    
    async def _request_with_retry(self, url: str, params: Dict) -> Optional[str]:
        """带重试的请求"""
        client = await self._get_client()
        
        for attempt in range(self.config.retry_count):
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.text
            except httpx.HTTPError as e:
                logger.warning(f"PubMed 请求失败 (尝试 {attempt + 1}/{self.config.retry_count}): {e}")
                if attempt < self.config.retry_count - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        return None
    
    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        sort: str = "relevance",
    ) -> List[str]:
        """
        搜索 PubMed 文章
        
        Args:
            query: 搜索词（支持 PubMed 高级搜索语法）
            max_results: 最大结果数
            min_date: 最小日期 (YYYY/MM/DD)
            max_date: 最大日期 (YYYY/MM/DD)
            sort: 排序方式 ("relevance" 或 "date")
            
        Returns:
            PMID 列表
        """
        max_results = max_results or self.config.max_results
        
        params = self._build_params({
            "term": query,
            "retmax": max_results,
            "sort": sort,
            "usehistory": "n",
        })
        
        if min_date:
            params["mindate"] = min_date
        if max_date:
            params["maxdate"] = max_date
        if min_date or max_date:
            params["datetype"] = "pdat"  # publication date
        
        logger.info(f"PubMed 搜索: {query[:50]}... (max={max_results})")
        
        xml_text = await self._request_with_retry(PUBMED_SEARCH_URL, params)
        if not xml_text:
            logger.error("PubMed 搜索请求失败")
            return []
        
        # 解析 XML 获取 PMID 列表
        try:
            root = ET.fromstring(xml_text)
            pmids = [id_elem.text for id_elem in root.findall(".//Id") if id_elem.text]
            logger.info(f"PubMed 搜索返回 {len(pmids)} 条结果")
            return pmids
        except ET.ParseError as e:
            logger.error(f"PubMed XML 解析失败: {e}")
            return []
    
    async def fetch_article(self, pmid: str) -> Optional[PubMedArticle]:
        """
        获取单篇文章详情
        
        Args:
            pmid: PubMed ID
            
        Returns:
            PubMedArticle 或 None
        """
        articles = await self.fetch_articles([pmid])
        return articles[0] if articles else None
    
    async def fetch_articles(self, pmids: List[str]) -> List[PubMedArticle]:
        """
        批量获取文章详情
        
        Args:
            pmids: PMID 列表
            
        Returns:
            PubMedArticle 列表
        """
        if not pmids:
            return []
        
        params = self._build_params({
            "id": ",".join(pmids),
            "rettype": "abstract",
        })
        
        logger.info(f"PubMed 获取 {len(pmids)} 篇文章详情")
        
        xml_text = await self._request_with_retry(PUBMED_FETCH_URL, params)
        if not xml_text:
            logger.error("PubMed fetch 请求失败")
            return []
        
        return self._parse_articles(xml_text)
    
    def _parse_articles(self, xml_text: str) -> List[PubMedArticle]:
        """解析 PubMed XML 响应"""
        articles = []
        
        try:
            root = ET.fromstring(xml_text)
            
            for article_elem in root.findall(".//PubmedArticle"):
                article = self._parse_single_article(article_elem)
                if article:
                    articles.append(article)
                    
        except ET.ParseError as e:
            logger.error(f"PubMed XML 解析失败: {e}")
        
        return articles
    
    def _parse_single_article(self, elem: ET.Element) -> Optional[PubMedArticle]:
        """解析单篇文章"""
        try:
            # PMID
            pmid_elem = elem.find(".//PMID")
            if pmid_elem is None or not pmid_elem.text:
                return None
            pmid = pmid_elem.text
            
            # 标题
            title_elem = elem.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None and title_elem.text else ""
            
            # 摘要
            abstract_parts = []
            for abstract_elem in elem.findall(".//AbstractText"):
                label = abstract_elem.get("Label", "")
                text = abstract_elem.text or ""
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = "\n".join(abstract_parts)
            
            # 作者
            authors = []
            for author_elem in elem.findall(".//Author"):
                last_name = author_elem.findtext("LastName", "")
                fore_name = author_elem.findtext("ForeName", "")
                if last_name:
                    authors.append(f"{last_name} {fore_name}".strip())
            
            # 期刊
            journal_elem = elem.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None and journal_elem.text else ""
            
            # 发表日期
            pub_date = None
            pub_date_elem = elem.find(".//PubDate")
            if pub_date_elem is not None:
                year = pub_date_elem.findtext("Year", "")
                month = pub_date_elem.findtext("Month", "")
                day = pub_date_elem.findtext("Day", "")
                if year:
                    pub_date = year
                    if month:
                        pub_date += f"-{month}"
                        if day:
                            pub_date += f"-{day}"
            
            # DOI
            doi = None
            for article_id in elem.findall(".//ArticleId"):
                if article_id.get("IdType") == "doi":
                    doi = article_id.text
                    break
            
            # 关键词
            keywords = [
                kw.text for kw in elem.findall(".//Keyword")
                if kw.text
            ]
            
            # MeSH 主题词
            mesh_terms = [
                mesh.text for mesh in elem.findall(".//MeshHeading/DescriptorName")
                if mesh.text
            ]
            
            return PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                pub_date=pub_date,
                doi=doi,
                keywords=keywords,
                mesh_terms=mesh_terms,
            )
            
        except Exception as e:
            logger.error(f"解析文章失败: {e}")
            return None
    
    async def search_and_fetch(
        self,
        query: str,
        max_results: Optional[int] = None,
        **search_kwargs,
    ) -> List[PubMedArticle]:
        """
        搜索并获取文章详情（合并操作）
        
        Args:
            query: 搜索词
            max_results: 最大结果数
            **search_kwargs: 传递给 search() 的额外参数
            
        Returns:
            PubMedArticle 列表
        """
        pmids = await self.search(query, max_results=max_results, **search_kwargs)
        if not pmids:
            return []
        return await self.fetch_articles(pmids)
    
    async def get_recent_articles(
        self,
        topic: str,
        days: int = 30,
        max_results: int = 10,
    ) -> List[PubMedArticle]:
        """
        获取最近发表的相关文章
        
        Args:
            topic: 主题
            days: 最近天数
            max_results: 最大结果数
            
        Returns:
            PubMedArticle 列表
        """
        from datetime import timedelta
        
        max_date = datetime.now()
        min_date = max_date - timedelta(days=days)
        
        return await self.search_and_fetch(
            query=topic,
            max_results=max_results,
            min_date=min_date.strftime("%Y/%m/%d"),
            max_date=max_date.strftime("%Y/%m/%d"),
            sort="date",
        )


# ============== 便捷函数 ==============

async def search_pubmed(
    query: str,
    max_results: int = 10,
    fetch_details: bool = True,
    **kwargs,
) -> List[PubMedArticle]:
    """
    搜索 PubMed 的便捷函数
    
    Args:
        query: 搜索词
        max_results: 最大结果数
        fetch_details: 是否获取详情
        **kwargs: 额外搜索参数
        
    Returns:
        PubMedArticle 列表
    """
    client = PubMedClient()
    try:
        if fetch_details:
            return await client.search_and_fetch(query, max_results=max_results, **kwargs)
        else:
            pmids = await client.search(query, max_results=max_results, **kwargs)
            return [PubMedArticle(pmid=pmid) for pmid in pmids]
    finally:
        await client.close()


def search_pubmed_sync(
    query: str,
    max_results: int = 10,
    **kwargs,
) -> List[PubMedArticle]:
    """
    同步搜索 PubMed
    
    Args:
        query: 搜索词
        max_results: 最大结果数
        
    Returns:
        PubMedArticle 列表
    """
    return asyncio.run(search_pubmed(query, max_results=max_results, **kwargs))
