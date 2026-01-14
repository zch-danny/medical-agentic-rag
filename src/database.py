"""
数据库模块 - 文件状态持久化

支持多种数据库后端（通过更改连接字符串）:
- SQLite: sqlite:///data/app.db (默认，无需额外配置)
- PostgreSQL: postgresql://user:pass@host/db
- MySQL: mysql://user:pass@host/db
"""
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Enum as SQLEnum
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from loguru import logger


Base = declarative_base()


class DocumentStatus(str, Enum):
    """文档状态"""
    PENDING = "pending"      # 已上传，待索引
    INDEXING = "indexing"    # 索引中
    INDEXED = "indexed"      # 已索引
    FAILED = "failed"        # 索引失败


class Document(Base):
    """文档模型"""
    __tablename__ = "documents"

    id = Column(String(32), primary_key=True)
    filename = Column(String(512), nullable=False)
    original_filename = Column(String(512), nullable=False)
    path = Column(Text, nullable=False)
    size_bytes = Column(Integer, default=0)
    status = Column(SQLEnum(DocumentStatus), default=DocumentStatus.PENDING)
    chunk_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """转换为字典"""
        return {
            "file_id": self.id,
            "filename": self.original_filename,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "status": self.status.value,
            "chunk_count": self.chunk_count,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class APIKey(Base):
    """API Key 模型（可选，用于数据库管理 Key）"""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(128), unique=True, nullable=False, index=True)
    name = Column(String(256), nullable=True)  # Key 名称/描述
    is_active = Column(Integer, default=1)     # 是否启用
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)


class Database:
    """数据库管理类"""

    def __init__(self, database_url: str = None):
        """
        初始化数据库连接

        Args:
            database_url: 数据库连接字符串，默认使用 SQLite
        """
        if database_url is None:
            # 默认使用 SQLite，存储在 data 目录
            db_path = Path(__file__).parent.parent / "data" / "app.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            database_url = f"sqlite:///{db_path}"

        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            echo=False,  # 设为 True 可打印 SQL 语句
            # SQLite 特定配置
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
        )
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)

        # 创建表
        Base.metadata.create_all(self.engine)
        logger.info(f"数据库初始化完成: {database_url}")

    def get_session(self) -> Session:
        """获取数据库会话"""
        return self.SessionLocal()

    # ============== Document 操作 ==============

    def create_document(
        self,
        file_id: str,
        filename: str,
        original_filename: str,
        path: str,
        size_bytes: int = 0,
    ) -> Document:
        """创建文档记录"""
        with self.get_session() as session:
            doc = Document(
                id=file_id,
                filename=filename,
                original_filename=original_filename,
                path=path,
                size_bytes=size_bytes,
                status=DocumentStatus.PENDING,
            )
            session.add(doc)
            session.commit()
            session.refresh(doc)
            return doc

    def get_document(self, file_id: str) -> Optional[Document]:
        """获取文档"""
        with self.get_session() as session:
            return session.query(Document).filter(Document.id == file_id).first()

    def get_all_documents(self) -> List[Document]:
        """获取所有文档"""
        with self.get_session() as session:
            return session.query(Document).order_by(Document.created_at.desc()).all()

    def get_pending_documents(self) -> List[Document]:
        """获取待索引的文档"""
        with self.get_session() as session:
            return session.query(Document).filter(
                Document.status == DocumentStatus.PENDING
            ).all()

    def update_document_status(
        self,
        file_id: str,
        status: DocumentStatus,
        chunk_count: int = None,
        error_message: str = None,
    ) -> Optional[Document]:
        """更新文档状态"""
        with self.get_session() as session:
            doc = session.query(Document).filter(Document.id == file_id).first()
            if doc:
                doc.status = status
                if chunk_count is not None:
                    doc.chunk_count = chunk_count
                if error_message is not None:
                    doc.error_message = error_message
                doc.updated_at = datetime.utcnow()
                session.commit()
                session.refresh(doc)
            return doc

    def delete_document(self, file_id: str) -> bool:
        """删除文档记录"""
        with self.get_session() as session:
            doc = session.query(Document).filter(Document.id == file_id).first()
            if doc:
                session.delete(doc)
                session.commit()
                return True
            return False

    # ============== API Key 操作 ==============

    def create_api_key(self, key: str, name: str = None) -> APIKey:
        """创建 API Key"""
        with self.get_session() as session:
            api_key = APIKey(key=key, name=name)
            session.add(api_key)
            session.commit()
            session.refresh(api_key)
            return api_key

    def validate_api_key(self, key: str) -> bool:
        """验证 API Key"""
        with self.get_session() as session:
            api_key = session.query(APIKey).filter(
                APIKey.key == key,
                APIKey.is_active == 1,
            ).first()
            if api_key:
                # 更新最后使用时间
                api_key.last_used_at = datetime.utcnow()
                session.commit()
                return True
            return False

    def list_api_keys(self) -> List[APIKey]:
        """列出所有 API Key"""
        with self.get_session() as session:
            return session.query(APIKey).all()


# 全局数据库实例（延迟初始化）
_db: Optional[Database] = None


def get_database() -> Database:
    """获取数据库实例（单例）"""
    global _db
    if _db is None:
        from config.settings import settings
        database_url = getattr(settings, "DATABASE_URL", None)
        _db = Database(database_url)
    return _db


def init_database(database_url: str = None) -> Database:
    """初始化数据库"""
    global _db
    _db = Database(database_url)
    return _db
