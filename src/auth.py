"""
API 认证模块 - API Key 验证

支持两种模式:
1. 环境变量模式: 配置 API_KEYS 环境变量（逗号分隔多个 Key）
2. 数据库模式: 从数据库读取有效 Key

使用方法:
    from src.auth import require_api_key
    
    @app.get("/protected")
    async def protected_endpoint(api_key: str = Depends(require_api_key)):
        return {"message": "authenticated"}
"""
import secrets
from typing import List, Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery
from loguru import logger


# API Key 可以通过 Header 或 Query 参数传递
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


def get_configured_api_keys() -> List[str]:
    """从配置获取有效的 API Key 列表"""
    from config.settings import settings
    
    api_keys_str = getattr(settings, "API_KEYS", "")
    if not api_keys_str:
        return []
    
    # 支持逗号分隔多个 Key
    keys = [k.strip() for k in api_keys_str.split(",") if k.strip()]
    return keys


def is_auth_enabled() -> bool:
    """检查是否启用了 API 认证"""
    from config.settings import settings
    
    # 如果配置了 API_KEYS 或 API_AUTH_ENABLED=true，则启用认证
    api_keys = get_configured_api_keys()
    auth_enabled = getattr(settings, "API_AUTH_ENABLED", False)
    
    return bool(api_keys) or auth_enabled


def validate_api_key(api_key: str) -> bool:
    """
    验证 API Key 是否有效
    
    Args:
        api_key: 待验证的 Key
        
    Returns:
        是否有效
    """
    if not api_key:
        return False
    
    # 1. 先检查环境变量中的 Key
    configured_keys = get_configured_api_keys()
    if api_key in configured_keys:
        return True
    
    # 2. 再检查数据库中的 Key（如果启用了数据库验证）
    from config.settings import settings
    if getattr(settings, "API_AUTH_USE_DB", False):
        try:
            from src.database import get_database
            db = get_database()
            return db.validate_api_key(api_key)
        except Exception as e:
            logger.warning(f"数据库 API Key 验证失败: {e}")
    
    return False


async def get_api_key(
    api_key_header: str = Security(api_key_header),
    api_key_query: str = Security(api_key_query),
) -> Optional[str]:
    """
    从请求中获取 API Key
    
    优先级: Header > Query
    """
    if api_key_header:
        return api_key_header
    if api_key_query:
        return api_key_query
    return None


async def require_api_key(
    api_key_header: str = Security(api_key_header),
    api_key_query: str = Security(api_key_query),
) -> str:
    """
    要求 API Key 认证的依赖
    
    Usage:
        @app.get("/protected")
        async def endpoint(api_key: str = Depends(require_api_key)):
            ...
    """
    # 如果未启用认证，直接通过
    if not is_auth_enabled():
        return "auth_disabled"
    
    # 获取 Key
    api_key = api_key_header or api_key_query
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 API Key，请在 Header (X-API-Key) 或 Query (api_key) 中提供",
        )
    
    # 验证 Key
    if not validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 API Key",
        )
    
    return api_key


def generate_api_key(length: int = 32) -> str:
    """
    生成随机 API Key
    
    Args:
        length: Key 长度
        
    Returns:
        新生成的 API Key
    """
    return secrets.token_urlsafe(length)


# ============== 辅助函数 ==============

def create_api_key_in_db(name: str = None) -> dict:
    """
    在数据库中创建新的 API Key
    
    Args:
        name: Key 的名称/描述
        
    Returns:
        包含 key 和 name 的字典
    """
    from src.database import get_database
    
    key = generate_api_key()
    db = get_database()
    api_key = db.create_api_key(key=key, name=name)
    
    return {
        "key": key,
        "name": api_key.name,
        "created_at": api_key.created_at.isoformat(),
    }


def list_api_keys_from_db() -> list:
    """列出数据库中所有 API Key（脱敏显示）"""
    from src.database import get_database
    
    db = get_database()
    keys = db.list_api_keys()
    
    return [
        {
            "id": k.id,
            "name": k.name,
            "key_prefix": k.key[:8] + "..." if k.key else None,
            "is_active": bool(k.is_active),
            "created_at": k.created_at.isoformat() if k.created_at else None,
            "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
        }
        for k in keys
    ]
