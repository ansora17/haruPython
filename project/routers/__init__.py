"""
API 라우터들
각 기능별로 엔드포인트 분리
"""

from .chatbot import router as chatbot_router
from .image import router as image_router  
from .system import router as system_router

__all__ = [
    "chatbot_router",
    "image_router", 
    "system_router"
]