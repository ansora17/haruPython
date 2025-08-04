"""
챗봇 모듈
운동 및 영양 관련 질문 처리
"""

from .models import Question, ChatResponse, UserProfile
from .knowledge_base import KnowledgeBaseManager
from .question_processor import QuestionProcessor

__all__ = [
    "Question",
    "ChatResponse", 
    "UserProfile",
    "KnowledgeBaseManager",
    "QuestionProcessor"
]