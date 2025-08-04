"""
챗봇 관련 데이터 모델들
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any

class UserProfile(BaseModel):
    """사용자 프로필 모델"""
    age: int = 25
    gender: str = "여성"
    weight: int = 55
    height: Optional[int] = None  # 키 정보 (선택적)
    activity_level: Optional[str] = "보통"  # 활동량 (선택적)

class Question(BaseModel):
    """질문 요청 모델 - 사용자 정보는 쿠키에서 자동 추출"""
    question: str
    
    class Config:
        schema_extra = {
            "example": {
                "question": "10분 러닝으로 몇 칼로리 태울 수 있나요?"
            }
        }

class ChatResponse(BaseModel):
    """챗봇 응답 모델"""
    answer: str
    type: str
    status: Optional[str] = None
    user_info: Optional[Dict[str, Any]] = None
    sources_count: Optional[int] = None
    command: Optional[str] = None