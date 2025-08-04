"""
챗봇 관련 API 라우터
질문 처리, 명령어 목록 등
"""

from fastapi import APIRouter, Request
from chatbot.models import Question, ChatResponse
from chatbot.knowledge_base import KnowledgeBaseManager
from chatbot.question_processor import QuestionProcessor
from chatbot.cookie_utils import get_user_info_from_cookies
from pathlib import Path

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

# 전역 변수들
knowledge_manager = KnowledgeBaseManager()
question_processor = None

def initialize_chatbot():
    """챗봇 초기화"""
    global question_processor
    
    print("🚀 지식베이스 초기화 시작...")
    data_dir = Path(__file__).parent.parent.parent / "data"
    files = list(data_dir.glob("*.*"))
    knowledge_manager.init_knowledge_base(files)
    print("✅ 지식베이스 초기화 완료!")
    
    question_processor = QuestionProcessor(knowledge_manager)

@router.post("/ask", response_model=ChatResponse)
async def ask_question(question_data: Question, request: Request):
    """
    🤖 개인화 운동 챗봇에게 질문하기
    
    **사용법:**
    - 질문만 입력하면 됩니다
    - 사용자 정보는 쿠키에서 자동으로 읽어옵니다
    
    **예시 질문:**
    
    **🏃‍♀️ 운동 관련:**
    - "10분 러닝으로 몇 칼로리 태울 수 있나요?"
    - "30분 운동으로 200칼로리 태우려면?"
    - "체중 감량에 좋은 운동 추천해주세요"
    
    **🍎 음식/영양성분:**
    - "바나나 1개의 칼로리는 얼마나 되나요?"
    - "닭가슴살 100g의 단백질 함량은?"
    - "현미밥과 백미밥의 영양성분 차이점은?"
    
    **개인화 정보:**
    - 쿠키에서 자동 추출
    - 쿠키가 없으면 개인화 없이 일반 답변 제공
    """
    global question_processor
    
    if question_processor is None:
        return ChatResponse(
            answer="⚠️ 챗봇이 아직 초기화되지 않았습니다. 잠시 후 다시 시도해주세요.",
            type="system_message",
            status="chatbot_not_ready"
        )
    
    # 쿠키에서 사용자 정보 추출
    user_info = get_user_info_from_cookies(request)
    
    return await question_processor.process_question(question_data, user_info)

@router.get("/commands")
async def get_available_commands():
    """사용 가능한 명령어 리스트"""
    return {
        "commands": [
            {"command": "/음식", "description": "개인 식사 기록 분석"},
            {"command": "/식사", "description": "최근 식사 패턴 분석"}, 
            {"command": "/식단", "description": "식단 기록 조회"},
            {"command": "/칼로리", "description": "칼로리 섭취 분석"},
            {"command": "/오늘음식", "description": "오늘 섭취한 음식 분석"},
            {"command": "/어제음식", "description": "어제 섭취한 음식 분석"}
        ]
    }