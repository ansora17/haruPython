"""
질문 처리 핵심 로직
질문 분류, 응답 생성 담당
"""

from fastapi.concurrency import run_in_threadpool
from langchain_community.chat_models import ChatOpenAI
from .models import Question, ChatResponse
from .knowledge_base import KnowledgeBaseManager
from .utils import (
    detect_command, 
    detect_food_question, 
    call_external_api
)


class QuestionProcessor:
    """질문 처리 클래스"""
    
    def __init__(self, knowledge_manager: KnowledgeBaseManager):
        self.knowledge_manager = knowledge_manager
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0,
            max_tokens=3000,
        )
    
    async def process_question(self, request: Question, user_info: dict = None) -> ChatResponse:
        """질문 처리 메인 로직"""
        
        # 1단계: 명령어 감지
        is_command, command_type = detect_command(request.question)
        
        if is_command:
            return await self._handle_command(command_type, user_info, request)
        
        # 음식/영양성분 질문 처리
        elif detect_food_question(request.question):
            return await self._handle_food_question(request, user_info)
        
        else:
            # 일반 질문 처리 (운동 관련)
            return await self._handle_general_question(request, user_info)
    
    async def _handle_command(self, command_type: str, user_info: dict, request: Question) -> ChatResponse:
        """명령어 처리"""
        print(f"개인화 명령어 감지: {command_type}")
        
        if user_info:
            user_id = user_info.get("id", "unknown_user")
            # 외부 API에서 개인 데이터 가져오기
            api_data = await call_external_api(command_type, user_id)
            
            if "error" in api_data:
                return ChatResponse(
                    answer="죄송합니다. 개인 데이터를 불러올 수 없습니다. 잠시 후 다시 시도해주세요.",
                    type="error"
                )
            
            return ChatResponse(
                answer="개인화 기능은 곧 구현될 예정입니다.",
                type="personalized",
                command=command_type,
                user_info=user_info
            )
        else:
            return ChatResponse(
                answer="🍪 개인화 기능을 사용하려면 로그인이 필요합니다.",
                type="login_required",
                command=command_type
            )
    
    async def _handle_food_question(self, request: Question, user_info: dict) -> ChatResponse:
        """음식/영양성분 질문 처리"""
        print("음식/영양성분 질문으로 처리")
        print(f"🍪 쿠키에서 받은 사용자 정보: {user_info}")
        
        # 지식베이스에서 관련 문서 검색
        docs = self.knowledge_manager.search(request.question, k=4)
        
        if user_info:
            # 🎯 개인화된 영양 조언
            enhanced_prompt = f"""
You are a nutrition expert. Based on the provided data, please give personalized nutrition advice in a friendly manner.

Question: {request.question}

User Information:
- Age: {user_info['age']} years old
- Gender: {user_info['gender']}  
- Weight: {user_info['weight']}kg
- Activity Level: {user_info.get('activity_level', '보통')}

Please provide:
1. Accurate nutritional information
2. Personalized recommendations based on user's profile
3. Calorie and nutrition calculations considering their weight ({user_info['weight']}kg)
4. Healthy eating suggestions
5. Please summarize it in 4 lines or less

IMPORTANT: Please respond in Korean language only. All answers must be in Korean.

Related nutrition data:
"""
        else:
            # 🔍 일반적인 영양 정보 (개인화 없음)
            enhanced_prompt = f"""
You are a nutrition expert. Please provide general nutrition information in a helpful manner.

Question: {request.question}

Please provide:
1. General nutritional information
2. Average calorie and nutrition values
3. Healthy eating recommendations
4. Food composition details
5. Please summarize it in 4 lines or less

IMPORTANT: Please respond in Korean language only. All answers must be in Korean.

Related nutrition data:
"""
        
        # 컨텍스트와 함께 답변 생성
        if docs:
            context = enhanced_prompt + "\n".join([doc.page_content for doc in docs])
        else:
            context = enhanced_prompt + "\n관련 영양성분 데이터가 현재 로드되지 않아 일반적인 영양 정보로 답변드립니다."
        
        response = await run_in_threadpool(self.llm.predict, context)
        
        return ChatResponse(
            answer=response,
            type="food" if user_info else "food_not_personalized",
            sources_count=len(docs),
            user_info=user_info
        )
    
    async def _handle_general_question(self, request: Question, user_info: dict) -> ChatResponse:
        """일반 질문 처리 (운동 관련)"""
        print("일반 질문으로 처리")
        print(f"🍪 쿠키에서 받은 사용자 정보: {user_info}")
        
        # 지식베이스에서 관련 문서 검색
        docs = self.knowledge_manager.search(request.question, k=4)
        
        if user_info:
            # 🎯 개인화된 답변 생성
            enhanced_prompt = f"""
You are a personalized fitness expert. Based on the provided data, please give optimized answers to users in a friendly and warm manner like a close friend.

Question: {request.question}

User Information:
- Age: {user_info['age']} years old
- Gender: {user_info['gender']}  
- Weight: {user_info['weight']}kg
- Activity Level: {user_info.get('activity_level', '보통')}

When answering, please MUST include the following:
1. Accurate calorie calculations considering the user's weight ({user_info['weight']}kg)
2. Specific exercise duration and methods
3. Provide 3-5 exercise options
4. Offer choices by exercise intensity level
5. Please summarize it in 4 lines or less

IMPORTANT: Please respond in Korean language only. All answers must be in Korean.

Related exercise data:
"""
        else:
            # 🔍 일반적인 답변 (개인화 없음)
            enhanced_prompt = f"""
You are a fitness expert. Please give general fitness advice in a friendly and helpful manner.

Question: {request.question}

Please provide:
1. General exercise recommendations
2. Approximate calorie calculations for average adults
3. 3-5 exercise options
4. Different intensity levels
5. Please summarize it in 4 lines or less

IMPORTANT: Please respond in Korean language only. All answers must be in Korean.

Related exercise data:
"""
        
        # 컨텍스트와 함께 답변 생성
        if docs:
            context = enhanced_prompt + "\n".join([doc.page_content for doc in docs])
        else:
            context = enhanced_prompt + "\n관련 운동 데이터가 현재 로드되지 않아 일반적인 운동 정보로 답변드립니다."
        
        response = await run_in_threadpool(self.llm.predict, context)
        
        return ChatResponse(
            answer=response,
            type="general" if user_info else "general_not_personalized",
            sources_count=len(docs),
            user_info=user_info
        )
    
