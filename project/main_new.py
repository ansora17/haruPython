"""
FastAPI 메인 애플리케이션
라우터 등록 및 기본 설정만 담당
"""

#python -m uvicorn main_new:app --reload --host 0.0.0.0 --port 8000
#http://localhost:8000/docs

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 라우터들 import
from routers import chatbot_router, image_router, system_router
from routers.chatbot import initialize_chatbot

# 환경변수 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI(
    title="운동 & 영양 챗봇 API",
    description="운동 및 영양성분 관련 질문답변 서비스",
    version="2.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(system_router)     # /, /health, /crawl
app.include_router(chatbot_router)    # /chatbot/*
app.include_router(image_router)      # /api/food/*

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 초기화"""
    print("🚀 애플리케이션 시작...")
    
    # 챗봇 초기화
    try:
        initialize_chatbot()
        print("✅ 모든 서비스 초기화 완료!")
    except Exception as e:
        print(f"⚠️ 초기화 중 오류 발생: {e}")
        print("🔄 서버는 계속 실행되지만 일부 기능이 제한될 수 있습니다.")

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 정리"""
    print("🛑 애플리케이션 종료...")

# 직접 실행 가능하도록 하는 코드
if __name__ == "__main__":
    import uvicorn
    print("🚀 FastAPI 서버를 시작합니다...")
    print("📍 서버 주소: http://localhost:8000")
    print("📖 API 문서: http://localhost:8000/docs")
    print("🛑 종료: Ctrl+C")
    
    uvicorn.run(
        "main_new:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )