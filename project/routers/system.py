"""
시스템 관련 API 라우터
헬스체크, 크롤링, 기본 정보 등
"""

from fastapi import APIRouter

router = APIRouter(tags=["system"])

@router.get("/")
def root():
    """메인 페이지 - 서버 상태 및 기능 목록"""
    return {
        "status": "FastAPI is running",
        "features": [
            "운동 & 영양 챗봇 (/chatbot/ask)",
            "음식 이미지 분석 (/api/food/analyze)",
            "웹 크롤링 (/crawl)",
            "시스템 상태 (/health)"
        ]
    }

@router.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    return {"status": "healthy", "service": "chatbot_api"}

@router.get("/crawl")
async def crawl(url: str):
    """웹 크롤링 (현재 비활성화)"""
    print("STEP 1: received url to crawl:", url)
    # result = await crawl_kjcn_article(url) # 주석 처리된 모듈 사용 시 오류 발생
    print("STEP 2: Finished crawl_kjcn_article, returning result")
    return {"message": "웹 크롤링 기능은 현재 미완성되어 사용할 수 없습니다."}