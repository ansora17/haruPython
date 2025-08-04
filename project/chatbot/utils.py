"""
챗봇 유틸리티 함수들
질문 감지, 사용자 프로필, 운동 계산 등
"""

import re
import httpx
from typing import Dict, Tuple

def detect_command(question: str) -> Tuple[bool, str]:
    """명령어 감지 함수"""
    command_mappings = {
        "/음식": "food",
        "/식사": "meal", 
        "/식단": "diet",
        "/칼로리": "calorie_intake",
        "/오늘음식": "today_food",
        "/어제음식": "yesterday_food"
    }
    
    for command, command_type in command_mappings.items():
        if question.strip().startswith(command):
            return True, command_type
    
    return False, ""

def detect_food_question(question: str) -> bool:
    """음식/칼로리/영양성분 관련 질문 감지 (운동 관련 칼로리 제외)"""
    
    # 운동 관련 칼로리 키워드 (음식 아님)
    exercise_calorie_keywords = [
        "칼로리 태우", "칼로리 소진", "칼로리 소모", "칼로리 태울", 
        "칼로리 소비", "운동", "태우려면", "소진하려면", "소모하려면"
    ]
    
    # 운동 관련이면 False 반환
    if any(keyword in question for keyword in exercise_calorie_keywords):
        return False
    
    # 음식 관련 키워드
    food_keywords = [
        "영양성분", "영양소", "단백질", "탄수화물", "지방", 
        "비타민", "미네랄", "나트륨", "음식", "식품", "요리",
        "먹으면", "섭취", "영양", "성분", "포함", "들어있",
        "당분", "당류", "함량", "시리얼", "김치", "된장", "라면", 
        "밥", "고기", "생선", "과일", "야채", "우유", "계란",
        "칼로리가", "칼로리는", "칼로리 함량", "칼로리 섭취"
    ]
    
    return any(keyword in question for keyword in food_keywords)

async def call_external_api(command_type: str, user_id: str = None) -> dict:
    """외부 API 호출 함수"""
    external_api_url = "http://localhost:8000"  # 실제 API URL로 변경
    
    # API 엔드포인트 매핑
    endpoint_mapping = {
        "food": "/image",
        "meal": "/meals", 
        "diet": "/diet_records",
        "calorie_intake": "/calorie_summary",
        "today_food": "/today_meals",
        "yesterday_food": "/yesterday_meals"
    }
    
    endpoint = endpoint_mapping.get(command_type, "/image")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{external_api_url}{endpoint}",
                headers={"Content-Type": "application/json"},
                json={
                    "query": command_type,
                    "user_id": user_id
                },
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"외부 API 호출 오류: {e}")
        return {"error": "외부 데이터를 가져올 수 없습니다."}

async def get_user_profile(user_id: str = None) -> dict:
    """외부 API에서 사용자 프로필 조회"""
    if not user_id:
        # user_id가 없으면 기본 프로필 반환
        return {
            "age": 25,
            "gender": "여성", 
            "weight": 55
        }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://localhost:8001/user/profile/{user_id}",  # 사용자 프로필 API
                timeout=10.0
            )
            response.raise_for_status()
            profile_data = response.json()
            
            # API 응답 형식에 맞게 변환
            return {
                "age": profile_data.get("age", 25),
                "gender": profile_data.get("gender", "여성"),
                "weight": profile_data.get("weight", 55)
            }
    except Exception as e:
        print(f"사용자 프로필 조회 오류: {e}")
        # 오류 시 기본 프로필 반환
        return {
            "age": 25,
            "gender": "여성", 
            "weight": 55
        }

def extract_user_info(question: str, base_profile: dict) -> dict:
    """질문에서 사용자 정보 추출하여 기본 프로필에 덮어쓰기"""
    
    # 기본 프로필로 시작 (외부 API에서 가져온 데이터)
    user_info = base_profile.copy()
    
    # 질문에서 다른 정보가 있으면 덮어쓰기
    age_match = re.search(r'(\d+)대', question)
    if age_match:
        user_info["age"] = age_match.group(1) + "대"
    
    # 성별 추출
    if "여성" in question or "여자" in question:
        user_info["gender"] = "여성"
    elif "남성" in question or "남자" in question:
        user_info["gender"] = "남성"
    
    # 체중 추출
    weight_match = re.search(r'(\d+)\s*(?:kg|킬로|키로)', question)
    if weight_match:
        user_info["weight"] = int(weight_match.group(1))
    
    return user_info

def calculate_exercise_time(target_calories: int, weight_kg: int = 60) -> str:
    """목표 칼로리에 맞는 정확한 운동 시간 계산"""
    
    # exercise_dataset.csv 기반 계산 (70kg 기준 데이터를 weight로 조정)
    exercises = [
        ("러닝(일반)", 563, "중강도"),
        ("러닝(빠름)", 950, "고강도"), 
        ("자전거타기", 563, "중강도"),
        ("줄넘기(보통)", 704, "고강도"),
        ("줄넘기(빠름)", 844, "고강도"),
        ("수영", 563, "중강도"),
        ("에어로빅", 493, "중강도"),
        ("조깅", 422, "저강도")
    ]
    
    result = []
    for name, cal_per_hour_70kg, intensity in exercises:
        # 체중 보정 (70kg 기준 데이터를 사용자 체중으로 조정)
        adjusted_cal = (cal_per_hour_70kg * weight_kg) / 55
        time_needed = (target_calories / adjusted_cal) * 60  # 분 단위
        
        if time_needed <= 90:  # 90분 이내 운동만 추천
            result.append(f"• {name} ({intensity}): {time_needed:.0f}분")
    
    return "\n".join(result[:5])  # 상위 5개만 추천