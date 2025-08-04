"""
쿠키에서 사용자 정보를 추출하는 유틸리티
"""

import json
from urllib.parse import unquote
from typing import Optional, Dict, Any
from .models import UserProfile


def parse_user_cookie(cookie_value: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    쿠키에서 사용자 정보 파싱
    
    Args:
        cookie_value: 쿠키 값 (JSON 문자열)
        
    Returns:
        사용자 정보 딕셔너리 또는 None
    """
    
    if not cookie_value:
        return None
    
    try:
        
        # URL 디코딩 후 JSON 파싱
        decoded_cookie = unquote(cookie_value)
        
        user_data = json.loads(decoded_cookie)
        
        # 필수 필드 검증 및 타입 변환
        result = {
            "id": user_data.get("id", "unknown_user"),
            "age": int(user_data.get("age", 25)),
            "gender": user_data.get("gender", "여성"),
            "weight": int(user_data.get("weight", 55)),
            "height": int(user_data.get("height")) if user_data.get("height") else None,
            "activity_level": user_data.get("activity_level", "보통")
        }
        return result
        
    except json.JSONDecodeError as e:
        return None
    except (ValueError, KeyError) as e:
        return None


def create_user_profile_from_cookie(cookie_value: Optional[str]) -> Optional[UserProfile]:
    """
    쿠키에서 UserProfile 객체 생성
    
    Args:
        cookie_value: 쿠키 값
        
    Returns:
        UserProfile 객체 또는 None
    """
    user_data = parse_user_cookie(cookie_value)
    
    if not user_data:
        return None
        
    return UserProfile(
        age=user_data["age"],
        gender=user_data["gender"],
        weight=user_data["weight"],
        height=user_data["height"],
        activity_level=user_data["activity_level"]
    )


def parse_member_cookie(cookie_value: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    프론트엔드 'member' 쿠키에서 사용자 정보 파싱 및 필드 매핑
    
    Args:
        cookie_value: member 쿠키 값
        
    Returns:
        백엔드 형식으로 변환된 사용자 정보 딕셔너리 또는 None
    """
    
    if not cookie_value:
        return None
    
    try:
        
        # URL 디코딩 후 JSON 파싱
        decoded_cookie = unquote(cookie_value)
        
        member_data = json.loads(decoded_cookie)
        
        # 🎯 프론트엔드 필드를 백엔드 필드로 매핑
        # memberId -> id, name -> age 계산 등
        
        # age 계산 (name 필드에서 나이 추출하거나 기본값 사용)
        age = 25  # 기본값
        if member_data.get("name"):
            # name에서 나이 추출 시도 (예: "김철수(28세)" 형태라면)
            import re
            age_match = re.search(r'\((\d+)세\)', member_data.get("name", ""))
            if age_match:
                age = int(age_match.group(1))
        
        # gender 추론 (기본값 또는 별도 필드가 있다면 사용)
        gender = "여성"  # 기본값
        # 만약 member_data에 gender 필드가 있다면: member_data.get("gender", "여성")
        
        # activityLevel 매핑
        activity_mapping = {
            "LOW": "낮음",
            "MEDIUM": "보통", 
            "HIGH": "높음",
            "낮음": "낮음",
            "보통": "보통",
            "높음": "높음"
        }
        
        result = {
            "id": member_data.get("memberId") or member_data.get("email", "unknown_user"),
            "age": age,
            "gender": gender,
            "weight": int(member_data.get("weight")) if member_data.get("weight") else None,
            "height": int(member_data.get("height")) if member_data.get("height") else None,
            "activity_level": activity_mapping.get(member_data.get("activityLevel"), "보통")
        }
        
        return result
        
    except json.JSONDecodeError as e:
        return None
    except (ValueError, KeyError) as e:
        return None


def get_user_info_dict(cookie_value: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    쿠키에서 사용자 정보를 딕셔너리로 반환
    
    Args:
        cookie_value: 쿠키 값
        
    Returns:
        사용자 정보 딕셔너리 또는 None (기본값 없음)
    """
    return parse_user_cookie(cookie_value)


def get_user_info_from_cookies(request) -> Optional[Dict[str, Any]]:
    """
    여러 쿠키에서 사용자 정보를 찾아서 반환
    
    Args:
        request: FastAPI Request 객체
        
    Returns:
        사용자 정보 딕셔너리 또는 None
    """
    # 1순위: user_info 쿠키 (테스트용)
    user_info_cookie = request.cookies.get("user_info")
    if user_info_cookie:
        return get_user_info_dict(user_info_cookie)
    
    # 2순위: member 쿠키 (프론트엔드 실제 쿠키)
    member_cookie = request.cookies.get("member")
    if member_cookie:
        return parse_member_cookie(member_cookie)
    
    return None