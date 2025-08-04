"""
이미지 분석 관련 API 라우터
음식 이미지 분석 기능
"""

import os
import base64
import openai
import json
import re
from fastapi import APIRouter, UploadFile, File
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/api/food", tags=["image"])

def encode_image(file: UploadFile):
    """이미지 파일을 base64로 인코딩"""
    try:
        file.file.seek(0)
        content = file.file.read()
        
        if not content:
            raise ValueError("업로드된 파일이 비어있습니다")
        
        if len(content) > 10 * 1024 * 1024:
            raise ValueError("파일 크기가 너무 큽니다 (최대 10MB)")
        
        encoded = base64.b64encode(content).decode("utf-8")
        
        if not encoded:
            raise ValueError("이미지 인코딩에 실패했습니다")
        
        return encoded
        
    except Exception as e:
        raise ValueError(f"이미지 인코딩 중 오류가 발생했습니다: {str(e)}")

@router.post("/analyze")
async def analyze_food_image(file: UploadFile = File(...)):
    """음식 이미지를 분석하여 영양성분 정보 제공"""
    try:
        # OpenAI API 키 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {
                "success": False,
                "error": "OpenAI API 키가 설정되지 않았습니다",
                "type": "config_error"
            }
        
        # 파일 유효성 검증
        if not file.content_type or not file.content_type.startswith('image/'):
            return {
                "success": False,
                "error": "이미지 파일만 업로드 가능합니다",
                "type": "file_validation_error"
            }
        
        # 파일 확장자 검증
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        file_extension = os.path.splitext(file.filename or "")[1].lower()
        if file_extension not in allowed_extensions:
            return {
                "success": False,
                "error": f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(allowed_extensions)}",
                "type": "file_validation_error"
            }
        
        # 이미지 인코딩
        try:
            encoded = encode_image(file)
        except ValueError as e:
            return {
                "success": False,
                "error": str(e),
                "type": "encoding_error"
            }
        
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": """
You are a food image analysis expert. Analyze the food image and provide nutritional information.

Please provide the analysis in JSON format:
{
    "foodName": "음식 이름",
    "quantity": 1,
    "calories": 숫자값,
    "carbohydrate": 숫자값,
    "protein": 숫자값,
    "fat": 숫자값,
    "sodium": 숫자값,
    "fiber": 숫자값,
    "totalAmount": 숫자값,
    "foodCategory": "한식"
}

IMPORTANT: 
- Return ONLY valid JSON format
- All text in Korean
- foodCategory must be EXACTLY ONE of: "한식", "중식", "일식", "양식", "분식", "음료"
"""
            }, {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"}
            }]
        }]
        
        # OpenAI API 호출
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        
        # JSON 파싱
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                
                # foodCategory 유효성 검사
                valid_categories = ["한식", "중식", "일식", "양식", "분식", "음료"]
                if result.get("foodCategory") not in valid_categories:
                    result["foodCategory"] = "기타"
                
                return {
                    "success": True,
                    "result": result,
                    "type": "image_analysis",
                    "model": "gpt-4o",
                    "filename": file.filename
                }
            except json.JSONDecodeError:
                pass
        
        return {
            "success": False,
            "error": "AI 응답을 파싱할 수 없습니다",
            "type": "parse_error"
        }
        
    except openai.OpenAIError as e:
        if "insufficient_quota" in str(e) or "429" in str(e):
            # API 할당량 초과 시 기본 응답
            return {
                "success": True,
                "result": {
                    "foodName": "음식 (할당량 초과)",
                    "quantity": 1,
                    "calories": 300,
                    "carbohydrate": 20,
                    "protein": 15,
                    "fat": 10,
                    "sodium": 2.0,
                    "fiber": 3,
                    "totalAmount": 300,
                    "foodCategory": "기타"
                },
                "type": "quota_exceeded",
                "note": "OpenAI API 할당량 초과로 기본값 반환"
            }
        else:
            return {
                "success": False,
                "error": f"AI 분석 중 오류가 발생했습니다: {str(e)}",
                "type": "openai_api_error"
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"이미지 분석 중 오류가 발생했습니다: {str(e)}",
            "type": "general_error"
        }