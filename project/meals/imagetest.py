# uvicorn imagetest:app --reload --host 0.0.0.0 --port 8000
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import base64, os, openai
from PIL import Image
import itertools
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OpenAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def encode_image(file: UploadFile):
    """이미지 파일을 base64로 인코딩"""
    try:
        # 파일 포인터를 시작으로 이동
        file.file.seek(0)
        content = file.file.read()
        
        # 파일이 비어있는지 확인
        if not content:
            raise ValueError("업로드된 파일이 비어있습니다")
        
        # 파일 크기 확인 (너무 큰 파일 방지)
        if len(content) > 10 * 1024 * 1024:  # 10MB 제한
            raise ValueError("파일 크기가 너무 큽니다 (최대 10MB)")
        
        # base64 인코딩
        encoded = base64.b64encode(content).decode("utf-8")
        
        # 인코딩된 문자열이 유효한지 확인
        if not encoded:
            raise ValueError("이미지 인코딩에 실패했습니다")
        
        return encoded
    except Exception as e:
        print(f"이미지 인코딩 오류: {e}")
        raise ValueError(f"이미지 인코딩 중 오류가 발생했습니다: {str(e)}")

@app.post("/api/food/analyze")
async def analyze_food(file: UploadFile = File(...)):
    """음식 이미지를 분석하여 영양성분 정보 제공"""
    try:
        # 파일 형식 검증
        if not file.content_type or not file.content_type.startswith('image/'):
            return {
                "success": False,
                "error": "이미지 파일만 업로드 가능합니다",
                "type": "image_analysis"
            }
        
        # 파일 확장자 검증
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in allowed_extensions:
            return {
                "success": False,
                "error": f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(allowed_extensions)}",
                "type": "image_analysis"
            }
        
        # 이미지 인코딩
        try:
            encoded = encode_image(file)
        except ValueError as e:
            return {
                "success": False,
                "error": str(e),
                "type": "image_analysis"
            }
        
        # OpenAI API 호출을 위한 메시지 구성
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
You are a food image analysis expert with deep knowledge in culinary arts. 
Please analyze the food image provided below carefully, considering its appearance, ingredients, and regional characteristics.  

IMPORTANT: Analyze ALL foods visible in the image, no matter how many there are. Each food should be a separate object in the array.

Please provide the analysis in JSON format with the following structure:

For single food:
{
    "foodName": "음식 이름",
    "calories": 숫자값,
    "carbohydrate": 숫자값,
    "protein": 숫자값,
    "fat": 숫자값,
    "sodium": 숫자값,
    "fiber": 숫자값,
    "totalAmount": 숫자값,
    "foodCategory": "한식/중식/일식/양식/분식/음료 중 하나"
}

For multiple foods (2 or more):
[
    {
        "foodName": "음식 이름 1",
        "calories": 숫자값,
        "carbohydrate": 숫자값,
        "protein": 숫자값,
        "fat": 숫자값,
        "sodium": 숫자값,
        "fiber": 숫자값,
        "totalAmount": 숫자값,
        "foodCategory": "한식/중식/일식/양식/분식/음료 중 하나"
    },
    {
        "foodName": "음식 이름 2",
        "calories": 숫자값,
        "carbohydrate": 숫자값,
        "protein": 숫자값,
        "fat": 숫자값,
        "sodium": 숫자값,
        "fiber": 숫자값,
        "totalAmount": 숫자값,
        "foodCategory": "한식/중식/일식/양식/분식/음료 중 하나"
    },
    {
        "foodName": "음식 이름 3",
        "calories": 숫자값,
        "carbohydrate": 숫자값,
        "protein": 숫자값,
        "fat": 숫자값,
        "sodium": 숫자값,
        "fiber": 숫자값,
        "totalAmount": 숫자값,
        "foodCategory": "한식/중식/일식/양식/분식/음료 중 하나"
    }
]

⚠ IMPORTANT: 
1. Return ONLY valid JSON format
2. All numeric values should be numbers (not strings)
3. All text values should be in Korean
4. Do not include any additional text or explanations
5. Make sure all quotes are properly escaped
6. If there's only one food, return a single object. If there are multiple foods, return an array of objects.
7. Include ALL foods visible in the image, even if there are many
8. Each food should be analyzed separately with its own nutritional values
"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{file.content_type};base64,{encoded}"
                        }
                    }
                ]
            }
        ]

        # OpenAI API 호출
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.1
        )
        
        # JSON 응답 파싱
        import json
        import re
        
        content = response.choices[0].message.content.strip()
        print(f"OpenAI 응답: {content}")
        
        # 배열과 객체 모두 처리할 수 있도록 개선
        json_patterns = [
            r'\[.*\]',  # 배열 패턴
            r'\{.*\}',  # 객체 패턴
        ]
        
        for pattern in json_patterns:
            json_match = re.search(pattern, content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    result_json = json.loads(json_str)
                    return {
                        "success": True,
                        "result": result_json,
                        "type": "image_analysis",
                        "model": "gpt-4-turbo"
                    }
                except json.JSONDecodeError as e:
                    print(f"JSON 파싱 오류: {e}")
                    print(f"파싱 시도한 문자열: {json_str}")
                    continue
        
        # 모든 패턴이 실패한 경우
        return {
            "success": False,
            "error": "JSON 형식을 찾을 수 없습니다",
            "result": content,
            "type": "image_analysis",
            "model": "gpt-4-turbo"
        }
    except Exception as e:
        print(f"이미지 분석 중 오류: {e}")
        return {
            "success": False,
            "error": f"이미지 분석 중 오류가 발생했습니다: {str(e)}",
            "type": "image_analysis"
        }

# 텍스트 기반 음식 분석 엔드포인트
@app.post("/api/food/analyze/text")
async def analyze_food_text(request: dict):
    """텍스트로 음식 분석 요청"""
    try:
        food_text = request.get("food_name", "")
        if not food_text:
            return {"error": "음식 이름을 입력해주세요"}
        
        messages = [
            {
                "role": "user",
                "content": f"""
You are a food image analysis expert with deep knowledge in culinary arts. 
Please analyze the food described below and provide nutritional information.

Food: {food_text}

Please provide the analysis in JSON format with the following structure:

{{
    "foodName": "음식 이름",
    "calories": 숫자값,
    "carbohydrate": 숫자값,
    "protein": 숫자값,
    "fat": 숫자값,
    "sodium": 숫자값,
    "fiber": 숫자값,
    "totalAmount": 숫자값,
    "foodCategory": "한식/중식/일식/양식/분식/음료 중 하나"
}}

⚠ IMPORTANT: 
1. Return ONLY valid JSON format
2. All numeric values should be numbers (not strings)
3. All text values should be in Korean
4. Do not include any additional text or explanations
5. Make sure all quotes are properly escaped
"""
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=300,
            temperature=0.1
        )
        
        # JSON 응답 파싱
        import json
        import re
        
        content = response.choices[0].message.content.strip()
        print(f"OpenAI 응답: {content}")  # 디버깅용
        
        # JSON 부분만 추출
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                result_json = json.loads(json_str)
                return {
                    "success": True,
                    "result": result_json,
                    "type": "text_analysis",
                    "model": "gpt-4-turbo"
                }
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {e}")
                print(f"파싱 시도한 문자열: {json_str}")
                return {
                    "success": False,
                    "error": f"JSON 파싱 실패: {str(e)}",
                    "result": content,
                    "type": "text_analysis",
                    "model": "gpt-4-turbo"
                }
        else:
            return {
                "success": False,
                "error": "JSON 형식을 찾을 수 없습니다",
                "result": content,
                "type": "text_analysis",
                "model": "gpt-4-turbo"
            }
    except Exception as e:
        print(f"OpenAI API 오류: {e}")
        return {"error": f"텍스트 분석 중 오류가 발생했습니다: {str(e)}"}

# 테스트용 엔드포인트
@app.get("/test")
async def test_endpoint():
    return {
        "message": "이미지 분석 서버가 정상 작동 중입니다",
        "endpoint": "/api/food/analyze",
        "method": "POST"
    }