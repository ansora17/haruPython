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
    content = file.file.read()
    return base64.b64encode(content).decode("utf-8")

@app.post("/api/food/analyze")
async def analyze_food(file: UploadFile = File(...)):
    encoded = encode_image(file)

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
                        "url": f"data:image/png;base64,{encoded}"
                    }
                }
            ]
        }
    ]

    try:
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
        print(f"OpenAI 응답: {content}")  # 디버깅용
        
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
        print(f"OpenAI API 오류: {e}")
        return {"error": f"이미지 분석 중 오류가 발생했습니다: {str(e)}"}

# 테스트용 엔드포인트
@app.get("/test")
async def test_endpoint():
    return {
        "message": "이미지 분석 서버가 정상 작동 중입니다",
        "endpoint": "/api/food/analyze",
        "method": "POST"
    }