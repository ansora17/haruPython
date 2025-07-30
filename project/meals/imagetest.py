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
If there are more than two food photos, please add the two values together. 
Please analyze the food image provided below carefully, considering its appearance, ingredients, and regional characteristics.  

Please provide the analysis in JSON format with the following structure:
{
    "foodName": "음식 이름",
    "calories": 숫자값,
    "carbohydrates": 숫자값,
    "protein": 숫자값,
    "fat": 숫자값,
    "sodium": 숫자값,
    "fiber": 숫자값,
    "total_amount": 숫자값,
    "food_category": "한식/중식/일식/양식/분식/음료 중 하나"
}

⚠ IMPORTANT: 
1. Return ONLY valid JSON format
2. All numeric values should be numbers (not strings)
3. All text values should be in Korean
4. Do not include any additional text or explanations
5. Make sure all quotes are properly escaped
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
        
        # JSON 부분만 추출 (중괄호로 시작하고 끝나는 부분)
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
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
                return {
                    "success": False,
                    "error": f"JSON 파싱 실패: {str(e)}",
                    "result": content,
                    "type": "image_analysis",
                    "model": "gpt-4-turbo"
                }
        else:
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