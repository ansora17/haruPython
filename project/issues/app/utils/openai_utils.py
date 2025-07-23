import openai
import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트의 .env 파일 로드
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

def summarize_text(text: str, max_tokens: int = 500) -> str:
    """
    OpenAI API를 사용하여 텍스트를 요약하는 함수
    
    Args:
        text: 요약할 텍스트
        max_tokens: 최대 토큰 수
        
    Returns:
        str: 요약된 텍스트
    """
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "다음 텍스트를 한국어로 간결하게 요약해주세요. 핵심 내용만 포함하고 불필요한 부분은 제거해주세요."
                },
                {
                    "role": "user", 
                    "content": text
                }
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"OpenAI API 오류: {e}")
        # API 오류 시 원본 텍스트의 일부를 반환
        return text[:500] + "..." if len(text) > 500 else text 