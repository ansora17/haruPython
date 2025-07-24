from fastapi import FastAPI, Header, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx
import base64
# import jwt  # JWT 사용 시 주석 해제
from dotenv import load_dotenv
# 🔄 최신 langchain-openai 패키지 사용
from langchain_openai import OpenAIEmbeddings  
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import openai
import pandas as pd
from pathlib import Path
from typing import List, Optional

# 🚫 미완성 이슈 모듈 주석처리
# from issues.crawler import crawl_kjcn_article

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")  # JWT용 시크릿 키

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str
    user_id: Optional[str] = None  # 개인화 기능을 위한 사용자 ID

knowledge_base = None

@app.on_event("startup")
def startup_event():
    global knowledge_base
    # ✅ 지식베이스 다시 활성화 (데이터 문제 해결됨)
    print("🚀 지식베이스 초기화 시작...")
    data_dir = Path(__file__).parent.parent / "data"
    files = list(data_dir.glob("*.*"))
    knowledge_base = init_knowledge_base(files)
    print("✅ 지식베이스 초기화 완료!")
    
    # # 🚫 임시 비활성화 (더 이상 필요 없음)
    # print("⚠️ 지식베이스 초기화 임시 비활성화 - 서버 구동 테스트")  
    # knowledge_base = None

def process_large_food_csv(file_path: Path, chunk_size: int = 1000) -> List[str]:
    """대용량 음식 CSV 파일을 청크 단위로 처리 (test_langchain.py 로직 적용)"""
    chunks = []
    
    try:
        print(f"대용량 파일 처리 시작: {file_path.name}")
        
        # 🔄 다양한 인코딩 시도 (한글 CSV 파일 대응)
        encodings_to_try = ['utf-8', 'cp949', 'euc-kr', 'latin-1']
        chunk_iter = None
        
        for encoding in encodings_to_try:
            try:
                chunk_iter = pd.read_csv(file_path, chunksize=chunk_size, encoding=encoding)
                print(f"  성공한 인코딩: {encoding}")
                break
            except UnicodeDecodeError:
                print(f"  실패한 인코딩: {encoding}")
                continue
        
        if chunk_iter is None:
            print(f"  모든 인코딩 실패: {file_path.name}")
            return []
        
        for i, chunk_df in enumerate(chunk_iter):
            # 각 청크를 의미있는 텍스트로 변환
            processed_texts = []
            
            for _, row in chunk_df.iterrows():
                try:
                    # 행을 자연어 문장으로 변환
                    row_text = convert_nutrition_row_to_text(row, chunk_df.columns)
                    if row_text:
                        processed_texts.append(row_text)
                except Exception as e:
                    continue
            
            # 청크별로 텍스트 결합
            if processed_texts:
                chunk_text = f"=== {file_path.name} 청크 {i+1} ===\n" + "\n".join(processed_texts)
                chunks.append(chunk_text)
            
            # 진행상황 출력
            if i % 5 == 0:
                print(f"  처리됨: {i * chunk_size}행")
                
    except Exception as e:
        print(f"대용량 CSV 처리 오류 ({file_path.name}): {e}")
        return []
    
    print(f"완료: {file_path.name} - {len(chunks)}개 청크 생성")
    return chunks

def convert_nutrition_row_to_text(row, columns) -> str:
    """영양성분 데이터 행을 자연어 문장으로 변환"""
    
    # 컬럼명에서 주요 정보 추출
    food_name = None
    nutrition_info = {}
    
    for col in columns:
        value = row[col]
        if pd.isna(value) or value == '':
            continue
            
        col_lower = str(col).lower()
        
        # 음식명 추출 (첫 번째 컬럼이나 '명' 포함 컬럼)
        if food_name is None and ('명' in col or col == columns[0]):
            food_name = str(value)
        
        # 영양성분 추출
        elif any(keyword in col_lower for keyword in ['칼로리', '열량', 'kcal']):
            nutrition_info['칼로리'] = value
        elif any(keyword in col_lower for keyword in ['단백질', 'protein']):
            nutrition_info['단백질'] = value
        elif any(keyword in col_lower for keyword in ['탄수화물', 'carb']):
            nutrition_info['탄수화물'] = value
        elif any(keyword in col_lower for keyword in ['지방', 'fat']):
            nutrition_info['지방'] = value
        elif any(keyword in col_lower for keyword in ['나트륨', 'sodium']):
            nutrition_info['나트륨'] = value
        elif any(keyword in col_lower for keyword in ['당류', '당분', 'sugar']):
            nutrition_info['당류'] = value
        elif any(keyword in col_lower for keyword in ['포화지방']):
            nutrition_info['포화지방'] = value
    
    # 자연어 문장 생성
    if not food_name:
        return ""
    
    text_parts = [f"{food_name}"]
    
    if nutrition_info:
        nutrition_parts = []
        for key, value in nutrition_info.items():
            if value and str(value) != 'nan':
                nutrition_parts.append(f"{key} {value}")
        
        if nutrition_parts:
            text_parts.append("영양성분: " + ", ".join(nutrition_parts))
    
    return " - ".join(text_parts) + "."

def init_knowledge_base(file_paths: List[str]):
    """개선된 지식베이스 초기화 (test_langchain.py 방식 적용)"""
    all_texts = []
    
    for file_path in file_paths:
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
        
        print(f"파일 처리: {file_path.name} ({file_size:.1f}MB)")
        
        # 🚫 대용량 파일 스킵 (임베딩 API 제한 대응)
        if file_size > 10:  # 10MB 이상 파일 제외
            print(f"  ⚠️ 대용량 파일 스킵: {file_path.name} (임베딩 처리 제한)")
            continue
        
        if suffix == ".txt":
            # 기존 텍스트 파일 처리
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                all_texts.append(content)
            except Exception as e:
                print(f"텍스트 파일 오류: {e}")
                
        elif suffix == ".csv":
            # 🔄 소용량 파일만 처리 (인코딩 개선)
            try:
                # 🔄 다양한 인코딩 시도
                encodings_to_try = ['utf-8', 'cp949', 'euc-kr', 'latin-1']
                df = None
                
                for encoding in encodings_to_try:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        print(f"  {file_path.name} 성공한 인코딩: {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is not None:
                    # 🔄 데이터 샘플링으로 크기 줄이기
                    if len(df) > 1000:
                        df = df.sample(n=1000, random_state=42)
                        print(f"  ✂️ 데이터 샘플링: {len(df)}행으로 축소")
                    
                    all_texts.append(df.to_string(index=False))
                else:
                    print(f"  {file_path.name} 모든 인코딩 실패")
                    
            except Exception as e:
                print(f"CSV 파일 오류: {e}")
    
    # CharacterTextSplitter로 최종 청크 분할 (🔄 청크 크기 증가)
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=3000,  # 🔄 증가: 1500 -> 3000
        chunk_overlap=300   # 🔄 증가: 200 -> 300
    )
    
    final_chunks = []
    for text in all_texts:
        if text:
            text_chunks = text_splitter.split_text(text)
            final_chunks.extend(text_chunks)
    
    print(f"최종 청크 수: {len(final_chunks)}")
    
    # 🔄 청크 수 제한 (OpenAI API 제한 대응)
    if len(final_chunks) > 2000:
        final_chunks = final_chunks[:2000]
        print(f"  ✂️ 청크 수 제한: {len(final_chunks)}개로 축소")
    
    # OpenAI Embeddings로 벡터화 (🔄 배치 크기 조정)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        chunk_size=100  # 🔄 기본값(1000)에서 100으로 축소
    )
    return FAISS.from_texts(final_chunks, embeddings)

def detect_command(question: str) -> tuple[bool, str]:
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
    import re
    
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

# JWT 토큰에서 user_id 추출 함수 (주석처리)
# def extract_user_id_from_token(authorization: str = None) -> str:
#     """JWT 토큰에서 user_id 추출"""
#     if not authorization or not authorization.startswith("Bearer "):
#         return None
#     
#     try:
#         token = authorization.split(" ")[1]
#         payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
#         return payload.get("user_id")
#     except Exception as e:
#         print(f"토큰 디코딩 오류: {e}")
#         return None

@app.post("/ask")
async def ask_question(
    request: Question,
    # authorization: str = Header(None)  # JWT 토큰용 (주석처리)
):
    global knowledge_base
    
    # # 🚫 지식베이스가 비활성화된 경우 처리
    # if knowledge_base is None:
    #     return {
    #         "answer": "⚠️ 현재 지식베이스가 초기화되지 않았습니다...", 
    #         "type": "system_message",
    #         "status": "knowledge_base_disabled"
    #     }
    
    # JWT 토큰에서 user_id 추출 (주석처리)
    # user_id = extract_user_id_from_token(authorization)
    # if not user_id:
    #     user_id = request.user_id  # 토큰이 없으면 request에서 가져오기
    
    # 임시: request에서 user_id 사용 (개발용)
    user_id = request.user_id or "default_user"
    
    # 1단계: 명령어 감지
    is_command, command_type = detect_command(request.question)
    
    if is_command:
        # 트랙 2: 개인화된 API 기반 답변 (기존 코드 유지)
        print(f"개인화 명령어 감지: {command_type}")
        
        # 외부 API에서 개인 데이터 가져오기
        api_data = await call_external_api(command_type, user_id)
        
        if "error" in api_data:
            return {"answer": "죄송합니다. 개인 데이터를 불러올 수 없습니다. 잠시 후 다시 시도해주세요."}
        
        # 개인화된 프롬프트 생성 (추후 구현)
        # personalized_prompt = get_personalized_prompt(command_type, api_data, request.question)
        
        # 임시 응답
        return {
            "answer": "개인화 기능은 곧 구현될 예정입니다.",
            "type": "personalized",
            "command": command_type
        }
    
    # elif detect_food_question(request.question):
    #     # �� 음식/영양성분 기능 임시 비활성화 (데이터 오류로 인해)
    #     return {
    #         "answer": "⚠️ 음식/영양성분 데이터에 문제가 있어 현재 해당 기능이 비활성화되었습니다. 일반 운동 관련 질문은 계속 이용 가능합니다.", 
    #         "type": "food_disabled",
    #         "status": "food_feature_disabled"
    #     }
        
    #     # # 🚫 트랙 3: 음식/영양성분 기반 답변 (임시 주석처리)
    #     # print("음식/영양성분 질문으로 처리")
    #     # 
    #     # docs = knowledge_base.similarity_search(request.question, k=4)
    #     # base_profile = await get_user_profile(user_id)
    #     # user_info = extract_user_info(request.question, base_profile)
    #     # 
    #     # # 음식 전용 프롬프트
    #     # food_prompt = f"""
    # # You are a personalized nutrition expert and dietitian. Based on the provided data, please give optimized nutritional advice to users in a friendly and warm manner like a close friend.
    # # 
    # # Question: {request.question}
    # # 
    # # User Information:
    # # - Age: {user_info['age']} years old
    # # - Gender: {user_info['gender']}  
    # # - Weight: {user_info['weight']}kg
    # # 
    # # When answering, please MUST include the following:
    # # 1. Accurate nutritional analysis based on the food data
    # # 2. Calorie and macronutrient breakdown (carbs, protein, fat)
    # # 3. Health benefits or concerns about this food
    # # 4. Recommendations considering user's profile (age, gender, weight)
    # # 5. Serving size suggestions or alternatives if needed
    # # 6. Please summarize it in 4 lines or less
    # # 
    # # IMPORTANT: Please respond in Korean language only. All answers must be in Korean.
    # # 
    # # Related nutrition data:
    # # """
    # #         
    # #         llm = ChatOpenAI(
    # #             model="gpt-3.5-turbo", 
    # #             temperature=0.1,
    # #             max_tokens=3000,
    # #         )
    # # 
    # #         context = food_prompt + "\n".join([doc.page_content for doc in docs])
    # #         response = await run_in_threadpool(llm.predict, context)
    # #         
    # #         return {
    # #             "answer": response, 
    # #             "type": "nutrition",
    # #             "sources_count": len(docs),
    # #             "user_info": user_info
    # #         }
    
    else:
        # 트랙 1: 기존 지식 베이스 기반 답변
        print("일반 질문으로 처리")
        print(f"사용 중인 user_id: {user_id}")
        
        docs = knowledge_base.similarity_search(request.question, k=4)
        
        # 🆕 외부 API에서 사용자 프로필 조회
        base_profile = await get_user_profile(user_id)
        print(f"사용자 기본 프로필: {base_profile}")
        
        # 🆕 질문에서 추가 정보 추출하여 프로필 업데이트
        user_info = extract_user_info(request.question, base_profile)
        print(f"최종 사용자 정보: {user_info}")
        
        # 운동 전용 프롬프트
        enhanced_prompt = f"""
You are a personalized fitness expert. Based on the provided data, please give optimized answers to users in a friendly and warm manner like a close friend.

Question: {request.question}

User Information:
- Age: {user_info['age']} years old
- Gender: {user_info['gender']}  
- Weight: {user_info['weight']}kg

When answering, please MUST include the following:
1. Accurate calorie calculations considering the user's weight ({user_info['weight']}kg)
2. Specific exercise duration and methods
3. Provide 3-5 exercise options
4. Offer choices by exercise intensity level
5. Please summarize it in 4 lines or less


IMPORTANT: Please respond in Korean language only. All answers must be in Korean.

Related exercise data:
"""
        
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0,
            max_tokens=3000,
        )

        # 컨텍스트와 함께 답변 생성
        context = enhanced_prompt + "\n".join([doc.page_content for doc in docs])
        response = await run_in_threadpool(llm.predict, context)
        
        return {
            "answer": response, 
            "type": "general",
            "sources_count": len(docs),
            "user_info": user_info,
            "base_profile_source": "external_api" if user_id != "default_user" else "default",
            # "authenticated": user_id != "default_user"  # JWT 활성화 시 사용
        }

# JWT 활성화 시 사용할 로그인 엔드포인트 (주석처리)
# @app.post("/login")
# async def login(username: str, password: str):
#     # 사용자 인증 로직
#     if authenticate_user(username, password):  # 실제 인증 함수 구현 필요
#         payload = {
#             "user_id": username,
#             "exp": datetime.utcnow() + timedelta(hours=24)
#         }
#         token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
#         return {"access_token": token, "token_type": "bearer"}
#     else:
#         raise HTTPException(status_code=401, detail="Invalid credentials")

# 새로운 엔드포인트: 사용 가능한 명령어 리스트
@app.get("/commands")
async def get_available_commands():
    return {
        "commands": [
            {"command": "/음식", "description": "개인 식사 기록 분석"},
            {"command": "/식사", "description": "최근 식사 패턴 분석"}, 
            {"command": "/식단", "description": "식단 기록 조회"},
            {"command": "/칼로리", "description": "칼로리 섭취 분석"},
            {"command": "/오늘음식", "description": "오늘 섭취한 음식 분석"},
            {"command": "/어제음식", "description": "어제 섭취한 음식 분석"}
        ]
    }

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    return {"status": "healthy", "knowledge_base_loaded": knowledge_base is not None}

@app.get("/crawl")
async def crawl(url: str):
    print("STEP 1: received url to crawl:", url)
    # result = await crawl_kjcn_article(url) # 주석 처리된 모듈 사용 시 오류 발생
    print("STEP 2: Finished crawl_kjcn_article, returning result")
    return {"message": "웹 크롤링 기능은 현재 미완성되어 사용할 수 없습니다."}

# 🍽️ 이미지 기반 음식 분석 기능
def encode_image(file: UploadFile):
    """이미지 파일을 base64로 인코딩"""
    file.file.seek(0)  # 파일 포인터를 시작으로 이동
    content = file.file.read()
    return base64.b64encode(content).decode("utf-8")

@app.post("/api/food/analyze")
async def analyze_food_image(file: UploadFile = File(...)):
    """음식 이미지를 분석하여 영양성분 정보 제공"""
    try:
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
Provide the following information:

- Dish name
- exact calories (in kcal)
- carbohydrates in the food(grams)
- protein in the food(grams)
- fat in the food(grams)
- Sodium in this food(grams)
- Dietary fiber in that food(grams)
- Number of foods and total amount (grams)

⚠ IMPORTANT: Your response must be written in Korean at the end

Format your response exactly like this:

- 요리명: (dish name in Korean)
- 칼로리: (exact calories in kcal)
- 탄수화물: (carbohydrates in the food(grams))
- 단백질: (protein in the food(grams))
- 지방: (fat in the food(grams))
- 나트륨: (Sodium in this food(grams))
- 식이섬유: (Dietary fiber in that food(grams))
- 총량: (Number of foods and total amount (grams))
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

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=300
        )
        
        return {
            "result": response.choices[0].message.content,
            "type": "image_analysis",
            "model": "gpt-4-turbo"
        }
        
    except Exception as e:
        return {"error": f"이미지 분석 중 오류가 발생했습니다: {str(e)}"}

@app.get("/")
def root():
    return {
        "status": "FastAPI is running",
        "features": [
            "운동 & 영양 챗봇 (/ask)",
            "음식 이미지 분석 (/api/food/analyze)",
            "웹 크롤링 (/crawl)",
            "시스템 상태 (/health)"
        ]
    }

    
