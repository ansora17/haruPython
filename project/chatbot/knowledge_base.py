"""
지식베이스 관리 모듈
데이터 로딩, 처리, 벡터화 담당
"""

import pandas as pd
from pathlib import Path
from typing import List
from langchain_openai import OpenAIEmbeddings  
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter


class KnowledgeBaseManager:
    """지식베이스 관리 클래스"""
    
    def __init__(self):
        self.knowledge_base = None
    
    def process_large_food_csv(self, file_path: Path, chunk_size: int = 1000) -> List[str]:
        """대용량 음식 CSV 파일을 청크 단위로 처리"""
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
                        row_text = self.convert_nutrition_row_to_text(row, chunk_df.columns)
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

    def convert_nutrition_row_to_text(self, row, columns) -> str:
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

    def init_knowledge_base(self, file_paths: List[str]):
        """개선된 지식베이스 초기화"""
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
        
        self.knowledge_base = FAISS.from_texts(final_chunks, embeddings)
        return self.knowledge_base

    def get_knowledge_base(self):
        """현재 지식베이스 반환"""
        return self.knowledge_base
    
    def search(self, query: str, k: int = 4):
        """지식베이스에서 유사한 문서 검색"""
        if self.knowledge_base is None:
            return []
        return self.knowledge_base.similarity_search(query, k=k)