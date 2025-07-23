import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트의 .env 파일 로드
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

try:
    from supabase import create_client, Client
    
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
    
    if SUPABASE_URL and SUPABASE_KEY:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase 클라이언트 초기화 성공")
    else:
        print("⚠️ Supabase 환경변수가 설정되지 않았습니다.")
        print("SUPABASE_URL과 SUPABASE_ANON_KEY를 .env 파일에 추가해주세요.")
        
        # 더미 클라이언트 (테스트용)
        class DummySupabase:
            def table(self, table_name):
                return DummyTable()
        
        class DummyTable:
            def select(self, *args):
                return self
            def insert(self, data):
                return self
            def eq(self, *args):
                return self
            def execute(self):
                return type('obj', (object,), {'data': []})()
        
        supabase = DummySupabase()
        
except ImportError:
    print("⚠️ supabase 패키지가 설치되지 않았습니다.")
    print("pip install supabase 명령으로 설치해주세요.")
    
    # 더미 클라이언트
    class DummySupabase:
        def table(self, table_name):
            return DummyTable()
    
    class DummyTable:
        def select(self, *args):
            return self
        def insert(self, data):
            return self
        def eq(self, *args):
            return self
        def execute(self):
            return type('obj', (object,), {'data': []})()
    
    supabase = DummySupabase() 