import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings #백터화
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import pandas as pd
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

data =[
    "파이썬은 매우 인기 있는 프로그래밍 언어입니다.",
    "자연어 처리에서 토큰화는 중요한 단계입니다.",
    "LangChain은 LLM 기반 앱을 쉽게 만들도록 도와줍니다.",
    "벡터 DB는 유사한 문장을 빠르게 검색할 수 있게 합니다."
]

df=pd.DataFrame(data, columns=["text"])

#print(df)

def embedding_func(text):
    embedding_result = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return embedding_result.data[0].embedding

df['embedding'] = df['text'].apply(embedding_func)

print(df)