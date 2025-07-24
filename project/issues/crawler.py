import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

from .app.utils.chunk_utils import chunk_text
from .app.utils.openai_utils import summarize_text
from .app.supabase_client import supabase
from .db import save_summary_to_db

# 프로젝트 루트의 .env 파일 로드
from pathlib import Path
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

async def crawl_kjcn_article(url: str) -> dict:
    print("STEP 2: Inside crawl_kjcn_article")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title (optional: fallback logic can be improved)
        title_tag = soup.select_one("div#container.article div.titArea")
        title = title_tag.get_text(strip=True) if title_tag else "제목 없음"

        # Extract body sections under <div class="contents"><div class="articleCon">
        article_container = soup.select_one("div.contents div.articleCon")
        if not article_container:
            return {"error": "본문을 찾을 수 없습니다.", "original_url": url}

        sections = []
        for header in article_container.select("h4.link-target"):
            section_title = header.get_text(strip=True)
            next_dd = header.find_next_sibling("dd")
            if next_dd:
                section_text = next_dd.get_text(strip=True)
                sections.append(f"[{section_title}]\n{section_text}")

        full_text = "\n\n".join(sections)

        if not full_text:
            return {"error": "본문 내용이 비어 있습니다.", "original_url": url}

        print("STEP 3: Finished extracting text, starting summarization")

        chunks = chunk_text(full_text)
        summaries = [summarize_text(chunk) for chunk in chunks]
        full_summary = "\n".join(summaries)

        print("STEP 4: Finished summarization")

        # after a project in Supabase is created
        save_summary_to_db(title, full_summary, url)

        return {
            "title": title,
            "summary": full_summary,
            "original_url": url
        }