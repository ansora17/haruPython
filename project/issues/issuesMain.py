from fastapi import FastAPI
from crawler import crawl_kjcn_article

app = FastAPI()

@app.get("/crawl")
async def crawl(url: str):
    print("STEP 1: received url to crawl:", url)
    result = await crawl_kjcn_article(url)
    print("STEP 2: Finished crawl_kjcn_article, returning result")
    return result

@app.get("/")
def root():
    return {"status": "FastAPI is running"}
