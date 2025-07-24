from .app.supabase_client import supabase

def is_duplicate(url: str) -> bool:
    try:
        response = supabase.table("summaries").select("id").eq("original_url", url).execute()
        return len(response.data) > 0
    except Exception as e:
        print(f"[DB] Error checking duplication: {e}")
        return False  # Be cautious: assume not duplicate if error

def save_summary_to_db(title: str, summary: str, url: str) -> bool:
    try:
        if is_duplicate(url):
            print("[DB] Duplicate URL found. Skipping insert.")
            return False

        data = {
            "title": title,
            "summary": summary,
            "original_url": url
        }
        response = supabase.table("summaries").insert(data).execute()
        print("[DB] Insert successful:", response.data)
        return True

    except Exception as e:
        print(f"[DB] Error inserting summary: {e}")
        return False