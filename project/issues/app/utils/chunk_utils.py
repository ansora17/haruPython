def chunk_text(text: str, max_length: int = 2000, overlap: int = 200) -> list:
    """
    긴 텍스트를 작은 청크로 나누는 함수
    
    Args:
        text: 분할할 텍스트
        max_length: 각 청크의 최대 길이
        overlap: 청크 간 겹치는 부분의 길이
    
    Returns:
        list: 분할된 텍스트 청크들의 리스트
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        
        # 문장 경계에서 자르기 위해 마지막 마침표나 줄바꿈을 찾음
        if end < len(text):
            # 마지막 마침표나 줄바꿈 위치 찾기
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            
            # 더 가까운 구분점 선택
            if last_period > start or last_newline > start:
                end = max(last_period, last_newline) + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # 다음 시작점 설정 (overlap 고려)
        start = max(start + 1, end - overlap)
        
        # 무한 루프 방지
        if start >= len(text):
            break
    
    return chunks 