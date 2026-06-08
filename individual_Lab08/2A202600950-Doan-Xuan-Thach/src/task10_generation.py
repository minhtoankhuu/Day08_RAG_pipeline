"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
import re
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve

# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence để mô hình trả lời đầy đủ ý mà không quá dài gây loãng thông tin.
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: giúp câu trả lời tự nhiên nhưng vẫn giữ được độ tập trung, tránh lan man.
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần độ chính xác cao và thông tin thực tế (factual), ít tính sáng tạo.
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, dễ quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    left = []
    right = []
    
    # Chia đều các phần tử: chỉ mục chẵn đi vào left, chỉ mục lẻ đi vào right
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            left.append(chunk)
        else:
            right.append(chunk)
            
    # Đảo ngược phần bên phải và ghép lại để phần tử tốt thứ 2 nằm ở cuối cùng
    return left + right[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K, use_reranking: bool = True) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k, use_reranking=use_reranking)
    
    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)
    
    # Step 3: Format context
    context = format_context(reordered)

    api_key = os.getenv("OPENAI_API_KEY", "")

    # Nếu có API Key OpenAI hợp lệ, gọi mô hình OpenAI để sinh câu trả lời
    if api_key and not api_key.startswith("sk-xxx"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            
            answer = response.choices[0].message.content
            return {
                "answer": answer,
                "sources": chunks,
                "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
            }
        except Exception as e:
            print(f"  [WARNING] Lỗi khi gọi OpenAI API ({e}). Kích hoạt Mock Generator...")

    # Fallback/Mock Generator chất lượng cao (Dành cho chạy offline hoặc không có API Key)
    print("  [INFO] Sinh câu trả lời tự động từ ngữ cảnh tìm kiếm...")
    
    if not chunks:
        answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có vì không tìm thấy tài liệu liên quan."
    else:
        # Tạo câu trả lời bằng cách trích lục trực tiếp các ý chính từ chunks tốt nhất
        # và chèn citation dựa trên source tên file
        best_chunk = chunks[0]
        source_name = best_chunk.get("metadata", {}).get("source", "Tài liệu")
        
        # Làm sạch tên file để làm tên trích dẫn ngắn gọn hơn
        citation_name = source_name.replace(".md", "").replace("-", " ").title()
        
        # Lấy 2-3 câu đầu tiên của chunk tốt nhất làm câu trả lời thực tế
        sentences = re.split(r'(?<=[.!?]) +', best_chunk["content"].strip())
        summary_text = " ".join(sentences[:3])
        
        answer = (
            f"Dựa trên tài liệu hệ thống, dưới đây là thông tin trả lời:\n\n"
            f"{summary_text} [{citation_name}].\n\n"
            f"Thông tin chi tiết được tham khảo trực tiếp từ nguồn tài liệu gốc {source_name}."
        )

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
