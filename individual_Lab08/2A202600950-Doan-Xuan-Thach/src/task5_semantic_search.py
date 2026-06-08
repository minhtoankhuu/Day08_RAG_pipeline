"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import pickle
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

VECTORSTORE_PATH = Path(__file__).parent.parent / "data" / "vectorstore.pkl"
EMBEDDING_MODEL = "BAAI/bge-m3"

# Sử dụng biến toàn cục để cache dữ liệu, giúp các truy vấn liên tục không phải load lại model/file
_model = None
_chunks = None


def get_model():
    """Tải và trả về đối tượng SentenceTransformer (singleton)."""
    global _model
    if _model is None:
        # Load mô hình từ cache local
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_chunks():
    """Tải và trả về danh sách chunks từ vector store cục bộ (singleton)."""
    global _chunks
    if _chunks is None:
        if not VECTORSTORE_PATH.exists():
            return []
        with open(VECTORSTORE_PATH, "rb") as f:
            _chunks = pickle.load(f)
    return _chunks


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity (Cosine Similarity).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    chunks = get_chunks()
    if not chunks:
        print("  [WARNING] Vector store chưa có dữ liệu hoặc file vectorstore.pkl không tồn tại.")
        return []

    # Lấy mô hình và biểu diễn câu truy vấn thành vector
    model = get_model()
    query_emb = model.encode(query)
    
    # Chuyển sang numpy array và chuẩn hóa vector truy vấn về độ dài L2 = 1
    query_vector = np.array(query_emb)
    query_norm = np.linalg.norm(query_vector)
    if query_norm > 0:
        query_vector = query_vector / query_norm

    results = []
    
    # Tính Cosine Similarity giữa câu truy vấn và từng chunk
    for chunk in chunks:
        chunk_vector = np.array(chunk["embedding"])
        chunk_norm = np.linalg.norm(chunk_vector)
        if chunk_norm > 0:
            chunk_vector = chunk_vector / chunk_norm
            
        # Do cả hai vector đều đã chuẩn hóa, dot product chính là cosine similarity
        score = float(np.dot(chunk_vector, query_vector))
        
        results.append({
            "content": chunk["content"],
            "score": score,
            "metadata": chunk["metadata"]
        })

    # Sắp xếp kết quả giảm dần theo score tương đồng ngữ nghĩa
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    
    return results[:top_k]


if __name__ == "__main__":
    # Test thử tìm kiếm ngữ nghĩa
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    print("\n--- Kết quả Semantic Search ---")
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
