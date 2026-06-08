"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25
"""

import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi

VECTORSTORE_PATH = Path(__file__).parent.parent / "data" / "vectorstore.pkl"

# Biến toàn cục để lưu trữ corpus và đối tượng index BM25
CORPUS: list[dict] = []
_bm25 = None


def load_corpus_from_store():
    """Tải corpus từ file vector store cục bộ."""
    global CORPUS
    if not CORPUS:
        if not VECTORSTORE_PATH.exists():
            print("  [WARNING] Vector store chưa có dữ liệu. Vui lòng chạy task 4 trước!")
            return []
        with open(VECTORSTORE_PATH, "rb") as f:
            CORPUS = pickle.load(f)
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    global _bm25
    if _bm25 is None and corpus:
        # Tokenize cơ bản cho tiếng Việt bằng cách viết thường và chia khoảng trắng (split)
        tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
        _bm25 = BM25Okapi(tokenized_corpus)
    return _bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    corpus = load_corpus_from_store()
    if not corpus:
        return []
        
    bm25 = build_bm25_index(corpus)
    if bm25 is None:
        return []

    # Tokenize câu truy vấn
    tokenized_query = query.lower().split()
    
    # Tính điểm BM25 cho tất cả tài liệu trong corpus
    scores = bm25.get_scores(tokenized_query)

    # Lấy chỉ số các kết quả có điểm cao nhất
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        # Giữ lại các kết quả có điểm số lớn hơn 0 (có ít nhất một từ khớp) để lọc nhiễu
        # Nếu không có kết quả nào > 0, bài test vẫn yêu cầu trả về danh sách, nên ta có thể trả về tất cả
        # nhưng tốt nhất là trả về danh sách được sắp xếp giảm dần theo điểm.
        results.append({
            "content": corpus[idx]["content"],
            "score": float(scores[idx]),
            "metadata": corpus[idx]["metadata"]
        })
        
    return results


if __name__ == "__main__":
    # Test thử tìm kiếm từ khóa
    results = lexical_search("Điều 249 tàng trữ trái phép chất ma tuý", top_k=5)
    print("\n--- Kết quả Lexical Search (BM25) ---")
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
