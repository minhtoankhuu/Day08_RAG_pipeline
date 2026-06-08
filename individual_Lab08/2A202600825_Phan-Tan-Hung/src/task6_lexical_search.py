"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from .task4_chunking_indexing import load_documents, chunk_documents

# Corpus được tải lazily
CORPUS: list[dict] = []
bm25_index = None


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    global CORPUS, bm25_index
    CORPUS = corpus
    if not CORPUS:
        bm25_index = None
        return None
    # Tokenize đơn giản bằng cách viết thường và chia khoảng trắng
    tokenized_corpus = [doc["content"].lower().split() for doc in CORPUS]
    bm25_index = BM25Okapi(tokenized_corpus)
    return bm25_index


def get_bm25_index():
    """Lấy hoặc tự động khởi tạo BM25 index từ file hệ thống."""
    global CORPUS, bm25_index
    if bm25_index is None:
        docs = load_documents()
        corpus = chunk_documents(docs)
        build_bm25_index(corpus)
    return bm25_index


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
    index = get_bm25_index()
    if not index or not CORPUS:
        return []

    tokenized_query = query.lower().split()
    scores = index.get_scores(tokenized_query)

    # Lấy ra các index có score cao nhất
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        # Chúng ta chỉ lấy các kết quả có điểm BM25 > 0 để lọc bớt nhiễu
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
            
    # Đảm bảo kết quả được sắp xếp giảm dần theo điểm score
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    print("=" * 60)
    print("Lexical Search (BM25) Results:")
    print("=" * 60)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:120]}...")
