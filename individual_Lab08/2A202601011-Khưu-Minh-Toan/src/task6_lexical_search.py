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
import json

# Load corpus từ data/standardized/ hoặc từ vector store
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
BM25_INDEX = None


def load_corpus_from_cache():
    global CORPUS, BM25_INDEX
    if not CORPUS:
        cache_path = Path(__file__).parent.parent / "data" / "vector_cache.json"
        if cache_path.exists():
            try:
                CORPUS = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Error loading corpus from cache: {e}")
        
        # Nếu chưa có file cache, tự động load và chunk
        if not CORPUS:
            from .task4_chunking_indexing import load_documents, chunk_documents
            try:
                docs = load_documents()
                CORPUS = chunk_documents(docs)
            except Exception as e:
                print(f"Error loading and chunking: {e}")
                
        if CORPUS:
            BM25_INDEX = build_bm25_index(CORPUS)


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi
    
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


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
    load_corpus_from_cache()
    if not BM25_INDEX or not CORPUS:
        print("  [ERROR] Corpus hoặc BM25 index rỗng. Vui lòng chạy task4 trước.")
        return []
        
    tokenized_query = query.lower().split()
    scores = BM25_INDEX.get_scores(tokenized_query)
    
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "content": CORPUS[idx]["content"],
            "score": float(scores[idx]),
            "metadata": CORPUS[idx]["metadata"]
        })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
