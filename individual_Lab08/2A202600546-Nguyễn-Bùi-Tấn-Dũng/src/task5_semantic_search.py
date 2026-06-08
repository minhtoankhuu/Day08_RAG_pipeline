import sys
import io
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# Cấu hình mã hóa UTF-8 cho terminal trên Windows để tránh lỗi UnicodeEncodeError
if sys.stdout.encoding.lower() != 'utf-8':
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
if sys.stderr.encoding.lower() != 'utf-8':
    if hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

DB_FILE = Path(__file__).parent.parent / "data" / "vectorstore.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cache model và db_data để tối ưu hóa tốc độ chạy test
_model = None
_db_data = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_db_data():
    global _db_data
    if _db_data is None:
        if DB_FILE.exists():
            _db_data = json.loads(DB_FILE.read_text(encoding="utf-8"))
        else:
            _db_data = []
    return _db_data


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

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
    chunks = get_db_data()
    if not chunks:
        print("[WARNING] Vector store is empty or not found.")
        return []

    # 1. Embed query
    model = get_model()
    query_emb = model.encode(query)

    # 2. Tính độ tương đồng cosine
    query_vec = np.array(query_emb)
    chunk_vecs = np.array([c["embedding"] for c in chunks])

    dot_products = np.dot(chunk_vecs, query_vec)
    query_norm = np.linalg.norm(query_vec)
    chunk_norms = np.linalg.norm(chunk_vecs, axis=1)

    # Tránh chia cho 0
    scores = dot_products / (query_norm * chunk_norms + 1e-10)

    # 3. Định dạng kết quả đầu ra
    results = []
    for chunk, score in zip(chunks, scores):
        results.append({
            "content": chunk["content"],
            "score": float(score),
            "metadata": chunk.get("metadata", {})
        })

    # Sắp xếp giảm dần theo điểm số
    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]


if __name__ == "__main__":
    # Test
    print("=" * 50)
    print("Testing Semantic Search...")
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
    print("=" * 50)
