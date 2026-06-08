"""
Task 6 — Lexical Search Module (BM25).
"""

import sys
import io
import json
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi

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

_corpus = None
_bm25 = None

# Tách từ tiếng Việt sử dụng underthesea với fallback phân tách bằng khoảng trắng
try:
    from underthesea import word_tokenize
    def tokenize_vi(text: str) -> list[str]:
        return word_tokenize(text.lower())
except ImportError:
    def tokenize_vi(text: str) -> list[str]:
        return text.lower().split()


def get_corpus() -> list[dict]:
    global _corpus
    if _corpus is None:
        if DB_FILE.exists():
            try:
                _corpus = json.loads(DB_FILE.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[WARNING] Error reading vectorstore: {e}")
                _corpus = []
        else:
            _corpus = []
    return _corpus


# Module-level CORPUS for compatibility
CORPUS: list[dict] = get_corpus()


def get_bm25():
    global _bm25
    if _bm25 is None:
        corpus = get_corpus()
        if corpus:
            # Tokenize sử dụng bộ phân đoạn từ tiếng Việt
            tokenized_corpus = [tokenize_vi(doc["content"]) for doc in corpus]
            _bm25 = BM25Okapi(tokenized_corpus)
    return _bm25


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [tokenize_vi(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


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
    corpus = get_corpus()
    bm25 = get_bm25()
    if not corpus or bm25 is None:
        return []

    tokenized_query = tokenize_vi(query)
    scores = bm25.get_scores(tokenized_query)

    # Get sorted indices descending
    sorted_indices = np.argsort(scores)[::-1]

    results = []
    for idx in sorted_indices[:top_k]:
        results.append({
            "content": corpus[idx]["content"],
            "score": float(scores[idx]),
            "metadata": corpus[idx].get("metadata", {})
        })

    return results


if __name__ == "__main__":
    # Test
    print("=" * 50)
    print("Testing Lexical Search...")
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
    print("=" * 50)
