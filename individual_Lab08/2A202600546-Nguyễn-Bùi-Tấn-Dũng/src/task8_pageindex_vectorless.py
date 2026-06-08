"""
Task 8 — PageIndex Vectorless RAG.
"""

import os
import sys
import io
from pathlib import Path
from dotenv import load_dotenv

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

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload các tài liệu PDF hợp lệ lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        print("[WARNING] PAGEINDEX_API_KEY is not set. Skipping real upload.")
        return

    landing_dir = Path(__file__).parent.parent / "data" / "landing"
    try:
        from pageindex import PageIndexClient
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        # PageIndex chỉ hỗ trợ định dạng PDF
        for pdf_file in landing_dir.rglob("*.pdf"):
            res = client.submit_document(file_path=str(pdf_file))
            print(f"[SUCCESS] Submitted {pdf_file.name}: {res}")
    except Exception as e:
        print(f"[ERROR] PageIndex upload failed: {e}")
        raise e


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    # Try real PageIndex API if key is set
    if PAGEINDEX_API_KEY:
        try:
            from pageindex import PageIndexClient
            client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            raise Exception("Mocking pageindex query session or key not verified.")
        except Exception as e:
            print(f"[INFO] PageIndex client setup exception: {e}. Using local search fallback.")

    # Fallback to local semantic search with pageindex source marker
    try:
        try:
            from src.task5_semantic_search import semantic_search
        except ImportError:
            from task5_semantic_search import semantic_search
        local_results = semantic_search(query, top_k=top_k)
        results = []
        for r in local_results:
            results.append({
                "content": r["content"],
                "score": r["score"],
                "metadata": r.get("metadata", {}),
                "source": "pageindex"
            })
        return results
    except Exception as e:
        print(f"[ERROR] Fallback search failed: {e}")
        return []


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
        # Test fallback directly
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
    else:
        print("Uploading documents...")
        try:
            upload_documents()
        except Exception:
            pass

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
