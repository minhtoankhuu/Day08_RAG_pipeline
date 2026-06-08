"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
import json
import time
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


class PageIndexResult:
    """Đại diện cho kết quả trả về từ PageIndex query."""
    def __init__(self, text: str, score: float, metadata: dict):
        self.text = text
        self.score = score
        self.metadata = metadata


class PageIndex:
    """Adapter class để bọc PageIndexClient và cung cấp API upload/query đơn giản."""
    def __init__(self, api_key: str):
        from pageindex import PageIndexClient
        self.client = PageIndexClient(api_key=api_key)

    def upload_file(self, file_path: str):
        """Upload file PDF thực tế lên PageIndex."""
        try:
            res = self.client.submit_document(file_path)
            doc_id = res.get("doc_id")
            if doc_id:
                self._save_doc_id(doc_id)
                print(f"  [SUCCESS] Uploaded {Path(file_path).name} to PageIndex. Doc ID: {doc_id}")
        except Exception as e:
            print(f"  [WARN] PageIndex upload failed for {Path(file_path).name}: {e}")

    def _save_doc_id(self, doc_id: str):
        cache_file = Path(__file__).parent.parent / "data" / "pageindex_docs.json"
        doc_ids = []
        if cache_file.exists():
            try:
                doc_ids = json.loads(cache_file.read_text(encoding="utf-8"))
            except:
                pass
        if doc_id not in doc_ids:
            doc_ids.append(doc_id)
        cache_file.write_text(json.dumps(doc_ids), encoding="utf-8")

    def query(self, query: str, top_k: int = 5) -> list[PageIndexResult]:
        cache_file = Path(__file__).parent.parent / "data" / "pageindex_docs.json"
        if not cache_file.exists():
            return []
        try:
            doc_ids = json.loads(cache_file.read_text(encoding="utf-8"))
        except:
            return []

        results = []
        for doc_id in doc_ids:
            try:
                # Chờ document sẵn sàng (tối đa 20 lần thử, mỗi lần 5 giây)
                print(f"Waiting for document {doc_id} to be ready on PageIndex...")
                ready = False
                for i in range(20):
                    if self.client.is_retrieval_ready(doc_id):
                        ready = True
                        break
                    time.sleep(5)
                if not ready:
                    print(f"  [WARN] Document {doc_id} is still processing. Skipping query for now.")
                    continue

                # Gửi truy vấn
                res = self.client.submit_query(doc_id, query)
                retrieval_id = res.get("retrieval_id")
                if retrieval_id:
                    # Chờ kết quả xử lý xong
                    for _ in range(5):
                        ret_res = self.client.get_retrieval(retrieval_id)
                        if ret_res.get("status") == "completed":
                            for r in ret_res.get("results", []):
                                text = r.get("text") or r.get("content") or ""
                                score = r.get("score") or 0.0
                                meta = r.get("metadata") or {}
                                results.append(PageIndexResult(text, score, meta))
                            break
                        time.sleep(2)
            except Exception as e:
                print(f"  [WARN] PageIndex query failed for doc {doc_id}: {e}")

        # Sắp xếp kết quả theo điểm tương đồng
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


def upload_documents():
    """
    Quét và upload các file PDF thực tế từ thư mục data/landing/ lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        raise Exception("PAGEINDEX_API_KEY not set in .env")

    pi = PageIndex(api_key=PAGEINDEX_API_KEY)

    landing_dir = Path(__file__).parent.parent / "data" / "landing"
    pdf_files = list(landing_dir.rglob("*.pdf"))
    
    if not pdf_files:
        print("  [WARN] No PDF files found in data/landing/ to upload to PageIndex.")
        return
        
    for pdf_file in pdf_files:
        pi.upload_file(str(pdf_file))


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
    if not PAGEINDEX_API_KEY:
        raise Exception("PAGEINDEX_API_KEY not set in .env")

    pi = PageIndex(api_key=PAGEINDEX_API_KEY)
    results = pi.query(query=query, top_k=top_k)

    return [
        {
            "content": r.text,
            "score": r.score,
            "metadata": r.metadata,
            "source": "pageindex"
        }
        for r in results
    ]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
