"""
Task 8 — PageIndex Vectorless RAG.

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from pageindex import PageIndexClient

# Đảm bảo đọc .env trước
load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
# Thư mục chứa PDF gốc
LEGAL_LANDING_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def get_pageindex_client():
    """Tạo client kết nối tới PageIndex."""
    if not PAGEINDEX_API_KEY:
        raise ValueError("PAGEINDEX_API_KEY is not set in .env file.")
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def upload_documents():
    """
    Upload toàn bộ file PDF trong data/landing/legal/ lên PageIndex.
    Tránh upload trùng lặp bằng cách đối chiếu tên file.
    """
    client = get_pageindex_client()
    
    # Lấy danh sách tài liệu hiện có trên hệ thống
    existing_docs = client.list_documents()
    existing_names = [d["name"] for d in existing_docs.get("documents", [])]

    uploaded_count = 0
    for filepath in LEGAL_LANDING_DIR.iterdir():
        # Chỉ upload tài liệu PDF
        if filepath.is_file() and filepath.suffix.lower() == ".pdf":
            if filepath.name in existing_names:
                print(f"Document '{filepath.name}' is already uploaded on PageIndex.")
                continue

            print(f"Uploading '{filepath.name}' to PageIndex...")
            try:
                res = client.submit_document(file_path=str(filepath))
                doc_id = res.get("doc_id")
                print(f"  [OK] Uploaded successfully. doc_id: {doc_id}")
                uploaded_count += 1
            except Exception as e:
                print(f"  [Error] Failed to upload {filepath.name}: {e}")

    print(f"Finished uploading documents. Total new uploads: {uploaded_count}")


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
            'source': 'pageindex'
        }
    """
    try:
        client = get_pageindex_client()
    except Exception as e:
        print(f"PageIndex client connection error: {e}")
        return []

    # 1. Lấy tất cả tài liệu có trạng thái completed
    docs_resp = client.list_documents()
    documents = docs_resp.get("documents", [])
    
    completed_doc_ids = [
        d["id"] for d in documents 
        if d.get("status") == "completed" or d.get("status") == "ready"
    ]

    if not completed_doc_ids:
        print("No completed/ready documents found on PageIndex.")
        return []

    results = []
    
    # 2. Truy vấn từng tài liệu hoàn thành
    for doc_id in completed_doc_ids:
        try:
            print(f"Querying document {doc_id} on PageIndex...")
            # Submit query
            ret_resp = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = ret_resp.get("retrieval_id")
            
            if not retrieval_id:
                print(f"  Failed to get retrieval_id for document {doc_id}")
                continue

            # Poll cho đến khi có kết quả hoặc hết thời gian chờ (tăng lên 30 lần để an toàn)
            max_attempts = 30
            retrieval = None
            for attempt in range(max_attempts):
                retrieval = client.get_retrieval(retrieval_id)
                status = retrieval.get("status")
                if status == "completed":
                    break
                elif status == "failed":
                    print(f"  Retrieval {retrieval_id} failed.")
                    break
                time.sleep(1.0)
            
            # Trích xuất kết quả
            if retrieval and retrieval.get("status") == "completed":
                nodes = retrieval.get("retrieved_nodes", [])
                if not nodes:
                    nodes = retrieval.get("results", [])
                if not nodes and "output" in retrieval:
                    nodes = retrieval.get("output", [])
                
                for node in nodes:
                    # Trích xuất nội dung từ relevant_contents có cấu trúc lồng nhau
                    contents = []
                    for sublist in node.get("relevant_contents", []):
                        if isinstance(sublist, list):
                            for item in sublist:
                                if isinstance(item, dict) and "relevant_content" in item:
                                    contents.append(item["relevant_content"])
                                elif isinstance(item, str):
                                    contents.append(item)
                        elif isinstance(sublist, str):
                            contents.append(sublist)
                    
                    content = "\n".join(contents).strip()
                    # Điểm tương quan
                    score = float(node.get("score", node.get("relevance", 0.5)))
                    meta = node.get("metadata", {})
                    
                    # Phân tích metadata linh hoạt
                    meta_dict = {}
                    source_name = "pageindex_document"
                    if isinstance(meta, dict):
                        meta_dict = meta
                        source_name = meta.get("filename", node.get("title", "pageindex_document"))
                    elif isinstance(meta, list):
                        if len(meta) > 1 and isinstance(meta[1], str):
                            source_name = meta[1]
                        meta_dict = {"list": meta}

                    if content:
                        results.append({
                            "content": content,
                            "score": score,
                            "metadata": {
                                "source": source_name,
                                "doc_id": doc_id,
                                **meta_dict
                            },
                            "source": "pageindex"
                        })
        except Exception as e:
            print(f"  Error querying document {doc_id}: {e}")

    # 3. Sort và giới hạn kết quả
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    if not PAGEINDEX_API_KEY:
        print("Please set PAGEINDEX_API_KEY in the .env file.")
    else:
        print("Checking documents status and uploading...")
        upload_documents()

        print("\nTesting retrieval...")
        res = pageindex_search("Luật Phòng chống ma túy 2021", top_k=3)
        for idx, r in enumerate(res, 1):
            print(f"[{idx}] [{r['score']:.3f}] Source: {r['metadata'].get('source')}")
            print(f"    Content: {r['content'][:150]}...")
