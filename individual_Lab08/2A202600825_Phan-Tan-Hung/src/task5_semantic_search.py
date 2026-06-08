"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# Cấu hình phải đồng bộ với Task 4
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
chroma_path = Path(__file__).parent.parent / "chromadb_data"


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity trên ChromaDB.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, type, chunk_index
        }
        Sorted by score descending.
    """
    try:
        # 1. Connect tới ChromaDB
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection(name="DrugLawDocs")
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}. Return empty results.")
        return []

    # 2. Embed câu truy vấn
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode(query).tolist()

    # 3. Truy vấn vector store
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    if not results or not results["documents"] or len(results["documents"][0]) == 0:
        return []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # 4. Map kết quả sang định dạng chuẩn
    formatted_results = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        # Khoảng cách trong ChromaDB khi chọn hnsw:space: cosine là (1 - cosine_similarity).
        # Vì vậy, similarity score = 1.0 - distance.
        similarity_score = 1.0 - float(dist)
        formatted_results.append({
            "content": doc,
            "score": similarity_score,
            "metadata": meta
        })

    # Sắp xếp giảm dần theo similarity score
    formatted_results = sorted(formatted_results, key=lambda x: x["score"], reverse=True)
    return formatted_results


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    print("=" * 60)
    print("Semantic Search Results:")
    print("=" * 60)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:120]}...")
