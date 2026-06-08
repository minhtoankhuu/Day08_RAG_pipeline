"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


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
    from sentence_transformers import SentenceTransformer

    # Import config từ Task 4 để đồng bộ model
    from .task4_chunking_indexing import EMBEDDING_MODEL
    
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_emb = model.encode(query).tolist()
    
    # Thử kết nối Weaviate
    try:
        import weaviate
        from weaviate.classes.query import MetadataQuery
        
        client = weaviate.connect_to_local()
        collection = client.collections.get("DrugLawDocs")
        
        results = collection.query.near_vector(
            near_vector=query_emb,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True)
        )
        
        output = []
        for obj in results.objects:
            score = 1.0 - (obj.metadata.distance if obj.metadata.distance is not None else 0.5)
            output.append({
                "content": obj.properties.get("content", ""),
                "score": score,
                "metadata": {
                    "source": obj.properties.get("source", "Unknown"),
                    "type": obj.properties.get("doc_type", "Unknown"),
                    "chunk_index": obj.properties.get("chunk_index", 0)
                }
            })
        client.close()
        return output
    except Exception as e:
        # Fallback sử dụng file cache local
        import json
        from pathlib import Path
        
        cache_path = Path(__file__).parent.parent / "data" / "vector_cache.json"
        if not cache_path.exists():
            print("  [ERROR] Local vector cache file does not exist. Please run task4_chunking_indexing first!")
            return []
            
        chunks = json.loads(cache_path.read_text(encoding="utf-8"))
        
        # Hàm tính cosine similarity thủ công
        def cosine_similarity(v1, v2):
            dot_product = sum(a * b for a, b in zip(v1, v2))
            norm_a = sum(a * a for a in v1) ** 0.5
            norm_b = sum(b * b for b in v2) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot_product / (norm_a * norm_b)
            
        scored_chunks = []
        for c in chunks:
            if "embedding" in c:
                score = cosine_similarity(query_emb, c["embedding"])
                scored_chunks.append({
                    "content": c["content"],
                    "score": score,
                    "metadata": c["metadata"]
                })
                
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
