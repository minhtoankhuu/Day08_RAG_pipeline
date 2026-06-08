"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# TODO: Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 500        # Chọn 500 vì độ dài này (~100-150 từ) vừa đủ chứa trọn vẹn một điều luật hoặc một đoạn tin tức, giúp giữ thông tin cô đọng cho mô hình embedding.
CHUNK_OVERLAP = 50      # Chọn 50 (~10-15 từ) để đảm bảo ngữ cảnh ở vị trí biên giữa các chunk không bị đứt đoạn hay mất thông tin liên kết.
CHUNKING_METHOD = "recursive"  # Dùng recursive vì nó tôn trọng các ký tự phân tách tự nhiên (\n\n, \n, câu, từ) để giữ câu văn toàn vẹn nhất.

# TODO: Chọn embedding model và giải thích
EMBEDDING_MODEL = "BAAI/bge-m3"  # Bge-m3 là model mã nguồn mở hàng đầu cho tiếng Việt hiện nay, hỗ trợ độ dài ngữ cảnh tốt và độ chính xác retrieval rất cao.
EMBEDDING_DIM = 1024

# TODO: Chọn vector store
VECTOR_STORE = "weaviate"  # Weaviate hỗ trợ tìm kiếm hybrid mạnh mẽ (BM25 + Vector). Code cũng lưu cache local phòng khi máy không chạy Weaviate Docker.


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file.parent) or "legal" in md_file.name else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    # 1. Luôn lưu vào local cache file để làm fallback
    cache_path = Path(__file__).parent.parent / "data" / "vector_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    cache_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [SUCCESS] Saved {len(chunks)} chunks to local cache: {cache_path}")

    # 2. Thử lưu vào Weaviate
    if VECTOR_STORE == "weaviate":
        import weaviate
        from weaviate.classes.config import Configure, Property, DataType
        
        try:
            print("Connecting to local Weaviate...")
            client = weaviate.connect_to_local()
            
            # Xóa collection cũ nếu tồn tại
            if client.collections.exists("DrugLawDocs"):
                client.collections.delete("DrugLawDocs")
                
            # Tạo collection mới
            collection = client.collections.create(
                name="DrugLawDocs",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="doc_type", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                ]
            )
            
            # Chèn các chunk vào Weaviate
            with collection.batch.dynamic() as batch:
                for chunk in chunks:
                    batch.add_object(
                        properties={
                            "content": chunk["content"],
                            "source": chunk["metadata"].get("source", "Unknown"),
                            "doc_type": chunk["metadata"].get("type", "Unknown"),
                            "chunk_index": chunk["metadata"].get("chunk_index", 0),
                        },
                        vector=chunk["embedding"]
                    )
            print("  [SUCCESS] Indexed to Weaviate collection: DrugLawDocs")
            client.close()
        except Exception as e:
            print(f"  [WARN] Weaviate indexing skipped/failed: {e}.")
            print("  (Don't worry, local fallback cache is saved and semantic search will use it!)")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n[INFO] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[INFO] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[INFO] Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("[SUCCESS] Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
