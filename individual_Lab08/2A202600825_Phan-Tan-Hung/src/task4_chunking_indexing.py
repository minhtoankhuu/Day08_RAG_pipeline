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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chunking Configuration
# Lý do chọn CHUNK_SIZE = 500: Đủ nhỏ để giữ thông tin cụ thể (ví dụ điều khoản luật hoặc đoạn tin tức)
# và không vượt quá context window của LLM khi kết hợp nhiều chunk.
# Lý do chọn CHUNK_OVERLAP = 50: Tránh bị đứt câu hoặc đứt nghĩa giữa các chunk liền kề.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# Embedding Configuration
# Lý do chọn sentence-transformers/all-MiniLM-L6-v2: Model nhẹ (90MB), chạy nhanh trên CPU local,
# đủ để demo và chạy test nhanh chóng mà không cần download model dung lượng lớn (như BGE-M3).
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Vector Store Configuration
# Vì Weaviate local không chạy sẵn, chúng ta sử dụng ChromaDB làm local vector database bền vững (persistent).
VECTOR_STORE = "chromadb"


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
        print(f"Warning: STANDARDIZED_DIR does not exist: {STANDARDIZED_DIR}")
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        if md_file.is_file():
            content = md_file.read_text(encoding="utf-8")
            doc_type = "legal" if "legal" in str(md_file.parent) else "news"
            documents.append({
                "content": content,
                "metadata": {"source": md_file.name, "type": doc_type}
            })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents sử dụng RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            # Cắt bớt khoảng trắng thừa
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "source": doc["metadata"]["source"],
                    "type": doc["metadata"]["type"],
                    "chunk_index": i
                }
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng SentenceTransformer model.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    if not chunks:
        return []

    print(f"Embedding {len(chunks)} chunks using {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào ChromaDB vector store.
    """
    if not chunks:
        print("No chunks to index.")
        return

    chroma_path = Path(__file__).parent.parent / "chromadb_data"
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    # Xóa collection cũ nếu tồn tại để tránh trùng lặp
    try:
        client.delete_collection(name="DrugLawDocs")
        print("Cleared existing ChromaDB collection 'DrugLawDocs'.")
    except Exception:
        pass
        
    collection = client.get_or_create_collection(
        name="DrugLawDocs",
        metadata={"hnsw:space": "cosine"} # sử dụng khoảng cách cosine
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = [c["embedding"] for c in chunks]
    documents = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Batch inserts
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
    print(f"Successfully indexed {len(chunks)} chunks to ChromaDB.")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
