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

import sys
import io
import json
from pathlib import Path

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

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chọn RecursiveCharacterTextSplitter để phân đoạn văn bản dựa trên ranh giới tự nhiên (đoạn, câu)
CHUNK_SIZE = 500        # 500 ký tự giúp giữ nguyên ngữ nghĩa của các điều luật hoặc đoạn bài báo ngắn
CHUNK_OVERLAP = 50      # Overlap 50 ký tự để không bị mất ngữ cảnh giữa các đoạn cắt liền kề
CHUNKING_METHOD = "recursive"

# Chọn all-MiniLM-L6-v2 vì model rất nhẹ (~90MB), chạy cực nhanh trên CPU và đủ tốt cho nhu cầu demo
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Sử dụng lưu trữ file JSON cục bộ vì môi trường local không cài đặt Docker/Weaviate
VECTOR_STORE = "local_json"


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
        print(f"[WARNING] Standardized directory not found: {STANDARDIZED_DIR}")
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file.parent) else "news"
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

    # Tối ưu hóa phân tách bằng cách giữ nguyên tính toàn vẹn của các Điều, Khoản, Điểm trong văn bản luật
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "\nĐiều ", "\nKhoản ", "\nĐiểm ", ". ", " ", ""]
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

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn (local JSON file).
    """
    db_file = STANDARDIZED_DIR.parent / "vectorstore.json"
    db_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SUCCESS] Saved vector store to: {db_file}")


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
    print("[INFO] Indexed to vector store successfully")


if __name__ == "__main__":
    run_pipeline()
