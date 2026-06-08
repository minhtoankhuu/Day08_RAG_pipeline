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

import pickle
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
VECTORSTORE_PATH = Path(__file__).parent.parent / "data" / "vectorstore.pkl"

# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# CHUNK_SIZE = 500: Phù hợp với độ dài trung bình của một Khoản hoặc Điều nhỏ trong luật Việt Nam,
# giúp giữ trọn vẹn ý nghĩa của điều khoản pháp lý và tránh làm loãng vector embedding.
CHUNK_SIZE = 500        

# CHUNK_OVERLAP = 50: Giúp giữ tính liên tục của văn cảnh giữa các chunk liền kề,
# tránh việc mất thông tin tại các điểm ngắt đoạn và hỗ trợ tìm kiếm từ khóa chéo ranh giới.
CHUNK_OVERLAP = 50      

CHUNKING_METHOD = "recursive"  # Sử dụng RecursiveCharacterTextSplitter để tách tự nhiên theo cấu trúc đoạn văn, dòng và câu.

# BAAI/bge-m3: Mô hình đa ngôn ngữ hàng đầu hiện nay, đặc biệt tốt trong việc xử lý ngữ nghĩa tiếng Việt
# và hỗ trợ độ dài context khổng lồ lên tới 8192 tokens.
EMBEDDING_MODEL = "BAAI/bge-m3"  
EMBEDDING_DIM = 1024

# local_pickle: Sử dụng giải pháp lưu trữ tệp tin cục bộ để đảm bảo hệ thống RAG chạy hoàn toàn in-process,
# offline, không cần thiết lập cổng mạng hay cài đặt/chạy Docker server phức tạp, tối đa hóa độ tin cậy.
VECTOR_STORE = "local_pickle"  


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
    
    # Duyệt qua toàn bộ file .md trong thư mục data/standardized/
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        if md_file.is_file() and not md_file.name.startswith("."):
            content = md_file.read_text(encoding="utf-8")
            # Xác định loại tài liệu dựa trên thư mục cha (legal hoặc news)
            doc_type = "legal" if "legal" in str(md_file.parent) else "news"
            documents.append({
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type
                }
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
                "metadata": {
                    "source": doc["metadata"]["source"],
                    "type": doc["metadata"]["type"],
                    "chunk_index": i
                }
            })
            
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    # Load mô hình từ cache cục bộ (hoặc tự động tải từ HF nếu chưa có)
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Lấy toàn bộ nội dung text để nhúng hàng loạt
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Gán vector embedding vào từng chunk dưới dạng danh sách số thực (list of floats)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
        
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    # Tạo thư mục lưu trữ dữ liệu nếu chưa tồn tại
    VECTORSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Lưu trữ dưới dạng tệp tin nhị phân bằng Pickle
    with open(VECTORSTORE_PATH, "wb") as f:
        pickle.dump(chunks, f)
        
    print(f"[SUCCESS] Indexed {len(chunks)} chunks to local vector store at: {VECTORSTORE_PATH}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("[OK] Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
