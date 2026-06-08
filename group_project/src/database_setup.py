import json
import os
import shutil
from pathlib import Path
from markitdown import MarkItDown
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import cấu hình dùng chung của nhóm
from . import config

def clean_news_content(text: str) -> str:
    """Loại bỏ header navigation và footer junk khỏi bài báo tin tức."""
    lines = text.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("#"):
            start_idx = i
            break
            
    lines = lines[start_idx:]
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Dừng thu thập khi gặp các phần social/footer nhiễu
        if any(marker in stripped for marker in [
            "[ Facebook ](https://www.facebook.com/sharer",
            "[ Chia sẻ ](javascript:",
            "[ Zalo ](javascript:",
            "CHIA SẺ",
            "Tặng sao",
            "Chuyển sao tặng cho thành viên",
            "Bình luận (",
            "Ý kiến của bạn sẽ được biên tập",
            "#### Khám phá thêm chủ đề",
            "![footer__logo]",
            "© Copyright",
            "Hotline",
            "Email:",
            "Tuyển tập tin liên quan",
            "Đăng ký email - Mở cổng thông tin"
        ]):
            break
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines).strip()

def convert_to_markdown():
    """Convert toàn bộ file trong data/landing/ thành Markdown và lưu vào data/standardized/"""
    print("\n" + "=" * 50)
    print("BƯỚC 1: Chuyển đổi dữ liệu sang Markdown (MarkItDown)")
    print("=" * 50)

    landing_dir = config.DATA_DIR / "landing"
    legal_landing = landing_dir / "legal"
    news_landing = landing_dir / "news"

    legal_output = config.STANDARDIZED_DIR / "legal"
    news_output = config.STANDARDIZED_DIR / "news"

    # Đảm bảo thư mục đầu ra tồn tại
    legal_output.mkdir(parents=True, exist_ok=True)
    news_output.mkdir(parents=True, exist_ok=True)

    # 1. Convert tài liệu pháp luật (PDF/DOCX)
    md = MarkItDown()
    print("\n--- [1/2] Converting Legal Documents ---")
    if legal_landing.exists():
        for filepath in legal_landing.iterdir():
            if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
                print(f"Đang convert: {filepath.name}")
                try:
                    result = md.convert(str(filepath))
                    output_path = legal_output / f"{filepath.stem}.md"
                    output_path.write_text(result.text_content, encoding="utf-8")
                    print(f"  [SUCCESS] Đã lưu: {output_path.name}")
                except Exception as e:
                    print(f"  [ERROR] Lỗi chuyển đổi {filepath.name}: {e}")
    else:
        print("[WARN] Thư mục legal landing trống hoặc không tồn tại.")

    # 2. Convert tin tức (JSON)
    print("\n--- [2/2] Converting News Articles ---")
    if news_landing.exists():
        for filepath in news_landing.iterdir():
            if filepath.suffix.lower() == ".json":
                print(f"Đang convert: {filepath.name}")
                try:
                    data = json.loads(filepath.read_text(encoding="utf-8"))
                    output_path = news_output / f"{filepath.stem}.md"

                    cleaned_body = clean_news_content(data.get("content_markdown", ""))

                    # Ghép header metadata
                    header = f"**Title:** {data.get('title', 'Unknown')}\n"
                    header += f"**Source:** {data.get('url', 'N/A')}\n"
                    header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

                    content = header + cleaned_body
                    output_path.write_text(content, encoding="utf-8")
                    print(f"  [SUCCESS] Đã lưu: {output_path.name}")
                except Exception as e:
                    print(f"  [ERROR] Lỗi chuyển đổi {filepath.name}: {e}")
    else:
        print("[WARN] Thư mục news landing trống hoặc không tồn tại.")

    print(f"\n[DONE] Hoàn tất chuyển đổi sang Markdown tại: {config.STANDARDIZED_DIR}")

def load_standardized_documents() -> list[dict]:
    """Đọc toàn bộ file markdown từ data/standardized/"""
    documents = []
    if not config.STANDARDIZED_DIR.exists():
        return documents
    
    # Đọc files từ thư mục legal
    legal_dir = config.STANDARDIZED_DIR / "legal"
    if legal_dir.exists():
        for md_file in legal_dir.rglob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            documents.append({
                "content": content,
                "metadata": {"source": md_file.name, "type": "legal"}
            })

    # Đọc files từ thư mục news
    news_dir = config.STANDARDIZED_DIR / "news"
    if news_dir.exists():
        for md_file in news_dir.rglob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            documents.append({
                "content": content,
                "metadata": {"source": md_file.name, "type": "news"}
            })
            
    return documents

def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia nhỏ các tài liệu thành các đoạn văn ngắn."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
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

def index_and_embed():
    """Lập chỉ mục, tính vector embedding và lưu vào Database / Cache file."""
    print("\n" + "=" * 50)
    print("BƯỚC 2: Chunking, Embedding & Indexing")
    print("=" * 50)

    docs = load_standardized_documents()
    print(f"\n[INFO] Đã tải {len(docs)} tài liệu markdown chuẩn hóa.")
    if not docs:
        print("[ERROR] Không tìm thấy tài liệu nào. Vui lòng chạy gộp dữ liệu và chuyển đổi trước!")
        return

    # 1. Chunking
    chunks = chunk_documents(docs)
    print(f"[INFO] Đã chia nhỏ thành {len(chunks)} chunks.")

    # 2. Embedding
    print(f"[INFO] Bắt đầu sinh vector embedding với model {config.EMBEDDING_MODEL}...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    print("  [SUCCESS] Sinh vector embeddings thành công.")

    # 3. Indexing to local JSON cache
    cache_path = config.DATA_DIR / "vector_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[SUCCESS] Đã lưu {len(chunks)} chunks đã embed vào cache local: {cache_path}")

    # 4. Indexing to Weaviate (nếu dùng)
    if config.VECTOR_STORE == "weaviate":
        import weaviate
        from weaviate.classes.config import Configure, Property, DataType

        try:
            print("\nĐang kết nối tới Weaviate local...")
            client = weaviate.connect_to_local()
            
            # Xóa collection cũ của nhóm nếu có
            if client.collections.exists("GroupDrugLawDocs"):
                client.collections.delete("GroupDrugLawDocs")
                
            # Tạo collection mới
            collection = client.collections.create(
                name="GroupDrugLawDocs",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="doc_type", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                ]
            )
            
            # Chèn các chunk kèm embedding
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
            print("  [SUCCESS] Đã lập chỉ mục vào Weaviate collection: GroupDrugLawDocs")
            client.close()
        except Exception as e:
            print(f"  [WARN] Không thể kết nối hoặc index vào Weaviate: {e}")
            print("  (Hệ thống sẽ chạy fallback tìm kiếm trực tiếp trên file vector_cache.json)")

def main():
    convert_to_markdown()
    index_and_embed()

if __name__ == "__main__":
    main()
