"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            try:
                result = md.convert(str(filepath))
                output_path = output_dir / f"{filepath.stem}.md"
                output_path.write_text(result.text_content, encoding="utf-8")
                print(f"  [SUCCESS] Saved: {output_path.name}")
            except Exception as e:
                print(f"  [ERROR] Failed to convert {filepath.name}: {e}")


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


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                output_path = output_dir / f"{filepath.stem}.md"

                # Làm sạch nội dung bài báo
                cleaned_body = clean_news_content(data.get("content_markdown", ""))

                # Thêm metadata header (không lặp lại tiêu đề dạng # nữa)
                header = f"**Title:** {data.get('title', 'Unknown')}\n"
                header += f"**Source:** {data.get('url', 'N/A')}\n"
                header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

                content = header + cleaned_body
                output_path.write_text(content, encoding="utf-8")
                print(f"  [SUCCESS] Saved: {output_path.name}")
            except Exception as e:
                print(f"  [ERROR] Failed to convert {filepath.name}: {e}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
