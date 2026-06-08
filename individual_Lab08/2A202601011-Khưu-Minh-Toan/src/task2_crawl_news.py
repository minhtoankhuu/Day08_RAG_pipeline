"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    "https://tuoitre.vn/vu-ma-tuy-lien-quan-tiep-vien-vietnam-airlines-truy-to-nguoi-mau-an-tay-chi-dan-va-225-bi-can-20260402112720784.htm",
    "https://tuoitre.vn/bat-nguoi-mau-an-tay-ca-si-chi-dan-co-tien-truc-phuong-do-lien-quan-ma-tuy-20241114114826655.htm",
    "https://thanhnien.vn/chuyen-an-bi-so-vn10-ca-si-chi-dan-ru-re-gop-tien-choi-ma-tuy-185260403093444362.htm",
    "https://vietnamnet.vn/loat-anh-ca-si-chi-dan-nguoi-mau-an-tay-va-co-tien-truc-phuong-khi-bi-bat-2341935.html",
    "https://vov.vn/phap-luat/vu-an-ma-tuy-lien-quan-den-ca-si-chi-dan-vi-sao-nguoi-mau-an-tay-bi-truy-to-post1281931.vov",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler
    import requests
    from bs4 import BeautifulSoup

    print(f"  -> Attempting Crawl4AI for: {url}")
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result and result.success and result.markdown:
                title = result.metadata.get("title") or "Unknown"
                if title == "Unknown" and hasattr(result, 'html') and result.html:
                    soup = BeautifulSoup(result.html, 'html.parser')
                    title = soup.title.string if soup.title else "Unknown"
                return {
                    "url": url,
                    "title": title.strip() if title else "Unknown",
                    "date_crawled": datetime.now().isoformat(),
                    "content_markdown": result.markdown,
                }
    except Exception as e:
        print(f"  [WARN] Crawl4AI failed: {e}. Falling back to standard requests+BeautifulSoup...")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }
        res = requests.get(url, headers=headers, timeout=15)
        res.encoding = 'utf-8'
        soup = BeautifulSoup(res.text, 'html.parser')
        
        title = "Unknown"
        if soup.title:
            title = soup.title.string
        elif soup.find('h1'):
            title = soup.find('h1').text

        paragraphs = soup.find_all('p')
        text_content = "\n\n".join([p.text.strip() for p in paragraphs if p.text.strip()])
        
        return {
            "url": url,
            "title": title.strip() if title else "Unknown",
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": text_content,
        }
    except Exception as e:
        print(f"  [ERROR] Fallback crawl failed for {url}: {e}")
        return {
            "url": url,
            "title": "Failed to crawl",
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": f"Failed to retrieve content for {url} due to error: {e}",
        }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [SUCCESS] Saved: {filename}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
