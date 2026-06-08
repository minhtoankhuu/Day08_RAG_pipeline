import os
import shutil
from pathlib import Path

def prepare_group_data():
    # Định nghĩa các đường dẫn thư mục
    base_dir = Path(__file__).parent.parent.parent
    individual_dir = base_dir / "individual_Lab08"
    group_data_dir = base_dir / "group_project" / "data"
    group_legal_dir = group_data_dir / "landing" / "legal"
    group_news_dir = group_data_dir / "landing" / "news"

    # Tạo các thư mục nhóm
    group_legal_dir.mkdir(parents=True, exist_ok=True)
    group_news_dir.mkdir(parents=True, exist_ok=True)

    print("=== Khởi tạo quy trình gom dữ liệu nhóm ===")
    
    # 1. Gom tài liệu pháp lý (chỉ cần lấy từ 1 thành viên vì chúng giống nhau)
    toan_legal_dir = individual_dir / "2A202601011-Khưu-Minh-Toan" / "data" / "landing" / "legal"
    if toan_legal_dir.exists():
        print("\n[1/2] Đang sao chép các tài liệu pháp lý...")
        for legal_file in toan_legal_dir.glob("*"):
            if legal_file.is_file() and legal_file.name != ".gitkeep":
                dest_file = group_legal_dir / legal_file.name
                shutil.copy2(legal_file, dest_file)
                print(f"  -> Đã copy: {legal_file.name}")
    else:
        print("\n[WARN] Không tìm thấy thư mục pháp lý cá nhân của Toàn để copy làm mẫu.")

    # 2. Gom tin tức (news) từ tất cả thành viên và đổi tên
    print("\n[2/2] Đang thu thập tin tức từ các thành viên...")
    
    # Danh sách các thành viên trong nhóm
    members = [
        {"folder": "2A202601011-Khưu-Minh-Toan", "prefix": "toan"},
        {"folder": "2A202600546-Nguyễn-Bùi-Tấn-Dũng", "prefix": "dung"},
        {"folder": "2A202600950-Doan-Xuan-Thach", "prefix": "thach"}
    ]

    for member in members:
        member_news_dir = individual_dir / member["folder"] / "data" / "landing" / "news"
        prefix = member["prefix"]
        
        if member_news_dir.exists():
            print(f"  * Đang quét thư mục của: {member['folder']}")
            count = 0
            for news_file in member_news_dir.glob("*.json"):
                if news_file.is_file():
                    # Đổi tên file để tránh bị ghi đè, ví dụ: article_01.json -> toan_article_01.json
                    new_name = f"{prefix}_{news_file.name}"
                    dest_file = group_news_dir / new_name
                    shutil.copy2(news_file, dest_file)
                    count += 1
            print(f"    => Đã copy {count} bài viết với tiền tố '{prefix}_'")
        else:
            print(f"  [WARN] Không tìm thấy thư mục tin tức của thành viên: {member['folder']}")

    print("\n=== Hoàn thành! ===")
    print(f"Tổng số tài liệu pháp lý: {len(list(group_legal_dir.glob('*')))} file")
    print(f"Tổng số file tin tức: {len(list(group_news_dir.glob('*.json')))} file")

if __name__ == "__main__":
    prepare_group_data()
