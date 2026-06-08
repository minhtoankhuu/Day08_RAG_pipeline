import sys
from pathlib import Path

# Đảm bảo import được group_project
sys.path.append(str(Path(__file__).parent.parent.parent))

from group_project.src.rag_pipeline import GroupRAGPipeline

def test_pipeline():
    print("=== Bắt đầu kiểm tra GroupRAGPipeline ===")
    
    try:
        pipeline = GroupRAGPipeline()
    except Exception as e:
        print(f"[FAIL] Lỗi khởi tạo pipeline: {e}")
        return

    # Test query
    query = "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật?"
    print(f"\n[QUERY] Hỏi thử: '{query}'")
    
    try:
        result = pipeline.generate(query)
        print(f"\n[ANSWER]\n{result['answer']}")
        print(f"\n[SOURCE TYPE] Tìm kiếm qua: {result['retrieval_source']}")
        print(f"[SOURCES COUNT] Số lượng chunk tìm thấy: {len(result['sources'])}")
        
        if result['sources']:
            print("\n[TOP SOURCE CHUNKS]:")
            for i, s in enumerate(result['sources'][:2], 1):
                source_name = s.get("metadata", {}).get("source", "Không rõ nguồn")
                print(f"  {i}. Source: {source_name} (Score: {s['score']:.3f})")
                print(f"     Nội dung: {s['content'][:150]}...")
        
        print("\n[SUCCESS] Pipeline chạy thử nghiệm thành công không gặp lỗi cú pháp!")
    except Exception as e:
        print(f"[FAIL] Lỗi chạy thử nghiệm query: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
