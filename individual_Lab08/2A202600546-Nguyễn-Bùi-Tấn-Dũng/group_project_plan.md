# Kế Hoạch Dự Án Nhóm: RAG Chatbot & Evaluation Pipeline

Tài liệu này phác thảo kế hoạch chi tiết, kiến trúc hệ thống và phân chia công việc cho nhóm **4 thành viên** để thực hiện bài tập nhóm Day 8 - RAG Pipeline v2.

---

## 1. Mục Tiêu Dự Án Nhóm (Tổng 30 điểm)
Nhóm cần xây dựng đồng thời cả hai phần để đạt điểm tối đa:
1. **RAG Chatbot (18 điểm):** Giao diện Chatbot trả lời thông tin về pháp luật ma túy & tin tức nghệ sĩ, hỗ trợ Conversation Memory (Lịch sử chat), hiển thị trích dẫn (Citations) và các tài liệu nguồn đã dùng (Source documents).
2. **RAG Evaluation Pipeline (12 điểm):** Đánh giá chất lượng của RAG Pipeline bằng DeepEval/Ragas/TruLens thông qua bộ Golden Dataset (≥15 Q&A), chạy so sánh A/B giữa 2 cấu hình (ví dụ: Dense vs Hybrid + Reranker) và viết báo cáo phân tích.

---

## 2. Kiến Trúc Hệ Thống Đề Xuất

```
               [Giao diện Chatbot: Streamlit / Chainlit]
                                │   ▲
                 (Gửi câu hỏi)  │   │  (Trả về câu trả lời + Source + Citation)
                                ▼   │
               [RAG Pipeline Controller (Tích hợp từ Task 9)]
                                │
         ┌──────────────────────┴──────────────────────┐
         ▼                                             ▼
[Semantic Search (Task 5)]                     [Lexical Search (Task 6)]
(Vector Store: Weaviate/Chroma)               (BM25 trên Corpus)
         │                                             │
         └──────────────────────┬──────────────────────┘
                                ▼
                   [Merge & Rerank (Task 7)]
                     (Jina Reranker / RRF)
                                │
                     (Kiểm tra Threshold)
                                ├─► [Nếu thấp] ──► [PageIndex Fallback (Task 8)]
                                ▼
                 [Reordered Context (Task 10)]
                     (Tránh Lost-in-the-Middle)
                                │
                                ▼
                 [Prompt Generation & LLM Call]
                     (Trả lời có kèm citation)
```

---

## 3. Phân Chia Công Việc Cho Nhóm 4 Người

Để tối ưu hóa hiệu suất và chuyên môn hóa công việc, dự án được chia làm 4 vai trò chính:

### Người 1: Backend Developer & System Integrator (Nhóm Trưởng)
* **Nhiệm vụ chính:** Tích hợp pipeline, quản lý Repository và điều phối kiến trúc.
* **Chi tiết công việc:**
  * Thu thập code của cả 4 thành viên cho các task cá nhân, chuẩn hóa thành các module dùng chung (`search`, `rerank`, `generation`, `utils`).
  * Khởi tạo và quản lý database (Vector Store chung cho cả nhóm, nạp đầy đủ dữ liệu từ phần cá nhân của các thành viên).
  * Xây dựng module `retrieval_pipeline.py` hoàn chỉnh, kết nối Semantic Search + Lexical Search + Reranker + Fallback PageIndex.
  * Hỗ trợ Member 2 kết nối Backend với UI và Member 3/4 kết nối Backend với Evaluation pipeline.
* **Sản phẩm bàn giao:** Thư mục `group_project/src/` chứa mã nguồn tích hợp hoàn chỉnh và file `.env` mẫu.

### Người 2: Frontend Developer (UI/UX)
* **Nhiệm vụ chính:** Xây dựng giao diện Chatbot tương tác và quản lý hội thoại.
* **Chi tiết công việc:**
  * Sử dụng **Streamlit** hoặc **Chainlit** để dựng giao diện Chatbot thân thiện, chuyên nghiệp.
  * Thiết kế khung chat hỗ trợ **Conversation Memory** (cho phép người dùng hỏi các câu hỏi tiếp nối - follow-up questions).
  * Hiển thị các tài liệu nguồn đã dùng (Source documents) ở dạng trực quan (ví dụ: dùng Sidebar hoặc Accordion/Expander để thu gọn/mở rộng).
  * Hiển thị câu trả lời dạng Markdown có định dạng trích dẫn nguồn nổi bật (ví dụ: click vào số trích dẫn sẽ cuộn tới tài liệu nguồn tương ứng).
* **Sản phẩm bàn giao:** File `group_project/app.py` chạy giao diện Chatbot.

### Người 3: Data Specialist & Evaluator 1 (Thiết kế Bộ Đánh Giá)
* **Nhiệm vụ chính:** Xây dựng Golden Dataset và thiết lập khung đánh giá.
* **Chi tiết công việc:**
  * Biên soạn **Golden Dataset** tối thiểu 15 cặp Q&A chất lượng cao (`question`, `expected_answer`, `expected_context`) dựa trên các văn bản pháp luật và tin tức thực tế trong cơ sở dữ liệu của nhóm.
  * Lưu trữ bộ dataset này dưới dạng tệp `group_project/evaluation/golden_dataset.json`.
  * Lựa chọn framework đánh giá (Khuyến nghị **DeepEval** hoặc **Ragas**).
  * Viết mã nguồn khởi tạo cho quá trình đánh giá (load dataset, định nghĩa các metric: *Faithfulness*, *Answer Relevance*, *Context Recall*, *Context Precision*).
* **Sản phẩm bàn giao:** `group_project/evaluation/golden_dataset.json` và một phần mã nguồn đánh giá cơ bản.

### Người 4: Evaluator 2 & Analyst (Chạy Eval & Viết Báo Cáo)
* **Nhiệm vụ chính:** Chạy đánh giá so sánh A/B và phân tích kết quả cải tiến.
* **Chi tiết công việc:**
  * Viết script chạy đánh giá hoàn chỉnh `group_project/evaluation/eval_pipeline.py`.
  * Thực hiện đánh giá **So sánh A/B** trên ít nhất 2 cấu hình RAG khác nhau:
    * *Cấu hình A (Baseline):* Chỉ dùng Dense Semantic Search, không dùng Reranker.
    * *Cấu hình B (Cải tiến):* Hybrid Search (Dense + Lexical) + Jina Reranker + Fallback PageIndex.
  * Phân tích các trường hợp chạy tệ nhất (worst performers) của hệ thống: Tại sao điểm Faithfulness hoặc Context Recall lại thấp?
  * Đề xuất phương án cải tiến và tổng hợp kết quả thành file báo cáo chi tiết.
* **Sản phẩm bàn giao:** File script `group_project/evaluation/eval_pipeline.py` và báo cáo phân tích `group_project/evaluation/results.md`.

---

## 4. Kế Hoạch Triển Khai & Phối Hợp (Timeline)

| Giai đoạn | Công việc chính | Người chịu trách nhiệm | Thời hạn |
| :--- | :--- | :--- | :--- |
| **Giai đoạn 1: Chuẩn bị** | Thống nhất cấu trúc thư mục nhóm, tổng hợp dữ liệu pháp luật/tin tức chung. | Cả nhóm | Ngày 1 |
| **Giai đoạn 2: Phát triển song song** | - Người 1: Tích hợp RAG Pipeline & Database.<br>- Người 2: Dựng khung UI Chatbot.<br>- Người 3: Soạn Golden Dataset (15+ Q&A). | Người 1, 2, 3 | Ngày 2-3 |
| **Giai đoạn 3: Kết nối & Eval** | - Người 1 & 2: Kết nối UI với RAG Pipeline.<br>- Người 3 & 4: Viết script `eval_pipeline.py` và chạy thử nghiệm. | Cả nhóm | Ngày 4 |
| **Giai đoạn 4: So sánh & Báo cáo** | Chạy đánh giá A/B, thu thập kết quả, viết báo cáo `results.md` và tinh chỉnh chatbot. | Người 4 (chính) & Người 1 | Ngày 5 |
| **Giai đoạn 5: Hoàn thiện** | Test toàn bộ hệ thống, viết README mô tả dự án và phân công, chuẩn bị demo thuyết trình. | Cả nhóm | Ngày 6 |

---

## 5. Hướng Dẫn Tích Hợp (Dành cho Người 1)

Để tích hợp mã nguồn của các thành viên một cách sạch sẽ:
1. Đảm bảo cấu trúc thư mục của nhóm giống như cấu trúc gợi ý trong bài tập nhóm.
2. Tạo file `group_project/src/config.py` để quản lý tập trung các biến môi trường và cấu hình (LLM API keys, Model name, Chunk size, Database connection info).
3. Đóng gói các hàm tìm kiếm và sinh văn bản thành một Class hoặc tập hợp các function chuẩn hóa:
   ```python
   # group_project/src/rag_pipeline.py
   
   class GroupRAGPipeline:
       def __init__(self, config):
           # Khởi tạo vector store, embedding, reranker, LLM
           pass
           
       def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
           # Chạy Hybrid search + Reranker + Fallback
           pass
           
       def generate(self, query: str, conversation_history: list = None) -> dict:
           # Trả về câu trả lời có citation kèm danh sách context nguồn
           pass
   ```
4. Người 2 và Người 4 chỉ cần gọi class này để phục vụ UI chatbot và đánh giá.
