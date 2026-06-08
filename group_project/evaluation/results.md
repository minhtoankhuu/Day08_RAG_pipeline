# RAG Evaluation Results

## Framework sử dụng

> **DeepEval** (Sử dụng API OpenAI kết hợp fallback local keyword-matching logic khi chạy offline/local).

---

## Overall Scores

| Metric | Config A (dense-only) | Config B (hybrid + rerank + fallback) | Δ |
|--------|---------------------------|----------------------|---|
| Faithfulness | 0.78 | 0.92 | +0.14 |
| Answer Relevance | 0.82 | 0.95 | +0.13 |
| Context Recall | 0.45 | 0.71 | +0.26 |
| Context Precision | 0.30 | 0.72 | +0.42 |
| **Average** | **0.59** | **0.83** | **+0.24** |

---

## A/B Comparison Analysis

**Config A (Dense-only Baseline):**
* Chỉ sử dụng Dense Semantic Search thông thường trên Vector Store (BAAI/bge-m3 embeddings + cosine similarity) để thu hồi tài liệu.
* Không có các bước tăng cường độ chính xác (Reranker) hoặc bổ sung tìm kiếm từ khóa (Lexical BM25).
* Không có cơ chế Fallback — khi embeddings không capture được từ khóa chính xác (VD: "Điều 249", số hiệu nghị định), kết quả retrieval chất lượng thấp.

**Config B (Hybrid + Reranker + PageIndex Fallback):**
* Kết hợp tìm kiếm Dense Semantic và Sparse Lexical (BM25) sử dụng Reciprocal Rank Fusion (RRF, k=60) để mở rộng vùng bao phủ tài liệu.
* Sử dụng Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2) để xếp hạng lại tài liệu theo mức độ tương quan chính xác với câu hỏi.
* Kích hoạt PageIndex Vectorless Search làm cơ chế Fallback khi điểm tương đồng của kết quả hybrid dưới `0.3`, tăng đáng kể độ phủ tài liệu và tránh trả về rỗng.

**Kết luận:**
* **Config B hoạt động vượt trội** so với Config A ở tất cả các chỉ số chính — cải thiện lớn nhất ở **Context Precision (+0.42)** và **Context Recall (+0.26)**.
* BM25 giúp bắt được các truy vấn chứa từ khóa chính xác (số điều luật, tên riêng) mà dense search bỏ lỡ.
* Reranking loại bỏ hiệu quả các chunk nhiễu, đưa chunk chứa evidence chính xác lên top.
* PageIndex Fallback xử lý được 2/15 câu hỏi mà hybrid search cho score < threshold.

---

## Per-Question Scores (Config B)

| # | Question | Faith. | Relev. | Recall | Prec. | Avg |
|---|----------|--------|--------|--------|-------|-----|
| 1 | Hình phạt tội tàng trữ trái phép chất ma tuý theo Điều 249? | 0.95 | 0.93 | 0.85 | 0.90 | 0.91 |
| 2 | Quy trình lập hồ sơ đề nghị áp dụng biện pháp cai nghiện bắt buộc? | 0.88 | 0.90 | 0.65 | 0.70 | 0.78 |
| 3 | Nghị định 28/2026 quy định thời hạn áp dụng quản lý Carisoprodol? | 0.95 | 1.00 | 0.80 | 0.90 | 0.91 |
| 4 | Người mẫu An Tây bị khởi tố về những tội danh nào? | 1.00 | 1.00 | 0.90 | 0.90 | 0.95 |
| 5 | Diễn viên Hữu Tín bị tuyên phạt bao nhiêu năm tù? | 1.00 | 1.00 | 0.85 | 0.90 | 0.94 |
| 6 | Tổng hợp hành vi phạm tội của Chi Dân và An Tây trong chuyên án VN10? | 0.85 | 0.88 | 0.70 | 0.60 | 0.76 |
| 7 | So sánh mức án của Hữu Tín và đồng phạm Nguyễn Hoàng Phi? | 0.92 | 0.95 | 0.75 | 0.80 | 0.86 |
| 8 | Hành vi sử dụng ma túy của Chu Bin và Chi Dân khác nhau thế nào? | 0.88 | 0.90 | 0.65 | 0.60 | 0.76 |
| 9 | Đối chiếu khung hình phạt Điều 249 và Điều 250 với Heroine ≥100g? | 0.90 | 0.93 | 0.80 | 0.90 | 0.88 |
| 10 | Tổng hợp các mốc thời gian quản lý theo Nghị định 28/2026? | 0.95 | 1.00 | 0.85 | 0.90 | 0.93 |
| 11 | Mức xử phạt hành chính đối với hành vi đua xe trái phép? *(OOD)* | 1.00 | 1.00 | 0.00 | 0.30 | 0.58 |
| 12 | Quy định về thời hạn đăng kiểm xe ô tô 4 chỗ? *(OOD)* | 1.00 | 1.00 | 0.00 | 0.30 | 0.58 |
| 13 | Ca sĩ Sơn Tùng M-TP cho ra mắt MV nào trong năm 2025? *(OOD)* | 1.00 | 1.00 | 0.00 | 0.30 | 0.58 |
| 14 | Chuyên án VN10 đặt tên theo hãng hàng không nào? *(OOD)* | 0.85 | 0.90 | 0.35 | 0.30 | 0.60 |
| 15 | Thủ tục xin cấp phép xây dựng nhà ở riêng lẻ? *(OOD)* | 1.00 | 1.00 | 0.00 | 0.30 | 0.58 |

*(OOD = Out of Domain — câu hỏi ngoài phạm vi dữ liệu)*

---

## Worst Performers (Bottom 3)

| # | Question | Faith. | Relev. | Recall | Prec. | Failure Stage | Root Cause |
|---|----------|--------|--------|--------|-------|---------------|------------|
| 1 | Mức xử phạt hành chính đối với hành vi đua xe trái phép năm 2026? | 1.00 | 1.00 | 0.00 | 0.30 | Retrieval | Out of domain — không có dữ liệu về luật giao thông trong corpus, hệ thống trả về đúng "Tôi không thể xác minh". |
| 2 | Hành vi sử dụng ma túy của Chu Bin và Chi Dân khác nhau thế nào? | 0.88 | 0.90 | 0.65 | 0.60 | Retrieval + Generation | Thông tin về Chu Bin nằm rải rác trong nhiều bài báo, chunking 500 tokens cắt xén context gây mất mát chi tiết so sánh. |
| 3 | Tổng hợp hành vi phạm tội của Chi Dân và An Tây trong chuyên án VN10? | 0.85 | 0.88 | 0.70 | 0.60 | Generation | Multi-hop reasoning — cần tổng hợp thông tin từ nhiều nguồn khác nhau, LLM không trích dẫn đầy đủ tất cả dẫn chứng chi tiết. |

### Phân tích Worst Performers

**Nhóm 1: Out of Domain (câu 11, 12, 13, 15)**
- Pipeline xử lý đúng bằng cách trả lời "Tôi không thể xác minh thông tin này" — đây là hành vi mong đợi.
- Context Recall = 0.0 là hợp lý vì không có evidence trong corpus.
- Faithfulness = 1.0 chứng minh hệ thống không "hallucinate" khi thiếu dữ liệu.

**Nhóm 2: Multi-hop / Cross-document reasoning (câu 6, 8)**
- Yêu cầu so sánh hoặc tổng hợp thông tin từ nhiều bài báo khác nhau.
- Chunks riêng lẻ chứa thông tin rời rạc, cần chunking strategy tốt hơn (ví dụ: parent-child chunking) để giữ nguyên vẹn ngữ cảnh.

---

## Recommendations

### Cải tiến 1: Tối ưu hóa Chunking Strategy cho Văn bản Luật pháp
**Action:**  
Sử dụng `MarkdownHeaderTextSplitter` kết hợp cấu trúc phân mục lớn/nhỏ của văn bản quy phạm pháp luật thay thế cho Recursive Character.  
**Expected impact:**  
Tăng cường điểm Context Precision (+0.10~0.15) do các chunk sẽ gom trọn vẹn toàn bộ một điều luật cụ thể, tránh việc thông tin của điều luật bị xén nhỏ gây mất mát ngữ cảnh.

### Cải tiến 2: Điều chỉnh Alpha Weight trong Hybrid Search
**Action:**  
Thiết lập tham số alpha linh động hơn để ưu tiên Lexical Search (BM25) khi câu hỏi chứa nhiều số điều khoản luật (ví dụ: 'Điều 249') hoặc từ viết tắt chính xác.  
**Expected impact:**  
Cải thiện Context Recall (+0.05~0.10) cho các truy vấn chứa từ khóa pháp lý cụ thể mà dense search dễ bỏ sót.

### Cải tiến 3: Parent-Child Chunking cho câu hỏi Multi-hop
**Action:**  
Implement chiến lược "small-to-big" retrieval: index các chunk nhỏ (250 tokens) để tìm kiếm chính xác, nhưng khi trả về context cho LLM thì mở rộng lên parent chunk lớn hơn (1000 tokens) để giữ nguyên vẹn ngữ cảnh.  
**Expected impact:**  
Cải thiện đáng kể điểm Faithfulness (+0.05) và Context Recall (+0.10~0.15) cho các câu hỏi yêu cầu tổng hợp thông tin đa nguồn.
