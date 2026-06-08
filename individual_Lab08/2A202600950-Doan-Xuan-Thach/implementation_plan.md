# Kế hoạch triển khai RAG Pipeline v2 (Day 8 Lab & Group Project)

Dự án này yêu cầu xây dựng một RAG (Retrieval-Augmented Generation) pipeline thực tế, end-to-end từ việc thu thập dữ liệu pháp luật và báo chí liên quan đến ma túy, qua quá trình xử lý, index vào vector database, thực hiện search hybrid (semantic + lexical), rerank kết quả, fallback thông qua PageIndex (vectorless) và generate câu trả lời có trích dẫn nguồn (citation). Đồng thời xây dựng sản phẩm nhóm gồm chatbot UI và evaluation pipeline.

---

## User Review Required

> [!IMPORTANT]
> **API Keys cần chuẩn bị:**
> Dự án yêu cầu các API keys sau để hoạt động đầy đủ:
> 1. `OPENAI_API_KEY`: Cho nhiệm vụ generation (Task 10) và evaluation (DeepEval/RAGAS).
> 2. `JINA_API_KEY`: Cho nhiệm vụ Reranking bằng Cross-encoder (Task 7) (nếu chọn Jina Reranker).
> 3. `PAGEINDEX_API_KEY`: Cho nhiệm vụ Vectorless RAG fallback (Task 8).
>
> Vui lòng cấu hình các API Key này trong file `.env`.

> [!TIP]
> **Lựa chọn Vector Store & Reranker:**
> - **Vector Store:** Chúng tôi đề xuất sử dụng **ChromaDB** hoặc **FAISS** thay vì Weaviate nếu không muốn chạy Docker cục bộ. Cả hai thư viện này đều chạy in-process và rất dễ dàng để cài đặt thông qua Python.
> - **Reranker:** Chúng tôi đề xuất sử dụng **RRF (Reciprocal Rank Fusion)** hoặc **MMR (Maximal Marginal Relevance)** tự triển khai (local) để tránh phụ thuộc vào API Key của Jina AI và giúp chạy nhanh hơn.

---

## Open Questions

> [!IMPORTANT]
> 1. **Bạn muốn sử dụng Vector Store nào?**
>    - **Option 1 (Khuyên dùng):** **ChromaDB** (lưu file local, nhẹ nhàng, hỗ trợ metadata filtering).
>    - **Option 2:** **Weaviate** (yêu cầu Docker container chạy local hoặc Weaviate Cloud instance).
>    - **Option 3:** **FAISS** (in-memory hoặc lưu file local, chỉ hỗ trợ dense search, cần tự map metadata).
>
> 2. **Bạn đã có tài khoản và API Key cho PageIndex chưa?**
>    - Nếu chưa, chúng ta có thể đăng ký miễn phí tại [pageindex.ai](https://pageindex.ai/) hoặc mock một phần nếu không muốn đăng ký.
>
> 3. **Lựa chọn Framework giao diện cho Chatbot Nhóm:**
>    - Chúng tôi đề xuất sử dụng **Streamlit** vì tính đơn giản, dễ build và có sẵn tích hợp giao diện chat rất trực quan.

---

## Proposed Changes

Chúng ta sẽ chỉnh sửa các file mã nguồn hiện có trong thư mục `src/` và `group_project/` để hiện thực hóa các chức năng của từng task.

### 1. Thu thập và chuyển đổi dữ liệu (Task 1 - 3)

#### [MODIFY] [task1_collect_legal_docs.py](file:///d:/ai_course/Day_8/Day08_Lab08-2A202600950-DoanXuanThach/src/task1_collect_legal_docs.py)
- Triển khai tải/chuẩn bị tối thiểu 3 văn bản pháp luật về ma túy (định dạng PDF/DOCX) lưu vào `data/landing/legal/`.
- File gợi ý:
  - `luat-phong-chong-ma-tuy-2021.pdf`
  - `nghi-dinh-105-2021-nd-cp.pdf`
  - `bo-luat-hinh-su-2015-chuong-xx.pdf`

#### [MODIFY] [task2_crawl_news.py](file:///d:/ai_course/Day_8/Day08_Lab08-2A202600950-DoanXuanThach/src/task2_crawl_news.py)
- Sử dụng Crawl4AI (hoặc HTTP requests & BeautifulSoup/Markdownify fallback) để cào tối thiểu 5 bài báo từ VnExpress/ThanhNien/Tuoitre liên quan tới nghệ sĩ Việt Nam và ma túy.
- Lưu trữ dưới dạng JSON với đầy đủ metadata: `url`, `title`, `date_crawled`, `content_markdown`.

#### [MODIFY] [task3_convert_markdown.py](file:///d:/ai_course/Day_8/Day08_Lab08-2A202600950-DoanXuanThach/src/task3_convert_markdown.py)
- Sử dụng thư viện `markitdown` để chuyển đổi tài liệu PDF/DOCX từ `data/landing/legal/` và trích xuất dữ liệu JSON từ `data/landing/news/` sang định dạng `.md` lưu vào `data/standardized/`.

---

### 2. Chunking & Search Modules (Task 4 - 6)

#### [MODIFY] [task4_chunking_indexing.py](file:///d:/ai_course/Day_8/Day08_Lab08-2A202600950-DoanXuanThach/src/task4_chunking_indexing.py)
- Áp dụng `RecursiveCharacterTextSplitter` với `CHUNK_SIZE = 500` và `CHUNK_OVERLAP = 50`.
- Sử dụng embedding model `BAAI/bge-m3` (chạy offline qua HuggingFace SentenceTransformers).
- Triển khai lưu trữ vào Vector Store đã chọn (mặc định đề xuất ChromaDB để dễ chạy local).

#### [MODIFY] [task5_semantic_search.py](file:///d:/ai_course/Day_8/Day08_Lab08-2A202600950-DoanXuanThach/src/task5_semantic_search.py)
- Nhận diện câu truy vấn, chuyển đổi sang embedding vector, tìm kiếm các vector tương đồng nhất trong Vector Store.
- Trả về danh sách định dạng chuẩn: `[{'content': str, 'score': float, 'metadata': dict}]` sắp xếp giảm dần theo score.

#### [MODIFY] [task6_lexical_search.py](file:///d:/ai_course/Day_8/Day08_Lab08-2A202600950-DoanXuanThach/src/task6_lexical_search.py)
- Triển khai thuật toán **BM25** sử dụng thư viện `rank-bm25` trên tập corpus các chunks được tạo từ Task 4.
- Trả về danh sách định dạng chuẩn, sắp xếp theo BM25 score giảm dần.

---

### 3. Advanced Retrieval & Fallback (Task 7 - 9)

#### [MODIFY] [task7_reranking.py](file:///d:/ai_course/Day_8/Day08_Lab08-2A202600950-DoanXuanThach/src/task7_reranking.py)
- Hiện thực hóa 3 phương pháp Rerank:
  1. `rerank_cross_encoder` (gọi Jina Reranker API).
  2. `rerank_mmr` (Maximal Marginal Relevance) tự triển khai bằng cosine similarity để tăng tính đa dạng cho thông tin.
  3. `rerank_rrf` (Reciprocal Rank Fusion) để merge kết quả rank từ Dense và Sparse retrieval.

#### [MODIFY] [task8_pageindex_vectorless.py](file:///d:/ai_course/Day_8/Day08_Lab08-2A202600950-DoanXuanThach/src/task8_pageindex_vectorless.py)
- Kết nối tới PageIndex qua SDK `pageindex`, upload tài liệu Markdown lên dịch vụ và truy vấn dữ liệu dưới dạng vectorless search.

#### [MODIFY] [task9_retrieval_pipeline.py](file:///d:/ai_course/Day_8/Day08_Lab08-2A202600950-DoanXuanThach/src/task9_retrieval_pipeline.py)
- Kết hợp: Semantic Search + Lexical Search -> Merge (RRF) -> Rerank.
- Fallback logic: Nếu top score của hybrid search thấp hơn `score_threshold` (mặc định 0.3), tự động fallback gọi PageIndex search.

---

### 4. Generation & Chatbot (Task 10 & Group Project)

#### [MODIFY] [task10_generation.py](file:///d:/ai_course/Day_8/Day08_Lab08-2A202600950-DoanXuanThach/src/task10_generation.py)
- Triển khai reordering để tránh "lost in the middle" (nhập `[1, 2, 3, 4, 5]` -> xuất `[1, 3, 5, 4, 2]`).
- Format context và xây dựng Prompt yêu cầu LLM (OpenAI/Gemini) trả lời bằng tiếng Việt kèm trích dẫn (ví dụ: `[Luật Phòng chống ma tuý 2021, Điều 3]`).
- Trả về câu trả lời hoàn thiện hoặc thông báo "Tôi không thể xác minh thông tin..." nếu thiếu context.

#### [NEW] [app.py](file:///d:/ai_course/Day_8/Day08_Lab08-2A202600950-DoanXuanThach/app.py)
- Giao diện Streamlit chatbot: cho phép nhập câu hỏi, lưu lịch sử trò chuyện (memory), hiển thị câu trả lời trích dẫn nguồn, và show các source document chunks được truy xuất làm minh chứng.

#### [MODIFY] [golden_dataset.json](file:///d:/ai_course/Day_8/Day08_Lab08-2A202600950-DoanXuanThach/group_project/evaluation/golden_dataset.json)
- Bổ sung tối thiểu 15 câu hỏi - đáp - ngữ cảnh liên quan đến ma túy để làm bộ dữ liệu kiểm thử vàng (Golden Dataset).

#### [MODIFY] [eval_pipeline.py](file:///d:/ai_course/Day_8/Day08_Lab08-2A202600950-DoanXuanThach/group_project/evaluation/eval_pipeline.py)
- Hiện thực hóa đánh giá bằng **DeepEval** hoặc **RAGAS** với 4 chỉ số: Faithfulness, Answer Relevance, Context Recall, Context Precision.
- Triển khai chức năng so sánh A/B giữa 2 cấu hình (ví dụ: Rerank vs No Rerank).
- Xuất kết quả chi tiết ra file `results.md`.

---

## Verification Plan

### Automated Tests
Chúng ta sẽ chạy bộ test tự động của bài lab:
```powershell
# Chạy toàn bộ test cá nhân
pytest tests/test_individual.py -v
```

### Manual Verification
1. Chạy các file đơn lẻ `task1_collect_legal_docs.py`, `task2_crawl_news.py`, `task3_convert_markdown.py`, `task4_chunking_indexing.py` để kiểm tra dữ liệu được tạo ra trong thư mục `data/`.
2. Khởi chạy Streamlit ứng dụng chatbot để kiểm nghiệm giao diện và tính chính xác của câu trả lời:
   ```powershell
   streamlit run app.py
   ```
3. Chạy `eval_pipeline.py` để thực hiện đánh giá RAG và tạo báo cáo `results.md`.
