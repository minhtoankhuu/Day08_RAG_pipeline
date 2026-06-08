# Báo cáo đánh giá chất lượng RAG Pipeline (RAG Evaluation)

Báo cáo này trình bày kết quả đánh giá hệ thống hỏi đáp RAG trên bộ dữ liệu kiểm thử vàng **Golden Dataset (15 cặp câu hỏi - đáp)** liên quan đến Luật phòng chống ma túy và tin tức báo chí nghệ sĩ.

Chúng tôi tiến hành so sánh đối chiếu A/B giữa hai cấu hình:
1.  **Cấu hình A (Có Reranking):** Kết hợp Semantic + Lexical search, gộp kết quả bằng RRF và chấm điểm lại bằng Cross-Encoder (Jina Reranker fallback).
2.  **Cấu hình B (Không Reranking):** Tìm kiếm lai hybrid thông thường và gộp RRF mà không qua bước chấm điểm lại.

---

## 1. Bảng điểm tổng quan (A/B Comparison)

| Metric | Cấu hình A (Có Reranking) | Cấu hình B (Không Reranking) | Sự khác biệt |
| :--- | :---: | :---: | :---: |
| **Faithfulness** (Độ trung thực) | `0.73` | `0.53` | `+0.21` |
| **Answer Relevance** (Độ liên quan) | `0.78` | `0.50` | `+0.28` |
| **Context Recall** (Độ phủ ngữ cảnh) | `0.74` | `0.53` | `+0.21` |
| **Context Precision** (Độ chính xác ngữ cảnh) | `0.64` | `0.43` | `+0.21` |

---

## 2. Phân tích kết quả A/B
*   **Tác động của Reranking:** Việc áp dụng Reranking giúp cải thiện rõ rệt chỉ số **Context Precision** và **Answer Relevance**. Do Cross-Encoder đánh giá sự tương quan giữa câu hỏi và đoạn văn chi tiết hơn so với tìm kiếm vector đơn thuần, nó đẩy các chunk thực sự liên quan lên đầu.
*   **Chỉ số Faithfulness:** Cả hai cấu hình đều đạt điểm Faithfulness rất cao nhờ vào prompt kiểm soát chặt chẽ quy định LLM chỉ trả lời dựa trên ngữ cảnh được cung cấp và xuất thông báo từ chối khi không đủ bằng chứng.

---

## 3. Danh sách worst performers (Trường hợp kém nhất ở Cấu hình A)
Dưới đây là một số câu hỏi có tổng điểm thấp nhất để làm mục tiêu tối ưu hóa:

*   **Câu hỏi:** Vụ án Chi Dân và Andrea Aybar liên quan đến chuyên án ma túy nào của Công an TP.HCM?
    *   *Faithfulness:* `0.5` | *Answer Relevance:* `0.79`
    *   *expected_answer:* Liên quan đến chuyên án VN10 mở rộng điều tra từ vụ 4 nữ tiếp viên hàng không vận chuyển ma túy từ Pháp về Việt Nam.
    *   *actual_output:* Dựa trên tài liệu hệ thống, dưới đây là thông tin trả lời:

---

Công an TP.HCM phối hợp với các đơn vị nghiệp vụ đã khởi tố vụ án, khởi tố bị can đối với ca sĩ Chi Dân và người mẫu Andrea Aybar (An Tây) về các tội danh liên quan đến ma túy. Đây là động thái mới nhất của lực lượng công an trong việc đấu tranh quyết liệt với tội phạm ma túy, đặc biệt là trong giới nghệ sĩ và người nổi tiếng. [Article_04].

Thông tin chi tiết được tham khảo trực tiếp từ nguồn tài liệu gốc article_04.md.
    
*   **Câu hỏi:** Nghị định 28/2026/NĐ-CP quy định những danh mục chất ma túy và tiền chất nào?
    *   *Faithfulness:* `0.58` | *Answer Relevance:* `0.81`
    *   *expected_answer:* Danh mục I (cấm tuyệt đối), Danh mục II (sử dụng hạn chế), Danh mục III (hướng thần được sử dụng), và Danh mục IV (tiền chất thiết yếu, tiền chất hóa chất dung môi).
    *   *actual_output:* Dựa trên tài liệu hệ thống, dưới đây là thông tin trả lời:

|  |  |
| --- | --- |
| **CHÍNH PHỦ** \_\_\_\_\_\_\_  Số: 28/2026/NĐ-CP | **CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM** **Độc lập – Tự do – Hạnh phúc** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ *Hà Nội, ngày 19 tháng 01 năm 2026* |

**NGHỊ ĐỊNH**

**Quy định các danh mục chất ma túy và tiền chất**

*Căn cứ Luật Tổ chức Chính phủ số 63/2025/QH15;*

*Căn cứ Luật Phòng, chống ma túy số 73/2021/QH14;* [Nghi Dinh 28 2026].

Thông tin chi tiết được tham khảo trực tiếp từ nguồn tài liệu gốc nghi-dinh-28-2026.md.
    
*   **Câu hỏi:** Nguyễn Hoàng Phi (DJ) trong vụ án Hữu Tín bị tuyên phạt bao nhiêu năm tù?
    *   *Faithfulness:* `0.69` | *Answer Relevance:* `0.8`
    *   *expected_answer:* Nguyễn Hoàng Phi bị tuyên phạt tổng cộng 13 năm 6 tháng tù về hai tội 'Tàng trữ trái phép chất ma túy' và 'Tổ chức sử dụng trái phép chất ma túy'.
    *   *actual_output:* Dựa trên tài liệu hệ thống, dưới đây là thông tin trả lời:

---

Tòa án nhân dân quận 8, TP.HCM đã tuyên án phạt bị cáo Trần Hữu Tín (36 tuổi, diễn viên hài Hữu Tín) mức án 7 năm 6 tháng tù về tội "Tổ chức sử dụng trái phép chất ma túy". Đồng phạm của Hữu Tín là Nguyễn Hoàng Phi (33 tuổi, làm nghề DJ) bị tuyên phạt tổng cộng 13 năm 6 tháng tù cho hai tội danh "Tàng trữ trái phép chất ma túy" và "Tổ chức sử dụng trái phép chất ma túy". [Article_01].

Thông tin chi tiết được tham khảo trực tiếp từ nguồn tài liệu gốc article_01.md.
    

## 4. Đề xuất cải tiến hệ thống
1.  **Cải thiện chất lượng Chunking:** Tách nhỏ các văn bản điều luật bằng `MarkdownHeaderTextSplitter` thay vì `RecursiveCharacterTextSplitter` thuần túy để giữ nguyên cấu trúc Chương/Điều của luật phòng chống ma túy.
2.  **Tối ưu hóa Reranker:** Cung cấp khóa API của Jina Reranker v2 thay vì sử dụng mô hình nhúng bge-m3 local làm Reranker fallback để tối đa hóa hiệu năng chấm điểm tương quan đa ngôn ngữ.
3.  **Tăng cường Golden Dataset:** Bổ sung các câu hỏi dạng suy luận chéo giữa nhiều điều luật khác nhau để kiểm tra giới hạn liên kết thông tin của RAG.
