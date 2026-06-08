import streamlit as st
import os
import sys
import time
from pathlib import Path

# Cấu hình mã hóa UTF-8 để tránh lỗi hiển thị Unicode trên Windows
if sys.stdout.encoding.lower() != 'utf-8':
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

# Khởi tạo đường dẫn thư mục làm việc để có thể import các module từ src
current_dir = Path(__file__).parent.resolve()
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Thử import pipeline thực tế
real_backend_available = False
try:
    from src.task10_generation import generate_with_citation  # type: ignore
    real_backend_available = True
except ImportError:
    try:
        from task10_generation import generate_with_citation  # type: ignore
        real_backend_available = True
    except ImportError:
        pass

# =============================================================================
# 1. MOCK RAG ENGINE
# =============================================================================
class MockRAGEngine:
    """Giả lập RAG pipeline phục vụ phát triển & kiểm thử giao diện Frontend độc lập."""
    
    @staticmethod
    def generate(query: str) -> dict:
        time.sleep(1.5)  # Giả lập thời gian xử lý của LLM & Retrieval
        q_lower = query.lower()
        
        # Scenario 1: Cai nghiện / Luật 2021
        if "cai nghiện" in q_lower or "luật 2021" in q_lower or "luat 2021" in q_lower:
            answer = (
                "Theo quy định tại **Luật Phòng, chống ma túy 2021** (Luật số 73/2021/QH15), "
                "biện pháp cai nghiện ma túy bao gồm cai nghiện tự nguyện và cai nghiện bắt buộc [Luật Phòng chống ma tuý 2021, Điều 28].\n\n"
                "Người nghiện ma túy từ đủ 18 tuổi trở lên bị áp dụng biện pháp đưa vào cơ sở cai nghiện bắt buộc "
                "khi thuộc một trong các trường hợp sau đây [Luật Phòng chống ma tuý 2021, Điều 32]:\n"
                "1. Không đăng ký, không thực hiện hoặc tự ý chấm dứt cai nghiện tự nguyện.\n"
                "2. Bị phát hiện sử dụng trái phép chất ma túy trong thời gian cai nghiện tự nguyện.\n"
                "3. Người nghiện ma túy các chất dạng thuốc phiện không đăng ký điều trị thay thế."
            )
            sources = [
                {
                    "content": "Điều 28. Biện pháp cai nghiện ma túy. Có các biện pháp cai nghiện ma túy sau đây: a) Cai nghiện ma túy tự nguyện; b) Cai nghiện ma túy bắt buộc.",
                    "score": 0.925,
                    "metadata": {"source": "luat-phong-chong-ma-tuy-2021.md", "type": "legal"}
                },
                {
                    "content": "Điều 32. Đối tượng bị áp dụng biện pháp đưa vào cơ sở cai nghiện bắt buộc. Người nghiện ma túy từ đủ 18 tuổi trở lên bị áp dụng biện pháp đưa vào cơ sở cai nghiện bắt buộc khi thuộc một trong các trường hợp sau: a) Không đăng ký cai nghiện tự nguyện...",
                    "score": 0.887,
                    "metadata": {"source": "luat-phong-chong-ma-tuy-2021.md", "type": "legal"}
                }
            ]
            retrieval_source = "hybrid"
            
        # Scenario 2: Hình phạt / Tàng trữ / Tù
        elif "hình phạt" in q_lower or "tàng trữ" in q_lower or "tang tru" in q_lower or "tù" in q_lower:
            answer = (
                "Căn cứ theo quy định của **Bộ luật Hình sự 2015 (sửa đổi, bổ sung 2017)** tại Chương XX: Các tội phạm về ma túy:\n\n"
                "- Tội tàng trữ trái phép chất ma túy (Điều 249) sẽ bị phạt tù từ 01 năm đến 05 năm đối với trường hợp tàng trữ nhựa thuốc phiện, nhựa cần sa có khối lượng từ 500 gam đến dưới 01 kilôgam; Heroine, Cocaine, Methamphetamine có khối lượng từ 0,1 gam đến dưới 05 gam [Bộ luật Hình sự 2015, Điều 249].\n"
                "- Nếu tàng trữ với số lượng lớn hơn, hình phạt có thể tăng lên đến 20 năm tù hoặc tù chung thân [Bộ luật Hình sự 2015, Điều 249]."
            )
            sources = [
                {
                    "content": "Điều 249. Tội tàng trữ trái phép chất ma túy. 1. Người nào tàng trữ trái phép chất ma túy mà không nhằm mục đích mua bán, vận chuyển, sản xuất trái phép chất ma túy thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 01 năm đến 05 năm...",
                    "score": 0.941,
                    "metadata": {"source": "bo-luat-hinh-su-2015.md", "type": "legal"}
                },
                {
                    "content": "4. Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 15 năm đến 20 năm hoặc tù chung thân: a) Nhựa thuốc phiện, nhựa cần sa hoặc cao côca có khối lượng 05 kilôgam trở lên; b) Heroine, Cocaine, Methamphetamine có khối lượng 100 gam trở lên...",
                    "score": 0.895,
                    "metadata": {"source": "bo-luat-hinh-su-2015.md", "type": "legal"}
                }
            ]
            retrieval_source = "hybrid"
            
        # Scenario 3: Nghệ sĩ / Ca sĩ / Diễn viên / Chi Dân / Andrea / An Tây
        elif any(name in q_lower for name in ["nghệ sĩ", "ca sĩ", "diễn viên", "chi dân", "andrea", "an tây"]):
            answer = (
                "Dựa trên các tin tức báo chí thu thập được vào cuối năm 2024, nhiều nghệ sĩ Việt đã bị điều tra và bắt giữ vì liên quan đến chất cấm:\n\n"
                "- Ca sĩ **Chi Dân** (Nguyễn Trung Hiếu) bị lực lượng chức năng tạm giữ tại quận Tân Bình, TP.HCM để điều tra về hành vi tổ chức và sử dụng trái phép chất ma túy [Tuổi Trẻ, 2024].\n"
                "- Người mẫu **Andrea Aybar** (tên tiếng Việt là An Tây) bị khởi tố và tạm giam về các tội danh tàng trữ trái phép chất ma túy và tổ chức sử dụng trái phép chất ma túy [VnExpress, 2024].\n"
                "- Các vụ việc này đã làm dấy lên làn sóng phản đối mạnh mẽ từ công chúng và thúc đẩy các cơ quan văn hóa siết chặt quy trình kiểm soát đạo đức nghệ sĩ [Nhân Dân, 2024]."
            )
            sources = [
                {
                    "content": "Ca sĩ Chi Dân bị công an quận Tân Bình tạm giữ cùng một số người khác để điều tra hành vi liên quan đến sử dụng, tổ chức sử dụng trái phép chất ma túy...",
                    "score": 0.963,
                    "metadata": {"source": "news_chi_dan_tam_giu.html", "type": "news"}
                },
                {
                    "content": "Công an TP HCM khởi tố bị can, bắt tạm giam người mẫu An Tây (Andrea Aybar) về tội Tàng trữ trái phép chất ma túy và Tổ chức sử dụng trái phép chất ma túy...",
                    "score": 0.938,
                    "metadata": {"source": "news_an_tay_khoi_to.html", "type": "news"}
                }
            ]
            retrieval_source = "hybrid"
            
        # Scenario 4: Query không khớp (kích hoạt Fallback PageIndex)
        else:
            answer = (
                "Hệ thống không tìm thấy thông tin cụ thể khớp với câu hỏi của bạn trong dữ liệu nội bộ RAG.\n\n"
                "Kích hoạt cơ chế **PageIndex Fallback (Vectorless)** để trả về thông tin bổ trợ: "
                "Quy trình xử lý hành vi liên quan đến ma túy luôn yêu cầu có kết luận giám định từ cơ quan chuyên môn y tế hoặc pháp y để làm căn cứ xử lý hành chính hoặc hình sự [PageIndex, 2025]."
            )
            sources = [
                {
                    "content": "[PageIndex Fallback Document] Quy trình giám định hàm lượng chất ma túy bắt buộc phải được tiến hành bởi cơ quan giám định kỹ thuật hình sự có thẩm quyền để đưa ra kết luận định lượng chính xác.",
                    "score": 0.285,
                    "metadata": {"source": "PageIndex API", "type": "vectorless_fallback"}
                }
            ]
            retrieval_source = "pageindex"
            
        return {
            "answer": answer,
            "sources": sources,
            "retrieval_source": retrieval_source
        }

# =============================================================================
# 2. UI SETUP & STYLING
# =============================================================================
st.set_page_config(
    page_title="DrugLaw RAG Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Nhúng CSS tùy chỉnh để làm giao diện premium và hiện đại
st.markdown("""
<style>
/* Font chữ Outfit hiện đại */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Glassmorphic header */
.main-header {
    background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.8rem;
    margin-bottom: 0.2rem;
}

.sub-header {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-bottom: 1.5rem;
}

/* Glassmorphism Card cho sidebar hoặc thông báo */
.info-card {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 15px;
    margin-bottom: 15px;
}

/* Hiệu ứng mượt cho các nút bấm */
button {
    transition: all 0.2s ease-in-out !important;
}
button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2) !important;
}

/* Tinh chỉnh khu vực chat */
.stChatMessage {
    border-radius: 10px;
    margin-bottom: 10px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 3. SIDEBAR CONFIGURATION
# =============================================================================
with st.sidebar:
    st.markdown("### 🛠️ CẤU HÌNH PIPELINE")
    
    # Lựa chọn Engine (Mock vs Real)
    if real_backend_available:
        engine_mode = st.radio(
            "Chế độ Backend:",
            ["Mock Engine (Giả Lập)", "Real RAG Pipeline (Thực Tế)"],
            help="Chọn Mock để kiểm thử giao diện hoặc chọn Real để chạy thực tế với mô hình RAG của bạn."
        )
    else:
        st.warning("⚠️ Không tìm thấy `task10_generation.py` trong `src/`. Mặc định chạy Mock Engine.")
        engine_mode = "Mock Engine (Giả Lập)"
        
    st.markdown("---")
    st.markdown("### 🎛️ THAM SỐ THẾ HỆ (LLM)")
    
    model_name = st.selectbox(
        "Mô hình ngôn ngữ (LLM):",
        ["gpt-4o-mini (Mặc định)", "gemini-1.5-flash", "gemini-1.5-pro", "gpt-4o"],
        index=0
    )
    
    temperature = st.slider(
        "Độ sáng tạo (Temperature):",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Giá trị thấp giúp câu trả lời thực tế và chính xác hơn."
    )
    
    st.markdown("---")
    st.markdown("### 🔍 THAM SỐ RETRIEVAL")
    
    top_k = st.slider(
        "Số lượng văn bản gợi ý (Top K):",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )
    
    score_threshold = st.slider(
        "Ngưỡng điểm tương đồng (Threshold):",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Nếu điểm của văn bản hybrid thấp hơn ngưỡng này, hệ thống sẽ kích hoạt Fallback PageIndex."
    )
    
    st.markdown("---")
    if st.button("🗑️ Xóa Lịch Sử Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# =============================================================================
# 4. MAIN CHAT INTERFACE
# =============================================================================
st.markdown('<div class="main-header">⚖️ DrugLaw RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hệ thống hỏi đáp Luật Phòng chống ma túy và tin tức nghệ sĩ Việt liên quan có trích dẫn nguồn</div>', unsafe_allow_html=True)

# Khởi tạo lịch sử chat trong session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Xin chào! Tôi là trợ lý AI tìm kiếm và phân tích pháp luật ma túy. "
                "Tôi có thể hỗ trợ giải đáp các quy định pháp luật ma túy Việt Nam (Luật 2021, Bộ luật Hình sự 2015) "
                "và tin tức liên quan đến các nghệ sĩ bị bắt giữ vì chất cấm.\n\n"
                "**Bạn có thể thử đặt câu hỏi như:**\n"
                "- *Hình phạt cho tội tàng trữ trái phép chất ma tuý là gì?*\n"
                "- *Quy trình cai nghiện bắt buộc theo Luật 2021 quy định ra sao?*\n"
                "- *Những nghệ sĩ nào bị bắt vì ma túy cuối năm 2024?*"
            ),
            "sources": []
        }
    ]

# Hiển thị các tin nhắn cũ trong lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Nếu là trợ lý và có nguồn tài liệu, hiển thị bên dưới
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("🔍 Xem nguồn tài liệu tham khảo", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    meta = src.get("metadata", {})
                    src_file = meta.get("source", "Tài liệu")
                    src_type = meta.get("type", "unknown")
                    score = src.get("score", 0.0)
                    
                    st.markdown(f"**[{i}] Nguồn:** `{src_file}` | **Loại:** `{src_type}` | **Điểm tương quan:** `{score:.3f}`")
                    st.info(src["content"])

# Nhận truy vấn từ người dùng
if user_query := st.chat_input("Nhập câu hỏi của bạn tại đây..."):
    # Hiển thị câu hỏi của người dùng ngay lập tức
    with st.chat_message("user"):
        st.markdown(user_query)
    
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Gọi Backend xử lý
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner("Đang truy xuất tài liệu và tạo câu trả lời..."):
            if "Mock Engine" in engine_mode:
                # Chạy chế độ giả lập
                result = MockRAGEngine.generate(user_query)
            else:
                # Chạy chế độ thật
                try:
                    # Ghi đè cấu hình tạm thời (nếu task10 hỗ trợ tham số)
                    result = generate_with_citation(user_query, top_k=top_k)
                except Exception as e:
                    st.error(f"Lỗi khi chạy Backend RAG thực tế: {e}")
                    result = {
                        "answer": "Đã xảy ra lỗi khi gọi Backend thực tế. Vui lòng kiểm tra API Key hoặc log terminal.",
                        "sources": [],
                        "retrieval_source": "error"
                    }
            
            # Hiển thị câu trả lời
            response_placeholder.markdown(result["answer"])
            
            # Hiển thị nguồn tài liệu nếu có
            if result.get("sources"):
                with st.expander("🔍 Xem nguồn tài liệu tham khảo", expanded=False):
                    for i, src in enumerate(result["sources"], 1):
                        meta = src.get("metadata", {})
                        src_file = meta.get("source", "Tài liệu")
                        src_type = meta.get("type", "unknown")
                        score = src.get("score", 0.0)
                        
                        st.markdown(f"**[{i}] Nguồn:** `{src_file}` | **Loại:** `{src_type}` | **Điểm tương quan:** `{score:.3f}`")
                        st.info(src["content"])
                        
            # Thông tin thêm về luồng retrieval
            retrieval_info = f"*(Sử dụng: **{result.get('retrieval_source', 'n/a').upper()}** search)*"
            st.caption(retrieval_info)
            
            # Lưu vào lịch sử chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"] + f"\n\n{retrieval_info}",
                "sources": result.get("sources", [])
            })
