import streamlit as st
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load cấu hình môi trường
load_dotenv()

# Import RAG pipeline từ phần cá nhân
from src.task10_generation import generate_with_citation
from src.task9_retrieval_pipeline import retrieve

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="DrugLaw RAG - Trợ lý Pháp luật Ma túy",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Styling (Dark Theme & Glassmorphism)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sleek gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.9);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    /* Chat message container styles */
    .chat-message {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .chat-message.user {
        background-color: rgba(99, 102, 241, 0.15);
        border-left: 5px solid #6366f1;
    }
    
    .chat-message.assistant {
        background-color: rgba(30, 41, 59, 0.7);
        border-left: 5px solid #10b981;
    }
    
    .chat-sender {
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .chat-message.user .chat-sender {
        color: #818cf8;
    }
    
    .chat-message.assistant .chat-sender {
        color: #34d399;
    }
    
    /* Header card design */
    .header-container {
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.2) 0%, rgba(16, 185, 129, 0.1) 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #818cf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Source list styles */
    .source-card {
        background-color: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
    }
    
    .source-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: #94a3b8;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 0.3rem;
        margin-bottom: 0.5rem;
    }
    
    .source-score {
        font-weight: bold;
        color: #10b981;
    }
    
    .source-content {
        font-size: 0.85rem;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR CONFIGURATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1041/1041883.png", width=70)
    st.markdown("### ⚙️ Cấu hình RAG Pipeline")
    st.markdown("Tùy biến các tham số truy vấn bên dưới để tinh chỉnh chất lượng tìm kiếm.")

    # Chọn số lượng chunks đưa vào ngữ cảnh
    top_k = st.slider("Số lượng ngữ cảnh (K)", min_value=1, max_value=10, value=5, step=1)
    
    # Thiết lập ngưỡng điểm kích hoạt PageIndex fallback
    score_threshold = st.slider("Ngưỡng điểm lai (Threshold)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    
    # Chọn áp dụng Reranking hay không
    use_reranking = st.checkbox("Áp dụng Reranking", value=True)
    
    # Chọn giải thuật Rerank
    rerank_method = st.selectbox(
        "Phương pháp Rerank",
        options=["cross_encoder", "mmr", "rrf"],
        index=0,
        disabled=not use_reranking
    )
    
    st.markdown("---")
    
    # Trình diễn danh sách tài liệu hiện có trong Vector Store
    st.markdown("📂 **Tài liệu pháp lý & báo chí đã nạp:**")
    legal_docs = ["bo-luat-hinh-su-2015-ve-toi-pham-ma-tuy.md", "luat-phong-chong-ma-tuy-2021.md", "nghi-dinh-105-2021.md", "nghi-dinh-28-2026.md"]
    for doc in legal_docs:
        st.markdown(f"- 📄 `{doc}`")

    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.rerun()

# -----------------------------------------------------------------------------
# HEADER CARD
# -----------------------------------------------------------------------------
st.markdown("""
<div class="header-container">
    <div class="header-title">⚖️ Trợ Lý Pháp Luật & Tin Tức Ma Túy</div>
    <div style="font-size: 1.1rem; color: #cbd5e1;">Hệ thống hỏi đáp RAG v2 tối ưu hóa ngữ nghĩa cho luật phòng chống ma túy Việt Nam</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CONVERSATION MEMORY INITIALIZATION
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for idx, msg in enumerate(st.session_state.messages):
    sender = "user" if msg["role"] == "user" else "assistant"
    sender_title = "Bạn" if msg["role"] == "user" else "Trợ Lý AI"
    
    with st.container():
        st.markdown(f"""
        <div class="chat-message {sender}">
            <div class="chat-sender">{sender_title}</div>
            <div style="font-size: 1rem; line-height: 1.6; white-space: pre-wrap;">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CHAT INPUT & EXECUTION
# -----------------------------------------------------------------------------
if prompt := st.chat_input("Hãy hỏi tôi về hình phạt ma túy, luật cai nghiện hoặc các tin tức nghệ sĩ..."):
    # Hiển thị câu hỏi người dùng lập tức
    with st.container():
        st.markdown(f"""
        <div class="chat-message user">
            <div class="chat-sender">Bạn</div>
            <div style="font-size: 1rem; line-height: 1.6; white-space: pre-wrap;">{prompt}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Gọi RAG pipeline và sinh câu trả lời
    with st.spinner("AI đang tìm kiếm tài liệu pháp luật và soạn câu trả lời..."):
        start_time = time.time()
        
        # Gọi mô hình nhúng và sinh câu trả lời
        # Ghi đè cấu hình tạm thời từ UI lên file cấu hình của module
        import src.task9_retrieval_pipeline
        src.task9_retrieval_pipeline.DEFAULT_TOP_K = top_k
        src.task9_retrieval_pipeline.SCORE_THRESHOLD = score_threshold
        src.task9_retrieval_pipeline.RERANK_METHOD = rerank_method
        
        result = generate_with_citation(prompt, top_k=top_k, use_reranking=use_reranking)
        
        elapsed_time = time.time() - start_time

    # Hiển thị câu trả lời của trợ lý
    answer_text = result["answer"]
    with st.container():
        st.markdown(f"""
        <div class="chat-message assistant">
            <div class="chat-sender">Trợ Lý AI</div>
            <div style="font-size: 1rem; line-height: 1.6; white-space: pre-wrap;">{answer_text}</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 1rem; text-align: right;">
                Thời gian xử lý: {elapsed_time:.2f}s | Nguồn: {result["retrieval_source"].upper()}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer_text})

    # Hiển thị tài liệu minh chứng (Source Documents) ở cột bên phải hoặc khu vực accordion
    with st.expander("📚 Xem danh sách tài liệu minh chứng đã truy xuất", expanded=True):
        col1, col2 = st.columns([1, 1])
        
        for i, source in enumerate(result["sources"]):
            with col1 if i % 2 == 0 else col2:
                src_name = source.get("metadata", {}).get("source", "Tài liệu không tên")
                src_type = source.get("metadata", {}).get("type", "unknown").upper()
                src_score = source.get("score", 0.0)
                
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-header">
                        <span>📄 <b>{src_name}</b> ({src_type})</span>
                        <span class="source-score">Độ tương đồng: {src_score:.3f}</span>
                    </div>
                    <div class="source-content">{source["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
