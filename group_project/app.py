import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st


current_dir = Path(__file__).parent.resolve()
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))


real_backend_available = False
DEFAULT_TOP_K = 5
try:
    from src.task10_generation import generate_with_citation  # type: ignore

    real_backend_available = True
except ImportError:
    try:
        from task10_generation import generate_with_citation  # type: ignore

        real_backend_available = True
    except ImportError:
        generate_with_citation = None  # type: ignore


class MockRAGEngine:
    """Mock backend so the frontend can be demoed before the real RAG pipeline is ready."""

    @staticmethod
    def generate(query: str, conversation_history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        time.sleep(0.6)
        q_lower = query.lower()
        history_hint = ""
        if conversation_history and len(conversation_history) > 2:
            history_hint = "\n\nTôi cũng đã xét đến ngữ cảnh hội thoại gần nhất để hiểu câu hỏi tiếp nối."

        if "cai nghiện" in q_lower or "luật 2021" in q_lower or "luat 2021" in q_lower:
            return {
                "answer": (
                    "Theo Luật Phòng, chống ma túy 2021, biện pháp cai nghiện ma túy gồm "
                    "cai nghiện tự nguyện và cai nghiện bắt buộc [1]. Người nghiện từ đủ 18 tuổi "
                    "có thể bị áp dụng biện pháp đưa vào cơ sở cai nghiện bắt buộc nếu không đăng ký, "
                    "không thực hiện hoặc tự ý chấm dứt cai nghiện tự nguyện [2]."
                    f"{history_hint}"
                ),
                "sources": [
                    {
                        "content": "Điều 28. Biện pháp cai nghiện ma túy gồm cai nghiện ma túy tự nguyện và cai nghiện ma túy bắt buộc.",
                        "score": 0.925,
                        "metadata": {"source": "luat-phong-chong-ma-tuy-2021.md", "type": "legal"},
                    },
                    {
                        "content": "Điều 32. Người nghiện ma túy từ đủ 18 tuổi trở lên thuộc một số trường hợp sẽ bị áp dụng biện pháp đưa vào cơ sở cai nghiện bắt buộc.",
                        "score": 0.887,
                        "metadata": {"source": "luat-phong-chong-ma-tuy-2021.md", "type": "legal"},
                    },
                ],
                "retrieval_source": "hybrid",
            }

        if "hình phạt" in q_lower or "tàng trữ" in q_lower or "tang tru" in q_lower or "tù" in q_lower:
            return {
                "answer": (
                    "Tội tàng trữ trái phép chất ma túy được quy định tại Điều 249 Bộ luật Hình sự. "
                    "Ở khung cơ bản, người phạm tội có thể bị phạt tù từ 01 năm đến 05 năm [1]. "
                    "Nếu khối lượng lớn hoặc có tình tiết tăng nặng, hình phạt có thể tăng lên các khung cao hơn, "
                    "bao gồm 15-20 năm tù hoặc tù chung thân [2]."
                    f"{history_hint}"
                ),
                "sources": [
                    {
                        "content": "Điều 249. Tội tàng trữ trái phép chất ma túy. Khung cơ bản có mức phạt tù từ 01 năm đến 05 năm.",
                        "score": 0.941,
                        "metadata": {"source": "bo-luat-hinh-su-2015.md", "type": "legal"},
                    },
                    {
                        "content": "Một số trường hợp phạm tội với khối lượng lớn có thể bị phạt tù từ 15 năm đến 20 năm hoặc tù chung thân.",
                        "score": 0.895,
                        "metadata": {"source": "bo-luat-hinh-su-2015.md", "type": "legal"},
                    },
                ],
                "retrieval_source": "hybrid",
            }

        if any(keyword in q_lower for keyword in ["nghệ sĩ", "ca sĩ", "diễn viên", "chi dân", "andrea", "an tây"]):
            return {
                "answer": (
                    "Dựa trên nhóm tin tức trong kho dữ liệu, một số vụ việc nghệ sĩ liên quan đến chất cấm "
                    "được hệ thống truy xuất gồm ca sĩ Chi Dân và người mẫu Andrea Aybar/An Tây [1][2]. "
                    "Khi dùng cho demo, phần này nên được trình bày như thông tin báo chí có nguồn kèm theo, "
                    "tránh khẳng định vượt quá nội dung tài liệu."
                    f"{history_hint}"
                ),
                "sources": [
                    {
                        "content": "Tin tức về việc ca sĩ Chi Dân bị tạm giữ để điều tra hành vi liên quan đến sử dụng, tổ chức sử dụng trái phép chất ma túy.",
                        "score": 0.963,
                        "metadata": {"source": "news_chi_dan_tam_giu.html", "type": "news"},
                    },
                    {
                        "content": "Tin tức về việc người mẫu An Tây bị khởi tố trong vụ án liên quan đến tàng trữ và tổ chức sử dụng trái phép chất ma túy.",
                        "score": 0.938,
                        "metadata": {"source": "news_an_tay_khoi_to.html", "type": "news"},
                    },
                ],
                "retrieval_source": "hybrid",
            }

        return {
            "answer": (
                "Tôi chưa tìm thấy đoạn tài liệu nội bộ thật sự khớp với câu hỏi này. "
                "Trong bản demo, hệ thống sẽ kích hoạt nhánh fallback để trả lời thận trọng và yêu cầu kiểm chứng nguồn [1]."
                f"{history_hint}"
            ),
            "sources": [
                {
                    "content": "Fallback PageIndex: khi điểm truy xuất thấp, hệ thống chuyển sang nguồn bổ trợ và gắn nhãn rõ để người dùng biết mức độ tin cậy.",
                    "score": 0.285,
                    "metadata": {"source": "PageIndex API", "type": "vectorless_fallback"},
                }
            ],
            "retrieval_source": "pageindex",
        }


def get_conversation_history(limit: int = 8) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in st.session_state.get("messages", [])[-limit:]:
        if message.get("role") in {"user", "assistant"}:
            history.append({"role": message["role"], "content": message.get("content", "")})
    return history


def call_rag_backend(query: str) -> dict[str, Any]:
    history = get_conversation_history()
    if real_backend_available and generate_with_citation is not None:
        try:
            return generate_with_citation(query, top_k=DEFAULT_TOP_K, conversation_history=history)  # type: ignore[misc]
        except TypeError:
            try:
                return generate_with_citation(query, top_k=DEFAULT_TOP_K)  # type: ignore[misc]
            except NotImplementedError:
                pass
        except NotImplementedError:
            pass
    return MockRAGEngine.generate(query, conversation_history=history)


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": result.get("answer", "Không có câu trả lời từ hệ thống."),
        "sources": result.get("sources", []),
        "retrieval_source": result.get("retrieval_source", "unknown"),
    }


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.info("Chưa có nguồn tài liệu đi kèm cho câu trả lời này.")
        return

    for index, source in enumerate(sources, start=1):
        metadata = source.get("metadata", {})
        source_name = metadata.get("source", "Tài liệu")
        source_type = metadata.get("type", "unknown")
        score = source.get("score", 0.0)
        with st.expander(f"[{index}] {source_name} - {source_type} - score {score:.3f}", expanded=index == 1):
            st.write(source.get("content", "Không có nội dung trích xuất."))


def ensure_chat_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Xin chào, tôi là trợ lý RAG về phòng chống ma túy. "
                    "Bạn có thể hỏi về quy định pháp luật, cai nghiện, tàng trữ trái phép chất ma túy "
                    "hoặc các tin tức có trong kho dữ liệu."
                ),
                "sources": [],
                "retrieval_source": "intro",
            }
        ]


st.set_page_config(
    page_title="PSD RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

chat_is_open = st.query_params.get("chat", "closed") == "open"
ensure_chat_state()

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --brand: #087a4b;
    --brand-dark: #075f3c;
    --accent: #edb526;
    --ink: #173b2d;
    --muted: #607466;
    --line: #dce8df;
    --soft: #f3f8f4;
}

html, body, [class*="css"] {
    font-family: Inter, Arial, sans-serif;
}

.stApp {
    background: #ffffff;
}

.block-container {
    max-width: 1180px;
    padding-top: 0;
    padding-bottom: 4rem;
}

[data-testid="stSidebar"] {
    display: none;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.top-strip {
    margin: 0 calc(50% - 50vw);
    padding: 8px calc(50vw - 50%);
    background: var(--brand-dark);
    color: white;
    font-size: 13px;
}

.top-strip-inner,
.nav-inner,
.section-inner {
    max-width: 1180px;
    margin: 0 auto;
}

.top-strip-inner {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: center;
}

.consult-pill {
    background: var(--accent);
    color: #263214;
    padding: 6px 12px;
    border-radius: 999px;
    font-weight: 700;
}

.main-nav {
    margin: 0 calc(50% - 50vw);
    padding: 16px calc(50vw - 50%);
    background: white;
    border-bottom: 1px solid var(--line);
    box-shadow: 0 8px 24px rgba(7, 95, 60, 0.08);
    position: sticky;
    top: 0;
    z-index: 50;
}

.nav-inner {
    display: flex;
    justify-content: space-between;
    gap: 22px;
    align-items: center;
}

.brand-lockup {
    display: flex;
    align-items: center;
    gap: 12px;
    min-width: 260px;
}

.brand-mark {
    width: 54px;
    height: 54px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    background: linear-gradient(145deg, var(--brand), #11a86b);
    color: white;
    font-weight: 800;
    border: 4px solid #e8f5ec;
}

.brand-title {
    font-size: 15px;
    line-height: 1.25;
    color: var(--brand-dark);
    font-weight: 800;
    text-transform: uppercase;
}

.brand-subtitle {
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
}

.nav-links {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 8px 18px;
    color: var(--brand-dark);
    font-size: 13px;
    font-weight: 800;
    text-transform: uppercase;
}

.hero {
    margin: 0 calc(50% - 50vw);
    min-height: 520px;
    color: white;
    background:
        linear-gradient(90deg, rgba(5, 74, 46, 0.94), rgba(8, 122, 75, 0.68), rgba(8, 122, 75, 0.16)),
        url("https://images.unsplash.com/photo-1576091160550-2173dba999ef?auto=format&fit=crop&w=1800&q=80");
    background-size: cover;
    background-position: center;
    display: flex;
    align-items: center;
}

.hero-inner {
    max-width: 1180px;
    width: 100%;
    margin: 0 auto;
    padding: 58px 0 72px;
}

.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border: 1px solid rgba(255,255,255,0.34);
    border-radius: 999px;
    background: rgba(255,255,255,0.12);
    font-size: 13px;
    font-weight: 700;
}

.hero h1 {
    max-width: 760px;
    font-size: 48px;
    line-height: 1.08;
    margin: 20px 0 18px;
    letter-spacing: 0;
}

.hero p {
    max-width: 650px;
    color: rgba(255,255,255,0.92);
    font-size: 18px;
    line-height: 1.65;
}

.hero-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 30px;
}

.primary-link,
.secondary-link {
    display: inline-block;
    padding: 13px 18px;
    border-radius: 4px;
    font-weight: 800;
    text-decoration: none !important;
}

.primary-link {
    background: var(--accent);
    color: #263214 !important;
}

.secondary-link {
    border: 1px solid rgba(255,255,255,0.46);
    color: white !important;
}

.section-title {
    margin: 42px 0 18px;
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 18px;
}

.section-title h2 {
    color: var(--ink);
    margin: 0;
    font-size: 28px;
}

.section-title span {
    color: var(--brand);
    font-weight: 800;
    font-size: 13px;
    text-transform: uppercase;
}

.about-band {
    display: grid;
    grid-template-columns: 1.1fr 0.9fr;
    gap: 28px;
    align-items: center;
    padding: 34px;
    background: var(--soft);
    border-left: 6px solid var(--brand);
}

.about-band p,
.news-card p,
.service-card p,
.value-card p {
    color: var(--muted);
    line-height: 1.65;
}

.stat-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
}

.stat {
    background: white;
    border: 1px solid var(--line);
    padding: 18px;
}

.stat strong {
    color: var(--brand);
    display: block;
    font-size: 26px;
}

.service-grid,
.news-grid,
.value-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 18px;
}

.service-card,
.news-card,
.value-card {
    border: 1px solid var(--line);
    background: white;
    padding: 22px;
    min-height: 170px;
}

.service-card {
    border-top: 4px solid var(--brand);
}

.news-card {
    border-top: 4px solid var(--accent);
}

.card-label {
    color: var(--brand);
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
}

.service-card h3,
.news-card h3,
.value-card h3 {
    color: var(--ink);
    margin: 10px 0 8px;
    font-size: 19px;
}

.chat-launcher {
    position: fixed;
    right: 28px;
    bottom: 28px;
    width: 72px;
    height: 72px;
    border-radius: 50%;
    z-index: 9999;
    display: grid;
    place-items: center;
    background: linear-gradient(145deg, var(--brand), #0fa66c);
    color: white !important;
    text-decoration: none !important;
    box-shadow: 0 18px 38px rgba(8, 122, 75, 0.32);
    border: 5px solid white;
    font-weight: 900;
    font-size: 18px;
}

.chat-launcher::after {
    content: "Hỏi AI";
    position: absolute;
    right: 76px;
    white-space: nowrap;
    background: #173b2d;
    color: white;
    padding: 7px 10px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 700;
}

.chat-panel {
    border: 1px solid var(--line);
    border-top: 5px solid var(--brand);
    background: white;
    padding: 22px;
    box-shadow: 0 18px 48px rgba(23, 59, 45, 0.12);
}

.st-key-chat_drawer {
    position: fixed;
    right: 24px;
    bottom: 24px;
    width: min(420px, calc(100vw - 32px));
    max-height: calc(100vh - 48px);
    overflow-y: auto;
    z-index: 10000;
    border: 1px solid var(--line);
    border-top: 5px solid var(--brand);
    border-radius: 8px;
    background: white;
    padding: 18px;
    box-shadow: 0 24px 70px rgba(23, 59, 45, 0.26);
}

.st-key-chat_history {
    border: 1px solid #edf3ee;
    border-radius: 8px;
    background: #fbfdfb;
    padding: 8px;
}

.chat-panel-header {
    display: flex;
    justify-content: space-between;
    gap: 14px;
    align-items: center;
    margin-bottom: 12px;
}

.chat-panel-header h2 {
    color: var(--ink);
    margin: 0;
}

.chat-close {
    color: var(--brand) !important;
    font-weight: 800;
    text-decoration: none !important;
}

.source-note {
    color: var(--muted);
    font-size: 13px;
}

[data-testid="stChatMessage"] {
    border: 1px solid #edf3ee;
    border-radius: 8px;
    background: #fbfdfb;
}

.stButton button {
    border-radius: 4px;
    border: 1px solid var(--brand);
    color: var(--brand);
}

@media (max-width: 860px) {
    .top-strip-inner,
    .nav-inner,
    .hero-inner {
        padding-left: 18px;
        padding-right: 18px;
    }

    .nav-inner,
    .about-band {
        grid-template-columns: 1fr;
        display: grid;
    }

    .nav-links {
        justify-content: flex-start;
    }

    .hero {
        min-height: 500px;
    }

    .hero h1 {
        font-size: 36px;
    }

    .service-grid,
    .news-grid,
    .value-grid,
    .stat-row {
        grid-template-columns: 1fr;
    }

    .chat-launcher {
        right: 18px;
        bottom: 18px;
        width: 62px;
        height: 62px;
    }

    .chat-launcher::after {
        display: none;
    }

    .st-key-chat_drawer {
        right: 12px;
        bottom: 12px;
        width: calc(100vw - 24px);
        max-height: calc(100vh - 24px);
    }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="top-strip">
  <div class="top-strip-inner">
    <div>Tiếng Việt | English</div>
    <div><span class="consult-pill">Đăng ký tư vấn</span></div>
  </div>
</div>

<div class="main-nav">
  <div class="nav-inner">
    <div class="brand-lockup">
      <div class="brand-mark">PSD</div>
      <div>
        <div class="brand-title">Viện nghiên cứu và ứng dụng phòng chống ma túy</div>
        <div class="brand-subtitle">RAG Chatbot hỗ trợ tra cứu pháp luật và tin tức</div>
      </div>
    </div>
    <div class="nav-links">
      <span>Trang chủ</span>
      <span>Giới thiệu</span>
      <span>Tin tức</span>
      <span>Nghiên cứu</span>
      <span>Tư vấn cai nghiện</span>
      <span>Thư viện</span>
      <span>Liên hệ</span>
    </div>
  </div>
</div>

<section class="hero">
  <div class="hero-inner">
    <div class="hero-kicker">Cổng thông tin phòng chống ma túy tích hợp AI</div>
    <h1>Tra cứu pháp luật, tin tức và nguồn dẫn bằng trợ lý RAG</h1>
    <p>
      Website mô phỏng giao diện cổng thông tin phòng chống ma túy, kết hợp chatbot nhỏ ở góc màn hình
      để người dùng đặt câu hỏi và nhận câu trả lời có trích dẫn nguồn.
    </p>
    <div class="hero-actions">
      <a class="primary-link" href="?chat=open#rag-chatbot">Hỏi trợ lý AI</a>
      <a class="secondary-link" href="#tin-noi-bat">Xem tin nổi bật</a>
    </div>
  </div>
</section>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="section-title">
  <div>
    <span>Về chúng tôi</span>
    <h2>Hệ thống hỏi đáp cho bài tập nhóm Day 8</h2>
  </div>
</div>

<div class="about-band">
  <div>
    <h3>Trọng tâm là truy xuất có kiểm chứng</h3>
    <p>
      Giao diện này đóng vai trò lớp trình bày cho RAG pipeline: người dùng hỏi tự nhiên, hệ thống truy xuất
      tài liệu liên quan, sinh câu trả lời và hiển thị nguồn đã sử dụng. Phần website bên ngoài giúp demo
      giống một sản phẩm thực tế thay vì chỉ là màn hình chat đơn giản.
    </p>
  </div>
  <div class="stat-row">
    <div class="stat"><strong>15+</strong><span>câu hỏi đánh giá</span></div>
    <div class="stat"><strong>4</strong><span>nhóm metric RAG</span></div>
    <div class="stat"><strong>2</strong><span>cấu hình A/B</span></div>
  </div>
</div>

<div class="section-title">
  <div>
    <span>Dịch vụ</span>
    <h2>Các khối chức năng chính</h2>
  </div>
</div>

<div class="service-grid">
  <div class="service-card">
    <div class="card-label">Tra cứu</div>
    <h3>Văn bản pháp luật</h3>
    <p>Hỏi đáp về Luật Phòng, chống ma túy, Bộ luật Hình sự và các quy định liên quan.</p>
  </div>
  <div class="service-card">
    <div class="card-label">Tin tức</div>
    <h3>Nguồn báo chí</h3>
    <p>Tìm các tin trong kho dữ liệu và hiển thị nguồn để người dùng tự kiểm chứng.</p>
  </div>
  <div class="service-card">
    <div class="card-label">Tư vấn</div>
    <h3>Hỏi đáp tiếp nối</h3>
    <p>Lưu lịch sử hội thoại để hỗ trợ follow-up questions trong phiên làm việc.</p>
  </div>
</div>

<div class="section-title" id="tin-noi-bat">
  <div>
    <span>Tin nổi bật</span>
    <h2>Nội dung mô phỏng trên trang chủ</h2>
  </div>
</div>

<div class="news-grid">
  <div class="news-card">
    <div class="card-label">Phòng ngừa</div>
    <h3>Nhận diện ma túy núp bóng</h3>
    <p>Các dạng chất cấm mới thường được ngụy trang tinh vi, cần kết hợp truyền thông và giáo dục cộng đồng.</p>
  </div>
  <div class="news-card">
    <div class="card-label">Cai nghiện</div>
    <h3>Phục hồi là một hành trình</h3>
    <p>Cai nghiện không chỉ là cắt cơn mà còn cần hỗ trợ tâm lý, gia đình và tái hòa nhập xã hội.</p>
  </div>
  <div class="news-card">
    <div class="card-label">Pháp luật</div>
    <h3>Tra cứu quy định xử lý</h3>
    <p>Chatbot có thể hỗ trợ tìm nhanh điều khoản, mức phạt và nguồn tài liệu liên quan.</p>
  </div>
</div>

<div class="section-title">
  <div>
    <span>Giá trị</span>
    <h2>Tại sao dùng RAG cho website này</h2>
  </div>
</div>

<div class="value-grid">
  <div class="value-card">
    <h3>Có căn cứ</h3>
    <p>Câu trả lời luôn đi kèm source documents thay vì chỉ trả lời chung chung.</p>
  </div>
  <div class="value-card">
    <h3>Dễ demo</h3>
    <p>Người dùng chỉ cần nhập câu hỏi, hệ thống tự truy xuất tài liệu phù hợp và trả lời kèm nguồn.</p>
  </div>
  <div class="value-card">
    <h3>Sẵn sàng mở rộng</h3>
    <p>Adapter backend giúp chuyển từ mock sang pipeline thật với ít thay đổi giao diện.</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if chat_is_open:
    with st.container(key="chat_drawer"):
        st.markdown(
            """
<div id="rag-chatbot" class="chat-panel-header">
  <div>
    <h2>Trợ lý hỏi đáp phòng chống ma túy</h2>
    <div class="source-note">Trả lời kèm citation và tài liệu nguồn.</div>
  </div>
  <a class="chat-close" href="?chat=closed">Đóng</a>
</div>
""",
            unsafe_allow_html=True,
        )

        if st.button("Xóa lịch sử", use_container_width=False, key="clear_chat"):
            st.session_state.messages = []
            ensure_chat_state()
            st.rerun()

        with st.container(height=420, border=False, key="chat_history", autoscroll=True):
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and message.get("sources"):
                        render_sources(message["sources"])

        with st.form("chat_form", clear_on_submit=True):
            user_query = st.text_input(
                "Câu hỏi",
                placeholder="Nhập câu hỏi của bạn...",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Gửi", use_container_width=True)

        if submitted and user_query.strip():
            st.session_state.messages.append({"role": "user", "content": user_query.strip()})
            with st.spinner("Đang truy xuất tài liệu và tạo câu trả lời..."):
                result = normalize_result(call_rag_backend(user_query.strip()))
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "retrieval_source": result["retrieval_source"],
                }
            )
            st.rerun()
else:
    st.markdown('<a class="chat-launcher" href="?chat=open#rag-chatbot" title="Mở chatbot">AI</a>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="section-title" id="rag-chatbot">
  <div>
    <span>Chatbot</span>
    <h2>Bấm logo AI ở góc phải để mở trợ lý hỏi đáp</h2>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
