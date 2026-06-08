import os
from pathlib import Path
from dotenv import load_dotenv

# Tải các biến môi trường từ .env
load_dotenv()

# Đường dẫn thư mục
SRC_DIR = Path(__file__).parent
BASE_DIR = SRC_DIR.parent
DATA_DIR = BASE_DIR / "data"
STANDARDIZED_DIR = DATA_DIR / "standardized"

# Cấu hình RAG
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
VECTOR_STORE = "weaviate"

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"

# Các khóa API lấy từ file .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
