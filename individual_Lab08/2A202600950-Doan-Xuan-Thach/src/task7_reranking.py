"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import os
import requests
import numpy as np
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY", "")


def cosine_sim(a: list, b: list) -> float:
    """Tính cosine similarity giữa hai vector."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by score descending.
    """
    if not candidates:
        return []

    # Option A: Sử dụng Jina Reranker API nếu có API Key
    if JINA_API_KEY:
        try:
            print("  [INFO] Gọi Jina Reranker API...")
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {JINA_API_KEY}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c["content"] for c in candidates],
                    "top_n": top_k
                },
                timeout=10
            )
            if response.status_code == 200:
                reranked = response.json()["results"]
                return [
                    {**candidates[r["index"]], "score": float(r["relevance_score"])}
                    for r in reranked
                ]
            else:
                print(f"  [WARNING] Jina API trả về lỗi {response.status_code}. Chuyển sang fallback.")
        except Exception as e:
            print(f"  [WARNING] Lỗi kết nối Jina API ({e}). Chuyển sang fallback.")

    # Option B: Fallback sử dụng mô hình nhúng cục bộ BAAI/bge-m3 để tính cosine similarity
    print("  [INFO] Sử dụng local BAAI/bge-m3 làm Reranker fallback...")
    try:
        from .task5_semantic_search import get_model
        model = get_model()
        
        query_emb = model.encode(query)
        doc_texts = [c["content"] for c in candidates]
        doc_embs = model.encode(doc_texts)
        
        scored_candidates = []
        for cand, emb in zip(candidates, doc_embs):
            score = cosine_sim(query_emb, emb)
            scored_candidates.append({**cand, "score": score})
            
        scored_candidates = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)
        return scored_candidates[:top_k]
    except Exception as e:
        print(f"  [WARNING] Không thể chạy local Rerank fallback ({e}). Giữ nguyên thứ tự ban đầu.")
        return candidates[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []

    # Đảm bảo tất cả candidates đều có embedding
    for i, cand in enumerate(candidates):
        if "embedding" not in cand or cand["embedding"] is None:
            try:
                from .task5_semantic_search import get_model
                model = get_model()
                cand["embedding"] = model.encode(cand["content"]).tolist()
            except Exception as e:
                print(f"  [WARNING] Không thể sinh embedding cho candidate {i}: {e}")
                cand["embedding"] = [0.0] * 1024

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            # Độ tương đồng với câu truy vấn (Relevance)
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])

            # Độ tương đồng lớn nhất với các tài liệu đã chọn trước đó (Diversity penalty)
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            # Công thức MMR
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
        else:
            break

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker khác nhau.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List của các danh sách kết quả đã sắp xếp (mỗi danh sách từ 1 retriever)
        top_k: Số lượng kết quả cuối cùng
        k: Hằng số điều hòa smoothing constant (default=60)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores = {}  # content -> score
    content_map = {}  # content -> full dict

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            # Tính điểm RRF dựa trên vị trí xếp hạng (rank)
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            # Lưu trữ metadata của phần tử
            content_map[key] = item

    # Sắp xếp các tài liệu theo điểm RRF giảm dần
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)

    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        try:
            from .task5_semantic_search import get_model
            model = get_model()
            query_embedding = model.encode(query).tolist()
            return rerank_mmr(query_embedding, candidates, top_k)
        except Exception as e:
            print(f"  [WARNING] Loi khi embed query cho MMR: {e}. Fallback sang cross_encoder.")
            return rerank_cross_encoder(query, candidates, top_k)
    elif method == "rrf":
        # RRF cần nhận vào một danh sách các danh sách xếp hạng.
        # Nếu candidates được truyền trực tiếp như danh sách phẳng, chúng ta giả lập
        # nó là một danh sách đơn lẻ hoặc trả về trực tiếp.
        if candidates and isinstance(candidates[0], list):
            return rerank_rrf(candidates, top_k)
        else:
            return rerank_rrf([candidates], top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Chạy thử nghiệm với dữ liệu mẫu
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
