"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement
"""

import os
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model local để làm fallback hoặc phục vụ MMR
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def get_model():
    """Lazily load SentenceTransformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def cosine_sim(a, b):
    """Tính cosine similarity giữa hai vector."""
    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model (Jina Reranker v2 hoặc local similarity fallback).

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by score descending.
    """
    if not candidates:
        return []

    jina_key = os.getenv("JINA_API_KEY", "")
    
    # Sử dụng Jina Reranker API nếu có API key hợp lệ
    if jina_key and jina_key != "jina_xxx":
        try:
            print("Using Jina Reranker API...")
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {jina_key}"},
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
                print(f"Jina Reranker returned error {response.status_code}. Falling back to local semantic similarity.")
        except Exception as e:
            print(f"Jina Reranker API error: {e}. Falling back to local semantic similarity.")

    # Local fallback: Calculate similarity using local SentenceTransformer embeddings
    print("Using local sentence-transformer semantic similarity for reranking...")
    model = get_model()
    query_emb = model.encode(query)
    
    reranked_candidates = []
    for c in candidates:
        doc_emb = c.get("embedding")
        if doc_emb is None:
            doc_emb = model.encode(c["content"])
        
        sim = cosine_sim(query_emb, doc_emb)
        new_c = c.copy()
        new_c["score"] = sim
        reranked_candidates.append(new_c)

    # Sắp xếp giảm dần theo điểm score mới
    reranked_candidates = sorted(reranked_candidates, key=lambda x: x["score"], reverse=True)
    return reranked_candidates[:top_k]


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

    model = get_model()
    # Đảm bảo các candidates đều có vector embedding
    for c in candidates:
        if "embedding" not in c or c["embedding"] is None:
            c["embedding"] = model.encode(c["content"]).tolist()

    selected_indices = []
    remaining_indices = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining_indices:
            # Relevance to query
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])

            # Max similarity to already selected documents
            max_sim_to_selected = 0.0
            if selected_indices:
                sims = [
                    cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                    for sel_idx in selected_indices
                ]
                max_sim_to_selected = max(sims)

            # MMR formula
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    return [candidates[i] for i in selected_indices]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores = {}  # content -> score
    content_map = {}  # content -> full dict

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            # Tính điểm RRF cộng dồn cho từng chunk
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            # Giữ lại thông tin đầy đủ của chunk
            if key not in content_map:
                content_map[key] = item

    # Sắp xếp giảm dần theo điểm RRF
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
    method: str = "cross_encoder",
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
        model = get_model()
        query_embedding = model.encode(query).tolist()
        return rerank_mmr(query_embedding, candidates, top_k)
    elif method == "rrf":
        # RRF cần danh sách của nhiều danh sách (ranked_lists)
        # Nếu chỉ truyền candidates đơn lẻ, xem như là 1 ranked_list đơn lẻ
        return rerank_rrf([candidates], top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    
    print("=" * 60)
    print("Reranking test:")
    print("=" * 60)
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
