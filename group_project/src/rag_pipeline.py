import json
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

# Import cấu hình dùng chung
from . import config

class GroupRAGPipeline:
    """
    RAG Pipeline tích hợp dùng chung cho cả nhóm.
    Hỗ trợ: Hybrid Search (Semantic + Lexical), Reranking (Cross-Encoder, MMR, RRF),
    PageIndex Fallback, Lost-in-the-Middle reordering, và Sinh câu trả lời có Citation.
    """
    def __init__(self):
        self._embedding_model = None
        self._cross_encoder_model = None
        self._bm25_index = None
        self._corpus: List[Dict[str, Any]] = []
        self._pageindex_client = None

        # Tải dữ liệu và lập chỉ mục BM25
        self._load_corpus()

    def _get_embedding_model(self):
        """Lazy load model embedding BAAI/bge-m3."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[INFO] Khởi tạo embedding model: {config.EMBEDDING_MODEL}")
            self._embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        return self._embedding_model

    def _get_cross_encoder_model(self):
        """Lazy load model cross-encoder để rerank."""
        if self._cross_encoder_model is None:
            from sentence_transformers import CrossEncoder
            print("[INFO] Khởi tạo cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2")
            self._cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._cross_encoder_model

    def _load_corpus(self):
        """Đọc dữ liệu từ cache local và xây dựng BM25 index."""
        cache_path = config.DATA_DIR / "vector_cache.json"
        if cache_path.exists():
            try:
                self._corpus = json.loads(cache_path.read_text(encoding="utf-8"))
                print(f"[INFO] RAGPipeline loaded {len(self._corpus)} chunks từ cache local.")
            except Exception as e:
                print(f"[ERROR] Lỗi đọc file cache dữ liệu: {e}")
        
        if self._corpus:
            # Build BM25 index
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [doc["content"].lower().split() for doc in self._corpus]
            self._bm25_index = BM25Okapi(tokenized_corpus)
            print("[INFO] Đã xây dựng BM25 index thành công.")
        else:
            print("[WARN] Chưa có dữ liệu cache vector store. Vui lòng chạy database_setup.py trước!")

    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Tìm kiếm ngữ nghĩa (Dense Retrieval) sử dụng Vector Database hoặc local cache fallback."""
        model = self._get_embedding_model()
        query_emb = model.encode(query).tolist()
        
        # Thử kết nối Weaviate local
        if config.VECTOR_STORE == "weaviate":
            try:
                import weaviate
                from weaviate.classes.query import MetadataQuery
                
                client = weaviate.connect_to_local()
                # Tương thích với collection nhóm của database_setup.py
                collection = client.collections.get("GroupDrugLawDocs")
                
                results = collection.query.near_vector(
                    near_vector=query_emb,
                    limit=top_k,
                    return_metadata=MetadataQuery(distance=True)
                )
                
                output = []
                for obj in results.objects:
                    score = 1.0 - (obj.metadata.distance if obj.metadata.distance is not None else 0.5)
                    output.append({
                        "content": obj.properties.get("content", ""),
                        "score": score,
                        "metadata": {
                            "source": obj.properties.get("source", "Unknown"),
                            "type": obj.properties.get("doc_type", "Unknown"),
                            "chunk_index": obj.properties.get("chunk_index", 0)
                        }
                    })
                client.close()
                return output
            except Exception as e:
                # Fallback sang so khớp vector local trên file cache
                pass

        # Local fallback using cosine similarity
        if not self._corpus:
            return []
            
        def cosine_similarity(v1, v2):
            dot_product = sum(a * b for a, b in zip(v1, v2))
            norm_a = sum(a * a for a in v1) ** 0.5
            norm_b = sum(b * b for b in v2) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot_product / (norm_a * norm_b)

        scored_chunks = []
        for c in self._corpus:
            if "embedding" in c:
                score = cosine_similarity(query_emb, c["embedding"])
                scored_chunks.append({
                    "content": c["content"],
                    "score": score,
                    "metadata": c["metadata"]
                })
                
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]

    def lexical_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Tìm kiếm từ khóa chính xác sử dụng thuật toán BM25."""
        if not self._bm25_index or not self._corpus:
            return []
            
        tokenized_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenized_query)
        
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "content": self._corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": self._corpus[idx]["metadata"]
            })
        return results

    def pageindex_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Tìm kiếm Vectorless RAG qua PageIndex Cloud (Fallback)."""
        if not config.PAGEINDEX_API_KEY:
            print("[WARN] PAGEINDEX_API_KEY không được đặt trong file .env")
            return []

        cache_file = config.DATA_DIR / "pageindex_docs.json"
        if not cache_file.exists():
            print("[WARN] Không tìm thấy file cache tài liệu PageIndex (pageindex_docs.json).")
            return []
            
        try:
            doc_ids = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[ERROR] Lỗi đọc pageindex_docs.json: {e}")
            return []

        from pageindex import PageIndexClient
        client = PageIndexClient(api_key=config.PAGEINDEX_API_KEY)
        
        results = []
        for doc_id in doc_ids:
            try:
                # Đảm bảo document đã sẵn sàng
                if not client.is_retrieval_ready(doc_id):
                    continue
                    
                res = client.submit_query(doc_id, query)
                retrieval_id = res.get("retrieval_id")
                if retrieval_id:
                    # Chờ kết quả xử lý
                    for _ in range(5):
                        ret_res = client.get_retrieval(retrieval_id)
                        if ret_res.get("status") == "completed":
                            for r in ret_res.get("results", []):
                                text = r.get("text") or r.get("content") or ""
                                score = r.get("score") or 0.0
                                meta = r.get("metadata") or {}
                                results.append({
                                    "content": text,
                                    "score": score,
                                    "metadata": meta,
                                    "source": "pageindex"
                                })
                            break
                        time.sleep(1)
            except Exception as e:
                print(f"[WARN] Lỗi query PageIndex với doc {doc_id}: {e}")

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int, method: str) -> List[Dict[str, Any]]:
        """Tái xếp hạng kết quả (Hỗ trợ Cross-Encoder, MMR, hoặc RRF)."""
        if not candidates:
            return []

        if method == "cross_encoder":
            try:
                model = self._get_cross_encoder_model()
                pairs = [[query, c["content"]] for c in candidates]
                scores = model.predict(pairs)
                
                reranked = []
                for c, score in zip(candidates, scores):
                    item = c.copy()
                    item["score"] = float(score)
                    reranked.append(item)
                    
                reranked.sort(key=lambda x: x["score"], reverse=True)
                return reranked[:top_k]
            except Exception as e:
                print(f"  [WARN] Cross-Encoder rerank thất bại: {e}. Sử dụng thứ hạng ban đầu.")
                return candidates[:top_k]

        elif method == "mmr":
            # Maximal Marginal Relevance
            def cosine_sim(v1, v2):
                dot_product = sum(a * b for a, b in zip(v1, v2))
                norm_a = sum(a * a for a in v1) ** 0.5
                norm_b = sum(b * b for b in v2) ** 0.5
                if norm_a == 0 or norm_b == 0:
                    return 0.0
                return dot_product / (norm_a * norm_b)

            # Đảm bảo các candidates có embedding
            emb_model = self._get_embedding_model()
            for c in candidates:
                if "embedding" not in c or not c["embedding"]:
                    # Tìm trong corpus
                    found = False
                    for corpus_item in self._corpus:
                        if corpus_item["content"] == c["content"] and "embedding" in corpus_item:
                            c["embedding"] = corpus_item["embedding"]
                            found = True
                            break
                    if not found:
                        c["embedding"] = emb_model.encode(c["content"]).tolist()

            query_embedding = emb_model.encode(query).tolist()
            lambda_param = 0.7
            selected = []
            remaining = list(range(len(candidates)))

            for _ in range(min(top_k, len(candidates))):
                best_idx = None
                best_score = float('-inf')

                for idx in remaining:
                    relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])
                    max_sim_to_selected = 0.0
                    for sel_idx in selected:
                        sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                        max_sim_to_selected = max(max_sim_to_selected, sim)

                    mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim_to_selected

                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx

                if best_idx is not None:
                    selected.append(best_idx)
                    remaining.remove(best_idx)
                else:
                    break

            return [candidates[i] for i in selected]

        elif method == "rrf":
            # Reciprocal Rank Fusion
            rrf_scores = {}
            content_map = {}
            k = 60

            for rank, item in enumerate(candidates, 1):
                key = item["content"]
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
                if key not in content_map:
                    content_map[key] = item.copy()

            sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            results = []
            for content, score in sorted_items[:top_k]:
                item = content_map[content].copy()
                item["score"] = float(score)
                results.append(item)
            return results

        else:
            # Fallback
            return candidates[:top_k]

    def _rrf_merge(self, ranked_lists: List[List[Dict[str, Any]]], top_k: int = 10) -> List[Dict[str, Any]]:
        """Gộp kết quả từ dense và sparse search bằng thuật toán RRF."""
        rrf_scores = {}
        content_map = {}
        k = 60

        for ranked_list in ranked_lists:
            for rank, item in enumerate(ranked_list, 1):
                key = item["content"]
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
                if key not in content_map:
                    content_map[key] = item.copy()

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for content, score in sorted_items[:top_k]:
            item = content_map[content].copy()
            item["score"] = float(score)
            results.append(item)
        return results

    def retrieve(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """Hàm tìm kiếm tích hợp: Hybrid Search + Reranking + PageIndex Fallback."""
        # 1. Chạy song song Dense và Sparse search
        dense_results = self.semantic_search(query, top_k=top_k * 2)
        sparse_results = self.lexical_search(query, top_k=top_k * 2)

        # 2. Gộp kết quả
        merged = self._rrf_merge([dense_results, sparse_results], top_k=top_k * 2)
        for item in merged:
            item["source"] = "hybrid"

        # 3. Reranking
        final_results = self.rerank(query, merged, top_k=top_k, method=config.RERANK_METHOD)

        # 4. Kiểm tra threshold và fallback PageIndex
        best_score = final_results[0]["score"] if final_results else 0.0
        if not final_results or best_score < config.SCORE_THRESHOLD:
            print(f"  [WARN] Hybrid score ({best_score:.3f}) < threshold ({config.SCORE_THRESHOLD}). Fallback -> PageIndex")
            try:
                fallback = self.pageindex_search(query, top_k=top_k)
                if fallback:
                    return fallback
            except Exception as e:
                print(f"  [WARN] Lỗi tìm kiếm PageIndex fallback: {e}")

        return final_results[:top_k]

    def reorder_for_llm(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sắp xếp lại các chunks để tránh hiệu ứng 'Lost in the middle' của LLM (Best ở đầu và cuối, worst ở giữa)."""
        if len(chunks) <= 2:
            return chunks

        reordered = []
        # Vị trí lẻ (0, 2, 4...) đặt lên đầu
        for i in range(0, len(chunks), 2):
            reordered.append(chunks[i])
        # Vị trí chẵn (1, 3, 5...) đảo ngược đặt về cuối
        start_even = len(chunks) - 1
        if start_even % 2 == 0:
            start_even -= 1
        for i in range(start_even, 0, -2):
            reordered.append(chunks[i])

        return reordered

    def generate(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Sinh câu trả lời tiếng Việt hoàn chỉnh, định dạng trích dẫn (citations) từ context và tích hợp chat history."""
        # 1. Retrieve các chunks liên quan nhất
        chunks = self.retrieve(query, top_k=config.DEFAULT_TOP_K)

        # 2. Tránh Lost-in-the-Middle
        reordered_chunks = self.reorder_for_llm(chunks)

        # 3. Format Context dạng rõ ràng có gán nhãn Document để LLM trích dẫn
        context_parts = []
        for i, chunk in enumerate(reordered_chunks, 1):
            source = chunk.get("metadata", {}).get("source", f"Tài liệu {i}")
            doc_type = chunk.get("metadata", {}).get("type", "unknown")
            context_parts.append(
                f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
                f"{chunk['content']}\n"
            )
        context = "\n---\n".join(context_parts)

        # 4. Xây dựng prompt sinh câu trả lời
        system_prompt = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets linking to the specific document index or source name (e.g., [Document 1] or [dung_article_01.md]).

If the information is not explicitly stated in the provided context, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than guessing or making it up.

Rules:
- Only use information from the provided context.
- Every factual claim MUST have a citation.
- If context is insufficient, say so clearly.
- Structure your answer with clear paragraphs."""

        # 5. Xây dựng cấu trúc Messages (Tích hợp Lịch sử trò chuyện - Conversation Memory)
        messages = [{"role": "system", "content": system_prompt}]
        
        # Thêm lịch sử hội thoại nếu có
        if conversation_history:
            for message in conversation_history:
                messages.append({
                    "role": message.get("role", "user"),
                    "content": message.get("content", "")
                })

        # Thêm thông tin context và câu hỏi hiện tại
        user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
        messages.append({"role": "user", "content": user_message})

        # 6. Gọi API sinh câu trả lời (Thử OpenAI -> Gemini -> Fallback Mock Local)
        answer = ""
        
        # Thử OpenAI
        if config.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=config.OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.3,
                    top_p=0.9
                )
                answer = response.choices[0].message.content
            except Exception as e:
                print(f"  [WARN] OpenAI API Call failed: {e}")

        # Thử Gemini
        if not answer and config.GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=config.GEMINI_API_KEY)
                
                # Format messages cho định dạng Gemini API nếu cần, hoặc dùng prompt phẳng
                model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    system_instruction=system_prompt
                )
                
                # Chuyển lịch sử + tin nhắn hiện tại thành định dạng chat của Gemini
                chat = model.start_chat(history=[])
                if conversation_history:
                    gemini_history = []
                    for h in conversation_history:
                        role = "user" if h["role"] == "user" else "model"
                        gemini_history.append({"role": role, "parts": [h["content"]]})
                    chat = model.start_chat(history=gemini_history)
                
                response = chat.send_message(
                    user_message,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        top_p=0.9
                    )
                )
                answer = response.text
            except Exception as e:
                print(f"  [WARN] Gemini API Call failed: {e}")

        # Fallback Local Mock nếu không có API keys
        if not answer:
            print("  [WARN] Không tìm thấy API keys hoặc cuộc gọi API thất bại. Sinh câu trả lời từ context...")
            if not chunks:
                answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
            else:
                lines = []
                for i, chunk in enumerate(chunks[:2], 1):
                    source = chunk.get("metadata", {}).get("source", "Tài liệu")
                    snippet = chunk["content"][:200].replace("\n", " ").strip()
                    lines.append(f"Dựa trên tài liệu [{source}]: {snippet}...")
                answer = " ".join(lines)

        return {
            "answer": answer,
            "sources": chunks,
            "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
        }
