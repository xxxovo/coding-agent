# lexical score：关键词匹配
# ector score：embedding + FAISS 相似度
# rerank bonus：策略加权

from __future__ import annotations

import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, List

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import sys
from pathlib import Path

# 获取当前文件所在目录（rag）并加入 Python 路径，包容任意维度的项目级 Import
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from code_indexer import CodeIndexer, DEFAULT_CACHE_DIR
from graph_builder import CodeGraphBuilder


class HybridRetriever:
    """升级版混合检索器。

    当前实现组合了：
    - lexical retrieval
    - embedding + FAISS vector retrieval
    - lightweight rerank
    - graph expansion
    """

    def __init__(
        self,
        code_units: Iterable[dict],
        graph_builder: CodeGraphBuilder,
        code_indexer: CodeIndexer,
    ):
        self.code_units = list(code_units)
        self.graph_builder = graph_builder
        self.code_indexer = code_indexer
        self.documents = [self._build_document(unit) for unit in self.code_units]
        self.doc_lookup = {unit["id"]: doc for unit, doc in zip(self.code_units, self.documents)}
        
        # 将内存 BM25 替换为了 Elasticsearch Indexer
        # 此处要求 ES 服务已运行并建立完毕索引
        # 可以通过 self.es_indexer = ESIndexer(); self.es_indexer.build_index(self.code_units) 提前初始化
        # from es_indexer import ESIndexer
        # self.es_retriever = ESIndexer()
        
        # 为了不强制打断旧代码流（如果您还没有拉起 docker），
        # 兼容策略：如果传入的 kwargs 或配置带 es_url 可以直接替换。在这里演示概念：
        self.use_es = True
        if self.use_es:
            from es_indexer import ESIndexer
            self.es_retriever = ESIndexer()
        else:
            tokenized_corpus = [self._tokenize(doc) for doc in self.documents]
            self.bm25_model = BM25Okapi(tokenized_corpus)
        
        # Load local Cross-Encoder reranker model
        # 强制断开公网连接，仅从本地加载，解决网络超时导致的冷启动卡死问题
        # 在 HuggingFace 缓存路径下，实际的模型权重文件存在于 snapshots 的哈希目录下
        model_path = '/Users/zrj/Documents/项目/coding-agent/.cache/huggingface/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
        self.reranker = CrossEncoder(
            model_path, 
            max_length=1024,
            local_files_only=True
        )
        
        self.edge_lookup = self._build_edge_lookup()

    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        expand_hops: int = 1,
        w_vec: float = 0.8,
        w_bm25: float = 0.2
    ) -> List[dict]:
        start_time = time.time()
        
        # Step 1: Base Retrieval (Recall)
        recall_start = time.time()
        candidate_pool_size = max(top_k * 4, 30)
        bm25_scores = self._bm25_search(query)
        vector_scores = self._vector_search(query, top_k=candidate_pool_size)

        bm25_ranking = self._top_k_ids_from_scores(bm25_scores, candidate_pool_size)
        vector_ranking = self._top_k_ids_from_scores(vector_scores, candidate_pool_size)
        
        # Step 2: RRF Fusion
        rrf_scores = self._apply_weighted_rrf(
            rankings_with_weights=[
                (vector_ranking, w_vec), 
                (bm25_ranking, w_bm25)
            ]
        )

        candidate_ids = [
            unit_id for unit_id, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:candidate_pool_size]
        ]
        candidate_units = [unit for unit in self.code_units if unit["id"] in candidate_ids]
        recall_end = time.time()
        print(f"[Timing] 召回阶段 (Recall): {recall_end - recall_start:.4f}s")

        if not candidate_units:
            print(f"[Timing] 总检索耗时 (Total Search): {time.time() - start_time:.4f}s")
            return []

        # Step 3: Model-based Reranking with Pre-computed AST Chunks & Max-Pooling
        rerank_start = time.time()
        flat_pairs = []
        chunk_to_unit_map = []  # 记录每个 chunk 对应的 unit 索引
        
        # O(N) 预计算当前向量库里的映射关系，避免嵌套循环
        has_ast_chunks = getattr(self.code_indexer, "enable_chunking", False) and hasattr(self.code_indexer, "chunk_mapping")
        unit_to_chunks = {}
        if has_ast_chunks:
            for idx, mapped_id in enumerate(self.code_indexer.chunk_mapping):
                # mapped_id 是 Code Unit 的 ID
                if mapped_id not in unit_to_chunks:
                    unit_to_chunks[mapped_id] = []
                unit_to_chunks[mapped_id].append(self.code_indexer.documents[idx])
        
        for i, unit in enumerate(candidate_units):
            unit_id = unit["id"]
            
            # 直接复用 CodeIndexer 阶段基于 Tree-Sitter 切分好的语义 Chunk
            # 彻底抛弃基于文本长度的滑动窗口，防止 Rerank 阶段破坏代码结构
            chunks = unit_to_chunks.get(unit_id, [])
            
            # 降级容错：如果未开启 Chunking 或当前 Unit 没有被分块，则使用完整全文本
            if not chunks:
                chunks = [self.doc_lookup[unit_id]]
            
            for chunk in chunks:
                flat_pairs.append([query, chunk])
                chunk_to_unit_map.append(i)
        
        if flat_pairs:
            char_lengths = [len(q) + len(doc) for q, doc in flat_pairs]
            print(f"[Reranker Stats] 候选符号: {len(candidate_units)} | 拆分 Chunk 总数: {len(flat_pairs)} | Chunk 最大字符数: {max(char_lengths)}")
        
        # Predict Chunk relevancy scores
        chunk_scores = self.reranker.predict(flat_pairs)

        # 聚合分块得分 (Max Pooling)
        unit_max_scores = [-float('inf')] * len(candidate_units)
        for chunk_idx, score in enumerate(chunk_scores):
            unit_idx = chunk_to_unit_map[chunk_idx]
            if score > unit_max_scores[unit_idx]:
                unit_max_scores[unit_idx] = float(score)

        for unit, max_score in zip(candidate_units, unit_max_scores):
            unit["_rerank_score"] = max_score

        ranked_units = sorted(
            candidate_units,
            key=lambda unit: unit["_rerank_score"],
            reverse=True,
        )[:top_k]

        results = []
        for unit in ranked_units:
            result = dict(unit)
            result["retrieval_score"] = round(result.pop("_rerank_score"), 4)
            result["retrieval_reason"] = "cross_encoder_rerank"
            results.append(result)

        rerank_end = time.time()
        print(f"[Timing] 重排阶段 (Rerank): {rerank_end - rerank_start:.4f}s")
        print(f"[Timing] 总检索耗时 (Total Search): {time.time() - start_time:.4f}s")

        return results

    def expand_context(self, results: List[dict], expand_hops: int = 1) -> List[dict]:
        """基于主检索结果做 graph 邻域扩展，作为二阶段上下文补全。"""
        seed_ids = [result["id"] for result in results]
        expanded_units = self.graph_builder.expand_neighbors(seed_ids, max_hops=expand_hops)
        primary_ids = set(seed_ids)

        expanded_context: List[dict] = []
        for unit in expanded_units:
            if unit["id"] in primary_ids:
                continue
            enriched_unit = dict(unit)
            enriched_unit["retrieval_score"] = round(self._graph_neighbor_score(unit["id"], results, expand_hops), 4)
            enriched_unit["retrieval_reason"] = "graph_neighbor"
            expanded_context.append(enriched_unit)

        return sorted(
            expanded_context,
            key=lambda unit: unit.get("retrieval_score", 0.0),
            reverse=True,
        )

    def _bm25_search(self, query: str) -> dict[str, float]:
        if getattr(self, "use_es", False):
            return self.es_retriever.search(query, top_k=50)

        query_terms = self._tokenize(query)
        bm25_scores = self.bm25_model.get_scores(query_terms)
        
        scores: dict[str, float] = {}
        for idx, unit in enumerate(self.code_units):
            if bm25_scores[idx] > 0:
                scores[unit["id"]] = float(bm25_scores[idx])

        return scores

    def _vector_search(self, query: str, top_k: int) -> dict[str, float]:
        results = self.code_indexer.search(query, top_k=top_k)
        return {result.unit_id: result.score for result in results}

    def _sigmoid(self, x: float) -> float:
        """将 Logit 转换为 (0, 1) 的概率分布"""
        if x < -20:  # 防止 math.exp 溢出
            return 0.0
        elif x > 20:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    def _inverse_sigmoid(self, p: float) -> float:
        """将概率转换回 Logit"""
        # 防止对 0 或 1 取对数
        p = max(min(p, 0.9999), 0.0001)
        return math.log(p / (1.0 - p))

    def _graph_neighbor_score(self, unit_id: str, ranked_units: List[dict], expand_hops: int = 1) -> float:
        best_prob = 0.0  # 我们在概率空间进行衰减计算，最小为0
        
        hop_distance = getattr(self.graph_builder, 'node_distances', {}).get(unit_id, expand_hops)

        for ranked_unit in ranked_units:
            parent_id = ranked_unit["id"]
            # 拿到的是带有正负的原始Logit打分
            parent_logit = ranked_unit.get("retrieval_score", 0.0) 
            # 步骤1： 将父节点的评分平滑映射到 [0, 1] 的概率空间，变得拥有物理/数学上的绝对含义
            parent_prob = self._sigmoid(parent_logit)
            
            fwd_edge = self.edge_lookup.get((parent_id, unit_id))
            rev_edge = self.edge_lookup.get((unit_id, parent_id))
            
            # 使用边类型映射衰减比例系数 (在真实概率概率下做乘法，1代表不衰减)
            edge_decay = 0.0
            if fwd_edge == "calls": edge_decay = 0.95
            elif fwd_edge == "contains": edge_decay = 0.90
            elif fwd_edge == "inherits": edge_decay = 0.85
            elif fwd_edge == "imports": edge_decay = 0.80
                
            if rev_edge == "calls": edge_decay = max(edge_decay, 0.90)
            elif rev_edge == "contains": edge_decay = max(edge_decay, 0.85)
            elif rev_edge == "inherits": edge_decay = max(edge_decay, 0.80)
            elif rev_edge == "imports": edge_decay = max(edge_decay, 0.75)
            
            if edge_decay > 0.0:
                # 步骤2：在标准的概率空间进行安全的连乘衰减（例如 0.8概率 * 0.9衰减 = 0.72概率）
                current_prob = parent_prob * edge_decay
                best_prob = max(best_prob, current_prob)

        # 针对完全没有任何边的孤岛节点或者没被成功扩展的，给予一个强制底限惩罚
        if best_prob == 0.0:
            # 找全局最低的得分（大概率是负数）
            min_logit = min([u.get("retrieval_score", 0.0) for u in ranked_units]) if ranked_units else 0.0
            # 还得继续往下扣除距离惩罚
            return min_logit - hop_distance

        # 步骤3：将衰减后的概率重新映射回原始的 Logit 空间，融入大部队比较
        return self._inverse_sigmoid(best_prob)

    def _build_edge_lookup(self) -> dict[tuple[str, str], str]:
        edge_lookup: dict[tuple[str, str], str] = {}
        for source, edges in self.graph_builder.adjacency.items():
            for edge in edges:
                edge_lookup[(source, edge.target)] = edge.edge_type
        return edge_lookup

    def _top_k_ids_from_scores(self, scores: dict[str, float], top_k: int) -> list[str]:
        return [
            unit_id
            for unit_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        ]

    def _apply_weighted_rrf(self, rankings_with_weights: List[tuple[List[str], float]], k: int = 60) -> dict[str, float]:
        rrf_scores = {}
        for ranking, weight in rankings_with_weights:
            for rank, item_id in enumerate(ranking):
                if item_id not in rrf_scores:
                    rrf_scores[item_id] = 0.0
                rrf_scores[item_id] += weight * (1.0 / (k + rank + 1))
        return rrf_scores

    def _normalize_scores(self, scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}

        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)
        if math.isclose(min_score, max_score):
            return {key: (1.0 if value > 0 else 0.0) for key, value in scores.items()}

        return {
            key: (value - min_score) / (max_score - min_score)
            for key, value in scores.items()
        }

    def _build_document(self, unit: dict) -> str:
        fields = [
            unit.get("name", ""),
            unit.get("type", ""),
            unit.get("module", ""),
            unit.get("file", ""),
            unit.get("signature", ""),
            unit.get("docstring", "") or "",
            " ".join(unit.get("imports") or []),
            " ".join(unit.get("calls") or []),
            unit.get("code", ""),
        ]
        return "\n".join(field for field in fields if field)

    def _tokenize(self, text: str) -> list[str]:
        return [token for token in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if token]